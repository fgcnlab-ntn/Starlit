from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from simulator.runlog import SCHEMA_VERSION, load_run


EARTH_RADIUS_KM = 6371.0
EARTH_VIEW_X_OFFSET_KM = EARTH_RADIUS_KM * 0.08
SIDEREAL_DAY_S = 86164.0905
EARTH_AMBIENT_LIGHT = 0.35
EARTH_SUNLIGHT = 1.0 - EARTH_AMBIENT_LIGHT
SCENE_BACKGROUND_RGB = (0.005, 0.01, 0.025)
SUNLIT_SATELLITE_RGB = (1.0, 0.76, 0.12)
ECLIPSE_SATELLITE_RGB = (0.05, 0.72, 0.90)
SELECTED_SATELLITE_RGB = (1.0, 0.24, 0.14)
ORBIT_MARKER_RGB = (0.30, 1.0, 0.58)
ORBIT_PATH_RGB = (0.72, 0.38, 1.0)
INFO_TEXT_RGB = (0.92, 0.95, 1.0)
INFO_BACKGROUND_RGB = (0.035, 0.045, 0.065)
INFO_PANEL_MARGIN_PX = 20
INFO_PANEL_TEXT_HEIGHT_PX = 15
INFO_PANEL_BORDER_PX = 10
BATTERY_CHART_WINDOW_STEPS = 120
UTILIZATION_VIOLIN_POINTS = 81
UTILIZATION_VIOLIN_BANDWIDTH_PCT = 5.0
UTILIZATION_VIOLIN_HALF_WIDTH = 0.34


@dataclass(frozen=True, slots=True)
class LocalActivity:
    completed: int | None
    failed: int | None
    queue_remaining: int | None


@dataclass(frozen=True, slots=True)
class SatelliteMetadata:
    sat_id: int
    plane: int


@dataclass(frozen=True, slots=True)
class SatellitePosition:
    sat_id: int
    position_km: tuple[float, float, float]
    sunlit: bool
    battery_j: float
    task_compute_time_s: float
    new_ground_tasks: int
    local_activity: LocalActivity


@dataclass(frozen=True, slots=True)
class Frame:
    time_s: float
    time_label: str
    sun_eci_unit: tuple[float, float, float] | None
    satellites: tuple[SatellitePosition, ...]


@dataclass(frozen=True, slots=True)
class OrbitMarkerGeometry:
    radius_km: float
    normal_eci: tuple[float, float, float]


class StateFrames:
    """Random-access view of a Starlit states.jsonl file.

    Only line offsets are retained in memory. A full constellation run can be
    very large, and previous/next navigation does not justify copying every
    satellite position into another in-memory representation.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.path = output_dir / "states.jsonl"
        if not self.path.is_file():
            raise FileNotFoundError(
                f"missing {self.path}; run the simulation with "
                'logging.state_steps set to "full"'
            )

        offsets: list[int] = []
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    raise ValueError(f"{self.path} contains a blank JSONL record")
                offsets.append(offset)

        if not offsets:
            raise ValueError(f"{self.path} contains no state frames")
        self._offsets = tuple(offsets)

        first = self[0]
        self._satellite_ids = tuple(sat.sat_id for sat in first.satellites)
        self._metadata_by_satellite, self.battery_capacity_j = _load_run_metadata(
            output_dir,
            self._satellite_ids,
        )
        self.planes = tuple(
            sorted({metadata.plane for metadata in self._metadata_by_satellite.values()})
        )
        self._satellite_ids_by_plane = {
            plane: tuple(
                sat_id
                for sat_id in self._satellite_ids
                if self._metadata_by_satellite[sat_id].plane == plane
            )
            for plane in self.planes
        }

    def __len__(self) -> int:
        return len(self._offsets)

    @lru_cache(maxsize=4)
    def __getitem__(self, index: int) -> Frame:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        with self.path.open("rb") as stream:
            stream.seek(self._offsets[index])
            raw = stream.readline()
        source = f"{self.path}:{index + 1}"
        try:
            record = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON in {source}: {exc}") from exc
        frame = _decode_frame(record, source=source)

        expected_ids = getattr(self, "_satellite_ids", None)
        actual_ids = tuple(sat.sat_id for sat in frame.satellites)
        if expected_ids is not None and actual_ids != expected_ids:
            raise ValueError(
                f"{source} satellite ids {actual_ids!r} do not match "
                f"the first frame {expected_ids!r}"
            )
        return frame

    def plane_for(self, sat_id: int) -> int:
        return self.metadata_for(sat_id).plane

    def metadata_for(self, sat_id: int) -> SatelliteMetadata:
        return self._metadata_by_satellite[sat_id]

    def satellites_in_plane(self, plane: int) -> tuple[int, ...]:
        return self._satellite_ids_by_plane[plane]


def _load_run_metadata(
    output_dir: Path,
    satellite_ids: tuple[int, ...],
) -> tuple[dict[int, SatelliteMetadata], float | None]:
    fallback = {
        sat_id: SatelliteMetadata(sat_id=sat_id, plane=0)
        for sat_id in satellite_ids
    }
    try:
        manifest = load_run(output_dir)
    except (FileNotFoundError, ValueError):
        return fallback, None

    battery_capacity_j = _battery_capacity(manifest)

    raw_catalog = manifest.get("satellites")
    if not isinstance(raw_catalog, list):
        return fallback, battery_capacity_j

    result: dict[int, SatelliteMetadata] = {}
    try:
        for entry in raw_catalog:
            if not isinstance(entry, dict):
                raise TypeError
            sat_id = int(entry["id"])
            plane = int(entry["plane"])
            if sat_id in result or plane < 0:
                raise ValueError
            result[sat_id] = SatelliteMetadata(sat_id, plane)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{output_dir / 'run.json'} has an invalid satellite catalog"
        ) from exc

    if set(result) != set(satellite_ids):
        raise ValueError(
            f"{output_dir / 'run.json'} satellite catalog does not match states.jsonl"
        )
    return result, battery_capacity_j


def _battery_capacity(manifest: dict[str, object]) -> float | None:
    try:
        config = manifest["config"]
        if not isinstance(config, dict):
            return None
        battery = config["battery"]
        if not isinstance(battery, dict):
            return None
        capacity = float(battery["capacity_j"])
    except (KeyError, TypeError, ValueError):
        return None
    return capacity if math.isfinite(capacity) and capacity > 0 else None


def _decode_frame(value: object, *, source: str) -> Frame:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{source} has unsupported schema_version "
            f"{value.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )

    try:
        time_s = float(value["time_s"])
        time_label = str(value["time_iso"])
        raw_satellites = value["satellites"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source} has invalid frame metadata") from exc
    if not math.isfinite(time_s):
        raise ValueError(f"{source} time_s must be finite")
    if not isinstance(raw_satellites, list) or not raw_satellites:
        raise ValueError(f"{source} satellites must be a non-empty array")

    sun_eci_unit = _decode_sun_eci_unit(value.get("snapshot_context"), source)

    satellites: list[SatellitePosition] = []
    seen_ids: set[int] = set()
    for raw_satellite in raw_satellites:
        if not isinstance(raw_satellite, dict):
            raise ValueError(f"{source} contains an invalid satellite record")
        try:
            sat_id = int(raw_satellite["id"])
            raw_position = raw_satellite["position_km"]
            position = tuple(float(component) for component in raw_position)
            sunlit = bool(raw_satellite["sunlit"])
            battery_j = float(raw_satellite["battery_j"])
            raw_task_load = raw_satellite["task_load"]
            raw_task_counts = raw_satellite["task_counts"]
            if not isinstance(raw_task_load, dict) or not isinstance(
                raw_task_counts, dict
            ):
                raise TypeError
            task_compute_time_s = float(raw_task_load["compute_time_s"])
            new_ground_tasks = int(raw_task_counts["generated"])
            raw_local_activity = raw_satellite.get("local_activity")
            local_activity = (
                LocalActivity(completed=None, failed=None, queue_remaining=None)
                if raw_local_activity is None
                else LocalActivity(
                    completed=int(raw_local_activity["completed"]),
                    failed=(
                        None
                        if raw_local_activity.get("failed") is None
                        else int(raw_local_activity["failed"])
                    ),
                    queue_remaining=int(raw_local_activity["queue_remaining"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source} contains an invalid satellite record") from exc
        if sat_id in seen_ids:
            raise ValueError(f"{source} contains duplicate satellite id {sat_id}")
        if len(position) != 3 or not all(math.isfinite(x) for x in position):
            raise ValueError(
                f"{source} satellite {sat_id} position_km must contain "
                "three finite values"
            )
        if not math.isfinite(battery_j) or not math.isfinite(task_compute_time_s):
            raise ValueError(
                f"{source} satellite {sat_id} battery and task load must be finite"
            )
        if any(
            value is not None and value < 0
            for value in (
                local_activity.completed,
                local_activity.failed,
                local_activity.queue_remaining,
            )
        ):
            raise ValueError(
                f"{source} satellite {sat_id} local activity must be non-negative"
            )
        seen_ids.add(sat_id)
        satellites.append(
            SatellitePosition(
                sat_id,
                position,
                sunlit,
                battery_j,
                task_compute_time_s,
                new_ground_tasks,
                local_activity,
            )
        )

    satellites.sort(key=lambda sat: sat.sat_id)
    return Frame(time_s, time_label, sun_eci_unit, tuple(satellites))


def _decode_sun_eci_unit(
    snapshot_context: object,
    source: str,
) -> tuple[float, float, float] | None:
    if snapshot_context is None:
        return None
    if not isinstance(snapshot_context, dict):
        raise ValueError(f"{source} snapshot_context must be a JSON object")

    raw_vector = snapshot_context.get("sun_eci_unit")
    if raw_vector is None:
        return None
    try:
        vector = tuple(float(component) for component in raw_vector)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} sun_eci_unit must contain three numbers") from exc
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{source} sun_eci_unit must contain three finite numbers")

    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0:
        raise ValueError(f"{source} sun_eci_unit must not be zero")
    return tuple(component / magnitude for component in vector)


@dataclass(slots=True)
class Playback:
    frame_count: int
    index: int = 0
    paused: bool = False

    def __post_init__(self) -> None:
        if self.frame_count <= 0:
            raise ValueError("frame_count must be positive")

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def previous(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.index = max(0, self.index - steps)
        self.paused = False

    def next(self, steps: int = 1) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive")
        self.index = min(self.frame_count - 1, self.index + steps)
        self.paused = False

    def reset(self) -> None:
        self.index = 0
        self.paused = False

    def advance(self) -> bool:
        if self.paused:
            return False
        if self.index == self.frame_count - 1:
            self.paused = True
            return False
        self.index += 1
        return True


class PlaneSelection:
    def __init__(self, planes: tuple[int, ...]) -> None:
        if not planes:
            raise ValueError("planes must not be empty")
        self.planes = planes
        self.selected = set(planes)

    def set_selected(self, plane: int, selected: bool) -> None:
        if plane not in self.planes:
            raise ValueError(f"unknown plane {plane}")
        if selected:
            self.selected.add(plane)
        else:
            self.selected.discard(plane)

    def select_all(self) -> None:
        self.selected = set(self.planes)

    def select_none(self) -> None:
        self.selected.clear()

    def contains(self, plane: int) -> bool:
        return plane in self.selected


def _eci_to_scene(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    """Map ECI's north axis (+z) to the canvas's vertical axis (+y)."""

    x, y, z = vector
    return (x, z, -y)


def _earth_rotation_angle(elapsed_s: float) -> float:
    return 2.0 * math.pi * elapsed_s / SIDEREAL_DAY_S


def _format_time_label(time_label: str) -> str:
    display = time_label.replace("T", " ", 1)
    if display.endswith("+00:00"):
        return f"{display[:-6]} UTC"
    if display.endswith("Z"):
        return f"{display[:-1]} UTC"
    return display


def _format_plane_id(plane: int, width: int) -> str:
    return str(plane + 1).rjust(width, "\N{FIGURE SPACE}")


def _eci_to_lat_lon(
    position_km: tuple[float, float, float],
    time_label: str,
) -> tuple[float, float]:
    """Return the spherical Earth subpoint of an ECI position in degrees."""

    x_km, y_km, z_km = position_km
    latitude = math.degrees(math.atan2(z_km, math.hypot(x_km, y_km)))
    inertial_longitude = math.atan2(y_km, x_km)
    longitude = math.degrees(inertial_longitude - _greenwich_sidereal_angle(time_label))
    return latitude, (longitude + 180.0) % 360.0 - 180.0


def _orbital_altitude_km(position_km: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in position_km)) - EARTH_RADIUS_KM


def _greenwich_sidereal_angle(time_label: str) -> float:
    """Return Greenwich mean sidereal angle for an ISO-8601 UTC timestamp."""

    timestamp = dt.datetime.fromisoformat(time_label.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    timestamp = timestamp.astimezone(dt.timezone.utc)
    unix_days = timestamp.timestamp() / 86400.0
    julian_date = 2440587.5 + unix_days
    centuries = (julian_date - 2451545.0) / 36525.0
    angle_deg = (
        280.46061837
        + 360.98564736629 * (julian_date - 2451545.0)
        + 0.000387933 * centuries * centuries
        - centuries * centuries * centuries / 38710000.0
    )
    return math.radians(angle_deg % 360.0)


def _format_dms(value_deg: float, *, positive: str, negative: str) -> str:
    hemisphere = positive if value_deg >= 0.0 else negative
    total_seconds = round(abs(value_deg) * 3600.0, 1)
    degrees, remainder = divmod(total_seconds, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    return (
        f"{int(degrees)}\N{DEGREE SIGN} {int(minutes):02d}\N{PRIME} "
        f"{seconds:04.1f}\N{DOUBLE PRIME} {hemisphere}"
    )


def _battery_chart_value(
    battery_j: float,
    battery_capacity_j: float | None,
) -> float:
    if battery_capacity_j is None:
        return battery_j
    return 100.0 * battery_j / battery_capacity_j


def _battery_chart_x_bounds(
    first_time_s: float,
    current_time_s: float,
    step_s: float,
    window_steps: int = BATTERY_CHART_WINDOW_STEPS,
) -> tuple[float, float]:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    step_h = max(1.0, abs(step_s)) / 3600.0
    first_time_h = first_time_s / 3600.0
    current_time_h = current_time_s / 3600.0
    xmin = max(
        first_time_h,
        current_time_h - step_h * (window_steps - 1),
    )
    return xmin, xmin + step_h * window_steps


def _satellite_utilizations(
    frame: Frame,
    step_s: float,
    *,
    sunlit: bool,
) -> list[float]:
    """Return per-satellite compute utilization for one simulation step.

    Utilization is defined as task_compute_time_s / step_s and clipped to
    [0, 100] percent.  This assumes task_compute_time_s is the amount of
    compute scheduled/executed during the current simulation step.
    """

    if not math.isfinite(step_s) or step_s == 0:
        raise ValueError("step_s must be a finite non-zero value")
    duration_s = abs(step_s)
    return [
        max(0.0, min(100.0, 100.0 * satellite.task_compute_time_s / duration_s))
        for satellite in frame.satellites
        if satellite.sunlit is sunlit
    ]


def _violin_kde(
    values: list[float],
    *,
    points: int = UTILIZATION_VIOLIN_POINTS,
    bandwidth: float = UTILIZATION_VIOLIN_BANDWIDTH_PCT,
) -> list[tuple[float, float]]:
    """Evaluate a lightweight Gaussian KDE over the fixed 0--100% domain."""

    if points < 2:
        raise ValueError("points must be at least 2")
    if bandwidth <= 0 or not math.isfinite(bandwidth):
        raise ValueError("bandwidth must be a positive finite value")
    if not values:
        return []

    scale = 1.0 / (len(values) * bandwidth * math.sqrt(2.0 * math.pi))
    result: list[tuple[float, float]] = []
    for index in range(points):
        utilization = 100.0 * index / (points - 1)
        density = 0.0
        for value in values:
            z = (utilization - value) / bandwidth
            density += math.exp(-0.5 * z * z)
        result.append((utilization, density * scale))
    return result


def _violin_outline_points(
    kde: list[tuple[float, float]],
    *,
    center_x: float,
    max_density: float,
    side: float,
) -> list[tuple[float, float]]:
    if side not in (-1.0, 1.0):
        raise ValueError("side must be -1.0 or 1.0")
    if not kde or max_density <= 0:
        return []
    return [
        (
            center_x
            + side * UTILIZATION_VIOLIN_HALF_WIDTH * density / max_density,
            utilization,
        )
        for utilization, density in kde
    ]


def _info_panel_center_y(canvas_height: int, text: str) -> float:
    line_count = text.count("\n") + 1
    text_height = INFO_PANEL_TEXT_HEIGHT_PX * (1.0 + 1.3 * (line_count - 1))
    panel_height = text_height + 2.0 * INFO_PANEL_BORDER_PX
    return canvas_height - INFO_PANEL_MARGIN_PX - panel_height / 2.0


def _camera_yaw_to_target(
    forward_scene: tuple[float, float, float],
    target_scene: tuple[float, float, float],
) -> float | None:
    camera_x = -forward_scene[0]
    camera_z = -forward_scene[2]
    target_x = target_scene[0]
    target_z = target_scene[2]
    camera_length = math.hypot(camera_x, camera_z)
    target_length = math.hypot(target_x, target_z)
    if camera_length == 0 or target_length == 0:
        return None

    camera_x /= camera_length
    camera_z /= camera_length
    target_x /= target_length
    target_z /= target_length
    cross_y = camera_z * target_x - camera_x * target_z
    dot = camera_x * target_x + camera_z * target_z
    return math.atan2(cross_y, dot)


def _orbit_marker_geometry(
    positions_km: list[tuple[float, float, float]],
) -> OrbitMarkerGeometry | None:
    if len(positions_km) < 2:
        return None

    radii = [
        math.sqrt(sum(component * component for component in position))
        for position in positions_km
    ]
    first = positions_km[0]
    best_cross = (0.0, 0.0, 0.0)
    best_magnitude_sq = 0.0
    for other in positions_km[1:]:
        cross = (
            first[1] * other[2] - first[2] * other[1],
            first[2] * other[0] - first[0] * other[2],
            first[0] * other[1] - first[1] * other[0],
        )
        magnitude_sq = sum(component * component for component in cross)
        if magnitude_sq > best_magnitude_sq:
            best_cross = cross
            best_magnitude_sq = magnitude_sq

    if best_magnitude_sq == 0:
        return None
    magnitude = math.sqrt(best_magnitude_sq)
    normal = tuple(component / magnitude for component in best_cross)
    return OrbitMarkerGeometry(sum(radii) / len(radii), normal)


class Visualizer:
    def __init__(
        self,
        frames: StateFrames,
        *,
        title: str = "Starlit simulation",
        satellite_radius_km: float = 60.0,
    ) -> None:
        if satellite_radius_km <= 0:
            raise ValueError("satellite_radius_km must be positive")
        self.frames = frames
        self.title = title
        self.satellite_radius_km = satellite_radius_km

    def play(self, rate: int = 10) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        try:
            import vpython as vp
        except ImportError as exc:
            raise SystemExit(
                "VPython is required for the interactive visualizer. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc

        playback = Playback(len(self.frames))
        satellite_plane_selection = PlaneSelection(self.frames.planes)
        path_plane_selection = PlaneSelection(self.frames.planes)
        path_plane_selection.select_none()
        dirty = True
        graph_args: dict[str, Any] = {
            "width": 450,
            "height": 280,
            "align": "none",
            "title": "Select a satellite",
            "xtitle": "Simulation time (h)",
            "ytitle": (
                "Battery (%)"
                if self.frames.battery_capacity_j is not None
                else "Battery (J)"
            ),
            "foreground": vp.color.black,
            "background": vp.color.white,
            "ymin": 0,
            "fast": True,
        }
        if self.frames.battery_capacity_j is not None:
            graph_args["ymax"] = 100
        battery_graph = vp.graph(**graph_args)

        utilization_graph = vp.graph(
            width=450,
            height=320,
            align="none",
            title="Satellite utilization by illumination<br>Sunlit / Eclipse",
            xtitle="1 = Sunlit                     2 = Eclipse",
            ytitle="Utilization (%)",
            xmin=0.5,
            xmax=2.5,
            ymin=0,
            ymax=100,
            foreground=vp.color.black,
            background=vp.color.white,
            fast=True,
        )

        scene = vp.canvas(
            title=self.title,
            width=930,
            height=720,
            align="left",
            center=vp.vec(EARTH_VIEW_X_OFFSET_KM, 0, 0),
            background=vp.vec(*SCENE_BACKGROUND_RGB),
        )
        scene.autoscale = False
        scene.userspin = False
        scene.userpan = False
        scene.range = EARTH_RADIUS_KM * 1.35
        scene.lights = []
        scene.ambient = vp.color.gray(EARTH_AMBIENT_LIGHT)
        scene.caption = (
            '<style>#glowscript{position:relative;min-width:1440px}'
            '.glowscript-graph{position:absolute!important;top:18px;left:950px;'
            'float:none!important;background:#fff!important}'
            '.glowscript-graph + .glowscript-graph{top:360px!important}</style>'
            '<div style="clear:both"></div>'
        )

        first = self.frames[0]
        initial_sun = _eci_to_scene(first.sun_eci_unit or (1.0, 0.0, 0.0))
        sunlight = vp.distant_light(
            direction=vp.vec(*initial_sun),
            color=vp.color.gray(EARTH_SUNLIGHT),
        )
        earth = vp.sphere(
            pos=vp.vec(0, 0, 0),
            radius=EARTH_RADIUS_KM,
            texture=vp.textures.earth,
            shininess=0,
        )
        orbit_marker = vp.ring(
            pos=vp.vec(0, 0, 0),
            axis=vp.vec(0, 1, 0),
            radius=EARTH_RADIUS_KM,
            thickness=max(18.0, self.satellite_radius_km * 0.45),
            color=vp.vec(*ORBIT_MARKER_RGB),
            opacity=0.95,
            emissive=True,
            visible=False,
        )
        orbit_paths: dict[int, Any] = {}

        def orbit_path_for(plane: int) -> Any | None:
            existing = orbit_paths.get(plane)
            if existing is not None:
                return existing
            geometry = _orbit_marker_geometry(
                [
                    satellite.position_km
                    for satellite in first.satellites
                    if self.frames.plane_for(satellite.sat_id) == plane
                ]
            )
            if geometry is None:
                return None
            path = vp.ring(
                pos=vp.vec(0, 0, 0),
                axis=vp.vec(*_eci_to_scene(geometry.normal_eci)),
                radius=geometry.radius_km,
                thickness=max(12.0, self.satellite_radius_km * 0.28),
                color=vp.vec(*ORBIT_PATH_RGB),
                opacity=0.82,
                emissive=True,
                visible=False,
            )
            orbit_paths[plane] = path
            return path

        def update_orbit_path_visibility() -> None:
            for plane, path in orbit_paths.items():
                path.visible = path_plane_selection.contains(plane)
            for plane in path_plane_selection.selected:
                path = orbit_path_for(plane)
                if path is not None:
                    path.visible = True

        satellite_colors = {
            True: vp.vec(*SUNLIT_SATELLITE_RGB),
            False: vp.vec(*ECLIPSE_SATELLITE_RGB),
        }
        selected_satellite_color = vp.vec(*SELECTED_SATELLITE_RGB)

        objects = {
            sat.sat_id: vp.sphere(
                pos=vp.vec(*_eci_to_scene(sat.position_km)),
                radius=self.satellite_radius_km,
                color=satellite_colors[sat.sunlit],
                emissive=True,
            )
            for sat in first.satellites
        }
        satellite_by_object_id = {obj.idx: sat_id for sat_id, obj in objects.items()}
        selected_satellite_id: int | None = None
        earth_time_s = first.time_s
        pending_rotation = 0.0
        battery_curve: Any | None = None
        battery_chart_satellite_id: int | None = None
        battery_chart_last_index: int | None = None
        utilization_curves: list[Any] = []
        utilization_chart_last_index: int | None = None

        if len(self.frames) > 1:
            utilization_step_s = self.frames[1].time_s - first.time_s
        else:
            utilization_step_s = 1.0

        def clear_utilization_chart() -> None:
            nonlocal utilization_curves
            for curve in utilization_curves:
                curve.delete()
            utilization_curves = []

        def update_utilization_chart(frame: Frame) -> None:
            nonlocal utilization_chart_last_index, utilization_curves
            if utilization_chart_last_index == playback.index:
                return

            sunlit_values = _satellite_utilizations(
                frame, utilization_step_s, sunlit=True
            )
            eclipse_values = _satellite_utilizations(
                frame, utilization_step_s, sunlit=False
            )
            sunlit_kde = _violin_kde(sunlit_values)
            eclipse_kde = _violin_kde(eclipse_values)

            clear_utilization_chart()
            sunlit_color = vp.vec(*SUNLIT_SATELLITE_RGB)
            eclipse_color = vp.vec(*ECLIPSE_SATELLITE_RGB)
            for center_x, kde, color in (
                (1.0, sunlit_kde, sunlit_color),
                (2.0, eclipse_kde, eclipse_color),
            ):
                max_density = max(
                    (density for _, density in kde),
                    default=0.0,
                )
                if not kde or max_density <= 0:
                    continue
                left = _violin_outline_points(
                    kde, center_x=center_x, max_density=max_density, side=-1.0
                )
                right = _violin_outline_points(
                    kde, center_x=center_x, max_density=max_density, side=1.0
                )
                outline = left + list(reversed(right)) + [left[0]]
                utilization_curves.append(
                    vp.gcurve(
                        graph=utilization_graph,
                        data=outline,
                        color=color,
                        width=3,
                    )
                )

            utilization_graph.title = (
                "Satellite utilization by illumination"
            )
            utilization_chart_last_index = playback.index

        def clear_battery_chart() -> None:
            nonlocal battery_curve
            nonlocal battery_chart_satellite_id, battery_chart_last_index
            if battery_curve is not None:
                battery_curve.delete()
            battery_curve = None
            battery_graph.title = "Select a satellite"
            battery_chart_satellite_id = None
            battery_chart_last_index = None

        def update_battery_chart(satellite: SatellitePosition) -> None:
            nonlocal battery_curve
            nonlocal battery_chart_satellite_id, battery_chart_last_index
            if (
                battery_chart_satellite_id == satellite.sat_id
                and battery_chart_last_index == playback.index
            ):
                return

            sequential = (
                battery_curve is not None
                and battery_chart_satellite_id == satellite.sat_id
                and battery_chart_last_index is not None
                and playback.index == battery_chart_last_index + 1
            )
            if sequential:
                battery_curve.plot(
                    self.frames[playback.index].time_s / 3600.0,
                    _battery_chart_value(
                        satellite.battery_j,
                        self.frames.battery_capacity_j,
                    ),
                )
            else:
                points = _battery_chart_points(
                    self.frames,
                    satellite.sat_id,
                    playback.index,
                )
                if battery_curve is not None:
                    battery_curve.delete()
                battery_curve = vp.gcurve(
                    graph=battery_graph,
                    data=points,
                    color=vp.vec(*ORBIT_MARKER_RGB),
                    width=3,
                    dot=True,
                    dot_color=vp.vec(*SELECTED_SATELLITE_RGB),
                    dot_radius=5,
                )

            battery_graph.title = f"Satellite {satellite.sat_id}"
            if len(self.frames) > 1:
                step_s = self.frames[1].time_s - first.time_s
            else:
                step_s = 1.0
            battery_graph.xmin, battery_graph.xmax = _battery_chart_x_bounds(
                first.time_s,
                self.frames[playback.index].time_s,
                step_s,
            )
            battery_chart_satellite_id = satellite.sat_id
            battery_chart_last_index = playback.index

        def center_earth_left() -> None:
            screen_right = vp.norm(vp.cross(scene.forward, scene.up))
            scene.center = screen_right * EARTH_VIEW_X_OFFSET_KM

        def rotate_view(angle: float) -> None:
            scene.forward = scene.forward.rotate(
                angle=angle,
                axis=vp.vec(0, 1, 0),
            )
            scene.up = scene.up.rotate(
                angle=angle,
                axis=vp.vec(0, 1, 0),
            )
            center_earth_left()

        def face_direction(target: tuple[float, float, float]) -> None:
            nonlocal pending_rotation
            forward = (
                scene.forward.x,
                scene.forward.y,
                scene.forward.z,
            )
            angle = _camera_yaw_to_target(forward, target)
            if angle is None:
                return
            pending_rotation = 0.0
            rotate_view(angle)

        face_direction(initial_sun)

        def set_paused_label() -> None:
            pause_button.text = "Play" if playback.paused else "Pause"

        def previous_100(_button: Any) -> None:
            nonlocal dirty
            playback.previous(100)
            set_paused_label()
            dirty = True

        vp.button(text="-100", bind=previous_100)

        def previous(_button: Any) -> None:
            nonlocal dirty
            playback.previous()
            set_paused_label()
            dirty = True

        vp.button(text="-1", bind=previous)

        def pause(_button: Any) -> None:
            nonlocal dirty
            playback.toggle_pause()
            set_paused_label()
            dirty = True

        pause_button = vp.button(text="Pause", bind=pause)

        def next_frame(_button: Any) -> None:
            nonlocal dirty
            playback.next()
            set_paused_label()
            dirty = True

        vp.button(text="+1", bind=next_frame)

        def next_100(_button: Any) -> None:
            nonlocal dirty
            playback.next(100)
            set_paused_label()
            dirty = True

        vp.button(text="+100", bind=next_100)

        def reset(_button: Any) -> None:
            nonlocal dirty
            playback.reset()
            face_direction(initial_sun)
            set_paused_label()
            dirty = True

        vp.button(text="Restart", bind=reset)

        def reset_view(_button: Any) -> None:
            scene.forward = vp.vec(0, 0, -1)
            scene.up = vp.vec(0, 1, 0)
            center_earth_left()
            scene.range = EARTH_RADIUS_KM * 1.35
            frame = self.frames[playback.index]
            update_utilization_chart(frame)
            sun_direction = _eci_to_scene(
                frame.sun_eci_unit or (1.0, 0.0, 0.0)
            )
            face_direction(sun_direction)

        vp.button(text="Reset camera", bind=reset_view)

        dragging = False
        drag_moved = False
        resume_after_drag = False
        previous_pointer_x = 0.0

        def pointer_x() -> float:
            right = vp.norm(vp.cross(scene.forward, scene.up))
            return vp.dot(vp.norm(scene.mouse.ray), right)

        def start_drag(event: Any) -> None:
            nonlocal dragging, drag_moved, previous_pointer_x, resume_after_drag
            if event.which != 1:
                return
            dragging = True
            drag_moved = False
            resume_after_drag = not playback.paused
            playback.paused = True
            set_paused_label()
            previous_pointer_x = pointer_x()

        def drag_view(_event: Any) -> None:
            nonlocal drag_moved, previous_pointer_x, pending_rotation
            if not dragging:
                return
            current_pointer_x = pointer_x()
            delta = current_pointer_x - previous_pointer_x
            if abs(delta) > 1.0e-4:
                drag_moved = True
            pending_rotation += max(-0.15, min(0.15, -2.0 * delta))
            previous_pointer_x = current_pointer_x

        def stop_drag(_event: Any) -> None:
            nonlocal dragging, resume_after_drag
            dragging = False
            if resume_after_drag:
                playback.paused = False
                set_paused_label()
            resume_after_drag = False

        scene.bind("mousedown", start_drag)
        scene.bind("mousemove", drag_view)
        scene.bind("mouseup mouseleave", stop_drag)

        plane_menu: Any | None = None
        satellite_menu: Any | None = None

        def clear_satellite_selection() -> None:
            nonlocal dirty, selected_satellite_id
            selected_satellite_id = None
            satellite_info.visible = False
            orbit_marker.visible = False
            clear_battery_chart()
            if satellite_menu is not None:
                satellite_menu.selected = "None"
            dirty = True

        selection_mode = "Satellites"
        vp.wtext(text="<br>Configure: ")

        def active_plane_selection() -> PlaneSelection:
            if selection_mode == "Satellites":
                return satellite_plane_selection
            return path_plane_selection

        plane_checkboxes: dict[int, Any] = {}

        def sync_plane_checkboxes() -> None:
            selection = active_plane_selection()
            all_planes_checkbox.checked = len(selection.selected) == len(
                selection.planes
            )
            for plane, checkbox in plane_checkboxes.items():
                checkbox.checked = selection.contains(plane)

        def choose_selection_mode(menu: Any) -> None:
            nonlocal selection_mode
            selection_mode = menu.selected
            sync_plane_checkboxes()

        vp.menu(
            choices=["Satellites", "Orbit paths"],
            index=0,
            bind=choose_selection_mode,
        )
        vp.wtext(text="<br>Orbit planes: ")

        def select_all_planes(checkbox: Any) -> None:
            nonlocal dirty
            selection = active_plane_selection()
            if checkbox.checked:
                selection.select_all()
            else:
                selection.select_none()
            for plane_checkbox in plane_checkboxes.values():
                plane_checkbox.checked = all_planes_checkbox.checked
            if selection_mode == "Satellites":
                if (
                    selected_satellite_id is not None
                    and not satellite_plane_selection.contains(
                        self.frames.plane_for(selected_satellite_id)
                    )
                ):
                    clear_satellite_selection()
                _set_visibility(objects, self.frames, satellite_plane_selection)
            else:
                update_orbit_path_visibility()
            dirty = True

        all_planes_checkbox = vp.checkbox(
            text="All",
            checked=True,
            bind=select_all_planes,
        )
        vp.wtext(text="<br>")

        plane_id_width = max(len(str(plane + 1)) for plane in self.frames.planes)
        for index, plane in enumerate(self.frames.planes):
            if index and index % 12 == 0:
                vp.wtext(text="<br>")

            def select_plane(checkbox: Any, selected_plane: int = plane) -> None:
                nonlocal dirty
                selection = active_plane_selection()
                selection.set_selected(selected_plane, checkbox.checked)
                all_planes_checkbox.checked = len(selection.selected) == len(
                    selection.planes
                )
                if selection_mode == "Satellites":
                    if (
                        selected_satellite_id is not None
                        and not satellite_plane_selection.contains(
                            self.frames.plane_for(selected_satellite_id)
                        )
                    ):
                        clear_satellite_selection()
                    _set_visibility(
                        objects,
                        self.frames,
                        satellite_plane_selection,
                    )
                else:
                    update_orbit_path_visibility()
                dirty = True

            plane_checkboxes[plane] = vp.checkbox(
                text=f"{_format_plane_id(plane, plane_id_width)}　",
                checked=True,
                bind=select_plane,
            )

        plane_by_label = {str(plane + 1): plane for plane in self.frames.planes}

        def satellite_choices(plane: int) -> list[str]:
            return [
                "None",
                *(str(sat_id) for sat_id in self.frames.satellites_in_plane(plane)),
            ]

        def sync_inspector(sat_id: int) -> None:
            plane = self.frames.plane_for(sat_id)
            assert plane_menu is not None
            assert satellite_menu is not None
            plane_menu.selected = str(plane + 1)
            satellite_menu.choices = satellite_choices(plane)
            satellite_menu.selected = str(sat_id)

        def locate_satellite(sat_id: int) -> None:
            satellite = _satellite_in_frame(self.frames[playback.index], sat_id)
            target = _eci_to_scene(satellite.position_km)
            face_direction(target)

        def select_satellite_by_id(
            sat_id: int,
            *,
            ensure_visible: bool,
            locate: bool,
        ) -> None:
            nonlocal dirty, selected_satellite_id
            plane = self.frames.plane_for(sat_id)
            if ensure_visible and not satellite_plane_selection.contains(plane):
                satellite_plane_selection.set_selected(plane, True)
                if selection_mode == "Satellites":
                    sync_plane_checkboxes()
                _set_visibility(objects, self.frames, satellite_plane_selection)
            if locate:
                locate_satellite(sat_id)
            selected_satellite_id = sat_id
            playback.paused = True
            set_paused_label()
            sync_inspector(sat_id)
            dirty = True

        def choose_inspector_plane(menu: Any) -> None:
            assert satellite_menu is not None
            clear_satellite_selection()
            plane = plane_by_label[menu.selected]
            satellite_menu.choices = satellite_choices(plane)
            satellite_menu.selected = "None"

        def choose_inspector_satellite(menu: Any) -> None:
            if menu.selected == "None":
                clear_satellite_selection()
                return
            select_satellite_by_id(
                int(menu.selected),
                ensure_visible=True,
                locate=True,
            )

        vp.wtext(text="<br><br>Inspect satellite: Orbit plane ")
        plane_menu = vp.menu(
            choices=list(plane_by_label),
            index=0,
            bind=choose_inspector_plane,
        )
        vp.wtext(text="　Satellite ID ")
        satellite_menu = vp.menu(
            choices=satellite_choices(self.frames.planes[0]),
            index=0,
            bind=choose_inspector_satellite,
        )
        vp.wtext(text="<br><br>")

        satellite_info = vp.label(
            pos=vp.vec(
                scene.width,
                _info_panel_center_y(scene.height, ""),
                0,
            ),
            pixel_pos=True,
            text="",
            align="left",
            height=INFO_PANEL_TEXT_HEIGHT_PX,
            color=vp.vec(*INFO_TEXT_RGB),
            background=vp.vec(*INFO_BACKGROUND_RGB),
            opacity=0.94,
            border=INFO_PANEL_BORDER_PX,
            xoffset=-INFO_PANEL_MARGIN_PX,
            box=True,
            line=False,
            visible=False,
        )

        def position_satellite_info(_event: Any) -> None:
            satellite_info.pos = vp.vec(
                scene.width,
                _info_panel_center_y(scene.height, satellite_info.text),
                0,
            )

        scene.bind("resize", position_satellite_info)

        def select_satellite(event: Any) -> None:
            nonlocal dirty, selected_satellite_id
            if drag_moved:
                return
            picked = scene.mouse.pick
            sat_id = (
                None
                if picked is None
                else satellite_by_object_id.get(getattr(picked, "idx", -1))
            )
            if sat_id is None or sat_id == selected_satellite_id:
                clear_satellite_selection()
                return
            select_satellite_by_id(
                sat_id,
                ensure_visible=False,
                locate=False,
            )

        scene.bind("click", select_satellite)

        while True:
            vp.rate(rate)
            if pending_rotation:
                rotate_view(pending_rotation)
                pending_rotation = 0.0
            if playback.advance():
                dirty = True
            elif playback.paused:
                set_paused_label()
            if not dirty:
                continue

            frame = self.frames[playback.index]
            update_utilization_chart(frame)
            sun_direction = _eci_to_scene(
                frame.sun_eci_unit or (1.0, 0.0, 0.0)
            )
            sunlight.direction = vp.vec(*sun_direction)
            earth.rotate(
                angle=_earth_rotation_angle(frame.time_s - earth_time_s),
                axis=vp.vec(0, 1, 0),
            )
            earth_time_s = frame.time_s
            scene.title = (
                f"{self.title} | {_format_time_label(frame.time_label)} | "
                f"Step {playback.index + 1} / {len(self.frames)}"
            )
            for sat in frame.satellites:
                obj = objects[sat.sat_id]
                visible = satellite_plane_selection.contains(
                    self.frames.plane_for(sat.sat_id)
                )
                obj.visible = visible
                if not visible:
                    continue
                obj.pos = vp.vec(*_eci_to_scene(sat.position_km))
                if sat.sat_id == selected_satellite_id:
                    obj.radius = self.satellite_radius_km * 2.0
                    obj.color = selected_satellite_color
                else:
                    obj.radius = self.satellite_radius_km
                    obj.color = satellite_colors[sat.sunlit]
            if selected_satellite_id is not None:
                selected = _satellite_in_frame(frame, selected_satellite_id)
                selected_plane = self.frames.plane_for(selected_satellite_id)
                marker_geometry = _orbit_marker_geometry(
                    [
                        satellite.position_km
                        for satellite in frame.satellites
                        if self.frames.plane_for(satellite.sat_id) == selected_plane
                    ]
                )
                if marker_geometry is not None:
                    orbit_marker.axis = vp.vec(
                        *_eci_to_scene(marker_geometry.normal_eci)
                    )
                    orbit_marker.radius = marker_geometry.radius_km
                    orbit_marker.visible = True
                else:
                    orbit_marker.visible = False
                satellite_info.text = _satellite_info(
                    selected,
                    self.frames.metadata_for(selected_satellite_id),
                    self.frames.battery_capacity_j,
                    frame.time_label,
                )
                position_satellite_info(None)
                satellite_info.visible = True
                update_battery_chart(selected)
            dirty = False


def _satellite_in_frame(frame: Frame, sat_id: int) -> SatellitePosition:
    for satellite in frame.satellites:
        if satellite.sat_id == sat_id:
            return satellite
    raise ValueError(f"satellite {sat_id} is missing from frame")


def _battery_chart_points(
    frames: StateFrames,
    sat_id: int,
    end_index: int,
    *,
    window_steps: int = BATTERY_CHART_WINDOW_STEPS,
) -> list[tuple[float, float]]:
    if window_steps <= 0:
        raise ValueError("window_steps must be positive")
    start_index = max(0, end_index - window_steps + 1)
    return [
        (
            frame.time_s / 3600.0,
            _battery_chart_value(
                _satellite_in_frame(frame, sat_id).battery_j,
                frames.battery_capacity_j,
            ),
        )
        for frame in (frames[index] for index in range(start_index, end_index + 1))
    ]


def _satellite_info(
    satellite: SatellitePosition,
    metadata: SatelliteMetadata,
    battery_capacity_j: float,
    time_label: str,
) -> str:
    latitude, longitude = _eci_to_lat_lon(satellite.position_km, time_label)
    altitude_km = _orbital_altitude_km(satellite.position_km)
    battery = f"{100.0 * satellite.battery_j / battery_capacity_j:.1f}%"
    completed = satellite.local_activity.completed
    failed = satellite.local_activity.failed
    queue_remaining = satellite.local_activity.queue_remaining
    completed_text = "N/A" if completed is None else f"{completed}"
    failed_text = "N/A" if failed is None else f"{failed}"
    queue_text = "N/A" if queue_remaining is None else f"{queue_remaining}"
    return (
        f"Satellite ID: {satellite.sat_id}\n"
        f"Orbit plane: {metadata.plane + 1}\n"
        f"Coordinates\n"
        f"  Latitude: {_format_dms(latitude, positive='N', negative='S')}\n"
        f"  Longitude: {_format_dms(longitude, positive='E', negative='W')}\n"
        f"Orbital altitude: {altitude_km:,.2f} km\n"
        f"Illumination: {'Sunlit' if satellite.sunlit else 'Eclipse'}\n"
        f"Battery: {battery}\n"
        f"New tasks from ground: {satellite.new_ground_tasks}\n"
        f"\nLocal activity this step\n"
        f"  Tasks completed: {completed_text}\n"
        f"  Tasks failed: {failed_text}\n"
        f"  Queue remaining: {queue_text}\n"
        f"  Compute time: {satellite.task_compute_time_s:,.2f} s"
    )


def _set_visibility(
    objects: dict[int, Any],
    frames: StateFrames,
    selection: PlaneSelection,
) -> None:
    for sat_id, obj in objects.items():
        obj.visible = selection.contains(frames.plane_for(sat_id))


def _run_title(output_dir: Path) -> str:
    try:
        manifest = load_run(output_dir)
    except (FileNotFoundError, ValueError):
        return output_dir.name or "Starlit simulation"
    config = manifest.get("config")
    if isinstance(config, dict):
        run = config.get("run")
        if isinstance(run, dict) and isinstance(run.get("name"), str):
            return f"{run['name']}"
    return output_dir.name or "Starlit simulation"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play satellite positions from a Starlit output directory",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="run output directory containing states.jsonl",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=10,
        help="maximum rendered frames per second (default: 10)",
    )
    parser.add_argument(
        "--satellite-radius-km",
        type=float,
        default=60.0,
        help="display radius of satellites in km (default: 60)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        frames = StateFrames(args.output_dir)
        visualizer = Visualizer(
            frames,
            title=_run_title(args.output_dir),
            satellite_radius_km=args.satellite_radius_km,
        )
        visualizer.play(rate=args.rate)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
online visualizer - inject task intensity
updated task failure rate
"""
