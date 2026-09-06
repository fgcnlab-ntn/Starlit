from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from .models import ComputeConfig, DemandDistribution, DemandPoint, Task, TaskConfig
from .route_cost import compute_cycles
from .runtime import EnvironmentRuntime, SatelliteRuntime

DEADLINE_SEED_SALT = 0x444541444C494E45


def demand_distribution(points: Iterable[DemandPoint]) -> DemandDistribution:
    loaded = tuple(points)
    cumulative: list[float] = []
    total = 0.0
    for point in loaded:
        if not math.isfinite(point.weight) or point.weight <= 0.0:
            raise ValueError("demand distribution weights must be finite and positive")
        total += point.weight
        cumulative.append(total)
    return DemandDistribution(loaded, tuple(cumulative), total)


def load_demand_points(path: Path) -> DemandDistribution:
    points: list[DemandPoint] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"lat", "lon", "weight"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("demand points CSV must contain lat, lon, weight columns")
        for row in reader:
            lat = float(row["lat"])
            lon = float(row["lon"])
            weight = float(row["weight"])
            if not -90.0 <= lat <= 90.0:
                raise ValueError(f"invalid demand latitude: {lat}")
            if not -180.0 <= lon <= 180.0:
                raise ValueError(f"invalid demand longitude: {lon}")
            if weight < 0.0:
                raise ValueError(f"invalid demand weight: {weight}")
            if weight > 0.0:
                points.append(DemandPoint(lat_deg=lat, lon_deg=lon, weight=weight))
    if not points:
        raise ValueError(f"no positive demand points in {path}")
    return demand_distribution(points)


def demand_points_provenance(path: Path) -> dict[str, object]:
    distribution = load_demand_points(path)
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    conversion: dict[str, object] | None = None
    if metadata_path.exists():
        value = json.loads(metadata_path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"demand-point metadata must be an object: {metadata_path}")
        conversion = value
    return {
        "file": str(path),
        "points": len(distribution.points),
        "total_weight": distribution.total_weight,
        "metadata_file": str(metadata_path) if metadata_path.exists() else None,
        "conversion": conversion,
    }


def validate_task_config(task_config: TaskConfig) -> None:
    if task_config.interval_s <= 0:
        raise ValueError("task interval must be positive")
    if task_config.tasks_per_step < 0:
        raise ValueError("tasks per step must be non-negative")
    if task_config.input_bits < 0 or task_config.output_bits < 0:
        raise ValueError("task input/output bits must be non-negative")
    if task_config.deadline_s <= 0:
        raise ValueError("task deadline must be positive")
    if task_config.deadline_min_s <= 0:
        raise ValueError("task minimum deadline must be positive")
    if task_config.deadline_min_s >= task_config.deadline_s:
        raise ValueError(
            "task minimum deadline must be less than the mean normal deadline"
        )
    if not 0.0 <= task_config.min_elevation_deg <= 90.0:
        raise ValueError("minimum elevation must be within [0, 90]")
    if not task_config.demand_distribution:
        raise ValueError("demand-points task generation requires a demand_points_file")


def deadline_random_seed(random_seed: int | None) -> int | None:
    """Derive an independent seed without perturbing the workload RNG stream."""

    return None if random_seed is None else random_seed ^ DEADLINE_SEED_SALT


def sample_deadline_s(env: EnvironmentRuntime, task_config: TaskConfig) -> float:
    minimum_s = task_config.deadline_min_s
    scale_s = (task_config.deadline_s - minimum_s) / 3.0
    while True:
        deadline_s = env.deadline_rng.normalvariate(
            task_config.deadline_s,
            scale_s,
        )
        if deadline_s >= minimum_s:
            return deadline_s


def choose_demand_point(
    rng: random.Random,
    distribution: DemandDistribution,
) -> DemandPoint:
    return rng.choices(
        distribution.points,
        cum_weights=distribution.cumulative_weights,
        k=1,
    )[0]


@lru_cache(maxsize=1)
def skyfield_timescale():
    try:
        from skyfield.api import load
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for demand-point coordinates. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return load.timescale()


@lru_cache(maxsize=1)
def skyfield_visibility_modules():
    try:
        from skyfield.api import wgs84
        from skyfield.constants import AU_KM
        from skyfield.positionlib import Geocentric
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for demand-point visibility. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return wgs84, AU_KM, Geocentric


def ground_position_km(point: DemandPoint, time_utc: dt.datetime) -> tuple[float, float, float]:
    wgs84, _, _ = skyfield_visibility_modules()
    t = skyfield_timescale().from_datetime(time_utc)
    position = wgs84.latlon(point.lat_deg, point.lon_deg).at(t).position.km
    return tuple(float(component) for component in position)


def satellite_altitude_distance(
    sat: SatelliteRuntime,
    point: DemandPoint,
    time_utc: dt.datetime,
) -> tuple[float, float]:
    wgs84, au_km, geocentric = skyfield_visibility_modules()
    t = skyfield_timescale().from_datetime(time_utc)
    ground = wgs84.latlon(point.lat_deg, point.lon_deg).at(t)
    return satellite_altitude_distance_from_ground(sat, t, ground, au_km, geocentric)


def satellite_altitude_distance_from_ground(
    sat: SatelliteRuntime,
    t,
    ground,
    au_km: float,
    geocentric,
) -> tuple[float, float]:
    satellite = geocentric(
        [component / au_km for component in sat.pos_km],
        t=t,
    )
    altitude, _, distance = (satellite - ground).altaz()
    return float(altitude.degrees), float(distance.km)


def nearest_satellite_id(
    satellites: Iterable[SatelliteRuntime],
    point: DemandPoint,
    time_utc: dt.datetime,
    min_elevation_deg: float = 30.0,
) -> int | None:
    best_sat_id = -1
    best_distance_km = float("inf")
    wgs84, au_km, geocentric = skyfield_visibility_modules()
    t = skyfield_timescale().from_datetime(time_utc)
    ground = wgs84.latlon(point.lat_deg, point.lon_deg).at(t)
    for sat in satellites:
        altitude_deg, distance_km = satellite_altitude_distance_from_ground(
            sat, t, ground, au_km, geocentric
        )
        if altitude_deg >= min_elevation_deg and distance_km < best_distance_km:
            best_sat_id = sat.sat_id
            best_distance_km = distance_km
    return None if best_sat_id < 0 else best_sat_id


def nearest_satellite_ids_vectorized(
    satellites: Sequence[SatelliteRuntime],
    points: Sequence[DemandPoint],
    time_utc: dt.datetime,
    min_elevation_deg: float = 30.0,
) -> list[int | None]:
    """Find nearest visible satellites for many ground points at one time.

    The scalar path above calls Skyfield ``altaz()`` for every
    point/satellite pair.  That is the wrong data structure for a sweep:
    geometry is a matrix.  Build the ground-position and satellite-position
    matrices once, then use dot products for elevation and vector norms for
    distance.
    """

    if not points:
        return []
    if not satellites:
        return [None for _ in points]

    try:
        import numpy as np
    except ImportError:
        return [
            nearest_satellite_id(
                satellites,
                point,
                time_utc,
                min_elevation_deg,
            )
            for point in points
        ]

    ground_positions = np.array(
        [ground_position_km(point, time_utc) for point in points],
        dtype=float,
    )
    satellite_positions = np.array([sat.pos_km for sat in satellites], dtype=float)
    satellite_ids = np.array([sat.sat_id for sat in satellites], dtype=int)
    satellite_norms = np.linalg.norm(satellite_positions, axis=1)
    valid_satellites = satellite_norms > 0.0
    if not np.any(valid_satellites):
        return [None for _ in points]
    satellite_positions = satellite_positions[valid_satellites]
    satellite_ids = satellite_ids[valid_satellites]
    satellite_norms = satellite_norms[valid_satellites]

    ground_norms = np.linalg.norm(ground_positions, axis=1)
    if np.any(ground_norms <= 0.0):
        raise ValueError("ground position norm must be positive")
    local_up = ground_positions / ground_norms[:, None]
    satellite_up = satellite_positions / satellite_norms[:, None]
    candidate_mask = elevation_candidate_mask(
        local_up,
        satellite_up,
        ground_norms,
        satellite_norms,
        min_elevation_deg,
    )

    min_sin_elevation = math.sin(math.radians(min_elevation_deg))
    result: list[int | None] = []
    for point_index, point_position in enumerate(ground_positions):
        candidate_indices = np.flatnonzero(candidate_mask[point_index])
        if len(candidate_indices) == 0:
            result.append(None)
            continue

        line_of_sight = satellite_positions[candidate_indices] - point_position
        distances = np.linalg.norm(line_of_sight, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sin_elevation = line_of_sight @ local_up[point_index] / distances

        visible = np.isfinite(sin_elevation) & (sin_elevation >= min_sin_elevation)
        if not np.any(visible):
            result.append(None)
            continue

        visible_indices = candidate_indices[visible]
        visible_distances = distances[visible]
        nearest_visible_index = int(np.argmin(visible_distances))
        result.append(int(satellite_ids[int(visible_indices[nearest_visible_index])]))
    return result


def elevation_candidate_mask(
    ground_unit_vectors,
    satellite_unit_vectors,
    ground_radii_km,
    satellite_radii_km,
    min_elevation_deg: float,
):
    """Return point/satellite pairs that can satisfy the elevation limit.

    This is the cheap rejection pass.  Visibility only depends on the central
    angle between the ground radial vector and the satellite radial vector,
    plus the two radii.  A dot-product matrix gives that angle without
    allocating the full point/satellite line-of-sight vector matrix.
    """

    import numpy as np

    min_elevation_rad = math.radians(min_elevation_deg)
    cos_elevation = math.cos(min_elevation_rad)
    sin_elevation = math.sin(min_elevation_rad)

    radius_ratio = ground_radii_km[:, None] / satellite_radii_km[None, :]
    horizon_term = np.maximum(0.0, 1.0 - (radius_ratio * cos_elevation) ** 2)
    min_central_cosine = (
        radius_ratio * cos_elevation * cos_elevation
        + sin_elevation * np.sqrt(horizon_term)
    )
    central_cosine = ground_unit_vectors @ satellite_unit_vectors.T
    return central_cosine >= min_central_cosine


def generate_step_tasks(
    env: EnvironmentRuntime,
    task_config: TaskConfig,
    compute_config: ComputeConfig,
) -> tuple[list[Task], list[Task]]:
    if env.time_s > 0 and env.time_s % task_config.interval_s == 0:
        env.pending_tasks.extend(
            generate_demand_point_tasks(env, task_config, compute_config)
        )
    return resolve_pending_tasks(env, task_config)


def generate_demand_point_tasks(
    env: EnvironmentRuntime,
    task_config: TaskConfig,
    compute_config: ComputeConfig,
) -> list[Task]:
    if env.time_utc is None:
        raise ValueError("demand-point task generation requires an absolute UTC simulation time")
    tasks: list[Task] = []
    for _ in range(task_config.tasks_per_step):
        point = choose_demand_point(env.rng, task_config.demand_distribution)
        task = Task(
            task_id=env.next_task_id,
            created_time_s=env.time_s,
            source_sat=None,
            input_bits=task_config.input_bits,
            output_bits=task_config.output_bits,
            deadline_s=sample_deadline_s(env, task_config),
            lat_deg=point.lat_deg,
            lon_deg=point.lon_deg,
        )
        tasks.append(task)
        emit_generated_task(env, task, compute_config)
        env.next_task_id += 1
    return tasks


def emit_generated_task(
    env: EnvironmentRuntime,
    task: Task,
    compute_config: ComputeConfig,
) -> None:
    env.emit_task_event(
        "task_generated",
        task.task_id,
        source_sat=task.source_sat,
        location=(
            None
            if task.lat_deg is None
            else {"lat_deg": task.lat_deg, "lon_deg": task.lon_deg}
        ),
        workload={
            "compute_cycles": compute_cycles(task, compute_config),
            "compute_time_s": None,
            "input_bits": task.input_bits,
            "output_bits": task.output_bits,
        },
        deadline_s=task.deadline_s,
    )


def resolve_pending_tasks(
    env: EnvironmentRuntime,
    task_config: TaskConfig,
) -> tuple[list[Task], list[Task]]:
    if env.time_utc is None:
        raise ValueError("demand-point task generation requires an absolute UTC simulation time")
    ready: list[Task] = []
    expired: list[Task] = []
    still_pending: list[Task] = []

    unique_points_by_coord: dict[tuple[float, float], DemandPoint] = {}
    for task in env.pending_tasks:
        if task.lat_deg is None or task.lon_deg is None:
            raise ValueError("pending demand-point task requires latitude and longitude")
        coord = (task.lat_deg, task.lon_deg)
        unique_points_by_coord.setdefault(
            coord,
            DemandPoint(lat_deg=task.lat_deg, lon_deg=task.lon_deg, weight=1.0),
        )

    unique_coords = tuple(unique_points_by_coord)
    source_by_coord = dict(
        zip(
            unique_coords,
            nearest_satellite_ids_vectorized(
                env.satellites,
                tuple(unique_points_by_coord.values()),
                env.time_utc,
                task_config.min_elevation_deg,
            ),
        )
    )

    for task in env.pending_tasks:
        assert task.lat_deg is not None
        assert task.lon_deg is not None
        source_sat = source_by_coord[(task.lat_deg, task.lon_deg)]
        if source_sat is not None:
            env.emit_task_event(
                "task_coverage_acquired",
                task.task_id,
                source_sat=source_sat,
                waiting_time_s=env.time_s - task.created_time_s,
            )
            ready.append(
                Task(
                    task_id=task.task_id,
                    created_time_s=task.created_time_s,
                    source_sat=source_sat,
                    input_bits=task.input_bits,
                    output_bits=task.output_bits,
                    deadline_s=task.deadline_s,
                    lat_deg=task.lat_deg,
                    lon_deg=task.lon_deg,
                )
            )
        elif env.time_s - task.created_time_s >= task.deadline_s:
            expired.append(task)
        else:
            env.emit_task_event(
                "task_waiting_for_coverage",
                task.task_id,
                waiting_time_s=env.time_s - task.created_time_s,
            )
            still_pending.append(task)
    env.pending_tasks = still_pending
    return ready, expired
