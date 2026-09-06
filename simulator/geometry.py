from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import EARTH_RADIUS_KM


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class CircularPlane:
    radial_basis: Vector3
    transverse_basis: Vector3


def circular_plane(inclination_rad: float, raan_rad: float) -> CircularPlane:
    sin_inclination = math.sin(inclination_rad)
    cos_inclination = math.cos(inclination_rad)
    sin_raan = math.sin(raan_rad)
    cos_raan = math.cos(raan_rad)
    return CircularPlane(
        radial_basis=(cos_raan, sin_raan, 0.0),
        transverse_basis=(
            -sin_raan * cos_inclination,
            cos_raan * cos_inclination,
            sin_inclination,
        ),
    )


def circular_state(
    plane: CircularPlane,
    radius_km: float,
    speed_km_s: float,
    argument_rad: float,
) -> tuple[Vector3, Vector3]:
    radial_x, radial_y, radial_z = plane.radial_basis
    transverse_x, transverse_y, transverse_z = plane.transverse_basis
    cos_argument = math.cos(argument_rad)
    sin_argument = math.sin(argument_rad)
    position = (
        radius_km * (cos_argument * radial_x + sin_argument * transverse_x),
        radius_km * (cos_argument * radial_y + sin_argument * transverse_y),
        radius_km * (cos_argument * radial_z + sin_argument * transverse_z),
    )
    velocity = (
        speed_km_s * (-sin_argument * radial_x + cos_argument * transverse_x),
        speed_km_s * (-sin_argument * radial_y + cos_argument * transverse_y),
        speed_km_s * (-sin_argument * radial_z + cos_argument * transverse_z),
    )
    return position, velocity


def xy_unit(v: Vector3) -> tuple[float, float] | None:
    x, y, _ = v
    norm = math.hypot(x, y)
    if norm == 0:
        return None
    return (x / norm, y / norm)


def vector_unit(v: Vector3) -> Vector3 | None:
    norm = math.sqrt(sum(component * component for component in v))
    if norm == 0:
        return None
    return tuple(component / norm for component in v)


def is_sunlit_cylindrical_shadow(
    pos_km: Vector3,
    sun_unit: Vector3 = (1.0, 0.0, 0.0),
) -> bool:
    x, y, z = pos_km
    sx, sy, sz = sun_unit
    along_sun = x * sx + y * sy + z * sz
    if along_sun >= 0:
        return True
    px = x - along_sun * sx
    py = y - along_sun * sy
    pz = z - along_sun * sz
    perpendicular_distance_sq = px * px + py * py + pz * pz
    return perpendicular_distance_sq >= EARTH_RADIUS_KM * EARTH_RADIUS_KM
