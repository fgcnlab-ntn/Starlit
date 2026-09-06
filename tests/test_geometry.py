from __future__ import annotations

import math

import pytest

from simulator.geometry import circular_plane, circular_state


def test_circular_state_in_equatorial_plane() -> None:
    plane = circular_plane(inclination_rad=0.0, raan_rad=0.0)

    position, velocity = circular_state(
        plane,
        radius_km=10.0,
        speed_km_s=2.0,
        argument_rad=0.0,
    )

    assert position == pytest.approx((10.0, 0.0, 0.0))
    assert velocity == pytest.approx((0.0, 2.0, 0.0))


def test_circular_state_in_polar_plane() -> None:
    plane = circular_plane(inclination_rad=math.pi / 2.0, raan_rad=0.0)

    position, velocity = circular_state(
        plane,
        radius_km=10.0,
        speed_km_s=2.0,
        argument_rad=math.pi / 2.0,
    )

    assert position == pytest.approx((0.0, 0.0, 10.0), abs=1.0e-12)
    assert velocity == pytest.approx((-2.0, 0.0, 0.0), abs=1.0e-12)
