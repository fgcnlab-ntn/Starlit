from __future__ import annotations

import math

from simulator.constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from simulator.orbit import build_circular_illumination_timeline


def test_circular_timeline_lookups_point_to_exact_sampled_transitions() -> None:
    satellites = 8
    planes = 2
    step_s = 30
    radius_km = EARTH_RADIUS_KM + 550.0
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / radius_km**3)
    timeline = build_circular_illumination_timeline(
        start=None,
        sun_ephemeris=None,
        satellites=satellites,
        planes=planes,
        sats_per_plane=satellites // planes,
        duration_s=6000,
        step_s=step_s,
        radius_km=radius_km,
        inclination_rad=math.radians(53.05),
        raan_spread_rad=2.0 * math.pi,
        walker_phase=1,
        mean_motion=mean_motion,
    )

    for step_index, row in enumerate(timeline.sunlit_by_step):
        for sat_id, is_sunlit in enumerate(row):
            eclipse_step = timeline.next_eclipse_step[step_index][sat_id]
            sunlit_step = timeline.next_sunlit_step[step_index][sat_id]

            if is_sunlit:
                if eclipse_step < 0 or sunlit_step < 0:
                    continue
                assert eclipse_step > step_index
                assert not timeline.sunlit_by_step[eclipse_step][sat_id]
                assert all(
                    timeline.sunlit_by_step[index][sat_id]
                    for index in range(step_index, eclipse_step)
                )
                assert sunlit_step > eclipse_step
            else:
                assert eclipse_step == step_index
                if sunlit_step < 0:
                    continue
                assert sunlit_step > step_index

            assert timeline.sunlit_by_step[sunlit_step][sat_id]
            assert all(
                not timeline.sunlit_by_step[index][sat_id]
                for index in range(eclipse_step, sunlit_step)
            )


def test_circular_timeline_transition_times_are_step_aligned() -> None:
    radius_km = EARTH_RADIUS_KM + 550.0
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / radius_km**3)
    timeline = build_circular_illumination_timeline(
        start=None,
        sun_ephemeris=None,
        satellites=4,
        planes=1,
        sats_per_plane=4,
        duration_s=6000,
        step_s=30,
        radius_km=radius_km,
        inclination_rad=math.radians(53.05),
        raan_spread_rad=2.0 * math.pi,
        walker_phase=0,
        mean_motion=mean_motion,
    )

    for eclipse_row, sunlit_row in zip(
        timeline.next_eclipse_step, timeline.next_sunlit_step
    ):
        for transition_step in (*eclipse_row, *sunlit_row):
            if transition_step >= 0:
                assert (transition_step * timeline.step_s) % timeline.step_s == 0
