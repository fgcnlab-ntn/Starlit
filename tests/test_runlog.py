from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from satmulator.models import SatelliteState
from satmulator.runlog import RunLog


def satellite_state(time_s: int, sat_id: int, *, safe: bool) -> SatelliteState:
    return SatelliteState(
        time_s=time_s,
        sat_id=sat_id,
        name=f"sat-{sat_id}",
        plane=0,
        slot=sat_id,
        x_km=0.0,
        y_km=0.0,
        z_km=0.0,
        vx_km_s=0.0,
        vy_km_s=0.0,
        vz_km_s=0.0,
        lat_deg=None,
        lon_deg=None,
        elevation_km=None,
        sunlit=True,
        battery_j=100.0 if safe else 40.0,
        battery_pct=100.0 if safe else 40.0,
        harvested_j=0.0,
        consumed_j=0.0,
        safe_battery=safe,
        generated_tasks=0,
        completed_tasks=0,
        failed_tasks=0,
        task_energy_j=0.0,
    )


def test_summary_averages_unsafe_ratio_over_all_satellites(
    tmp_path: Path,
) -> None:
    runlog = RunLog(
        tmp_path,
        dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        {"logging": {"task_events": "off", "state_steps": "off"}},
    )
    runlog.write_step(
        [
            satellite_state(0, 0, safe=False),
            satellite_state(0, 1, safe=True),
        ]
    )
    runlog.write_step(
        [
            satellite_state(30, 0, safe=False),
            satellite_state(30, 1, safe=False),
        ]
    )

    summary = runlog.complete()

    assert summary["battery_violations"]["avg_unsafe_ratio"] == pytest.approx(
        0.75
    )
