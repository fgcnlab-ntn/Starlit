#!/usr/bin/env python3
"""Sample a task-free Walker constellation and plot per-satellite sunlight CDFs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from simulator.geometry import (
    circular_plane,
    circular_state,
    is_sunlit_cylindrical_shadow,
    vector_unit,
)
from simulator.orbit import load_sun_ephemeris, sun_vector_from_ephemeris
from tools.plot_output import save_png_pdf

DEFAULT_CONFIG = ROOT / "configs/sunlit-ratio/constellations.json"


def sample_constellation(config: dict, sun_units: list[tuple[float, float, float]], step_s: int) -> list[dict]:
    if not sun_units or step_s <= 0:
        raise ValueError("at least one Sun sample and a positive step are required")
    satellites = int(config["satellites"])
    planes = int(config["planes"])
    if satellites <= 0 or planes <= 0 or satellites % planes:
        raise ValueError("satellites must be positive and divisible by planes")
    slots = satellites // planes
    radius_km = EARTH_RADIUS_KM + float(config["altitude_km"])
    if radius_km <= EARTH_RADIUS_KM:
        raise ValueError("altitude_km must be positive")
    motion = math.sqrt(EARTH_MU_KM3_S2 / radius_km**3)
    speed_km_s = radius_km * motion
    phase = int(config["walker_phase"])
    raan_spread = float(config["raan_spread_deg"])
    if not 0 < raan_spread <= 360:
        raise ValueError("raan_spread_deg must be within (0, 360]")
    geometry = [
        circular_plane(
            math.radians(float(config["inclination_deg"])),
            math.radians(raan_spread) * plane / planes,
        )
        for plane in range(planes)
    ]
    sunlit_counts = [0] * satellites

    for index, sun_unit in enumerate(sun_units):
        time_s = index * step_s
        for plane, basis in enumerate(geometry):
            plane_phase = math.tau * phase * plane / satellites
            for slot in range(slots):
                sat_id = plane * slots + slot
                argument = math.tau * slot / slots + plane_phase + motion * time_s
                position, _ = circular_state(basis, radius_km, speed_km_s, argument)
                sunlit_counts[sat_id] += is_sunlit_cylindrical_shadow(position, sun_unit)

    samples = len(sun_units)
    return [
        {
            "satellite_id": sat_id,
            "plane": sat_id // slots,
            "slot": sat_id % slots,
            "sunlit_samples": count,
            "samples": samples,
            "sunlit_ratio_pct": 100.0 * count / samples,
        }
        for sat_id, count in enumerate(sunlit_counts)
    ]


def write_cdf(path: Path, results: dict[str, list[dict]]) -> None:
    cache = Path(tempfile.gettempdir()) / "simulator-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name, rows in results.items():
        ratios = sorted(row["sunlit_ratio_pct"] for row in rows)
        x = [ratios[0], *ratios]
        y = [0.0, *(index / len(ratios) for index in range(1, len(ratios) + 1))]
        ax.step(x, y, where="post", label=name, linewidth=2)
    lowest = min(row["sunlit_ratio_pct"] for rows in results.values() for row in rows)
    x_min = max(0, min(95, 5 * math.floor(lowest / 5)))
    ax.set(xlabel="Sunlit ratio (%)", ylabel="CDF", xlim=(x_min, 100), ylim=(0, 1))
    ax.grid(alpha=0.25)
    ax.legend()
    save_png_pdf(fig, path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-utc", default="2026-05-22T12:00:00Z")
    parser.add_argument("--duration-s", type=int, default=86400)
    parser.add_argument("--step-s", type=int, default=30)
    parser.add_argument("--ephemeris", type=Path, default=ROOT / "de440s.bsp")
    parser.add_argument("--out", type=Path, default=ROOT / "output/sunlit-ratio")
    args = parser.parse_args()
    if args.step_s <= 0 or args.duration_s <= 0 or args.duration_s % args.step_s:
        parser.error("duration-s must be positive and divisible by positive step-s")
    if not args.ephemeris.is_file():
        parser.error(f"ephemeris not found: {args.ephemeris}")
    start = dt.datetime.fromisoformat(args.start_utc.replace("Z", "+00:00"))
    if start.tzinfo is None:
        parser.error("start-utc must include a UTC offset")
    start = start.astimezone(dt.timezone.utc)
    constellations = json.loads(args.config.read_text())
    if not isinstance(constellations, dict) or not constellations:
        parser.error("config must contain at least one constellation")

    ephemeris = load_sun_ephemeris(str(args.ephemeris))
    sun_units = []
    for time_s in range(0, args.duration_s, args.step_s):
        vector = sun_vector_from_ephemeris(ephemeris, start + dt.timedelta(seconds=time_s))
        sun_unit = vector_unit(vector)
        if sun_unit is None:
            raise ValueError("Sun vector norm must be positive")
        sun_units.append(sun_unit)

    results = {
        name: sample_constellation(config, sun_units, args.step_s)
        for name, config in constellations.items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "satellite_sunlit_ratios.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "constellation", "satellite_id", "plane", "slot",
            "sunlit_samples", "samples", "sunlit_ratio_pct",
        ])
        writer.writeheader()
        for name, rows in results.items():
            writer.writerows({"constellation": name, **row} for row in rows)
    (args.out / "run_config.json").write_text(
        json.dumps(
            {
                "start_utc": start.isoformat(),
                "duration_s": args.duration_s,
                "step_s": args.step_s,
                "ephemeris": str(args.ephemeris),
                "constellations": constellations,
            },
            indent=2,
        ) + "\n"
    )
    write_cdf(args.out / "satellite_sunlit_ratio_cdf", results)
    print(f"Wrote {sum(map(len, results.values()))} satellite ratios to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
