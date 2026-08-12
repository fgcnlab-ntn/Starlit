from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from satmulator.plot_styles import (
    METHOD_ORDER,
    canonical_method,
    line_kwargs,
    method_style,
)
from tools.plot_output import format_written, save_png_pdf

FIXED_X_MAX = 50.0
X_TICK_STEP = 5.0
Y_TICKS = [i / 5.0 for i in range(6)]
CDF_POINT_LEVELS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def _sort_key(run_name: str) -> int:
    try:
        return METHOD_ORDER.index(canonical_method(run_name))
    except ValueError:
        return len(METHOD_ORDER)


def _pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "satmulator-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "font.size": 11,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def load_capacity_j(run_dir: Path) -> float:
    run = json.loads((run_dir / "run.json").read_text())
    return float(run["config"]["battery"]["capacity_j"])


def default_label(run_dir: Path) -> str:
    return method_style(run_dir.name.replace("_", "-")).label


def load_eclipse_dod_values(run_dir: Path) -> list[float]:
    capacity_j = load_capacity_j(run_dir)
    states_path = run_dir / "states.jsonl"
    values: list[float] = []

    with states_path.open() as f:
        for line in f:
            snapshot = json.loads(line)
            for sat in snapshot.get("satellites", []):
                if sat.get("sunlit", True):
                    continue
                battery_j = float(sat["battery_j"])
                dod_pct = 100.0 * (1.0 - battery_j / capacity_j)
                values.append(dod_pct)
    return values


def build_cdf_curve(values: list[float], *, x_max: float, levels: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []

    sorted_values = sorted(values)
    total = len(sorted_values)
    x_values: list[float] = []
    y_values: list[float] = []

    for level in levels:
        if level <= 0.0:
            rank_index = 0
        elif level >= 1.0:
            rank_index = total - 1
        else:
            rank_index = max(0, min(total - 1, int(level * total) - 1))

        x = sorted_values[rank_index]
        x_values.append(max(0.0, min(x_max, x)))
        y_values.append(level)

    return x_values, y_values


def write_figure(
    path: Path,
    series: list[dict[str, object]],
    *,
    title: str = "CDF of Eclipse Satellite-Time DoD",
) -> tuple[Path, Path]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    for item in series:
        values = item["values"]
        if not values:
            continue
        x_values, cdf = build_cdf_curve(
            values,
            x_max=FIXED_X_MAX,
            levels=CDF_POINT_LEVELS,
        )
        if not x_values:
            continue
        ax.plot(
            x_values,
            cdf,
            linestyle="-",
            linewidth=2.2,
            label=str(item["label"]),
            **line_kwargs(str(item["method"])),
        )

    ax.set_xlabel("Eclipse satellite-time DoD (%)")
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0.0, FIXED_X_MAX)
    ax.set_yticks(Y_TICKS)
    ax.set_xticks([tick for tick in range(0, int(FIXED_X_MAX) + 1, int(X_TICK_STEP))])
    ax.grid(True, alpha=0.7)
    ax.legend(loc="lower right", framealpha=0.94)
    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare eclipse satellite-time DoD CDFs across multiple runs."
    )
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories with states.jsonl")
    parser.add_argument("--labels", nargs="*", help="Optional labels for each run")
    parser.add_argument(
        "--title",
        default="CDF of Eclipse Satellite-Time DoD",
        help="Figure title",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path or prefix for the combined CDF figure. Writes .png and .pdf.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.runs):
        raise ValueError("--labels count must match the number of run directories")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    series = []
    for index, run_dir in enumerate(args.runs):
        run_name = run_dir.name.replace("_", "-")
        method_name = canonical_method(run_name)
        if args.labels is not None:
            label = args.labels[index]
        else:
            label = method_style(method_name).label
        item = {
            "label": label,
            "values": load_eclipse_dod_values(run_dir),
            "style": method_style(method_name),
            "method": method_name,
            "sort_key": _sort_key(run_name),
            "input_index": index,
        }
        series.append(item)

    series.sort(key=lambda item: (int(item["sort_key"]), int(item["input_index"])))

    written = write_figure(args.out, series, title=args.title)
    print(f"Wrote {format_written(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
