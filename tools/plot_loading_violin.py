from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cache_dir = Path(tempfile.gettempdir()) / "simulator-matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from simulator.plot_styles import EDGE_COLOR, find_method_run_dir, method_style
from tools.plot_output import format_written, save_png_pdf


RUN_METHODS = (
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "phoenix",
    "starlit",
)
COMPLETED_CACHE_NAME = "loading-completed-tasks.csv"
ILLUMINATION_CACHE_NAME = "loading-illumination-compute-ratio.csv"

sns.set_theme(
    context="paper",
    style="whitegrid",
    rc={
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "font.size": 25,
        "axes.labelsize": 25,
        "axes.titlesize": 25,
        "xtick.labelsize": 25,
        "ytick.labelsize": 25,
        "legend.fontsize": 25,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.8,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    },
)


def discover_run_dirs(base_dir: Path) -> list[Path]:
    return [
        run_dir
        for method in RUN_METHODS
        if (run_dir := find_method_run_dir(base_dir, method)) is not None
    ]


def load_run(run_dir: Path) -> dict:
    path = run_dir / "run.json"
    if not path.exists():
        raise FileNotFoundError(f"missing run.json: {path}")
    return json.loads(path.read_text())


def satellite_count(run: dict) -> int:
    return int(run["config"]["orbit"]["satellites"])


def iter_snapshots(run_dir: Path):
    path = run_dir / "states.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing states.jsonl: {path}")
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc


def aggregate_completed_tasks(run_dir: Path) -> list[float]:
    count = satellite_count(load_run(run_dir))
    values = [0.0] * count
    for line_number, snapshot in iter_snapshots(run_dir):
        for satellite in snapshot.get("satellites", []):
            sat_id = int(satellite["id"])
            if not 0 <= sat_id < count:
                raise ValueError(
                    f"{run_dir}/states.jsonl:{line_number}: invalid satellite id {sat_id}"
                )
            values[sat_id] += float(satellite["task_counts"]["completed"])
    return values


def aggregate_illumination_utilization(run_dir: Path) -> dict[str, list[float]]:
    run = load_run(run_dir)
    count = satellite_count(run)
    step_s = float(run["config"]["time"]["step_s"])
    if step_s <= 0.0:
        raise ValueError(f"{run_dir}: time.step_s must be positive")

    compute_s = {
        "sunlit": [0.0] * count,
        "eclipse": [0.0] * count,
    }
    duration_s = {
        "sunlit": [0.0] * count,
        "eclipse": [0.0] * count,
    }
    for line_number, snapshot in iter_snapshots(run_dir):
        if float(snapshot.get("time_s", 0.0)) <= 0.0:
            continue
        for satellite in snapshot.get("satellites", []):
            sat_id = int(satellite["id"])
            if not 0 <= sat_id < count:
                raise ValueError(
                    f"{run_dir}/states.jsonl:{line_number}: invalid satellite id {sat_id}"
                )
            task_load = satellite.get("task_load")
            if not isinstance(task_load, dict) or "compute_time_s" not in task_load:
                raise ValueError(
                    f"{run_dir}/states.jsonl:{line_number}: satellite {sat_id} "
                    "has no task_load.compute_time_s"
                )
            state = "sunlit" if bool(satellite["sunlit"]) else "eclipse"
            compute_s[state][sat_id] += float(task_load["compute_time_s"])
            duration_s[state][sat_id] += step_s

    values = {"sunlit": [], "eclipse": []}
    for state in values:
        values[state] = [
            0.0 if duration == 0.0 else compute / duration
            for compute, duration in zip(compute_s[state], duration_s[state])
        ]
    return values


def load_or_build_completed_tasks(
    run_dir: Path,
    *,
    use_cache: bool,
) -> list[float]:
    path = run_dir / COMPLETED_CACHE_NAME
    if use_cache and path.exists():
        with path.open(newline="") as stream:
            return [float(row["loading"]) for row in csv.DictReader(stream)]

    values = aggregate_completed_tasks(run_dir)
    if use_cache:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["sat_id", "loading"])
            writer.writeheader()
            for sat_id, value in enumerate(values):
                writer.writerow({"sat_id": sat_id, "loading": f"{value:.12g}"})
    return values


def load_or_build_illumination_utilization(
    run_dir: Path,
    *,
    use_cache: bool,
) -> dict[str, list[float]]:
    path = run_dir / ILLUMINATION_CACHE_NAME
    if use_cache and path.exists():
        values = {"sunlit": [], "eclipse": []}
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                values["sunlit"].append(float(row["sunlit"]))
                values["eclipse"].append(float(row["eclipse"]))
        return values

    values = aggregate_illumination_utilization(run_dir)
    if use_cache:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["sat_id", "sunlit", "eclipse"],
            )
            writer.writeheader()
            for sat_id, (sunlit, eclipse) in enumerate(
                zip(values["sunlit"], values["eclipse"])
            ):
                writer.writerow(
                    {
                        "sat_id": sat_id,
                        "sunlit": f"{sunlit:.12g}",
                        "eclipse": f"{eclipse:.12g}",
                    }
                )
    return values


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def gini(values: list[float]) -> float:
    ordered = sorted(max(0.0, value) for value in values)
    total = sum(ordered)
    if not ordered:
        return math.nan
    if total == 0.0:
        return 0.0
    weighted = sum((index + 1) * value for index, value in enumerate(ordered))
    count = len(ordered)
    return (2.0 * weighted) / (count * total) - (count + 1.0) / count


def statistics_row(method: str, values: list[float]) -> dict[str, object]:
    return {
        "method": method,
        "satellites": len(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "max": max(values),
        "gini": gini(values),
    }


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "satellites", "mean", "median", "p95", "max", "gini"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.12g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def plot_completed_tasks(
    path: Path,
    series: list[dict[str, object]],
) -> tuple[Path, Path]:
    labels = [str(item["label"]) for item in series]
    method_column: list[str] = []
    value_column: list[float] = []
    for item in series:
        values = list(item["values"])
        method_column.extend([str(item["label"])] * len(values))
        value_column.extend(values)

    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    sns.violinplot(
        data={"method": method_column, "completed": value_column},
        x="method",
        y="completed",
        order=labels,
        color="white",
        density_norm="width",
        cut=0,
        inner=None,
        linewidth=1.0,
        ax=ax,
    )
    bodies = [
        collection
        for collection in ax.collections
        if isinstance(collection, PolyCollection)
    ]
    for body, item in zip(bodies, series):
        style = method_style(str(item["method"]))
        body.set_facecolor("none")
        body.set_edgecolor(style.color)
        body.set_hatch(style.hatch)
        body.set_linewidth(1.2)
        body.set_alpha(1.0)

    for position, item in enumerate(series):
        median = percentile(list(item["values"]), 50.0)
        ax.hlines(
            median,
            position - 0.18,
            position + 0.18,
            color=EDGE_COLOR,
            linewidth=2.0,
            zorder=3,
        )

    ax.set_xlabel("")
    ax.set_ylabel(r"$10^3$ tasks / Sat.")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:g}")
    )
    ax.yaxis.set_major_locator(MultipleLocator(2000))
    ax.yaxis.set_minor_locator(MultipleLocator(1000))
    ax.tick_params(axis="y", which="major", left=True, length=7, width=1.0)
    ax.tick_params(axis="y", which="minor", left=True, length=5, width=1.0)
    ax.grid(True, axis="y", alpha=0.7)
    ax.legend(
        handles=[Line2D([0], [0], color=EDGE_COLOR, linewidth=2.0, label="Median")],
        loc="upper left",
        borderpad=0.3,
    )
    ax.set_ylim(0, 14_000)
    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def plot_illumination_utilization(
    path: Path,
    series: list[dict[str, object]],
) -> tuple[Path, Path]:
    labels = [str(item["label"]) for item in series]
    method_column: list[str] = []
    state_column: list[str] = []
    value_column: list[float] = []
    state_labels = {
        "sunlit": "Sunlit",
        "eclipse": "Eclipse",
    }
    colors = {
        "Sunlit": "#F7D077",
        "Eclipse": "#7534AD",
    }
    for item in series:
        values_by_state = item["values_by_state"]
        if not isinstance(values_by_state, dict):
            raise TypeError("illumination series is missing values_by_state")
        for state in ("sunlit", "eclipse"):
            values = list(values_by_state[state])
            if len(values) < 2 or math.isclose(
                min(values),
                max(values),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                continue
            method_column.extend([str(item["label"])] * len(values))
            state_column.extend([state_labels[state]] * len(values))
            value_column.extend(values)

    fig, ax = plt.subplots(figsize=(9.4, 6.3))
    sns.violinplot(
        data={
            "method": method_column,
            "illumination": state_column,
            "utilization": value_column,
        },
        x="method",
        y="utilization",
        hue="illumination",
        order=labels,
        hue_order=["Sunlit", "Eclipse"],
        palette=colors,
        split=True,
        density_norm="width",
        common_norm=False,
        width=0.9,
        cut=0,
        inner=None,
        linewidth=1.0,
        saturation=1.0,
        legend=False,
        ax=ax,
    )
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_edgecolor(EDGE_COLOR)
            collection.set_alpha(0.62)

    offsets = {"sunlit": -0.18, "eclipse": 0.18}
    for position, item in enumerate(series):
        values_by_state = item["values_by_state"]
        if not isinstance(values_by_state, dict):
            raise TypeError("illumination series is missing values_by_state")
        for state in ("sunlit", "eclipse"):
            median = percentile(list(values_by_state[state]), 50.0)
            center = position + offsets[state]
            ax.hlines(
                median,
                center - 0.10,
                center + 0.10,
                color=EDGE_COLOR,
                linewidth=2.0,
                zorder=3,
            )

    fig.legend(
        handles=[
            Patch(
                facecolor=colors["Sunlit"],
                edgecolor=EDGE_COLOR,
                alpha=0.62,
                label="Sunlit",
            ),
            Patch(
                facecolor=colors["Eclipse"],
                edgecolor=EDGE_COLOR,
                alpha=0.62,
                label="Eclipse",
            ),
            Line2D([0], [0], color=EDGE_COLOR, linewidth=2.0, label="Median"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.51, 0.98),
        framealpha=0.94,
        ncols=3,
        columnspacing=1.0,
        # handletextpad=0.4,
        # handlelength=1.5,
        # borderpad=0.3,
    )
    ax.set_xlabel("")
    ax.set_ylabel("%", rotation=0, va="center", labelpad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{100 * value:g}"))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(axis="y", which="major", left=True, length=7, width=1.0)
    ax.tick_params(axis="y", which="minor", left=True, length=5, width=1.0)
    ax.grid(True, axis="y", alpha=0.7)
    ax.margins(y=0.08)
    ax.set_ylim(bottom=0.0)
    fig.subplots_adjust(top=0.84)
    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def build_completed_series(
    run_dirs: Iterable[Path],
    *,
    labels: list[str] | None,
    use_cache: bool,
) -> list[dict[str, object]]:
    series = []
    for index, run_dir in enumerate(run_dirs):
        method = run_dir.name
        series.append(
            {
                "method": method,
                "label": labels[index] if labels is not None else method_style(method).label,
                "values": load_or_build_completed_tasks(
                    run_dir,
                    use_cache=use_cache,
                ),
            }
        )
    return series


def build_illumination_series(
    run_dirs: Iterable[Path],
    *,
    labels: list[str] | None,
    use_cache: bool,
) -> list[dict[str, object]]:
    series = []
    for index, run_dir in enumerate(run_dirs):
        method = run_dir.name
        series.append(
            {
                "method": method,
                "label": labels[index] if labels is not None else method_style(method).label,
                "values_by_state": load_or_build_illumination_utilization(
                    run_dir,
                    use_cache=use_cache,
                ),
            }
        )
    return series


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-satellite completed tasks or sunlit/eclipse compute "
            "utilization across scheduler runs."
        )
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Directory containing one run subdirectory per method.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        type=Path,
        help="Explicit run directories; defaults to the standard five methods.",
    )
    parser.add_argument("--labels", nargs="*", help="Labels matching the run count")
    parser.add_argument(
        "--plot",
        choices=["illumination-relative", "total"],
        default="illumination-relative",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output figure path or prefix; writes PNG and PDF.",
    )
    parser.add_argument("--summary-csv", type=Path, help="Summary CSV path")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write per-run aggregation caches.",
    )
    args = parser.parse_args()

    run_dirs = args.runs if args.runs is not None else discover_run_dirs(args.base_dir)
    if not run_dirs:
        raise ValueError(f"no run directories found under {args.base_dir}")
    if args.labels is not None and len(args.labels) != len(run_dirs):
        raise ValueError("--labels count must match the number of runs")

    use_cache = not args.no_cache
    if args.plot == "total":
        series = build_completed_series(
            run_dirs,
            labels=args.labels,
            use_cache=use_cache,
        )
        out = args.out or args.base_dir / "loading-violin"
        written = plot_completed_tasks(out, series)
        summary_rows = [
            statistics_row(str(item["method"]), list(item["values"]))
            for item in series
        ]
    else:
        series = build_illumination_series(
            run_dirs,
            labels=args.labels,
            use_cache=use_cache,
        )
        out = args.out or args.base_dir / "loading-illumination-violin"
        written = plot_illumination_utilization(out, series)
        summary_rows = []
        for item in series:
            values_by_state = item["values_by_state"]
            if not isinstance(values_by_state, dict):
                raise TypeError("illumination series is missing values_by_state")
            for state in ("sunlit", "eclipse"):
                row = statistics_row(str(item["method"]), list(values_by_state[state]))
                row["illumination"] = state
                summary_rows.append(row)

    summary_path = args.summary_csv or out.with_name(f"{out.stem}-summary.csv")
    if args.plot == "illumination-relative":
        fields = [
            "method",
            "illumination",
            "satellites",
            "mean",
            "median",
            "p95",
            "max",
            "gini",
        ]
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(
                    {
                        key: f"{value:.12g}" if isinstance(value, float) else value
                        for key, value in row.items()
                    }
                )
    else:
        write_summary_csv(summary_path, summary_rows)

    print(f"Wrote {format_written(written)}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
