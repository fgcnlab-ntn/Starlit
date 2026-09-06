from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator.plot_styles import (
    canonical_method,
    find_method_run_dir,
    method_style,
    ordered_methods,
)
from tools.plot_output import save_png_pdf

DEFAULT_RUN_METHODS = [
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "phoenix",
    "starlit",
]
LOADING_RE = re.compile(r"^r(?P<pct>\d+(?:\.\d+)?)$")


def _pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "simulator-matplotlib"
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
            "font.size": 24,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def loading_pct(group_dir: Path) -> float:
    match = LOADING_RE.match(group_dir.name)
    if not match:
        raise ValueError(f"cannot parse loading percentage from {group_dir.name!r}")
    return float(match.group("pct"))


def task_failure_ratio(summary: dict) -> float:
    tasks = summary["tasks"]
    generated = int(tasks["generated"])
    return 0.0 if generated == 0 else float(tasks["failed"]) / generated


def method_order_map(methods: list[str]) -> dict[str, int]:
    return {
        canonical_method(method): index
        for index, method in enumerate(ordered_methods(methods))
    }


def method_summary_file(group_dir: Path, method: str) -> Path | None:
    run_dir = find_method_run_dir(group_dir, method)
    if run_dir is None:
        return None
    summary_file = run_dir / "summary.json"
    return summary_file if summary_file.exists() else None


def collect_rows(
    base_dir: Path,
    *,
    methods: list[str],
    runs: set[str] | None = None,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for group_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        if group_dir.name == "r30":
            continue
        if runs is not None and group_dir.name not in runs:
            continue
        if not any(method_summary_file(group_dir, method) is not None for method in methods):
            continue
        loading = loading_pct(group_dir)
        for method in methods:
            summary_file = method_summary_file(group_dir, method)
            if summary_file is None:
                continue
            summary = json.loads(summary_file.read_text())
            battery = summary.get("battery_violations", {})
            legacy_objective = summary.get("objective", {})
            if "avg_unsafe_ratio" not in battery:
                raise ValueError(
                    f"{summary_file}: missing battery_violations.avg_unsafe_ratio; "
                    "rerun the simulation with the current logger"
                )
            rows.append(
                {
                    "run": group_dir.name,
                    "method": method,
                    "label": method_style(method).label,
                    "loading_pct": loading,
                    "below_e_safe_ratio": float(
                        battery.get(
                            "avg_eclipse_unsafe_ratio",
                            legacy_objective.get("avg_eclipse_unsafe_ratio", 0.0),
                        )
                    ),
                    "unique_below_e_safe_ratio": float(
                        battery.get("unique_breached_ratio", 0.0)
                    ),
                    "avg_below_e_safe_ratio": float(
                        battery["avg_unsafe_ratio"]
                    ),
                    "task_failure_ratio": task_failure_ratio(summary),
                }
            )
    order_index = method_order_map(methods)
    return sorted(
        rows,
        key=lambda row: (
            float(row["loading_pct"]),
            order_index[canonical_method(str(row["method"]))],
        ),
    )


def write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "method",
        "label",
        "loading_pct",
        "below_e_safe_ratio",
        "unique_below_e_safe_ratio",
        "avg_below_e_safe_ratio",
        "task_failure_ratio",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.12g}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def plot_metric(
    rows: list[dict[str, float | str]],
    path: Path,
    *,
    methods: list[str],
    metric: str,
    ylabel: str,
    y_max: int = 80,
    ylabel_rotation: int = 0,
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.4))

    method_index = method_order_map(methods)
    plotted_methods = sorted(
        {str(row["method"]) for row in rows},
        key=lambda method: method_index[canonical_method(method)],
    )
    for method in plotted_methods:
        points = [row for row in rows if row["method"] == method]
        if not points:
            continue
        xs = [float(row["loading_pct"]) for row in points]
        ys = [100.0 * float(row[metric]) for row in points]
        style = method_style(method)
        label = style.label
        ax.plot(
            xs,
            ys,
            linewidth=2,
            markersize=10,
            label=label,
            color=style.color,
            alpha=style.alpha,
            marker=style.marker,
        )

    ax.set_xlabel("Task intensity (%)")
    ax.set_ylabel(
        ylabel,
        rotation=ylabel_rotation,
        va="center",
        labelpad=16,
    )
    ax.grid(True, axis="both", alpha=0.75)
    x_values = [float(row["loading_pct"]) for row in rows]
    x_min = min(x_values)
    x_max = max(x_values)
    padding = max(1.0, (x_max - x_min) * 0.08)
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_xticks(sorted({float(row["loading_pct"]) for row in rows}))
    ax.set_ylim(-3, y_max)
    ax.set_yticks(list(range(0, y_max + 1, 20)))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis="y", which="major", left=True, length=7, width=1.0)
    ax.tick_params(axis="y", which="minor", left=True, length=5, width=1.0)
    fig.legend(
        ncol=len(plotted_methods),
        loc="upper center",
        bbox_to_anchor=(0.51, 0.97),
        borderaxespad=0.0,
        frameon=True,
        columnspacing=0.4,
        handletextpad=0.4,
        handlelength=1.3,
    )
    fig.subplots_adjust(top=0.84)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_png_pdf(fig, path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot final-run task loading against E_safe breaches and task failures."
    )
    parser.add_argument("--base-dir", type=Path, default=Path("output"))
    parser.add_argument("--out-dir", type=Path, default=Path("output/compare"))
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_RUN_METHODS,
        help="Method directories to include, in plot order.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Optional run group names to include, such as r70 r80 r90.",
    )
    args = parser.parse_args()

    rows = collect_rows(
        args.base_dir,
        methods=args.methods,
        runs=set(args.runs) if args.runs is not None else None,
    )
    if not rows:
        raise SystemExit(f"no run groups found under {args.base_dir}")

    write_csv(rows, args.out_dir / "final-loading-effects.csv")
    plot_metric(
        rows,
        args.out_dir / "final-loading-below-e-safe",
        methods=args.methods,
        metric="avg_below_e_safe_ratio",
        ylabel="Satellites below E_safe (%)",
        y_max=100,
        ylabel_rotation=90,
    )
    plot_metric(
        rows,
        args.out_dir / "final-loading-task-fail-ratio",
        methods=args.methods,
        metric="task_failure_ratio",
        ylabel="%",
    )


if __name__ == "__main__":
    main()
