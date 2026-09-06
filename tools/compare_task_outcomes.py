from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from matplotlib.ticker import MultipleLocator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.plot_styles import method_style, run_display_label
from tools.plot_output import format_written, save_png_pdf


FALLBACK_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#B279A2",
    "#72B7B2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]
FALLBACK_HATCHES = ["", "//", "\\", "xx", "--", "++", "oo", "**", ".."]


def plot_style(method: str, index: int):
    try:
        return method_style(method)
    except ValueError:
        from types import SimpleNamespace

        return SimpleNamespace(
            method=method,
            label=method,
            color=FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
            alpha=0.75,
            hatch=FALLBACK_HATCHES[index % len(FALLBACK_HATCHES)],
            marker="o",
        )


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
            "font.size": 22,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def default_method(run_dir: Path) -> str:
    return run_display_label(run_dir)


def load_fail_rate(run_dir: Path) -> tuple[int, float]:
    summary = json.loads((run_dir / "summary.json").read_text())
    tasks = summary["tasks"]
    generated = int(tasks["generated"])
    completed = int(tasks["completed"])
    pending = int(tasks["pending"])
    failed = int(tasks["failed"])
    if completed + pending + failed != generated:
        raise ValueError(
            f"{run_dir}: completed + pending + failed does not equal generated"
        )
    rate = 0.0 if generated == 0 else 100.0 * failed / generated
    return failed, rate


def write_figure(path: Path, series: list[dict[str, object]]) -> tuple[Path, Path]:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 5.1))

    styles = [plot_style(str(item["method"]), index) for index, item in enumerate(series)]
    labels = [style.label for style in styles]
    failed = [int(item["failed"]) for item in series]
    rates = [float(item["fail_rate"]) for item in series]
    x = list(range(len(series)))

    for xpos, style, rate in zip(x, styles, rates):
        ax.bar(
            xpos,
            rate,
            0.68,
            facecolor="none",
            edgecolor=style.color,
            hatch=style.hatch,
            linewidth=1.2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("%", rotation=0, va="center", labelpad=18)
    ax.tick_params(axis="y", which="major", left=True, length=7, width=1.0)
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis="y", which="minor", left=True, length=5, width=1.0)
    ax.set_ylim(0.0, 80.0)
    ax.grid(True, axis="y", alpha=0.7)

    for xpos, rate, count in zip(x, rates, failed):
        if count == 0:
            continue

        ax.annotate(
            f"{rate:.1f}%",
            xy=(xpos, rate),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#222222",
            fontsize=22,
        )

    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare task fail rates across runs."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        type=Path,
        help="Run directories with summary.json",
    )
    parser.add_argument("--labels", nargs="*", help="Optional method names for each run")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output figure path or prefix. Writes .png and .pdf.",
    )
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.runs):
        raise ValueError("--labels count must match the number of run directories")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    series = []
    for index, run_dir in enumerate(args.runs):
        failed, fail_rate = load_fail_rate(run_dir)
        series.append(
            {
                "method": (
                    args.labels[index]
                    if args.labels is not None
                    else default_method(run_dir)
                ),
                "failed": failed,
                "fail_rate": fail_rate,
            }
        )

    written = write_figure(args.out, series)
    print(f"Wrote {format_written(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
