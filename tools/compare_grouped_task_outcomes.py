from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.plot_styles import find_method_run_dir, method_style
from tools.plot_output import format_written, save_png_pdf


DEFAULT_METHODS = [
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "phoenix",
    "starlit",
]
MIN_GROUPS = 4
MAX_GROUPS = 18
METHOD_COUNT = 5
BASE_FONT_SIZE = 25
Y_AXIS_LABEL_FONT_SIZE = 25
Y_AXIS_TICK_FONT_SIZE = 25
DEFAULT_GROUP_LABEL_FONT_SIZE = 25
LEGEND_FONT_SIZE = 26
X_AXIS_LABEL_FONT_SIZE = 26
SEASON_LABELS = {
    "spring-equinox": "Spring\nequinox",
    "summer-solstice": "Summer\nsolstice",
    "autumn-equinox": "Autumn\nequinox",
    "winter-solstice": "Winter\nsolstice",
}
SEASON_LABEL_ALIASES = {
    "spring": "spring-equinox",
    "summer": "summer-solstice",
    "autumn": "autumn-equinox",
    "winter": "winter-solstice",
}


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
            "font.size": BASE_FONT_SIZE,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )
    return plt


def validate_groups(groups: list[str]) -> None:
    if not MIN_GROUPS <= len(groups) <= MAX_GROUPS:
        raise ValueError(
            f"expected {MIN_GROUPS} to {MAX_GROUPS} groups, got {len(groups)}"
        )
    if len(set(groups)) != len(groups):
        raise ValueError("group names must be unique")


def validate_methods(methods: list[str]) -> None:
    if len(methods) != METHOD_COUNT:
        raise ValueError(f"expected {METHOD_COUNT} methods, got {len(methods)}")
    canonical_methods = [method_style(method).method for method in methods]
    if len(set(canonical_methods)) != len(canonical_methods):
        raise ValueError("methods must be unique")


def format_group_labels(group_labels: list[str]) -> tuple[list[str], bool]:
    raw_keys = [
        "-".join(label.replace("\n", " ").lower().split())
        for label in group_labels
    ]
    keys = [SEASON_LABEL_ALIASES.get(key, key) for key in raw_keys]
    is_four_season = len(keys) == 4 and set(keys) == set(SEASON_LABELS)
    if is_four_season:
        return [SEASON_LABELS[key] for key in keys], True
    if len(group_labels) == 5:
        return [
            label if "\n" in label else f"{label}\n "
            for label in group_labels
        ], False
    return group_labels, False


def load_fail_rate(summary_path: Path) -> tuple[int, float]:
    summary = json.loads(summary_path.read_text())
    tasks = summary["tasks"]
    generated = int(tasks["generated"])
    completed = int(tasks["completed"])
    pending = int(tasks["pending"])
    failed = int(tasks["failed"])
    if completed + pending + failed != generated:
        raise ValueError(
            f"{summary_path}: completed + pending + failed does not equal generated"
        )
    rate = 0.0 if generated == 0 else 100.0 * failed / generated
    return failed, rate


def collect_rows(
    base_dir: Path,
    *,
    groups: list[str],
    group_labels: list[str],
    methods: list[str],
) -> list[dict[str, int | float | str]]:
    return collect_rows_from_dirs(
        [base_dir / group for group in groups],
        group_keys=groups,
        group_labels=group_labels,
        methods=methods,
    )


def collect_rows_from_dirs(
    group_dirs: list[Path],
    *,
    group_keys: list[str],
    group_labels: list[str],
    methods: list[str],
) -> list[dict[str, int | float | str]]:
    validate_groups(group_keys)
    if len(group_dirs) != len(group_keys):
        raise ValueError("group directory count must match group key count")
    if len({str(path) for path in group_dirs}) != len(group_dirs):
        raise ValueError("group directories must be unique")
    if len(group_labels) != len(group_keys):
        raise ValueError("--group-labels count must match run group count")
    validate_methods(methods)

    rows: list[dict[str, int | float | str]] = []
    for group_dir, group, group_label in zip(
        group_dirs,
        group_keys,
        group_labels,
    ):
        for method in methods:
            run_dir = find_method_run_dir(group_dir, method)
            if run_dir is None:
                raise FileNotFoundError(
                    f"missing output directory for {method} under {group_dir}"
                )
            summary_path = run_dir / "summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"missing summary file: {summary_path}")
            failed, fail_rate = load_fail_rate(summary_path)
            rows.append(
                {
                    "group": group,
                    "group_label": group_label,
                    "method": method,
                    "failed": failed,
                    "fail_rate": fail_rate,
                }
            )
    return rows


def write_figure(
    path: Path,
    rows: list[dict[str, int | float | str]],
    *,
    groups: list[str],
    group_labels: list[str],
    methods: list[str],
    xlabel: str,
    group_label_font_size: float = DEFAULT_GROUP_LABEL_FONT_SIZE,
) -> tuple[Path, Path]:
    plt = _pyplot()
    from matplotlib.ticker import MultipleLocator

    figure_width = max(8.8, 1.25 * len(groups) + 2.0)
    fig, ax = plt.subplots(figsize=(17.0, 5.1))

    rows_by_key = {
        (str(row["group"]), str(row["method"])): row
        for row in rows
    }
    group_spacing = 1.5
    x_positions = [index * group_spacing for index in range(len(groups))]
    group_width = 1.25
    bar_width = group_width / len(methods)
    tick_labels, _ = format_group_labels(group_labels)

    for method_index, method in enumerate(methods):
        style = method_style(method)
        offset = -group_width / 2.0 + bar_width * (method_index + 0.5)
        bar_positions = [x + offset for x in x_positions]
        rates = [
            float(rows_by_key[(group, method)]["fail_rate"])
            for group in groups
        ]
        ax.bar(
            bar_positions,
            rates,
            bar_width,
            label=style.label,
            facecolor="none",
            edgecolor=style.color,
            hatch=style.hatch,
            linewidth=1.2,
        )

    separator_offset = 0.03
    for left, right in zip(x_positions, x_positions[1:]):
        ax.axvline(
            (left + right) / 2.0 - separator_offset,
            color="#bfbfbf",
            linewidth=0.8,
            alpha=0.8,
            zorder=0,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=group_label_font_size)
    edge_padding = group_width * 0.35
    ax.set_xlim(
        x_positions[0] - group_width / 2.0 - edge_padding,
        x_positions[-1] + group_width / 2.0 + edge_padding,
    )
    ax.set_xlabel(xlabel, fontsize=X_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(
        "%",
        fontsize=Y_AXIS_LABEL_FONT_SIZE,
        rotation=0,
        va="center",
        labelpad=16,
    )
    max_rate = max(float(row["fail_rate"]) for row in rows)
    y_max = max(80.0, min(100.0, math.ceil(max_rate / 10.0) * 10.0))
    ax.set_ylim(0.0, y_max)
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(
        axis="y",
        which="major",
        left=True,
        length=7,
        width=1.0,
        labelsize=Y_AXIS_TICK_FONT_SIZE,
    )
    ax.tick_params(axis="y", which="minor", left=True, length=4, width=1.0)
    ax.grid(True, axis="y", alpha=0.7)
    ax.legend(
        ncol=len(methods),
        loc="upper center",
        bbox_to_anchor=(0.51, 1.02),
        bbox_transform=fig.transFigure,
        borderpad=0.15,
        # handlelength=0.7,
        columnspacing=0.7,
        # handletextpad=1,
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
    )
    fig.subplots_adjust(top=0.88)

    written = save_png_pdf(fig, path)
    plt.close(fig)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare task failure rates for five methods across four to eighteen "
            "run groups."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Directory containing the names passed to --groups.",
    )
    group_source = parser.add_mutually_exclusive_group(required=True)
    group_source.add_argument(
        "--groups",
        nargs="+",
        help="Four to eighteen run-group directory names, in plot order.",
    )
    group_source.add_argument(
        "--group-dirs",
        nargs="+",
        type=Path,
        help="Four to eighteen run-group paths, which may have different parents.",
    )
    parser.add_argument(
        "--group-labels",
        nargs="+",
        help="Optional x-axis labels. Defaults to the group directory names.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Method directory names, in bar and legend order.",
    )
    parser.add_argument(
        "--xlabel",
        default="Run group",
        help="X-axis label.",
    )
    parser.add_argument(
        "--group-label-font-size",
        type=float,
        default=DEFAULT_GROUP_LABEL_FONT_SIZE,
        help="Font size for the run-group names along the x-axis.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output figure path or prefix. Writes .png and .pdf.",
    )
    args = parser.parse_args()
    if args.group_label_font_size <= 0:
        parser.error("--group-label-font-size must be positive")

    if args.groups is not None:
        if args.base_dir is None:
            parser.error("--base-dir is required with --groups")
        group_keys = args.groups
        group_dirs = [args.base_dir / group for group in args.groups]
    else:
        if args.base_dir is not None:
            parser.error("--base-dir cannot be used with --group-dirs")
        group_dirs = args.group_dirs
        group_keys = [str(index) for index in range(len(group_dirs))]

    group_labels = args.group_labels or [
        path.name for path in group_dirs
    ]
    rows = collect_rows_from_dirs(
        group_dirs,
        group_keys=group_keys,
        group_labels=group_labels,
        methods=args.methods,
    )
    written = write_figure(
        args.out,
        rows,
        groups=group_keys,
        group_labels=group_labels,
        methods=args.methods,
        xlabel=args.xlabel,
        group_label_font_size=args.group_label_font_size,
    )
    print(f"Wrote {format_written(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
