from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


METHOD_ALPHA = 0.65
EDGE_COLOR = "#222222"


@dataclass(frozen=True)
class MethodPlotStyle:
    method: str
    label: str
    color: str
    alpha: float
    hatch: str
    marker: str


METHOD_ORDER: tuple[str, ...] = (
    "local-only",
    "nearest-sunlit",
    "greedy-energy",
    "phoenix",
    "starlit",
)

_METHOD_STYLES: dict[str, MethodPlotStyle] = {
    "local-only": MethodPlotStyle(
        method="local-only",
        label="LO",
        color="#7F7F7F",
        alpha=METHOD_ALPHA,
        hatch="\\",
        marker="o",
    ),
    "nearest-sunlit": MethodPlotStyle(
        method="nearest-sunlit",
        label="NS",
        color="#1F77B4",
        alpha=METHOD_ALPHA,
        hatch=".",
        marker="s",
    ),
    "greedy-energy": MethodPlotStyle(
        method="greedy-energy",
        label="LEB",
        color="#FF7F0E",
        alpha=METHOD_ALPHA,
        hatch="xx",
        marker="^",
    ),
    "phoenix": MethodPlotStyle(
        method="phoenix",
        label="PHX",
        color="#2CA02C",
        alpha=METHOD_ALPHA,
        hatch="//",
        marker="D",
    ),
    "starlit": MethodPlotStyle(
        method="starlit",
        label="Starlit",
        color="#D62728",
        alpha=METHOD_ALPHA,
        hatch="**",
        marker="*",
    ),
}

_METHOD_ALIASES = {
    "local-only": "local-only",
    "nearest-sunlit": "nearest-sunlit",
    "greedy-energy": "greedy-energy",
    "phoenix2": "phoenix",
    "phoenix": "phoenix",
    "starlit": "starlit",
    "method7": "starlit",
}

_RUN_DIRECTORY_NAMES = {
    "local-only": ("local-only",),
    "nearest-sunlit": ("nearest-sunlit",),
    "greedy-energy": ("greedy-energy",),
    "phoenix": ("phoenix", "phoenix2"),
    "starlit": ("starlit", "method7"),
}


def canonical_method(method: str) -> str:
    try:
        return _METHOD_ALIASES[method]
    except KeyError as exc:
        raise ValueError(f"unknown method: {method}") from exc


def run_display_label(run_dir: str | Path) -> str:
    name = Path(run_dir).name.replace("_", "-")
    if name == "phoenix2":
        return "phoenix"
    return name


def find_method_run_dir(parent: Path, method: str) -> Path | None:
    """Find a current or historical output directory for a method."""

    canonical = canonical_method(method)
    for directory_name in _RUN_DIRECTORY_NAMES[canonical]:
        run_dir = parent / directory_name
        if run_dir.is_dir():
            return run_dir
    return None


def method_style(method: str) -> MethodPlotStyle:
    return _METHOD_STYLES[canonical_method(method)]


def ordered_methods(methods: Iterable[str] | None = None) -> list[str]:
    if methods is None:
        return list(METHOD_ORDER)

    present = {canonical_method(method) for method in methods}
    return [method for method in METHOD_ORDER if method in present]


def method_labels(methods: Iterable[str] | None = None) -> list[str]:
    return [method_style(method).label for method in ordered_methods(methods)]


def method_colors(methods: Iterable[str] | None = None) -> list[str]:
    return [method_style(method).color for method in ordered_methods(methods)]


def method_hatches(methods: Iterable[str] | None = None) -> list[str]:
    return [method_style(method).hatch for method in ordered_methods(methods)]


def method_markers(methods: Iterable[str] | None = None) -> list[str]:
    return [method_style(method).marker for method in ordered_methods(methods)]


def bar_kwargs(method: str) -> dict[str, object]:
    style = method_style(method)
    return {
        "facecolor": style.color,
        "edgecolor": EDGE_COLOR,
        "alpha": style.alpha,
        "hatch": style.hatch,
    }


def violin_body_kwargs(method: str) -> dict[str, object]:
    style = method_style(method)
    return {
        "facecolor": style.color,
        "edgecolor": EDGE_COLOR,
        "alpha": style.alpha,
    }


def line_kwargs(method: str) -> dict[str, object]:
    style = method_style(method)
    return {
        "color": style.color,
        "alpha": style.alpha,
        "marker": style.marker,
    }


def css_rgba(method: str) -> str:
    style = method_style(method)
    red = int(style.color[1:3], 16)
    green = int(style.color[3:5], 16)
    blue = int(style.color[5:7], 16)
    return f"rgba({red}, {green}, {blue}, {style.alpha:.2f})"
