from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import compare_eclipse_dod_cdf, plot_final_loading_effects


def test_eclipse_cdf_accepts_historical_method_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "method7"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(
        json.dumps({"config": {"battery": {"capacity_j": 100.0}}})
    )
    (run_dir / "states.jsonl").write_text(
        json.dumps(
            {
                "satellites": [
                    {"sunlit": False, "battery_j": 75.0},
                ]
            }
        )
        + "\n"
    )

    captured: dict[str, object] = {}

    def capture_figure(path, series, *, title):
        captured["path"] = path
        captured["series"] = series
        captured["title"] = title
        return path.with_suffix(".png"), path.with_suffix(".pdf")

    output = tmp_path / "cdf"
    monkeypatch.setattr(compare_eclipse_dod_cdf, "write_figure", capture_figure)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_eclipse_dod_cdf.py",
            str(run_dir),
            "--out",
            str(output),
        ],
    )

    assert compare_eclipse_dod_cdf.main() == 0
    assert captured["path"] == output
    assert captured["series"] == [
        {
            "label": "Starlit",
            "values": [25.0],
            "style": compare_eclipse_dod_cdf.method_style("starlit"),
            "method": "starlit",
            "sort_key": 4,
            "input_index": 0,
        }
    ]


def test_loading_effects_plots_unique_satellite_breach_ratio(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "run": "r40",
            "method": "local-only",
            "label": "LO",
            "loading_pct": 40.0,
            "below_e_safe_ratio": 0.25,
            "unique_below_e_safe_ratio": 0.5,
            "avg_below_e_safe_ratio": 0.4,
            "task_failure_ratio": 0.1,
        }
    ]
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        plot_final_loading_effects,
        "collect_rows",
        lambda *args, **kwargs: rows,
    )
    monkeypatch.setattr(
        plot_final_loading_effects,
        "write_csv",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        plot_final_loading_effects,
        "plot_metric",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_final_loading_effects.py",
            "--base-dir",
            str(tmp_path),
            "--out-dir",
            str(tmp_path),
        ],
    )

    plot_final_loading_effects.main()

    assert calls[0]["metric"] == "avg_below_e_safe_ratio"
    assert calls[0]["y_max"] == 100
    assert calls[0]["ylabel_rotation"] == 90
    assert calls[1]["metric"] == "task_failure_ratio"
