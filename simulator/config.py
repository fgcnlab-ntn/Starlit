from __future__ import annotations

import copy
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from .models import (
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    TaskConfig,
)
from .workload import demand_points_provenance, load_demand_points


JsonObject = dict[str, object]

REQUIRED_CONFIG_KEYS = {
    "run": {"name", "description"},
    "orbit": {
        "sun_position_file",
        "satellites",
        "planes",
        "altitude_km",
        "inclination_deg",
        "walker_phase",
    },
    "time": {"start_utc", "duration_s", "step_s"},
    "battery": {
        "capacity_j",
        "initial_pct",
        "min_safe_pct",
        "harvest_w",
        "idle_w",
    },
    "task": {
        "interval_s",
        "random_seed",
        "tasks_per_step",
        "input_bits",
        "output_bits",
        "demand_points_file",
        "min_elevation_deg",
        "deadline_s",
        "deadline_min_s",
    },
    "compute": {"cycles_per_input_bit", "cpu_frequency_hz", "cpu_power_w"},
    "isl": {"rate_bps", "tx_power_w", "max_range_km"},
    "scheduler": {"name"},
    "output": {"path"},
    "logging": {"task_events"},
}

OPTIONAL_CONFIG_DEFAULTS = {
    "logging": {
        "state_steps": "full",
        "summary_start_s": None,
        "summary_duration_s": None,
    }
}


@dataclass(frozen=True)
class SimulationConfig:
    start: dt.datetime
    duration_s: int
    step_s: int
    output_path: Path
    satellites: int
    planes: int
    altitude_km: float
    inclination_deg: float
    sun_position_file: str
    walker_phase: int
    battery: BatteryConfig
    compute: ComputeConfig
    task: TaskConfig
    isl: ISLConfig
    scheduler_name: str
    effective: JsonObject


def _load_sections(path: Path) -> dict[str, JsonObject]:
    with path.open() as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON config must be an object")

    expected_sections = set(REQUIRED_CONFIG_KEYS)
    actual_sections = set(data)
    if actual_sections != expected_sections:
        missing = sorted(expected_sections - actual_sections)
        extra = sorted(actual_sections - expected_sections)
        details = []
        if missing:
            details.append(f"missing sections: {', '.join(missing)}")
        if extra:
            details.append(f"unknown sections: {', '.join(extra)}")
        raise ValueError(
            "standalone config must define every section ("
            + "; ".join(details)
            + ")"
        )

    sections: dict[str, JsonObject] = {}
    for section, required_keys in REQUIRED_CONFIG_KEYS.items():
        value = data[section]
        if not isinstance(value, dict):
            raise ValueError(f"config section {section!r} must be an object")
        optional_keys = set(OPTIONAL_CONFIG_DEFAULTS.get(section, {}))
        expected_keys = required_keys | optional_keys
        actual_keys = set(value)
        if not required_keys <= actual_keys <= expected_keys:
            missing = sorted(required_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            details = []
            if missing:
                details.append(f"missing keys: {', '.join(missing)}")
            if extra:
                details.append(f"unknown keys: {', '.join(extra)}")
            raise ValueError(
                f"standalone config section {section!r} must define every key ("
                + "; ".join(details)
                + ")"
            )
        sections[section] = value
    return sections


def _parse_utc_datetime(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _validate_values(
    sections: dict[str, JsonObject], logging: JsonObject
) -> None:
    time_config = sections["time"]
    battery = sections["battery"]
    task = sections["task"]
    compute = sections["compute"]
    isl = sections["isl"]

    duration_s = time_config["duration_s"]
    step_s = time_config["step_s"]
    if duration_s < 0:
        raise ValueError("time.duration_s must be non-negative")
    if step_s <= 0:
        raise ValueError("time.step_s must be positive")
    if not 0 <= battery["initial_pct"] <= 100:
        raise ValueError("battery.initial_pct must be within [0, 100]")
    if not 0 <= battery["min_safe_pct"] <= 100:
        raise ValueError("battery.min_safe_pct must be within [0, 100]")
    if task["interval_s"] <= 0:
        raise ValueError("task.interval_s must be positive")
    if task["tasks_per_step"] < 0:
        raise ValueError("task.tasks_per_step must be non-negative")
    if task["deadline_s"] <= 0:
        raise ValueError("task.deadline_s must be positive")
    if task["deadline_min_s"] <= 0:
        raise ValueError("task.deadline_min_s must be positive")
    if task["deadline_min_s"] >= task["deadline_s"]:
        raise ValueError("task.deadline_min_s must be less than task.deadline_s")
    if not 0.0 <= task["min_elevation_deg"] <= 90.0:
        raise ValueError("task.min_elevation_deg must be within [0, 90]")
    if compute["cycles_per_input_bit"] <= 0:
        raise ValueError("compute.cycles_per_input_bit must be positive")
    if compute["cpu_frequency_hz"] <= 0:
        raise ValueError("compute.cpu_frequency_hz must be positive")
    if compute["cpu_power_w"] < 0:
        raise ValueError("compute.cpu_power_w must be non-negative")
    if isl["rate_bps"] <= 0:
        raise ValueError("isl.rate_bps must be positive")
    if isl["tx_power_w"] < 0:
        raise ValueError("isl.tx_power_w must be non-negative")
    if isl["max_range_km"] is None or isl["max_range_km"] <= 0.0:
        raise ValueError("isl.max_range_km must be positive")

    task_events = logging["task_events"]
    if task_events not in {"full", "lifecycle", "summary", "off"}:
        raise ValueError(
            "logging.task_events must be full, lifecycle, summary, or off"
        )
    if logging["state_steps"] not in {"full", "off"}:
        raise ValueError("logging.state_steps must be full or off")

    summary_start_s = logging["summary_start_s"]
    summary_duration_s = logging["summary_duration_s"]
    if (summary_start_s is None) != (summary_duration_s is None):
        raise ValueError(
            "logging.summary_start_s and logging.summary_duration_s "
            "must be specified together"
        )
    if summary_start_s is None:
        return
    if not isinstance(summary_start_s, int) or isinstance(summary_start_s, bool):
        raise ValueError("logging.summary_start_s must be an integer")
    if not isinstance(summary_duration_s, int) or isinstance(
        summary_duration_s, bool
    ):
        raise ValueError("logging.summary_duration_s must be an integer")
    if summary_start_s < 0:
        raise ValueError("logging.summary_start_s must be non-negative")
    if summary_duration_s <= 0:
        raise ValueError("logging.summary_duration_s must be positive")
    if summary_start_s % step_s or summary_duration_s % step_s:
        raise ValueError("logging summary window must align with time.step_s")
    if summary_start_s + summary_duration_s > duration_s:
        raise ValueError("logging summary window exceeds time.duration_s")


def _effective_config(
    sections: dict[str, JsonObject], demand_points_file: Path, logging: JsonObject
) -> JsonObject:
    effective = copy.deepcopy(sections)
    effective["task"]["demand_points_provenance"] = demand_points_provenance(
        demand_points_file
    )

    effective_logging = effective["logging"]
    if logging["state_steps"] == "full":
        effective_logging.pop("state_steps", None)
    else:
        effective_logging["state_steps"] = logging["state_steps"]
    if logging["summary_start_s"] is None:
        effective_logging.pop("summary_start_s", None)
        effective_logging.pop("summary_duration_s", None)
    else:
        effective_logging["summary_start_s"] = logging["summary_start_s"]
        effective_logging["summary_duration_s"] = logging["summary_duration_s"]
    return effective


def load_config(path: Path) -> SimulationConfig:
    sections = _load_sections(path)
    logging = dict(OPTIONAL_CONFIG_DEFAULTS["logging"])
    logging.update(sections["logging"])
    _validate_values(sections, logging)

    orbit = sections["orbit"]
    time_config = sections["time"]
    battery_values = sections["battery"]
    task_values = sections["task"]
    compute_values = sections["compute"]
    isl_values = sections["isl"]
    demand_points_file = Path(task_values["demand_points_file"])
    demand_distribution = load_demand_points(demand_points_file)

    battery = BatteryConfig(
        capacity_j=battery_values["capacity_j"],
        initial_j=(
            battery_values["capacity_j"] * battery_values["initial_pct"] / 100.0
        ),
        min_safe_j=(
            battery_values["capacity_j"] * battery_values["min_safe_pct"] / 100.0
        ),
        harvest_w=battery_values["harvest_w"],
        idle_w=battery_values["idle_w"],
    )
    task = TaskConfig(
        interval_s=task_values["interval_s"],
        random_seed=task_values["random_seed"],
        tasks_per_step=task_values["tasks_per_step"],
        input_bits=task_values["input_bits"],
        output_bits=task_values["output_bits"],
        deadline_s=task_values["deadline_s"],
        deadline_min_s=task_values["deadline_min_s"],
        demand_distribution=demand_distribution,
        min_elevation_deg=task_values["min_elevation_deg"],
    )
    compute = ComputeConfig(
        cycles_per_input_bit=compute_values["cycles_per_input_bit"],
        cpu_frequency_hz=compute_values["cpu_frequency_hz"],
        cpu_power_w=compute_values["cpu_power_w"],
    )
    isl = ISLConfig(
        rate_bps=isl_values["rate_bps"],
        tx_power_w=isl_values["tx_power_w"],
        max_range_km=isl_values["max_range_km"],
    )
    return SimulationConfig(
        start=_parse_utc_datetime(time_config["start_utc"]),
        duration_s=time_config["duration_s"],
        step_s=time_config["step_s"],
        output_path=Path(sections["output"]["path"]),
        satellites=orbit["satellites"],
        planes=orbit["planes"],
        altitude_km=orbit["altitude_km"],
        inclination_deg=orbit["inclination_deg"],
        sun_position_file=orbit["sun_position_file"],
        walker_phase=orbit["walker_phase"],
        battery=battery,
        compute=compute,
        task=task,
        isl=isl,
        scheduler_name=sections["scheduler"]["name"],
        effective=_effective_config(sections, demand_points_file, logging),
    )
