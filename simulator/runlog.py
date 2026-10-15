from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from .models import SatelliteState, SnapshotContext


SCHEMA_VERSION = 1
JsonObject = dict[str, object]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_json_line(
    stream: TextIO,
    value: object,
    *,
    flush: bool = False,
) -> None:
    stream.write(json.dumps(value, separators=(",", ":")) + "\n")
    if flush:
        stream.flush()


def _validate_record(value: object, *, source: str) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    version = value.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{source} has unsupported schema_version {version!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return value


def load_run(output_dir: Path) -> JsonObject:
    path = output_dir / "run.json"
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    return _validate_record(value, source=str(path))


def _iter_jsonl(path: Path) -> Iterator[JsonObject]:
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            source = f"{path}:{line_number}"
            if not line.strip():
                raise ValueError(f"{source} must contain a JSON object")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {source}: {exc.msg}") from exc
            yield _validate_record(value, source=source)


def iter_state_steps(output_dir: Path) -> Iterator[JsonObject]:
    return _iter_jsonl(output_dir / "states.jsonl")


def iter_task_events(output_dir: Path) -> Iterator[JsonObject]:
    return _iter_jsonl(output_dir / "tasks.jsonl")


def satellite_catalog(states: list[SatelliteState]) -> list[dict[str, object]]:
    return [
        {
            "id": state.sat_id,
            "name": state.name,
            "plane": state.plane,
            "slot": state.slot,
        }
        for state in states
    ]


def eclipse_energy_summary(states: list[SatelliteState]) -> dict[str, float]:
    idle_j = 0.0
    task_j = 0.0
    for state in states:
        if state.sunlit:
            continue
        idle_j += state.consumed_j
        task_j += state.task_energy_j
    return {
        "idle_j": idle_j,
        "task_j": task_j,
        "total_j": idle_j + task_j,
    }


def eclipse_unsafe_ratio(states: list[SatelliteState]) -> float:
    eclipse_states = [state for state in states if not state.sunlit]
    if not eclipse_states:
        return 0.0
    unsafe = sum(1 for state in eclipse_states if not state.safe_battery)
    return unsafe / len(eclipse_states)


def unsafe_ratio(states: list[SatelliteState]) -> float:
    if not states:
        return 0.0
    unsafe = sum(1 for state in states if not state.safe_battery)
    return unsafe / len(states)


def state_record(
    start: dt.datetime,
    states: list[SatelliteState],
    context: SnapshotContext | None = None,
    new_breach_ids: set[int] | None = None,
    new_eclipse_breach_ids: set[int] | None = None,
) -> dict[str, object]:
    time_s = states[0].time_s
    new_breach_ids = set() if new_breach_ids is None else new_breach_ids
    new_eclipse_breach_ids = (
        set() if new_eclipse_breach_ids is None else new_eclipse_breach_ids
    )
    below_min_safe = [state for state in states if not state.safe_battery]
    eclipse_states = [state for state in states if not state.sunlit]
    eclipse_below_min_safe = [
        state for state in eclipse_states if not state.safe_battery
    ]
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "time_s": time_s,
        "time_iso": (start + dt.timedelta(seconds=time_s)).isoformat(),
        "energy_summary": {
            "eclipse": eclipse_energy_summary(states),
        },
        "battery_violation_summary": {
            "below_min_safe": len(below_min_safe),
            "below_min_safe_ratio": len(below_min_safe) / len(states),
            "eclipse_below_min_safe": len(eclipse_below_min_safe),
            "eclipse_below_min_safe_ratio": len(eclipse_below_min_safe)
            / len(states),
            "eclipse_below_min_safe_among_eclipse_ratio": (
                0.0
                if not eclipse_states
                else len(eclipse_below_min_safe) / len(eclipse_states)
            ),
            "unsafe_eclipse_ratio": eclipse_unsafe_ratio(states),
            "new_breaches": len(new_breach_ids),
            "new_eclipse_breaches": len(new_eclipse_breach_ids),
        },
        "satellites": [
            {
                "id": state.sat_id,
                "position_km": [state.x_km, state.y_km, state.z_km],
                "velocity_km_s": [state.vx_km_s, state.vy_km_s, state.vz_km_s],
                "geodetic": (
                    None
                    if state.lat_deg is None
                    else {
                        "lat_deg": state.lat_deg,
                        "lon_deg": state.lon_deg,
                        "elevation_km": state.elevation_km,
                    }
                ),
                "sunlit": state.sunlit,
                "battery_j": state.battery_j,
                "energy_delta_j": {
                    "harvested": state.harvested_j,
                    "consumed": state.consumed_j,
                    "tasks": state.task_energy_j,
                },
                "task_load": {
                    "compute_time_s": state.task_compute_time_s,
                    "compute_energy_j": state.task_compute_energy_j,
                    "transmission_energy_j": state.task_transmission_energy_j,
                },
                "local_activity": {
                    "completed": state.local_completed_tasks,
                    "failed": state.local_failed_tasks,
                    "queue_remaining": state.queue_remaining_tasks,
                },
                "battery_status": {
                    "below_min_safe": not state.safe_battery,
                    "first_breach": state.sat_id in new_breach_ids,
                    "first_eclipse_breach": state.sat_id
                    in new_eclipse_breach_ids,
                },
                "task_counts": {
                    "generated": state.generated_tasks,
                    "completed": state.completed_tasks,
                    "failed": state.failed_tasks,
                    "deferred": state.deferred_tasks,
                },
            }
            for state in states
        ],
    }
    if context is not None:
        record["snapshot_context"] = {
            "projection_label": context.projection_label,
            "sun_eci_unit": context.sun_eci_unit,
            "sun_xy_unit": context.sun_xy_unit,
        }
    return record


class RunLog:
    """完整狀態每步寫入磁碟；不記錄完整狀態時仍可產生摘要"""

    def __init__(self, output_dir: Path, start: dt.datetime, config: dict[str, object]):
        self.output_dir = output_dir
        self.start = start
        self._started_monotonic = time.monotonic()
        self.run_path = output_dir / "run.json"
        self.summary_path = output_dir / "summary.json"
        self._task_event_mode = task_event_mode(config)
        self._state_step_mode = state_step_mode(config)
        self._summary_window = summary_window(config)
        self._states = (
            (output_dir / "states.jsonl").open("w")
            if self._state_step_mode == "full"
            else None
        )
        self._tasks = (output_dir / "tasks.jsonl").open("w")
        self._cohort_first_task_id: int | None = None
        self._cohort_end_task_id: int | None = None
        self._generated = 0
        self._terminal = 0
        self._completed = 0
        self._failed = 0
        self._deferred = 0
        self._steps = 0
        self._eclipse_idle_j = 0.0
        self._eclipse_task_j = 0.0
        self._unsafe_ratio_sum = 0.0
        self._eclipse_unsafe_ratio_sum = 0.0
        self._final_states: list[SatelliteState] | None = None
        self._breached_sat_ids: set[int] = set()
        self._eclipse_breached_sat_ids: set[int] = set()
        self._event_breached_sat_ids: set[int] = set()
        self._event_eclipse_breached_sat_ids: set[int] = set()
        self._first_breach_time_s: int | None = None
        self._last_breach_time_s: int | None = None
        self._first_eclipse_breach_time_s: int | None = None
        self._last_eclipse_breach_time_s: int | None = None
        self._manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at": utc_now_iso(),
            "config": config,
        }
        write_json(self.run_path, self._manifest)

    def _elapsed_wall_s(self) -> float:
        return max(0.0, time.monotonic() - self._started_monotonic)

    def write_step(
        self,
        states: list[SatelliteState],
        context: SnapshotContext | None = None,
    ) -> None:
        if "satellites" not in self._manifest:
            self._manifest["satellites"] = satellite_catalog(states)
            write_json(self.run_path, self._manifest)
        below_min_safe = {
            state.sat_id for state in states if not state.safe_battery
        }
        eclipse_below_min_safe = {
            state.sat_id
            for state in states
            if not state.sunlit and not state.safe_battery
        }
        new_event_breach_ids = below_min_safe - self._event_breached_sat_ids
        new_event_eclipse_breach_ids = (
            eclipse_below_min_safe - self._event_eclipse_breached_sat_ids
        )

        self._event_breached_sat_ids.update(new_event_breach_ids)
        self._event_eclipse_breached_sat_ids.update(
            new_event_eclipse_breach_ids
        )

        if self._states is not None:
            append_json_line(
                self._states,
                state_record(
                    self.start,
                    states,
                    context,
                    new_event_breach_ids,
                    new_event_eclipse_breach_ids,
                ),
                flush=True,
            )
        self._write_battery_breach_events(
            states,
            new_event_breach_ids,
            eclipse=False,
        )
        self._write_battery_breach_events(
            states,
            new_event_eclipse_breach_ids,
            eclipse=True,
        )
        if not self._time_in_summary_window(states[0].time_s):
            return

        new_breach_ids = below_min_safe - self._breached_sat_ids
        new_eclipse_breach_ids = (
            eclipse_below_min_safe - self._eclipse_breached_sat_ids
        )
        if new_breach_ids:
            self._first_breach_time_s = (
                states[0].time_s
                if self._first_breach_time_s is None
                else self._first_breach_time_s
            )
            self._last_breach_time_s = states[0].time_s
        if new_eclipse_breach_ids:
            self._first_eclipse_breach_time_s = (
                states[0].time_s
                if self._first_eclipse_breach_time_s is None
                else self._first_eclipse_breach_time_s
            )
            self._last_eclipse_breach_time_s = states[0].time_s
        self._breached_sat_ids.update(below_min_safe)
        self._eclipse_breached_sat_ids.update(eclipse_below_min_safe)

        eclipse_energy = eclipse_energy_summary(states)
        self._eclipse_idle_j += eclipse_energy["idle_j"]
        self._eclipse_task_j += eclipse_energy["task_j"]
        self._unsafe_ratio_sum += unsafe_ratio(states)
        self._eclipse_unsafe_ratio_sum += eclipse_unsafe_ratio(states)
        self._steps += 1
        self._final_states = states

    def _time_in_summary_window(self, time_s: int) -> bool:
        if self._summary_window is None:
            return True
        start_s, end_s = self._summary_window
        return start_s <= time_s < end_s

    def _write_battery_breach_events(
        self,
        states: list[SatelliteState],
        breached_ids: set[int],
        *,
        eclipse: bool,
    ) -> None:
        if self._task_event_mode in {"summary", "off"} or not breached_ids:
            return
        by_id = {state.sat_id: state for state in states}
        event_type = "battery_eclipse_breach" if eclipse else "battery_breach"
        for sat_id in sorted(breached_ids):
            state = by_id[sat_id]
            append_json_line(
                self._tasks,
                {
                    "schema_version": SCHEMA_VERSION,
                    "type": event_type,
                    "time_s": state.time_s,
                    "time_iso": (
                        self.start + dt.timedelta(seconds=state.time_s)
                    ).isoformat(),
                    "sat_id": sat_id,
                    "battery_j": state.battery_j,
                    "battery_pct": state.battery_pct,
                    "min_safe_pct": self._min_safe_pct(),
                    "sunlit": state.sunlit,
                },
            )

    def _min_safe_pct(self) -> float | None:
        config = self._manifest.get("config")
        if not isinstance(config, dict):
            return None
        battery = config.get("battery")
        if not isinstance(battery, dict):
            return None
        value = battery.get("min_safe_pct")
        return float(value) if isinstance(value, (int, float)) else None

    def write_task_event(self, event: dict[str, object]) -> None:
        record = {"schema_version": SCHEMA_VERSION, **event}
        time_s = event.get("time_s")
        if isinstance(time_s, int):
            record["time_iso"] = (self.start + dt.timedelta(seconds=time_s)).isoformat()
        task_id = event.get("task_id")
        event_type = event.get("type")
        if isinstance(task_id, int):
            if event_type == "task_generated":
                if self._generated_task_is_in_summary(event):
                    if self._summary_window is not None:
                        self._record_cohort_task(task_id)
                    self._generated += 1
            elif (
                event_type == "task_completed"
                and self._task_is_in_cohort(task_id)
            ):
                self._terminal += 1
                self._completed += 1
            elif event_type == "task_failed" and self._task_is_in_cohort(task_id):
                self._terminal += 1
                self._failed += 1
            elif (
                event_type == "task_deferred"
                and self._task_is_in_cohort(task_id)
            ):
                self._deferred += 1
        if not self._should_write_task_event(event_type):
            return
        append_json_line(self._tasks, record)

    def _generated_task_is_in_summary(self, event: dict[str, object]) -> bool:
        if self._summary_window is None:
            return True
        time_s = event.get("time_s")
        return isinstance(time_s, int) and self._time_in_summary_window(time_s)

    def _record_cohort_task(self, task_id: int) -> None:
        if self._cohort_first_task_id is None:
            self._cohort_first_task_id = task_id
            self._cohort_end_task_id = task_id + 1
            return
        if task_id != self._cohort_end_task_id:
            raise ValueError("summary-window task ids must be contiguous")
        self._cohort_end_task_id = task_id + 1

    def _task_is_in_cohort(self, task_id: int) -> bool:
        if self._summary_window is None:
            return True
        if self._cohort_first_task_id is None or self._cohort_end_task_id is None:
            return False
        return self._cohort_first_task_id <= task_id < self._cohort_end_task_id

    def _should_write_task_event(self, event_type: object) -> bool:
        if not isinstance(event_type, str):
            return self._task_event_mode == "full"
        if not event_type.startswith("task_"):
            return self._task_event_mode != "off"
        if self._task_event_mode == "full":
            return True
        if self._task_event_mode == "lifecycle":
            return event_type in {
                "task_generated",
                "task_deferred",
                "task_completed",
                "task_failed",
            }
        return False

    def complete(
        self, all_steps: list[list[SatelliteState]] | None = None
    ) -> JsonObject:
        if all_steps is not None and self._summary_window is None:
            final_states = all_steps[-1]
            steps = len(all_steps)
            eclipse_idle_j = 0.0
            eclipse_task_j = 0.0
            unsafe_ratio_sum = 0.0
            eclipse_unsafe_ratio_sum = 0.0
            for states in all_steps:
                eclipse_energy = eclipse_energy_summary(states)
                eclipse_idle_j += eclipse_energy["idle_j"]
                eclipse_task_j += eclipse_energy["task_j"]
                unsafe_ratio_sum += unsafe_ratio(states)
                eclipse_unsafe_ratio_sum += eclipse_unsafe_ratio(states)
        elif self._final_states is not None:
            final_states = self._final_states
            steps = self._steps
            eclipse_idle_j = self._eclipse_idle_j
            eclipse_task_j = self._eclipse_task_j
            unsafe_ratio_sum = self._unsafe_ratio_sum
            eclipse_unsafe_ratio_sum = self._eclipse_unsafe_ratio_sum
        else:
            raise ValueError("cannot complete a run with no state records")
        generated_tasks = self._generated
        failed_tasks = self._failed
        task_failure_ratio = (
            0.0 if generated_tasks == 0 else failed_tasks / generated_tasks
        )
        avg_eclipse_unsafe_ratio = (
            0.0 if steps == 0 else eclipse_unsafe_ratio_sum / steps
        )
        avg_unsafe_ratio = 0.0 if steps == 0 else unsafe_ratio_sum / steps
        summary = {
            "schema_version": SCHEMA_VERSION,
            "elapsed_wall_s": self._elapsed_wall_s(),
            "steps": steps,
            "final_time_s": final_states[0].time_s,
            "satellites": len(final_states),
            "tasks": {
                "generated": generated_tasks,
                "completed": self._completed,
                "deferred": self._deferred,
                "failed": failed_tasks,
                "pending": generated_tasks - self._terminal,
                "failure_ratio": task_failure_ratio,
                "pending_policy": "count_as_success",
            },
            "final_battery_j": {
                "minimum": min(state.battery_j for state in final_states),
                "average": sum(state.battery_j for state in final_states)
                / len(final_states),
            },
            "energy": {
                "eclipse": {
                    "idle_j": eclipse_idle_j,
                    "task_j": eclipse_task_j,
                    "total_j": eclipse_idle_j + eclipse_task_j,
                },
            },
            "battery_violations": {
                "avg_unsafe_ratio": avg_unsafe_ratio,
                "avg_eclipse_unsafe_ratio": avg_eclipse_unsafe_ratio,
                "unique_breached_satellites": len(self._breached_sat_ids),
                "unique_breached_ratio": len(self._breached_sat_ids)
                / len(final_states),
                "unique_eclipse_breached_satellites": len(
                    self._eclipse_breached_sat_ids
                ),
                "unique_eclipse_breached_ratio": len(
                    self._eclipse_breached_sat_ids
                )
                / len(final_states),
                "first_breach_time_s": self._first_breach_time_s,
                "last_breach_time_s": self._last_breach_time_s,
                "first_eclipse_breach_time_s": self._first_eclipse_breach_time_s,
                "last_eclipse_breach_time_s": self._last_eclipse_breach_time_s,
            },
        }
        if self._summary_window is not None:
            start_s, end_s = self._summary_window
            summary["measurement_window"] = {
                "start_time_s": start_s,
                "end_time_s_exclusive": end_s,
                "start_utc": (
                    self.start + dt.timedelta(seconds=start_s)
                ).isoformat(),
                "end_utc_exclusive": (
                    self.start + dt.timedelta(seconds=end_s)
                ).isoformat(),
                "task_cohort": "generated_within_window",
            }
        write_json(self.summary_path, summary)
        self._manifest.update(
            {
                "status": "completed",
                "finished_at": utc_now_iso(),
                "elapsed_wall_s": summary["elapsed_wall_s"],
                "summary_file": self.summary_path.name,
            }
        )
        write_json(self.run_path, self._manifest)
        self.close()
        return summary

    def fail(self, exc: BaseException) -> None:
        self._manifest.update(
            {
                "status": "failed",
                "finished_at": utc_now_iso(),
                "elapsed_wall_s": self._elapsed_wall_s(),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        write_json(self.run_path, self._manifest)
        self.close()

    def close(self) -> None:
        if self._states is not None and not self._states.closed:
            self._states.flush()
            self._states.close()
        if not self._tasks.closed:
            self._tasks.flush()
            self._tasks.close()


def task_event_mode(config: dict[str, object]) -> str:
    logging = config.get("logging")
    if not isinstance(logging, dict):
        return "full"
    mode = logging.get("task_events", "full")
    if mode not in {"full", "lifecycle", "summary", "off"}:
        raise ValueError("logging.task_events must be full, lifecycle, summary, or off")
    return str(mode)


def state_step_mode(config: dict[str, object]) -> str:
    logging = config.get("logging")
    if not isinstance(logging, dict):
        return "full"
    mode = logging.get("state_steps", "full")
    if mode not in {"full", "off"}:
        raise ValueError("logging.state_steps must be full or off")
    return str(mode)


def summary_window(config: dict[str, object]) -> tuple[int, int] | None:
    logging = config.get("logging")
    if not isinstance(logging, dict):
        return None
    start_s = logging.get("summary_start_s")
    duration_s = logging.get("summary_duration_s")
    if start_s is None and duration_s is None:
        return None
    if not isinstance(start_s, int) or isinstance(start_s, bool) or start_s < 0:
        raise ValueError("logging.summary_start_s must be a non-negative integer")
    if (
        not isinstance(duration_s, int)
        or isinstance(duration_s, bool)
        or duration_s <= 0
    ):
        raise ValueError("logging.summary_duration_s must be a positive integer")
    return start_s, start_s + duration_s
