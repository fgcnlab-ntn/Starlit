from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .runlog import JsonObject


TERMINAL_EVENTS = {"task_completed", "task_failed"}
STATUS_BY_EVENT = {
    "task_generated": "pending",
    "task_waiting_for_coverage": "pending",
    "task_coverage_acquired": "pending",
    "task_assigned": "assigned",
    "task_deferred": "deferred",
    "task_completed": "completed",
    "task_failed": "failed",
}


@dataclass(frozen=True)
class TaskLifecycle:
    task_id: int
    created_time_s: int
    status: str
    source_sat: int | None
    target_sat: int | None
    mode: str | None
    events: tuple[JsonObject, ...]

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def terminal_event(self) -> JsonObject | None:
        if self.status not in {"completed", "failed"}:
            return None
        return self.events[-1]


def project_task_lifecycles(events: Iterable[JsonObject]) -> list[TaskLifecycle]:
    events_by_task: dict[int, list[JsonObject]] = {}

    for event in events:
        task_id = event.get("task_id")
        event_type = event.get("type")
        time_s = event.get("time_s")
        if not isinstance(event_type, str):
            raise ValueError("event requires a string type")
        if not event_type.startswith("task_"):
            continue
        if not isinstance(task_id, int):
            raise ValueError("task event requires an integer task_id")
        if not isinstance(time_s, int):
            raise ValueError(f"task {task_id} event requires an integer time_s")

        task_events = events_by_task.setdefault(task_id, [])
        if not task_events and event_type != "task_generated":
            raise ValueError(f"task {task_id} lifecycle must start with task_generated")
        if task_events:
            previous = task_events[-1]
            if previous["type"] in TERMINAL_EVENTS:
                raise ValueError(f"task {task_id} has an event after its terminal event")
            if time_s < previous["time_s"]:
                raise ValueError(f"task {task_id} events are not ordered by time_s")
        if event_type == "task_generated" and task_events:
            raise ValueError(f"task {task_id} has multiple task_generated events")
        task_events.append(event)

    return [_project_task(task_id, events) for task_id, events in events_by_task.items()]


def _project_task(task_id: int, events: list[JsonObject]) -> TaskLifecycle:
    generated = events[0]
    status = "pending"
    source_sat = _optional_int(generated.get("source_sat"), task_id, "source_sat")
    target_sat = None
    mode = None

    for event in events:
        event_type = event["type"]
        status = STATUS_BY_EVENT.get(event_type, status)
        if "source_sat" in event:
            source_sat = _optional_int(event["source_sat"], task_id, "source_sat")
        if "target_sat" in event:
            target_sat = _optional_int(event["target_sat"], task_id, "target_sat")
        if "mode" in event:
            event_mode = event["mode"]
            if not isinstance(event_mode, str):
                raise ValueError(f"task {task_id} mode must be a string")
            mode = event_mode

    return TaskLifecycle(
        task_id=task_id,
        created_time_s=generated["time_s"],
        status=status,
        source_sat=source_sat,
        target_sat=target_sat,
        mode=mode,
        events=tuple(events),
    )


def _optional_int(value: object, task_id: int, field: str) -> int | None:
    if value is None or isinstance(value, int):
        return value
    raise ValueError(f"task {task_id} {field} must be an integer or null")
