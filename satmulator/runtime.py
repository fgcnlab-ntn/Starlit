from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Callable

from .battery import battery_is_safe
from .models import (
    BatteryConfig,
    QueuedTaskView,
    Route,
    SatelliteState,
    SatelliteView,
    Task,
)


Vector3 = tuple[float, float, float]
TaskEventSink = Callable[[dict[str, object]], None]


@dataclass
class SatelliteRuntime:
    """Mutable per-satellite simulation state.

    This is the owner of state that changes during the simulation.  Snapshot
    dataclasses are for output only; policies and energy accounting should not
    pass around parallel arrays for battery, load, and task ownership.
    """

    sat_id: int
    name: str
    plane: int
    slot: int
    battery_j: float
    task_queue: list[RunningTask] = field(default_factory=list)
    pos_km: Vector3 = (0.0, 0.0, 0.0)
    vel_km_s: Vector3 = (0.0, 0.0, 0.0)
    lat_deg: float | None = None
    lon_deg: float | None = None
    elevation_km: float | None = None
    sunlit: bool = False
    next_sunlit_time_s: float | None = None
    next_eclipse_time_s: float | None = None
    illumination_horizon_time_s: float | None = None

    def update_orbit(
        self,
        *,
        pos_km: Vector3,
        vel_km_s: Vector3,
        sunlit: bool,
        lat_deg: float | None = None,
        lon_deg: float | None = None,
        elevation_km: float | None = None,
    ) -> None:
        self.pos_km = pos_km
        self.vel_km_s = vel_km_s
        self.sunlit = sunlit
        self.lat_deg = lat_deg
        self.lon_deg = lon_deg
        self.elevation_km = elevation_km

    def view(
        self,
        *,
        pending_task_energy_j: float = 0.0,
        include_queue_tasks: bool = False,
    ) -> SatelliteView:
        return SatelliteView(
            sat_id=self.sat_id,
            x_km=self.pos_km[0],
            y_km=self.pos_km[1],
            z_km=self.pos_km[2],
            sunlit=self.sunlit,
            battery_j=self.battery_j,
            queue_backlog_s=sum(
                task.remaining_compute_time_s for task in self.task_queue
            ),
            plane=self.plane,
            slot=self.slot,
            next_sunlit_time_s=self.next_sunlit_time_s,
            next_eclipse_time_s=self.next_eclipse_time_s,
            illumination_horizon_time_s=self.illumination_horizon_time_s,
            pending_task_energy_j=pending_task_energy_j,
            queued_tasks=(
                tuple(
                    QueuedTaskView(
                        task_id=running.task.task_id,
                        absolute_deadline_s=(
                            running.task.created_time_s + running.task.deadline_s
                        ),
                        remaining_compute_time_s=running.remaining_compute_time_s,
                        started=running.executed_compute_time_s > 1.0e-12,
                        transmission_time_s=running.transmission_time_s,
                    )
                    for running in self.task_queue
                )
                if include_queue_tasks
                else ()
            ),
        )

    def snapshot(
        self,
        *,
        time_s: int,
        harvested_j: float,
        consumed_j: float,
        battery: BatteryConfig,
        generated_tasks: int,
        completed_tasks: int,
        failed_tasks: int,
        deferred_tasks: int,
        task_energy_j: float,
        task_compute_time_s: float = 0.0,
        task_compute_energy_j: float = 0.0,
        task_transmission_energy_j: float = 0.0,
        local_completed_tasks: int = 0,
        local_failed_tasks: int = 0,
        queue_remaining_tasks: int = 0,
    ) -> SatelliteState:
        return SatelliteState(
            time_s=time_s,
            sat_id=self.sat_id,
            name=self.name,
            plane=self.plane,
            slot=self.slot,
            x_km=self.pos_km[0],
            y_km=self.pos_km[1],
            z_km=self.pos_km[2],
            vx_km_s=self.vel_km_s[0],
            vy_km_s=self.vel_km_s[1],
            vz_km_s=self.vel_km_s[2],
            lat_deg=self.lat_deg,
            lon_deg=self.lon_deg,
            elevation_km=self.elevation_km,
            sunlit=self.sunlit,
            battery_j=self.battery_j,
            battery_pct=100.0 * self.battery_j / battery.capacity_j,
            harvested_j=harvested_j,
            consumed_j=consumed_j,
            safe_battery=battery_is_safe(self.battery_j, battery.min_safe_j),
            generated_tasks=generated_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            deferred_tasks=deferred_tasks,
            task_energy_j=task_energy_j,
            task_compute_time_s=task_compute_time_s,
            task_compute_energy_j=task_compute_energy_j,
            task_transmission_energy_j=task_transmission_energy_j,
            local_completed_tasks=local_completed_tasks,
            local_failed_tasks=local_failed_tasks,
            queue_remaining_tasks=queue_remaining_tasks,
        )


@dataclass
class RunningTask:
    task: Task
    route: Route
    mode: str
    total_compute_time_s: float
    remaining_compute_time_s: float
    executed_compute_time_s: float
    transmission_time_s: float
    transmission_energy_by_sat: dict[int, float]
    energy_by_sat: dict[int, float]
    score: float = 0.0

    @property
    def source_sat(self) -> int:
        return self.route.source_sat

    @property
    def target_sat(self) -> int:
        return self.route.target_sat


@dataclass
class EnvironmentRuntime:
    """Mutable simulation state shared by orbit, scheduler, and accounting."""

    satellites: list[SatelliteRuntime]
    rng: random.Random = field(default_factory=random.Random)
    time_s: int = 0
    time_utc: dt.datetime | None = None
    next_task_id: int = 0
    pending_tasks: list[Task] = field(default_factory=list)
    completed_tasks: list[int] = field(default_factory=list)
    deferred_tasks: list[Task] = field(default_factory=list)
    failed_tasks: list[int] = field(default_factory=list)
    task_event_sink: TaskEventSink | None = None
    deadline_rng: random.Random = field(default_factory=random.Random)

    @property
    def running_tasks(self) -> list[RunningTask]:
        """Flattened view of per-satellite execution queues."""

        return [task for sat in self.satellites for task in sat.task_queue]

    def views(self, *, include_queue_tasks: bool = False) -> list[SatelliteView]:
        pending_energy_by_sat: dict[int, float] = {}
        for running in self.running_tasks:
            for sat_id, energy_j in running.transmission_energy_by_sat.items():
                pending_energy_by_sat[sat_id] = (
                    pending_energy_by_sat.get(sat_id, 0.0) + energy_j
                )
        return [
            sat.view(
                pending_task_energy_j=pending_energy_by_sat.get(sat.sat_id, 0.0),
                include_queue_tasks=include_queue_tasks,
            )
            for sat in self.satellites
        ]

    def emit_task_event(self, event_type: str, task_id: int, **details: object) -> None:
        if self.task_event_sink is None:
            return
        self.task_event_sink(
            {
                "type": event_type,
                "time_s": self.time_s,
                "task_id": task_id,
                **details,
            }
        )
