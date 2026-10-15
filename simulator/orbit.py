from __future__ import annotations

import datetime as dt
import math
import random
from array import array
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

from .battery import (
    battery_step,
    validate_battery_config,
)
from .constants import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from .geometry import (
    circular_plane,
    circular_state,
    is_sunlit_cylindrical_shadow,
    vector_unit,
    xy_unit,
)
from .isl import build_isl_graph, grid_constellation_layout
from .models import (
    Assignment,
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    SatelliteState,
    SatelliteView,
    SnapshotContext,
    Task,
    TaskConfig,
    TaskIntensity,
)
from .route_cost import estimate_route_cost
from .runtime import EnvironmentRuntime, RunningTask, SatelliteRuntime, TaskEventSink
from .scheduler import Scheduler, enforce_edf_queue_feasibility
from .workload import deadline_random_seed, generate_step_tasks, validate_task_config


StepSink = Callable[[list[SatelliteState], SnapshotContext], None]


def validate_compute_config(compute: ComputeConfig) -> None:
    if compute.cycles_per_input_bit <= 0:
        raise ValueError("compute cycles per input bit must be positive")
    if compute.cpu_frequency_hz <= 0:
        raise ValueError("CPU frequency must be positive")
    if compute.cpu_power_w < 0:
        raise ValueError("CPU power must be non-negative")


@dataclass(frozen=True)
class CircularIlluminationTimeline:
    step_s: int
    duration_s: int
    sunlit_by_step: tuple[bytes, ...]
    next_eclipse_step: tuple[array, ...]
    next_sunlit_step: tuple[array, ...]

    @property
    def horizon_time_s(self) -> float:
        # 最後一次取樣的日照狀態仍套用於完整時間步
        return float(self.duration_s + self.step_s)

    def apply(self, *, env: EnvironmentRuntime, step_index: int) -> None:
        sunlit = self.sunlit_by_step[step_index]
        next_eclipse = self.next_eclipse_step[step_index]
        next_sunlit = self.next_sunlit_step[step_index]
        for sat in env.satellites:
            sat_id = sat.sat_id
            sat.sunlit = bool(sunlit[sat_id])
            eclipse_step = next_eclipse[sat_id]
            sunlit_step = next_sunlit[sat_id]
            sat.next_eclipse_time_s = (
                None if eclipse_step < 0 else float(eclipse_step * self.step_s)
            )
            sat.next_sunlit_time_s = (
                None if sunlit_step < 0 else float(sunlit_step * self.step_s)
            )
            sat.illumination_horizon_time_s = self.horizon_time_s


def build_circular_illumination_timeline(
    *,
    start: dt.datetime | None,
    sun_ephemeris,
    satellites: int,
    planes: int,
    sats_per_plane: int,
    duration_s: int,
    step_s: int,
    radius_km: float,
    inclination_rad: float,
    raan_spread_rad: float,
    walker_phase: int,
    mean_motion: float,
) -> CircularIlluminationTimeline:
    times = tuple(range(0, duration_s + 1, step_s))
    orbital_speed_km_s = radius_km * mean_motion
    plane_geometry = tuple(
        circular_plane(
            inclination_rad,
            raan_spread_rad * plane / planes,
        )
        for plane in range(planes)
    )
    sunlit_rows: list[bytes] = []
    for time_s in times:
        time_utc = None if start is None else start + dt.timedelta(seconds=time_s)
        sun_vector = (
            None
            if time_utc is None or sun_ephemeris is None
            else sun_vector_from_ephemeris(sun_ephemeris, time_utc)
        )
        sun_unit = (1.0, 0.0, 0.0) if sun_vector is None else vector_unit(sun_vector)
        if sun_unit is None:
            raise ValueError("Sun vector norm must be positive")

        row = bytearray(satellites)
        for plane, geometry in enumerate(plane_geometry):
            plane_phase = 2.0 * math.pi * walker_phase * plane / satellites
            for slot in range(sats_per_plane):
                sat_id = plane * sats_per_plane + slot
                arg = (
                    2.0 * math.pi * slot / sats_per_plane
                    + plane_phase
                    + mean_motion * time_s
                )
                pos, _vel = circular_state(
                    geometry,
                    radius_km,
                    orbital_speed_km_s,
                    arg,
                )
                row[sat_id] = is_sunlit_cylindrical_shadow(pos, sun_unit)
        sunlit_rows.append(bytes(row))

    step_count = len(sunlit_rows)
    next_eclipse = [array("i", [-1]) * satellites for _ in range(step_count)]
    next_sunlit = [array("i", [-1]) * satellites for _ in range(step_count)]

    for sat_id in range(satellites):
        run_start = 0
        while run_start < step_count:
            state = bool(sunlit_rows[run_start][sat_id])
            run_end = run_start + 1
            while (
                run_end < step_count
                and bool(sunlit_rows[run_end][sat_id]) == state
            ):
                run_end += 1

            if state:
                eclipse_step = run_end if run_end < step_count else -1
                sunlit_step = -1
                if eclipse_step >= 0:
                    sunlit_step = eclipse_step
                    while (
                        sunlit_step < step_count
                        and not bool(sunlit_rows[sunlit_step][sat_id])
                    ):
                        sunlit_step += 1
                    if sunlit_step >= step_count:
                        sunlit_step = -1
                for step_index in range(run_start, run_end):
                    next_eclipse[step_index][sat_id] = eclipse_step
                    next_sunlit[step_index][sat_id] = sunlit_step
            else:
                sunlit_step = run_end if run_end < step_count else -1
                for step_index in range(run_start, run_end):
                    next_eclipse[step_index][sat_id] = step_index
                    next_sunlit[step_index][sat_id] = sunlit_step
            run_start = run_end

    return CircularIlluminationTimeline(
        step_s=step_s,
        duration_s=duration_s,
        sunlit_by_step=tuple(sunlit_rows),
        next_eclipse_step=tuple(next_eclipse),
        next_sunlit_step=tuple(next_sunlit),
    )


@dataclass
class SatelliteStepStats:
    generated_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    deferred_tasks: int = 0
    local_completed_tasks: int = 0
    local_failed_tasks: int = 0
    task_energy_j: float = 0.0
    task_compute_time_s: float = 0.0
    task_compute_energy_j: float = 0.0
    task_transmission_energy_j: float = 0.0
    harvested_j: float = 0.0
    consumed_j: float = 0.0


def assign_step_tasks(
    *,
    scheduler: Scheduler,
    tasks: list[Task],
    satellite_views: list[SatelliteView],
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
    isl_config: ISLConfig,
    isl_graph,
) -> list[Assignment]:
    if any(task.source_sat is None for task in tasks):
        raise ValueError("cannot assign tasks without a visible source satellite")

    assignments = scheduler.assign_tasks(
        tasks=tasks,
        satellite_views=satellite_views,
        time_s=time_s,
        step_s=step_s,
        battery=battery,
        compute_config=compute_config,
        isl_config=isl_config,
        isl_graph=isl_graph,
    )
    if scheduler.queue_discipline == "edf":
        return enforce_edf_queue_feasibility(
            tasks=tasks,
            assignments=assignments,
            satellite_views=satellite_views,
            time_s=time_s,
            compute_config=compute_config,
            isl_config=isl_config,
        )
    return assignments


def pop_deferred_tasks(env: EnvironmentRuntime) -> tuple[list[Task], list[Task]]:
    ready: list[Task] = []
    expired: list[Task] = []

    for task in env.deferred_tasks:
        if env.time_s - task.created_time_s >= task.deadline_s:
            expired.append(task)
        else:
            ready.append(task)

    env.deferred_tasks.clear()
    return ready, expired


def order_task_queue(sat: SatelliteRuntime, queue_discipline: str) -> None:
    if queue_discipline == "fifo":
        return
    if queue_discipline != "edf":
        raise ValueError(f"unknown queue discipline: {queue_discipline}")

    active = [
        running
        for running in sat.task_queue
        if running.executed_compute_time_s > 1.0e-12
    ]
    waiting = sorted(
        (
            running
            for running in sat.task_queue
            if running.executed_compute_time_s <= 1.0e-12
        ),
        key=lambda running: (
            running.task.created_time_s + running.task.deadline_s,
            running.task.task_id,
        ),
    )
    sat.task_queue[:] = active + waiting


def apply_step(
    *,
    env: EnvironmentRuntime,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
    isl_config: ISLConfig,
    tasks: list[Task],
    assignments: list[Assignment],
    expired_tasks: list[Task] | None = None,
    queue_discipline: str = "fifo",
) -> list[SatelliteState]:
    """任務可延後、立即失敗或入列"""

    stats_by_sat = {sat.sat_id: SatelliteStepStats() for sat in env.satellites}
    task_by_id = {task.task_id: task for task in tasks}
    cpu_time_left_s = {sat.sat_id: step_s for sat in env.satellites}
    satellite_by_id = {sat.sat_id: sat for sat in env.satellites}

    def remaining_deadline_s(task: Task) -> float:
        return task.created_time_s + task.deadline_s - env.time_s

    expired_tasks = [] if expired_tasks is None else expired_tasks

    # 延後任務重新進入此 time step 時，不算新產生的任務
    for task in tasks:
        if task.source_sat is not None and task.created_time_s == env.time_s:
            stats_by_sat[task.source_sat].generated_tasks += 1

    # 到期任務包含未獲覆蓋的需求及延後的任務
    for task in expired_tasks:
        waiting_time_s = env.time_s - task.created_time_s
        env.failed_tasks.append(task.task_id)
        source_sat = task.source_sat if task.source_sat is not None else -1

        if source_sat in stats_by_sat:
            stats_by_sat[source_sat].failed_tasks += 1
            failed_reason = "deadline"
        else:
            failed_reason = "no_coverage"

        env.emit_task_event(
            "task_failed",
            task.task_id,
            source_sat=source_sat,
            reason=failed_reason,
            waiting_time_s=waiting_time_s,
        )

    for assignment in assignments:
        task = task_by_id[assignment.task_id]
        waiting_time_s = env.time_s - task.created_time_s
        env.emit_task_event(
            "task_assigned",
            task.task_id,
            source_sat=assignment.source_sat,
            target_sat=assignment.target_sat,
            route=list(assignment.route.nodes),
            mode=assignment.mode,
            score=assignment.score,
            waiting_time_s=waiting_time_s,
        )
        source_stats = stats_by_sat[assignment.source_sat]

        # 延後不消耗任務能量，也不計為失敗
        if assignment.mode == "defer":
            env.deferred_tasks.append(task)
            source_stats.deferred_tasks += 1
            env.emit_task_event(
                "task_deferred",
                task.task_id,
                source_sat=assignment.source_sat,
                route=list(assignment.route.nodes),
                score=assignment.score,
                waiting_time_s=waiting_time_s,
            )

            continue

        if assignment.mode == "fail":
            source_stats.failed_tasks += 1
            env.failed_tasks.append(task.task_id)
            env.emit_task_event(
                "task_failed",
                task.task_id,
                source_sat=assignment.source_sat,
                target_sat=assignment.target_sat,
                route=list(assignment.route.nodes),
                mode="fail",
                reason=assignment.failed_reason or "scheduler_fail",
                score=assignment.score,
                waiting_time_s=waiting_time_s,
            )

            continue

        cost = estimate_route_cost(
            task=task,
            route=assignment.route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        transmission_energy_by_sat = {
            sat_id: energy_j
            for sat_id, energy_j in cost.energy_by_sat.items()
            if sat_id != assignment.target_sat
        }
        satellite_by_id[assignment.target_sat].task_queue.append(
            RunningTask(
                task=task,
                route=assignment.route,
                mode=assignment.mode,
                total_compute_time_s=cost.compute_time_s,
                remaining_compute_time_s=cost.compute_time_s,
                executed_compute_time_s=0.0,
                transmission_time_s=cost.transmission_time_s,
                transmission_energy_by_sat=transmission_energy_by_sat,
                energy_by_sat={},
                score=assignment.score,
            )
        )

    def accumulated_energy(running: RunningTask) -> tuple[float, float, float]:
        source_energy_j = running.energy_by_sat.get(running.source_sat, 0.0)
        target_energy_j = (
            0.0
            if running.target_sat == running.source_sat
            else running.energy_by_sat.get(running.target_sat, 0.0)
        )
        return source_energy_j, target_energy_j, sum(running.energy_by_sat.values())

    def fail_running_task(
        running: RunningTask,
        *,
        waiting_time_s: float,
        transmission_time_s: float,
        total_time_s: float,
    ) -> None:
        stats_by_sat[running.source_sat].failed_tasks += 1
        stats_by_sat[running.target_sat].local_failed_tasks += 1
        env.failed_tasks.append(running.task.task_id)
        env.emit_task_event(
            "task_failed",
            running.task.task_id,
            source_sat=running.source_sat,
            target_sat=running.target_sat,
            route=list(running.route.nodes),
            mode=running.mode,
            reason="deadline",
            waiting_time_s=waiting_time_s,
            executed_compute_time_s=running.executed_compute_time_s,
            remaining_compute_time_s=running.remaining_compute_time_s,
            transmission_time_s=transmission_time_s,
            total_time_s=total_time_s,
            score=running.score,
        )

    if queue_discipline not in {"fifo", "edf"}:
        raise ValueError(f"unknown queue discipline: {queue_discipline}")
    if queue_discipline == "edf":
        for sat in env.satellites:
            order_task_queue(sat, queue_discipline)

    for sat in env.satellites:
        cpu_time_capacity = max(0.0, cpu_time_left_s[sat.sat_id])
        cpu_time_left = cpu_time_capacity
        while sat.task_queue:
            running = sat.task_queue[0]
            task = running.task
            waiting_time_s = env.time_s - task.created_time_s

            if remaining_deadline_s(task) <= 0.0 and running.remaining_compute_time_s > 0.0:
                fail_running_task(
                    running,
                    waiting_time_s=waiting_time_s,
                    transmission_time_s=0.0,
                    total_time_s=waiting_time_s,
                )
                sat.task_queue.pop(0)
                continue

            if cpu_time_left <= 1.0e-12:
                break

            elapsed_cpu_time_s = cpu_time_capacity - cpu_time_left
            target_cpu_time_s = min(running.remaining_compute_time_s, cpu_time_left)
            cpu_time_left -= target_cpu_time_s
            running.remaining_compute_time_s -= target_cpu_time_s

            transmission_energy_by_sat = dict(running.transmission_energy_by_sat)
            running.transmission_energy_by_sat.clear()
            compute_energy_j = target_cpu_time_s * compute_config.cpu_power_w

            for sat_id, energy_j in transmission_energy_by_sat.items():
                stats_by_sat[sat_id].task_transmission_energy_j += energy_j

            if compute_energy_j:
                stats_by_sat[running.target_sat].task_compute_time_s += (
                    target_cpu_time_s
                )
                stats_by_sat[running.target_sat].task_compute_energy_j += (
                    compute_energy_j
                )

            energy_by_sat = transmission_energy_by_sat
            if compute_energy_j:
                energy_by_sat[running.target_sat] = (
                    energy_by_sat.get(running.target_sat, 0.0) + compute_energy_j
                )
            for sat_id, energy_j in energy_by_sat.items():
                stats_by_sat[sat_id].task_energy_j += energy_j
                running.energy_by_sat[sat_id] = (
                    running.energy_by_sat.get(sat_id, 0.0) + energy_j
                )

            running.executed_compute_time_s += target_cpu_time_s

            if running.remaining_compute_time_s > 1.0e-12:
                break

            finish_time_s = (
                env.time_s
                + elapsed_cpu_time_s
                + target_cpu_time_s
                + running.transmission_time_s
            )
            total_time_s = finish_time_s - task.created_time_s
            if total_time_s > task.deadline_s:
                fail_running_task(
                    running,
                    waiting_time_s=waiting_time_s,
                    transmission_time_s=running.transmission_time_s,
                    total_time_s=total_time_s,
                )
                sat.task_queue.pop(0)
                continue

            stats_by_sat[running.source_sat].completed_tasks += 1
            stats_by_sat[running.target_sat].local_completed_tasks += 1
            env.completed_tasks.append(task.task_id)
            source_energy_j, target_energy_j, total_energy_j = accumulated_energy(running)
            env.emit_task_event(
                "task_completed",
                task.task_id,
                source_sat=running.source_sat,
                target_sat=running.target_sat,
                route=list(running.route.nodes),
                mode=running.mode,
                waiting_time_s=waiting_time_s,
                executed_compute_time_s=running.executed_compute_time_s,
                remaining_compute_time_s=0.0,
                transmission_time_s=running.transmission_time_s,
                total_time_s=total_time_s,
                energy_j={
                    "source": source_energy_j,
                    "target": target_energy_j,
                    "total": total_energy_j,
                },
                energy_by_sat={
                    str(sat_id): energy_j
                    for sat_id, energy_j in running.energy_by_sat.items()
                },
                score=running.score,
            )
            sat.task_queue.pop(0)
        cpu_time_left_s[sat.sat_id] = cpu_time_left

    states: list[SatelliteState] = []

    for sat in env.satellites:
        stats = stats_by_sat[sat.sat_id]

        # t=0 不結算電量，保留初始值
        battery_now, harvested_j, consumed_j = battery_step(
            battery_now=sat.battery_j,
            sunlit=sat.sunlit,
            step_s=step_s,
            battery=battery,
            task_energy_j=stats.task_energy_j,
            update=env.time_s > 0,
        )

        sat.battery_j = battery_now
        stats.harvested_j = harvested_j
        stats.consumed_j = consumed_j

        states.append(
            sat.snapshot(
                time_s=env.time_s,
                harvested_j=stats.harvested_j,
                consumed_j=stats.consumed_j,
                battery=battery,
                generated_tasks=stats.generated_tasks,
                completed_tasks=stats.completed_tasks,
                failed_tasks=stats.failed_tasks,
                deferred_tasks=stats.deferred_tasks,
                task_energy_j=stats.task_energy_j,
                task_compute_time_s=stats.task_compute_time_s,
                task_compute_energy_j=stats.task_compute_energy_j,
                task_transmission_energy_j=stats.task_transmission_energy_j,
                local_completed_tasks=stats.local_completed_tasks,
                local_failed_tasks=stats.local_failed_tasks,
                queue_remaining_tasks=len(sat.task_queue),
            )
        )

    return states


def iter_circular_states(
    *,
    start: dt.datetime | None = None,
    sun_position_file: str | None = None,
    satellites: int,
    planes: int,
    altitude_km: float,
    inclination_deg: float,
    duration_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
    task_config: TaskConfig,
    isl_config: ISLConfig,
    scheduler: Scheduler,
    walker_phase: int = 0,
    task_event_sink: TaskEventSink | None = None,
    step_sink: StepSink | None = None,
    task_intensity: TaskIntensity | None = None,
) -> Iterable[list[SatelliteState]]:
    """每步更新軌道與日照，產生並排程任務，再執行與更新電量"""

    if satellites <= 0:
        raise ValueError("satellites must be positive")
    if planes <= 0 or satellites % planes != 0:
        raise ValueError("planes must be positive and divide satellites")
    if step_s <= 0:
        raise ValueError("step must be positive")
    validate_battery_config(battery)
    validate_compute_config(compute_config)
    validate_task_config(task_config)

    sats_per_plane = satellites // planes
    radius_km = EARTH_RADIUS_KM + altitude_km
    inclination_rad = math.radians(inclination_deg)
    raan_spread_rad = math.tau
    mean_motion = math.sqrt(EARTH_MU_KM3_S2 / (radius_km**3))
    orbital_speed_km_s = radius_km * mean_motion
    plane_geometry = tuple(
        circular_plane(
            inclination_rad,
            raan_spread_rad * plane / planes,
        )
        for plane in range(planes)
    )
    sun_ephemeris = (
        None
        if start is None or sun_position_file is None
        else load_sun_ephemeris(sun_position_file)
    )
    illumination_timeline = build_circular_illumination_timeline(
        start=start,
        sun_ephemeris=sun_ephemeris,
        satellites=satellites,
        planes=planes,
        sats_per_plane=sats_per_plane,
        duration_s=duration_s,
        step_s=step_s,
        radius_km=radius_km,
        inclination_rad=inclination_rad,
        raan_spread_rad=raan_spread_rad,
        walker_phase=walker_phase,
        mean_motion=mean_motion,
    )

    env = EnvironmentRuntime(
        rng=random.Random(task_config.random_seed),
        deadline_rng=random.Random(deadline_random_seed(task_config.random_seed)),
        task_event_sink=task_event_sink,
        satellites=[
            SatelliteRuntime(
                sat_id=sat_id,
                name=f"sat_{sat_id}",
                plane=sat_id // sats_per_plane,
                slot=sat_id % sats_per_plane,
                battery_j=battery.initial_j,
            )
            for sat_id in range(satellites)
        ],
    )
    isl_candidate_graph = grid_constellation_layout(
        env.views(), walker_phase=walker_phase
    )

    for time_s in range(0, duration_s + 1, step_s):
        step_index = time_s // step_s
        env.time_s = time_s
        env.time_utc = None if start is None else start + dt.timedelta(seconds=time_s)
        sun_vector = (
            None
            if env.time_utc is None or sun_ephemeris is None
            else sun_vector_from_ephemeris(sun_ephemeris, env.time_utc)
        )
        sun_unit = (1.0, 0.0, 0.0) if sun_vector is None else vector_unit(sun_vector)
        if sun_unit is None:
            raise ValueError("Sun vector norm must be positive")

        for plane, geometry in enumerate(plane_geometry):
            plane_phase = 2.0 * math.pi * walker_phase * plane / satellites

            for slot in range(sats_per_plane):
                sat_id = plane * sats_per_plane + slot
                arg = (
                    2.0 * math.pi * slot / sats_per_plane
                    + plane_phase
                    + mean_motion * time_s
                )
                pos, vel = circular_state(
                    geometry,
                    radius_km,
                    orbital_speed_km_s,
                    arg,
                )
                sunlit = bool(
                    illumination_timeline.sunlit_by_step[step_index][sat_id]
                )

                env.satellites[sat_id].update_orbit(
                    pos_km=pos,
                    vel_km_s=vel,
                    sunlit=sunlit,
                )

        illumination_timeline.apply(
            env=env,
            step_index=step_index,
        )

        new_tasks, expired_tasks = generate_step_tasks(
            env,
            task_config,
            compute_config,
            tasks_per_step=(
                None
                if task_intensity is None
                else task_intensity.tasks_for_time(time_s)
            ),
        )
        deferred_tasks, expired_deferred_tasks = pop_deferred_tasks(env)

        tasks = deferred_tasks + new_tasks
        expired_tasks = expired_tasks + expired_deferred_tasks

        satellite_views = env.views(
            include_queue_tasks=scheduler.queue_discipline == "edf"
        )
        assignments = assign_step_tasks(
            scheduler=scheduler,
            tasks=tasks,
            satellite_views=satellite_views,
            time_s=env.time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
            isl_config=isl_config,
            isl_graph=build_isl_graph(
                satellite_views,
                isl_config,
                candidate_graph=isl_candidate_graph,
            ),
        )

        states = apply_step(
            env=env,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
            isl_config=isl_config,
            tasks=tasks,
            assignments=assignments,
            expired_tasks=expired_tasks,
            queue_discipline=scheduler.queue_discipline,
        )

        if step_sink is not None:
            step_sink(states, circular_snapshot_context(sun_vector))
        yield states



def circular_snapshot_context(
    sun_vector: tuple[float, float, float] | None = None,
) -> SnapshotContext:
    if sun_vector is None:
        return SnapshotContext(
            projection_label="ECI x-y projection; circular orbit model uses fixed +x sun direction",
            sun_xy_unit=(1.0, 0.0),
            sun_eci_unit=(1.0, 0.0, 0.0),
        )
    return SnapshotContext(
        projection_label="ECI x-y projection; circular orbit model uses ephemeris Sun vector",
        sun_xy_unit=xy_unit(sun_vector),
        sun_eci_unit=vector_unit(sun_vector),
    )


@lru_cache(maxsize=4)
def load_sun_ephemeris(sun_position_file: str):
    try:
        from skyfield.api import load
    except ImportError as exc:
        raise SystemExit(
            "Skyfield is required for ephemeris Sun positions. Install it with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc
    return load.timescale(), load(sun_position_file)


def sun_vector_from_ephemeris(
    sun_ephemeris,
    time_utc: dt.datetime,
) -> tuple[float, float, float]:
    ts, eph = sun_ephemeris
    t = ts.from_datetime(time_utc)
    earth = eph["earth"].at(t)
    sun = eph["sun"].at(t)
    return tuple(float(x) for x in (sun.position.km - earth.position.km))
