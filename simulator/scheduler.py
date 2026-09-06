from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .battery import battery_is_safe, battery_step
from .isl import (
    ISLGraph,
    build_route_tree,
    route_from_parents,
    route_nodes_reversed,
    routes_from_source,
    routes_to_targets,
    shortest_route,
)
from .models import (
    Assignment,
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    Route,
    SatelliteView,
    Task,
)
from .route_cost import (
    RouteCost,
    RouteTiming,
    compute_cycles,
    estimate_route_cost,
    estimate_route_timing,
    task_compute_time_s,
    transmission_energy_j,
)


def route_or_raise(graph: ISLGraph, source_sat: int, target_sat: int) -> Route:
    route = shortest_route(graph, source_sat, target_sat)
    if route is None:
        raise ValueError(f"no ISL route from {source_sat} to {target_sat}")
    return route


@dataclass(frozen=True)
class PhoenixCandidateCache:
    sunlit_by_plane: dict[int, tuple[SatelliteView, ...]]
    sunlit_global: tuple[SatelliteView, ...]
    sunlit_counts_by_plane: dict[int, int]


@dataclass(frozen=True)
class GreedyEnergyCandidate:
    assignment: Assignment
    finish_time_s: float
    energy_j: float
    battery_cost_j: float


@dataclass(frozen=True)
class ProjectedQueueTask:
    task_id: int
    absolute_deadline_s: float
    remaining_compute_time_s: float
    transmission_time_s: float
    started: bool = False


@dataclass(frozen=True)
class EDFQueueProjection:
    tasks: tuple[ProjectedQueueTask, ...]
    finish_time_by_task: dict[int, float]


def project_nonpreemptive_edf_queue(
    tasks: list[ProjectedQueueTask],
    *,
    time_s: float,
) -> EDFQueueProjection | None:
    """Order waiting work by deadline and reject an infeasible projection."""

    active = [task for task in tasks if task.started]
    waiting = sorted(
        (task for task in tasks if not task.started),
        key=lambda task: (task.absolute_deadline_s, task.task_id),
    )
    ordered = active + waiting
    finish_time_by_task: dict[int, float] = {}
    cpu_available_time_s = time_s
    for task in ordered:
        cpu_available_time_s += task.remaining_compute_time_s
        finish_time_s = cpu_available_time_s + task.transmission_time_s
        finish_time_by_task[task.task_id] = finish_time_s
        if finish_time_s > task.absolute_deadline_s:
            return None
    return EDFQueueProjection(tuple(ordered), finish_time_by_task)


def enforce_edf_queue_feasibility(
    *,
    tasks: list[Task],
    assignments: list[Assignment],
    satellite_views: list[SatelliteView],
    time_s: int,
    compute_config: ComputeConfig,
    isl_config: ISLConfig,
) -> list[Assignment]:
    """Admit assignments only when the target's full EDF queue stays feasible."""

    task_by_id = {task.task_id: task for task in tasks}
    projected_queues = {
        sat.sat_id: [
            ProjectedQueueTask(
                task_id=queued.task_id,
                absolute_deadline_s=queued.absolute_deadline_s,
                remaining_compute_time_s=queued.remaining_compute_time_s,
                transmission_time_s=queued.transmission_time_s,
                started=queued.started,
            )
            for queued in sat.queued_tasks
        ]
        for sat in satellite_views
    }
    checked: list[Assignment] = []
    for assignment in assignments:
        if assignment.mode in {"defer", "fail"}:
            checked.append(assignment)
            continue

        task = task_by_id[assignment.task_id]
        timing = estimate_route_timing(
            task=task,
            route=assignment.route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        candidate = ProjectedQueueTask(
            task_id=task.task_id,
            absolute_deadline_s=task.created_time_s + task.deadline_s,
            remaining_compute_time_s=timing.compute_time_s,
            transmission_time_s=timing.transmission_time_s,
        )
        target_queue = projected_queues[assignment.target_sat]
        projection = project_nonpreemptive_edf_queue(
            target_queue + [candidate],
            time_s=float(time_s),
        )
        if projection is None:
            checked.append(
                Assignment(
                    task_id=task.task_id,
                    route=assignment.route,
                    mode="fail",
                    score=float("inf"),
                    failed_reason="edf_queue_infeasible",
                )
            )
            continue

        projected_queues[assignment.target_sat] = list(projection.tasks)
        checked.append(assignment)
    return checked


def reserved_energy_for_sat(reserved_energy, sat_id: int) -> float:
    if isinstance(reserved_energy, BatteryReservation):
        return reserved_energy.spent_transmission_j.get(sat_id, 0.0)
    if isinstance(reserved_energy, dict):
        return reserved_energy.get(sat_id, 0.0)
    if 0 <= sat_id < len(reserved_energy):
        return reserved_energy[sat_id]
    return 0.0


class BatteryReservation:
    """Batch-local battery headroom and transmission-energy ledger."""

    def __init__(
        self,
        *,
        remaining_j: dict[int, float],
        free_sunlit_compute_s: dict[int, float],
        compute_config: ComputeConfig,
    ) -> None:
        self.remaining_j = remaining_j
        self.free_sunlit_compute_s = free_sunlit_compute_s
        self.spent_transmission_j = {sat_id: 0.0 for sat_id in remaining_j}
        self.compute_config = compute_config

    @classmethod
    def build(
        cls,
        *,
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
    ) -> BatteryReservation:
        remaining_j = {}
        free_sunlit_compute_s = {}
        for sat in satellite_views:
            minimum_j = minimum_projected_battery_until_recharge(
                sat=sat,
                available_time_s=float(time_s) + sat.queue_backlog_s,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
                extra_energy_j=sat.pending_task_energy_j,
            )
            remaining_j[sat.sat_id] = max(0.0, minimum_j - battery.min_safe_j)
            next_eclipse = (
                sat.next_eclipse_time_s
                if sat.next_eclipse_time_s is not None
                else sat.illumination_horizon_time_s
            )
            free_sunlit_compute_s[sat.sat_id] = (
                max(
                    0.0,
                    next_eclipse
                    - float(time_s)
                    - sat.queue_backlog_s,
                )
                if (
                    sat.sunlit
                    and next_eclipse is not None
                    and next_eclipse > float(time_s)
                    and battery.harvest_w
                    >= battery.idle_w + compute_config.cpu_power_w
                )
                else 0.0
            )
        return cls(
            remaining_j=remaining_j,
            free_sunlit_compute_s=free_sunlit_compute_s,
            compute_config=compute_config,
        )

    def allows(self, *, route: Route, route_cost: RouteCost) -> bool:
        target_compute_j = (
            route_cost.compute_time_s * self.compute_config.cpu_power_w
        )
        charge_target_compute = (
            route_cost.compute_time_s
            > self.free_sunlit_compute_s.get(route.target_sat, 0.0) + 1.0e-9
        )
        for sat_id, energy_j in route_cost.energy_by_sat.items():
            if sat_id == route.target_sat:
                energy_j = max(0.0, energy_j - target_compute_j)
                if charge_target_compute:
                    energy_j += target_compute_j
            if energy_j > self.remaining_j.get(sat_id, 0.0) + 1.0e-9:
                return False
        return True

    def allows_compute(self, *, sat_id: int, compute_time_s: float) -> bool:
        free_compute_s = self.free_sunlit_compute_s.get(sat_id, 0.0)
        compute_j = (
            0.0
            if compute_time_s <= free_compute_s + 1.0e-9
            else compute_time_s * self.compute_config.cpu_power_w
        )
        return compute_j <= self.remaining_j.get(sat_id, 0.0) + 1.0e-9

    def reserve(self, *, route: Route, route_cost: RouteCost) -> None:
        transmission_by_sat = route_transmission_energy_by_sat(
            route=route,
            route_cost=route_cost,
            compute_config=self.compute_config,
        )
        for sat_id, energy_j in transmission_by_sat.items():
            self.spent_transmission_j[sat_id] = (
                self.spent_transmission_j.get(sat_id, 0.0) + energy_j
            )
        if (
            route_cost.compute_time_s
            > self.free_sunlit_compute_s.get(route.target_sat, 0.0) + 1.0e-9
        ):
            transmission_by_sat[route.target_sat] = (
                transmission_by_sat.get(route.target_sat, 0.0)
                + route_cost.compute_time_s * self.compute_config.cpu_power_w
            )
        for sat_id, energy_j in transmission_by_sat.items():
            self.remaining_j[sat_id] = self.remaining_j.get(sat_id, 0.0) - energy_j
        self.free_sunlit_compute_s[route.target_sat] = max(
            0.0,
            self.free_sunlit_compute_s.get(route.target_sat, 0.0)
            - route_cost.compute_time_s,
        )


def hard_limit_reserved_energy_by_sat(
    *,
    satellite_views: list[SatelliteView],
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
) -> BatteryReservation:
    """Build the batch-local energy ledger from the exact transition lookup."""

    return BatteryReservation.build(
        satellite_views=satellite_views,
        time_s=time_s,
        step_s=step_s,
        battery=battery,
        compute_config=compute_config,
    )


def eclipse_route_respects_hard_limit(
    *,
    route_cost,
    satellite_by_id: dict[int, SatelliteView],
    reserved_energy,
    battery: BatteryConfig,
) -> bool:
    for sat_id, energy_j in route_cost.energy_by_sat.items():
        sat = satellite_by_id[sat_id]
        if sat.sunlit:
            continue
        projected = sat.battery_j - reserved_energy_for_sat(reserved_energy, sat_id)
        projected -= energy_j
        if not battery_is_safe(projected, battery.min_safe_j):
            return False
    return True


def reserve_route_energy(
    *,
    route_cost,
    reserved_energy,
) -> None:
    for sat_id, energy_j in route_cost.energy_by_sat.items():
        if isinstance(reserved_energy, dict):
            reserved_energy[sat_id] = reserved_energy.get(sat_id, 0.0) + energy_j
        elif 0 <= sat_id < len(reserved_energy):
            reserved_energy[sat_id] += energy_j


def route_transmission_energy_by_sat(
    *,
    route: Route,
    route_cost: RouteCost,
    compute_config: ComputeConfig,
) -> dict[int, float]:
    """Return route energy excluding target CPU compute energy."""

    target_compute_j = route_cost.compute_time_s * compute_config.cpu_power_w
    transmission_by_sat = {}
    for sat_id, energy_j in route_cost.energy_by_sat.items():
        if sat_id == route.target_sat:
            energy_j = max(0.0, energy_j - target_compute_j)
        if energy_j > 0.0:
            transmission_by_sat[sat_id] = energy_j
    return transmission_by_sat


def reserve_route_transmission_energy(
    *,
    route: Route,
    route_cost: RouteCost,
    compute_config: ComputeConfig,
    reserved_energy,
) -> None:
    if isinstance(reserved_energy, BatteryReservation):
        reserved_energy.reserve(route=route, route_cost=route_cost)
        return
    for sat_id, energy_j in route_transmission_energy_by_sat(
        route=route,
        route_cost=route_cost,
        compute_config=compute_config,
    ).items():
        if isinstance(reserved_energy, dict):
            reserved_energy[sat_id] = reserved_energy.get(sat_id, 0.0) + energy_j
        elif 0 <= sat_id < len(reserved_energy):
            reserved_energy[sat_id] += energy_j


def _projection_horizon_s(
    *,
    sat: SatelliteView,
    time_s: int,
    step_s: int,
    workload_s: float,
) -> float:
    """Return the horizon through the next unavoidable discharge interval."""

    now = float(time_s)
    fallback = now + max(float(step_s), workload_s)
    horizon = sat.illumination_horizon_time_s
    if sat.sunlit:
        next_sunlit = sat.next_sunlit_time_s
        if next_sunlit is not None and next_sunlit > now:
            return next_sunlit
        return max(fallback, horizon) if horizon is not None else fallback

    next_sunlit = sat.next_sunlit_time_s
    if next_sunlit is not None and next_sunlit > now:
        return next_sunlit
    return max(fallback, horizon) if horizon is not None else fallback


def _project_interval(
    *,
    battery_j: float,
    minimum_j: float,
    duration_s: float,
    sunlit: bool,
    compute_s: float,
    battery: BatteryConfig,
    compute_power_w: float,
) -> tuple[float, float]:
    if duration_s <= 0.0:
        return battery_j, minimum_j

    harvest_w = battery.harvest_w if sunlit else 0.0
    compute_s = max(0.0, min(duration_s, compute_s))
    idle_s = duration_s - compute_s

    if compute_s > 0.0:
        net_w = harvest_w - battery.idle_w - compute_power_w
        battery_j += net_w * compute_s
        battery_j = min(battery.capacity_j, battery_j)
        minimum_j = min(minimum_j, battery_j)

    if idle_s > 0.0:
        net_w = harvest_w - battery.idle_w
        battery_j += net_w * idle_s
        battery_j = min(battery.capacity_j, battery_j)
        minimum_j = min(minimum_j, battery_j)

    return battery_j, minimum_j


def minimum_projected_battery_until_recharge(
    *,
    sat: SatelliteView,
    available_time_s: float,
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig | None = None,
    compute_power_w: float | None = None,
    extra_compute_time_s: float = 0.0,
    extra_energy_j: float = 0.0,
) -> float:
    """Project the minimum battery through the next eclipse/recharge window.

    ``available_time_s`` encodes compute already queued for the satellite.
    ``extra_compute_time_s`` is the candidate work appended to that queue.
    ``extra_energy_j`` is non-compute route energy already reserved or added by
    the candidate.  Runtime still updates the real battery step by step; this
    function only answers whether the scheduler may spend the margin.
    """

    if compute_power_w is None:
        if compute_config is None:
            raise ValueError("compute_config or compute_power_w is required")
        compute_power_w = compute_config.cpu_power_w

    now = float(time_s)
    queued_compute_s = max(0.0, available_time_s - now)
    workload_s = queued_compute_s + max(0.0, extra_compute_time_s)
    horizon_s = _projection_horizon_s(
        sat=sat,
        time_s=time_s,
        step_s=step_s,
        workload_s=workload_s,
    )

    battery_j = sat.battery_j - max(0.0, extra_energy_j)
    minimum_j = min(sat.battery_j, battery_j)
    remaining_compute_s = workload_s
    cursor_s = now

    def consume_until(end_s: float, sunlit: bool) -> None:
        nonlocal battery_j, minimum_j, remaining_compute_s, cursor_s
        end_s = min(end_s, horizon_s)
        if end_s <= cursor_s:
            return
        duration_s = end_s - cursor_s
        compute_s = min(remaining_compute_s, duration_s)
        battery_j, minimum_j = _project_interval(
            battery_j=battery_j,
            minimum_j=minimum_j,
            duration_s=duration_s,
            sunlit=sunlit,
            compute_s=compute_s,
            battery=battery,
            compute_power_w=compute_power_w,
        )
        remaining_compute_s -= compute_s
        cursor_s = end_s

    if sat.sunlit:
        next_eclipse = sat.next_eclipse_time_s
        if next_eclipse is not None and next_eclipse <= now:
            consume_until(horizon_s, False)
        elif next_eclipse is not None and next_eclipse < horizon_s:
            consume_until(next_eclipse, True)
            consume_until(horizon_s, False)
        else:
            consume_until(horizon_s, True)
    else:
        consume_until(horizon_s, False)

    return minimum_j


def route_respects_battery_projection(
    *,
    route: Route,
    route_cost: RouteCost,
    satellite_by_id: dict[int, SatelliteView],
    reserved_available_time: dict[int, float],
    reserved_energy,
    time_s: int,
    step_s: int,
    battery: BatteryConfig,
    compute_config: ComputeConfig,
) -> bool:
    if isinstance(reserved_energy, BatteryReservation):
        return reserved_energy.allows(route=route, route_cost=route_cost)

    transmission_by_sat = route_transmission_energy_by_sat(
        route=route,
        route_cost=route_cost,
        compute_config=compute_config,
    )
    touched_sat_ids = set(route_cost.energy_by_sat)
    touched_sat_ids.add(route.target_sat)
    for sat_id in touched_sat_ids:
        sat = satellite_by_id[sat_id]
        extra_compute_time_s = (
            route_cost.compute_time_s if sat_id == route.target_sat else 0.0
        )
        extra_energy_j = (
            reserved_energy_for_sat(reserved_energy, sat_id)
            + transmission_by_sat.get(sat_id, 0.0)
        )
        minimum_j = minimum_projected_battery_until_recharge(
            sat=sat,
            available_time_s=reserved_available_time.get(
                sat_id,
                float(time_s) + sat.queue_backlog_s,
            ),
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
            extra_compute_time_s=extra_compute_time_s,
            extra_energy_j=extra_energy_j,
        )
        if not battery_is_safe(minimum_j, battery.min_safe_j):
            return False
    return True


class Scheduler:
    name = "base"
    queue_discipline = "fifo"

    def assign_task(
        self,
        *,
        task: Task,
        satellite_views: list[SatelliteView],
        isl_graph: ISLGraph,
    ) -> Assignment:
        raise NotImplementedError

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
    ) -> list[Assignment]:
        return [
            self.assign_task(
                task=task,
                satellite_views=satellite_views,
                isl_graph=isl_graph,
            )
            for task in tasks
        ]


class LocalOnlyScheduler(Scheduler):
    name = "local"

    def assign_task(
        self,
        *,
        task: Task,
        satellite_views: list[SatelliteView],
        isl_graph: ISLGraph,
    ) -> Assignment:
        if task.source_sat is None:
            raise ValueError("task.source_sat is required for local scheduling")
        if task.source_sat not in isl_graph.adjacency:
            raise ValueError(
                f"source satellite {task.source_sat} is not present in the ISL graph"
            )
        return Assignment(
            task_id=task.task_id,
            route=Route((task.source_sat,)),
            mode=self.name,
        )

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
    ) -> list[Assignment]:
        if battery is None or compute_config is None or isl_config is None:
            return [
                self.assign_task(
                    task=task,
                    satellite_views=satellite_views,
                    isl_graph=isl_graph,
                )
                for task in tasks
            ]

        by_id = {sat.sat_id: sat for sat in satellite_views}
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s
            for sat in satellite_views
        }
        assignments: list[Assignment] = []

        for task in tasks:
            assert task.source_sat is not None
            route = Route((task.source_sat,))
            route_cost = estimate_route_cost(
                task=task,
                route=route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if route_respects_battery_projection(
                route=route,
                route_cost=route_cost,
                satellite_by_id=by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            ):
                assignments.append(
                    Assignment(task_id=task.task_id, route=route, mode=self.name)
                )
                reserve_route_transmission_energy(
                    route=route,
                    route_cost=route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )
                reserved_available_time[route.target_sat] += route_cost.compute_time_s
            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route,
                        mode="defer",
                        score=float("inf"),
                    )
                )

        return assignments


class NearestSunlitScheduler(Scheduler):
    name = "nearest-sunlit"

    def _assignment_for_source(
        self,
        *,
        task_id: int,
        source: SatelliteView,
        sunlit_targets: tuple[SatelliteView, ...],
        isl_graph: ISLGraph,
    ) -> Assignment:
        mode = "local"
        route = route_or_raise(isl_graph, source.sat_id, source.sat_id)
        if not source.sunlit:
            routes_by_target = routes_from_source(isl_graph, source.sat_id)
            reachable_sunlit_targets = [
                sat for sat in sunlit_targets if sat.sat_id in routes_by_target
            ]
            if reachable_sunlit_targets:
                target = min(
                    reachable_sunlit_targets,
                    key=lambda sat: routes_by_target[sat.sat_id].hop_count,
                )
                route = routes_by_target[target.sat_id]
                mode = "offload"
        return Assignment(
            task_id=task_id,
            route=route,
            mode=mode,
        )

    def assign_task(
        self,
        *,
        task: Task,
        satellite_views: list[SatelliteView],
        isl_graph: ISLGraph,
    ) -> Assignment:
        assert task.source_sat is not None
        satellite_by_id = {sat.sat_id: sat for sat in satellite_views}
        source = satellite_by_id[task.source_sat]
        return self._assignment_for_source(
            task_id=task.task_id,
            source=source,
            sunlit_targets=tuple(sat for sat in satellite_views if sat.sunlit),
            isl_graph=isl_graph,
        )

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
    ) -> list[Assignment]:
        if battery is None or compute_config is None or isl_config is None:
            return [
                self.assign_task(
                    task=task,
                    satellite_views=satellite_views,
                    isl_graph=isl_graph,
                )
                for task in tasks
            ]

        satellite_by_id = {sat.sat_id: sat for sat in satellite_views}
        sunlit_targets = tuple(sat for sat in satellite_views if sat.sunlit)
        assignment_by_source: dict[int, Assignment] = {}
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s
            for sat in satellite_views
        }
        assignments: list[Assignment] = []

        for task in tasks:
            assert task.source_sat is not None
            template = assignment_by_source.get(task.source_sat)
            if template is None:
                template = self._assignment_for_source(
                    task_id=task.task_id,
                    source=satellite_by_id[task.source_sat],
                    sunlit_targets=sunlit_targets,
                    isl_graph=isl_graph,
                )
                assignment_by_source[task.source_sat] = template
            route_cost = estimate_route_cost(
                task=task,
                route=template.route,
                compute_config=compute_config,
                isl_config=isl_config,
            )

            finish_time_s = (
                max(
                    float(time_s) + route_cost.transmission_time_s,
                    reserved_available_time[template.route.target_sat],
                )
                + route_cost.compute_time_s
            )
            if finish_time_s > task.created_time_s + task.deadline_s:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=template.route,
                        mode="fail",
                        score=float("inf"),
                        failed_reason="deadline",
                    )
                )
                continue

            if route_respects_battery_projection(
                route=template.route,
                route_cost=route_cost,
                satellite_by_id=satellite_by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=template.route,
                        mode=template.mode,
                        score=template.score,
                        failed_reason=template.failed_reason,
                    )
                )
                reserve_route_transmission_energy(
                    route=template.route,
                    route_cost=route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )
                reserved_available_time[template.route.target_sat] += (
                    route_cost.compute_time_s
                )
            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=template.route,
                        mode="fail",
                        score=float("inf"),
                        failed_reason="battery_hard_constraint",
                    )
                )

        return assignments


class GreedyEnergyScheduler(Scheduler):
    """Greedy baseline adapted from the LEO energy-allocation paper.

    Ground stations are intentionally not modeled.  The paper's DVFS compute
    model is represented by the simulator's existing compute-power model, and
    Friis link loss is represented by the existing tx-power-per-bit model.

    Feasibility is enforced first: deadline, source-local CPU quota, and the
    shadow battery guard are hard constraints.  Among feasible local and sunlit
    relay candidates, choose the lowest shadow-battery-impact option.  Energy
    spent on sunlit satellites is treated as cheaper than energy drained from
    eclipse satellites, matching the paper's battery-preservation intent.
    """

    name = "greedy-energy"
    max_remote_candidates_per_source = 64
    shadow_soft_guard_ratio = 0.65

    def _step_capacity_cycles(
        self,
        *,
        step_s: int,
        compute_config: ComputeConfig,
    ) -> float:
        return step_s * compute_config.cpu_frequency_hz

    def _local_quota_cycles(
        self,
        *,
        sat: SatelliteView,
        step_s: int,
        time_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
    ) -> float:
        cpu_quota_cycles = self._step_capacity_cycles(
            step_s=step_s,
            compute_config=compute_config,
        )
        if sat.sunlit:
            return cpu_quota_cycles

        soft_guard_j = self.shadow_soft_guard_ratio * battery.capacity_j
        idle_energy_j = battery.idle_w * step_s if time_s > 0 else 0.0
        task_energy_budget_j = sat.battery_j - soft_guard_j - idle_energy_j
        if task_energy_budget_j <= 0.0:
            return 0.0

        battery_quota_cycles = (
            task_energy_budget_j
            / compute_config.cpu_power_w
            * compute_config.cpu_frequency_hz
        )
        return min(cpu_quota_cycles, battery_quota_cycles)

    def _local_quota_by_sat(
        self,
        *,
        satellite_views: list[SatelliteView],
        step_s: int,
        time_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
    ) -> dict[int, float]:
        return {
            sat.sat_id: self._local_quota_cycles(
                sat=sat,
                step_s=step_s,
                time_s=time_s,
                battery=battery,
                compute_config=compute_config,
            )
            for sat in satellite_views
        }

    def _battery_cost_j(
        self,
        *,
        energy_by_sat: dict[int, float],
        satellite_views_by_id: dict[int, SatelliteView],
    ) -> float:
        return sum(
            energy_j
            for sat_id, energy_j in energy_by_sat.items()
            if not satellite_views_by_id[sat_id].sunlit
        )

    def _candidate_for_route(
        self,
        *,
        task: Task,
        route: Route,
        mode: str,
        time_s: int,
        reserved_available_time: dict[int, float],
        satellite_views_by_id: dict[int, SatelliteView],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> GreedyEnergyCandidate | None:
        timing = estimate_route_timing(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        finish_time_s = (
            max(
                float(time_s) + timing.transmission_time_s,
                reserved_available_time[route.target_sat],
            )
            + timing.compute_time_s
        )
        deadline_time_s = task.created_time_s + task.deadline_s
        if finish_time_s > deadline_time_s:
            return None

        cost = estimate_route_cost(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        energy_j = cost.total_energy_j
        battery_cost_j = self._battery_cost_j(
            energy_by_sat=cost.energy_by_sat,
            satellite_views_by_id=satellite_views_by_id,
        )
        return GreedyEnergyCandidate(
            assignment=Assignment(
                task_id=task.task_id,
                route=route,
                mode=mode,
                score=battery_cost_j,
            ),
            finish_time_s=finish_time_s,
            energy_j=energy_j,
            battery_cost_j=battery_cost_j,
        )

    def _local_candidate(
        self,
        *,
        task: Task,
        source: SatelliteView,
        time_s: int,
        reserved_available_time: dict[int, float],
        satellite_views_by_id: dict[int, SatelliteView],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> GreedyEnergyCandidate | None:
        return self._candidate_for_route(
            task=task,
            route=Route((source.sat_id,)),
            mode="local",
            time_s=time_s,
            reserved_available_time=reserved_available_time,
            satellite_views_by_id=satellite_views_by_id,
            compute_config=compute_config,
            isl_config=isl_config,
        )

    def _remote_compute_candidates(
        self,
        *,
        task: Task,
        remote_routes: tuple[Route, ...],
        time_s: int,
        reserved_available_time: dict[int, float],
        satellite_views_by_id: dict[int, SatelliteView],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> list[GreedyEnergyCandidate]:
        candidates: list[GreedyEnergyCandidate] = []
        for route in remote_routes:
            candidate = self._candidate_for_route(
                task=task,
                route=route,
                mode="relay",
                time_s=time_s,
                reserved_available_time=reserved_available_time,
                satellite_views_by_id=satellite_views_by_id,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if candidate is not None:
                candidates.append(candidate)
        return candidates

    def _nearest_sunlit_compute_routes(
        self,
        *,
        source: SatelliteView,
        satellite_views_by_id: dict[int, SatelliteView],
        isl_graph: ISLGraph,
    ) -> tuple[Route, ...]:
        """Return a small stable set of nearest sunlit compute routes.

        The simplified link-energy model charges per hop, not per distance.
        Scanning every sunlit satellite per task is therefore just wasted work:
        lower-hop routes dominate higher-hop routes on energy.  Keep a bounded
        set so queue/deadline tie-breaks still have alternatives without
        turning each scheduling slot into tasks x constellation_size work.
        """

        if source.sat_id not in isl_graph.adjacency:
            return ()

        parents: dict[int, int | None] = {source.sat_id: None}
        queue: deque[int] = deque([source.sat_id])
        routes: list[Route] = []

        while queue and len(routes) < self.max_remote_candidates_per_source:
            current = queue.popleft()
            for neighbor in isl_graph.neighbors(current):
                if neighbor in parents:
                    continue
                parents[neighbor] = current
                sat = satellite_views_by_id.get(neighbor)
                if sat is not None and sat.sunlit:
                    route = route_from_parents(parents, neighbor)
                    assert route is not None
                    routes.append(route)
                    if len(routes) >= self.max_remote_candidates_per_source:
                        break
                queue.append(neighbor)

        return tuple(routes)

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
    ) -> list[Assignment]:
        by_id = {sat.sat_id: sat for sat in satellite_views}
        local_quota_cycles = self._local_quota_by_sat(
            satellite_views=satellite_views,
            step_s=step_s,
            time_s=time_s,
            battery=battery,
            compute_config=compute_config,
        )
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )
        reserved_local_cycles = {
            sat.sat_id: sat.queue_backlog_s * compute_config.cpu_frequency_hz
            for sat in satellite_views
        }
        remote_routes_by_source: dict[int, tuple[Route, ...]] = {}
        assignments: list[Assignment] = []
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]
            candidates: list[GreedyEnergyCandidate] = []
            task_cycles = compute_cycles(task, compute_config)
            local_fits_quota = (
                reserved_local_cycles[source.sat_id] + task_cycles
                <= local_quota_cycles[source.sat_id]
            )

            if local_fits_quota:
                local = self._local_candidate(
                    task=task,
                    source=source,
                    time_s=time_s,
                    reserved_available_time=reserved_available_time,
                    satellite_views_by_id=by_id,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
                if local is not None:
                    candidates.append(local)

            remote_routes = remote_routes_by_source.get(source.sat_id)
            if remote_routes is None:
                remote_routes = self._nearest_sunlit_compute_routes(
                    source=source,
                    satellite_views_by_id=by_id,
                    isl_graph=isl_graph,
                )
                remote_routes_by_source[source.sat_id] = remote_routes
            candidates.extend(
                self._remote_compute_candidates(
                    task=task,
                    remote_routes=remote_routes,
                    time_s=time_s,
                    reserved_available_time=reserved_available_time,
                    satellite_views_by_id=by_id,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
            )

            if not candidates:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="defer",
                        score=float("inf"),
                    )
                )
                continue

            safe_candidates = []
            for candidate in candidates:
                candidate_cost = estimate_route_cost(
                    task=task,
                    route=candidate.assignment.route,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
                if route_respects_battery_projection(
                    route=candidate.assignment.route,
                    route_cost=candidate_cost,
                    satellite_by_id=by_id,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    time_s=time_s,
                    step_s=step_s,
                    battery=battery,
                    compute_config=compute_config,
                ):
                    safe_candidates.append((candidate, candidate_cost))

            if not safe_candidates:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=candidates[0].assignment.route,
                        mode="fail",
                        score=float("inf"),
                        failed_reason="battery_hard_constraint",
                    )
                )
                continue

            chosen, chosen_cost = min(
                safe_candidates,
                key=lambda candidate: (
                    candidate[0].battery_cost_j,
                    candidate[0].finish_time_s,
                    candidate[0].energy_j,
                    candidate[0].assignment.hop_count,
                    candidate[0].assignment.target_sat,
                ),
            )

            assignments.append(chosen.assignment)
            reserved_available_time[chosen.assignment.target_sat] = chosen.finish_time_s
            reserve_route_transmission_energy(
                route=chosen.assignment.route,
                route_cost=chosen_cost,
                compute_config=compute_config,
                reserved_energy=reserved_energy,
            )
            if chosen.assignment.mode == "local":
                reserved_local_cycles[source.sat_id] += task_cycles

        return assignments


class StarlitScheduler(Scheduler):
    """Schedule battery-safe work across local, sunlit, and eclipse peers."""

    name = "starlit"
    queue_discipline = "edf"

    @staticmethod
    def _clip_unit(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _execution_load_cost(
        self,
        *,
        available_time_s: float,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
    ) -> tuple[float, float] | None:
        """Return normalized projected slot occupancy and finish time."""
        import math

        now = float(time_s)
        finish_time = max(now, available_time_s) + compute_time_s
        if finish_time > deadline_time:
            return None

        remaining_time_s = deadline_time - now
        remaining_slots = max(1, math.floor(remaining_time_s / step_s))
        committed_workload_s = max(0.0, available_time_s - now)
        occupied_slots = math.ceil(
            (committed_workload_s + compute_time_s) / step_s
        )
        load_cost = self._clip_unit(occupied_slots / remaining_slots)
        return load_cost, finish_time

    def _normalized_local_cost(
        self,
        *,
        sat: SatelliteView,
        available_time_s: float,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_power_w: float,
        battery: BatteryConfig,
    ) -> tuple[float, float, bool] | None:
        execution = self._execution_load_cost(
            available_time_s=available_time_s,
            time_s=time_s,
            step_s=step_s,
            deadline_time=deadline_time,
            compute_time_s=compute_time_s,
        )
        if execution is None:
            return None

        load_cost, finish_time = execution
        if sat.sunlit:
            return load_cost, finish_time, True

        # Local eclipse work cannot be moved after it enters the execution
        # queue.  Project the battery through all work already committed to
        # this satellite, including reservations made earlier in this batch.
        committed_workload_s = max(0.0, available_time_s - float(time_s))
        projected_workload_s = committed_workload_s + compute_time_s
        projected_energy_j = (
            battery.idle_w + compute_power_w
        ) * projected_workload_s
        projected_battery_j = sat.battery_j - projected_energy_j
        projected_battery_safe = projected_battery_j >= battery.min_safe_j

        safe_span_j = battery.capacity_j - battery.min_safe_j
        if safe_span_j <= 0.0:
            battery_cost = (
                0.0 if projected_battery_j >= battery.capacity_j else 1.0
            )
        else:
            battery_cost = self._clip_unit(
                (battery.capacity_j - projected_battery_j) / safe_span_j
            )

        return (
            max(load_cost, battery_cost),
            finish_time,
            projected_battery_safe,
        )

    def _normalized_defer_cost(
        self,
        *,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        deferred_workload_s: float,
        has_feasible_sunlit_execution: bool,
        local_eclipse_cost: float | None,
    ) -> float:
        remaining_time_s = deadline_time - float(time_s)
        if remaining_time_s <= 0.0:
            return float("inf")

        finish_after_defer = (
            float(time_s) + step_s + deferred_workload_s + compute_time_s
        )
        if finish_after_defer > deadline_time:
            return float("inf")

        urgency_cost = self._clip_unit(step_s / remaining_time_s)
        deferred_workload_cost = self._clip_unit(
            (deferred_workload_s + compute_time_s) / step_s
        )
        if has_feasible_sunlit_execution:
            # Deferring would discard an immediately feasible, energy-safe
            # execution opportunity.  Execution wins the cost-1 tie.
            immediate_opportunity_cost = 1.0
        elif local_eclipse_cost is not None:
            # With no usable sunlit target, wait only when local eclipse risk
            # is worse than the complementary value of preserving it.
            immediate_opportunity_cost = 1.0 - self._clip_unit(
                local_eclipse_cost
            )
        else:
            immediate_opportunity_cost = 0.0

        return max(
            urgency_cost,
            deferred_workload_cost,
            immediate_opportunity_cost,
        )

    def _peek_least_loaded_sunlit_mod(
        self,
        *,
        sunlit_heap,
        reserved_available_time: dict[int, float],
        time_s: int,
        exclude_sat_id: int,
    ) -> tuple[int, float] | None:
        """Peek a valid candidate without consuming its heap entry."""
        import heapq

        excluded_entries = []
        candidate = None

        while sunlit_heap:
            recorded_load, sat_id = heapq.heappop(sunlit_heap)
            current_load = max(
                0.0,
                reserved_available_time[sat_id] - float(time_s),
            )

            # Updated reservations always push a new entry, so an old entry
            # can be discarded instead of being inserted again.
            if abs(recorded_load - current_load) > 1e-9:
                continue

            if sat_id == exclude_sat_id:
                excluded_entries.append((recorded_load, sat_id))
                continue

            candidate = (sat_id, reserved_available_time[sat_id])
            heapq.heappush(sunlit_heap, (recorded_load, sat_id))
            break

        for entry in excluded_entries:
            heapq.heappush(sunlit_heap, entry)

        return candidate

    def __init__(self) -> None:
        super().__init__()
        self._route_cache_key = None
        self._route_parents_cache: dict[int, dict[int, int | None]] = {}
        self._route_cost_cache: dict[tuple, RouteCost] = {}

    def _short_circuit_local_action(
        self,
        *,
        source: SatelliteView,
        local_hard_safe: bool,
        local_finish: float | None,
        time_s: int,
        step_s: int,
    ) -> bool:
        """Whether a safe sunlit local action is cheap enough to stop search.

        This is an optimization, not policy.  It is only valid while local
        execution does not add cross-slot queue pressure.  Under high loading,
        blindly accepting every safe sunlit local task hides the load term from
        the original cost model and starves offload opportunities.
        """

        return (
            source.sunlit
            and local_hard_safe
            and local_finish is not None
            and local_finish <= float(time_s) + step_s
        )

    def _blocked_route_relays(
        self,
        *,
        ordered_tasks: list[Task],
        satellite_by_id: dict[int, SatelliteView],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        isl_config: ISLConfig,
    ) -> set[int]:
        max_relay_energy_j = max(
            (
                transmission_energy_j(task.input_bits, isl_config)
                + transmission_energy_j(task.output_bits, isl_config)
                for task in ordered_tasks
            ),
            default=0.0,
        )
        return {
            sat_id
            for sat_id, sat in satellite_by_id.items()
            if not sat.sunlit
            and sat.battery_j
            - reserved_energy_for_sat(reserved_energy, sat_id)
            - max_relay_energy_j
            < battery.min_safe_j
        }

    def _peek_least_loaded_safe_eclipse_mod(
        self,
        *,
        eclipse_heap,
        compute_rejections: list[tuple[float, int]],
        route_cost_for,
        reserved_available_time: dict[int, float],
        reserved_energy: dict[int, float],
        satellite_by_id: dict[int, SatelliteView],
        route_parents: dict[int, int | None],
        task: Task,
        time_s: int,
        step_s: int,
        deadline_time: float,
        compute_time_s: float,
        compute_power_w: float,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        exclude_sat_id: int,
        protected_cost: float | None,
    ) -> tuple[float, float, int, Route, RouteCost] | None:
        """Return the least-loaded reachable safe eclipse peer, if any."""
        import heapq

        skipped_entries = []
        chosen = None
        compute_energy_j = compute_time_s * compute_power_w
        while (
            compute_rejections
            and -compute_rejections[0][0] > compute_energy_j + 1.0e-9
        ):
            _neg_rejected_compute_j, sat_id = heapq.heappop(compute_rejections)
            heapq.heappush(
                eclipse_heap,
                (
                    max(
                        0.0,
                        reserved_available_time[sat_id] - float(time_s),
                    ),
                    sat_id,
                ),
            )
        eclipse_floor = (
            0.0
            if protected_cost is None
            else 1.0 - self._clip_unit(protected_cost)
        )

        while eclipse_heap:
            recorded_load, sat_id = heapq.heappop(eclipse_heap)
            current_load = max(
                0.0,
                reserved_available_time[sat_id] - float(time_s),
            )

            if abs(recorded_load - current_load) > 1e-9:
                heapq.heappush(eclipse_heap, (current_load, sat_id))
                continue

            if sat_id == exclude_sat_id:
                skipped_entries.append((recorded_load, sat_id))
                continue

            if (
                isinstance(reserved_energy, BatteryReservation)
                and not reserved_energy.allows_compute(
                    sat_id=sat_id,
                    compute_time_s=compute_time_s,
                )
            ):
                heapq.heappush(
                    compute_rejections,
                    (-compute_energy_j, sat_id),
                )
                continue

            result = self._normalized_local_cost(
                sat=satellite_by_id[sat_id],
                available_time_s=reserved_available_time[sat_id],
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_power_w=compute_power_w,
                battery=battery,
            )
            if result is None:
                skipped_entries.append((recorded_load, sat_id))
                break

            cost, finish_time, _battery_safe = result
            reversed_route_nodes = route_nodes_reversed(
                route_parents,
                sat_id,
            )
            assert reversed_route_nodes is not None

            route = Route(tuple(reversed(reversed_route_nodes)))
            route_cost = route_cost_for(task, route)
            route_is_safe = route_respects_battery_projection(
                route=route,
                route_cost=route_cost,
                satellite_by_id=satellite_by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            )
            if not route_is_safe:
                skipped_entries.append((recorded_load, sat_id))
                continue

            chosen = (
                max(cost, eclipse_floor),
                finish_time,
                sat_id,
                route,
                route_cost,
            )
            skipped_entries.append((recorded_load, sat_id))
            break

        for entry in skipped_entries:
            heapq.heappush(eclipse_heap, entry)

        return chosen

    def assign_tasks(
        self,
        *,
        tasks: list[Task],
        satellite_views: list[SatelliteView],
        time_s: int,
        step_s: int,
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
        isl_graph: ISLGraph,
    ) -> list[Assignment]:
        import heapq

        by_id = {sat.sat_id: sat for sat in satellite_views}

        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s
            for sat in satellite_views
        }
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )

        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )

        def route_cost_for(task: Task, route: Route) -> RouteCost:
            key = (
                route.nodes,
                task.input_bits,
                task.output_bits,
                compute_config.cycles_per_input_bit,
                compute_config.cpu_frequency_hz,
                compute_config.cpu_power_w,
                isl_config.rate_bps,
                isl_config.tx_power_w,
            )
            cost = self._route_cost_cache.get(key)
            if cost is None:
                cost = estimate_route_cost(
                    task=task,
                    route=route,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )
                if len(self._route_cost_cache) >= 65536:
                    self._route_cost_cache.clear()
                self._route_cost_cache[key] = cost
            return cost

        blocked_route_relays = self._blocked_route_relays(
            ordered_tasks=ordered_tasks,
            satellite_by_id=by_id,
            reserved_energy=reserved_energy,
            battery=battery,
            isl_config=isl_config,
        )
        route_cache_key = (
            tuple(isl_graph.adjacency.items()),
            frozenset(blocked_route_relays),
        )
        if route_cache_key != self._route_cache_key:
            self._route_cache_key = route_cache_key
            self._route_parents_cache = {}
        route_parents_by_source = self._route_parents_cache

        def route_parents_for_source(source_sat: int) -> dict[int, int | None]:
            parents = route_parents_by_source.get(source_sat)
            if parents is None:
                parents = (
                    build_route_tree(
                        isl_graph,
                        source_sat,
                        blocked_route_relays - {source_sat},
                    )
                    if blocked_route_relays
                    else build_route_tree(isl_graph, source_sat)
                )
                route_parents_by_source[source_sat] = parents
            return parents

        sunlit_heap = []
        for sat in satellite_views:
            if sat.sunlit:
                heapq.heappush(
                    sunlit_heap,
                    (
                        max(
                            0.0,
                            reserved_available_time[sat.sat_id] - float(time_s),
                        ),
                        sat.sat_id,
                    ),
                )
        eclipse_heaps_by_source: dict[int, list[tuple[float, int]]] = {}
        eclipse_compute_rejections_by_source: dict[
            int,
            list[tuple[float, int]],
        ] = {}

        def eclipse_heap_for_source(
            source_sat: int,
            route_parents: dict[int, int | None],
        ) -> list[tuple[float, int]]:
            heap = eclipse_heaps_by_source.get(source_sat)
            if heap is None:
                heap = [
                    (
                        max(
                            0.0,
                            reserved_available_time[sat_id] - float(time_s),
                        ),
                        sat_id,
                    )
                    for sat_id in route_parents
                    if sat_id != source_sat and not by_id[sat_id].sunlit
                ]
                heapq.heapify(heap)
                eclipse_heaps_by_source[source_sat] = heap
            return heap

        assignments: list[Assignment] = []
        deferred_workload_s = {sat.sat_id: 0.0 for sat in satellite_views}

        for task in ordered_tasks:
            assert task.source_sat is not None

            source = by_id[task.source_sat]
            deadline_time = task.created_time_s + task.deadline_s
            compute_time_s = task_compute_time_s(task, compute_config)
            local_route = Route((source.sat_id,))

            local_result = self._normalized_local_cost(
                sat=source,
                available_time_s=reserved_available_time[source.sat_id],
                time_s=time_s,
                step_s=step_s,
                deadline_time=deadline_time,
                compute_time_s=compute_time_s,
                compute_power_w=compute_config.cpu_power_w,
                battery=battery,
            )

            local_cost = float("inf")
            local_finish = None
            local_hard_safe = False
            if local_result is not None:
                candidate_cost, candidate_finish, _battery_safe = local_result
                local_route_cost = RouteCost(
                    compute_time_s=compute_time_s,
                    transmission_time_s=0.0,
                    energy_by_sat={
                        source.sat_id: compute_time_s * compute_config.cpu_power_w
                    },
                )
                if route_respects_battery_projection(
                    route=local_route,
                    route_cost=local_route_cost,
                    satellite_by_id=by_id,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    time_s=time_s,
                    step_s=step_s,
                    battery=battery,
                    compute_config=compute_config,
                ):
                    local_cost = candidate_cost
                    local_finish = candidate_finish
                    local_hard_safe = True
            prefer_local = self._short_circuit_local_action(
                source=source,
                local_hard_safe=local_hard_safe,
                local_finish=local_finish,
                time_s=time_s,
                step_s=step_s,
            )

            sun_cost = float("inf")
            sun_finish = None
            sun_sat_id = None
            sun_route = None

            best_sunlit = (
                self._peek_least_loaded_sunlit_mod(
                    sunlit_heap=sunlit_heap,
                    reserved_available_time=reserved_available_time,
                    time_s=time_s,
                    exclude_sat_id=source.sat_id,
                )
                if not prefer_local
                else None
            )

            if best_sunlit is not None:
                candidate_sat_id, candidate_available_time = best_sunlit
                reversed_route_nodes = route_nodes_reversed(
                    route_parents_for_source(source.sat_id),
                    candidate_sat_id,
                )

                if reversed_route_nodes is not None:
                    sun_result = self._normalized_local_cost(
                        sat=by_id[candidate_sat_id],
                        available_time_s=candidate_available_time,
                        time_s=time_s,
                        step_s=step_s,
                        deadline_time=deadline_time,
                        compute_time_s=compute_time_s,
                        compute_power_w=compute_config.cpu_power_w,
                        battery=battery,
                    )

                    if sun_result is not None:
                        candidate_cost, candidate_finish, _battery_safe = sun_result
                        route = Route(tuple(reversed(reversed_route_nodes)))
                        route_cost = route_cost_for(task, route)
                        route_is_safe = route_respects_battery_projection(
                            route=route,
                            route_cost=route_cost,
                            satellite_by_id=by_id,
                            reserved_available_time=reserved_available_time,
                            reserved_energy=reserved_energy,
                            time_s=time_s,
                            step_s=step_s,
                            battery=battery,
                            compute_config=compute_config,
                        )
                        if route_is_safe:
                            sun_cost = candidate_cost
                            sun_finish = candidate_finish
                            sun_sat_id = candidate_sat_id
                            sun_route = route
                            sun_route_cost = route_cost

            prefer_sunlit = sun_cost < float("inf")

            protected_costs = []
            if source.sunlit and local_hard_safe and local_cost < float("inf"):
                protected_costs.append(local_cost)
            if sun_cost < float("inf"):
                protected_costs.append(sun_cost)
            protected_cost = min(protected_costs) if protected_costs else None

            eclipse_choice = (
                self._peek_least_loaded_safe_eclipse_mod(
                    eclipse_heap=eclipse_heap_for_source(
                        source.sat_id,
                        route_parents_for_source(source.sat_id),
                    ),
                    compute_rejections=eclipse_compute_rejections_by_source.setdefault(
                        source.sat_id,
                        [],
                    ),
                    route_cost_for=route_cost_for,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    satellite_by_id=by_id,
                    route_parents=route_parents_for_source(source.sat_id),
                    task=task,
                    time_s=time_s,
                    step_s=step_s,
                    deadline_time=deadline_time,
                    compute_time_s=compute_time_s,
                    compute_power_w=compute_config.cpu_power_w,
                    battery=battery,
                    compute_config=compute_config,
                    exclude_sat_id=source.sat_id,
                    protected_cost=protected_cost,
                )
                if not prefer_local and not prefer_sunlit
                else None
            )
            eclipse_cost = float("inf")
            eclipse_finish = None
            eclipse_sat_id = None
            eclipse_route = None
            eclipse_route_cost = None
            if eclipse_choice is not None:
                (
                    eclipse_cost,
                    eclipse_finish,
                    eclipse_sat_id,
                    eclipse_route,
                    eclipse_route_cost,
                ) = eclipse_choice

            has_immediate_execution = (
                (local_hard_safe and local_cost < float("inf"))
                or sun_cost < float("inf")
                or eclipse_cost < float("inf")
            )
            if has_immediate_execution:
                defer_cost = 1.0
            else:
                defer_cost = self._normalized_defer_cost(
                    time_s=time_s,
                    step_s=step_s,
                    deadline_time=deadline_time,
                    compute_time_s=compute_time_s,
                    deferred_workload_s=deferred_workload_s[source.sat_id],
                    has_feasible_sunlit_execution=False,
                    local_eclipse_cost=None,
                )

            if prefer_local:
                action = "local"
            elif prefer_sunlit:
                action = "sunlit"
            elif source.sunlit and local_hard_safe:
                action_costs = [
                    ("local", local_cost),
                    ("sunlit", sun_cost),
                    ("eclipse", eclipse_cost),
                    ("defer", defer_cost),
                ]
                action, _ = min(action_costs, key=lambda x: x[1])
            elif local_hard_safe:
                action_costs = [
                    ("sunlit", sun_cost),
                    ("local", local_cost),
                    ("eclipse", eclipse_cost),
                    ("defer", defer_cost),
                ]
                action, _ = min(action_costs, key=lambda x: x[1])
            else:
                action_costs = [
                    ("sunlit", sun_cost),
                    ("eclipse", eclipse_cost),
                    ("defer", defer_cost),
                ]
                action, _ = min(action_costs, key=lambda x: x[1])

            if action == "local" and local_finish is not None:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=local_route,
                        mode="local",
                        score=local_cost,
                    )
                )

                reserved_available_time[source.sat_id] = local_finish
                reserve_route_transmission_energy(
                    route=local_route,
                    route_cost=local_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )

                if source.sunlit:
                    heapq.heappush(
                        sunlit_heap,
                        (
                            max(0.0, local_finish - float(time_s)),
                            source.sat_id,
                        ),
                    )

            elif (
                action == "sunlit"
                and sun_sat_id is not None
                and sun_finish is not None
                and sun_route is not None
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=sun_route,
                        mode="offload",
                        score=sun_cost,
                    )
                )

                reserved_available_time[sun_sat_id] = sun_finish
                assert sun_route_cost is not None
                reserve_route_transmission_energy(
                    route=sun_route,
                    route_cost=sun_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )
                heapq.heappush(
                    sunlit_heap,
                    (
                        max(0.0, sun_finish - float(time_s)),
                        sun_sat_id,
                    ),
                )

            elif (
                action == "eclipse"
                and eclipse_sat_id is not None
                and eclipse_finish is not None
                and eclipse_route is not None
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=eclipse_route,
                        mode="offload",
                        score=eclipse_cost,
                    )
                )

                reserved_available_time[eclipse_sat_id] = eclipse_finish
                assert eclipse_route_cost is not None
                reserve_route_transmission_energy(
                    route=eclipse_route,
                    route_cost=eclipse_route_cost,
                    compute_config=compute_config,
                    reserved_energy=reserved_energy,
                )

            elif defer_cost < float("inf"):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="defer",
                        score=defer_cost,
                    )
                )
                deferred_workload_s[source.sat_id] += compute_time_s

            else:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=Route((source.sat_id,)),
                        mode="fail",
                        score=float("inf"),
                        failed_reason="no_safe_capacity_before_deadline",
                    )
                )

        return assignments



class PhoenixScheduler(Scheduler):
    """Schedule tasks using bounded PHOENIX energy-aware peer selection."""

    name = "phoenix"
    queue_discipline = "edf"

    def __init__(self) -> None:
        self.plane_load_by_plane: dict[int, float] = {}

    def _candidate_finish_time(
        self,
        *,
        task: Task,
        route: Route,
        target: SatelliteView,
        time_s: int,
        reserved_available_time: dict[int, float],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[float, RouteTiming] | None:
        timing = estimate_route_timing(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        arrival_time = float(time_s) + timing.transmission_time_s
        start_time = max(arrival_time, reserved_available_time[target.sat_id])
        finish_time = start_time + timing.compute_time_s
        deadline_time = task.created_time_s + task.deadline_s
        if finish_time > deadline_time:
            return None
        return finish_time, timing

    @staticmethod
    def _plane_of(sat: SatelliteView) -> int | None:
        if sat.plane is None or sat.plane < 0:
            return None
        return sat.plane

    def _plane_sunlit_counts(
        self,
        satellite_views: list[SatelliteView],
    ) -> dict[int, int]:
        counts: dict[int, int] = {}
        for sat in satellite_views:
            plane = self._plane_of(sat)
            if plane is None:
                continue
            counts.setdefault(plane, 0)
            if sat.sunlit:
                counts[plane] += 1
        return counts

    def _candidate_cache(
        self,
        satellite_views: list[SatelliteView],
    ) -> PhoenixCandidateCache:
        sunlit_by_plane: dict[int, list[SatelliteView]] = {}
        sunlit_global: list[SatelliteView] = []
        sunlit_counts_by_plane: dict[int, int] = {}

        for sat in satellite_views:
            plane = self._plane_of(sat)
            if plane is not None:
                sunlit_counts_by_plane.setdefault(plane, 0)
            if not sat.sunlit:
                continue

            sunlit_global.append(sat)
            if plane is not None:
                sunlit_counts_by_plane[plane] += 1
                sunlit_by_plane.setdefault(plane, []).append(sat)

        def by_battery(candidates: list[SatelliteView]) -> tuple[SatelliteView, ...]:
            return tuple(
                sorted(
                    candidates,
                    key=lambda sat: (-sat.battery_j, sat.sat_id),
                )
            )

        return PhoenixCandidateCache(
            sunlit_by_plane={
                plane: by_battery(candidates)
                for plane, candidates in sunlit_by_plane.items()
            },
            sunlit_global=by_battery(sunlit_global),
            sunlit_counts_by_plane=sunlit_counts_by_plane,
        )

    def _target_plane(
        self,
        candidate_cache: PhoenixCandidateCache,
    ) -> int | None:
        sunlit_counts = candidate_cache.sunlit_counts_by_plane
        planes_with_sunlight = [
            plane for plane, sunlit_count in sunlit_counts.items() if sunlit_count > 0
        ]
        if not planes_with_sunlight:
            return None
        return min(
            planes_with_sunlight,
            key=lambda plane: (
                self.plane_load_by_plane.get(plane, 0.0) / max(1, sunlit_counts[plane]),
                self.plane_load_by_plane.get(plane, 0.0),
                plane,
            ),
        )

    def _candidate_energy_score(
        self,
        *,
        task: Task,
        route: Route,
        target: SatelliteView,
        finish_time: float,
        time_s: int,
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> float:
        """Estimate PHOENIX-style residual energy for a peer candidate.

        The paper uses future solar energy, current battery, and queued work.
        The simulator does not keep a full future sunlight matrix here, so use
        the cheap state we already have: current sunlight over this task window,
        current battery, already-reserved work, and this route's target energy.
        """

        route_cost = estimate_route_cost(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        window_s = max(0.0, finish_time - float(time_s))
        harvest_j = battery.harvest_w * window_s if target.sunlit else 0.0
        return min(
            battery.capacity_j,
            target.battery_j
            + harvest_j
            - reserved_energy_for_sat(reserved_energy, target.sat_id)
            - route_cost.energy_for(target.sat_id),
        )

    def _best_peer_in_planes(
        self,
        *,
        task: Task,
        source: SatelliteView,
        candidates: tuple[SatelliteView, ...],
        routes_by_target: dict[int, Route],
        satellite_views_by_id: dict[int, SatelliteView],
        time_s: int,
        step_s: int,
        reserved_available_time: dict[int, float],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[Assignment, float] | None:
        best_assignment = None
        best_finish = None
        best_key = (float("inf"), float("inf"), float("inf"), float("inf"))

        for target in candidates:
            if target.sat_id == source.sat_id:
                continue

            route = routes_by_target.get(target.sat_id)
            if route is None:
                continue

            feasible = self._candidate_finish_time(
                task=task,
                route=route,
                target=target,
                time_s=time_s,
                reserved_available_time=reserved_available_time,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if feasible is None:
                continue
            finish_time, _timing = feasible
            route_cost = estimate_route_cost(
                task=task,
                route=route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if not route_respects_battery_projection(
                route=route,
                route_cost=route_cost,
                satellite_by_id=satellite_views_by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            ):
                continue
            energy_score = self._candidate_energy_score(
                task=task,
                route=route,
                target=target,
                finish_time=finish_time,
                time_s=time_s,
                reserved_energy=reserved_energy,
                battery=battery,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            key = (-energy_score, finish_time, route.hop_count, target.sat_id)

            if key < best_key:
                best_assignment = Assignment(
                    task_id=task.task_id,
                    route=route,
                    mode="offload",
                    score=energy_score,
                )
                best_finish = finish_time
                best_key = key

        if best_assignment is None:
            return None
        assert best_finish is not None
        return best_assignment, best_finish

    def _choose_peer(
        self,
        *,
        task: Task,
        source: SatelliteView,
        isl_graph: ISLGraph,
        routes_by_source: dict[int, dict[int, Route]],
        searched_targets_by_source: dict[int, set[int]],
        candidate_cache: PhoenixCandidateCache,
        satellite_views_by_id: dict[int, SatelliteView],
        time_s: int,
        step_s: int,
        reserved_available_time: dict[int, float],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[Assignment, float] | None:
        target_plane = self._target_plane(candidate_cache)
        candidates = (
            candidate_cache.sunlit_global
            if target_plane is None
            else candidate_cache.sunlit_by_plane.get(target_plane, ())
        )
        if not candidates:
            return None

        routes_by_target = routes_by_source.setdefault(source.sat_id, {})
        searched_targets = searched_targets_by_source.setdefault(source.sat_id, set())
        candidate_target_ids = {target.sat_id for target in candidates}
        missing_target_ids = candidate_target_ids - searched_targets
        if missing_target_ids:
            routes_by_target.update(
                routes_to_targets(isl_graph, source.sat_id, missing_target_ids)
            )
            searched_targets.update(missing_target_ids)

        return self._best_peer_in_planes(
            task=task,
            source=source,
            candidates=candidates,
            routes_by_target=routes_by_target,
            satellite_views_by_id=satellite_views_by_id,
            time_s=time_s,
            step_s=step_s,
            reserved_available_time=reserved_available_time,
            reserved_energy=reserved_energy,
            battery=battery,
            compute_config=compute_config,
            isl_config=isl_config,
        )

    def _choose_local(
        self,
        *,
        task: Task,
        source: SatelliteView,
        isl_graph: ISLGraph,
        satellite_views_by_id: dict[int, SatelliteView],
        time_s: int,
        step_s: int,
        reserved_available_time: dict[int, float],
        reserved_energy: dict[int, float],
        battery: BatteryConfig,
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
        mode: str,
    ) -> tuple[Assignment, float] | None:
        route = route_or_raise(isl_graph, source.sat_id, source.sat_id)
        feasible = self._candidate_finish_time(
            task=task,
            route=route,
            target=source,
            time_s=time_s,
            reserved_available_time=reserved_available_time,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        if feasible is None:
            return None
        finish_time, _timing = feasible
        route_cost = estimate_route_cost(
            task=task,
            route=route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        if not route_respects_battery_projection(
            route=route,
            route_cost=route_cost,
            satellite_by_id=satellite_views_by_id,
            reserved_available_time=reserved_available_time,
            reserved_energy=reserved_energy,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        ):
            return None
        return (
            Assignment(
                task_id=task.task_id,
                route=route,
                mode=mode,
                score=finish_time,
            ),
            finish_time,
        )

    def _remember_assignment_load(
        self,
        assignment: Assignment,
        by_id: dict[int, SatelliteView],
        load_j: float,
    ) -> None:
        target = by_id[assignment.target_sat]
        plane = self._plane_of(target)
        if plane is not None:
            self.plane_load_by_plane[plane] = (
                self.plane_load_by_plane.get(plane, 0.0) + load_j
            )


    def _defer_time_if_deadline_safe_with_reservation(
        self,
        *,
        task: Task,
        source: SatelliteView,
        time_s: int,
        step_s: int,
        deferred_available_time: dict[int, float],
        compute_config: ComputeConfig,
        isl_config: ISLConfig,
    ) -> tuple[float, float] | None:
        local_route = Route((source.sat_id,))
        timing = estimate_route_timing(
            task=task,
            route=local_route,
            compute_config=compute_config,
            isl_config=isl_config,
        )
        defer_until = source.next_sunlit_time_s
        if defer_until is None or defer_until <= float(time_s):
            defer_until = float(time_s + step_s)

        start_after_wait = max(
            defer_until,
            deferred_available_time[source.sat_id],
        )
        finish_after_wait = start_after_wait + timing.compute_time_s
        if finish_after_wait <= task.created_time_s + task.deadline_s:
            return defer_until, finish_after_wait
        return None

    def assign_tasks(
        self,
        *,
        tasks,
        satellite_views,
        time_s,
        step_s,
        battery,
        compute_config,
        isl_config,
        isl_graph,
    ):
        by_id = {sat.sat_id: sat for sat in satellite_views}
        reserved_available_time = {
            sat.sat_id: float(time_s) + sat.queue_backlog_s for sat in satellite_views
        }
        deferred_available_time = dict(reserved_available_time)
        reserved_energy = hard_limit_reserved_energy_by_sat(
            satellite_views=satellite_views,
            time_s=time_s,
            step_s=step_s,
            battery=battery,
            compute_config=compute_config,
        )
        ordered_tasks = sorted(
            tasks,
            key=lambda task: (task.created_time_s + task.deadline_s, task.task_id),
        )
        candidate_cache = self._candidate_cache(satellite_views)

        # PHOENIX's orbit-level load is a scheduling-horizon signal, not a
        # lifetime counter.  Use one assign_tasks() batch as the horizon and
        # expose the last batch for diagnostics.
        self.plane_load_by_plane = {}

        assignments = []
        routes_by_source: dict[int, dict[int, Route]] = {}
        searched_targets_by_source: dict[int, set[int]] = {}

        for task in ordered_tasks:
            assert task.source_sat is not None
            source = by_id[task.source_sat]

            chosen = None
            deferred = None

            if source.sunlit:
                chosen = self._choose_local(
                    task=task,
                    source=source,
                    isl_graph=isl_graph,
                    satellite_views_by_id=by_id,
                    time_s=time_s,
                    step_s=step_s,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    compute_config=compute_config,
                    isl_config=isl_config,
                    mode="local",
                )
            else:
                deferred = self._defer_time_if_deadline_safe_with_reservation(
                    task=task,
                    source=source,
                    time_s=time_s,
                    step_s=step_s,
                    deferred_available_time=deferred_available_time,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )

            if not source.sunlit and deferred is not None:
                defer_until, deferred_finish_time = deferred
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                        mode="defer",
                        score=defer_until,
                    )
                )
                deferred_available_time[source.sat_id] = deferred_finish_time
                continue

            if chosen is None:
                chosen = self._choose_peer(
                    task=task,
                    source=source,
                    isl_graph=isl_graph,
                    routes_by_source=routes_by_source,
                    searched_targets_by_source=searched_targets_by_source,
                    candidate_cache=candidate_cache,
                    satellite_views_by_id=by_id,
                    time_s=time_s,
                    step_s=step_s,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    compute_config=compute_config,
                    isl_config=isl_config,
                )

            if chosen is None and not source.sunlit:
                chosen = self._choose_local(
                    task=task,
                    source=source,
                    isl_graph=isl_graph,
                    satellite_views_by_id=by_id,
                    time_s=time_s,
                    step_s=step_s,
                    reserved_available_time=reserved_available_time,
                    reserved_energy=reserved_energy,
                    battery=battery,
                    compute_config=compute_config,
                    isl_config=isl_config,
                    mode="local",
                )

            if chosen is None:
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=route_or_raise(isl_graph, source.sat_id, source.sat_id),
                        mode="fail",
                        score=float("inf"),
                        failed_reason="no_feasible_candidate",
                    )
                )
                continue

            assignment, finish_time = chosen
            cost = estimate_route_cost(
                task=task,
                route=assignment.route,
                compute_config=compute_config,
                isl_config=isl_config,
            )
            if not route_respects_battery_projection(
                route=assignment.route,
                route_cost=cost,
                satellite_by_id=by_id,
                reserved_available_time=reserved_available_time,
                reserved_energy=reserved_energy,
                time_s=time_s,
                step_s=step_s,
                battery=battery,
                compute_config=compute_config,
            ):
                assignments.append(
                    Assignment(
                        task_id=task.task_id,
                        route=assignment.route,
                        mode="fail",
                        score=float("inf"),
                        failed_reason="battery_hard_constraint",
                    )
                )
                continue

            assignments.append(assignment)
            reserved_available_time[assignment.target_sat] = finish_time
            reserve_route_transmission_energy(
                route=assignment.route,
                route_cost=cost,
                compute_config=compute_config,
                reserved_energy=reserved_energy,
            )
            self._remember_assignment_load(
                assignment,
                by_id,
                load_j=cost.energy_for(assignment.target_sat),
            )

        return assignments


SCHEDULER_TYPES: dict[str, type[Scheduler]] = {
    scheduler_type.name: scheduler_type
    for scheduler_type in (
        LocalOnlyScheduler,
        NearestSunlitScheduler,
        GreedyEnergyScheduler,
        StarlitScheduler,
        PhoenixScheduler,
    )
}


def create_scheduler(name: str) -> Scheduler:
    try:
        return SCHEDULER_TYPES[name]()
    except KeyError as exc:
        raise ValueError(f"unknown scheduler: {name}") from exc
