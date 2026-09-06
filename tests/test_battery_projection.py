from __future__ import annotations

from dataclasses import replace

import pytest

from simulator.isl import ISLGraph
from simulator.models import (
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    SatelliteView,
    Task,
)
from simulator.scheduler import (
    BatteryReservation,
    create_scheduler,
    minimum_projected_battery_until_recharge,
    reserve_route_transmission_energy,
    route_respects_battery_projection,
)
from simulator.models import Route
from simulator.route_cost import RouteCost


def _battery() -> BatteryConfig:
    return BatteryConfig(
        capacity_j=100.0,
        initial_j=100.0,
        min_safe_j=50.0,
        harvest_w=2.0,
        idle_w=1.0,
    )


def _compute() -> ComputeConfig:
    return ComputeConfig(
        cycles_per_input_bit=1.0,
        cpu_frequency_hz=1.0,
        cpu_power_w=1.0,
    )



def _sunlit_satellite() -> SatelliteView:
    return SatelliteView(
        sat_id=0,
        x_km=0.0,
        y_km=0.0,
        z_km=0.0,
        sunlit=True,
        battery_j=80.0,
        queue_backlog_s=0.0,
        plane=0,
        slot=0,
        next_eclipse_time_s=10.0,
        next_sunlit_time_s=40.0,
    )


def _task() -> Task:
    return Task(
        task_id=1,
        created_time_s=0,
        source_sat=0,
        input_bits=15.0,
        output_bits=0.0,
        deadline_s=100.0,
    )


def test_projection_reserves_upcoming_eclipse_idle() -> None:
    sat = _sunlit_satellite()

    without_task = minimum_projected_battery_until_recharge(
        sat=sat,
        available_time_s=0.0,
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
    )
    with_task = minimum_projected_battery_until_recharge(
        sat=sat,
        available_time_s=0.0,
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
        extra_compute_time_s=15.0,
    )

    assert without_task >= _battery().min_safe_j
    assert with_task < _battery().min_safe_j


def test_batch_reservation_projects_once_then_uses_constant_time_budget(
    monkeypatch,
) -> None:
    import simulator.scheduler as scheduler_module

    calls = 0
    real_projection = scheduler_module.minimum_projected_battery_until_recharge

    def counted_projection(**kwargs):
        nonlocal calls
        calls += 1
        return real_projection(**kwargs)

    monkeypatch.setattr(
        scheduler_module,
        "minimum_projected_battery_until_recharge",
        counted_projection,
    )
    sat = _sunlit_satellite()
    reservation = BatteryReservation.build(
        satellite_views=[sat],
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
    )
    route = Route((0,))
    cost = RouteCost(5.0, 0.0, {0: 5.0})
    kwargs = dict(
        route=route,
        route_cost=cost,
        satellite_by_id={0: sat},
        reserved_available_time={0: 0.0},
        reserved_energy=reservation,
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
    )

    assert calls == 1
    assert route_respects_battery_projection(**kwargs)
    assert route_respects_battery_projection(**kwargs)
    assert calls == 1

    reserve_route_transmission_energy(
        route=route,
        route_cost=cost,
        compute_config=_compute(),
        reserved_energy=reservation,
    )
    assert reservation.remaining_j[0] < 30.0


def test_battery_reservation_keeps_headroom_and_spending_separate() -> None:
    sat = _sunlit_satellite()
    reservation = BatteryReservation.build(
        satellite_views=[sat],
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
    )
    initial_headroom = reservation.remaining_j[0]
    route = Route((0,))
    cost = RouteCost(15.0, 0.0, {0: 15.0})

    assert not isinstance(reservation, dict)
    reservation.reserve(route=route, route_cost=cost)

    assert reservation.remaining_j[0] == pytest.approx(initial_headroom - 15.0)
    assert reservation.spent_transmission_j[0] == 0.0


def test_phoenix_tries_safe_peer_after_local_battery_rejection() -> None:
    source = _sunlit_satellite()
    peer = SatelliteView(
        sat_id=1,
        x_km=1.0,
        y_km=0.0,
        z_km=0.0,
        sunlit=True,
        battery_j=100.0,
        queue_backlog_s=0.0,
        plane=0,
        slot=1,
        next_eclipse_time_s=100.0,
        next_sunlit_time_s=140.0,
    )

    assignments = create_scheduler("phoenix").assign_tasks(
        tasks=[_task()],
        satellite_views=[source, peer],
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
        isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
        isl_graph=ISLGraph({0: (1,), 1: (0,)}),
    )

    assert len(assignments) == 1
    assert assignments[0].mode == "offload"
    assert assignments[0].target_sat == 1


def test_starlit_projects_once_per_satellite_not_once_per_task(monkeypatch) -> None:
    import simulator.scheduler as scheduler_module

    calls = 0
    real_projection = scheduler_module.minimum_projected_battery_until_recharge

    def counted_projection(**kwargs):
        nonlocal calls
        calls += 1
        return real_projection(**kwargs)

    monkeypatch.setattr(
        scheduler_module,
        "minimum_projected_battery_until_recharge",
        counted_projection,
    )
    scheduler = create_scheduler("starlit")
    tasks = [
        Task(
            task_id=task_id,
            created_time_s=0,
            source_sat=0,
            input_bits=1.0,
            output_bits=0.0,
            deadline_s=100.0,
        )
        for task_id in range(20)
    ]
    satellite_views = [
        SatelliteView(
            sat_id=sat_id,
            x_km=float(sat_id),
            y_km=0.0,
            z_km=0.0,
            sunlit=True,
            battery_j=80.0,
            next_eclipse_time_s=100.0,
            next_sunlit_time_s=None,
            illumination_horizon_time_s=120.0,
            plane=0,
            slot=sat_id,
        )
        for sat_id in range(3)
    ]

    scheduler.assign_tasks(
        tasks=tasks,
        satellite_views=satellite_views,
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
        isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
        isl_graph=ISLGraph({0: (1,), 1: (0, 2), 2: (1,)}),
    )

    assert calls == len(satellite_views)


def test_starlit_stops_after_safe_sunlit_local_action(monkeypatch) -> None:
    import simulator.scheduler as scheduler_module
    from simulator.scheduler import StarlitScheduler

    class NoEclipseSearchStarlit(StarlitScheduler):
        def _peek_least_loaded_safe_eclipse_mod(self, **kwargs):
            raise AssertionError("eclipse action must not be evaluated")

    def unexpected_route_search(*args, **kwargs):
        raise AssertionError("local action must not build a route tree")

    monkeypatch.setattr(
        scheduler_module,
        "build_route_tree",
        unexpected_route_search,
    )

    sat = SatelliteView(
        sat_id=0,
        x_km=0.0,
        y_km=0.0,
        z_km=0.0,
        sunlit=True,
        battery_j=80.0,
        next_eclipse_time_s=100.0,
        illumination_horizon_time_s=120.0,
        plane=0,
        slot=0,
    )
    task = Task(
        task_id=99,
        created_time_s=0,
        source_sat=0,
        input_bits=1.0,
        output_bits=0.0,
        deadline_s=100.0,
    )

    assignments = NoEclipseSearchStarlit().assign_tasks(
        tasks=[task],
        satellite_views=[sat],
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
        isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
        isl_graph=ISLGraph({0: ()}),
    )

    assert assignments[0].mode == "local"


def test_starlit_does_not_short_circuit_congested_sunlit_local() -> None:
    from simulator.scheduler import StarlitScheduler

    source = SatelliteView(
        sat_id=0,
        x_km=0.0,
        y_km=0.0,
        z_km=0.0,
        sunlit=True,
        battery_j=80.0,
        queue_backlog_s=30.0,
        next_eclipse_time_s=100.0,
        illumination_horizon_time_s=120.0,
        plane=0,
        slot=0,
    )
    target = SatelliteView(
        sat_id=1,
        x_km=1.0,
        y_km=0.0,
        z_km=0.0,
        sunlit=True,
        battery_j=80.0,
        queue_backlog_s=0.0,
        next_eclipse_time_s=100.0,
        illumination_horizon_time_s=120.0,
        plane=0,
        slot=1,
    )
    task = Task(
        task_id=99,
        created_time_s=0,
        source_sat=0,
        input_bits=1.0,
        output_bits=0.0,
        deadline_s=100.0,
    )

    assignments = StarlitScheduler().assign_tasks(
        tasks=[task],
        satellite_views=[source, target],
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
        isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
        isl_graph=ISLGraph({0: (1,), 1: (0,)}),
    )

    assert assignments[0].mode == "offload"
    assert assignments[0].target_sat == 1


def test_starlit_does_not_rescan_compute_rejected_eclipse_targets(
    monkeypatch,
) -> None:
    from simulator.scheduler import StarlitScheduler

    class UnblockedStarlit(StarlitScheduler):
        def _blocked_route_relays(self, **kwargs):
            return set()

    calls = 0
    real_allows_compute = BatteryReservation.allows_compute

    def counted_allows_compute(self, **kwargs):
        nonlocal calls
        calls += 1
        return real_allows_compute(self, **kwargs)

    monkeypatch.setattr(BatteryReservation, "allows_compute", counted_allows_compute)
    satellites = [
        SatelliteView(
            sat_id=sat_id,
            x_km=float(sat_id),
            y_km=0.0,
            z_km=0.0,
            sunlit=False,
            battery_j=80.0,
            plane=0,
            slot=sat_id,
            next_sunlit_time_s=40.0,
        )
        for sat_id in range(3)
    ]
    tasks = [replace(_task(), task_id=task_id) for task_id in (1, 2)]

    UnblockedStarlit().assign_tasks(
        tasks=tasks,
        satellite_views=satellites,
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
        isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
        isl_graph=ISLGraph({0: (1, 2), 1: (0,), 2: (0,)}),
    )

    assert calls == 2


@pytest.mark.parametrize(
    "scheduler_name",
    [
        "local",
        "nearest-sunlit",
        "greedy-energy",
        "starlit",
        "phoenix",
    ],
)
def test_schedulers_do_not_accept_work_that_spends_eclipse_reserve(
    scheduler_name: str,
) -> None:
    scheduler = create_scheduler(scheduler_name)
    assignments = scheduler.assign_tasks(
        tasks=[_task()],
        satellite_views=[_sunlit_satellite()],
        time_s=0,
        step_s=20,
        battery=_battery(),
        compute_config=_compute(),
        isl_config=ISLConfig(rate_bps=1.0e9, tx_power_w=0.0),
        isl_graph=ISLGraph({0: ()}),
    )

    assert len(assignments) == 1
    assert assignments[0].mode in {"defer", "fail"}


def test_method7_is_not_a_scheduler_alias() -> None:
    with pytest.raises(ValueError, match="unknown scheduler"):
        create_scheduler("method7")


def test_phoenix2_is_not_a_scheduler_alias() -> None:
    with pytest.raises(ValueError, match="unknown scheduler"):
        create_scheduler("phoenix2")
