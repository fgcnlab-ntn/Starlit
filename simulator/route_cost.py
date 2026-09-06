from __future__ import annotations

from dataclasses import dataclass

from .models import ComputeConfig, ISLConfig, Route, Task


@dataclass(frozen=True)
class RouteCost:
    compute_time_s: float
    transmission_time_s: float
    energy_by_sat: dict[int, float]

    @property
    def total_time_s(self) -> float:
        return self.compute_time_s + self.transmission_time_s

    @property
    def total_energy_j(self) -> float:
        return sum(self.energy_by_sat.values())

    def energy_for(self, sat_id: int) -> float:
        return self.energy_by_sat.get(sat_id, 0.0)


@dataclass(frozen=True)
class RouteTiming:
    compute_time_s: float
    transmission_time_s: float

    @property
    def total_time_s(self) -> float:
        return self.compute_time_s + self.transmission_time_s


def add_energy(energy_by_sat: dict[int, float], sat_id: int, energy_j: float) -> None:
    if energy_j == 0.0:
        return
    energy_by_sat[sat_id] = energy_by_sat.get(sat_id, 0.0) + energy_j


def transfer_time_s(bits: float, isl_config: ISLConfig) -> float:
    return bits / isl_config.rate_bps


def transmission_energy_j(bits: float, isl_config: ISLConfig) -> float:
    return isl_config.tx_power_w * transfer_time_s(bits, isl_config)


def compute_cycles(task: Task, compute_config: ComputeConfig) -> float:
    return task.input_bits * compute_config.cycles_per_input_bit


def task_compute_time_s(task: Task, compute_config: ComputeConfig) -> float:
    return compute_cycles(task, compute_config) / compute_config.cpu_frequency_hz


def estimate_route_timing(
    *,
    task: Task,
    route: Route,
    compute_config: ComputeConfig,
    isl_config: ISLConfig,
) -> RouteTiming:
    compute_time_s = task_compute_time_s(task, compute_config)
    transmission_time_s = route.hop_count * (
        transfer_time_s(task.input_bits, isl_config)
        + transfer_time_s(task.output_bits, isl_config)
    )
    return RouteTiming(
        compute_time_s=compute_time_s,
        transmission_time_s=transmission_time_s,
    )


def estimate_route_cost(
    *,
    task: Task,
    route: Route,
    compute_config: ComputeConfig,
    isl_config: ISLConfig,
) -> RouteCost:
    """Estimate execution cost for a task over a route.

    A single-node route means local execution.  A multi-node route sends input
    forward from source to target, computes at the target, then sends output
    back along the same path in reverse.  The two-node route is exactly the
    historical one-hop model.
    """

    timing = estimate_route_timing(
        task=task,
        route=route,
        compute_config=compute_config,
        isl_config=isl_config,
    )
    compute_energy_j = timing.compute_time_s * compute_config.cpu_power_w
    energy_by_sat: dict[int, float] = {}

    add_energy(energy_by_sat, route.target_sat, compute_energy_j)

    if route.hop_count == 0:
        return RouteCost(
            compute_time_s=timing.compute_time_s,
            transmission_time_s=0.0,
            energy_by_sat=energy_by_sat,
        )

    input_tx_energy_j = transmission_energy_j(task.input_bits, isl_config)
    output_tx_energy_j = transmission_energy_j(task.output_bits, isl_config)

    add_energy(energy_by_sat, route.source_sat, input_tx_energy_j)
    relay_energy_j = input_tx_energy_j + output_tx_energy_j
    for relay in route.nodes[1:-1]:
        add_energy(energy_by_sat, relay, relay_energy_j)
    add_energy(energy_by_sat, route.target_sat, output_tx_energy_j)

    return RouteCost(
        compute_time_s=timing.compute_time_s,
        transmission_time_s=timing.transmission_time_s,
        energy_by_sat=energy_by_sat,
    )
