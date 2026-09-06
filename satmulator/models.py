from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConfig:
    capacity_j: float
    initial_j: float
    min_safe_j: float
    harvest_w: float
    idle_w: float


@dataclass(frozen=True)
class DemandPoint:
    lat_deg: float
    lon_deg: float
    weight: float


@dataclass(frozen=True)
class DemandDistribution:
    points: tuple[DemandPoint, ...]
    cumulative_weights: tuple[float, ...]
    total_weight: float

    def __bool__(self) -> bool:
        return bool(self.points)


@dataclass(frozen=True)
class TaskConfig:
    interval_s: int
    random_seed: int | None
    tasks_per_step: int
    input_bits: float
    output_bits: float
    deadline_s: float
    deadline_min_s: float
    demand_distribution: DemandDistribution
    min_elevation_deg: float


@dataclass(frozen=True)
class ComputeConfig:
    cycles_per_input_bit: float
    cpu_frequency_hz: float
    cpu_power_w: float


@dataclass(frozen=True)
class ISLConfig:
    rate_bps: float
    tx_power_w: float
    max_range_km: float = 5000.0


@dataclass(frozen=True)
class Task:
    task_id: int
    created_time_s: int
    source_sat: int | None
    input_bits: float
    output_bits: float
    deadline_s: float
    lat_deg: float | None = None
    lon_deg: float | None = None


@dataclass(frozen=True)
class Route:
    nodes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("route must contain at least one satellite")
        if any(not isinstance(node, int) or node < 0 for node in self.nodes):
            raise ValueError("route nodes must be non-negative satellite ids")

    @property
    def source_sat(self) -> int:
        return self.nodes[0]

    @property
    def target_sat(self) -> int:
        return self.nodes[-1]

    @property
    def hop_count(self) -> int:
        return len(self.nodes) - 1


@dataclass(frozen=True)
class Assignment:
    task_id: int
    route: Route
    mode: str
    score: float = 0.0
    failed_reason: str = ""

    @property
    def source_sat(self) -> int:
        return self.route.source_sat

    @property
    def target_sat(self) -> int:
        return self.route.target_sat

    @property
    def hop_count(self) -> int:
        return self.route.hop_count


@dataclass(frozen=True)
class QueuedTaskView:
    task_id: int
    absolute_deadline_s: float
    remaining_compute_time_s: float
    started: bool
    transmission_time_s: float = 0.0


@dataclass(frozen=True)
class SatelliteView:
    sat_id: int
    x_km: float
    y_km: float
    z_km: float
    sunlit: bool
    battery_j: float = 0.0
    queue_backlog_s: float = 0.0
    plane: int | None = None
    slot: int | None = None
    next_sunlit_time_s: float | None = None
    next_eclipse_time_s: float | None = None
    illumination_horizon_time_s: float | None = None
    pending_task_energy_j: float = 0.0
    queued_tasks: tuple[QueuedTaskView, ...] = ()


@dataclass(frozen=True)
class SatelliteState:
    time_s: int
    sat_id: int
    name: str
    plane: int
    slot: int
    x_km: float
    y_km: float
    z_km: float
    vx_km_s: float
    vy_km_s: float
    vz_km_s: float
    lat_deg: float | None
    lon_deg: float | None
    elevation_km: float | None
    sunlit: bool
    battery_j: float
    battery_pct: float
    harvested_j: float
    consumed_j: float
    safe_battery: bool
    generated_tasks: int
    completed_tasks: int
    failed_tasks: int
    task_energy_j: float
    task_compute_time_s: float = 0.0
    task_compute_energy_j: float = 0.0
    task_transmission_energy_j: float = 0.0
    deferred_tasks: int = 0
    local_completed_tasks: int = 0
    local_failed_tasks: int = 0
    queue_remaining_tasks: int = 0


@dataclass(frozen=True)
class SnapshotContext:
    projection_label: str
    sun_xy_unit: tuple[float, float] | None = None
    sun_eci_unit: tuple[float, float, float] | None = None
