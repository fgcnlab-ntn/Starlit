from __future__ import annotations

from .models import BatteryConfig


# 容許浮點運算的微小誤差
ENERGY_EPSILON_J = 1.0e-6


def battery_is_safe(battery_j: float, min_safe_j: float) -> bool:
    """判斷電量是否在誤差容許範圍內高於安全值"""

    return battery_j + ENERGY_EPSILON_J >= min_safe_j


def validate_battery_config(battery: BatteryConfig) -> None:
    if battery.capacity_j <= 0:
        raise ValueError("battery capacity must be positive")
    if battery.harvest_w < 0 or battery.idle_w < 0:
        raise ValueError("battery power values must be non-negative")
    if not 0 <= battery.initial_j <= battery.capacity_j:
        raise ValueError("initial battery must be within [0, capacity]")
    if not 0 <= battery.min_safe_j <= battery.capacity_j:
        raise ValueError("minimum safe battery must be within [0, capacity]")


def battery_step(
    *,
    battery_now: float,
    sunlit: bool,
    step_s: int,
    battery: BatteryConfig,
    task_energy_j: float,
    update: bool,
) -> tuple[float, float, float]:
    """回傳更新後電量、收集能量與待機耗能，單位均為 J"""

    if not update:
        return battery_now, 0.0, 0.0

    consumed_j = battery.idle_w * step_s
    harvested_j = battery.harvest_w * step_s if sunlit else 0.0
    battery_now = max(
        0.0,
        min(
            battery.capacity_j,
            battery_now - consumed_j - task_energy_j + harvested_j,
        ),
    )
    return battery_now, harvested_j, consumed_j
