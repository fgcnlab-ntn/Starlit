from __future__ import annotations

import sys
import time

from .config import SimulationConfig
from .models import SatelliteState
from .orbit import iter_circular_states
from .runlog import JsonObject, RunLog
from .scheduler import create_scheduler


def _format_seconds(seconds: float) -> str:
    minutes, seconds = divmod(int(max(0, seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class _ProgressBar:
    def __init__(self, total_steps: int, width: int = 40) -> None:
        self.total_steps = total_steps
        self.width = width
        self.started_at = time.monotonic()
        self.last_printed_at = 0.0

    def update(self, steps: int) -> None:
        now = time.monotonic()
        if now - self.last_printed_at < 1.0 and steps != self.total_steps:
            return

        elapsed = now - self.started_at
        rate = steps / elapsed if elapsed > 0 else 0.0
        remaining_steps = max(0, self.total_steps - steps)
        eta = remaining_steps / rate if rate > 0 else 0.0
        progress = steps / self.total_steps
        filled = int(self.width * progress)
        bar = "#" * filled + "-" * (self.width - filled)

        sys.stderr.write(
            "\r"
            f"Simulating: [{bar}] "
            f"{steps}/{self.total_steps} "
            f"({progress * 100:5.1f}%) "
            f"elapsed {_format_seconds(elapsed)} "
            f"eta {_format_seconds(eta)}"
        )
        sys.stderr.flush()
        self.last_printed_at = now

    @staticmethod
    def finish() -> None:
        sys.stderr.write("\n")


def _sunlight_counts(states: list[SatelliteState]) -> tuple[int, int]:
    sunlit = sum(state.sunlit for state in states)
    return sunlit, len(states) - sunlit


def _print_summary(
    *,
    config: SimulationConfig,
    scheduler_name: str,
    first: list[SatelliteState],
    last: list[SatelliteState],
    steps: int,
    summary: JsonObject,
) -> None:
    first_sunlit, first_eclipse = _sunlight_counts(first)
    final_sunlit, final_eclipse = _sunlight_counts(last)
    task_summary = summary["tasks"]
    battery_violations = summary.get("battery_violations", {})

    print("Minimal orbit simulation complete")
    print(f"  scheduler: {scheduler_name}")
    print(f"  satellites: {len(first)}")
    print(f"  planes: {config.planes}")
    print(
        f"  steps: {steps}, duration: {config.duration_s}s, "
        f"step: {config.step_s}s"
    )
    print(f"  t=0 sunlit/eclipse: {first_sunlit}/{first_eclipse}")
    print(f"  final sunlit/eclipse: {final_sunlit}/{final_eclipse}")
    print(
        "  final battery min/avg: "
        f"{min(state.battery_pct for state in last):.2f}%/"
        f"{sum(state.battery_pct for state in last) / len(last):.2f}%"
    )
    print(
        "  tasks completed/deferred/failed/pending: "
        f"{task_summary['completed']}/"
        f"{task_summary.get('deferred', 0)}/"
        f"{task_summary['failed']}/"
        f"{task_summary.get('pending', 0)}"
    )
    print(
        "  battery breaches total/eclipse: "
        f"{battery_violations.get('unique_breached_satellites', 0)}/"
        f"{battery_violations.get('unique_eclipse_breached_satellites', 0)}"
    )
    print(f"  output: {config.output_path.resolve()}")


def run(config: SimulationConfig) -> int:
    config.output_path.mkdir(parents=True, exist_ok=True)
    scheduler = create_scheduler(config.scheduler_name)
    run_log = RunLog(config.output_path, config.start, config.effective)
    progress = _ProgressBar(config.duration_s // config.step_s + 1)

    first = None
    last = None
    steps = 0
    try:
        step_iterator = iter_circular_states(
            start=config.start,
            satellites=config.satellites,
            planes=config.planes,
            altitude_km=config.altitude_km,
            inclination_deg=config.inclination_deg,
            sun_position_file=config.sun_position_file,
            duration_s=config.duration_s,
            step_s=config.step_s,
            battery=config.battery,
            compute_config=config.compute,
            task_config=config.task,
            isl_config=config.isl,
            scheduler=scheduler,
            walker_phase=config.walker_phase,
            task_event_sink=run_log.write_task_event,
            step_sink=run_log.write_step,
        )
        for steps, states in enumerate(step_iterator, start=1):
            if first is None:
                first = states
            last = states
            progress.update(steps)

        progress.finish()
        summary = run_log.complete()
    except BaseException as exc:
        run_log.fail(exc)
        raise

    assert first is not None and last is not None
    _print_summary(
        config=config,
        scheduler_name=scheduler.name,
        first=first,
        last=last,
        steps=steps,
        summary=summary,
    )
    return 0
