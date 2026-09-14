from __future__ import annotations

import argparse
import copy
import datetime as dt
from collections import deque
from pathlib import Path
from typing import TextIO

from simulator.config import SimulationConfig, load_config
from simulator.models import SatelliteState, SnapshotContext, TaskIntensity
from simulator.orbit import iter_circular_states
from simulator.runlog import (
    SCHEMA_VERSION,
    RunLog,
    append_json_line,
    write_json,
)
from simulator.scheduler import create_scheduler
from tools.visualizer import (
    BATTERY_CHART_WINDOW_STEPS,
    LIVE_METRICS_WINDOW_STEPS,
    Frame,
    LiveMetrics,
    LocalActivity,
    SatelliteMetadata,
    SatellitePosition,
    Visualizer,
)


class OnlineControlLog:
    """Crash-safe intensity audit log plus a directly runnable replay config."""

    def __init__(
        self,
        output_dir: Path,
        config: dict[str, object],
        *,
        task_capacity: int,
    ) -> None:
        if task_capacity <= 0:
            raise ValueError("task_capacity must be positive")
        self.output_dir = output_dir
        self.controls_path = output_dir / "controls.jsonl"
        self.replay_config_path = output_dir / "replay-config.json"
        self.controls_path.unlink(missing_ok=True)
        self.replay_config_path.unlink(missing_ok=True)
        self._config = copy.deepcopy(config)
        self._task_capacity = task_capacity
        task = self._config.get("task")
        if not isinstance(task, dict):
            raise ValueError("config.task must be an object")
        self._schedule = {
            int(change["time_s"]): int(change["tasks_per_step"])
            for change in task.get("intensity_schedule", [])
            if isinstance(change, dict)
        }
        self._stream: TextIO | None = None
        self._sequence = 0

    def record(
        self,
        *,
        selected_at_time_s: int,
        effective_time_s: int,
        tasks_per_step: int,
    ) -> None:
        if self._stream is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._stream = self.controls_path.open("w")
        self._sequence += 1
        append_json_line(
            self._stream,
            {
                "schema_version": SCHEMA_VERSION,
                "type": "task_intensity_changed",
                "sequence": self._sequence,
                "selected_at_time_s": selected_at_time_s,
                "effective_time_s": effective_time_s,
                "tasks_per_step": tasks_per_step,
                "load_pct": 100.0 * tasks_per_step / self._task_capacity,
                "task_capacity": self._task_capacity,
            },
            flush=True,
        )
        self._schedule[effective_time_s] = tasks_per_step
        self._write_replay_config()

    def _write_replay_config(self) -> None:
        replay = copy.deepcopy(self._config)
        task = replay["task"]
        assert isinstance(task, dict)
        task["intensity_schedule"] = [
            {"time_s": time_s, "tasks_per_step": tasks_per_step}
            for time_s, tasks_per_step in sorted(self._schedule.items())
        ]
        output = replay["output"]
        assert isinstance(output, dict)
        output["path"] = str(self.output_dir / "replay-output")
        temporary = self.replay_config_path.with_suffix(".json.tmp")
        write_json(temporary, replay)
        temporary.replace(self.replay_config_path)

    def close(self) -> None:
        if self._stream is not None and not self._stream.closed:
            self._stream.close()


def frame_from_states(
    start: dt.datetime,
    states: list[SatelliteState],
    context: SnapshotContext | None,
) -> Frame:
    if not states:
        raise ValueError("a visualizer frame must contain satellites")
    time_s = states[0].time_s
    if any(state.time_s != time_s for state in states):
        raise ValueError("all satellites in a frame must have the same time")
    return Frame(
        time_s=float(time_s),
        time_label=(start + dt.timedelta(seconds=time_s)).isoformat(),
        sun_eci_unit=None if context is None else context.sun_eci_unit,
        satellites=tuple(
            SatellitePosition(
                sat_id=state.sat_id,
                position_km=(state.x_km, state.y_km, state.z_km),
                sunlit=state.sunlit,
                battery_j=state.battery_j,
                task_compute_time_s=state.task_compute_time_s,
                new_ground_tasks=state.generated_tasks,
                local_activity=LocalActivity(
                    completed=state.local_completed_tasks,
                    failed=state.local_failed_tasks,
                    queue_remaining=state.queue_remaining_tasks,
                ),
            )
            for state in states
        ),
    )


def metrics_from_states(
    states: list[SatelliteState],
    *,
    generated: int,
    completed: int,
    failed: int,
) -> LiveMetrics:
    if not states:
        raise ValueError("live metrics require satellite states")
    pending = generated - completed - failed
    if pending < 0:
        raise ValueError("completed and failed tasks exceed generated tasks")
    return LiveMetrics(
        time_s=float(states[0].time_s),
        task_failure_rate_pct=(
            0.0 if generated == 0 else 100.0 * failed / generated
        ),
        pending_task_rate_pct=(
            0.0 if generated == 0 else 100.0 * pending / generated
        ),
    )


class OnlineFrames:
    """A bounded, append-only frame collection for the live renderer."""

    def __init__(
        self,
        first_frame: Frame,
        first_metrics: LiveMetrics,
        states: list[SatelliteState],
        *,
        battery_capacity_j: float,
        history_steps: int = BATTERY_CHART_WINDOW_STEPS,
        metrics_history_steps: int = LIVE_METRICS_WINDOW_STEPS,
    ) -> None:
        if history_steps <= 0:
            raise ValueError("history_steps must be positive")
        if metrics_history_steps <= 0:
            raise ValueError("metrics_history_steps must be positive")
        if tuple(state.sat_id for state in states) != tuple(
            satellite.sat_id for satellite in first_frame.satellites
        ):
            raise ValueError("satellite metadata does not match the first frame")

        self.battery_capacity_j = battery_capacity_j
        self._metadata_by_satellite = {
            state.sat_id: SatelliteMetadata(state.sat_id, state.plane)
            for state in states
        }
        if len(self._metadata_by_satellite) != len(states):
            raise ValueError("satellite ids must be unique")
        self.planes = tuple(
            sorted(
                {
                    metadata.plane
                    for metadata in self._metadata_by_satellite.values()
                }
            )
        )
        self._satellite_ids = tuple(state.sat_id for state in states)
        self._satellite_ids_by_plane = {
            plane: tuple(
                sat_id
                for sat_id in self._satellite_ids
                if self._metadata_by_satellite[sat_id].plane == plane
            )
            for plane in self.planes
        }
        self._history: deque[Frame] = deque(maxlen=history_steps)
        self._metrics_history: deque[LiveMetrics] = deque(
            maxlen=metrics_history_steps
        )
        self._history.append(first_frame)
        self._metrics_history.append(first_metrics)
        self._count = 1
        self._first = first_frame

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int) -> Frame:
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        if index == 0:
            return self._first

        history_start = self._count - len(self._history)
        if index < history_start:
            raise IndexError(f"frame {index} is outside the live history window")
        return self._history[index - history_start]

    def append(self, frame: Frame, metrics: LiveMetrics) -> None:
        actual_ids = tuple(satellite.sat_id for satellite in frame.satellites)
        if actual_ids != self._satellite_ids:
            raise ValueError("satellite ids changed during the simulation")
        if frame.time_s <= self._history[-1].time_s:
            raise ValueError("live frame times must be strictly increasing")
        if metrics.time_s != frame.time_s:
            raise ValueError("live metrics time must match the frame")
        self._history.append(frame)
        self._metrics_history.append(metrics)
        self._count += 1

    def metrics_window(
        self,
        end_index: int,
        *,
        window_steps: int = LIVE_METRICS_WINDOW_STEPS,
    ) -> tuple[LiveMetrics, ...]:
        if window_steps <= 0:
            raise ValueError("window_steps must be positive")
        if end_index != self._count - 1:
            raise IndexError("live metrics are only available for the latest frame")
        return tuple(self._metrics_history)[-window_steps:]

    def plane_for(self, sat_id: int) -> int:
        return self.metadata_for(sat_id).plane

    def metadata_for(self, sat_id: int) -> SatelliteMetadata:
        return self._metadata_by_satellite[sat_id]

    def satellites_in_plane(self, plane: int) -> tuple[int, ...]:
        return self._satellite_ids_by_plane[plane]


class OnlineSimulation:
    def __init__(
        self,
        config: SimulationConfig,
        *,
        control_log: OnlineControlLog | None = None,
        run_log: RunLog | None = None,
    ) -> None:
        self.config = config
        self.task_intensity = TaskIntensity(
            config.task.tasks_per_step,
            getattr(config, "task_intensity_schedule", ()),
        )
        self.control_log = control_log
        self.run_log = run_log
        self._run_log_finalized = False
        self.frames: OnlineFrames | None = None
        self._captured: tuple[Frame, LiveMetrics, list[SatelliteState]] | None = None
        self._generated = 0
        self._completed_tasks = 0
        self._failed = 0
        self._completed = False
        self._iterator = iter(
            iter_circular_states(
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
                scheduler=create_scheduler(config.scheduler_name),
                walker_phase=config.walker_phase,
                task_event_sink=self._capture_task_event,
                step_sink=self._capture,
                task_intensity=self.task_intensity,
            )
        )

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def tasks_per_step(self) -> int:
        return self.task_intensity.tasks_per_step

    def set_tasks_per_step(self, value: int) -> None:
        if self.frames is None:
            raise RuntimeError("online simulation has not started")
        if self._completed:
            raise RuntimeError("online simulation has completed")
        selected_at_time_s = int(self.frames[-1].time_s)
        effective_time_s = selected_at_time_s + self.config.step_s
        if effective_time_s > self.config.duration_s:
            raise RuntimeError("task intensity cannot change after the final step")
        self.task_intensity.schedule_change(effective_time_s, value)
        if self.control_log is not None:
            self.control_log.record(
                selected_at_time_s=selected_at_time_s,
                effective_time_s=effective_time_s,
                tasks_per_step=value,
            )

    def _capture(
        self,
        states: list[SatelliteState],
        context: SnapshotContext | None,
    ) -> None:
        if self.run_log is not None:
            self.run_log.write_step(states, context)
        self._captured = (
            frame_from_states(self.config.start, states, context),
            metrics_from_states(
                states,
                generated=self._generated,
                completed=self._completed_tasks,
                failed=self._failed,
            ),
            states,
        )

    def _capture_task_event(self, event: dict[str, object]) -> None:
        if self.run_log is not None:
            self.run_log.write_task_event(event)
        event_type = event.get("type")
        if event_type == "task_generated":
            self._generated += 1
        elif event_type == "task_completed":
            self._completed_tasks += 1
        elif event_type == "task_failed":
            self._failed += 1

    def start(self) -> OnlineFrames:
        if self.frames is not None:
            raise RuntimeError("online simulation has already started")
        self._advance_iterator()
        if self._captured is None:
            raise RuntimeError("simulation did not produce an initial frame")
        frame, metrics, states = self._captured
        self.frames = OnlineFrames(
            frame,
            metrics,
            states,
            battery_capacity_j=self.config.battery.capacity_j,
        )
        return self.frames

    def advance(self) -> bool:
        if self.frames is None:
            raise RuntimeError("online simulation has not started")
        if self._completed:
            return False
        if not self._advance_iterator():
            return False
        if self._captured is None:
            raise RuntimeError("simulation step did not produce a frame")
        frame, metrics, _states = self._captured
        self.frames.append(frame, metrics)
        return True

    def _advance_iterator(self) -> bool:
        self._captured = None
        try:
            next(self._iterator)
        except StopIteration:
            self._completed = True
            if self.run_log is not None:
                self.run_log.complete()
                self._run_log_finalized = True
            if self.control_log is not None:
                self.control_log.close()
            print("Simulation complete")
            return False
        return True

    def fail(self, exc: BaseException) -> None:
        if self.run_log is not None and not self._run_log_finalized:
            self.run_log.fail(exc)
            self._run_log_finalized = True
        if self.control_log is not None:
            self.control_log.close()


def _run_title(config: SimulationConfig) -> str:
    run = config.effective.get("run")
    if isinstance(run, dict) and isinstance(run.get("name"), str):
        return str(run["name"])
    return config.output_path.name or "Starlit simulation"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and visualize a Starlit simulation as it is produced",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="complete standalone JSON config file",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="maximum simulation and rendered steps per second (default: 10)",
    )
    parser.add_argument(
        "--satellite-radius-km",
        type=float,
        default=60.0,
        help="display radius of satellites in km (default: 60)",
    )
    parser.add_argument(
        "--task-capacity",
        type=int,
        default=8000,
        help="tasks per generation represented by 100%% load (default: 8000)",
    )
    args = parser.parse_args(argv)
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.satellite_radius_km <= 0:
        parser.error("--satellite-radius-km must be positive")
    if args.task_capacity <= 0:
        parser.error("--task-capacity must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = copy.deepcopy(config.effective)
    run_log = RunLog(output_dir, config.start, run_config)
    control_log = OnlineControlLog(
        output_dir,
        config.source,
        task_capacity=args.task_capacity,
    )
    simulation: OnlineSimulation | None = None
    try:
        simulation = OnlineSimulation(
            config,
            control_log=control_log,
            run_log=run_log,
        )
        frames = simulation.start()
        Visualizer(
            frames,
            title=_run_title(config),
            satellite_radius_km=args.satellite_radius_km,
            step_s=config.step_s,
        ).play(
            rate=args.fps,
            advance_live=simulation.advance,
            task_intensity=simulation,
            task_capacity=args.task_capacity,
            live_metrics=frames,
        )
    except BaseException as exc:
        if simulation is not None:
            simulation.fail(exc)
        else:
            run_log.fail(exc)
            control_log.close()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
