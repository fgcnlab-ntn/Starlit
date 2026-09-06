import unittest

from simulator.battery import ENERGY_EPSILON_J, battery_is_safe, battery_step
from simulator.models import (
    Assignment,
    BatteryConfig,
    ComputeConfig,
    ISLConfig,
    Route,
    Task,
)
from simulator.orbit import apply_step
from simulator.runtime import EnvironmentRuntime, SatelliteRuntime



def compute_config() -> ComputeConfig:
    return ComputeConfig(
        cycles_per_input_bit=1.0,
        cpu_frequency_hz=1.0,
        cpu_power_w=1.0,
    )


class BatteryDoDTests(unittest.TestCase):
    def test_battery_step_clamps_to_physical_bounds(self) -> None:
        battery = BatteryConfig(
            capacity_j=100.0,
            initial_j=50.0,
            min_safe_j=20.0,
            harvest_w=10.0,
            idle_w=1.0,
        )

        depleted, _, _ = battery_step(
            battery_now=5.0,
            sunlit=False,
            step_s=10,
            battery=battery,
            task_energy_j=1.0,
            update=True,
        )
        charged, _, _ = battery_step(
            battery_now=95.0,
            sunlit=True,
            step_s=10,
            battery=battery,
            task_energy_j=0.0,
            update=True,
        )

        self.assertEqual(depleted, 0.0)
        self.assertEqual(charged, 100.0)

    def test_safe_limit_ignores_only_floating_point_noise(self) -> None:
        minimum_j = 151200.0

        self.assertTrue(battery_is_safe(minimum_j, minimum_j))
        self.assertTrue(battery_is_safe(107999.99999999894, 108000.0))
        self.assertTrue(
            battery_is_safe(minimum_j - ENERGY_EPSILON_J / 2.0, minimum_j)
        )
        self.assertFalse(
            battery_is_safe(minimum_j - ENERGY_EPSILON_J * 2.0, minimum_j)
        )

    def test_apply_step_executes_task_that_crosses_dod_limit(self) -> None:
        battery = BatteryConfig(
            capacity_j=100.0,
            initial_j=25.0,
            min_safe_j=20.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            input_bits=10.0,
            output_bits=0.0,
            deadline_s=30.0,
        )
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=25.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )

        states = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[
                Assignment(task_id=1, route=Route((0,)), mode="local")
            ],
        )

        self.assertEqual(env.completed_tasks, [1])
        self.assertEqual(states[0].battery_j, 15.0)
        self.assertFalse(states[0].safe_battery)

    def test_apply_step_keeps_oversized_task_running(self) -> None:
        battery = BatteryConfig(
            capacity_j=1000.0,
            initial_j=1000.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            input_bits=100.0,
            output_bits=0.0,
            deadline_s=1000.0,
        )
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=1000.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )

        states = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[Assignment(task_id=1, route=Route((0,)), mode="local")],
        )

        self.assertEqual(env.completed_tasks, [])
        self.assertEqual(len(env.running_tasks), 1)
        self.assertEqual(states[0].task_energy_j, 30.0)
        self.assertEqual(states[0].battery_j, 970.0)

    def test_apply_step_completes_running_task_later(self) -> None:
        battery = BatteryConfig(
            capacity_j=1000.0,
            initial_j=1000.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            input_bits=40.0,
            output_bits=0.0,
            deadline_s=120.0,
        )
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=1000.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )
        events = []
        env.task_event_sink = events.append

        apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[Assignment(task_id=1, route=Route((0,)), mode="local")],
        )
        env.time_s = 60
        states = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[],
            assignments=[],
        )

        self.assertEqual(len(env.running_tasks), 0)
        completed = [event for event in events if event["type"] == "task_completed"]
        self.assertEqual(completed[0]["executed_compute_time_s"], 40.0)
        self.assertEqual(completed[0]["total_time_s"], 40.0)
        self.assertEqual(states[0].task_energy_j, 10.0)
        self.assertEqual(states[0].battery_j, 960.0)

    def test_satellite_queue_runs_head_before_next_task(self) -> None:
        battery = BatteryConfig(
            capacity_j=1000.0,
            initial_j=1000.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        tasks = [
            Task(
                task_id=1,
                created_time_s=30,
                source_sat=0,
                input_bits=40.0,
                output_bits=0.0,
                deadline_s=1000.0,
            ),
            Task(
                task_id=2,
                created_time_s=30,
                source_sat=0,
                input_bits=10.0,
                output_bits=0.0,
                deadline_s=1000.0,
            ),
        ]
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=1000.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )
        events = []
        env.task_event_sink = events.append

        states = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=tasks,
            assignments=[
                Assignment(task_id=1, route=Route((0,)), mode="local"),
                Assignment(task_id=2, route=Route((0,)), mode="local"),
            ],
        )

        self.assertEqual(env.completed_tasks, [])
        self.assertEqual([task.task.task_id for task in env.satellites[0].task_queue], [1, 2])
        self.assertEqual(states[0].task_energy_j, 30.0)

        env.time_s = 60
        states = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[],
            assignments=[],
        )

        completed = [event for event in events if event["type"] == "task_completed"]
        self.assertEqual([event["task_id"] for event in completed], [1, 2])
        self.assertEqual([event["total_time_s"] for event in completed], [40.0, 50.0])
        self.assertEqual(len(env.running_tasks), 0)
        self.assertEqual(states[0].task_energy_j, 20.0)

    def test_apply_step_fails_running_task_that_finishes_after_deadline(self) -> None:
        battery = BatteryConfig(
            capacity_j=1000.0,
            initial_j=1000.0,
            min_safe_j=0.0,
            harvest_w=0.0,
            idle_w=0.0,
        )
        task = Task(
            task_id=1,
            created_time_s=30,
            source_sat=0,
            input_bits=40.0,
            output_bits=0.0,
            deadline_s=35.0,
        )
        env = EnvironmentRuntime(
            satellites=[
                SatelliteRuntime(
                    sat_id=0,
                    name="sat_0",
                    plane=0,
                    slot=0,
                    battery_j=1000.0,
                    sunlit=False,
                )
            ],
            time_s=30,
        )
        events = []
        env.task_event_sink = events.append

        apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[task],
            assignments=[Assignment(task_id=1, route=Route((0,)), mode="local")],
        )
        env.time_s = 60
        states = apply_step(
            env=env,
            step_s=30,
            battery=battery,
            compute_config=compute_config(),
            isl_config=ISLConfig(1.0, 0.0),
            tasks=[],
            assignments=[],
        )

        self.assertEqual(len(env.running_tasks), 0)
        failed = [event for event in events if event["type"] == "task_failed"]
        self.assertEqual(failed[0]["reason"], "deadline")
        self.assertEqual(failed[0]["executed_compute_time_s"], 40.0)
        self.assertEqual(states[0].failed_tasks, 1)

if __name__ == "__main__":
    unittest.main()
