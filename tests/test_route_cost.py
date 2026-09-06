import unittest

from simulator.models import ComputeConfig, ISLConfig, Route, Task
from simulator.route_cost import estimate_route_cost, estimate_route_timing


def task() -> Task:
    return Task(
        task_id=1,
        created_time_s=0,
        source_sat=0,
        input_bits=100.0,
        output_bits=10.0,
        deadline_s=30.0,
    )


class RouteCostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = task()
        self.compute = ComputeConfig(
            cycles_per_input_bit=10.0,
            cpu_frequency_hz=100.0,
            cpu_power_w=50.0,
        )
        self.isl = ISLConfig(
            rate_bps=10.0,
            tx_power_w=2.0,
        )

    def test_local_route_matches_local_execution(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0,)),
            compute_config=self.compute,
            isl_config=self.isl,
        )

        self.assertEqual(cost.compute_time_s, 10.0)
        self.assertEqual(cost.transmission_time_s, 0.0)
        self.assertEqual(cost.energy_by_sat, {0: 500.0})
        self.assertEqual(cost.total_energy_j, 500.0)

    def test_one_hop_route_charges_transmit_power(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0, 1)),
            compute_config=self.compute,
            isl_config=self.isl,
        )

        self.assertEqual(cost.compute_time_s, 10.0)
        self.assertEqual(cost.transmission_time_s, 11.0)
        self.assertEqual(cost.energy_by_sat, {0: 20.0, 1: 502.0})
        self.assertEqual(cost.total_energy_j, 522.0)

    def test_route_timing_omits_energy_breakdown(self) -> None:
        timing = estimate_route_timing(
            task=self.task,
            route=Route((0, 2, 1)),
            compute_config=self.compute,
            isl_config=self.isl,
        )

        self.assertEqual(timing.compute_time_s, 10.0)
        self.assertEqual(timing.transmission_time_s, 22.0)
        self.assertEqual(timing.total_time_s, 32.0)

    def test_multi_hop_charges_relay_for_forward_and_return(self) -> None:
        cost = estimate_route_cost(
            task=self.task,
            route=Route((0, 2, 1)),
            compute_config=self.compute,
            isl_config=self.isl,
        )

        self.assertEqual(cost.transmission_time_s, 22.0)
        self.assertEqual(cost.energy_by_sat[0], 20.0)
        self.assertEqual(cost.energy_by_sat[2], 22.0)
        self.assertEqual(cost.energy_by_sat[1], 502.0)

if __name__ == "__main__":
    unittest.main()
