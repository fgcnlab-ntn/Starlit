import unittest

from simulator.models import Assignment, Route


class AssignmentRouteTests(unittest.TestCase):
    def test_route_exposes_endpoints(self) -> None:
        route = Route((2, 5, 9))
        assignment = Assignment(task_id=7, route=route, mode="offload")

        self.assertEqual(assignment.source_sat, 2)
        self.assertEqual(assignment.target_sat, 9)
        self.assertEqual(assignment.hop_count, 2)
        self.assertIs(assignment.route, route)

    def test_rejects_empty_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "route must contain"):
            Route(())

    def test_rejects_invalid_route_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "route nodes"):
            Route((-1,))


if __name__ == "__main__":
    unittest.main()
