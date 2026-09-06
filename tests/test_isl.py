import unittest

from simulator.constants import EARTH_RADIUS_KM
from simulator.isl import (
    ISLGraph,
    build_isl_graph,
    grid_constellation_layout,
    has_line_of_sight,
    shortest_route,
)
from simulator.models import ISLConfig, SatelliteView, Task
from simulator.scheduler import LocalOnlyScheduler, NearestSunlitScheduler


def view(
    sat_id: int,
    *,
    sunlit: bool = False,
    x: float = 0.0,
    y: float = 0.0,
    plane: int | None = None,
    slot: int | None = None,
) -> SatelliteView:
    return SatelliteView(
        sat_id=sat_id,
        x_km=x,
        y_km=y,
        z_km=0.0,
        sunlit=sunlit,
        plane=plane,
        slot=slot,
    )


def point_view(
    sat_id: int,
    *,
    x: float,
    y: float = 0.0,
    z: float = 0.0,
) -> SatelliteView:
    return SatelliteView(
        sat_id=sat_id,
        x_km=x,
        y_km=y,
        z_km=z,
        sunlit=False,
    )


def isl_config(max_range_km: float) -> ISLConfig:
    return ISLConfig(1.0, 0.0, max_range_km=max_range_km)


class ISLGraphTests(unittest.TestCase):
    def test_grid_graph_connects_only_four_topological_neighbors(self) -> None:
        satellites = [
            view(
                plane * 3 + slot,
                x=10000.0 + 10.0 * plane,
                y=10.0 * slot,
                plane=plane,
                slot=slot,
            )
            for plane in range(3)
            for slot in range(3)
        ]

        graph = build_isl_graph(satellites, isl_config(100.0))

        self.assertEqual(graph.neighbors(0), (1, 2, 3, 6))
        self.assertEqual(graph.neighbors(4), (1, 3, 5, 7))
        self.assertNotIn(8, graph.neighbors(4))

    def test_grid_diagonal_requires_two_hops(self) -> None:
        satellites = [
            view(0, x=10000.0, y=0.0, plane=0, slot=0),
            view(1, x=10000.0, y=1.0, plane=0, slot=1),
            view(2, x=10001.0, y=0.0, plane=1, slot=0),
            view(3, x=10001.0, y=1.0, plane=1, slot=1),
        ]

        route = shortest_route(build_isl_graph(satellites, isl_config(10.0)), 0, 3)

        self.assertIsNotNone(route)
        self.assertEqual(route.nodes, (0, 1, 3))

    def test_walker_phase_offsets_cross_plane_seam(self) -> None:
        satellites = [
            view(
                plane * 4 + slot,
                plane=plane,
                slot=slot,
            )
            for plane in range(3)
            for slot in range(4)
        ]

        layout = grid_constellation_layout(satellites, walker_phase=1)

        self.assertEqual(layout.neighbors(8), (1, 4, 9, 11))
        self.assertNotIn(0, layout.neighbors(8))

    def test_static_layout_is_reused_when_link_availability_changes(self) -> None:
        near = [
            view(0, x=10000.0, plane=0, slot=0),
            view(1, x=10001.0, plane=0, slot=1),
        ]
        far = [
            view(0, x=10000.0, plane=0, slot=0),
            view(1, x=10010.0, plane=0, slot=1),
        ]
        layout = grid_constellation_layout(near)

        near_graph = build_isl_graph(
            near,
            isl_config(5.0),
            candidate_graph=layout,
        )
        far_graph = build_isl_graph(
            far,
            isl_config(5.0),
            candidate_graph=layout,
        )

        self.assertEqual(near_graph.neighbors(0), (1,))
        self.assertEqual(far_graph.neighbors(0), ())

    def test_grid_graph_filters_neighbors_by_range(self) -> None:
        graph = build_isl_graph(
            [
                view(0, x=10000.0, plane=0, slot=0),
                view(1, x=10003.0, plane=0, slot=1),
                view(2, x=10010.0, plane=0, slot=2),
            ],
            isl_config(5.0),
        )

        self.assertEqual(graph.neighbors(0), (1,))
        self.assertEqual(graph.neighbors(1), (0,))
        self.assertEqual(graph.neighbors(2), ())

    def test_grid_graph_includes_link_at_exact_maximum_range(self) -> None:
        satellites = [
            view(0, x=10000.0, plane=0, slot=0),
            view(1, x=10003.0, y=4.0, plane=0, slot=1),
        ]

        graph = build_isl_graph(satellites, isl_config(5.0))

        self.assertEqual(graph.neighbors(0), (1,))

    def test_line_of_sight_blocks_links_through_earth(self) -> None:
        first = point_view(0, x=7000.0)
        second = point_view(1, x=-7000.0)

        self.assertFalse(has_line_of_sight(first, second))

    def test_line_of_sight_allows_links_above_earth_limb(self) -> None:
        first = point_view(0, x=10000.0, y=0.0)
        second = point_view(1, x=0.0, y=10000.0)

        self.assertTrue(has_line_of_sight(first, second))

    def test_line_of_sight_rejects_link_tangent_to_earth(self) -> None:
        first = point_view(0, x=-1000.0, y=EARTH_RADIUS_KM)
        second = point_view(1, x=1000.0, y=EARTH_RADIUS_KM)

        self.assertFalse(has_line_of_sight(first, second))

    def test_grid_graph_rejects_earth_blocked_links(self) -> None:
        graph = build_isl_graph(
            [
                SatelliteView(
                    sat_id=0,
                    x_km=7000.0,
                    y_km=0.0,
                    z_km=0.0,
                    sunlit=False,
                    plane=0,
                    slot=0,
                ),
                SatelliteView(
                    sat_id=1,
                    x_km=-7000.0,
                    y_km=0.0,
                    z_km=0.0,
                    sunlit=False,
                    plane=0,
                    slot=1,
                ),
            ],
            isl_config(20000.0),
        )

        self.assertEqual(graph.neighbors(0), ())
        self.assertEqual(graph.neighbors(1), ())

    def test_build_isl_graph_filters_grid_by_range(self) -> None:
        graph = build_isl_graph(
            [
                view(0, x=10000.0, plane=0, slot=0),
                view(1, x=10003.0, plane=0, slot=1),
                view(2, x=10010.0, plane=0, slot=2),
            ],
            ISLConfig(1.0, 0.0, max_range_km=5.0),
        )

        self.assertEqual(graph.neighbors(0), (1,))
        self.assertEqual(graph.neighbors(2), ())

    def test_grid_topology_requires_positive_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive max_range_km"):
            build_isl_graph(
                [view(0, plane=0, slot=0)],
                ISLConfig(
                    1.0,
                    0.0,
                    max_range_km=0.0,
                ),
            )

    def test_grid_topology_requires_plane_and_slot_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "plane and slot metadata"):
            build_isl_graph([view(0)], isl_config(5.0))

    def test_grid_topology_rejects_incomplete_layout(self) -> None:
        with self.assertRaisesRegex(ValueError, "complete rectangular layout"):
            build_isl_graph(
                [
                    view(0, plane=0, slot=0),
                    view(1, plane=1, slot=1),
                ],
                isl_config(5.0),
            )

    def test_shortest_route_returns_direct_route_when_available(self) -> None:
        graph = ISLGraph({0: (1, 2), 1: (0,), 2: (0,)})

        route = shortest_route(graph, 0, 2)

        self.assertIsNotNone(route)
        self.assertEqual(route.nodes, (0, 2))

    def test_shortest_route_returns_multi_hop_route(self) -> None:
        graph = ISLGraph({0: (1,), 1: (0, 2), 2: (1,)})

        route = shortest_route(graph, 0, 2)

        self.assertIsNotNone(route)
        self.assertEqual(route.nodes, (0, 1, 2))

    def test_shortest_route_returns_none_when_disconnected(self) -> None:
        graph = ISLGraph({0: (), 1: ()})

        self.assertIsNone(shortest_route(graph, 0, 1))

    def test_nearest_sunlit_scheduler_uses_route_finder(self) -> None:
        task = Task(
            task_id=1,
            created_time_s=0,
            source_sat=0,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=30.0,
        )
        views = [
            view(0, sunlit=False, x=0.0),
            view(1, sunlit=False, x=1.0),
            view(2, sunlit=True, x=2.0),
        ]
        graph = ISLGraph({0: (1,), 1: (0, 2), 2: (1,)})

        assignment = NearestSunlitScheduler().assign_task(
            task=task,
            satellite_views=views,
            isl_graph=graph,
        )

        self.assertEqual(assignment.route.nodes, (0, 1, 2))
        self.assertEqual(assignment.mode, "offload")

    def test_nearest_sunlit_scheduler_prefers_shorter_grid_route(self) -> None:
        task = Task(
            task_id=1,
            created_time_s=0,
            source_sat=0,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=30.0,
        )
        views = [
            view(0, sunlit=False, x=0.0),
            view(1, sunlit=True, x=100.0),
            view(2, sunlit=True, x=1.0),
            view(3, sunlit=False, x=0.5),
        ]
        graph = ISLGraph({0: (1, 3), 1: (0,), 2: (3,), 3: (0, 2)})

        assignment = NearestSunlitScheduler().assign_task(
            task=task,
            satellite_views=views,
            isl_graph=graph,
        )

        self.assertEqual(assignment.route.nodes, (0, 1))
        self.assertEqual(assignment.mode, "offload")

    def test_nearest_sunlit_scheduler_preserves_order_for_equal_grid_route(
        self,
    ) -> None:
        task = Task(
            task_id=1,
            created_time_s=0,
            source_sat=0,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=30.0,
        )
        views = [
            view(0, sunlit=False, x=0.0),
            view(2, sunlit=True, x=100.0),
            view(1, sunlit=True, x=1.0),
        ]
        graph = ISLGraph({0: (1, 2), 1: (0,), 2: (0,)})

        assignment = NearestSunlitScheduler().assign_task(
            task=task,
            satellite_views=views,
            isl_graph=graph,
        )

        self.assertEqual(assignment.route.nodes, (0, 2))
        self.assertEqual(assignment.mode, "offload")

    def test_nearest_sunlit_batch_reuses_source_decision(self) -> None:
        tasks = [
            Task(
                task_id=task_id,
                created_time_s=0,
                source_sat=0,
                input_bits=1.0,
                output_bits=1.0,
                deadline_s=30.0,
            )
            for task_id in (1, 2)
        ]
        views = [
            view(0, sunlit=False, x=0.0),
            view(1, sunlit=True, x=1.0),
        ]
        graph = ISLGraph({0: (1,), 1: (0,)})

        assignments = NearestSunlitScheduler().assign_tasks(
            tasks=tasks,
            satellite_views=views,
            time_s=0,
            step_s=1,
            battery=None,
            compute_config=None,
            isl_config=None,
            isl_graph=graph,
        )

        self.assertEqual([assignment.task_id for assignment in assignments], [1, 2])
        self.assertEqual(
            [assignment.route.nodes for assignment in assignments],
            [(0, 1), (0, 1)],
        )
        self.assertEqual(
            [assignment.mode for assignment in assignments],
            ["offload", "offload"],
        )

    def test_nearest_sunlit_scheduler_ignores_unreachable_sunlit_target(self) -> None:
        task = Task(
            task_id=1,
            created_time_s=0,
            source_sat=0,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=30.0,
        )
        views = [
            view(0, sunlit=False, x=0.0),
            view(1, sunlit=True, x=1.0),
        ]
        graph = ISLGraph({0: (), 1: ()})

        assignment = NearestSunlitScheduler().assign_task(
            task=task,
            satellite_views=views,
            isl_graph=graph,
        )

        self.assertEqual(assignment.route.nodes, (0,))
        self.assertEqual(assignment.mode, "local")

    def test_local_only_scheduler_returns_single_node_route(self) -> None:
        task = Task(
            task_id=1,
            created_time_s=0,
            source_sat=0,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=30.0,
        )

        assignment = LocalOnlyScheduler().assign_task(
            task=task,
            satellite_views=[view(0)],
            isl_graph=ISLGraph({0: ()}),
        )

        self.assertEqual(assignment.route.nodes, (0,))
        self.assertEqual(assignment.mode, "local")

    def test_local_only_scheduler_rejects_unknown_source_satellite(self) -> None:
        task = Task(
            task_id=1,
            created_time_s=0,
            source_sat=9,
            input_bits=1.0,
            output_bits=1.0,
            deadline_s=30.0,
        )

        with self.assertRaisesRegex(ValueError, "not present in the ISL graph"):
            LocalOnlyScheduler().assign_task(
                task=task,
                satellite_views=[view(0)],
                isl_graph=ISLGraph({0: ()}),
            )


if __name__ == "__main__":
    unittest.main()
