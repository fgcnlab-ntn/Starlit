from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .constants import EARTH_RADIUS_KM
from .models import ISLConfig, Route, SatelliteView


@dataclass(frozen=True)
class ISLGraph:
    adjacency: dict[int, tuple[int, ...]]

    def neighbors(self, sat_id: int) -> tuple[int, ...]:
        return self.adjacency.get(sat_id, ())


def _link_geometry_sq(
    a: SatelliteView,
    b: SatelliteView,
) -> tuple[float, float | None]:
    """Return squared separation and closest squared geocentric distance."""

    ax, ay, az = a.x_km, a.y_km, a.z_km
    dx = b.x_km - ax
    dy = b.y_km - ay
    dz = b.z_km - az
    separation_sq = dx * dx + dy * dy + dz * dz
    if separation_sq == 0.0:
        return separation_sq, None

    closest_t = -(ax * dx + ay * dy + az * dz) / separation_sq
    closest_t = max(0.0, min(1.0, closest_t))
    closest_x = ax + closest_t * dx
    closest_y = ay + closest_t * dy
    closest_z = az + closest_t * dz
    clearance_sq = (
        closest_x * closest_x
        + closest_y * closest_y
        + closest_z * closest_z
    )
    return separation_sq, clearance_sq


def has_line_of_sight(
    a: SatelliteView,
    b: SatelliteView,
    earth_radius_km: float = EARTH_RADIUS_KM,
) -> bool:
    """Return true when the segment between satellites clears Earth."""

    _, clearance_sq = _link_geometry_sq(a, b)
    return clearance_sq is None or clearance_sq > earth_radius_km * earth_radius_km


def grid_constellation_layout(
    satellites: Iterable[SatelliteView],
    walker_phase: int = 0,
) -> ISLGraph:
    """Build the fixed four-neighbor topology for a Walker constellation."""

    sat_list = sorted(satellites, key=lambda sat: sat.sat_id)
    if len(sat_list) != len({sat.sat_id for sat in sat_list}):
        raise ValueError("duplicate satellite id")
    by_position: dict[tuple[int, int], SatelliteView] = {}
    for sat in sat_list:
        if sat.plane is None or sat.plane < 0 or sat.slot is None or sat.slot < 0:
            raise ValueError(
                "grid ISL topology requires non-negative plane and slot metadata"
            )
        position = (sat.plane, sat.slot)
        if position in by_position:
            raise ValueError(f"duplicate satellite grid position: {position}")
        by_position[position] = sat

    if not sat_list:
        return ISLGraph({})

    plane_count = max(plane for plane, _ in by_position) + 1
    slots_per_plane = max(slot for _, slot in by_position) + 1
    expected_positions = {
        (plane, slot)
        for plane in range(plane_count)
        for slot in range(slots_per_plane)
    }
    if set(by_position) != expected_positions:
        raise ValueError("grid ISL topology requires a complete rectangular layout")

    adjacency: dict[int, set[int]] = {sat.sat_id: set() for sat in sat_list}

    def add_link(
        first_position: tuple[int, int],
        second_position: tuple[int, int],
    ) -> None:
        first = by_position[first_position].sat_id
        second = by_position[second_position].sat_id
        if first == second:
            return
        adjacency[first].add(second)
        adjacency[second].add(first)

    for plane in range(plane_count):
        for slot in range(slots_per_plane):
            add_link((plane, slot), (plane, (slot + 1) % slots_per_plane))

    for plane in range(plane_count - 1):
        for slot in range(slots_per_plane):
            add_link((plane, slot), (plane + 1, slot))

    if plane_count > 1:
        seam_offset = walker_phase % slots_per_plane
        for slot in range(slots_per_plane):
            add_link(
                (plane_count - 1, slot),
                (0, (slot + seam_offset) % slots_per_plane),
            )

    return ISLGraph(
        {sat_id: tuple(sorted(neighbors)) for sat_id, neighbors in adjacency.items()}
    )


def build_isl_graph(
    satellites: Iterable[SatelliteView],
    config: ISLConfig,
    *,
    candidate_graph: ISLGraph | None = None,
    walker_phase: int = 0,
) -> ISLGraph:
    sat_list = tuple(satellites)
    if candidate_graph is None:
        candidate_graph = grid_constellation_layout(sat_list, walker_phase)
    if config.max_range_km <= 0.0:
        raise ValueError("ISL requires positive max_range_km")

    by_id = {sat.sat_id: sat for sat in sat_list}
    if len(by_id) != len(sat_list):
        raise ValueError("duplicate satellite id")
    if set(by_id) != set(candidate_graph.adjacency):
        raise ValueError("satellites do not match ISL candidate graph")

    max_range_sq = config.max_range_km * config.max_range_km
    earth_radius_sq = EARTH_RADIUS_KM * EARTH_RADIUS_KM
    adjacency: dict[int, list[int]] = {sat_id: [] for sat_id in by_id}
    for first_id, neighbors in candidate_graph.adjacency.items():
        first = by_id[first_id]
        for second_id in neighbors:
            if second_id <= first_id:
                continue
            separation_sq, clearance_sq = _link_geometry_sq(
                first,
                by_id[second_id],
            )
            if separation_sq > max_range_sq:
                continue
            if clearance_sq is not None and clearance_sq <= earth_radius_sq:
                continue
            adjacency[first_id].append(second_id)
            adjacency[second_id].append(first_id)

    return ISLGraph(
        {sat_id: tuple(sorted(neighbors)) for sat_id, neighbors in adjacency.items()}
    )


def shortest_route(graph: ISLGraph, source_sat: int, target_sat: int) -> Route | None:
    if source_sat == target_sat:
        return Route((source_sat,))
    if source_sat not in graph.adjacency or target_sat not in graph.adjacency:
        return None

    parents: dict[int, int | None] = {source_sat: None}
    queue: deque[int] = deque([source_sat])

    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            if neighbor == target_sat:
                return route_from_parents(parents, target_sat)
            queue.append(neighbor)
    return None


def build_route_tree(
    graph: ISLGraph,
    source_sat: int,
    blocked_relays: set[int] | None = None,
) -> dict[int, int | None]:
    """Return a shortest-route tree rooted at one source."""

    if source_sat not in graph.adjacency:
        return {}

    blocked = blocked_relays if blocked_relays is not None else ()
    parents: dict[int, int | None] = {source_sat: None}
    queue: deque[int] = deque([source_sat])
    while queue:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in parents or neighbor in blocked:
                continue
            parents[neighbor] = current
            queue.append(neighbor)
    return parents


def route_nodes_reversed(
    parents: dict[int, int | None],
    target_sat: int,
) -> list[int] | None:
    if target_sat not in parents:
        return None

    nodes = [target_sat]
    current = target_sat
    while parents[current] is not None:
        current = parents[current]
        nodes.append(current)
    return nodes


def route_from_parents(
    parents: dict[int, int | None],
    target_sat: int,
) -> Route | None:
    nodes = route_nodes_reversed(parents, target_sat)
    if nodes is None:
        return None
    nodes.reverse()
    return Route(tuple(nodes))


def routes_from_source(graph: ISLGraph, source_sat: int) -> dict[int, Route]:
    """Return shortest routes from one source to every reachable satellite."""

    parents = build_route_tree(graph, source_sat)
    routes: dict[int, Route] = {}
    for target_sat in parents:
        route = route_from_parents(parents, target_sat)
        assert route is not None
        routes[target_sat] = route
    return routes


def routes_to_targets(
    graph: ISLGraph,
    source_sat: int,
    target_sats: set[int],
) -> dict[int, Route]:
    """Return shortest routes from one source to requested reachable targets."""

    if not target_sats or source_sat not in graph.adjacency:
        return {}

    remaining = set(target_sats)
    parents: dict[int, int | None] = {source_sat: None}
    queue: deque[int] = deque([source_sat])
    remaining.discard(source_sat)

    while queue and remaining:
        current = queue.popleft()
        for neighbor in graph.neighbors(current):
            if neighbor in parents:
                continue
            parents[neighbor] = current
            remaining.discard(neighbor)
            if not remaining:
                break
            queue.append(neighbor)

    routes: dict[int, Route] = {}
    for target_sat in target_sats - remaining:
        route = route_from_parents(parents, target_sat)
        assert route is not None
        routes[target_sat] = route
    return routes
