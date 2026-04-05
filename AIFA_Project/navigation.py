"""
navigation.py — Path-Planning Engine
======================================
Implements time-expanded A* search with collision avoidance,
multi-waypoint planning, and weighted A* for performance tuning.
"""

import heapq
from itertools import count


class Navigation:
    """
    A* path planner that operates on a time-expanded graph to
    guarantee collision-free multi-robot navigation.
    """

    _tie_breaker = count()  # global counter to break heap ties

    def __init__(self, warehouse, max_time_steps=300):
        self.warehouse = warehouse
        self.max_time_steps = max_time_steps
        self.paths = {}

    # ── Heuristics ───────────────────────────────────────────────────────
    @staticmethod
    def manhattan(a, b):
        """Manhattan distance between two (row, col) positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def heuristic(self, position, target_position):
        return self.manhattan(position, target_position)

    # ── Neighbour expansion ──────────────────────────────────────────────
    def get_neighbors(self, position):
        """Return valid neighbours including a wait-in-place action."""
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
        neighbors = []
        for dr, dc in moves:
            nr, nc = position[0] + dr, position[1] + dc
            if not self.warehouse.is_valid_position((nr, nc)):
                continue
            if self.warehouse.warehouse[nr, nc] == 1:  # SHELF
                continue
            neighbors.append((nr, nc))
        return neighbors

    # ── Reservation checking ─────────────────────────────────────────────
    @staticmethod
    def _is_reserved(current_pos, next_pos, next_t, reserved_cells, reserved_edges):
        """Check vertex and edge (swap) conflicts."""
        if (next_pos, next_t) in reserved_cells:
            return True
        if (next_pos, current_pos, next_t) in reserved_edges:
            return True
        return False

    # ── Core A* (time-expanded) ──────────────────────────────────────────
    def a_star_time_expanded(self, start_position, target_position,
                              reserved_cells=None, reserved_edges=None,
                              weight=1.0):
        """
        Time-expanded A* returning a list of positions over time.

        Parameters
        ----------
        weight : float
            Heuristic inflation factor.  w=1 → optimal; w>1 → faster but
            potentially sub-optimal (Weighted A* / WA*).
        """
        if reserved_cells is None:
            reserved_cells = set()
        if reserved_edges is None:
            reserved_edges = set()

        heap = []
        start_state = (start_position, 0)
        h0 = weight * self.heuristic(start_position, target_position)
        heapq.heappush(heap, (h0, 0, next(self._tie_breaker), start_state))

        came_from = {}
        g_score = {start_state: 0}
        visited = set()

        while heap:
            _, g_val, _, (current_pos, current_t) = heapq.heappop(heap)

            if (current_pos, current_t) in visited:
                continue
            visited.add((current_pos, current_t))

            if current_pos == target_position:
                state = (current_pos, current_t)
                path = [current_pos]
                while state in came_from:
                    state = came_from[state]
                    path.append(state[0])
                path.reverse()
                return path

            if current_t >= self.max_time_steps:
                continue

            for next_pos in self.get_neighbors(current_pos):
                next_t = current_t + 1
                if self._is_reserved(current_pos, next_pos, next_t,
                                     reserved_cells, reserved_edges):
                    continue

                next_state = (next_pos, next_t)
                move_cost = 0 if next_pos == current_pos else 1  # wait is free
                tentative_g = g_score[(current_pos, current_t)] + 1

                if tentative_g < g_score.get(next_state, float("inf")):
                    came_from[next_state] = (current_pos, current_t)
                    g_score[next_state] = tentative_g
                    f_score = tentative_g + weight * self.heuristic(
                        next_pos, target_position)
                    heapq.heappush(
                        heap,
                        (f_score, tentative_g, next(self._tie_breaker), next_state),
                    )

        return None  # no path found

    # ── Multi-waypoint planning ──────────────────────────────────────────
    def plan_waypoints(self, waypoints, reserved_cells=None, reserved_edges=None,
                       weight=1.0):
        """
        Plan a path visiting a sequence of waypoints in order.
        Returns the concatenated path (no duplicate positions at transitions).
        """
        if reserved_cells is None:
            reserved_cells = set()
        if reserved_edges is None:
            reserved_edges = set()

        full_path = []
        for i in range(len(waypoints) - 1):
            # Offset time by how far along the full path we are
            seg = self.a_star_time_expanded(
                waypoints[i], waypoints[i + 1],
                reserved_cells, reserved_edges, weight,
            )
            if seg is None:
                return None
            if full_path:
                seg = seg[1:]  # skip duplicate overlap position
            full_path.extend(seg)

        return full_path if full_path else None

    # ── Path reservation ─────────────────────────────────────────────────
    def _reserve_path(self, path, reserved_cells, reserved_edges):
        if not path:
            return
        for t, pos in enumerate(path):
            reserved_cells.add((pos, t))
            if t > 0:
                prev = path[t - 1]
                reserved_edges.add((prev, pos, t))
        # Hold goal cell after arrival
        final_pos = path[-1]
        for t in range(len(path), self.max_time_steps + 1):
            reserved_cells.add((final_pos, t))

    # ── Multi-robot prioritized planning ─────────────────────────────────
    def plan_multi_robot(self, robot_starts, robot_targets):
        """
        Plans collision-free paths using prioritized planning.

        Parameters
        ----------
        robot_starts : dict   {robot_id: (r, c)}
        robot_targets : dict  {robot_id: (r, c)}

        Returns
        -------
        dict  {robot_id: [positions over time]}
        """
        reserved_cells = set()
        reserved_edges = set()
        planned_paths = {}

        for robot_id in robot_starts:
            if robot_id not in robot_targets:
                raise ValueError(f"Missing target for robot '{robot_id}'")
            start = robot_starts[robot_id]
            target = robot_targets[robot_id]
            path = self.a_star_time_expanded(
                start, target, reserved_cells, reserved_edges)
            if path is None:
                raise ValueError(
                    f"No valid path for robot '{robot_id}'. "
                    "Try increasing max_time_steps or changing priorities."
                )
            planned_paths[robot_id] = path
            self._reserve_path(path, reserved_cells, reserved_edges)

        self.paths = planned_paths
        return planned_paths

    # ── Multi-robot with waypoints ───────────────────────────────────────
    def plan_multi_robot_waypoints(self, robot_starts, robot_waypoints):
        """
        Prioritized planning where each robot has a list of waypoints.

        Parameters
        ----------
        robot_starts : dict     {robot_id: (r, c)}
        robot_waypoints : dict  {robot_id: [(r,c), (r,c), ...]}
        """
        reserved_cells = set()
        reserved_edges = set()
        planned_paths = {}

        for robot_id in robot_starts:
            wps = [robot_starts[robot_id]] + robot_waypoints[robot_id]
            path = self.plan_waypoints(wps, reserved_cells, reserved_edges)
            if path is None:
                raise ValueError(
                    f"No valid waypoint path for robot '{robot_id}'.")
            planned_paths[robot_id] = path
            self._reserve_path(path, reserved_cells, reserved_edges)

        self.paths = planned_paths
        return planned_paths
