"""planning/astar.py — Space-Time A* low-level path planner."""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from config import Config
from env.grid_utils import precompute_corner_cells


class SpaceTimeAStar:
    """Space-Time A* planner with kinodynamic corner penalties and constraint support."""

    def __init__(self, grid: np.ndarray, config: Config) -> None:
        """Store grid and config. Precompute corner cells for kinodynamic weighting."""
        self.grid = grid
        self.config = config
        self.H, self.W = grid.shape
        self.corner_cells: Set[Tuple[int, int]] = precompute_corner_cells(grid)

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        constraints: List[Tuple[int, int, int]],  # (row, col, time) forbidden nodes
        start_time: int = 0,
    ) -> Optional[List[Tuple[int, int, int]]]:
        """
        Space-Time A* search returning path as [(row, col, time), ...] or None.

        Movements: up, down, left, right, wait-in-place (5 actions).
        Constraints: list of (row, col, t) tuples that must be avoided.
        """
        constraint_set: Set[Tuple[int, int, int]] = set(constraints)

        # open_heap entries: (f_cost, -g_cost, (row, col, time))
        # Tie-break by -g_cost: prefer higher g (longer path = reached goal faster)
        start_node = (start[0], start[1], start_time)
        h0 = self._heuristic(start, goal)
        open_heap: List[Tuple[float, float, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (h0, 0.0, start_node))

        came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start_node: None}
        g_score: Dict[Tuple[int, int, int], float] = {start_node: 0.0}

        while open_heap:
            f, neg_g, node = heapq.heappop(open_heap)
            r, c, t = node

            # Goal check (position only, any time)
            if (r, c) == goal:
                return self._reconstruct_path(came_from, node)

            if t >= self.config.time_horizon:
                continue

            for nr, nc, nt in self._get_neighbors(r, c, t):
                # Vertex constraint check
                if (nr, nc, nt) in constraint_set:
                    continue
                # Edge conflict (swap) check
                if (nc, nr, nt) in constraint_set and (nr == r and nc == c):
                    continue
                # Detect swap: robot at (nr,nc) moving to (r,c) at same time
                if (r, c, nt) in constraint_set and (nr, nc, t) in constraint_set:
                    continue

                cell_cost = self._get_cell_cost(nr, nc)
                tentative_g = g_score[node] + cell_cost

                if (nr, nc, nt) not in g_score or tentative_g < g_score[(nr, nc, nt)]:
                    g_score[(nr, nc, nt)] = tentative_g
                    h = self._heuristic((nr, nc), goal)
                    f_new = tentative_g + h
                    heapq.heappush(open_heap, (f_new, -tentative_g, (nr, nc, nt)))
                    came_from[(nr, nc, nt)] = node

        return None  # No path found

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible)."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _get_cell_cost(self, row: int, col: int) -> float:
        """Return 1.0 + corner_penalty_lambda if cell is a corner cell, else 1.0."""
        if (row, col) in self.corner_cells:
            return 1.0 + self.config.corner_penalty_lambda
        return 1.0

    def _get_neighbors(
        self, row: int, col: int, t: int
    ) -> List[Tuple[int, int, int]]:
        """Return valid (row, col, t+1) neighbors including wait-in-place."""
        neighbors: List[Tuple[int, int, int]] = []
        nt = t + 1
        # 4-directional movement + wait
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.H and 0 <= nc < self.W and self.grid[nr, nc] != 1:
                neighbors.append((nr, nc, nt))
        return neighbors

    def _reconstruct_path(
        self,
        came_from: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]],
        current: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int]]:
        """Trace back the path from goal to start and return it in forward order."""
        path: List[Tuple[int, int, int]] = []
        node: Optional[Tuple[int, int, int]] = current
        while node is not None:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path
