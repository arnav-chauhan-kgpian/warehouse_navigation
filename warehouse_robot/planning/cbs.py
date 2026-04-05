"""planning/cbs.py — Conflict-Based Search high-level multi-agent path coordinator."""

from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import Config
from planning.astar import SpaceTimeAStar


@dataclass
class CBSNode:
    """A node in the CBS high-level search tree."""

    constraints: Dict[int, List[Tuple[int, int, int]]]  # robot_id -> [(r, c, t)]
    paths: Dict[int, List[Tuple[int, int, int]]]         # robot_id -> path
    cost: float                                           # sum of path lengths

    def __lt__(self, other: "CBSNode") -> bool:
        """Enable heap ordering by cost."""
        return self.cost < other.cost


class ConflictBasedSearch:
    """Conflict-Based Search algorithm for collision-free multi-robot path planning."""

    def __init__(self, grid: np.ndarray, config: Config) -> None:
        """Initialize CBS with the warehouse grid and configuration."""
        self.grid = grid
        self.config = config
        self.astar = SpaceTimeAStar(grid, config)
        self.nodes_expanded: int = 0

    def plan(
        self,
        starts: Dict[int, Tuple[int, int]],
        goals: Dict[int, Tuple[int, int]],
    ) -> Optional[Dict[int, List[Tuple[int, int, int]]]]:
        """
        Main CBS entry point.

        Returns dict mapping robot_id -> list of (row, col, time) tuples,
        or None if no solution found within cbs_max_nodes expansions.
        """
        self.nodes_expanded = 0

        # ── Root node: independent A* paths, no constraints ───────────────────
        root_paths: Dict[int, List[Tuple[int, int, int]]] = {}
        root_constraints: Dict[int, List[Tuple[int, int, int]]] = {rid: [] for rid in starts}

        for robot_id, start in starts.items():
            goal = goals[robot_id]
            path = self.astar.plan(start, goal, [])
            if path is None:
                return None  # Infeasible even without constraints
            root_paths[robot_id] = path

        root = CBSNode(
            constraints=root_constraints,
            paths=root_paths,
            cost=self._compute_sum_of_costs(root_paths),
        )
        open_heap: List[CBSNode] = [root]
        root_cost = root.cost
        best_partial: Optional[Dict[int, List]] = root.paths

        while open_heap:
            N = heapq.heappop(open_heap)
            self.nodes_expanded += 1

            # Node limit — prune heap and return best partial
            if self.nodes_expanded > self.config.cbs_max_nodes:
                return best_partial

            # Heap pruning: discard nodes with cost > 1.5× unconstrained cost
            if N.cost > 1.5 * root_cost and root_cost > 0:
                continue

            conflict = self._find_first_conflict(N.paths)
            if conflict is None:
                # Solution found
                return N.paths

            # Keep track of the deepest explored node's paths as best_partial
            best_partial = N.paths

            # ── Branch on conflict: constrain each involved agent ─────────────
            for agent in conflict["agents"]:
                N_prime_constraints = copy.deepcopy(N.constraints)
                N_prime_paths = copy.deepcopy(N.paths)

                if conflict["type"] == "vertex":
                    N_prime_constraints[agent].append(
                        (conflict["cell"][0], conflict["cell"][1], conflict["time"])
                    )
                elif conflict["type"] == "edge":
                    # For a swap A→B and B→A: constrain agent from entering
                    # the conflict cell at conflict_time AND from leaving its
                    # current position at conflict_time-1 (occupancy constraint).
                    # Practically: add vertex constraint at destination + source.
                    N_prime_constraints[agent].append(
                        (conflict["cell"][0], conflict["cell"][1], conflict["time"])
                    )
                    # Also constrain the agent at conflict_time - 1 at the OTHER
                    # agent's destination (the swap source), blocking the crossing.
                    t_prev = max(0, conflict["time"] - 1)
                    path_agent = N.paths.get(agent, [])
                    if len(path_agent) > t_prev:
                        src_r, src_c, _ = path_agent[t_prev]
                        N_prime_constraints[agent].append((src_r, src_c, conflict["time"]))

                # Replan only the constrained agent
                new_path = self.astar.plan(
                    starts[agent],
                    goals[agent],
                    N_prime_constraints[agent],
                )
                if new_path is not None:
                    N_prime_paths[agent] = new_path
                    new_cost = self._compute_sum_of_costs(N_prime_paths)
                    child = CBSNode(
                        constraints=N_prime_constraints,
                        paths=N_prime_paths,
                        cost=new_cost,
                    )
                    heapq.heappush(open_heap, child)

        return best_partial  # No complete solution found

    def _find_first_conflict(
        self, paths: Dict[int, List[Tuple[int, int, int]]]
    ) -> Optional[Dict]:
        """
        Scan all pairs of robot paths for vertex and edge conflicts.

        Pads shorter paths by repeating the last position (robot waits at goal).
        Returns first conflict found or None.
        """
        robot_ids = list(paths.keys())
        if not robot_ids:
            return None

        # Find max timestep across all paths
        max_t = max(len(p) for p in paths.values())

        # Pad paths
        padded: Dict[int, List[Tuple[int, int, int]]] = {}
        for rid, path in paths.items():
            if len(path) < max_t:
                last = path[-1]
                extra = [(last[0], last[1], last[2] + i + 1) for i in range(max_t - len(path))]
                padded[rid] = path + extra
            else:
                padded[rid] = path

        # Check all pairs
        for i in range(len(robot_ids)):
            for j in range(i + 1, len(robot_ids)):
                a, b = robot_ids[i], robot_ids[j]
                path_a = padded[a]
                path_b = padded[b]

                for t_idx in range(max_t):
                    ra, ca, _ = path_a[t_idx]
                    rb, cb, _ = path_b[t_idx]
                    t = t_idx  # Use index as logical time

                    # Vertex conflict
                    if (ra, ca) == (rb, cb):
                        return {
                            "type": "vertex",
                            "agents": [a, b],
                            "cell": (ra, ca),
                            "time": t,
                        }

                    # Edge conflict (swap positions)
                    if t_idx + 1 < max_t:
                        ra2, ca2, _ = path_a[t_idx + 1]
                        rb2, cb2, _ = path_b[t_idx + 1]
                        if (ra, ca) == (rb2, cb2) and (rb, cb) == (ra2, ca2):
                            return {
                                "type": "edge",
                                "agents": [a, b],
                                "cell": (ra2, ca2),
                                "time": t + 1,
                            }

        return None  # No conflict found

    def _compute_sum_of_costs(self, paths: Dict[int, List]) -> int:
        """Sum of all path lengths (number of timesteps including start)."""
        return sum(len(p) for p in paths.values())
