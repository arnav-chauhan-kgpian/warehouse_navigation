"""allocation/hungarian.py — Hungarian Algorithm task allocator for robots."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianAllocator:
    """Optimal task allocator using the Hungarian Algorithm (scipy wrapper)."""

    def __init__(self) -> None:
        """Initialize the allocator."""
        pass

    def assign(
        self,
        robot_positions: List[Tuple[int, int]],
        tasks: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        active_robots: Optional[List[int]] = None,
    ) -> Dict[int, int]:
        """
        Compute optimal assignment using scipy linear_sum_assignment.

        Handles rectangular matrices (more tasks than robots or vice versa).
        If active_robots is provided, only those robots are considered.

        Returns dict mapping robot_id -> task_id.
        """
        if active_robots is None:
            active_robots = list(range(len(robot_positions)))

        if not active_robots or not tasks:
            return {}

        k = len(active_robots)
        m = len(tasks)

        # Build cost matrix: C[i][j] = dist(robot → pickup) + dist(pickup → dropoff)
        C = np.zeros((k, m), dtype=np.float32)
        for idx, robot_id in enumerate(active_robots):
            for j, (pickup, dropoff) in enumerate(tasks):
                C[idx][j] = self.manhattan(robot_positions[robot_id], pickup) + \
                             self.manhattan(pickup, dropoff)

        row_ind, col_ind = linear_sum_assignment(C)

        assignment: Dict[int, int] = {}
        for i, j in zip(row_ind, col_ind):
            robot_id = active_robots[i]
            assignment[robot_id] = j

        return assignment

    def reassign_dynamic(
        self,
        robot_positions: List[Tuple[int, int]],
        tasks: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        task_status: List[str],
        robot_tasks: List[Optional[int]],
    ) -> Dict[int, int]:
        """
        Dynamic reassignment: only assign IDLE robots to PENDING tasks.

        Returns updated assignment dict for idle robots only.
        """
        idle_robots = [i for i, t in enumerate(robot_tasks) if t is None]
        pending_tasks_idx = [j for j, s in enumerate(task_status) if s == "pending"]

        if not idle_robots or not pending_tasks_idx:
            return {}

        pending_tasks = [tasks[j] for j in pending_tasks_idx]
        raw_assignment = self.assign(robot_positions, pending_tasks, active_robots=idle_robots)

        # Map back from pending_task_local_index to original task index
        result: Dict[int, int] = {}
        for robot_id, local_task_idx in raw_assignment.items():
            original_task_idx = pending_tasks_idx[local_task_idx]
            result[robot_id] = original_task_idx

        return result

    @staticmethod
    def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two grid positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
