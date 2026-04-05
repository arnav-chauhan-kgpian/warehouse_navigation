"""
task_allocator.py — Multi-Robot Task Allocation
================================================
Provides two allocation strategies:

  greedy_allocate()
      Assigns each pending task to the nearest idle robot (O(R·T)).
      Suitable for real-time / dynamic scenarios.

  hungarian_allocate()
      Solves the assignment problem optimally using the Hungarian algorithm
      (via scipy.optimize.linear_sum_assignment), minimising total
      robot→pickup travel distance.  Falls back to greedy if scipy is absent.

Both functions accept the current robot and task lists and return a list of
``(Robot, Task)`` pairs ready to be handed to the simulation engine.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from robot import Robot, RobotStatus, Task, TaskStatus

try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _idle_and_pending(
    robots: List[Robot], tasks: List[Task]
) -> Tuple[List[Robot], List[Task]]:
    """Filter to idle robots and pending tasks, sorted for determinism."""
    idle    = [r for r in robots if r.is_idle]
    pending = sorted(
        [t for t in tasks if t.status == TaskStatus.PENDING],
        key=lambda t: t.created_time,   # FIFO priority
    )
    return idle, pending


# ---------------------------------------------------------------------------
# Greedy allocation
# ---------------------------------------------------------------------------

def greedy_allocate(
    robots: List[Robot],
    tasks:  List[Task],
    current_time: int = 0,
) -> List[Tuple[Robot, Task]]:
    """
    Greedy nearest-robot task allocation.

    For each pending task (oldest first), find the closest idle robot and
    assign it.  Both robot and task are removed from consideration once
    matched.

    Returns
    -------
    List of ``(robot, task)`` pairs — may be empty if no robots are idle or
    no tasks are pending.
    """
    idle, pending = _idle_and_pending(robots, tasks)
    if not idle or not pending:
        return []

    used_robots = set()
    used_tasks  = set()
    assignments: List[Tuple[Robot, Task]] = []

    for task in pending:
        best_robot = None
        best_dist  = math.inf

        for robot in idle:
            if robot.id in used_robots:
                continue
            d = _manhattan(robot.pos, task.pickup)
            if d < best_dist or (d == best_dist and robot.id < best_robot.id):
                best_dist  = d
                best_robot = robot

        if best_robot is not None:
            assignments.append((best_robot, task))
            used_robots.add(best_robot.id)
            used_tasks.add(task.id)

    return assignments


# ---------------------------------------------------------------------------
# Hungarian (optimal) allocation
# ---------------------------------------------------------------------------

def hungarian_allocate(
    robots: List[Robot],
    tasks:  List[Task],
    current_time: int = 0,
) -> List[Tuple[Robot, Task]]:
    """
    Optimal task allocation via the Hungarian algorithm.

    Builds a cost matrix  C[i][j] = Manhattan distance from idle robot i to
    the pickup of pending task j, then finds the minimum-weight matching.
    When there are more tasks than robots (or vice versa) the matrix is padded
    with a large sentinel cost so unmatched rows/columns are never preferred.

    Falls back to ``greedy_allocate`` if *scipy* is not installed.

    Returns
    -------
    List of ``(robot, task)`` pairs.
    """
    if not _SCIPY:
        print(
            "[task_allocator] scipy not available — falling back to greedy allocation."
        )
        return greedy_allocate(robots, tasks, current_time)

    idle, pending = _idle_and_pending(robots, tasks)
    if not idle or not pending:
        return []

    R, T    = len(idle), len(pending)
    size    = max(R, T)
    BIG     = 10_000.0
    cost    = np.full((size, size), BIG, dtype=float)

    for i, robot in enumerate(idle):
        for j, task in enumerate(pending):
            cost[i, j] = _manhattan(robot.pos, task.pickup)

    row_ind, col_ind = linear_sum_assignment(cost)

    assignments: List[Tuple[Robot, Task]] = []
    for r_idx, t_idx in zip(row_ind, col_ind):
        if r_idx < R and t_idx < T and cost[r_idx, t_idx] < BIG:
            assignments.append((idle[r_idx], pending[t_idx]))

    return assignments
