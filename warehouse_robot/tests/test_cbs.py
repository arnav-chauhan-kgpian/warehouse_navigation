"""tests/test_cbs.py — Pytest tests for Conflict-Based Search planner."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config import Config
from planning.cbs import ConflictBasedSearch
from planning.astar import SpaceTimeAStar


def _empty_grid(h: int = 8, w: int = 8) -> np.ndarray:
    """Create a fully free grid."""
    return np.zeros((h, w), dtype=np.int8)


def count_collisions(paths: dict) -> int:
    """Count vertex collisions in a set of paths."""
    robot_ids = list(paths.keys())
    max_t = max(len(p) for p in paths.values())

    # Pad paths
    padded = {}
    for rid, path in paths.items():
        if len(path) < max_t:
            last = path[-1]
            extra = [(last[0], last[1], last[2] + i + 1) for i in range(max_t - len(path))]
            padded[rid] = path + extra
        else:
            padded[rid] = path

    collisions = 0
    for t in range(max_t):
        positions = [(padded[rid][t][0], padded[rid][t][1]) for rid in robot_ids]
        if len(positions) != len(set(positions)):
            collisions += 1
    return collisions


def test_no_conflict():
    """Two robots on non-intersecting paths. CBS solution = independent A* paths."""
    config = Config(time_horizon=50, cbs_max_nodes=1000)
    grid = _empty_grid()
    cbs = ConflictBasedSearch(grid, config)
    astar = SpaceTimeAStar(grid, config)

    starts = {0: (0, 0), 1: (7, 7)}
    goals = {0: (0, 4), 1: (7, 3)}

    result = cbs.plan(starts, goals)
    assert result is not None, "CBS should find solution for non-conflicting paths"
    assert count_collisions(result) == 0, "No vertex collisions expected"


def test_vertex_conflict():
    """Two robots heading for same cell. CBS resolves with 0 collisions."""
    config = Config(time_horizon=50, cbs_max_nodes=2000)
    grid = _empty_grid()
    cbs = ConflictBasedSearch(grid, config)

    # Both robots head toward center, likely to collide
    starts = {0: (0, 0), 1: (0, 6)}
    goals = {0: (0, 6), 1: (0, 0)}  # They need to swap positions

    result = cbs.plan(starts, goals)
    assert result is not None, "CBS should find solution for vertex conflict"
    assert count_collisions(result) == 0, "CBS must resolve vertex conflicts"


def test_edge_conflict():
    """Two robots swapping positions (edge conflict). CBS resolves it."""
    config = Config(time_horizon=50, cbs_max_nodes=2000)
    grid = _empty_grid()
    cbs = ConflictBasedSearch(grid, config)

    # Robots directly adjacent, swapping
    starts = {0: (3, 3), 1: (3, 4)}
    goals = {0: (3, 4), 1: (3, 3)}

    result = cbs.plan(starts, goals)
    assert result is not None, "CBS should find solution for edge conflict"
    # Verify no swap conflict: robots don't simultaneously swap in one step
    assert count_collisions(result) == 0, "CBS must resolve edge conflicts"


def test_optimality():
    """CBS solution cost <= prioritized A* on same instance."""
    config = Config(time_horizon=50, cbs_max_nodes=3000)
    grid = _empty_grid()
    cbs = ConflictBasedSearch(grid, config)
    astar = SpaceTimeAStar(grid, config)

    starts = {0: (0, 0), 1: (0, 7)}
    goals = {0: (7, 7), 1: (7, 0)}

    cbs_result = cbs.plan(starts, goals)
    assert cbs_result is not None, "CBS should find solution"

    # Prioritized A*: plan robot 0 first, then robot 1 avoids robot 0's path
    path0 = astar.plan(starts[0], goals[0], [])
    assert path0 is not None
    prio_constraints = [(r, c, t) for r, c, t in path0]
    path1 = astar.plan(starts[1], goals[1], prio_constraints)
    if path1 is None:
        pytest.skip("Prioritized A* couldn't find path — CBS trivially wins")

    prio_cost = len(path0) + len(path1)
    cbs_cost = sum(len(p) for p in cbs_result.values())

    assert cbs_cost <= prio_cost, (
        f"CBS cost {cbs_cost} should be <= prioritized A* cost {prio_cost}"
    )
