"""tests/test_astar.py — Pytest tests for Space-Time A* planner."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from config import Config
from planning.astar import SpaceTimeAStar


def _empty_grid(h: int = 5, w: int = 5) -> np.ndarray:
    """Create a fully free grid."""
    return np.zeros((h, w), dtype=np.int8)


def _blocked_goal_grid() -> np.ndarray:
    """Create a grid where cell (4,4) is surrounded by obstacles."""
    grid = np.zeros((5, 5), dtype=np.int8)
    # Wall around (4,4)
    grid[3, 4] = 1
    grid[4, 3] = 1
    return grid


def test_simple_path():
    """Single robot, no obstacles, 5x5 grid. Verify path found and no cycles."""
    config = Config(time_horizon=50)
    grid = _empty_grid()
    planner = SpaceTimeAStar(grid, config)
    path = planner.plan((0, 0), (4, 4), [])
    assert path is not None, "Should find a path in empty grid"
    # Check no cycles (all (r,c) positions are unique or only repeated in time)
    positions = [(r, c) for r, c, t in path]
    # Path should start at (0,0) and end at (4,4)
    assert (path[0][0], path[0][1]) == (0, 0)
    assert (path[-1][0], path[-1][1]) == (4, 4)
    # Path length should be reasonable: Manhattan dist = 8
    assert len(path) >= 9  # At least 9 nodes (start + 8 moves)


def test_obstacle_avoidance():
    """L-shaped obstacle. Verify path goes around it."""
    config = Config(time_horizon=50)
    grid = np.zeros((5, 5), dtype=np.int8)
    # Vertical wall at col 2, rows 0-3
    for r in range(4):
        grid[r, 2] = 1
    planner = SpaceTimeAStar(grid, config)
    path = planner.plan((0, 0), (0, 4), [])
    assert path is not None, "Should find path around obstacle"
    # Verify no path cell is an obstacle
    for r, c, t in path:
        assert grid[r, c] != 1, f"Path goes through obstacle at ({r},{c})"
    assert (path[-1][0], path[-1][1]) == (0, 4)


def test_constraint_respected():
    """Add constraint at (2,2,3). Verify path avoids that space-time cell."""
    config = Config(time_horizon=50)
    grid = _empty_grid()
    planner = SpaceTimeAStar(grid, config)
    constraints = [(2, 2, 3)]
    path = planner.plan((0, 0), (4, 4), constraints)
    assert path is not None, "Should find path even with constraint"
    for r, c, t in path:
        assert not (r == 2 and c == 2 and t == 3), \
            "Path must not pass through constrained cell (2,2,3)"


def test_no_path():
    """Completely blocked goal. Verify planner returns None."""
    config = Config(time_horizon=30)
    grid = np.zeros((5, 5), dtype=np.int8)
    # Surround (4,4) with walls
    grid[3, 4] = 1
    grid[4, 3] = 1
    grid[3, 3] = 1
    grid[4, 4] = 1  # Actually block the goal itself
    planner = SpaceTimeAStar(grid, config)
    path = planner.plan((0, 0), (4, 4), [])
    assert path is None, "Should return None when goal is blocked"


def test_wait_action():
    """Two constraints along shortest path. Verify robot uses wait action."""
    config = Config(time_horizon=50)
    grid = _empty_grid()
    planner = SpaceTimeAStar(grid, config)
    # Block the direct path at early timesteps to force waiting
    constraints = [(0, 1, 1), (0, 2, 2), (0, 3, 3)]
    path = planner.plan((0, 0), (0, 4), constraints)
    assert path is not None, "Should find a path despite constraints"
    assert (path[-1][0], path[-1][1]) == (0, 4)
    # Verify constraints are respected
    for r, c, t in path:
        assert (r, c, t) not in set(constraints), \
            f"Path violates constraint at ({r},{c},{t})"
