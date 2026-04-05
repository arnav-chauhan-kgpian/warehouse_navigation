"""tests/test_hungarian.py — Pytest tests for Hungarian task allocator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from allocation.hungarian import HungarianAllocator


def test_square_matrix():
    """3 robots, 3 tasks. Verify assignment is globally optimal."""
    allocator = HungarianAllocator()
    robot_positions = [(0, 0), (5, 5), (9, 9)]
    tasks = [
        ((1, 1), (2, 2)),   # Close to robot 0
        ((6, 6), (7, 7)),   # Close to robot 1
        ((8, 8), (9, 0)),   # Close to robot 2
    ]
    assignment = allocator.assign(robot_positions, tasks)
    assert len(assignment) == 3, "All 3 robots should be assigned"
    assert set(assignment.keys()) == {0, 1, 2}, "All robot IDs present"
    assert set(assignment.values()) == {0, 1, 2}, "All task IDs assigned"
    # Optimal: robot 0 → task 0 (cost 2), robot 1 → task 1 (cost 2+2=4?), robot 2 → task 2
    assert assignment[0] == 0, "Robot 0 should get nearest task (task 0)"
    assert assignment[1] == 1, "Robot 1 should get nearest task (task 1)"


def test_rectangular_more_tasks():
    """2 robots, 5 tasks. Verify only 2 tasks get assigned."""
    allocator = HungarianAllocator()
    robot_positions = [(0, 0), (9, 9)]
    tasks = [
        ((0, 1), (0, 2)),
        ((9, 8), (9, 7)),
        ((5, 5), (5, 6)),
        ((2, 3), (3, 4)),
        ((7, 8), (8, 9)),
    ]
    assignment = allocator.assign(robot_positions, tasks)
    assert len(assignment) == 2, "Only 2 robots should be assigned (one task each)"
    assert set(assignment.keys()) == {0, 1}
    # Each assigned task_id must be a valid index
    for robot_id, task_id in assignment.items():
        assert 0 <= task_id < len(tasks)


def test_rectangular_more_robots():
    """4 robots, 2 tasks. Verify only 2 robots get tasks."""
    allocator = HungarianAllocator()
    robot_positions = [(0, 0), (1, 1), (9, 9), (8, 8)]
    tasks = [
        ((0, 2), (0, 4)),    # Near robots 0 and 1
        ((9, 7), (9, 5)),    # Near robots 2 and 3
    ]
    assignment = allocator.assign(robot_positions, tasks)
    assert len(assignment) == 2, "Only 2 robots should be assigned (matching task count)"
    task_ids = list(assignment.values())
    assert sorted(task_ids) == [0, 1], "Both tasks should be covered"


def test_dynamic_reassign():
    """Mix of idle and busy robots. Verify only idle robots reassigned."""
    allocator = HungarianAllocator()
    robot_positions = [(0, 0), (5, 5), (9, 9), (3, 3)]
    tasks = [
        ((0, 1), (0, 2)),
        ((5, 6), (6, 6)),
        ((9, 8), (8, 8)),
        ((3, 4), (4, 4)),
    ]
    task_status = ["pending", "in_progress_pickup", "pending", "done"]
    robot_tasks = [None, 1, None, 3]  # Robots 1 and 3 are busy

    assignment = allocator.reassign_dynamic(
        robot_positions, tasks, task_status, robot_tasks
    )

    # Only robots 0 and 2 should be assigned (idle ones)
    for robot_id in assignment.keys():
        assert robot_tasks[robot_id] is None, \
            f"Robot {robot_id} is busy but got reassigned"

    # Only pending tasks (0 and 2) should appear
    for task_id in assignment.values():
        assert task_status[task_id] == "pending", \
            f"Task {task_id} is not pending but was assigned"
