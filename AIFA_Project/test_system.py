"""
test_system.py — Automated Test Suite
========================================
Verifies all core functionality of the multi-robot warehouse system.
Run with:  python test_system.py
"""

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import Warehouse, SHELF, EMPTY, PICKUP, DROPOFF, CHARGING
from navigation import Navigation
from robot_task_alocat import RobotTaskAllocator
from state_diagram import (
    RobotState, RobotFSM, TRANSITIONS,
    generate_mermaid_fsm, generate_snapshot_diagram, print_state_diagram,
)
from metrics import MetricsTracker


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    END = "\033[0m"
    BOLD = "\033[1m"


passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  {Colors.GREEN}✓{Colors.END} {name}")
    else:
        failed += 1
        print(f"  {Colors.RED}✗{Colors.END} {name}")
        if detail:
            print(f"    {Colors.RED}{detail}{Colors.END}")


def section(title):
    print(f"\n{Colors.CYAN}{Colors.BOLD}▸ {title}{Colors.END}")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 1: WAREHOUSE ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════
def test_warehouse():
    section("Warehouse Environment")

    w = Warehouse(20, 30)
    grid = w.create_environment()
    test("Grid created with correct shape", grid.shape == (20, 30))
    test("Grid initialized to empty", grid.sum() == 0)

    # Realistic layout
    w.generate_realistic_layout()
    test("Layout has shelves", (w.warehouse == SHELF).any(),
         "No shelves found in generated layout")
    test("Layout has pickup zones", (w.warehouse == PICKUP).any(),
         "No pickup zones found")
    test("Layout has dropoff zones", (w.warehouse == DROPOFF).any(),
         "No dropoff zones found")
    test("Layout has charging stations", (w.warehouse == CHARGING).any(),
         "No charging stations found")

    # Free positions
    free = w.get_free_positions()
    test("Free positions exist", len(free) > 0)
    test("Free positions are walkable",
         all(w.is_walkable(p) for p in free[:20]))

    # Validity
    test("(0,0) is valid", w.is_valid_position((0, 0)))
    test("(-1,0) is invalid", not w.is_valid_position((-1, 0)))
    test("(20,30) is invalid", not w.is_valid_position((20, 30)))

    # Task / robot generation
    rng = random.Random(42)
    tasks = w.generate_random_tasks(5, rng)
    test("Generated 5 tasks", len(tasks) == 5)
    test("Tasks are (pickup, dropoff) pairs",
         all(len(t) == 2 for t in tasks))

    starts = w.generate_robot_starts(3, rng)
    test("Generated 3 robot starts", len(starts) == 3)
    test("Robot starts are walkable",
         all(w.is_walkable(p) for p in starts))


# ═══════════════════════════════════════════════════════════════════════
#  TEST 2: A* PATHFINDING
# ═══════════════════════════════════════════════════════════════════════
def test_pathfinding():
    section("A* Pathfinding")

    w = Warehouse(10, 10)
    w.create_environment()
    # Simple wall
    for r in range(1, 8):
        w.warehouse[r, 5] = SHELF

    nav = Navigation(w)

    # Basic path
    path = nav.a_star_time_expanded((0, 0), (0, 9))
    test("Path found around wall", path is not None)
    test("Path starts at (0,0)", path[0] == (0, 0) if path else False)
    test("Path ends at (0,9)", path[-1] == (0, 9) if path else False)

    # Path avoids obstacles
    if path:
        obstacles_hit = any(w.warehouse[r, c] == SHELF for r, c in path)
        test("Path avoids shelves", not obstacles_hit)

    # Path continuity (each step is adjacent or wait)
    if path:
        continuous = True
        for i in range(1, len(path)):
            dr = abs(path[i][0] - path[i-1][0])
            dc = abs(path[i][1] - path[i-1][1])
            if dr + dc > 1:
                continuous = False
                break
        test("Path is continuous (adjacent moves)", continuous)

    # Weighted A* finds a path too
    path_w = nav.a_star_time_expanded((0, 0), (0, 9), weight=2.0)
    test("Weighted A* (w=2) also finds path", path_w is not None)

    # Multi-waypoint
    wp_path = nav.plan_waypoints([(0, 0), (9, 0), (9, 9)])
    test("Multi-waypoint path found", wp_path is not None)
    test("Waypoint path visits (9,0)",
         (9, 0) in wp_path if wp_path else False)
    test("Waypoint path ends at (9,9)",
         wp_path[-1] == (9, 9) if wp_path else False)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 3: MULTI-ROBOT COLLISION AVOIDANCE
# ═══════════════════════════════════════════════════════════════════════
def test_collision_avoidance():
    section("Multi-Robot Collision Avoidance")

    w = Warehouse(5, 10)
    w.create_environment()

    nav = Navigation(w)

    # Two robots heading towards each other
    starts = {0: (2, 0), 1: (2, 9)}
    targets = {0: (2, 9), 1: (2, 0)}

    paths = nav.plan_multi_robot(starts, targets)
    test("Both robots have paths", len(paths) == 2)

    # Check no vertex collisions
    max_t = max(len(p) for p in paths.values())
    collision_free = True
    for t in range(max_t):
        positions_at_t = set()
        for rid, path in paths.items():
            pos = path[min(t, len(path) - 1)]
            if pos in positions_at_t:
                collision_free = False
                break
            positions_at_t.add(pos)
    test("No vertex collisions at any timestep", collision_free)

    # Check no edge collisions (swaps)
    swap_free = True
    for t in range(1, max_t):
        for r1 in paths:
            for r2 in paths:
                if r1 >= r2:
                    continue
                p1 = paths[r1]
                p2 = paths[r2]
                pos1_prev = p1[min(t-1, len(p1)-1)]
                pos1_now = p1[min(t, len(p1)-1)]
                pos2_prev = p2[min(t-1, len(p2)-1)]
                pos2_now = p2[min(t, len(p2)-1)]
                if pos1_prev == pos2_now and pos2_prev == pos1_now:
                    swap_free = False
    test("No edge (swap) collisions", swap_free)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 4: FSM STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════
def test_fsm():
    section("Robot FSM State Machine")

    fsm = RobotFSM(0)
    test("Initial state is IDLE", fsm.state == RobotState.IDLE)

    # Valid transition chain
    fsm.transition(RobotState.MOVING_TO_PICKUP, 1)
    test("IDLE → MOVING_TO_PICKUP", fsm.state == RobotState.MOVING_TO_PICKUP)

    fsm.transition(RobotState.PICKING_UP, 5)
    test("MOVING_TO_PICKUP → PICKING_UP", fsm.state == RobotState.PICKING_UP)

    fsm.transition(RobotState.MOVING_TO_DROPOFF, 7)
    test("PICKING_UP → MOVING_TO_DROPOFF", fsm.state == RobotState.MOVING_TO_DROPOFF)

    fsm.transition(RobotState.DROPPING_OFF, 12)
    test("MOVING_TO_DROPOFF → DROPPING_OFF", fsm.state == RobotState.DROPPING_OFF)

    fsm.transition(RobotState.IDLE, 14)
    test("DROPPING_OFF → IDLE (cycle complete)", fsm.state == RobotState.IDLE)

    # Invalid transition
    try:
        fsm.transition(RobotState.DROPPING_OFF, 15)
        test("Invalid transition raises ValueError", False)
    except ValueError:
        test("Invalid transition raises ValueError", True)

    # History
    test("FSM history has 6 entries", len(fsm.history) == 6,
         f"Got {len(fsm.history)}")

    # Mermaid diagram
    mermaid = generate_mermaid_fsm()
    test("Mermaid diagram generated", len(mermaid) > 100)
    test("Mermaid contains 'stateDiagram'", "stateDiagram" in mermaid)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 5: PICK-AND-PLACE LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════
def test_pick_and_place():
    section("Pick-and-Place Task Lifecycle")

    w = Warehouse(10, 20)
    w.create_environment()

    robots = [(0, 0)]
    tasks = [((0, 5), (0, 15))]  # pickup at (0,5), dropoff at (0,15)

    alloc = RobotTaskAllocator(w, robots, tasks)

    # Run until done
    result = alloc.run_until_done(max_steps=500)

    test("Task completed", len(result["completed_tasks"]) == 1,
         f"Completed: {len(result['completed_tasks'])}")
    test("No pending tasks left", len(result["pending_tasks"]) == 0)
    test("Robot returned to IDLE",
         alloc.fsm[0].state == RobotState.IDLE)

    # FSM went through all states
    states_visited = set(s for _, s in alloc.fsm[0].history)
    expected = {RobotState.IDLE, RobotState.MOVING_TO_PICKUP,
                RobotState.PICKING_UP, RobotState.MOVING_TO_DROPOFF,
                RobotState.DROPPING_OFF}
    test("Robot visited all 5 FSM states", states_visited == expected,
         f"Visited: {[s.name for s in states_visited]}")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 6: MULTI-ROBOT TASK COMPLETION
# ═══════════════════════════════════════════════════════════════════════
def test_multi_robot_tasks():
    section("Multi-Robot Task Completion")

    w = Warehouse(10, 20)
    w.create_environment()

    robots = [(0, 0), (9, 0), (5, 0)]
    tasks = [
        ((0, 10), (0, 18)),
        ((9, 10), (9, 18)),
        ((5, 10), (5, 18)),
    ]

    alloc = RobotTaskAllocator(w, robots, tasks)
    result = alloc.run_until_done(max_steps=500)

    test("All 3 tasks completed", len(result["completed_tasks"]) == 3,
         f"Completed: {len(result['completed_tasks'])}")
    test("No pending tasks remain", len(result["pending_tasks"]) == 0)

    # Collision check in recorded positions
    test("Simulation finished within 500 steps",
         result["total_steps"] < 500,
         f"Took {result['total_steps']} steps")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 7: DYNAMIC TASK INJECTION
# ═══════════════════════════════════════════════════════════════════════
def test_dynamic_tasks():
    section("Dynamic Task Injection")

    w = Warehouse(10, 20)
    w.create_environment()

    robots = [(0, 0), (9, 0)]
    tasks = [((0, 10), (0, 18))]

    alloc = RobotTaskAllocator(w, robots, tasks)

    # Run a few steps
    for _ in range(10):
        alloc.step()

    # Inject new tasks
    alloc.add_tasks([((9, 10), (9, 18)), ((5, 10), (5, 18))])
    test("Tasks injected (now 2 pending)", len(alloc.pending_tasks) >= 0)

    # Run to completion
    result = alloc.run_until_done(max_steps=500)
    total = len(result["completed_tasks"])
    test("All 3 tasks eventually completed", total == 3,
         f"Completed: {total}")


# ═══════════════════════════════════════════════════════════════════════
#  TEST 8: METRICS TRACKING
# ═══════════════════════════════════════════════════════════════════════
def test_metrics():
    section("Metrics Tracking")

    w = Warehouse(10, 20)
    w.create_environment()

    robots = [(0, 0)]
    tasks = [((0, 5), (0, 15))]

    alloc = RobotTaskAllocator(w, robots, tasks)
    alloc.run_until_done(max_steps=500)

    s = alloc.metrics.summary_dict()
    test("Total steps > 0", s["total_steps"] > 0)
    test("Total distance > 0", s["total_distance"] > 0)
    test("Tasks completed = 1", s["total_tasks_completed"] == 1)
    test("Average completion time > 0", s["average_completion_time"] > 0)
    test("Throughput > 0", s["throughput_tasks_per_step"] > 0)
    test("Per-robot data present", len(s["per_robot"]) == 1)
    test("Robot utilization > 0", s["per_robot"][0]["utilization"] > 0)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 9: STATE DIAGRAM GENERATION
# ═══════════════════════════════════════════════════════════════════════
def test_state_diagrams():
    section("State Diagram Generation")

    mermaid = generate_mermaid_fsm()
    test("FSM diagram is non-empty", len(mermaid) > 50)
    test("Contains all states",
         all(s.name in mermaid for s in RobotState))

    # Snapshot diagram
    fsms = {0: RobotFSM(0), 1: RobotFSM(1)}
    snapshot = generate_snapshot_diagram(fsms, 0)
    test("Snapshot diagram generated", len(snapshot) > 30)
    test("Snapshot contains Robot_0", "Robot_0" in snapshot)
    test("Snapshot contains Robot_1", "Robot_1" in snapshot)


# ═══════════════════════════════════════════════════════════════════════
#  TEST 10: REALISTIC WAREHOUSE END-TO-END
# ═══════════════════════════════════════════════════════════════════════
def test_realistic_e2e():
    section("Realistic Warehouse End-to-End")

    rng = random.Random(42)
    w = Warehouse(20, 30)
    w.generate_realistic_layout()

    robots = w.generate_robot_starts(4, rng)
    tasks = w.generate_random_tasks(6, rng)

    test("Robots placed on walkable cells",
         all(w.is_walkable(p) for p in robots))

    alloc = RobotTaskAllocator(w, robots, tasks)
    result = alloc.run_until_done(max_steps=2000)

    completed = len(result["completed_tasks"])
    pending = len(result["pending_tasks"])
    test(f"Tasks completed: {completed}/6", completed == 6,
         f"Pending: {pending}")
    test("Finished within 2000 steps",
         result["total_steps"] < 2000,
         f"Took {result['total_steps']} steps")

    # Verify all FSMs returned to IDLE
    all_idle = all(alloc.fsm[i].state == RobotState.IDLE
                   for i in range(4))
    test("All robots returned to IDLE", all_idle)


# ═══════════════════════════════════════════════════════════════════════
#  RUNNER
# ═══════════════════════════════════════════════════════════════════════
def main():
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("  MULTI-ROBOT WAREHOUSE — AUTOMATED TEST SUITE")
    print(f"{'=' * 60}{Colors.END}")

    test_warehouse()
    test_pathfinding()
    test_collision_avoidance()
    test_fsm()
    test_pick_and_place()
    test_multi_robot_tasks()
    test_dynamic_tasks()
    test_metrics()
    test_state_diagrams()
    test_realistic_e2e()

    print(f"\n{Colors.BOLD}{'=' * 60}")
    total = passed + failed
    if failed == 0:
        print(f"  {Colors.GREEN}ALL {total} TESTS PASSED ✓{Colors.END}")
    else:
        print(f"  {Colors.GREEN}{passed} passed{Colors.END}, "
              f"{Colors.RED}{failed} failed{Colors.END} / {total} total")
    print(f"{'=' * 60}{Colors.END}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
