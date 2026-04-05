"""
main.py — Entry Point for Multi-Robot Warehouse Simulation
=============================================================
Configurable parameters, state diagram output, and simulation launch.
"""

import sys
import os
import random

# Ensure the project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import Warehouse
from robot_task_alocat import RobotTaskAllocator
from state_diagram import print_state_diagram, generate_mermaid_fsm
from simulation import main as run_simulation


def print_banner():
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ◈  MULTI-ROBOT WAREHOUSE NAVIGATION & TASK ALLOCATION  ◈   ║
    ║                                                              ║
    ║   A* Path Planning  •  Collision Avoidance  •  FSM Control   ║
    ║   Dynamic Tasks  •  Performance Metrics  •  State Diagrams   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_configuration(rows, cols, n_robots, n_tasks, seed):
    print("  Configuration:")
    print(f"    Grid Size:     {rows} × {cols}")
    print(f"    Robots:        {n_robots}")
    print(f"    Initial Tasks: {n_tasks}")
    print(f"    Random Seed:   {seed}")
    print()


def main():
    # ── Configuration ────────────────────────────────────────────────────
    ROWS = 20
    COLS = 30
    N_ROBOTS = 4
    N_TASKS = 8
    SEED = 42

    print_banner()
    print_configuration(ROWS, COLS, N_ROBOTS, N_TASKS, SEED)

    # ── Print FSM State Diagram ──────────────────────────────────────────
    print_state_diagram()

    # ── Ensure output directory exists ───────────────────────────────────
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Save FSM diagram before simulation starts
    with open(os.path.join(output_dir, "fsm_initial.md"), "w") as f:
        f.write("# Robot FSM State Diagram\n\n")
        f.write("```mermaid\n")
        f.write(generate_mermaid_fsm())
        f.write("\n```\n")
    print(f"  [output] Initial FSM diagram saved to output/fsm_initial.md\n")

    # ── Console-only mode (headless test) ────────────────────────────────
    if "--headless" in sys.argv:
        print("  Running in headless mode (no GUI)...\n")
        rng = random.Random(SEED)
        warehouse = Warehouse(ROWS, COLS)
        warehouse.generate_realistic_layout()
        robots = warehouse.generate_robot_starts(N_ROBOTS, rng)
        tasks = warehouse.generate_random_tasks(N_TASKS, rng)
        allocator = RobotTaskAllocator(warehouse, robots, tasks)

        result = allocator.run_until_done(max_steps=2000)
        allocator.metrics.print_report()
        allocator.metrics.plot_metrics(output_dir=output_dir)

        print(f"  Completed in {result['total_steps']} steps.")
        print(f"  Tasks completed: {len(result['completed_tasks'])}")
        print(f"  Tasks pending:   {len(result['pending_tasks'])}")
        return

    # ── Launch GUI simulation ────────────────────────────────────────────
    print("  Launching simulation window...\n")
    print("  Controls:")
    print("    Space  — Pause / Resume")
    print("    +/-    — Increase / Decrease speed")
    print("    D      — Inject dynamic tasks")
    print("    M      — Toggle metrics overlay")
    print("    Esc    — Quit\n")

    run_simulation()


if __name__ == "__main__":
    main()
