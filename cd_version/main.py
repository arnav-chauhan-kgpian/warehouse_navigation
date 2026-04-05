"""
main.py — Multi-Robot Warehouse Navigation & Task Allocation
=============================================================

Entry point for the simulation.  All scenario parameters are centralised
in the CONFIG dictionary at the top of this file — no other file needs
to be edited to change the experiment.

Quick start
-----------
    pip install matplotlib numpy scipy
    python main.py

What this project implements
-----------------------------
  • Warehouse grid  : 2-D shelf layout with aisles and margins.
  • Path planning   : Space-Time A* (collision-free, optimal per-robot paths)
                      with a standard A* fallback.
  • Collision avoid.: Prioritised planning via a shared reservation table.
                      Swap-conflict detection prevents robots from crossing.
  • Task allocation : Greedy nearest-robot OR Hungarian optimal assignment.
  • Dynamic tasks   : Tasks that arrive at scheduled simulation timesteps.
  • Visualisation   : Live animation + final metric charts.
"""

from __future__ import annotations

import random
from typing import List, Tuple

# ── Project modules ────────────────────────────────────────────────────────
from warehouse      import Warehouse
from robot          import Robot, Task, RobotStatus
from simulation     import Simulation
from visualizer     import WarehouseVisualizer, ROBOT_PALETTE


# ===========================================================================
# CONFIGURATION
# ===========================================================================

CONFIG: dict = {
    # ── Warehouse ──────────────────────────────────────────────────────────
    "warehouse_rows":   22,
    "warehouse_cols":   24,
    "shelf_height":      2,   # grid rows per shelf block
    "shelf_width":       3,   # grid cols per shelf block
    "aisle_gap":         2,   # free cells between shelves

    # ── Fleet ─────────────────────────────────────────────────────────────
    "num_robots":        4,

    # ── Static tasks (present from step 0) ────────────────────────────────
    "num_static_tasks":  8,

    # ── Dynamic tasks (arrive mid-simulation) ─────────────────────────────
    "num_dynamic_tasks": 4,
    "dyn_first_arrival": 40,   # step at which the first dynamic task arrives
    "dyn_interval":      30,   # steps between successive dynamic task arrivals

    # ── Planning ──────────────────────────────────────────────────────────
    # 'greedy'   → O(R·T) nearest-robot assignment  (fast, good for real-time)
    # 'hungarian'→ optimal global assignment         (requires scipy)
    "allocation_method": "greedy",
    "max_steps":         700,
    "replan_horizon":    130,   # max future steps Space-Time A* explores

    # ── Visualisation ─────────────────────────────────────────────────────
    "anim_interval_ms":  120,   # lower = faster playback
    "show_trails":       True,
    "show_paths":        True,
    "show_animation":    True,
    "show_metrics":      True,
    "show_final_map":    True,

    # ── Reproducibility ───────────────────────────────────────────────────
    "seed": 42,
}


# ===========================================================================
# FACTORY FUNCTIONS
# ===========================================================================

def build_warehouse(cfg: dict) -> Warehouse:
    return Warehouse(
        rows         = cfg["warehouse_rows"],
        cols         = cfg["warehouse_cols"],
        shelf_height = cfg["shelf_height"],
        shelf_width  = cfg["shelf_width"],
        aisle_gap    = cfg["aisle_gap"],
        seed         = cfg["seed"],
    )


def spawn_robots(
    num:  int,
    wh:   Warehouse,
    seed: int,
) -> List[Robot]:
    """
    Place robots on free cells in the lower quarter of the warehouse
    (staging area), ensuring no two robots share a cell.
    """
    rng = random.Random(seed + 77)

    # Prefer bottom staging area
    staging = [
        cell for cell in wh.free_cells
        if cell[0] >= int(wh.rows * 0.70)
    ]
    if len(staging) < num:
        staging = list(wh.free_cells)

    rng.shuffle(staging)
    positions = staging[:num]

    robots = []
    for i, pos in enumerate(positions):
        robots.append(Robot(
            id    = i,
            pos   = pos,
            color = ROBOT_PALETTE[i % len(ROBOT_PALETTE)],
        ))
        print(f"    Robot {i:2d}  spawned at  row={pos[0]:2d}, col={pos[1]:2d}")

    return robots


def generate_tasks(
    count:      int,
    wh:         Warehouse,
    occupied:   List[Tuple[int, int]],
    seed:       int,
    id_offset:  int = 0,
    created_at: int = 0,
) -> List[Task]:
    """
    Generate *count* pick-and-place tasks on randomly chosen free cells.
    Pickup and dropoff are guaranteed to be distinct and unoccupied.
    """
    rng   = random.Random(seed + 200)
    used  = set(occupied)
    tasks = []

    for i in range(count):
        # Retry a few times in case of RNG collision
        for _ in range(50):
            pu = wh.random_free_cell(exclude=list(used))
            do = wh.random_free_cell(exclude=list(used) + [pu])
            break
        used.add(pu); used.add(do)

        tasks.append(Task(
            id           = id_offset + i,
            pickup       = pu,
            dropoff      = do,
            created_time = created_at,
        ))
        print(
            f"    Task {id_offset+i:2d}  {pu} → {do}"
            + (f"  (appears at step {created_at})" if created_at else "")
        )

    return tasks


def schedule_dynamic_tasks(
    count:      int,
    wh:         Warehouse,
    occupied:   List[Tuple[int, int]],
    seed:       int,
    first_step: int,
    interval:   int,
    id_offset:  int,
) -> List[Tuple[int, Task]]:
    """
    Create tasks paired with their scheduled arrival timestep.
    Returns a list of (arrival_step, Task).
    """
    events = []
    used   = set(occupied)

    for i in range(count):
        arrival = first_step + i * interval
        task    = generate_tasks(
            1, wh, list(used), seed + i * 13,
            id_offset=id_offset + i,
            created_at=arrival,
        )[0]
        used.add(task.pickup)
        used.add(task.dropoff)
        events.append((arrival, task))

    return events


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    cfg  = CONFIG
    seed = cfg["seed"]

    _banner("Multi-Robot Warehouse Navigation & Task Allocation")

    # ── 1. Warehouse ───────────────────────────────────────────────────────
    _section("1 / 5  —  Building warehouse")
    wh = build_warehouse(cfg)
    print(f"    {wh.summary()}")

    # ── 2. Robots ──────────────────────────────────────────────────────────
    _section(f"2 / 5  —  Spawning {cfg['num_robots']} robots")
    robots   = spawn_robots(cfg["num_robots"], wh, seed)
    occupied = [r.pos for r in robots]

    # ── 3. Static tasks ────────────────────────────────────────────────────
    _section(f"3 / 5  —  Generating {cfg['num_static_tasks']} static tasks")
    static_tasks = generate_tasks(
        cfg["num_static_tasks"], wh, occupied, seed,
        id_offset=0, created_at=0,
    )
    occupied += [t.pickup for t in static_tasks] + [t.dropoff for t in static_tasks]

    # ── 4. Dynamic tasks ───────────────────────────────────────────────────
    _section(f"4 / 5  —  Scheduling {cfg['num_dynamic_tasks']} dynamic tasks")
    dynamic_events = schedule_dynamic_tasks(
        count      = cfg["num_dynamic_tasks"],
        wh         = wh,
        occupied   = occupied,
        seed       = seed,
        first_step = cfg["dyn_first_arrival"],
        interval   = cfg["dyn_interval"],
        id_offset  = len(static_tasks),
    )

    # ── 5. Simulation ──────────────────────────────────────────────────────
    _section(f"5 / 5  —  Running simulation  [{cfg['allocation_method']} allocation]")
    sim = Simulation(
        warehouse         = wh,
        robots            = robots,
        tasks             = static_tasks,
        allocation_method = cfg["allocation_method"],
        dynamic_tasks     = dynamic_events,
        max_steps         = cfg["max_steps"],
        replan_horizon    = cfg["replan_horizon"],
        verbose           = True,
    )
    sim.run()

    # ── Visualisation ──────────────────────────────────────────────────────
    _section("Visualisation")
    viz = WarehouseVisualizer(
        warehouse          = wh,
        robots             = robots,
        tasks              = sim.tasks,      # includes dynamically added tasks
        simulation         = sim,
        animation_interval = cfg["anim_interval_ms"],
        show_trails        = cfg["show_trails"],
        show_paths         = cfg["show_paths"],
    )

    if cfg["show_animation"]:
        print("  Launching animation …  (close the window to continue)")
        viz.animate()

    if cfg["show_metrics"]:
        print("  Showing metric charts …")
        viz.plot_metrics()

    if cfg["show_final_map"]:
        print("  Showing final warehouse map …")
        viz.plot_final_map()


# ===========================================================================
# Helpers
# ===========================================================================

def _banner(title: str) -> None:
    w = 62
    print("\n" + "═" * w)
    print(f"  {title}")
    print("═" * w + "\n")


def _section(name: str) -> None:
    print(f"\n┌─  {name}")


if __name__ == "__main__":
    main()
