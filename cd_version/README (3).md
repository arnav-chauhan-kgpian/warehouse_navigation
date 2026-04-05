# Multi-Robot Warehouse Navigation & Task Allocation

A complete Python simulation of autonomous warehouse robots that coordinate
pick-and-place tasks, plan collision-free paths, and visualise the results
in real time.

---

## Features

| Feature | Details |
|---|---|
| **Path planning** | Space-Time A\* (collision-free, optimal) + standard A\* fallback |
| **Collision avoidance** | Prioritised planning with a shared reservation table; swap-conflict detection |
| **Task allocation** | Greedy (nearest robot) or Hungarian (globally optimal via scipy) |
| **Dynamic tasks** | Tasks that arrive at configurable timesteps during the simulation |
| **Visualisation** | Live Matplotlib animation + metric bar-charts + final heatmap |
| **Metrics** | Per-robot distance & task count, task makespan, throughput |

---

## Project Structure

```
warehouse_robots/
├── main.py            ← Entry point & scenario configuration
├── warehouse.py       ← Warehouse grid generator
├── astar.py           ← A* and Space-Time A* algorithms
├── robot.py           ← Robot and Task data classes
├── task_allocator.py  ← Greedy and Hungarian allocation
├── simulation.py      ← Simulation engine (reservation table, step loop)
├── visualizer.py      ← Matplotlib animation and analysis plots
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install matplotlib numpy scipy

# 2. Run the simulation
python main.py
```

---

## Configuration

All parameters live in the `CONFIG` dict at the top of `main.py`:

```python
CONFIG = {
    # Warehouse size
    "warehouse_rows": 22,
    "warehouse_cols": 24,
    "shelf_height":    2,
    "shelf_width":     3,
    "aisle_gap":       2,

    # Fleet
    "num_robots":        4,

    # Tasks
    "num_static_tasks":  8,
    "num_dynamic_tasks": 4,
    "dyn_first_arrival": 40,   # step when first dynamic task arrives
    "dyn_interval":      30,   # steps between dynamic arrivals

    # Planning
    "allocation_method": "greedy",   # or "hungarian"
    "max_steps":         700,
    "replan_horizon":    130,

    # Visualisation
    "anim_interval_ms":  120,
    "show_animation":    True,
    "show_metrics":      True,
    "show_final_map":    True,

    "seed": 42,
}
```

---

## Algorithm Details

### Space-Time A\*

Standard A\* operates on a 2-D grid.  Space-Time A\* extends the search into
a 3-D space `(row, col, timestep)`, so the planner can schedule *when* a robot
occupies each cell.

**State space**: `(row, col, t)`  
**Actions**: move up / down / left / right, or *wait* (cost = 1 each)  
**Heuristic**: Manhattan distance to goal (admissible → optimal paths)  
**Constraints**:
  - A cell `(r, c, t)` is blocked if another robot reserved it at time `t`.
  - Swap conflicts are detected: two robots cannot pass through each other.

### Prioritised Planning

Robots are planned sequentially (highest-priority first).  Each robot's
planned path is written into the shared **reservation table** before the next
robot is planned, so later robots automatically route around earlier ones.

**Reservation table**: `dict[(row, col, timestep)] → robot_id`

When a robot completes a leg (pickup or dropoff), its old reservations are
cleared and the new path is re-planned under the updated table.

### Task Allocation

| Method | Complexity | Optimality |
|---|---|---|
| Greedy | O(R·T) | Locally optimal per task |
| Hungarian | O(n³) | Globally optimal matching |

Both methods work on idle robots vs. pending tasks.  If scipy is absent,
Hungarian automatically degrades to greedy.

---

## Mechanical Engineering Integration

The simulation accounts for:

- **Kinematics**: Grid moves correspond to discrete robot displacements;
  diagonal moves are excluded (non-holonomic constraint simplification).
- **Collision geometry**: Each robot occupies exactly one grid cell; the
  reservation table enforces hard exclusion zones.
- **Motion constraints**: The wait action models robot deceleration /
  holding at a junction — analogous to speed control in real AGVs.
- **Extensibility**: Replace the grid-based model with a continuous
  configuration space and integrate ROS Nav2 for real-hardware deployment.

---

## Sample Output

```
══════════════════════════════════════════════════════════════
  Multi-Robot Warehouse Navigation & Task Allocation
══════════════════════════════════════════════════════════════

┌─  1 / 5  —  Building warehouse
    Warehouse 22×24  |  Free: 336  |  Shelves: 192  |  Occupancy: 36.4%

┌─  2 / 5  —  Spawning 4 robots
    Robot  0  spawned at  row=16, col=22
    Robot  1  spawned at  row=17, col= 3

┌─  5 / 5  —  Running simulation  [greedy allocation]
  Step    0  |  Done  0/12  |  Active robots 4/4
  Step   25  |  Done  3/12  |  Active robots 4/4
  Step   50  |  Done  7/12  |  Active robots 3/4
  ✓ All tasks completed at step 134!

╔══════════════════════════════════════╗
║  Completed Tasks      : 12           ║
║  Avg Task Makespan    :  38.2 steps  ║
║  Throughput           :   8.96 tasks/100s ║
╚══════════════════════════════════════╝
```
