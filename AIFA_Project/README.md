# 🤖 Multi-Robot Warehouse Navigation & Task Allocation

> **AI-powered multi-robot coordination** for warehouse pick-and-place automation using A* path planning, collision avoidance, and dynamic task allocation.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.x-00CC00?logo=python&logoColor=white)](https://pygame.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## 📋 Problem Statement

Coordinate a fleet of autonomous warehouse robots to efficiently execute **pick-and-place tasks**. Each task involves navigating a robot from its position to a **pickup location**, collecting an item, and delivering it to a **drop-off location** — all while avoiding collisions with other robots and obstacles.

**Objectives:**
- Minimize total travel distance and task completion time
- Guarantee collision-free multi-robot navigation
- Support dynamic task injection at runtime
- Provide real-time visualization and performance metrics

---

## 🏗️ Architecture

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   main.py    │────▶│  simulation.py  │────▶│  environment.py  │
│  Entry Point │     │  Pygame UI      │     │  Warehouse Grid  │
└──────────────┘     └────────┬────────┘     └──────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ robot_task_alocat.py│
                    │ Task Allocator+FSM │
                    └──┬───────┬─────┬───┘
                       │       │     │
              ┌────────▼──┐ ┌──▼───┐ ┌▼────────────┐
              │navigation │ │state │ │  metrics.py  │
              │   .py     │ │diagram│ │ Performance  │
              │ A* Planner│ │ .py  │ │  Tracking    │
              └───────────┘ └──────┘ └──────────────┘
```

---

## 🚀 Features

### Core Algorithm
- **A\* Search** with time-expanded graph for optimal path planning
- **Weighted A\*** (configurable w>1) for faster search on large grids
- **Multi-waypoint planning** (start → pickup → drop-off in one call)
- **Collision avoidance** via vertex and edge conflict detection
- **Prioritized planning** — robots plan sequentially, respecting reservations

### Task Allocation
- **Pick-and-place lifecycle** — full workflow with pickup and delivery phases
- **Cost-matrix greedy assignment** — nearest-pickup matching for efficiency
- **Dynamic task injection** — add tasks at runtime via keyboard shortcut

### Robot FSM (Finite State Machine)
```
IDLE → MOVING_TO_PICKUP → PICKING_UP → MOVING_TO_DROPOFF → DROPPING_OFF → IDLE
```
Each robot transitions through 5 states with validated transitions and logged history.

### Visualization
- **Dark-themed Pygame UI** with auto-scaling to screen resolution
- **Color-coded robots** — different colors per FSM state with glow halos
- **Planned path trails** — dashed lines showing each robot's route
- **Animated markers** — pulsating pickup/dropoff indicators
- **HUD dashboard** — step count, task stats, robot state cards
- **Legend panel** — explains all visual elements
- **Metrics overlay** — press `M` for live utilization bars

### Performance Metrics
- Per-robot: distance traveled, tasks completed, utilization %
- Global: throughput (tasks/step), average completion time
- **Matplotlib charts** saved automatically on exit

### State Diagrams
- **Mermaid FSM diagrams** generated at startup
- **State snapshots** printed at every 25 timesteps
- Saved to `output/fsm_diagram.md`

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies
```bash
pip install pygame numpy matplotlib
```

---

## 🎮 Usage

### GUI Simulation
```bash
cd AIFA_Project
python main.py
```

### Headless Mode (No GUI)
```bash
python main.py --headless
```

### Run Tests
```bash
python test_system.py
```

### Controls

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume simulation |
| `+` / `-` | Increase / Decrease speed |
| `D` | Inject 3 dynamic tasks |
| `M` | Toggle metrics overlay |
| `Esc` | Quit |

---

## 📁 Project Structure

```
AIFA_Project/
├── main.py               # Entry point with configuration
├── simulation.py          # Premium Pygame visualization
├── environment.py         # Warehouse grid & layout generation
├── navigation.py          # A* path planner with collision avoidance
├── robot_task_alocat.py   # Task allocator with FSM lifecycle
├── state_diagram.py       # Robot FSM & Mermaid diagram generation
├── metrics.py             # Performance tracking & chart plotting
├── test_system.py         # Automated test suite (62 tests)
└── output/                # Generated artifacts
    ├── fsm_diagram.md         # Mermaid state diagrams
    ├── fsm_initial.md         # Initial FSM diagram
    ├── tasks_completed.png    # Cumulative task chart
    ├── robot_utilization.png  # Per-robot utilization
    ├── robot_distances.png    # Distance traveled chart
    └── utilization_over_time.png  # Fleet utilization timeline
```

---

## 🧠 Technical Details

### State Space Representation

| Component | Description |
|-----------|-------------|
| **Initial State** | Robot positions on grid + pending task list |
| **Goal State** | All tasks delivered, robots idle |
| **Actions** | Move (up/down/left/right), wait, pick up, drop off |
| **Transitions** | Each action updates robot position or task status |

### Heuristic Strategy

- **Manhattan Distance** — admissible heuristic for grid-based movement
- **Multi-waypoint sum** — combined distance along pickup → dropoff sequence
- **Time-expanded search** — plans in (position, timestep) space to prevent conflicts

### Collision Avoidance

1. **Vertex conflicts** — no two robots occupy the same cell at the same timestep
2. **Edge conflicts** — no two robots swap positions (cross paths) in one step
3. **Reservation table** — each planned path reserves cells across all future timesteps
4. **Goal holding** — completed robots hold their final position to prevent others from planning through them

---

## ✅ Test Results

```
══════════════════════════════════════════════════
  MULTI-ROBOT WAREHOUSE — AUTOMATED TEST SUITE
══════════════════════════════════════════════════

▸ Warehouse Environment         ✓ (11 tests)
▸ A* Pathfinding                ✓ (8 tests)
▸ Multi-Robot Collision Avoidance ✓ (4 tests)
▸ Robot FSM State Machine       ✓ (8 tests)
▸ Pick-and-Place Lifecycle      ✓ (5 tests)
▸ Multi-Robot Task Completion   ✓ (3 tests)
▸ Dynamic Task Injection        ✓ (2 tests)
▸ Metrics Tracking              ✓ (7 tests)
▸ State Diagram Generation      ✓ (5 tests)
▸ Realistic E2E Simulation      ✓ (4 tests)

  ALL 62 TESTS PASSED ✓
══════════════════════════════════════════════════
```

---

## 📊 Sample Metrics Output

```
══════════════════════════════════════════════════
  SIMULATION METRICS REPORT
══════════════════════════════════════════════════
  Total Steps:            76
  Total Distance:         245 cells
  Tasks Completed:        8
  Avg Completion Time:    39.75 steps
  Throughput:             0.0526 tasks/step
──────────────────────────────────────────────────
  Robot  0: dist=  47  tasks=2  util=98.7%
  Robot  1: dist=  70  tasks=2  util=98.7%
  Robot  2: dist=  70  tasks=2  util=98.7%
  Robot  3: dist=  58  tasks=2  util=98.7%
══════════════════════════════════════════════════
```

---

## 🔮 Future Enhancements

- CBS (Conflict-Based Search) for globally optimal multi-robot plans
- Heterogeneous robot types with varying speeds and capacities
- Priority-based task scheduling with deadlines
- ROS integration for real-world deployment
- Web-based dashboard with live monitoring

---

## 📄 License

This project is released under the MIT License.
