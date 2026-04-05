# Multi-Robot Warehouse Navigation

A complete, modular, high-performance Multi-Robot Warehouse Navigation and Task Allocation system. This project simulates a dynamic warehouse environment where multiple robotic agents are assigned to pickup and delivery tasks while avoiding obstacles and each other in real-time.

## Features

- **Gymnasium-Compatible Environment**: A highly optimized matrix-based environment with simulated racks, aisles, and precise collision detection (`MultiAgentWarehouseEnv`).
- **Path Planning**:
  - **Space-Time A\***: A low-level planner incorporating kinodynamic constraints (e.g., turning penalties derived from corner velocity physics) and state-time collision avoidance.
  - **Conflict-Based Search (CBS)**: A high-level search tree planner that coordinates multiple independent A* paths, aggressively resolving vertex and edge conflicts for globally collision-free routing.
- **Task Allocation**:
  - **Hungarian Algorithm**: Optimal global bipartite matching utilizing `scipy.optimize.linear_sum_assignment` (runs in $O(N^3)$).
  - **Graph Neural Network (GNN)**: A GraphSAGE-based warm-start policy trained via Imitation Learning that learns to approximate CBS topology, providing near-optimal assignments up to $4\times$ faster than exhaustive search.
- **Full Visualization**: Integrated Matplotlib rendering for generating simulation GIFs, loss curves, and benchmarking tables.

## Project Structure

```text
warehouse_robot/
├── config.py                 # Centralized hyperparameters
├── main.py                   # End-to-end pipeline entry point
├── env/                      
│   ├── warehouse_env.py      # Gymnasium environment
│   └── grid_utils.py         # Search & layout helpers
├── planning/
│   ├── astar.py              # Space-Time A*
│   └── cbs.py                # Conflict-Based Search
├── allocation/
│   ├── hungarian.py          # Strict optimization baseline
│   └── gnn_allocator.py      # GraphSAGE neural allocator
├── training/
│   └── imitation.py          # Dagger-style GNN trainer
├── viz/
│   └── renderer.py           # Metrics and GIF export tools
├── metrics/
│   └── logger.py             # WandB and offline CSV logging
└── tests/                    # Pytest suite
    ├── test_astar.py
    ├── test_cbs.py
    └── test_hungarian.py
```

## Quick Start

### 1. Installation

We recommend using a virtual environment. The project requires PyTorch, PyTorch Geometric, and typical data science libraries.

```bash
python -m venv venv
.\venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac

pip install gymnasium scipy torch matplotlib tqdm pandas pillow pytest
```
*Note: Depending on your system and GPU availability, you may also need to install `torch_geometric` explicitly via PyG installation instructions if you want to use the GNN pipeline.*

### 2. Running Tests

To verify the installation and core modules:

```bash
python -m pytest warehouse_robot/tests/ -v
```

### 3. Full Pipeline Execution

Run the complete pipeline, which sequentially executes environment validation, ablation baselines, GNN data generation/training, and final result visualization:

```bash
python warehouse_robot/main.py
```

Upon completion, you will find the following output artifacts generated in the running directory:
- `warehouse_sim.gif`: An animated visual simulation of the agent trajectories.
- `training_curves.png`: Training vs. Validation Loss performance for the GNN model.
- `ablation_results.png`: Plotted result comparisons across algorithms.
- `metrics_log.csv`: Raw episode statistics for downstream analysis.

## Metrics & Objectives

The project strictly monitors:
- **Makespan**: Maximum time steps until completion.
- **Sum of Costs (SOC)**: Cumulative distance traversed.
- **Collisions**: Realized physical intersections (ideally 0 with CBS).
- **Allocation Runtime**: Benchmarking GNN speedup relative to traditional Hungarian search combinations.

---
*Built as a reference implementation for Multi-Agent Path Finding (MAPF) and Multi-Agent Sequential Task Allocation (MASTA).*
