# AI Agent Prompt: Multi-Robot Warehouse Navigation and Task Allocation

---

## AGENT IDENTITY AND OBJECTIVE

You are an expert AI software engineer specializing in multi-agent systems, robotics simulation,
and machine learning. Your task is to fully implement, from scratch, a **Multi-Robot Warehouse
Navigation and Task Allocation** system. The final deliverable is a complete, runnable Python
codebase in a Kaggle Notebook environment (2× T4 GPUs) that demonstrates:

1. A Gymnasium-compatible 2D grid warehouse environment
2. Optimal task allocation using the Hungarian Algorithm
3. Collision-free multi-agent path planning using Conflict-Based Search (CBS) with Space-Time A*
4. A Graph Neural Network (GNN) warm-start that accelerates CBS via imitation learning
5. A visual simulation of robot movement with logged performance metrics

All code must run end-to-end without errors. Produce clean, modular, well-commented Python.

---

## CONSTRAINTS AND ENVIRONMENT

- **Platform:** Kaggle Notebook (Python 3.10, 2× NVIDIA T4 GPUs, ~13 GB RAM)
- **Primary libraries:** `torch`, `torch_geometric`, `scipy`, `numpy`, `matplotlib`,
  `gymnasium`, `wandb`, `pygame` (for visualization), `rware` (optional — can build custom env)
- **Install command to run first:**
  ```
  pip install gymnasium rware cbs-mapf scipy torch torch_geometric stable-baselines3
      wandb pygame matplotlib tqdm
  ```
- **No internet access during execution** (pre-download any models or datasets during setup)
- **GPU usage:** GNN training uses CUDA; CBS and Hungarian run on CPU
- **Target runtime:** Full pipeline (10k episodes of data generation + GNN training) must
  complete within Kaggle's 9-hour session limit
- **Code style:** PEP 8 compliant, type-annotated, modular (one class per file)

---

## PROJECT ARCHITECTURE

Organize the project as follows. Create every file listed:

```
warehouse_robot/
├── main.py                     # Entry point: runs full pipeline
├── config.py                   # All hyperparameters and constants
├── env/
│   ├── __init__.py
│   ├── warehouse_env.py        # Custom Gymnasium MultiAgentWarehouseEnv
│   └── grid_utils.py           # Grid helpers: obstacle map, Manhattan distance
├── planning/
│   ├── __init__.py
│   ├── astar.py                # Space-Time A* low-level planner
│   └── cbs.py                  # Conflict-Based Search high-level coordinator
├── allocation/
│   ├── __init__.py
│   ├── hungarian.py            # Hungarian Algorithm task allocator
│   └── gnn_allocator.py        # PyTorch Geometric GNN warm-start model
├── training/
│   ├── __init__.py
│   └── imitation.py            # Data generation + GNN training loop
├── viz/
│   ├── __init__.py
│   └── renderer.py             # Pygame/Matplotlib grid renderer + path animator
├── metrics/
│   ├── __init__.py
│   └── logger.py               # WandB + CSV metric logging
└── tests/
    ├── test_astar.py
    ├── test_cbs.py
    └── test_hungarian.py
```

---

## MODULE-BY-MODULE SPECIFICATIONS

Implement each module EXACTLY as specified. Do not skip any method or attribute.

---

### MODULE 1: `config.py`

Define a single `Config` dataclass with these fields and default values:

```python
@dataclass
class Config:
    # Environment
    grid_height: int = 16
    grid_width: int = 20
    num_robots: int = 8
    num_tasks: int = 20
    shelf_density: float = 0.30       # Fraction of cells that are shelves/obstacles
    max_timesteps: int = 500          # Max steps per episode before timeout

    # A* / CBS
    time_horizon: int = 100           # Max time dimension for space-time A*
    cbs_max_nodes: int = 5000         # CBS high-level tree node expansion limit

    # Kinodynamic weighting
    mu_traction: float = 0.6          # Traction coefficient
    g_gravity: float = 9.81           # m/s²
    turning_radius: float = 0.3       # metres
    v_max_straight: float = 1.5       # m/s max straight-line speed
    corner_penalty_lambda: float = 0.4 # Weight for corner cell cost inflation

    # GNN
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_learning_rate: float = 1e-3
    gnn_batch_size: int = 64
    gnn_epochs: int = 50
    gnn_train_episodes: int = 10000   # CBS solutions to generate as training labels
    gnn_top_k: int = 3                # Top-k assignment candidates GNN proposes to CBS

    # Logging
    wandb_project: str = "multi-robot-warehouse"
    log_every_n: int = 50
    checkpoint_path: str = "./checkpoints/gnn_best.pt"

    # Seed
    seed: int = 42
```

---

### MODULE 2: `env/warehouse_env.py`

Implement `MultiAgentWarehouseEnv(gym.Env)` with these exact specifications:

**Grid representation:**
- `np.ndarray` of shape `(H, W)` with dtype `int8`
- Cell values: `0 = free`, `1 = obstacle/shelf`, `2 = robot`, `3 = pickup`, `4 = dropoff`
- Shelves are placed in a realistic aisle pattern: alternate columns of 2-cell-wide shelf blocks
  separated by 1-cell-wide aisles, with 2-cell-wide horizontal corridors every 4 rows

**State:**
- `robot_positions`: `List[Tuple[int, int]]` — (row, col) for each robot
- `robot_tasks`: `List[Optional[int]]` — task index assigned to each robot (None if idle)
- `task_list`: `List[Tuple[Tuple, Tuple]]` — list of (pickup_pos, dropoff_pos) tuples
- `task_status`: `List[str]` — "pending", "in_progress_pickup", "in_progress_dropoff", "done"
- `robot_carrying`: `List[bool]` — whether robot currently holds an item

**Methods to implement:**
```python
def reset(self, seed=None) -> Tuple[Dict, Dict]:
    """Reset env. Randomly place robots in free cells. Randomly assign pickup/dropoff
    locations in free cells (not obstacle). Return (obs_dict, info_dict)."""

def step(self, actions: Dict[int, Tuple[int,int]]) -> Tuple[Dict, Dict, bool, bool, Dict]:
    """Actions is a dict mapping robot_id -> (new_row, new_col) next position.
    Validate moves (no obstacle, no out-of-bounds). Update task status when robot
    reaches pickup (set carrying=True) or dropoff (task done). Return
    (obs, rewards, terminated, truncated, info) where info contains
    {'makespan': int, 'tasks_completed': int, 'total_distance': float,
     'collisions': int}."""

def get_observation(self) -> Dict:
    """Return flat dict: {'robot_positions': np.array shape (k,2),
    'task_pickups': np.array shape (m,2), 'task_dropoffs': np.array shape (m,2),
    'task_status': np.array shape (m,), 'robot_carrying': np.array shape (k,)}"""

def get_cost_matrix(self) -> np.ndarray:
    """Return (k x m) cost matrix C[i][j] = manhattan(robot_i, pickup_j)
    + manhattan(pickup_j, dropoff_j) for pending tasks only."""

def render(self, mode='rgb_array') -> np.ndarray:
    """Return RGB numpy array of current grid state for visualization."""

def _generate_warehouse_layout(self) -> np.ndarray:
    """Generate realistic aisle-based warehouse grid with shelf_density fraction
    of cells occupied. Ensure connectivity: all free cells must be reachable."""

def _check_connectivity(self, grid: np.ndarray) -> bool:
    """BFS/flood-fill to verify all free cells are connected. Return True if connected."""
```

**Reward structure:**
- `+10` per task completed (dropoff reached)
- `-0.1` per timestep (time penalty encouraging efficiency)
- `-5` per collision (two robots in same cell)
- `+50` bonus when ALL tasks completed in episode

---

### MODULE 3: `planning/astar.py`

Implement `SpaceTimeAStar` with this exact interface:

```python
class SpaceTimeAStar:
    def __init__(self, grid: np.ndarray, config: Config):
        """Store grid and config. Precompute corner cells for kinodynamic weighting."""

    def plan(self,
             start: Tuple[int, int],
             goal: Tuple[int, int],
             constraints: List[Tuple[int, int, int]],  # (row, col, time) forbidden nodes
             start_time: int = 0) -> Optional[List[Tuple[int, int, int]]]:
        """
        Space-Time A* search.

        Node: (row, col, time)
        Heuristic: Manhattan distance to goal (admissible, ignores time)
        Edge cost: base 1.0 + corner_penalty_lambda if cell is a corner cell
        Constraints: skip any node (row, col, t) in constraints set

        Returns path as list of (row, col, time) tuples, or None if no path found.
        Path must start at (start[0], start[1], start_time) and end at
        (goal[0], goal[1], T) for some T.

        Movements: up, down, left, right, wait-in-place (5 actions total).
        """

    def _heuristic(self, pos: Tuple[int,int], goal: Tuple[int,int]) -> float:
        """Manhattan distance."""

    def _get_cell_cost(self, row: int, col: int) -> float:
        """Return 1.0 + corner_penalty_lambda if cell is adjacent to ≥2 obstacle cells
        (i.e., a corner), else 1.0. This implements kinodynamic weighting."""

    def _get_neighbors(self, row: int, col: int, t: int) -> List[Tuple[int,int,int]]:
        """Return valid (new_row, new_col, t+1) neighbors including wait action."""
```

**Priority queue:** Use `heapq` with entries `(f_cost, g_cost, node)`.
**Tie-breaking:** Break ties on `g_cost` (prefer longer paths, reaches goal faster).
**Constraint check:** Before expanding a node `(r, c, t)`, check it is not in the
constraints set AND check for edge conflicts (robot swapping positions with another).

---

### MODULE 4: `planning/cbs.py`

Implement the full Conflict-Based Search algorithm:

```python
@dataclass
class CBSNode:
    constraints: Dict[int, List[Tuple[int,int,int]]]  # robot_id -> [(r,c,t)]
    paths: Dict[int, List[Tuple[int,int,int]]]         # robot_id -> path
    cost: int                                           # sum of path lengths

    def __lt__(self, other): return self.cost < other.cost

class ConflictBasedSearch:
    def __init__(self, grid: np.ndarray, config: Config):
        self.astar = SpaceTimeAStar(grid, config)
        self.config = config
        self.nodes_expanded = 0   # Track for metrics

    def plan(self,
             starts: Dict[int, Tuple[int,int]],
             goals: Dict[int, Tuple[int,int]]) -> Optional[Dict[int, List]]:
        """
        Main CBS entry point.

        HIGH-LEVEL ALGORITHM:
        1. Create root node: no constraints, run low-level A* for each robot independently
        2. Push root onto min-heap ordered by sum-of-costs
        3. Pop lowest-cost node N
        4. Find FIRST conflict among all robot paths in N
        5. If no conflict → return N.paths (solution found)
        6. For each agent involved in the conflict:
           a. Create child node N' = copy of N
           b. Add constraint (agent, conflict_cell, conflict_time) to N'.constraints[agent]
           c. Replan ONLY the constrained agent using Space-Time A* with updated constraints
           d. If replanning succeeds, push N' onto heap
        7. Repeat until solution found or heap empty or node limit reached

        Returns dict mapping robot_id -> list of (row, col, time) tuples.
        Returns None if no solution within cbs_max_nodes expansions.
        """

    def _find_first_conflict(self,
                              paths: Dict[int, List]) -> Optional[Dict]:
        """
        Scan all pairs of robot paths for conflicts.
        Two conflict types:
        1. VERTEX conflict: robots i and j both at (r,c) at time t
        2. EDGE conflict: robot i moves (r1,c1)->(r2,c2) and robot j moves (r2,c2)->(r1,c1)
           at the same time (position swap)

        Return dict: {'type': 'vertex'|'edge', 'agents': [i,j],
                      'cell': (r,c), 'time': t} or None if no conflict.
        Pad shorter paths by repeating last position (robot waits at goal).
        """

    def _compute_sum_of_costs(self, paths: Dict[int, List]) -> int:
        """Sum of all path lengths (number of timesteps including start)."""
```

---

### MODULE 5: `allocation/hungarian.py`

```python
class HungarianAllocator:
    def __init__(self):
        pass

    def assign(self,
               robot_positions: List[Tuple[int,int]],
               tasks: List[Tuple[Tuple,Tuple]],
               active_robots: Optional[List[int]] = None) -> Dict[int, int]:
        """
        Build cost matrix C[i][j] = manhattan(robot_i, pickup_j) + manhattan(pickup_j, dropoff_j)
        for all idle robots i and pending tasks j.

        Use scipy.optimize.linear_sum_assignment to find optimal assignment.
        Handle rectangular matrices (more tasks than robots or vice versa).

        If active_robots is provided, only assign those robot indices.

        Returns dict mapping robot_id -> task_id.
        """

    def reassign_dynamic(self,
                          robot_positions: List[Tuple[int,int]],
                          tasks: List[Tuple[Tuple,Tuple]],
                          task_status: List[str],
                          robot_tasks: List[Optional[int]]) -> Dict[int, int]:
        """
        Dynamic reassignment: only reassign IDLE robots to PENDING tasks.
        Robots currently executing tasks are not reassigned.
        Returns updated assignment dict for idle robots only.
        """

    @staticmethod
    def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])
```

---

### MODULE 6: `allocation/gnn_allocator.py`

Implement the GNN using PyTorch Geometric:

```python
class WarehouseHeteroGraph:
    """Converts warehouse state to PyTorch Geometric HeteroData object."""

    @staticmethod
    def build(robot_positions: np.ndarray,
              task_pickups: np.ndarray,
              task_dropoffs: np.ndarray,
              task_status: np.ndarray) -> HeteroData:
        """
        Node features:
          robot: [row/H, col/W, is_idle (0/1), dist_to_nearest_task/max_dist] — shape (k, 4)
          task:  [pickup_r/H, pickup_c/W, dropoff_r/H, dropoff_c/W, status/4] — shape (m, 5)

        Edges: fully connected bipartite (robot -> task and task -> robot)
        Edge features: [manhattan_distance / max_distance] — shape (k*m, 1)

        Normalize all spatial features by grid dimensions.
        Only include PENDING tasks (filter by task_status == 0).
        """

class TaskAllocGNN(torch.nn.Module):
    """
    Heterogeneous GNN predicting assignment scores for robot-task pairs.
    Architecture: HeteroConv with SAGEConv layers → MLP head → score per (robot, task) pair
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        # Input projections
        self.robot_proj = nn.Linear(4, hidden_dim)
        self.task_proj = nn.Linear(5, hidden_dim)

        # HeteroConv layers
        self.convs = nn.ModuleList([
            HeteroConv({
                ('robot', 'to', 'task'): SAGEConv((-1, -1), hidden_dim),
                ('task', 'to', 'robot'): SAGEConv((-1, -1), hidden_dim),
            }) for _ in range(num_layers)
        ])

        # Scoring head: concatenate robot + task embeddings → score
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Returns score tensor of shape (num_robots, num_tasks).
        Higher score = better assignment.

        Steps:
        1. Project robot and task node features to hidden_dim
        2. Apply num_layers of HeteroConv with ReLU activation
        3. For each (robot_i, task_j) pair, concatenate their embeddings
        4. Pass through score_head
        5. Return score matrix reshaped to (num_robots, num_tasks)
        """

class GNNAllocator:
    """Wraps TaskAllocGNN for inference: proposes top-k assignments."""

    def __init__(self, model: TaskAllocGNN, config: Config):
        self.model = model
        self.config = config

    def propose_assignments(self,
                             robot_positions: np.ndarray,
                             tasks: List,
                             task_status: np.ndarray,
                             top_k: int = 3) -> List[Dict[int, int]]:
        """
        Build graph → run forward pass → extract top-k assignment permutations
        using greedy decoding on score matrix.

        Returns list of top_k assignment dicts (robot_id -> task_id),
        sorted by total score descending.
        These are candidates that CBS will verify and choose from.
        """
```

---

### MODULE 7: `training/imitation.py`

```python
class ImitationLearningTrainer:
    """
    Generates CBS-optimal assignment labels, then trains GNN via supervised learning.
    """

    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.env = MultiAgentWarehouseEnv(config)
        self.cbs = ConflictBasedSearch(self.env.grid, config)
        self.hungarian = HungarianAllocator()
        self.device = device

    def generate_dataset(self, num_episodes: int) -> List[Tuple[HeteroData, torch.Tensor]]:
        """
        For each episode:
        1. Reset env to get robot positions and tasks
        2. Run HungarianAllocator to get optimal assignment
        3. Build HeteroData graph from state
        4. Create label tensor: one-hot assignment matrix of shape (k, m)
           where label[i][j] = 1 if robot i is assigned to task j

        Returns list of (graph, label) pairs for GNN training.
        Log progress every 500 episodes.
        """

    def train(self,
              model: TaskAllocGNN,
              dataset: List[Tuple[HeteroData, torch.Tensor]],
              save_path: str) -> Dict[str, List[float]]:
        """
        Training loop:
        - Loss: Binary Cross-Entropy (BCEWithLogitsLoss) on assignment scores
        - Optimizer: AdamW with lr from config, weight_decay=1e-4
        - Scheduler: CosineAnnealingLR
        - Batch: DataLoader with config.gnn_batch_size
        - Early stopping: stop if val loss doesn't improve for 5 epochs
        - Save best model checkpoint to save_path
        - Return dict with 'train_loss' and 'val_loss' lists for plotting

        Use 80/20 train/val split.
        Log epoch metrics to WandB.
        """
```

---

### MODULE 8: `viz/renderer.py`

```python
class WarehouseRenderer:
    """
    Renders warehouse state using matplotlib (for Kaggle — no display needed).
    Produces both static frames and animated GIFs.
    """

    # Color scheme
    COLORS = {
        'free':     '#F7F6F2',
        'obstacle': '#28251D',
        'robot':    '#01696F',
        'pickup':   '#DA7101',
        'dropoff':  '#437A22',
        'path':     '#4F98A3',
        'robot_carrying': '#A12C7B',
    }

    def __init__(self, config: Config):
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

    def render_frame(self,
                     grid: np.ndarray,
                     robot_positions: List[Tuple],
                     paths: Dict[int, List],
                     tasks: List,
                     task_status: List[str],
                     robot_carrying: List[bool],
                     timestep: int,
                     metrics: Dict) -> np.ndarray:
        """
        Draw current state:
        - Grid cells colored by type (free/obstacle/pickup/dropoff)
        - Robots as filled circles (teal if not carrying, purple if carrying)
        - Robot paths as faint dotted lines
        - Robot ID labels on robots
        - Task pickup (orange dot) and dropoff (green dot) markers
        - Title: "Timestep {t} | Tasks Done: {x}/{total} | Collisions: {c}"
        - Metrics bar at bottom: makespan, sum-of-costs, CBS nodes expanded

        Returns RGB numpy array.
        """

    def save_animation(self,
                       frames: List[np.ndarray],
                       output_path: str = 'warehouse_sim.gif',
                       fps: int = 5):
        """Save list of RGB frames as animated GIF using matplotlib animation."""

    def plot_training_curves(self,
                              history: Dict[str, List[float]],
                              output_path: str = 'training_curves.png'):
        """
        Two-panel plot:
        Left: train loss vs val loss over epochs
        Right: CBS nodes expanded (baseline Hungarian vs GNN warm-start) over episodes

        Use the Nexus teal (#01696F) and orange (#DA7101) as line colors.
        Save to output_path.
        """

    def plot_ablation_table(self,
                             results: Dict[str, Dict],
                             output_path: str = 'ablation_results.png'):
        """
        Render a styled matplotlib table comparing:
        Rows: [Greedy A*, Prioritized A*, CBS+Greedy, CBS+Hungarian, CBS+Hungarian+GNN]
        Cols: [Method, Makespan, Sum-of-Costs, Collisions, Alloc Time (ms)]
        Color-code best values in each column with teal highlight.
        """
```

---

### MODULE 9: `metrics/logger.py`

```python
class MetricsLogger:
    def __init__(self, config: Config, use_wandb: bool = True):
        """Initialize WandB run and local CSV log."""

    def log_episode(self, episode: int, metrics: Dict):
        """Log episode-level metrics: makespan, sum_of_costs, tasks_completed,
        collisions, cbs_nodes_expanded, allocation_time_ms, method_name."""

    def log_training(self, epoch: int, train_loss: float, val_loss: float):
        """Log GNN training metrics."""

    def summary(self) -> pd.DataFrame:
        """Return all logged metrics as a DataFrame. Save to metrics_log.csv."""

    def close(self):
        """Finalize WandB run."""
```

---

### MODULE 10: `main.py`

The main script must execute the full pipeline in this exact order:

```python
def main():
    config = Config()
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = MetricsLogger(config)

    print("=== PHASE 1: Environment Validation ===")
    # Create env, run 5 random episodes using ONLY Hungarian + CBS (no GNN)
    # Verify zero collisions, all tasks complete, log baseline metrics

    print("=== PHASE 2: Ablation Baselines ===")
    # Run 100 episodes each for:
    # (a) Greedy A* + nearest-robot assignment
    # (b) Prioritized A* + greedy assignment
    # (c) CBS + greedy assignment
    # (d) CBS + Hungarian
    # Log makespan, SoC, collisions, alloc_time for each method

    print("=== PHASE 3: GNN Training ===")
    # Generate dataset: config.gnn_train_episodes CBS-labeled scenarios
    # Train TaskAllocGNN with ImitationLearningTrainer
    # Save best checkpoint
    # Plot training curves

    print("=== PHASE 4: GNN Evaluation ===")
    # Load best checkpoint
    # Run 100 episodes with CBS + Hungarian + GNN warm-start
    # Compare CBS nodes expanded vs Phase 2(d) baseline
    # Log all metrics

    print("=== PHASE 5: Visualization ===")
    # Run 1 full episode with GNN pipeline
    # Capture frames every timestep
    # Save animated GIF: warehouse_sim.gif
    # Save ablation table: ablation_results.png
    # Save training curves: training_curves.png

    print("=== PHASE 6: Final Report ===")
    # Print formatted table of ablation results
    # Print: GNN speedup ratio (CBS nodes baseline / CBS nodes with GNN)
    # Save complete metrics_log.csv

    logger.close()
```

---

## ALGORITHM PSEUDOCODE REFERENCE

Use these exact algorithms. Do not deviate.

### Space-Time A* (Low-Level Planner)

```
function SpaceTimeAStar(start, goal, constraints, grid, config):
    open_heap = [(h(start,goal), 0, (start.r, start.c, 0))]
    came_from = {}
    g_score = {(start.r, start.c, 0): 0}
    corner_cells = precompute_corners(grid)

    while open_heap not empty:
        f, g, node = heappop(open_heap)
        r, c, t = node

        if (r,c) == goal:
            return reconstruct_path(came_from, node)

        if t >= config.time_horizon:
            continue

        for each neighbor (nr, nc, nt) in get_neighbors(r,c,t):
            if (nr, nc, nt) in constraints:
                continue
            if edge_swap_conflict(r,c,nr,nc,t,constraints):
                continue

            cell_cost = 1.0 + config.corner_penalty_lambda * (nr,nc in corner_cells)
            tentative_g = g + cell_cost

            if (nr,nc,nt) not in g_score or tentative_g < g_score[(nr,nc,nt)]:
                g_score[(nr,nc,nt)] = tentative_g
                f = tentative_g + h((nr,nc), goal)
                heappush(open_heap, (f, tentative_g, (nr,nc,nt)))
                came_from[(nr,nc,nt)] = node

    return None  # No path found
```

### CBS (High-Level Planner)

```
function CBS(starts, goals, grid, config):
    root = CBSNode(
        constraints = {robot_id: [] for each robot},
        paths = {id: SpaceTimeAStar(starts[id], goals[id], []) for each robot},
        cost = sum_of_path_lengths(paths)
    )
    open_heap = [root]
    nodes_expanded = 0

    while open_heap not empty:
        N = heappop(open_heap)
        nodes_expanded += 1

        if nodes_expanded > config.cbs_max_nodes:
            return best_partial_solution(N)  # Return lowest-cost partial solution

        conflict = find_first_conflict(N.paths)

        if conflict is None:
            return N.paths  # SOLUTION FOUND

        for agent in conflict.agents:
            N_prime = deep_copy(N)
            if conflict.type == 'vertex':
                N_prime.constraints[agent].append((conflict.cell.r, conflict.cell.c, conflict.time))
            elif conflict.type == 'edge':
                # Add constraint preventing agent from entering conflict cell at conflict time
                N_prime.constraints[agent].append((conflict.cell.r, conflict.cell.c, conflict.time))

            new_path = SpaceTimeAStar(starts[agent], goals[agent], N_prime.constraints[agent])

            if new_path is not None:
                N_prime.paths[agent] = new_path
                N_prime.cost = sum_of_path_lengths(N_prime.paths)
                heappush(open_heap, N_prime)

    return None  # No solution
```

### Hungarian Algorithm Wrapper

```
function HungarianAssign(robot_positions, tasks):
    k = len(robot_positions)
    m = len(tasks)
    C = zeros(k, m)

    for i in range(k):
        for j in range(m):
            C[i][j] = manhattan(robot_positions[i], tasks[j].pickup)
                     + manhattan(tasks[j].pickup, tasks[j].dropoff)

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(C)
    return {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
```

---

## KINODYNAMIC WEIGHTING SPECIFICATION

The maximum safe speed at a corner cell is derived from vehicle dynamics:

```
v_max_turn = sqrt(mu * g * r)   # mu=0.6, g=9.81, r=0.3 → v_max_turn ≈ 1.33 m/s
lambda = (v_max_straight - v_max_turn) / v_max_straight   # ≈ 0.11 with above values
cell_cost_corner = 1.0 + lambda
cell_cost_straight = 1.0
```

A corner cell is defined as any free cell with ≥2 orthogonally adjacent obstacle cells.
This penalizes paths through tight corners, biasing the planner toward open aisle traversal.

---

## GNN TRAINING SPECIFICATION

### Data Generation
- For each episode: reset env → run Hungarian → record (graph, assignment_label)
- Assignment label: binary matrix `Y` of shape `(k, m)` where `Y[i][j]=1` iff robot `i`
  assigned to task `j` by Hungarian Algorithm
- Dataset split: 80% train, 20% validation (stratified by num_tasks)

### Loss Function
```python
loss = F.binary_cross_entropy_with_logits(
    scores.view(-1),       # Flatten (k*m,)
    labels.view(-1).float()
)
```

### Inference (Top-K Greedy Decoding)
```
Given score matrix S of shape (k, m):
  assignments = []
  S_copy = S.clone()
  for _ in range(min(k,m)):
      i, j = argmax(S_copy)    # Find highest-scoring robot-task pair
      assignments.append((i,j))
      S_copy[i, :] = -inf      # Robot i is assigned
      S_copy[:, j] = -inf      # Task j is assigned
  return assignments as dict
```

---

## EVALUATION METRICS SPECIFICATION

Log these metrics for EVERY episode, for EVERY method:

| Metric | Definition | Unit |
|--------|-----------|------|
| `makespan` | Timestep when last task completes | steps |
| `sum_of_costs` | Total path length across all robots | steps |
| `collisions` | Count of robot-robot cell overlaps | count |
| `tasks_completed` | Number of tasks finished | count |
| `cbs_nodes_expanded` | High-level CBS tree nodes expanded | count |
| `allocation_time_ms` | Wall-clock time for task assignment | ms |
| `total_distance_cells` | Sum of cells traversed by all robots | cells |
| `episode_timeout` | Whether max_timesteps was reached | bool |

**Ablation table target values** (your implementation should aim to reproduce these):

| Method | Makespan | SoC | Collisions | Alloc (ms) |
|--------|---------|-----|-----------|-----------|
| Greedy A* + Nearest Robot | ~87 | ~412 | ~14 | <1 |
| Prioritized A* + Greedy | ~74 | ~389 | ~3 | <1 |
| CBS + Greedy Assign | ~68 | ~341 | 0 | ~38 |
| CBS + Hungarian | ~61 | ~298 | 0 | ~41 |
| CBS + Hungarian + GNN | ~54 | ~271 | 0 | ~18 |

---

## TESTING SPECIFICATION

Write pytest tests for these cases. All tests must pass before running main.py.

### `tests/test_astar.py`
```
test_simple_path: Single robot, no obstacles, 5x5 grid. Verify path found, no cycles.
test_obstacle_avoidance: L-shaped obstacle. Verify path goes around.
test_constraint_respected: Add constraint at (2,2,3). Verify path avoids it.
test_no_path: Completely blocked goal. Verify returns None.
test_wait_action: Two constraints at goal path. Verify robot waits.
```

### `tests/test_cbs.py`
```
test_no_conflict: Two robots on non-intersecting paths. Verify solution = individual A* paths.
test_vertex_conflict: Two robots heading for same cell. Verify CBS resolves with 0 collisions.
test_edge_conflict: Two robots swapping positions. Verify CBS resolves edge conflict.
test_optimality: Verify CBS solution cost <= prioritized A* solution cost on same instance.
```

### `tests/test_hungarian.py`
```
test_square_matrix: 3 robots, 3 tasks. Verify assignment is globally optimal.
test_rectangular_more_tasks: 2 robots, 5 tasks. Verify only 2 tasks assigned.
test_rectangular_more_robots: 4 robots, 2 tasks. Verify only 2 robots get tasks.
test_dynamic_reassign: Mix of idle and busy robots. Verify only idle robots reassigned.
```

---

## OUTPUT FILES EXPECTED

After `main.py` completes, these files must exist in the working directory:

```
warehouse_sim.gif           # Animated simulation (≥30 frames)
training_curves.png         # GNN train/val loss + CBS nodes reduction
ablation_results.png        # Styled comparison table
metrics_log.csv             # Full metrics for all methods and episodes
checkpoints/gnn_best.pt     # Best GNN checkpoint (val loss)
```

---

## EXECUTION ORDER

Run files in this exact order:

```bash
# 1. Install dependencies
pip install gymnasium rware cbs-mapf scipy torch torch_geometric
    stable-baselines3 wandb pygame matplotlib tqdm pytest

# 2. Run tests first — fix all failures before proceeding
pytest warehouse_robot/tests/ -v

# 3. Run full pipeline
cd warehouse_robot && python main.py
```

Total expected runtime on Kaggle T4: ~2–3 hours (data gen: 45 min, training: 60 min,
evaluation: 30 min, visualization: 10 min).

---

## QUALITY STANDARDS

- **Zero errors:** `main.py` must run start to finish without exceptions
- **Zero collisions:** CBS + Hungarian method must achieve 0 collisions across 100 eval episodes
- **Reproducible:** Setting `config.seed = 42` must produce identical results across runs
- **Type annotations:** All function signatures must have complete type hints
- **Docstrings:** Every class and public method must have a one-line docstring
- **No magic numbers:** All constants go in `config.py`, referenced by name
- **Memory safety:** CBS node heap must be pruned if it exceeds 10,000 nodes
  (discard nodes with cost > 1.5× best found cost so far)
- **GPU utilization:** GNN training dataloader must use `pin_memory=True` and
  `num_workers=2` when CUDA is available
