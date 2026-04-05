"""config.py — All hyperparameters and constants for the warehouse navigation system."""

from dataclasses import dataclass, field


@dataclass
class Config:
    """Central configuration dataclass for the multi-robot warehouse system."""

    # ── Environment ───────────────────────────────────────────────────────────
    grid_height: int = 16
    grid_width: int = 20
    num_robots: int = 8
    num_tasks: int = 20
    shelf_density: float = 0.30       # Fraction of cells that are shelves/obstacles
    max_timesteps: int = 500          # Max steps per episode before timeout

    # ── A* / CBS ──────────────────────────────────────────────────────────────
    time_horizon: int = 100           # Max time dimension for space-time A*
    cbs_max_nodes: int = 5000         # CBS high-level tree node expansion limit

    # ── Kinodynamic weighting ─────────────────────────────────────────────────
    mu_traction: float = 0.6          # Traction coefficient
    g_gravity: float = 9.81           # m/s²
    turning_radius: float = 0.3       # metres
    v_max_straight: float = 1.5       # m/s max straight-line speed
    corner_penalty_lambda: float = 0.4  # Weight for corner cell cost inflation

    # ── GNN ───────────────────────────────────────────────────────────────────
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 3
    gnn_learning_rate: float = 1e-3
    gnn_batch_size: int = 64
    gnn_epochs: int = 50
    gnn_train_episodes: int = 10000   # CBS solutions to generate as training labels
    gnn_top_k: int = 3                # Top-k assignment candidates GNN proposes to CBS

    # ── Logging ───────────────────────────────────────────────────────────────
    wandb_project: str = "multi-robot-warehouse"
    log_every_n: int = 50
    checkpoint_path: str = "./checkpoints/gnn_best.pt"

    # ── Seed ──────────────────────────────────────────────────────────────────
    seed: int = 42
