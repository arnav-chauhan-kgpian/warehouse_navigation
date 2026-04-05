"""metrics/logger.py — WandB + CSV metrics logger for warehouse experiments."""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import pandas as pd

from config import Config

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class MetricsLogger:
    """Logs per-episode and training metrics to WandB and CSV."""

    def __init__(self, config: Config, use_wandb: bool = True) -> None:
        """Initialize WandB run (if available) and local CSV log."""
        self.config = config
        self.use_wandb = use_wandb and HAS_WANDB
        self._records: List[Dict] = []
        self._csv_path = "metrics_log.csv"
        self._fieldnames: Optional[List[str]] = None

        if self.use_wandb:
            try:
                wandb.init(
                    project=config.wandb_project,
                    config={
                        "grid_height": config.grid_height,
                        "grid_width": config.grid_width,
                        "num_robots": config.num_robots,
                        "num_tasks": config.num_tasks,
                        "gnn_epochs": config.gnn_epochs,
                        "seed": config.seed,
                    },
                    mode="offline",  # Safe for restricted environments
                )
                print("[Logger] WandB initialized (offline mode).")
            except Exception as e:
                print(f"[Logger] WandB init failed: {e}. Logging CSV only.")
                self.use_wandb = False

    def log_episode(self, episode: int, metrics: Dict) -> None:
        """Log episode-level metrics to WandB and internal buffer."""
        record = {"episode": episode, **metrics}
        self._records.append(record)

        if self.use_wandb:
            try:
                wandb.log({"episode": episode, **metrics})
            except Exception:
                pass

    def log_training(self, epoch: int, train_loss: float, val_loss: float) -> None:
        """Log GNN training metrics."""
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        self._records.append(record)

        if self.use_wandb:
            try:
                wandb.log(record)
            except Exception:
                pass

    def summary(self) -> pd.DataFrame:
        """Return all logged metrics as a DataFrame and save to CSV."""
        if not self._records:
            return pd.DataFrame()

        df = pd.DataFrame(self._records)
        df.to_csv(self._csv_path, index=False)
        print(f"[Logger] Metrics saved → {self._csv_path}")
        return df

    def close(self) -> None:
        """Finalize WandB run and flush remaining data."""
        self.summary()  # Ensure CSV is written
        if self.use_wandb:
            try:
                wandb.finish()
                print("[Logger] WandB run finalized.")
            except Exception:
                pass
