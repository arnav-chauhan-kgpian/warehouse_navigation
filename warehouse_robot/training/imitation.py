"""training/imitation.py — Imitation learning: dataset generation + GNN training."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import Config
from env.warehouse_env import MultiAgentWarehouseEnv
from allocation.hungarian import HungarianAllocator
from allocation.gnn_allocator import TaskAllocGNN, WarehouseHeteroGraph

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class ImitationLearningTrainer:
    """Generates CBS-optimal assignment labels, then trains GNN via supervised learning."""

    def __init__(self, config: Config, device: torch.device) -> None:
        """Initialize trainer with environment, allocator, and device."""
        self.config = config
        self.device = device
        self.env = MultiAgentWarehouseEnv(config)
        self.hungarian = HungarianAllocator()

    def generate_dataset(
        self, num_episodes: int
    ) -> List[Tuple["HeteroData", torch.Tensor]]:
        """
        Generate (graph, assignment_label) pairs by running Hungarian on reset envs.

        Label: binary matrix Y of shape (k, m) where Y[i][j]=1 iff robot i assigned to task j.
        """
        if not HAS_PYG:
            raise RuntimeError("torch_geometric is not installed. Cannot generate dataset.")

        dataset: List[Tuple["HeteroData", torch.Tensor]] = []

        for ep in range(num_episodes):
            try:
                obs, _ = self.env.reset(seed=ep + self.config.seed)
            except Exception:
                continue

            robot_positions = self.env.robot_positions
            tasks = self.env.task_list
            task_status = self.env.task_status

            # Get Hungarian assignment
            assignment = self.hungarian.reassign_dynamic(
                robot_positions, tasks, task_status,
                [None] * self.config.num_robots
            )

            if not assignment:
                continue

            k = self.config.num_robots
            m = len(tasks)

            # Build label matrix
            label = torch.zeros((k, m), dtype=torch.float)
            for robot_id, task_id in assignment.items():
                if robot_id < k and task_id < m:
                    label[robot_id][task_id] = 1.0

            # Build graph
            task_status_arr = np.array([0] * m, dtype=np.int32)  # All pending at start
            graph = WarehouseHeteroGraph.build(
                np.array(robot_positions),
                np.array([t[0] for t in tasks]),
                np.array([t[1] for t in tasks]),
                task_status_arr,
                grid_height=self.config.grid_height,
                grid_width=self.config.grid_width,
            )

            dataset.append((graph, label))

            if (ep + 1) % 500 == 0:
                print(f"  [Dataset] Episode {ep + 1}/{num_episodes} — {len(dataset)} samples")

        print(f"  [Dataset] Generated {len(dataset)} samples total.")
        return dataset

    def train(
        self,
        model: TaskAllocGNN,
        dataset: List[Tuple["HeteroData", torch.Tensor]],
        save_path: str,
    ) -> Dict[str, List[float]]:
        """
        Train the GNN on generated dataset.

        Loss: BCEWithLogitsLoss | Optimizer: AdamW | Scheduler: CosineAnnealingLR
        Early stopping: stop if val loss stagnates for 5 epochs.
        Returns history dict with 'train_loss' and 'val_loss' lists.
        """
        if not dataset:
            print("  [Train] Empty dataset, skipping training.")
            return {"train_loss": [], "val_loss": []}

        # Train/val split
        random.shuffle(dataset)
        split = int(0.8 * len(dataset))
        train_data = dataset[:split]
        val_data = dataset[split:]

        print(f"  [Train] {len(train_data)} train / {len(val_data)} val samples")

        model = model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.config.gnn_learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.gnn_epochs)

        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(1, self.config.gnn_epochs + 1):
            model.train()
            train_loss = self._run_epoch(model, train_data, optimizer)
            model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(model, val_data, optimizer=None)

            scheduler.step()
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            print(f"  Epoch {epoch:3d}/{self.config.gnn_epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

            if HAS_WANDB:
                try:
                    wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
                except Exception:
                    pass

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  [Train] Early stopping at epoch {epoch}")
                    break

        print(f"  [Train] Best val loss: {best_val_loss:.4f} — checkpoint saved to {save_path}")
        return history

    def _run_epoch(
        self,
        model: TaskAllocGNN,
        data_list: List[Tuple["HeteroData", torch.Tensor]],
        optimizer: Optional[AdamW],
    ) -> float:
        """Run one training or validation epoch; return mean loss."""
        total_loss = 0.0
        count = 0

        # Simple mini-batch loop (manual batching over list)
        for graph, label in data_list:
            graph = graph.to(self.device)
            label = label.to(self.device)

            scores = model(graph)  # (k, m_pending)

            # Only include pending task columns in loss
            m_pending = scores.size(1)
            if m_pending == 0:
                continue

            label_pending = label[:, :m_pending]  # Match pending-task shape
            loss = F.binary_cross_entropy_with_logits(
                scores.view(-1),
                label_pending.view(-1).float(),
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / max(count, 1)
