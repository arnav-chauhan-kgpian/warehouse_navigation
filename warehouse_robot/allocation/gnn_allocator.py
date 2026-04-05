"""allocation/gnn_allocator.py — PyTorch Geometric GNN warm-start allocator."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    HeteroData = object  # Fallback type hint

from config import Config


class WarehouseHeteroGraph:
    """Converts warehouse state to PyTorch Geometric HeteroData object."""

    @staticmethod
    def build(
        robot_positions: np.ndarray,
        task_pickups: np.ndarray,
        task_dropoffs: np.ndarray,
        task_status: np.ndarray,
        grid_height: int = 16,
        grid_width: int = 20,
    ) -> "HeteroData":
        """
        Build a bipartite heterogeneous graph.

        Robot node features: [row/H, col/W, is_idle, dist_to_nearest_task/max_dist] (k, 4)
        Task node features:  [pickup_r/H, pickup_c/W, dropoff_r/H, dropoff_c/W, status/4] (m, 5)
        Edges: fully connected bipartite (robot->task and task->robot)
        Only PENDING tasks (status == 0) are included.
        """
        if not HAS_PYG:
            raise RuntimeError("torch_geometric is not installed.")

        H, W = grid_height, grid_width
        max_dist = H + W  # Maximum Manhattan distance

        # Filter to pending tasks
        pending_mask = task_status == 0
        pending_pickups = task_pickups[pending_mask]
        pending_dropoffs = task_dropoffs[pending_mask]

        k = len(robot_positions)
        m = len(pending_pickups)

        data = HeteroData()

        if m == 0:
            # No pending tasks — create empty graph
            data["robot"].x = torch.zeros((k, 4), dtype=torch.float)
            data["task"].x = torch.zeros((0, 5), dtype=torch.float)
            data["robot", "to", "task"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["task", "to", "robot"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            return data

        # ── Robot node features ───────────────────────────────────────────────
        robot_feats = []
        for i, (r, c) in enumerate(robot_positions):
            # Minimum distance to any pending task pickup
            dists = [abs(r - pr) + abs(c - pc) for pr, pc in pending_pickups]
            min_dist = min(dists) / max_dist if dists else 0.0
            # is_idle: 1.0 (all robots treated as idle at allocation time)
            robot_feats.append([r / H, c / W, 1.0, min_dist])
        robot_x = torch.tensor(robot_feats, dtype=torch.float)

        # ── Task node features ────────────────────────────────────────────────
        task_feats = []
        for (pr, pc), (dr, dc) in zip(pending_pickups, pending_dropoffs):
            task_feats.append([pr / H, pc / W, dr / H, dc / W, 0.0 / 4.0])  # 0 = pending
        task_x = torch.tensor(task_feats, dtype=torch.float)

        # ── Fully connected bipartite edges ───────────────────────────────────
        robot_idx = torch.arange(k).repeat_interleave(m)   # [0,0,...,1,1,...,k-1,...]
        task_idx = torch.arange(m).repeat(k)               # [0,1,...,m-1,0,...,m-1]

        edge_r2t = torch.stack([robot_idx, task_idx], dim=0)  # (2, k*m)
        edge_t2r = torch.stack([task_idx, robot_idx], dim=0)  # (2, k*m)

        # Edge feature: normalized manhattan distance
        edge_feats = []
        for ri in range(k):
            for ti in range(m):
                pr, pc = pending_pickups[ti]
                rr, rc = robot_positions[ri]
                d = (abs(rr - pr) + abs(rc - pc)) / max_dist
                edge_feats.append([d])
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

        data["robot"].x = robot_x
        data["task"].x = task_x
        data["robot", "to", "task"].edge_index = edge_r2t
        data["robot", "to", "task"].edge_attr = edge_attr
        data["task", "to", "robot"].edge_index = edge_t2r
        data["task", "to", "robot"].edge_attr = edge_attr  # Symmetric

        return data


class TaskAllocGNN(nn.Module):
    """Heterogeneous GNN predicting assignment scores for robot-task pairs."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3) -> None:
        """Initialize GNN with projection layers, HeteroConv layers, and scoring head."""
        super().__init__()

        if not HAS_PYG:
            raise RuntimeError("torch_geometric is not installed.")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projections
        self.robot_proj = nn.Linear(4, hidden_dim)
        self.task_proj = nn.Linear(5, hidden_dim)

        # HeteroConv layers (SAGEConv for message passing)
        self.convs = nn.ModuleList([
            HeteroConv({
                ("robot", "to", "task"): SAGEConv((-1, -1), hidden_dim),
                ("task", "to", "robot"): SAGEConv((-1, -1), hidden_dim),
            }) for _ in range(num_layers)
        ])

        # Scoring head: concat robot + task embeddings -> scalar score
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: "HeteroData") -> torch.Tensor:
        """
        Forward pass returning score matrix of shape (num_robots, num_tasks).

        Higher score indicates better robot-task pairing.
        """
        x_dict = {
            "robot": F.relu(self.robot_proj(data["robot"].x)),
            "task": F.relu(self.task_proj(data["task"].x)),
        }
        edge_index_dict = {
            ("robot", "to", "task"): data["robot", "to", "task"].edge_index,
            ("task", "to", "robot"): data["task", "to", "robot"].edge_index,
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}

        robot_emb = x_dict["robot"]   # (k, hidden_dim)
        task_emb = x_dict["task"]     # (m, hidden_dim)

        k = robot_emb.size(0)
        m = task_emb.size(0)

        if m == 0:
            return torch.zeros((k, 0), device=robot_emb.device)

        # Build all (robot, task) pairs
        robot_exp = robot_emb.unsqueeze(1).expand(k, m, -1)   # (k, m, hidden_dim)
        task_exp = task_emb.unsqueeze(0).expand(k, m, -1)     # (k, m, hidden_dim)
        pairs = torch.cat([robot_exp, task_exp], dim=-1)       # (k, m, 2*hidden_dim)

        scores = self.score_head(pairs).squeeze(-1)            # (k, m)
        return scores


class GNNAllocator:
    """Wraps TaskAllocGNN for inference: proposes top-k task assignments."""

    def __init__(self, model: TaskAllocGNN, config: Config) -> None:
        """Initialize with trained GNN model and configuration."""
        self.model = model
        self.config = config

    def propose_assignments(
        self,
        robot_positions: np.ndarray,
        tasks: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        task_status: np.ndarray,
        top_k: int = 3,
        grid_height: int = 16,
        grid_width: int = 20,
    ) -> List[Dict[int, int]]:
        """
        Build graph → run GNN → extract top-k assignments via greedy decoding.

        Returns list of top_k assignment dicts (robot_id -> task_id).
        """
        if not HAS_PYG or not tasks:
            return [{}]

        task_pickups = np.array([t[0] for t in tasks])
        task_dropoffs = np.array([t[1] for t in tasks])

        data = WarehouseHeteroGraph.build(
            robot_positions, task_pickups, task_dropoffs, task_status,
            grid_height=grid_height, grid_width=grid_width,
        )

        device = next(self.model.parameters()).device
        data = data.to(device)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(data)  # (k, m_pending)

        if scores.numel() == 0:
            return [{}]

        scores_np = scores.cpu().numpy()

        # Pending task indices in original task list
        pending_indices = [j for j, s in enumerate(task_status) if s == 0]

        proposals: List[Dict[int, int]] = []
        for _ in range(top_k):
            S = scores_np.copy()
            assignment: Dict[int, int] = {}
            k, m = S.shape
            for _ in range(min(k, m)):
                flat_idx = np.argmax(S)
                ri, ti = divmod(int(flat_idx), m)
                if S[ri, ti] == -np.inf:
                    break
                original_task_id = pending_indices[ti] if ti < len(pending_indices) else ti
                assignment[ri] = original_task_id
                S[ri, :] = -np.inf
                S[:, ti] = -np.inf
            proposals.append(assignment)
            # Perturb scores slightly for diversity in subsequent proposals
            scores_np = scores_np + np.random.randn(*scores_np.shape) * 0.1

        return proposals
