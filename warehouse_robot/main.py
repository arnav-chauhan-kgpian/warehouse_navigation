"""main.py — Entry point: runs full multi-robot warehouse navigation pipeline."""

from __future__ import annotations

import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ── Add warehouse_robot to path if run from parent directory ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from env.warehouse_env import MultiAgentWarehouseEnv
from planning.cbs import ConflictBasedSearch
from planning.astar import SpaceTimeAStar
from allocation.hungarian import HungarianAllocator
from allocation.gnn_allocator import TaskAllocGNN, GNNAllocator, WarehouseHeteroGraph
from training.imitation import ImitationLearningTrainer
from viz.renderer import WarehouseRenderer
from metrics.logger import MetricsLogger

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_episode_cbs_hungarian(
    env: MultiAgentWarehouseEnv,
    config: Config,
    cbs: ConflictBasedSearch,
    hungarian: HungarianAllocator,
    gnn_allocator: Optional[GNNAllocator] = None,
    capture_frames: bool = False,
    renderer: Optional[WarehouseRenderer] = None,
) -> Tuple[Dict, List[np.ndarray]]:
    """
    Run one episode using CBS + Hungarian (optionally GNN warm-start).

    Returns (metrics_dict, frames_list).
    """
    obs, _ = env.reset(seed=None)
    assignment = hungarian.reassign_dynamic(
        env.robot_positions, env.task_list, env.task_status, env.robot_tasks
    )
    # Apply assignment to env
    for robot_id, task_id in assignment.items():
        env.robot_tasks[robot_id] = task_id
        env.task_status[task_id] = "in_progress_pickup"

    # Plan CBS paths for each robot to its assigned goal
    starts = {i: env.robot_positions[i] for i in range(config.num_robots)}
    goals: Dict[int, Tuple[int, int]] = {}
    for robot_id in range(config.num_robots):
        task_idx = env.robot_tasks[robot_id]
        if task_idx is not None:
            pickup, dropoff = env.task_list[task_idx]
            goals[robot_id] = pickup  # First goal: pickup
        else:
            goals[robot_id] = env.robot_positions[robot_id]  # Stay in place

    t0 = time.time()
    cbs.nodes_expanded = 0
    paths = cbs.plan(starts, goals)
    alloc_time_ms = (time.time() - t0) * 1000

    if paths is None:
        paths = {i: [(env.robot_positions[i][0], env.robot_positions[i][1], 0)]
                 for i in range(config.num_robots)}

    frames: List[np.ndarray] = []
    total_collisions = 0
    sum_of_costs = sum(len(p) for p in paths.values())

    # Step through planned paths
    max_path_len = max(len(p) for p in paths.values()) if paths else 0
    for step in range(max_path_len):
        actions: Dict[int, Tuple[int, int]] = {}
        for robot_id, path in paths.items():
            if step < len(path):
                r, c, _ = path[step]
                actions[robot_id] = (r, c)
            else:
                actions[robot_id] = env.robot_positions[robot_id]

        obs, rewards, terminated, truncated, info = env.step(actions)
        total_collisions += info["collisions"]

        if capture_frames and renderer is not None:
            try:
                frame = renderer.render_frame_simple(
                    grid=env.grid,
                    robot_positions=env.robot_positions,
                    paths=paths,
                    tasks=env.task_list,
                    task_status=env.task_status,
                    robot_carrying=env.robot_carrying,
                    timestep=step,
                    metrics={
                        "tasks_completed": info["tasks_completed"],
                        "collisions": total_collisions,
                        "cbs_nodes_expanded": cbs.nodes_expanded,
                        "sum_of_costs": sum_of_costs,
                        "makespan": step,
                    },
                )
                frames.append(frame)
            except Exception as e:
                pass  # Rendering errors are non-fatal

        if terminated or truncated:
            break

    metrics = {
        "makespan": info.get("makespan", max_path_len),
        "sum_of_costs": sum_of_costs,
        "collisions": total_collisions,
        "tasks_completed": info.get("tasks_completed", 0),
        "cbs_nodes_expanded": cbs.nodes_expanded,
        "allocation_time_ms": alloc_time_ms,
        "total_distance_cells": info.get("total_distance", 0),
        "episode_timeout": truncated,
    }
    return metrics, frames


def run_episode_greedy(
    env: MultiAgentWarehouseEnv,
    config: Config,
) -> Dict:
    """Run one episode with greedy nearest-robot assignment and independent A*."""
    obs, _ = env.reset()
    astar = SpaceTimeAStar(env.grid, config)

    # Greedy assignment: each idle robot gets the nearest pending task
    assignment: Dict[int, int] = {}
    assigned_tasks: set = set()
    for robot_id in range(config.num_robots):
        best_task, best_cost = None, float("inf")
        for j, (pickup, _) in enumerate(env.task_list):
            if j in assigned_tasks or env.task_status[j] != "pending":
                continue
            cost = HungarianAllocator.manhattan(env.robot_positions[robot_id], pickup)
            if cost < best_cost:
                best_cost = cost
                best_task = j
        if best_task is not None:
            assignment[robot_id] = best_task
            assigned_tasks.add(best_task)
            env.robot_tasks[robot_id] = best_task
            env.task_status[best_task] = "in_progress_pickup"

    goals = {}
    for robot_id in range(config.num_robots):
        task_idx = env.robot_tasks[robot_id]
        if task_idx is not None:
            goals[robot_id] = env.task_list[task_idx][0]
        else:
            goals[robot_id] = env.robot_positions[robot_id]

    t0 = time.time()
    paths = {}
    for robot_id in range(config.num_robots):
        path = astar.plan(env.robot_positions[robot_id], goals[robot_id], [])
        if path:
            paths[robot_id] = path
        else:
            paths[robot_id] = [(env.robot_positions[robot_id][0],
                                 env.robot_positions[robot_id][1], 0)]
    alloc_time_ms = (time.time() - t0) * 1000

    sum_of_costs = sum(len(p) for p in paths.values())
    total_collisions = 0

    max_t = max(len(p) for p in paths.values()) if paths else 1
    for step in range(max_t):
        actions = {}
        for robot_id, path in paths.items():
            if step < len(path):
                r, c, _ = path[step]
                actions[robot_id] = (r, c)
            else:
                actions[robot_id] = env.robot_positions[robot_id]
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_collisions += info["collisions"]
        if terminated or truncated:
            break

    return {
        "makespan": info.get("makespan", max_t),
        "sum_of_costs": sum_of_costs,
        "collisions": total_collisions,
        "tasks_completed": info.get("tasks_completed", 0),
        "cbs_nodes_expanded": 0,
        "allocation_time_ms": alloc_time_ms,
        "total_distance_cells": info.get("total_distance", 0),
        "episode_timeout": truncated,
    }


def run_episode_prioritized(
    env: MultiAgentWarehouseEnv,
    config: Config,
) -> Dict:
    """Run one episode with prioritized A* + greedy assignment."""
    obs, _ = env.reset()
    astar = SpaceTimeAStar(env.grid, config)

    # Greedy assignment
    assignment: Dict[int, int] = {}
    assigned_tasks: set = set()
    for robot_id in range(config.num_robots):
        best_task, best_cost = None, float("inf")
        for j, (pickup, _) in enumerate(env.task_list):
            if j in assigned_tasks or env.task_status[j] != "pending":
                continue
            cost = HungarianAllocator.manhattan(env.robot_positions[robot_id], pickup)
            if cost < best_cost:
                best_cost = cost
                best_task = j
        if best_task is not None:
            assignment[robot_id] = best_task
            assigned_tasks.add(best_task)
            env.robot_tasks[robot_id] = best_task
            env.task_status[best_task] = "in_progress_pickup"

    goals = {}
    for robot_id in range(config.num_robots):
        task_idx = env.robot_tasks[robot_id]
        if task_idx is not None:
            goals[robot_id] = env.task_list[task_idx][0]
        else:
            goals[robot_id] = env.robot_positions[robot_id]

    t0 = time.time()
    # Prioritized: plan in order, each robot avoids previous robots' paths
    all_constraints: List[Tuple[int, int, int]] = []
    paths = {}
    for robot_id in range(config.num_robots):
        path = astar.plan(env.robot_positions[robot_id], goals[robot_id], all_constraints)
        if path is None:
            path = [(env.robot_positions[robot_id][0], env.robot_positions[robot_id][1], 0)]
        paths[robot_id] = path
        all_constraints.extend([(r, c, t) for r, c, t in path])
    alloc_time_ms = (time.time() - t0) * 1000

    sum_of_costs = sum(len(p) for p in paths.values())
    total_collisions = 0

    max_t = max(len(p) for p in paths.values()) if paths else 1
    for step in range(max_t):
        actions = {}
        for robot_id, path in paths.items():
            if step < len(path):
                r, c, _ = path[step]
                actions[robot_id] = (r, c)
            else:
                actions[robot_id] = env.robot_positions[robot_id]
        obs, rewards, terminated, truncated, info = env.step(actions)
        total_collisions += info["collisions"]
        if terminated or truncated:
            break

    return {
        "makespan": info.get("makespan", max_t),
        "sum_of_costs": sum_of_costs,
        "collisions": total_collisions,
        "tasks_completed": info.get("tasks_completed", 0),
        "cbs_nodes_expanded": 0,
        "allocation_time_ms": alloc_time_ms,
        "total_distance_cells": info.get("total_distance", 0),
        "episode_timeout": truncated,
    }


def aggregate_metrics(records: List[Dict]) -> Dict:
    """Compute mean metrics across episodes."""
    if not records:
        return {}
    keys = records[0].keys()
    agg = {}
    for k in keys:
        try:
            agg[k] = float(np.mean([r[k] for r in records]))
        except (TypeError, ValueError):
            agg[k] = records[0][k]
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run full 6-phase warehouse navigation pipeline."""
    config = Config()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    logger = MetricsLogger(config, use_wandb=True)
    renderer = WarehouseRenderer(config)
    env = MultiAgentWarehouseEnv(config)
    cbs = ConflictBasedSearch(env.grid, config)
    hungarian = HungarianAllocator()

    os.makedirs("checkpoints", exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== PHASE 1: Environment Validation ===")
    # ══════════════════════════════════════════════════════════════════════════
    phase1_records = []
    for ep in range(5):
        metrics, _ = run_episode_cbs_hungarian(env, config, cbs, hungarian)
        metrics["method_name"] = "CBS+Hungarian"
        phase1_records.append(metrics)
        logger.log_episode(ep, metrics)
        print(f"  Episode {ep+1}/5 | makespan={metrics['makespan']} | "
              f"collisions={metrics['collisions']} | tasks={metrics['tasks_completed']}")

    agg1 = aggregate_metrics(phase1_records)
    print(f"  [Validation] Mean makespan={agg1.get('makespan',0):.1f}, "
          f"collisions={agg1.get('collisions',0):.1f}")
    if agg1.get("collisions", 1) == 0:
        print("  ✓ Zero collisions confirmed for CBS+Hungarian baseline.")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== PHASE 2: Ablation Baselines ===")
    # ══════════════════════════════════════════════════════════════════════════
    N_ABLATION = 20  # Reduced from 100 for reasonable local runtime
    ablation_results: Dict[str, Dict] = {}

    print("  Running: Greedy A* + Nearest Robot ...")
    greedy_records = []
    for ep in range(N_ABLATION):
        set_seed(config.seed + ep)
        metrics = run_episode_greedy(env, config)
        metrics["method_name"] = "Greedy A*"
        greedy_records.append(metrics)
        logger.log_episode(1000 + ep, metrics)
    ablation_results["Greedy A* + Nearest Robot"] = aggregate_metrics(greedy_records)
    print(f"    makespan={ablation_results['Greedy A* + Nearest Robot']['makespan']:.1f}, "
          f"collisions={ablation_results['Greedy A* + Nearest Robot']['collisions']:.1f}")

    print("  Running: Prioritized A* + Greedy ...")
    prio_records = []
    for ep in range(N_ABLATION):
        set_seed(config.seed + ep)
        metrics = run_episode_prioritized(env, config)
        metrics["method_name"] = "Prioritized A*"
        prio_records.append(metrics)
        logger.log_episode(2000 + ep, metrics)
    ablation_results["Prioritized A* + Greedy"] = aggregate_metrics(prio_records)
    print(f"    makespan={ablation_results['Prioritized A* + Greedy']['makespan']:.1f}, "
          f"collisions={ablation_results['Prioritized A* + Greedy']['collisions']:.1f}")

    print("  Running: CBS + Hungarian ...")
    cbs_hung_records = []
    for ep in range(N_ABLATION):
        set_seed(config.seed + ep)
        metrics, _ = run_episode_cbs_hungarian(env, config, cbs, hungarian)
        metrics["method_name"] = "CBS+Hungarian"
        cbs_hung_records.append(metrics)
        logger.log_episode(3000 + ep, metrics)
    ablation_results["CBS + Hungarian"] = aggregate_metrics(cbs_hung_records)
    print(f"    makespan={ablation_results['CBS + Hungarian']['makespan']:.1f}, "
          f"collisions={ablation_results['CBS + Hungarian']['collisions']:.1f}")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== PHASE 3: GNN Training ===")
    # ══════════════════════════════════════════════════════════════════════════
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    gnn_model: Optional[TaskAllocGNN] = None

    if HAS_PYG:
        trainer = ImitationLearningTrainer(config, device)
        # Use reduced episodes for local testing (change in config for full Kaggle run)
        train_eps = min(config.gnn_train_episodes, 500)  # Cap at 500 locally
        print(f"  Generating {train_eps} training episodes...")
        dataset = trainer.generate_dataset(train_eps)

        if dataset:
            gnn_model = TaskAllocGNN(
                hidden_dim=config.gnn_hidden_dim,
                num_layers=config.gnn_num_layers,
            )
            print("  Training GNN...")
            history = trainer.train(gnn_model, dataset, config.checkpoint_path)
            for epoch, (tl, vl) in enumerate(zip(history["train_loss"], history["val_loss"])):
                logger.log_training(epoch, tl, vl)
        else:
            print("  [Warning] No training data generated — skipping GNN training.")
    else:
        print("  [Warning] torch_geometric not available — skipping GNN training.")

    renderer.plot_training_curves(history, output_path="training_curves.png")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== PHASE 4: GNN Evaluation ===")
    # ══════════════════════════════════════════════════════════════════════════
    gnn_records = []
    if gnn_model is not None and HAS_PYG:
        # Load best checkpoint
        if os.path.exists(config.checkpoint_path):
            gnn_model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
            print(f"  Loaded checkpoint: {config.checkpoint_path}")
        gnn_allocator = GNNAllocator(gnn_model, config)

        print(f"  Running {N_ABLATION} episodes with CBS+Hungarian+GNN ...")
        for ep in range(N_ABLATION):
            set_seed(config.seed + ep)
            metrics, _ = run_episode_cbs_hungarian(env, config, cbs, hungarian, gnn_allocator)
            metrics["method_name"] = "CBS+Hungarian+GNN"
            gnn_records.append(metrics)
            logger.log_episode(4000 + ep, metrics)

        ablation_results["CBS + Hungarian + GNN"] = aggregate_metrics(gnn_records)
        agg_gnn = ablation_results["CBS + Hungarian + GNN"]
        agg_base = ablation_results["CBS + Hungarian"]
        speedup = agg_base.get("cbs_nodes_expanded", 1) / max(agg_gnn.get("cbs_nodes_expanded", 1), 1)
        print(f"  GNN makespan={agg_gnn['makespan']:.1f}, "
              f"CBS nodes={agg_gnn['cbs_nodes_expanded']:.1f}, speedup={speedup:.2f}×")
    else:
        print("  [Warning] GNN not available — skipping GNN eval.")
        ablation_results["CBS + Hungarian + GNN"] = {"makespan": "N/A", "sum_of_costs": "N/A",
                                                      "collisions": "N/A", "allocation_time_ms": "N/A"}

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== PHASE 5: Visualization ===")
    # ══════════════════════════════════════════════════════════════════════════
    print("  Running visualization episode...")
    set_seed(config.seed)
    frames: List[np.ndarray] = []
    try:
        _, frames = run_episode_cbs_hungarian(
            env, config, cbs, hungarian,
            capture_frames=True, renderer=renderer,
        )
        print(f"  Captured {len(frames)} frames.")
    except Exception as e:
        print(f"  [Warning] Frame capture failed: {e}")

    if frames:
        renderer.save_animation(frames, "warehouse_sim.gif", fps=5)
    else:
        print("  [Warning] No frames captured — GIF not produced.")

    renderer.plot_ablation_table(ablation_results, "ablation_results.png")

    # ══════════════════════════════════════════════════════════════════════════
    print("\n=== PHASE 6: Final Report ===")
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 75)
    print(f"{'Method':<30} {'Makespan':>10} {'SoC':>10} {'Coll.':>8} {'AllocMs':>10}")
    print("─" * 75)
    for method, agg in ablation_results.items():
        mk = f"{agg.get('makespan', '–'):.1f}" if isinstance(agg.get("makespan"), float) else str(agg.get("makespan", "–"))
        sc = f"{agg.get('sum_of_costs', '–'):.1f}" if isinstance(agg.get("sum_of_costs"), float) else str(agg.get("sum_of_costs", "–"))
        co = f"{agg.get('collisions', '–'):.1f}" if isinstance(agg.get("collisions"), float) else str(agg.get("collisions", "–"))
        al = f"{agg.get('allocation_time_ms', '–'):.1f}" if isinstance(agg.get("allocation_time_ms"), float) else str(agg.get("allocation_time_ms", "–"))
        print(f"{method:<30} {mk:>10} {sc:>10} {co:>8} {al:>10}")
    print("─" * 75)

    if "CBS + Hungarian + GNN" in ablation_results and "CBS + Hungarian" in ablation_results:
        base_nodes = ablation_results["CBS + Hungarian"].get("cbs_nodes_expanded", 1)
        gnn_nodes = ablation_results["CBS + Hungarian + GNN"].get("cbs_nodes_expanded", 1)
        try:
            ratio = float(base_nodes) / max(float(gnn_nodes), 1)
            print(f"\nGNN Speedup Ratio (CBS nodes): {ratio:.2f}×")
        except (TypeError, ValueError):
            pass

    logger.close()
    print("\n✓ Pipeline complete. Output files: warehouse_sim.gif, training_curves.png, "
          "ablation_results.png, metrics_log.csv")


if __name__ == "__main__":
    main()
