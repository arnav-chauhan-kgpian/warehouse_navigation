"""viz/renderer.py — Matplotlib-based warehouse renderer, animator, and plot generator."""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for Kaggle/headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np

from config import Config


class WarehouseRenderer:
    """Renders warehouse state using matplotlib. Produces frames, GIFs, and plots."""

    COLORS = {
        "free":           "#F7F6F2",
        "obstacle":       "#28251D",
        "robot":          "#01696F",
        "pickup":         "#DA7101",
        "dropoff":        "#437A22",
        "path":           "#4F98A3",
        "robot_carrying": "#A12C7B",
    }

    def __init__(self, config: Config) -> None:
        """Initialize renderer with configuration."""
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=(12, 10))

    def render_frame(
        self,
        grid: np.ndarray,
        robot_positions: List[Tuple[int, int]],
        paths: Dict[int, List[Tuple[int, int, int]]],
        tasks: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        task_status: List[str],
        robot_carrying: List[bool],
        timestep: int,
        metrics: Dict,
    ) -> np.ndarray:
        """
        Render current grid state to an RGB numpy array.

        Draws: grid cells, robot paths (faint), robots (circles), task markers, metrics.
        """
        self.ax.clear()
        H, W = grid.shape

        # ── Background cells ──────────────────────────────────────────────────
        for r in range(H):
            for c in range(W):
                val = grid[r, c]
                if val == 1:
                    color = self.COLORS["obstacle"]
                else:
                    color = self.COLORS["free"]
                rect = mpatches.Rectangle(
                    (c, H - r - 1), 1, 1,
                    linewidth=0.3, edgecolor="#CCCCCC", facecolor=color
                )
                self.ax.add_patch(rect)

        # ── Paths (faint dotted lines) ────────────────────────────────────────
        for robot_id, path in paths.items():
            if not path:
                continue
            xs = [p[1] + 0.5 for p in path]
            ys = [H - p[0] - 0.5 for p in path]
            self.ax.plot(xs, ys, color=self.COLORS["path"], alpha=0.35,
                         linewidth=1.2, linestyle="--")

        # ── Task markers ──────────────────────────────────────────────────────
        for j, (pickup, dropoff) in enumerate(tasks):
            if task_status[j] == "done":
                continue
            # Pickup: orange circle
            pr, pc = pickup
            self.ax.plot(pc + 0.5, H - pr - 0.5, "o",
                         color=self.COLORS["pickup"], markersize=8, zorder=3)
            self.ax.text(pc + 0.5, H - pr - 0.5, f"P{j}",
                         ha="center", va="center", fontsize=5, color="white", zorder=4)
            # Dropoff: green circle
            dr, dc = dropoff
            self.ax.plot(dc + 0.5, H - dr - 0.5, "s",
                         color=self.COLORS["dropoff"], markersize=8, zorder=3)
            self.ax.text(dc + 0.5, H - dr - 0.5, f"D{j}",
                         ha="center", va="center", fontsize=5, color="white", zorder=4)

        # ── Robots ────────────────────────────────────────────────────────────
        for i, (r, c) in enumerate(robot_positions):
            color = self.COLORS["robot_carrying"] if robot_carrying[i] else self.COLORS["robot"]
            circle = plt.Circle((c + 0.5, H - r - 0.5), 0.38,
                                 color=color, zorder=5)
            self.ax.add_patch(circle)
            self.ax.text(c + 0.5, H - r - 0.5, str(i),
                         ha="center", va="center", fontsize=7,
                         color="white", fontweight="bold", zorder=6)

        # ── Axes formatting ───────────────────────────────────────────────────
        tasks_done = metrics.get("tasks_completed", 0)
        total_tasks = len(tasks)
        collisions = metrics.get("collisions", 0)
        cbs_nodes = metrics.get("cbs_nodes_expanded", "–")
        makespan = metrics.get("makespan", timestep)
        soc = metrics.get("sum_of_costs", "–")

        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"Timestep {timestep} | Tasks Done: {tasks_done}/{total_tasks} | Collisions: {collisions}",
            fontsize=13, fontweight="bold", pad=10
        )
        self.ax.set_xlabel(
            f"Makespan: {makespan}  |  Sum-of-Costs: {soc}  |  CBS Nodes: {cbs_nodes}",
            fontsize=9
        )
        self.ax.axis("off")

        # Render to numpy array
        self.fig.tight_layout()
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
        buf.seek(0)
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()

        import cv2  # type: ignore
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def render_frame_simple(
        self,
        grid: np.ndarray,
        robot_positions: List[Tuple[int, int]],
        paths: Dict[int, List[Tuple[int, int, int]]],
        tasks: List[Tuple[Tuple[int, int], Tuple[int, int]]],
        task_status: List[str],
        robot_carrying: List[bool],
        timestep: int,
        metrics: Dict,
    ) -> np.ndarray:
        """Render frame without cv2 dependency (uses PIL instead)."""
        from PIL import Image  # type: ignore

        self.ax.clear()
        H, W = grid.shape

        for r in range(H):
            for c in range(W):
                val = grid[r, c]
                color = self.COLORS["obstacle"] if val == 1 else self.COLORS["free"]
                rect = mpatches.Rectangle((c, H - r - 1), 1, 1,
                                          linewidth=0.3, edgecolor="#CCCCCC", facecolor=color)
                self.ax.add_patch(rect)

        for robot_id, path in paths.items():
            if path:
                xs = [p[1] + 0.5 for p in path]
                ys = [H - p[0] - 0.5 for p in path]
                self.ax.plot(xs, ys, color=self.COLORS["path"],
                             alpha=0.35, linewidth=1.2, linestyle="--")

        for j, (pickup, dropoff) in enumerate(tasks):
            if task_status[j] != "done":
                pr, pc = pickup
                self.ax.plot(pc + 0.5, H - pr - 0.5, "o",
                             color=self.COLORS["pickup"], markersize=8, zorder=3)
                dr, dc = dropoff
                self.ax.plot(dc + 0.5, H - dr - 0.5, "s",
                             color=self.COLORS["dropoff"], markersize=8, zorder=3)

        for i, (r, c) in enumerate(robot_positions):
            color = self.COLORS["robot_carrying"] if robot_carrying[i] else self.COLORS["robot"]
            circle = plt.Circle((c + 0.5, H - r - 0.5), 0.38, color=color, zorder=5)
            self.ax.add_patch(circle)
            self.ax.text(c + 0.5, H - r - 0.5, str(i),
                         ha="center", va="center", fontsize=7,
                         color="white", fontweight="bold", zorder=6)

        tasks_done = metrics.get("tasks_completed", 0)
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(0, H)
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"Timestep {timestep} | Tasks Done: {tasks_done}/{len(tasks)} | "
            f"Collisions: {metrics.get('collisions', 0)}",
            fontsize=11, fontweight="bold"
        )
        self.ax.axis("off")
        self.fig.tight_layout()

        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=72)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        arr = np.array(img)
        buf.close()
        return arr

    def save_animation(
        self,
        frames: List[np.ndarray],
        output_path: str = "warehouse_sim.gif",
        fps: int = 5,
    ) -> None:
        """Save list of RGB numpy frames as animated GIF."""
        from PIL import Image  # type: ignore

        if not frames:
            print("[Renderer] No frames to save.")
            return

        pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=duration_ms,
        )
        print(f"[Renderer] Saved animation ({len(frames)} frames) -> {output_path}")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        cbs_baseline: Optional[List[float]] = None,
        cbs_gnn: Optional[List[float]] = None,
        output_path: str = "training_curves.png",
    ) -> None:
        """
        Two-panel plot: train/val loss curves and CBS nodes comparison.

        Uses teal (#01696F) and orange (#DA7101) as primary line colors.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#F9F9F9")

        # ── Left: Loss curves ─────────────────────────────────────────────────
        ax1 = axes[0]
        if history.get("train_loss"):
            ax1.plot(history["train_loss"], color="#01696F", linewidth=2, label="Train Loss")
        if history.get("val_loss"):
            ax1.plot(history["val_loss"], color="#DA7101", linewidth=2,
                     linestyle="--", label="Val Loss")
        ax1.set_title("GNN Training Loss", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("BCE Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor("#F0F0EE")

        # ── Right: CBS nodes comparison ───────────────────────────────────────
        ax2 = axes[1]
        if cbs_baseline:
            ax2.plot(cbs_baseline, color="#DA7101", linewidth=2,
                     label="CBS + Hungarian (Baseline)")
        if cbs_gnn:
            ax2.plot(cbs_gnn, color="#01696F", linewidth=2,
                     label="CBS + Hungarian + GNN")
        ax2.set_title("CBS Nodes Expanded", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Nodes Expanded")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor("#F0F0EE")

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[Renderer] Training curves saved -> {output_path}")

    def plot_ablation_table(
        self,
        results: Dict[str, Dict],
        output_path: str = "ablation_results.png",
    ) -> None:
        """
        Render a styled matplotlib comparison table.

        Rows: methods. Cols: Makespan, SoC, Collisions, Alloc Time (ms).
        Best values per column highlighted in teal.
        """
        methods = list(results.keys())
        columns = ["Method", "Makespan", "Sum-of-Costs", "Collisions", "Alloc (ms)"]

        table_data = []
        for method in methods:
            m = results[method]
            table_data.append([
                method,
                f"{m.get('makespan', '–'):.1f}" if isinstance(m.get("makespan"), float) else str(m.get("makespan", "–")),
                f"{m.get('sum_of_costs', '–'):.1f}" if isinstance(m.get("sum_of_costs"), float) else str(m.get("sum_of_costs", "–")),
                str(m.get("collisions", "–")),
                f"{m.get('allocation_time_ms', '–'):.1f}" if isinstance(m.get("allocation_time_ms"), float) else str(m.get("allocation_time_ms", "–")),
            ])

        fig, ax = plt.subplots(figsize=(14, max(3, len(methods) * 0.8 + 1.5)))
        fig.patch.set_facecolor("#F9F9F9")
        ax.axis("off")

        tbl = ax.table(
            cellText=table_data,
            colLabels=columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.8)

        # Header style
        for j in range(len(columns)):
            tbl[0, j].set_facecolor("#01696F")
            tbl[0, j].set_text_props(color="white", fontweight="bold")

        # Highlight best values (columns 1–4, lower is better for all)
        for col_idx in range(1, len(columns)):
            try:
                vals = []
                for row_idx, row in enumerate(table_data):
                    raw = row[col_idx].replace("~", "")
                    vals.append((row_idx, float(raw)))
                best_row = min(vals, key=lambda x: x[1])[0]
                tbl[best_row + 1, col_idx].set_facecolor("#01696F")
                tbl[best_row + 1, col_idx].set_text_props(color="white", fontweight="bold")
            except (ValueError, IndexError):
                pass

        # Alternate row shading
        for row_idx in range(1, len(table_data) + 1):
            bg = "#E8F4F4" if row_idx % 2 == 0 else "#FFFFFF"
            for col_idx in range(len(columns)):
                cell = tbl[row_idx, col_idx]
                if cell.get_facecolor() == matplotlib.colors.to_rgba("#FFFFFF") or \
                   cell.get_facecolor()[0:3] == (1.0, 1.0, 1.0):
                    cell.set_facecolor(bg)

        ax.set_title("Ablation Results — Multi-Robot Warehouse Navigation",
                     fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[Renderer] Ablation table saved -> {output_path}")
