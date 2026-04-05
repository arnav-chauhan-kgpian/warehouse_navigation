"""
visualizer.py — Warehouse Simulation Visualiser
================================================
Three visualisation modes:

  WarehouseVisualizer.animate()
      Replay the full simulation as a smooth Matplotlib animation.
      Shows the warehouse map, robot movements, planned-path overlays,
      movement trails and live bar-chart metrics.

  WarehouseVisualizer.plot_metrics()
      Static bar + histogram charts of final performance statistics.

  WarehouseVisualizer.plot_final_map()
      Top-down heatmap of robot trails with final positions and task markers.
"""

from __future__ import annotations

from typing import List

import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from robot import Robot, Task, TaskStatus
from simulation import Simulation
from warehouse import Warehouse

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

ROBOT_PALETTE = [
    "#E74C3C",   # vivid red
    "#3498DB",   # steel blue
    "#2ECC71",   # emerald green
    "#F39C12",   # amber
    "#9B59B6",   # amethyst
    "#1ABC9C",   # teal
    "#E67E22",   # carrot orange
    "#2980B9",   # belize blue
    "#D35400",   # pumpkin
    "#27AE60",   # nephritis
]

_SHELF_RGBA  = [0.18, 0.20, 0.25, 1.0]   # dark blue-grey shelves
_FLOOR_RGBA  = [0.95, 0.95, 0.92, 1.0]   # warm off-white floor
_BG_DARK     = "#0f0f23"
_PANEL_DARK  = "#1a1a3e"
_PANEL_MED   = "#16213e"


# ---------------------------------------------------------------------------
# Visualiser
# ---------------------------------------------------------------------------

class WarehouseVisualizer:
    """
    Interactive Matplotlib visualiser for the warehouse simulation.

    Parameters
    ----------
    warehouse           : The warehouse environment.
    robots              : List of Robot objects (post-simulation state).
    tasks               : All tasks (static + dynamic).
    simulation          : The completed Simulation instance (for history).
    animation_interval  : Milliseconds between animation frames.
    show_trails         : Overlay robot movement trails.
    show_paths          : Overlay planned-path previews.
    trail_max_len       : Maximum number of trail positions rendered per robot.
    """

    def __init__(
        self,
        warehouse:          Warehouse,
        robots:             List[Robot],
        tasks:              List[Task],
        simulation:         Simulation,
        animation_interval: int  = 130,
        show_trails:        bool = True,
        show_paths:         bool = True,
        trail_max_len:      int  = 80,
    ) -> None:
        self.wh        = warehouse
        self.robots    = robots
        self.tasks     = tasks
        self.sim       = simulation
        self.interval  = animation_interval
        self.trails    = show_trails
        self.paths     = show_paths
        self.trail_max = trail_max_len

        self._bg = self._build_bg()

    # ------------------------------------------------------------------
    # Background image
    # ------------------------------------------------------------------

    def _build_bg(self) -> np.ndarray:
        """RGBA background image: shelves vs. floor."""
        rows, cols = self.wh.rows, self.wh.cols
        img = np.ones((rows, cols, 4))
        for r in range(rows):
            for c in range(cols):
                img[r, c] = _SHELF_RGBA if self.wh.grid[r][c] else _FLOOR_RGBA
        return img

    def _color(self, robot_id: int) -> str:
        return ROBOT_PALETTE[robot_id % len(ROBOT_PALETTE)]

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------

    def animate(self) -> None:
        """Play back the simulation as an animated Matplotlib figure."""
        history = self.sim.history
        if not history:
            print("[Visualiser] No history to replay.")
            return

        # ── Layout ──────────────────────────────────────────────────────
        fig = plt.figure(figsize=(17, 9), facecolor=_BG_DARK)
        gs  = GridSpec(
            2, 3, figure=fig,
            left=0.04, right=0.98, top=0.93, bottom=0.07,
            hspace=0.40, wspace=0.30,
        )
        ax_main  = fig.add_subplot(gs[:, :2])
        ax_dist  = fig.add_subplot(gs[0, 2])
        ax_tasks = fig.add_subplot(gs[1, 2])

        for ax in (ax_main, ax_dist, ax_tasks):
            ax.set_facecolor(_PANEL_MED)
            ax.tick_params(colors="#999", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#333")

        # ── Warehouse background ─────────────────────────────────────────
        rows, cols = self.wh.rows, self.wh.cols
        ax_main.imshow(
            self._bg, origin="upper", aspect="equal",
            extent=[-0.5, cols - 0.5, rows - 0.5, -0.5],
        )
        ax_main.set_xlim(-0.5, cols - 0.5)
        ax_main.set_ylim(rows - 0.5, -0.5)
        ax_main.set_title(
            "Multi-Robot Warehouse Navigation & Task Allocation",
            color="white", fontsize=13, fontweight="bold", pad=8,
        )
        ax_main.set_xlabel("Column", color="#888", fontsize=8)
        ax_main.set_ylabel("Row",    color="#888", fontsize=8)

        # ── Static task markers ──────────────────────────────────────────
        pickup_pts  = {}
        dropoff_pts = {}
        for task in self.tasks:
            pp = ax_main.plot(
                task.pickup[1],  task.pickup[0],
                marker="D", markersize=7, color="#2ECC71",
                markeredgecolor="#1a6b3a", markeredgewidth=0.8,
                zorder=4, alpha=0.9, linestyle="None",
            )[0]
            dp = ax_main.plot(
                task.dropoff[1], task.dropoff[0],
                marker="s", markersize=7, color="#E74C3C",
                markeredgecolor="#8b1a1a", markeredgewidth=0.8,
                zorder=4, alpha=0.9, linestyle="None",
            )[0]
            pickup_pts[task.id]  = pp
            dropoff_pts[task.id] = dp

        # ── Dynamic robot artists ────────────────────────────────────────
        robot_circles = []
        robot_labels  = []
        trail_lines   = []
        path_lines    = []

        for robot in self.robots:
            clr = self._color(robot.id)

            circ = plt.Circle(
                (robot.pos[1], robot.pos[0]), 0.42,
                color=clr, zorder=6, alpha=0.92,
            )
            ax_main.add_patch(circ)
            robot_circles.append(circ)

            lbl = ax_main.text(
                robot.pos[1], robot.pos[0], str(robot.id),
                color="white", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=7,
            )
            robot_labels.append(lbl)

            tl, = ax_main.plot([], [], color=clr, alpha=0.28,
                               linewidth=2.0, zorder=3, solid_capstyle="round")
            trail_lines.append(tl)

            pl, = ax_main.plot([], [], color=clr, alpha=0.55,
                               linewidth=1.2, linestyle="--", zorder=5)
            path_lines.append(pl)

        # ── Metric bar charts ────────────────────────────────────────────
        rids   = [r.id for r in self.robots]
        rlabels = [f"R{r.id}" for r in self.robots]
        rcolors = [self._color(r.id) for r in self.robots]

        bars_dist  = ax_dist.bar(rids,  [0] * len(rids),  color=rcolors, alpha=0.85)
        bars_tasks = ax_tasks.bar(rids, [0] * len(rids),  color=rcolors, alpha=0.85)

        ax_dist.set_title("Distance Traveled",  color="white", fontsize=10)
        ax_tasks.set_title("Tasks Completed",   color="white", fontsize=10)
        for ax in (ax_dist, ax_tasks):
            ax.set_xticks(rids)
            ax.set_xticklabels(rlabels, color="#ccc", fontsize=8)
            ax.set_ylabel("Count", color="#aaa", fontsize=8)

        # ── HUD overlays ─────────────────────────────────────────────────
        hud_kw = dict(
            transform=ax_main.transAxes, color="white",
            fontsize=11, fontweight="bold", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor=_PANEL_DARK, alpha=0.85),
        )
        step_txt  = ax_main.text(0.02, 0.975, "Step: 0",    **hud_kw)
        task_txt  = ax_main.text(
            0.98, 0.975, "Tasks: 0/0",
            horizontalalignment="right", **hud_kw,
        )

        # ── Legend ───────────────────────────────────────────────────────
        legend_elems = [
            mpatches.Patch(facecolor="#2ECC71", edgecolor="#1a6b3a",
                           label="Pickup  ◆"),
            mpatches.Patch(facecolor="#E74C3C", edgecolor="#8b1a1a",
                           label="Drop-off ■"),
        ] + [
            mpatches.Patch(facecolor=self._color(r.id), label=f"Robot {r.id}")
            for r in self.robots
        ]
        ax_main.legend(
            handles=legend_elems, loc="lower left",
            facecolor=_PANEL_DARK, edgecolor="#555",
            labelcolor="white", fontsize=8,
            ncol=min(4, 2 + len(self.robots)),
        )

        # ── Update function ───────────────────────────────────────────────
        def update(frame: int):
            if frame >= len(history):
                return []

            snap        = history[frame]
            n_done      = snap["n_completed"]
            t_statuses  = snap["task_statuses"]

            step_txt.set_text(f"Step: {snap['step']}")
            task_txt.set_text(f"Tasks: {n_done}/{len(self.tasks)}")

            for i, robot in enumerate(self.robots):
                pos = snap["robot_positions"][robot.id]

                # Circle + label
                robot_circles[i].center = (pos[1], pos[0])
                robot_labels[i].set_position((pos[1], pos[0]))

                # Trail
                if self.trails:
                    trail = snap["robot_trails"][robot.id]
                    if len(trail) > 1:
                        tail = trail[-self.trail_max:]
                        trail_lines[i].set_data(
                            [p[1] for p in tail], [p[0] for p in tail]
                        )
                    else:
                        trail_lines[i].set_data([], [])

                # Path preview
                if self.paths:
                    p   = snap["robot_paths"][robot.id]
                    idx = snap["robot_path_idx"][robot.id]
                    rem = p[idx:]
                    if len(rem) > 1:
                        path_lines[i].set_data(
                            [q[1] for q in rem], [q[0] for q in rem]
                        )
                    else:
                        path_lines[i].set_data([], [])

            # Task marker visibility
            for task in self.tasks:
                done = t_statuses.get(task.id) == TaskStatus.COMPLETED
                pickup_pts[task.id].set_alpha(0.18 if done else 0.9)
                dropoff_pts[task.id].set_alpha(0.18 if done else 0.9)

            # Metric bars
            max_d, max_t = 1, 1
            for i, robot in enumerate(self.robots):
                d = snap["robot_distances"][robot.id]
                t = snap["robot_task_counts"][robot.id]
                bars_dist[i].set_height(d)
                bars_tasks[i].set_height(t)
                max_d = max(max_d, d)
                max_t = max(max_t, t)

            ax_dist.set_ylim(0, max_d * 1.25 + 5)
            ax_tasks.set_ylim(0, max_t * 1.4  + 1)

            return (
                [step_txt, task_txt]
                + robot_circles + robot_labels
                + trail_lines + path_lines
                + list(pickup_pts.values()) + list(dropoff_pts.values())
                + list(bars_dist) + list(bars_tasks)
            )

        anim = animation.FuncAnimation(
            fig, update,
            frames=len(history),
            interval=self.interval,
            blit=False,
            repeat=False,
        )

        fig.suptitle(
            "Space-Time A*  ·  Prioritised Multi-Robot Planning",
            color="#aaaacc", fontsize=9, y=0.995,
        )
        plt.show()

    # ------------------------------------------------------------------
    # Static performance plots
    # ------------------------------------------------------------------

    def plot_metrics(self) -> None:
        """Bar charts + histogram of final simulation metrics."""
        metrics = self.sim.metrics
        robots  = self.robots

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=_BG_DARK)
        fig.suptitle(
            "Simulation Performance — Final Metrics",
            color="white", fontsize=13, fontweight="bold", y=1.01,
        )

        colors = [self._color(r.id) for r in robots]
        xlabels = [f"R{r.id}" for r in robots]

        # Distance per robot
        ax = axes[0]
        ax.set_facecolor(_PANEL_MED)
        vals = [r.total_distance for r in robots]
        bars = ax.bar(xlabels, vals, color=colors, alpha=0.88, edgecolor="#222", width=0.5)
        ax.set_title("Distance Traveled per Robot", color="white", fontsize=11)
        ax.set_ylabel("Grid Steps", color="#aaa")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.5, str(v),
                    ha="center", va="bottom", color="white", fontsize=9)

        # Tasks per robot
        ax = axes[1]
        ax.set_facecolor(_PANEL_MED)
        vals = [r.tasks_completed for r in robots]
        bars = ax.bar(xlabels, vals, color=colors, alpha=0.88, edgecolor="#222", width=0.5)
        ax.set_title("Tasks Completed per Robot", color="white", fontsize=11)
        ax.set_ylabel("Count", color="#aaa")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, str(v),
                    ha="center", va="bottom", color="white", fontsize=9)

        # Makespan distribution
        ax = axes[2]
        ax.set_facecolor(_PANEL_MED)
        mk = metrics.makespan_list
        if mk:
            nbins = max(5, len(mk) // 2)
            ax.hist(mk, bins=nbins, color="#3498DB", alpha=0.85, edgecolor=_BG_DARK)
            ax.axvline(metrics.avg_makespan, color="#F39C12", linewidth=1.5,
                       linestyle="--", label=f"Mean {metrics.avg_makespan:.1f}")
            ax.legend(facecolor=_PANEL_DARK, labelcolor="white",
                      edgecolor="#555", fontsize=9)
        ax.set_title("Task Makespan Distribution", color="white", fontsize=11)
        ax.set_xlabel("Timesteps to Complete", color="#aaa")
        ax.set_ylabel("Frequency", color="#aaa")

        for ax in axes:
            ax.tick_params(colors="#999")
            for s in ax.spines.values():
                s.set_color("#333")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Final map
    # ------------------------------------------------------------------

    def plot_final_map(self) -> None:
        """Overhead map of the warehouse with robot trails and final positions."""
        fig, ax = plt.subplots(figsize=(11, 10), facecolor=_BG_DARK)
        ax.set_facecolor(_BG_DARK)

        ax.imshow(
            self._bg, origin="upper", aspect="equal",
            extent=[-0.5, self.wh.cols - 0.5, self.wh.rows - 0.5, -0.5],
        )

        # Trail heatmap overlay (optional density layer)
        density = np.zeros((self.wh.rows, self.wh.cols), dtype=float)
        for robot in self.robots:
            for (r, c) in robot.trail:
                density[r, c] += 1

        if density.max() > 0:
            density /= density.max()
            ax.imshow(
                density, origin="upper", aspect="equal", cmap="hot",
                alpha=0.30,
                extent=[-0.5, self.wh.cols - 0.5, self.wh.rows - 0.5, -0.5],
            )

        # Per-robot coloured trails
        for robot in self.robots:
            clr = self._color(robot.id)
            if len(robot.trail) > 1:
                ax.plot(
                    [p[1] for p in robot.trail],
                    [p[0] for p in robot.trail],
                    color=clr, alpha=0.50, linewidth=1.6,
                    solid_capstyle="round",
                    label=f"Robot {robot.id}",
                )
            # Final position
            circ = plt.Circle(
                (robot.pos[1], robot.pos[0]), 0.45,
                color=clr, zorder=6, alpha=0.92,
            )
            ax.add_patch(circ)
            ax.text(
                robot.pos[1], robot.pos[0], str(robot.id),
                color="white", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=7,
            )

        # Task markers
        for task in self.tasks:
            done  = task.status == TaskStatus.COMPLETED
            alpha = 0.22 if done else 0.88
            ax.plot(task.pickup[1],  task.pickup[0],  "D",
                    color="#2ECC71", markersize=9, alpha=alpha, zorder=5)
            ax.plot(task.dropoff[1], task.dropoff[0], "s",
                    color="#E74C3C", markersize=9, alpha=alpha, zorder=5)

        ax.set_xlim(-0.5, self.wh.cols - 0.5)
        ax.set_ylim(self.wh.rows - 0.5, -0.5)
        ax.set_title(
            "Warehouse — Final State & Movement Heatmap",
            color="white", fontsize=13, fontweight="bold",
        )
        ax.tick_params(colors="#777")
        for s in ax.spines.values():
            s.set_color("#444")

        ax.legend(
            facecolor=_PANEL_DARK, edgecolor="#444",
            labelcolor="white", fontsize=9, loc="lower right",
        )
        plt.tight_layout()
        plt.show()
