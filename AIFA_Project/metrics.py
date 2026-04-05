"""
metrics.py — Performance Metrics Tracker
==========================================
Tracks per-robot and global performance metrics for the
multi-robot warehouse simulation.
"""

import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MetricsTracker:
    """
    Records and reports simulation performance metrics.

    Per-robot metrics:
        - distance_traveled
        - tasks_completed
        - idle_steps / busy_steps

    Global metrics:
        - total_tasks_completed (time series)
        - throughput (tasks per step, windowed)
        - average_completion_time
    """

    def __init__(self, n_robots):
        self.n_robots = n_robots

        # Per-robot counters
        self.distance = [0] * n_robots
        self.tasks_completed = [0] * n_robots
        self.idle_steps = [0] * n_robots
        self.busy_steps = [0] * n_robots

        # Task timing
        self.task_start_times = {}       # robot_id → timestep when current task started
        self.completion_times = []       # list of individual task durations

        # Time-series
        self.total_completed_ts = []     # (step, total_completed)
        self.utilization_ts = []         # (step, fraction_busy)
        self._total_completed = 0
        self._step = 0

    def record_step(self, robot_positions_prev, robot_positions_now, robot_busy):
        """Call once per simulation step."""
        self._step += 1
        n_busy = 0
        for rid in range(self.n_robots):
            if robot_busy[rid]:
                self.busy_steps[rid] += 1
                n_busy += 1
                # Distance
                if robot_positions_prev[rid] != robot_positions_now[rid]:
                    self.distance[rid] += 1
            else:
                self.idle_steps[rid] += 1

        utilization = n_busy / max(self.n_robots, 1)
        self.utilization_ts.append((self._step, utilization))
        self.total_completed_ts.append((self._step, self._total_completed))

    def record_task_start(self, robot_id, timestep):
        self.task_start_times[robot_id] = timestep

    def record_task_complete(self, robot_id, timestep):
        self.tasks_completed[robot_id] += 1
        self._total_completed += 1
        start = self.task_start_times.pop(robot_id, timestep)
        duration = timestep - start
        self.completion_times.append(duration)

    # ── Reports ──────────────────────────────────────────────────────────
    def summary_dict(self):
        total_dist = sum(self.distance)
        avg_comp = (sum(self.completion_times) / len(self.completion_times)
                    if self.completion_times else 0)
        throughput = (self._total_completed / self._step
                      if self._step > 0 else 0)

        return {
            "total_steps": self._step,
            "total_distance": total_dist,
            "total_tasks_completed": self._total_completed,
            "average_completion_time": round(avg_comp, 2),
            "throughput_tasks_per_step": round(throughput, 4),
            "per_robot": [
                {
                    "robot_id": rid,
                    "distance": self.distance[rid],
                    "tasks_completed": self.tasks_completed[rid],
                    "idle_steps": self.idle_steps[rid],
                    "busy_steps": self.busy_steps[rid],
                    "utilization": round(
                        self.busy_steps[rid] /
                        max(self.busy_steps[rid] + self.idle_steps[rid], 1), 3),
                }
                for rid in range(self.n_robots)
            ],
        }

    def print_report(self):
        s = self.summary_dict()
        print("\n" + "=" * 60)
        print("  SIMULATION METRICS REPORT")
        print("=" * 60)
        print(f"  Total Steps:            {s['total_steps']}")
        print(f"  Total Distance:         {s['total_distance']}")
        print(f"  Tasks Completed:        {s['total_tasks_completed']}")
        print(f"  Avg Completion Time:    {s['average_completion_time']} steps")
        print(f"  Throughput:             {s['throughput_tasks_per_step']} tasks/step")
        print("-" * 60)
        for r in s["per_robot"]:
            print(f"  Robot {r['robot_id']:>2}: dist={r['distance']:>4}  "
                  f"tasks={r['tasks_completed']:>2}  "
                  f"util={r['utilization']:.1%}")
        print("=" * 60 + "\n")

    # ── Plotting ─────────────────────────────────────────────────────────
    def plot_metrics(self, output_dir="."):
        """Generate and save metric plots as PNGs."""
        if not HAS_MATPLOTLIB:
            print("[metrics] matplotlib not available; skipping plots.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Tasks completed over time
        if self.total_completed_ts:
            steps, completed = zip(*self.total_completed_ts)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(steps, completed, color="#6366f1", linewidth=2)
            ax.set_xlabel("Simulation Step")
            ax.set_ylabel("Tasks Completed")
            ax.set_title("Cumulative Tasks Completed Over Time")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "tasks_completed.png"), dpi=150)
            plt.close(fig)

        # 2. Robot utilization bar chart
        s = self.summary_dict()
        rids = [r["robot_id"] for r in s["per_robot"]]
        utils = [r["utilization"] for r in s["per_robot"]]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#a855f7",
                  "#06b6d4", "#ec4899", "#84cc16"]
        bar_colors = [colors[i % len(colors)] for i in rids]
        ax.bar([f"R{r}" for r in rids], utils, color=bar_colors, edgecolor="white")
        ax.set_ylabel("Utilization")
        ax.set_title("Robot Utilization")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "robot_utilization.png"), dpi=150)
        plt.close(fig)

        # 3. Distance per robot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([f"R{i}" for i in range(self.n_robots)],
               self.distance, color=bar_colors[:self.n_robots], edgecolor="white")
        ax.set_ylabel("Distance (cells)")
        ax.set_title("Distance Traveled per Robot")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "robot_distances.png"), dpi=150)
        plt.close(fig)

        # 4. Utilization over time
        if self.utilization_ts:
            steps, util = zip(*self.utilization_ts)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.fill_between(steps, util, alpha=0.3, color="#6366f1")
            ax.plot(steps, util, color="#6366f1", linewidth=1.5)
            ax.set_xlabel("Simulation Step")
            ax.set_ylabel("Fleet Utilization")
            ax.set_title("Fleet Utilization Over Time")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "utilization_over_time.png"), dpi=150)
            plt.close(fig)

        print(f"[metrics] Plots saved to {output_dir}/")
