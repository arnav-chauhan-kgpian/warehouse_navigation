"""
simulation.py — Multi-Robot Warehouse Simulation Engine
========================================================
Implements the discrete-time simulation loop.

Each timestep:
  1. Inject dynamically scheduled tasks.
  2. Allocate pending tasks to idle robots (greedy or Hungarian).
  3. For each newly assigned robot, plan a collision-free path using
     Space-Time A* consulting the shared reservation table.
  4. Advance every robot one step along its current path.
  5. Detect waypoint arrivals (pickup / dropoff) and trigger replanning.
  6. Snapshot the world state for later visualisation replay.

Collision-avoidance design
--------------------------
A reservation table maps (row, col, timestep) → robot_id.
Robots plan sequentially (prioritised planning): earlier-priority robots
lock cells in space-time; later-priority robots route around them.
After each planning event the robot's new path is written into the table.
Old reservations for a robot are cleared before replanning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from astar import astar, space_time_astar
from robot import Robot, RobotStatus, Task, TaskStatus
from task_allocator import greedy_allocate, hungarian_allocate
from warehouse import Warehouse

# (row, col, timestep) → robot_id
ReservationTable = Dict[Tuple[int, int, int], int]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class SimMetrics:
    """Aggregated performance statistics collected at simulation end."""

    total_tasks:      int = 0
    completed_tasks:  int = 0
    total_distance:   int = 0
    simulation_steps: int = 0
    makespan_list:    List[int] = field(default_factory=list)
    distance_per_robot: Dict[int, int] = field(default_factory=dict)
    tasks_per_robot:    Dict[int, int] = field(default_factory=dict)

    @property
    def avg_makespan(self) -> float:
        if not self.makespan_list:
            return 0.0
        return sum(self.makespan_list) / len(self.makespan_list)

    @property
    def throughput(self) -> float:
        """Tasks completed per 100 simulation steps."""
        if self.simulation_steps == 0:
            return 0.0
        return self.completed_tasks / self.simulation_steps * 100

    def __str__(self) -> str:
        lines = [
            "╔══════════════════════════════════════╗",
            "║      SIMULATION PERFORMANCE REPORT   ║",
            "╠══════════════════════════════════════╣",
            f"║  Total Tasks          : {self.total_tasks:<13}║",
            f"║  Completed Tasks      : {self.completed_tasks:<13}║",
            f"║  Success Rate         : {self.completed_tasks/max(1,self.total_tasks)*100:<13.1f}║",
            f"║  Total Steps          : {self.simulation_steps:<13}║",
            f"║  Total Distance       : {self.total_distance:<13}║",
            f"║  Avg Task Makespan    : {self.avg_makespan:<13.1f}║",
            f"║  Throughput           : {self.throughput:<13.2f}║",
            "╚══════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

_PRUNE_INTERVAL: int = 25

class Simulation:
    """
    Discrete-time multi-robot warehouse simulation.

    Parameters
    ----------
    warehouse         : The warehouse environment.
    robots            : List of Robot objects (initial positions set).
    tasks             : Initial batch of pick-and-place tasks.
    allocation_method : ``'greedy'`` or ``'hungarian'``.
    dynamic_tasks     : Sequence of ``(arrival_step, Task)`` pairs for tasks
                        that appear mid-simulation.
    max_steps         : Hard upper bound on simulation length.
    replan_horizon    : Max future steps considered in Space-Time A*.
    goal_reservation_pad : Extra timesteps the goal cell is reserved after
                           path end to prevent another robot from squatting.
    verbose           : Print progress to stdout.
    """

    def __init__(
        self,
        warehouse:            Warehouse,
        robots:               List[Robot],
        tasks:                List[Task],
        allocation_method:    str                           = "greedy",
        dynamic_tasks:        List[Tuple[int, Task]]       = None,
        max_steps:            int                           = 600,
        replan_horizon:       int                           = 120,
        goal_reservation_pad: int                           = 25,
        verbose:              bool                          = True,
    ) -> None:
        self.warehouse            = warehouse
        self.robots               = robots
        self.tasks: List[Task]    = list(tasks)
        self.allocation_method    = allocation_method
        self.dynamic_tasks        = sorted(dynamic_tasks or [], key=lambda x: x[0])
        self.max_steps            = max_steps
        self.replan_horizon       = replan_horizon
        self.goal_pad             = goal_reservation_pad
        self.verbose              = verbose

        self.step: int                = 0
        self.reservation: ReservationTable = {}
        self.metrics                  = SimMetrics(total_tasks=len(tasks))

        # Replay history — list of world snapshots (one per step)
        self.history: List[Dict] = []

        # Initialise robot trails with spawn position
        for robot in self.robots:
            robot.trail = [robot.pos]

    # ======================================================================
    # Public API
    # ======================================================================

    def run(self) -> SimMetrics:
        """Execute the full simulation and return performance metrics."""
        t_start = time.perf_counter()

        if self.verbose:
            print(
                f"\n  Robots: {len(self.robots)}  |  "
                f"Static tasks: {len(self.tasks)}  |  "
                f"Dynamic tasks: {len(self.dynamic_tasks)}  |  "
                f"Method: {self.allocation_method}\n"
                + "─" * 60
            )

        for step in range(self.max_steps):
            self.step = step

            self._inject_dynamic_tasks()
            self._refresh_idle_reservations()
            self._allocate_tasks()
            self._step_all_robots()
            self._check_completions()
            self._save_snapshot()

            if step % _PRUNE_INTERVAL == 0:
                self._prune_old_reservations()

            if self.verbose and step % 25 == 0:
                done = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
                active = sum(1 for r in self.robots if not r.is_idle)
                print(
                    f"  Step {step:4d}  |  "
                    f"Done {done:2d}/{len(self.tasks)}  |  "
                    f"Active robots {active}/{len(self.robots)}"
                )

            if self._all_done():
                if self.verbose:
                    print(f"\n  ✓ All tasks completed at step {step}!")
                    print(f"  Finished {self.max_steps - step} steps early")
                break

        # Finalise metrics
        self.metrics.simulation_steps   = self.step
        self.metrics.total_distance     = sum(r.total_distance for r in self.robots)
        self.metrics.completed_tasks    = sum(
            1 for t in self.tasks if t.status == TaskStatus.COMPLETED
        )
        self.metrics.distance_per_robot = {r.id: r.total_distance for r in self.robots}
        self.metrics.tasks_per_robot    = {r.id: r.tasks_completed for r in self.robots}

        elapsed = time.perf_counter() - t_start
        if self.verbose:
            print(f"\n  Wall-clock time: {elapsed:.2f}s")
            print(self.metrics)

        return self.metrics

    # ======================================================================
    # Step phases
    # ======================================================================

    def _inject_dynamic_tasks(self) -> None:
        """Add tasks whose scheduled arrival matches the current step."""
        for arrival_step, task in self.dynamic_tasks:
            if arrival_step == self.step and task not in self.tasks:
                self.tasks.append(task)
                self.metrics.total_tasks += 1
                if self.verbose:
                    print(
                        f"  [Step {self.step:4d}]  ⚡ Dynamic task {task.id} arrived  "
                        f"{task.pickup} → {task.dropoff}"
                    )

    def _refresh_idle_reservations(self) -> None:
        """
        Ensure every idle robot's current cell is reserved for the next
        ``goal_pad`` timesteps, so active robots' planners route around them.
        Called once per step before allocation / planning.
        """
        for robot in self.robots:
            if robot.is_idle:
                r, c = robot.pos
                for dt in range(self.goal_pad + 10):
                    key = (r, c, self.step + dt)
                    existing = self.reservation.get(key)
                    if existing is not None and existing != robot.id:
                        continue
                    if existing is None or existing == robot.id:
                        self.reservation[key] = robot.id

    def _allocate_tasks(self) -> None:
        """Match idle robots to pending tasks, then plan paths."""
        if self.allocation_method == "hungarian":
            pairs = hungarian_allocate(self.robots, self.tasks, self.step)
        else:
            pairs = greedy_allocate(self.robots, self.tasks, self.step)

        # Sort pairs by manhattan(robot.pos, task.pickup) ascending
        def manhattan(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        pairs.sort(key=lambda pair: manhattan(pair[0].pos, pair[1].pickup))

        for robot, task in pairs:
            self._assign(robot, task)

    def _step_all_robots(self) -> None:
        """Advance every robot one step along its planned path."""
        for robot in self.robots:
            if robot.path:
                robot.move_one_step()

    def _check_completions(self) -> None:
        """Detect pickup / dropoff arrivals and handle transitions."""
        for robot in self.robots:
            if robot.current_task is None:
                continue

            task = robot.current_task

            # ── Arrived at pickup ──────────────────────────────────────────
            if (
                robot.status == RobotStatus.TO_PICKUP
                and robot.pos == task.pickup
                and robot.at_path_end
            ):
                task.status      = TaskStatus.IN_PROGRESS
                robot.status     = RobotStatus.TO_DROPOFF
                self._clear_reservations(robot.id)

                path = self._plan(robot, task.dropoff)
                if path:
                    robot.assign_path(path)
                    self._reserve(robot.id, path)
                    if self.verbose:
                        print(
                            f"  [Step {self.step:4d}]  📦 Robot {robot.id} "
                            f"picked up Task {task.id} → heading to {task.dropoff}"
                        )
                else:
                    if self.verbose:
                        print(
                            f"  [WARNING] Robot {robot.id}: no path to dropoff "
                            f"{task.dropoff} — task {task.id} re-queued to PENDING."
                        )
                    task.status        = TaskStatus.PENDING
                    task.assigned_to   = None
                    robot.current_task = None
                    robot.status       = RobotStatus.IDLE
                    robot.reset_path()

            # ── Arrived at dropoff ─────────────────────────────────────────
            elif (
                robot.status == RobotStatus.TO_DROPOFF
                and robot.pos == task.dropoff
                and robot.at_path_end
            ):
                task.status          = TaskStatus.COMPLETED
                task.completed_time  = self.step
                robot.tasks_completed += 1
                robot.current_task   = None
                robot.status         = RobotStatus.IDLE
                robot.reset_path()
                self._clear_reservations(robot.id)
                # Reserve idle position so moving robots route around this robot
                r_idle, c_idle = robot.pos
                for extra in range(self.goal_pad + 10):
                    key = (r_idle, c_idle, self.step + extra)
                    if key not in self.reservation:
                        self.reservation[key] = robot.id

                if task.makespan is not None:
                    self.metrics.makespan_list.append(task.makespan)

                if self.verbose:
                    print(
                        f"  [Step {self.step:4d}]  ✓ Robot {robot.id} "
                        f"completed Task {task.id}  "
                        f"(makespan {task.makespan} steps)"
                    )

    # ======================================================================
    # Path planning helpers
    # ======================================================================

    def _assign(self, robot: Robot, task: Task) -> None:
        """Update book-keeping and plan the robot's first leg (→ pickup)."""
        task.status       = TaskStatus.ASSIGNED
        task.assigned_to  = robot.id
        task.assigned_time = self.step
        robot.current_task = task
        robot.status       = RobotStatus.TO_PICKUP

        # Clear any idle-position reservation before planning new path
        self._clear_reservations(robot.id)

        path = self._plan(robot, task.pickup)
        if path:
            robot.assign_path(path)
            self._reserve(robot.id, path)
            if self.verbose:
                print(
                    f"  [Step {self.step:4d}]  → Robot {robot.id} assigned "
                    f"Task {task.id}  {task.pickup} → {task.dropoff}"
                )
        else:
            # Rollback: re-queue the task
            if self.verbose:
                print(
                    f"  [WARNING] Robot {robot.id}: no path to pickup "
                    f"{task.pickup} — task {task.id} re-queued."
                )
            task.status      = TaskStatus.PENDING
            task.assigned_to = None
            robot.current_task = None
            robot.status       = RobotStatus.IDLE

    def _plan(
        self, robot: Robot, goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Plan a collision-free path for *robot* to *goal*.

        First attempts Space-Time A* against the reservation table.
        Falls back to standard A* (ignores other robots) if the horizon
        is exhausted — this may rarely produce conflicts but guarantees
        progress.
        """
        path = space_time_astar(
            grid              = self.warehouse.grid,
            start             = robot.pos,
            goal              = goal,
            reservation_table = self.reservation,
            robot_id          = robot.id,
            start_time        = self.step,
            max_time          = self.step + self.replan_horizon,
        )
        if path is None:
            if self.verbose:
                print(
                    f"  [Planner] Robot {robot.id}: STA* failed, "
                    f"using standard A* fallback."
                )
            path = astar(self.warehouse.grid, robot.pos, goal)
        return path

    def _reserve(self, robot_id: int, path: List[Tuple[int, int]]) -> None:
        """
        Write the planned path into the reservation table.

        Each position path[i] is reserved at absolute timestep
        ``self.step + i``.  The goal cell is additionally reserved for
        ``goal_pad`` extra steps to prevent another robot from occupying
        it just before this robot arrives.
        """
        for i, (r, c) in enumerate(path):
            self.reservation[(r, c, self.step + i)] = robot_id

        # Extend reservation at goal
        if path:
            gr, gc    = path[-1]
            last_t    = self.step + len(path) - 1
            for extra in range(1, self.goal_pad + 1):
                key = (gr, gc, last_t + extra)
                if key not in self.reservation:
                    self.reservation[key] = robot_id

    def _prune_old_reservations(self) -> None:
        """
        Remove reservation entries whose timestep has already passed.
        Keeps the dict bounded to at most ~(replan_horizon * n_robots) entries.
        Called every _PRUNE_INTERVAL steps from run().
        """
        stale = [k for k in self.reservation if k[2] < self.step]
        for k in stale:
            del self.reservation[k]

    def _clear_reservations(self, robot_id: int) -> None:
        """Remove all reservation-table entries belonging to *robot_id*."""
        keys = [k for k, v in self.reservation.items() if v == robot_id]
        for k in keys:
            del self.reservation[k]

    # ======================================================================
    # Termination & snapshot
    # ======================================================================

    def _all_done(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    def _save_snapshot(self) -> None:
        """
        Capture a lightweight snapshot of the world state.
        Stored in ``self.history`` for visualisation replay.
        """
        snap: Dict = {
            "step":            self.step,
            "robot_positions": {r.id: r.pos          for r in self.robots},
            "robot_statuses":  {r.id: r.status        for r in self.robots},
            "robot_paths":     {r.id: list(r.path) if r.path else [] for r in self.robots},
            "robot_path_idx":  {r.id: r.path_index    for r in self.robots},
            "robot_trails":    {r.id: list(r.trail[-200:]) for r in self.robots},
            "robot_distances": {r.id: r.total_distance for r in self.robots},
            "robot_task_counts":{r.id: r.tasks_completed for r in self.robots},
            "task_statuses":   {t.id: t.status        for t in self.tasks},
            "n_completed":     sum(
                1 for t in self.tasks if t.status == TaskStatus.COMPLETED
            ),
        }
        self.history.append(snap)
