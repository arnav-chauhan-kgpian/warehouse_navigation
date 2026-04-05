"""
robot_task_alocat.py — Multi-Robot Task Allocator
====================================================
Full pick-and-place lifecycle with FSM integration,
cost-matrix greedy allocation, and dynamic task support.
"""

import numpy as np
from navigation import Navigation
from state_diagram import RobotState, RobotFSM, print_snapshot
from metrics import MetricsTracker


class RobotTaskAllocator:
    """
    Manages a fleet of warehouse robots executing pick-and-place tasks.

    Each task is a (pickup, dropoff) pair.  Robots transition through:
        IDLE → MOVING_TO_PICKUP → PICKING_UP → MOVING_TO_DROPOFF → DROPPING_OFF → IDLE

    Features:
        • Collision-free prioritized planning (time-expanded A*)
        • Cost-matrix greedy assignment (nearest pickup)
        • Dynamic task injection via add_tasks()
        • Integrated FSM and metrics tracking
    """

    PICK_DURATION = 2   # timesteps to simulate picking up
    DROP_DURATION = 2   # timesteps to simulate dropping off

    def __init__(self, warehouse, robots, tasks):
        """
        Parameters
        ----------
        warehouse : Warehouse
        robots    : list of (row, col) initial positions
        tasks     : list of (pickup, dropoff) tuples
        """
        self.warehouse = warehouse
        self.robots = [tuple(r) for r in np.array(robots)]
        self.n_robots = len(self.robots)
        self.navigation = Navigation(warehouse, max_time_steps=300)

        # Task queue: list of (pickup_pos, dropoff_pos)
        self.pending_tasks = list(tasks)
        self.completed_tasks = []

        # Per-robot state
        self.robot_busy = [False] * self.n_robots
        self.robot_task = [None] * self.n_robots     # (pickup, dropoff) or None
        self.robot_phase = [None] * self.n_robots    # 'to_pickup' | 'picking' | 'to_dropoff' | 'dropping'
        self.robot_path = [[] for _ in range(self.n_robots)]
        self.robot_step_idx = [0] * self.n_robots
        self.robot_action_timer = [0] * self.n_robots  # countdown for pick/drop actions
        self.prev_positions = list(self.robots)

        # FSM per robot
        self.fsm = {i: RobotFSM(i) for i in range(self.n_robots)}

        # Metrics
        self.metrics = MetricsTracker(self.n_robots)
        self.timestep = 0

        # Snapshot logging
        self.state_snapshots = []   # list of (timestep, snapshot_text)

    # ── Dynamic task injection ───────────────────────────────────────────
    def add_tasks(self, new_tasks):
        """Add tasks at runtime.  *new_tasks*: list of (pickup, dropoff)."""
        self.pending_tasks.extend(new_tasks)

    # ── Reservation building ─────────────────────────────────────────────
    def _build_reservations(self):
        """Build reservations from all busy robots' remaining paths."""
        reserved_cells = set()
        reserved_edges = set()

        for rid in range(self.n_robots):
            if not self.robot_busy[rid]:
                continue
            if self.robot_phase[rid] in ('picking', 'dropping'):
                # Robot is stationary — reserve its cell
                pos = self.robots[rid]
                for t in range(self.navigation.max_time_steps + 1):
                    reserved_cells.add((pos, t))
                continue

            path = self.robot_path[rid]
            step_idx = self.robot_step_idx[rid]
            if not path:
                continue

            remaining = path[step_idx:]
            for t, pos in enumerate(remaining):
                reserved_cells.add((pos, t))
                if t > 0:
                    reserved_cells.add((remaining[t - 1], pos, t))
                    reserved_edges.add((remaining[t - 1], pos, t))

            final_pos = remaining[-1]
            for t in range(len(remaining), self.navigation.max_time_steps + 1):
                reserved_cells.add((final_pos, t))

        return reserved_cells, reserved_edges

    # ── Task allocation ──────────────────────────────────────────────────
    def _assign_idle_robots(self):
        """Assign pending tasks to idle robots using nearest-pickup greedy."""
        if not self.pending_tasks:
            return

        reserved_cells, reserved_edges = self._build_reservations()

        # Build cost matrix: (robot_id, task_idx, manhattan_to_pickup)
        idle_robots = [rid for rid in range(self.n_robots)
                       if not self.robot_busy[rid]]
        if not idle_robots:
            return

        # Sort by Manhattan distance to pickup (greedy)
        assignments = []
        for rid in idle_robots:
            for tidx, (pickup, dropoff) in enumerate(self.pending_tasks):
                dist = Navigation.manhattan(self.robots[rid], pickup)
                assignments.append((dist, rid, tidx))
        assignments.sort()

        assigned_robots = set()
        assigned_tasks = set()

        for dist, rid, tidx in assignments:
            if rid in assigned_robots or tidx in assigned_tasks:
                continue

            pickup, dropoff = self.pending_tasks[tidx]

            # Plan path: current → pickup
            path = self.navigation.a_star_time_expanded(
                self.robots[rid], pickup, reserved_cells, reserved_edges
            )
            if path is None:
                continue

            # Assign
            self.robot_task[rid] = (pickup, dropoff)
            self.robot_phase[rid] = 'to_pickup'
            self.robot_path[rid] = path
            self.robot_step_idx[rid] = 0
            self.robot_busy[rid] = True

            # FSM transition
            self.fsm[rid].transition(RobotState.MOVING_TO_PICKUP, self.timestep)

            # Metrics
            self.metrics.record_task_start(rid, self.timestep)

            # Reserve this path
            for t, pos in enumerate(path):
                reserved_cells.add((pos, t))
                if t > 0:
                    reserved_edges.add((path[t - 1], pos, t))
            final = path[-1]
            for t in range(len(path), self.navigation.max_time_steps + 1):
                reserved_cells.add((final, t))

            assigned_robots.add(rid)
            assigned_tasks.add(tidx)

        # Remove assigned tasks (in reverse order to preserve indices)
        for tidx in sorted(assigned_tasks, reverse=True):
            self.pending_tasks.pop(tidx)

    # ── Phase transitions ────────────────────────────────────────────────
    def _handle_arrival(self, rid):
        """Called when robot *rid* reaches end of its current path."""
        phase = self.robot_phase[rid]

        if phase == 'to_pickup':
            # Arrived at pickup — start picking
            self.robot_phase[rid] = 'picking'
            self.robot_action_timer[rid] = self.PICK_DURATION
            self.fsm[rid].transition(RobotState.PICKING_UP, self.timestep)

        elif phase == 'to_dropoff':
            # Arrived at dropoff — start dropping
            self.robot_phase[rid] = 'dropping'
            self.robot_action_timer[rid] = self.DROP_DURATION
            self.fsm[rid].transition(RobotState.DROPPING_OFF, self.timestep)

    def _handle_action_complete(self, rid):
        """Called when pick/drop timer reaches zero."""
        phase = self.robot_phase[rid]

        if phase == 'picking':
            # Done picking — plan path to dropoff
            pickup, dropoff = self.robot_task[rid]
            reserved_cells, reserved_edges = self._build_reservations()
            path = self.navigation.a_star_time_expanded(
                self.robots[rid], dropoff, reserved_cells, reserved_edges
            )
            if path is None:
                # Retry next step
                self.robot_action_timer[rid] = 1
                return

            self.robot_phase[rid] = 'to_dropoff'
            self.robot_path[rid] = path
            self.robot_step_idx[rid] = 0
            self.fsm[rid].transition(RobotState.MOVING_TO_DROPOFF, self.timestep)

        elif phase == 'dropping':
            # Done dropping — task complete!
            task = self.robot_task[rid]
            self.completed_tasks.append(task)
            self.metrics.record_task_complete(rid, self.timestep)

            self.robot_busy[rid] = False
            self.robot_task[rid] = None
            self.robot_phase[rid] = None
            self.robot_path[rid] = []
            self.robot_step_idx[rid] = 0
            self.fsm[rid].transition(RobotState.IDLE, self.timestep)

    # ── Main step ────────────────────────────────────────────────────────
    def step(self):
        """Advance the simulation by one timestep."""
        self.timestep += 1
        self.prev_positions = list(self.robots)

        # Assign idle robots
        self._assign_idle_robots()

        # Advance each robot
        for rid in range(self.n_robots):
            if not self.robot_busy[rid]:
                continue

            phase = self.robot_phase[rid]

            if phase in ('picking', 'dropping'):
                # Countdown timer
                self.robot_action_timer[rid] -= 1
                if self.robot_action_timer[rid] <= 0:
                    self._handle_action_complete(rid)
                continue

            # Moving along path
            path = self.robot_path[rid]
            next_idx = self.robot_step_idx[rid] + 1

            if next_idx < len(path):
                self.robot_step_idx[rid] = next_idx
                self.robots[rid] = path[next_idx]
            
            # Check if arrived at destination
            if self.robot_step_idx[rid] >= len(path) - 1:
                self._handle_arrival(rid)

        # Record metrics
        self.metrics.record_step(self.prev_positions, self.robots, self.robot_busy)

        # Log state snapshot at crucial moments
        if self.timestep == 1 or self.timestep % 25 == 0:
            self._log_snapshot()

    def _log_snapshot(self):
        """Record a state snapshot for crucial-turn documentation."""
        snapshot = {}
        for rid in range(self.n_robots):
            snapshot[rid] = {
                "position": self.robots[rid],
                "state": self.fsm[rid].state.name,
                "task": self.robot_task[rid],
                "phase": self.robot_phase[rid],
            }
        self.state_snapshots.append((self.timestep, snapshot))
        print_snapshot(self.fsm, self.timestep)

    # ── Query helpers ────────────────────────────────────────────────────
    def active_targets(self):
        """Return list of current target positions (pickup or dropoff)."""
        targets = []
        for rid in range(self.n_robots):
            if not self.robot_busy[rid] or self.robot_task[rid] is None:
                continue
            pickup, dropoff = self.robot_task[rid]
            phase = self.robot_phase[rid]
            if phase in ('to_pickup', 'picking'):
                targets.append(pickup)
            elif phase in ('to_dropoff', 'dropping'):
                targets.append(dropoff)
        return targets

    def active_pickups(self):
        """Return pickup positions of in-progress tasks."""
        return [self.robot_task[rid][0]
                for rid in range(self.n_robots)
                if self.robot_busy[rid] and self.robot_task[rid]
                and self.robot_phase[rid] in ('to_pickup', 'picking')]

    def active_dropoffs(self):
        """Return dropoff positions of in-progress tasks."""
        return [self.robot_task[rid][1]
                for rid in range(self.n_robots)
                if self.robot_busy[rid] and self.robot_task[rid]
                and self.robot_phase[rid] in ('to_dropoff', 'dropping')]

    def pending_pickups(self):
        """Return pickup positions of pending (unassigned) tasks."""
        return [t[0] for t in self.pending_tasks]

    def pending_dropoffs(self):
        """Return dropoff positions of pending (unassigned) tasks."""
        return [t[1] for t in self.pending_tasks]

    def is_done(self):
        """True if all tasks completed and no robots busy."""
        return not self.pending_tasks and not any(self.robot_busy)

    # ── Run until done ───────────────────────────────────────────────────
    def run_until_done(self, max_steps=2000):
        """Run the full simulation.  Returns summary dict."""
        for _ in range(max_steps):
            if self.is_done():
                break
            self.step()

        return {
            "total_steps": self.timestep,
            "robot_positions": list(self.robots),
            "completed_tasks": list(self.completed_tasks),
            "pending_tasks": list(self.pending_tasks),
            "metrics": self.metrics.summary_dict(),
        }