"""
robot.py — Robot and Task Data Structures
==========================================
Defines the core domain objects used throughout the simulation:
  - TaskStatus  : Lifecycle states for a pick-and-place task.
  - RobotStatus : Operational states for a warehouse robot.
  - Task        : A single pick-and-place work item.
  - Robot       : A mobile warehouse robot with path-following behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    PENDING    = "pending"     # Created but not yet assigned
    ASSIGNED   = "assigned"    # Assigned to a robot; robot heading to pickup
    IN_PROGRESS = "in_progress" # Robot has picked up item; heading to dropoff
    COMPLETED  = "completed"   # Item delivered


class RobotStatus(Enum):
    IDLE       = "idle"        # No current task
    TO_PICKUP  = "to_pickup"   # Travelling to pickup location
    TO_DROPOFF = "to_dropoff"  # Carrying item to dropoff location


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """
    A pick-and-place work item.

    Attributes
    ----------
    id           : Unique integer identifier.
    pickup       : (row, col) cell where the item must be collected.
    dropoff      : (row, col) cell where the item must be delivered.
    status       : Current lifecycle state.
    assigned_to  : ``robot.id`` of the robot assigned to this task, or None.
    created_time : Simulation timestep at which this task was generated.
    assigned_time: Timestep at which a robot was assigned.
    completed_time: Timestep at which delivery was confirmed.
    """

    id:             int
    pickup:         Tuple[int, int]
    dropoff:        Tuple[int, int]
    status:         TaskStatus       = TaskStatus.PENDING
    assigned_to:    Optional[int]    = None
    created_time:   int              = 0
    assigned_time:  Optional[int]    = None
    completed_time: Optional[int]    = None

    @property
    def makespan(self) -> Optional[int]:
        """Timesteps from creation to completion, or None if not yet done."""
        if self.completed_time is not None and self.created_time is not None:
            return self.completed_time - self.created_time
        return None


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

@dataclass
class Robot:
    """
    A mobile warehouse robot.

    The robot follows a pre-computed path stored as a list of (row, col)
    positions.  ``path_index`` tracks how far the robot has progressed.

    Attributes
    ----------
    id             : Unique integer identifier (0-based).
    pos            : Current (row, col) position on the grid.
    color          : Hex colour string used for visualisation.
    status         : Operational state.
    path           : Planned sequence of (row, col) positions.
    path_index     : Index into ``path`` of the robot's current position.
    current_task   : The task this robot is currently executing, or None.
    total_distance : Cumulative grid steps moved (waits excluded).
    tasks_completed: Number of tasks fully delivered.
    trail          : Full positional history for trail visualisation.
    """

    id:              int
    pos:             Tuple[int, int]
    color:           str
    status:          RobotStatus       = RobotStatus.IDLE
    path:            List[Tuple[int, int]] = field(default_factory=list)
    path_index:      int               = 0
    current_task:    Optional[Task]    = None
    total_distance:  int               = 0
    tasks_completed: int               = 0
    trail:           List[Tuple[int, int]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_idle(self) -> bool:
        return self.status == RobotStatus.IDLE

    @property
    def at_path_end(self) -> bool:
        """True when the robot has reached the last waypoint in its path."""
        return len(self.path) == 0 or self.path_index >= len(self.path) - 1

    # ------------------------------------------------------------------
    # Path management
    # ------------------------------------------------------------------

    def assign_path(self, path: List[Tuple[int, int]]) -> None:
        """Replace the robot's path and reset the index to the beginning."""
        self.path       = path
        self.path_index = 0

    def move_one_step(self) -> bool:
        """
        Advance the robot one position along its current path.

        Returns
        -------
        bool
            ``True`` if the robot moved or waited; ``False`` if already at
            the end of the path (or path is empty).

        Side-effects
        ------------
        * Updates ``self.pos`` and ``self.path_index``.
        * Increments ``self.total_distance`` only for actual spatial moves
          (not waits — where path[i] == path[i-1]).
        * Appends new position to ``self.trail`` on spatial moves.
        """
        if not self.path or self.path_index >= len(self.path) - 1:
            return False

        old_pos         = self.pos
        self.path_index += 1
        new_pos         = self.path[self.path_index]
        self.pos        = new_pos

        if new_pos != old_pos:            # actual spatial move (not a wait)
            self.total_distance += 1
            self.trail.append(new_pos)

        return True

    def reset_path(self) -> None:
        """Clear the current path (called after task completion)."""
        self.path       = [self.pos]
        self.path_index = 0
