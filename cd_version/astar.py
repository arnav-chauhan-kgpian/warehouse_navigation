"""
astar.py — A* and Space-Time A* Path Planning
==============================================
Provides two path planners:
  - astar()             : Standard A* on a static 2D grid.
  - space_time_astar()  : Space-Time A* that avoids reservations made
                          by other robots, enabling collision-free
                          multi-robot path planning via prioritised planning.
"""

import heapq
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Grid = List[List[int]]                      # 0 = free, 1 = obstacle
Pos  = Tuple[int, int]                      # (row, col)
ReservationTable = Dict[Tuple[int, int, int], int]  # (row, col, t) → robot_id

# Movement directions: up/down/left/right
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# Movement + wait-in-place
_DIRS_WAIT = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def manhattan(a: Pos, b: Pos) -> int:
    """Admissible Manhattan-distance heuristic for grid movement."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ---------------------------------------------------------------------------
# Standard A*
# ---------------------------------------------------------------------------

def astar(grid: Grid, start: Pos, goal: Pos) -> Optional[List[Pos]]:
    """
    Standard A* on a static 2D grid.

    Parameters
    ----------
    grid  : 2-D list where 0 = free cell, 1 = obstacle.
    start : (row, col) starting position (must be free).
    goal  : (row, col) goal position (must be free).

    Returns
    -------
    Ordered list of (row, col) positions from *start* to *goal* inclusive,
    or ``None`` if no path exists.
    """
    rows, cols = len(grid), len(grid[0])

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    if not in_bounds(*start) or not in_bounds(*goal):
        return None
    if grid[goal[0]][goal[1]] != 0:
        return None
    if start == goal:
        return [start]

    # heap entries: (f, tie_break, g, pos, path)
    counter = 0
    heap = [(manhattan(start, goal), counter, 0, start, [start])]
    best_g: Dict[Pos, int] = {}

    while heap:
        f, _, g, pos, path = heapq.heappop(heap)

        if pos == goal:
            return path

        if best_g.get(pos, float('inf')) <= g:
            continue
        best_g[pos] = g

        r, c = pos
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and grid[nr][nc] == 0:
                npos  = (nr, nc)
                ng    = g + 1
                nf    = ng + manhattan(npos, goal)
                if ng < best_g.get(npos, float('inf')):
                    counter += 1
                    heapq.heappush(heap, (nf, counter, ng, npos, path + [npos]))

    return None  # unreachable


# ---------------------------------------------------------------------------
# Space-Time A*
# ---------------------------------------------------------------------------

def space_time_astar(
    grid:              Grid,
    start:             Pos,
    goal:              Pos,
    reservation_table: ReservationTable,
    robot_id:          int,
    start_time:        int = 0,
    max_time:          int = 300,
) -> Optional[List[Pos]]:
    """
    Space-Time A* — finds a collision-free path in the joint space (row, col, time).

    A cell (r, c) at timestep t is considered *occupied* if
    ``reservation_table[(r, c, t)]`` exists and belongs to a *different* robot.
    The planner also detects edge (swap) conflicts where two robots would
    pass through each other.

    The robot may *wait* in place (cost 1) to let a blocker pass.

    Parameters
    ----------
    grid              : Static 2-D obstacle grid.
    start             : Starting (row, col) position at ``start_time``.
    goal              : Goal (row, col).
    reservation_table : Shared dict mapping (r, c, t) → robot_id.
    robot_id          : ID of the robot being planned for (ignored in lookup).
    start_time        : Absolute timestep the robot departs.
    max_time          : Planning horizon; search halts past this time.

    Returns
    -------
    List of (row, col) positions, one per timestep beginning at ``start_time``,
    ending when the robot first arrives at *goal*.
    Returns ``None`` if no conflict-free path found within the horizon.
    """
    rows, cols = len(grid), len(grid[0])

    if grid[goal[0]][goal[1]] != 0:
        return None
    if start == goal:
        return [start]

    def cell_free(r: int, c: int, t: int) -> bool:
        """True when (r,c) is unobstructed and not reserved by another robot."""
        if not (0 <= r < rows and 0 <= c < cols):
            return False
        if grid[r][c] != 0:
            return False
        owner = reservation_table.get((r, c, t))
        return owner is None or owner == robot_id

    def swap_conflict(r1: int, c1: int, r2: int, c2: int, t: int) -> bool:
        """
        Detects edge conflict: robot moves (r1,c1)→(r2,c2) at t→t+1,
        while another robot moves (r2,c2)→(r1,c1) simultaneously.
        """
        other_at_dest_now  = reservation_table.get((r2, c2, t))
        other_at_src_next  = reservation_table.get((r1, c1, t + 1))
        return (
            other_at_dest_now  is not None and other_at_dest_now  != robot_id and
            other_at_src_next  is not None and other_at_src_next  != robot_id and
            other_at_dest_now == other_at_src_next
        )

    # heap: (f, tie_break, g, (r,c,t), path)
    counter = 0
    init_h  = manhattan(start, goal)
    heap    = [(init_h, counter, 0, (start[0], start[1], start_time), [start])]
    best_g: Dict[Tuple[int, int, int], int] = {}

    while heap:
        f, _, g, state, path = heapq.heappop(heap)
        r, c, t = state

        if (r, c) == goal:
            return path

        if t >= max_time:
            continue

        if best_g.get(state, float('inf')) <= g:
            continue
        best_g[state] = g

        for dr, dc in _DIRS_WAIT:
            nr, nc, nt = r + dr, c + dc, t + 1

            # Skip waits that duplicate an already-explored state at same time
            is_wait = (dr == 0 and dc == 0)

            if not cell_free(nr, nc, nt):
                continue
            if not is_wait and swap_conflict(r, c, nr, nc, t):
                continue

            nstate = (nr, nc, nt)
            ng     = g + 1          # uniform cost: move or wait costs 1
            nf     = ng + manhattan((nr, nc), goal)

            if ng < best_g.get(nstate, float('inf')):
                counter += 1
                heapq.heappush(heap, (nf, counter, ng, nstate, path + [(nr, nc)]))

    return None
