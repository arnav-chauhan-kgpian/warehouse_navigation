import os

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

# --- main.py ---
main_py = read_file('cd_version/main.py')

# BUG 1
main_py = main_py.replace('''for _ in range(50):
            pu = wh.random_free_cell(exclude=list(used))
            do = wh.random_free_cell(exclude=list(used) + [pu])
            break''', '''for _ in range(50):
            try:
                pu = wh.random_free_cell(exclude=list(used))
                do = wh.random_free_cell(exclude=list(used) + [pu])
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(
                f"Could not place task {id_offset + i} after 50 attempts "
                f"— too many cells already occupied. Reduce num_robots or num_tasks."
            )''')

# BUG 2
main_py = main_py.replace('''rng   = random.Random(seed + 200)
    used  = set(occupied)
    tasks = []

    for i in range(count):''', '''saved_rng = wh._rng
    wh._rng = random.Random(seed + 200)
    try:
        used  = set(occupied)
        tasks = []

        for i in range(count):''')
main_py = main_py.replace('''        print(
            f"    Task {id_offset+i:2d}  {pu} → {do}"
            + (f"  (appears at step {created_at})" if created_at else "")
        )

    return tasks''', '''        print(
            f"    Task {id_offset+i:2d}  {pu} → {do}"
            + (f"  (appears at step {created_at})" if created_at else "")
        )

    finally:
        wh._rng = saved_rng

    return tasks''')

# IMPROVEMENT 1: CONFIG validation function
main_py = main_py.replace('''def main() -> None:
    cfg  = CONFIG
    seed = cfg["seed"]''', '''def _validate_config(cfg: dict) -> None:
    if cfg["num_robots"] < 1:
        raise ValueError("num_robots must be >= 1")
    if cfg["num_static_tasks"] < 0 or cfg["num_dynamic_tasks"] < 0:
        raise ValueError("num_static_tasks and num_dynamic_tasks must be >= 0")
    if cfg["max_steps"] < 10:
        raise ValueError("max_steps must be >= 10")
    if cfg["replan_horizon"] > cfg["max_steps"]:
        raise ValueError("replan_horizon must be <= max_steps")
    if cfg["allocation_method"] not in {"greedy", "hungarian"}:
        raise ValueError("allocation_method must be 'greedy' or 'hungarian'")
    if not isinstance(cfg["seed"], int):
        raise ValueError("seed must be an integer")

def main() -> None:
    cfg  = CONFIG
    _validate_config(cfg)
    seed = cfg["seed"]''')

# IMPROVEMENT 2 & Constants
main_py = main_py.replace('''def generate_tasks(
    count:      int,
    wh:         Warehouse,
    occupied:   List[Tuple[int, int]],
    seed:       int,
    id_offset:  int = 0,
    created_at: int = 0,
) -> List[Task]:''', '''_MAX_RETRY = 50

def generate_tasks(
    count:      int,
    wh:         Warehouse,
    occupied:   List[Tuple[int, int]],
    seed:       int,
    id_offset:  int,
    created_at: int = 0,
) -> List[Task]:''')

main_py = main_py.replace('for _ in range(50):', 'for _ in range(_MAX_RETRY):')
main_py = main_py.replace('after 50 attempts', 'after {_MAX_RETRY} attempts')

# IMPROVEMENT 3
main_py = main_py.replace('5 / 5  —  Running', '5 / 6  —  Running')
main_py = main_py.replace('Visualisation"', '6 / 6  —  Visualisation"')

# IMPROVEMENT 5
main_py = main_py.replace('''    for i in range(count):
        arrival = first_step + i * interval
        task    = generate_tasks(
            1, wh, list(used), seed + i * 13,
            id_offset=id_offset + i,
            created_at=arrival,
        )[0]''', '''    for i in range(count):
        arrival = first_step + i * interval
        saved_rng = wh._rng
        wh._rng = random.Random(seed + i * 13)
        try:
            task    = generate_tasks(
                1, wh, list(used), seed + i * 13,
                id_offset=id_offset + i,
                created_at=arrival,
            )[0]
        finally:
            wh._rng = saved_rng''')

write_file('cd_version/main.py', main_py)
print("main.py fixed")


# --- simulation.py ---
sim_py = read_file('cd_version/simulation.py')

# Add _PRUNE_INTERVAL
sim_py = sim_py.replace('''class Simulation:''', '''_PRUNE_INTERVAL: int = 25

class Simulation:''')

# BUG 3
sim_py = sim_py.replace('''    def _clear_reservations(self, robot_id: int) -> None:''', '''    def _prune_old_reservations(self) -> None:
        """
        Remove reservation entries whose timestep has already passed.
        Keeps the dict bounded to at most ~(replan_horizon * n_robots) entries.
        Called every _PRUNE_INTERVAL steps from run().
        """
        stale = [k for k in self.reservation if k[2] < self.step]
        for k in stale:
            del self.reservation[k]

    def _clear_reservations(self, robot_id: int) -> None:''')

sim_py = sim_py.replace('''            self._check_completions()
            self._save_snapshot()''', '''            self._check_completions()
            self._save_snapshot()

            if step % _PRUNE_INTERVAL == 0:
                self._prune_old_reservations()''')

# BUG 4
sim_py = sim_py.replace('''                else:
                    if self.verbose:
                        print(
                            f"  [WARNING] Robot {robot.id}: no path to dropoff "
                            f"{task.dropoff}"
                        )''', '''                else:
                    if self.verbose:
                        print(
                            f"  [WARNING] Robot {robot.id}: no path to dropoff "
                            f"{task.dropoff} — task {task.id} re-queued to PENDING."
                        )
                    task.status        = TaskStatus.PENDING
                    task.assigned_to   = None
                    robot.current_task = None
                    robot.status       = RobotStatus.IDLE
                    robot.reset_path()''')

# IMP 1: priority scheduling
sim_py = sim_py.replace('''    def _allocate_tasks(self) -> None:
        """Match idle robots to pending tasks, then plan paths."""
        if self.allocation_method == "hungarian":
            pairs = hungarian_allocate(self.robots, self.tasks, self.step)
        else:
            pairs = greedy_allocate(self.robots, self.tasks, self.step)

        for robot, task in pairs:
            self._assign(robot, task)''', '''    def _allocate_tasks(self) -> None:
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
            self._assign(robot, task)''')

# IMP 2: avoid over-writing active robot entries
sim_py = sim_py.replace('''                    existing = self.reservation.get(key)
                    if existing is None or existing == robot.id:
                        self.reservation[key] = robot.id''', '''                    existing = self.reservation.get(key)
                    if existing is not None and existing != robot.id:
                        continue
                    if existing is None or existing == robot.id:
                        self.reservation[key] = robot.id''')

# IMP 3: SimMetrics formatting
sim_py = sim_py.replace('''            f"║  Total Tasks          : {self.total_tasks:<13}║",
            f"║  Completed Tasks      : {self.completed_tasks:<13}║",
            f"║  Success Rate         : {self.completed_tasks/max(1,self.total_tasks)*100:>6.1f} %      ║",
            f"║  Total Steps          : {self.simulation_steps:<13}║",
            f"║  Total Distance       : {self.total_distance:<13}║",
            f"║  Avg Task Makespan    : {self.avg_makespan:>6.1f} steps    ║",
            f"║  Throughput           : {self.throughput:>6.2f} tasks/100s ║",''', '''            f"║  Total Tasks          : {self.total_tasks:<13}║",
            f"║  Completed Tasks      : {self.completed_tasks:<13}║",
            f"║  Success Rate         : {self.completed_tasks/max(1,self.total_tasks)*100:<13.1f}║",
            f"║  Total Steps          : {self.simulation_steps:<13}║",
            f"║  Total Distance       : {self.total_distance:<13}║",
            f"║  Avg Task Makespan    : {self.avg_makespan:<13.1f}║",
            f"║  Throughput           : {self.throughput:<13.2f}║",''')
			
# IMP 4: early exit logging
sim_py = sim_py.replace('''        if self._all_done():
                if self.verbose:
                    print(f"\\n  ✓ All tasks completed at step {step}!")
                break''', '''            if self._all_done():
                if self.verbose:
                    print(f"\\n  ✓ All tasks completed at step {step}!")
                    print(f"  Finished {self.max_steps - step} steps early")
                break''')

# IMP 5: exclude empty path robots, cap trail length
sim_py = sim_py.replace('''            "robot_paths":     {r.id: list(r.path)    for r in self.robots},
            "robot_path_idx":  {r.id: r.path_index    for r in self.robots},
            "robot_trails":    {r.id: list(r.trail)   for r in self.robots},''', '''            "robot_paths":     {r.id: list(r.path) if r.path else [] for r in self.robots},
            "robot_path_idx":  {r.id: r.path_index    for r in self.robots},
            "robot_trails":    {r.id: list(r.trail[-200:]) for r in self.robots},''')

write_file('cd_version/simulation.py', sim_py)
print("simulation.py fixed")


# --- astar.py ---
astar_py = read_file('cd_version/astar.py')

# CONSTANTS and EXPORTS
astar_py = astar_py.replace('''from typing import Dict, List, Optional, Tuple

Pos = Tuple[int, int]''', '''from typing import Dict, List, Optional, Tuple

__all__ = ["astar", "space_time_astar", "manhattan"]

Pos = Tuple[int, int]
_MAX_WAIT = 15''')

# heuristic improvements
astar_py = astar_py.replace('''def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])''', '''import math

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def octile(a: Pos, b: Pos) -> float:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (math.sqrt(2) - 1) * min(dr, dc)''')

astar_py = astar_py.replace('''def astar(
    grid:  List[List[int]],
    start: Pos,
    goal:  Pos,
) -> Optional[List[Pos]]:''', '''def astar(
    grid:  List[List[int]],
    start: Pos,
    goal:  Pos,
    heuristic: str = "manhattan"
) -> Optional[List[Pos]]:''')

astar_py = astar_py.replace('''    def h(pos: Pos) -> int:
        return manhattan(pos, goal)''', '''    def h(pos: Pos) -> float:
        if heuristic == "octile":
            return octile(pos, goal)
        return manhattan(pos, goal)''')

astar_py = astar_py.replace('''    # heap stores (f_score, counter, g_score, (r, c), path)
    heap: List[Tuple[int, int, int, Pos, List[Pos]]] = [
        (h(start), counter, 0, start, [start])
    ]
    visited: set[Pos] = {start}

    while heap:
        f, _, g, pos, path = heapq.heappop(heap)

        if pos == goal:
            return path

        r, c = pos
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 0:
                    npos = (nr, nc)
                    if npos not in visited:
                        visited.add(npos)
                        heapq.heappush(
                            heap,
                            (g + 1 + h(npos), next_counter(), g + 1, npos, path + [npos])
                        )

    return None''', '''    came_from: Dict[Pos, Optional[Pos]] = {start: None}
    heap = [(h(start), counter, 0, start)]
    visited = {start}

    while heap:
        f, _, g, pos = heapq.heappop(heap)
        
        if pos == goal:
            path_res, cur = [], pos
            while cur is not None:
                path_res.append(cur)
                cur = came_from[cur]
            return path_res[::-1]

        r, c = pos
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 0:
                    npos = (nr, nc)
                    if npos not in visited:
                        visited.add(npos)
                        came_from[npos] = pos
                        heapq.heappush(
                            heap,
                            (g + 1 + h(npos), next_counter(), g + 1, npos)
                        )

    return None''')

astar_py = astar_py.replace('''def space_time_astar(
    grid:              List[List[int]],
    start:             Pos,
    goal:              Pos,
    reservation_table: Dict[Tuple[int, int, int], int],
    robot_id:          int,
    start_time:        int,
    max_time:          int,
) -> Optional[List[Pos]]:''', '''def space_time_astar(
    grid:              List[List[int]],
    start:             Pos,
    goal:              Pos,
    reservation_table: Dict[Tuple[int, int, int], int],
    robot_id:          int,
    start_time:        int,
    max_time:          int,
    heuristic: str = "manhattan"
) -> Optional[List[Pos]]:''')

astar_py = astar_py.replace('''    def h(pos: Pos) -> int:
        return manhattan(pos, goal)''', '''    def h(pos: Pos) -> float:
        if heuristic == "octile":
            return octile(pos, goal)
        return manhattan(pos, goal)''')

astar_py = astar_py.replace('''    def h(pos: Pos) -> float:''', '''    def h(pos: Pos) -> float:''') # wait already replaced above

astar_py = astar_py.replace('''    # state: (r, c, t)
    # heap stores (f, tie, g, path_list)
    start_state = (start[0], start[1], start_time)
    heap: List[Tuple[int, int, int, List[Pos]]] = [
        (h(start), counter, 0, [start])
    ]
    visited_states: set[Tuple[int, int, int]] = {start_state}

    while heap:
        f, _, g, path = heapq.heappop(heap)
        
        pos = path[-1]
        r, c = pos
        t    = start_time + len(path) - 1
        
        if pos == goal:
            return path
            
        if t >= max_time:
            continue
            
        # Try moving (0,0 is wait-in-place)
        for dr, dc in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            
            # Grid bounds
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
                
            # Obstacles
            if grid[nr][nc] == 1:
                continue
                
            nt = t + 1
            npos = (nr, nc)
            nstate = (nr, nc, nt)
            
            if nstate in visited_states:
                continue
                
            # Vertex collision
            if reservation_table.get((nr, nc, nt), robot_id) != robot_id:
                continue
                
            # Edge/swap collision
            if dr != 0 or dc != 0:
                rob_r1 = reservation_table.get((nr, nc, t))
                rob_r2 = reservation_table.get((r, c, nt))
                if (
                    rob_r1 is not None 
                    and rob_r1 == rob_r2 
                    and rob_r1 != robot_id
                ):
                    continue
                    
            visited_states.add(nstate)
            heapq.heappush(
                heap,
                (g + 1 + h(npos), next_counter(), g + 1, path + [npos])
            )''', '''    # state: (r, c, t, wait_count)
    start_state = (start[0], start[1], start_time, 0)
    came_from: Dict[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]] = {start_state: None}
    heap = [(h(start), counter, 0, start_state)]
    visited_states = {start_state[:3]: 0} # map (r,c,t) -> wait_count

    while heap:
        f, _, g, state = heapq.heappop(heap)
        r, c, t, wait_count = state
        
        if (r, c) == goal:
            path_res = []
            cur = state
            while cur is not None:
                path_res.append((cur[0], cur[1]))
                cur = came_from[cur]
            return path_res[::-1]
            
        if t >= max_time:
            continue
            
        # Try moving (0,0 is wait-in-place)
        for dr, dc in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = r + dr, c + dc
            n_wait = wait_count + 1 if (dr == 0 and dc == 0) else 0

            # Wait limit guard
            if n_wait > _MAX_WAIT:
                continue
            
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:
                continue
                
            nt = t + 1
            npos = (nr, nc)
            nstate_key = (nr, nc, nt)
            
            if nstate_key in visited_states and visited_states[nstate_key] <= n_wait:
                continue
                
            # Vertex collision
            if reservation_table.get((nr, nc, nt), robot_id) != robot_id:
                continue
                
            # Edge/swap collision
            if dr != 0 or dc != 0:
                rob_r1 = reservation_table.get((nr, nc, t))
                rob_r2 = reservation_table.get((r, c, nt))
                if (
                    rob_r1 is not None 
                    and rob_r1 == rob_r2 
                    and rob_r1 != robot_id
                ):
                    continue
                    
            visited_states[nstate_key] = n_wait
            nstate = (nr, nc, nt, n_wait)
            came_from[nstate] = state
            heapq.heappush(
                heap,
                (g + 1 + h(npos), next_counter(), g + 1, nstate)
            )''')

write_file('cd_version/astar.py', astar_py)
print("astar.py fixed")


# --- robot.py ---
robot_py = read_file('cd_version/robot.py')

robot_py = robot_py.replace('''    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"''', '''    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    WAITING     = "waiting" # For future use''')

robot_py = robot_py.replace('''    TO_PICKUP  = "moving_to_pickup"
    TO_DROPOFF = "moving_to_dropoff"''', '''    TO_PICKUP  = "moving_to_pickup"
    TO_DROPOFF = "moving_to_dropoff"
    WAITING    = "waiting"''')

robot_py = robot_py.replace('''    completed_time: Optional[int] = None''', '''    completed_time: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id}, {self.pickup}→{self.dropoff}, "
            f"status={self.status.value})"
        )

    @property
    def makespan(self) -> Optional[int]:
        if self.completed_time is not None and self.created_time is not None:
            return max(0, self.completed_time - self.created_time)
        return None''')

robot_py = robot_py.replace('''    @property
    def makespan(self) -> Optional[int]:
        if self.completed_time is not None and self.created_time is not None:
            return self.completed_time - self.created_time
        return None''', '') # remove it from wherever it was to avoid duplicates and just let the previously inserted one serve


robot_py = robot_py.replace('''    tasks_completed: int = 0''', '''    tasks_completed: int = 0
    total_waits:     int = 0

    def __repr__(self) -> str:
        return (
            f"Robot(id={self.id}, pos={self.pos}, "
            f"status={self.status.value}, task={self.current_task.id if self.current_task else None})"
        )''')

robot_py = robot_py.replace('''        if new_pos != old_pos:
            self.total_distance += 1
            self.trail.append(new_pos)''', '''        if new_pos != old_pos:
            self.total_distance += 1
            self.trail.append(new_pos)
        else:
            self.total_waits += 1''')

write_file('cd_version/robot.py', robot_py)
print("robot.py fixed")


# --- warehouse.py ---
wh_py = read_file('cd_version/warehouse.py')

wh_py = wh_py.replace('''from typing import List, Tuple, Set''', '''from typing import Collection, List, Optional, Set, Tuple''')

wh_py = wh_py.replace('''    def random_free_cell(
        self,
        exclude: List[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """Return a random free cell not in `exclude`."""
        blocked = set(exclude or [])''', '''    def random_free_cell(
        self,
        exclude: Optional[Collection[Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        """Return a random free cell not in `exclude`."""
        blocked = set(exclude or [])''')

wh_py = wh_py.replace('''        self.free_cells: List[Tuple[int, int]] = self._collect_free_cells()''', '''        self.free_cells: List[Tuple[int, int]] = self._collect_free_cells()
        self.free_cells_set: frozenset = frozenset(self.free_cells)''')

wh_py = wh_py.replace('''    def is_free(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] == 0
        return False''', '''    def is_free(self, pos: Tuple[int, int]) -> bool:
        return pos in self.free_cells_set''')

wh_py = wh_py.replace('''    def render_ascii(self) -> str:
        chars = {0: "·", 1: "█"}
        rows = ["".join(chars[cell] for cell in row) for row in self.grid]
        return "\\n".join(rows)''', '''    def render_ascii(self) -> str:
        header = "    " + "".join(
            str(c % 10) if c % 5 != 0 else str(c // 10 % 10)
            for c in range(self.cols)
        )
        chars = {0: "·", 1: "█"}
        rows = [
            f"{r:3d} " + "".join(chars[cell] for cell in row)
            for r, row in enumerate(self.grid)
        ]
        return header + "\\n" + "\\n".join(rows)''')


wh_py = wh_py.replace('''    def summary(self) -> str:
        return f"Warehouse {self.rows}×{self.cols} ({len(self.free_cells)} free cells)"''', '''    def summary(self) -> str:
        total   = self.rows * self.cols
        free    = len(self.free_cells)
        shelves = total - free
        return (
            f"Warehouse {self.rows}×{self.cols} | "
            f"Free: {free} ({free/total*100:.1f}%) | "
            f"Shelves: {shelves} ({shelves/total*100:.1f}%)"
        )''')

write_file('cd_version/warehouse.py', wh_py)
print("warehouse.py fixed")


# --- task_allocator.py ---
ta_py = read_file('cd_version/task_allocator.py')

ta_py = ta_py.replace('''from robot import Robot, Task, TaskStatus''', '''from robot import Robot, Task, TaskStatus

__all__ = ["greedy_allocate", "hungarian_allocate"]''')

ta_py = ta_py.replace('''    # FIFO priority
    pending = sorted(
        [t for t in tasks if t.status == TaskStatus.PENDING],
        key=lambda t: t.created_time,
    )''', '''    # FIFO priority
    pending = sorted(
        [t for t in tasks if t.status == TaskStatus.PENDING],
        key=lambda t: (t.created_time, t.id),
    )''')

ta_py = ta_py.replace('''            if d < best_dist:
                best_dist  = d
                best_robot = robot''', '''            if d < best_dist or (d == best_dist and robot.id < best_robot.id):
                best_dist  = d
                best_robot = robot''')

ta_py = ta_py.replace('''except ImportError:
    HAS_SCIPY = False''', '''except ImportError:
    import sys
    print(
        f"[task_allocator] scipy not available (Python {sys.version.split()[0]}) "
        f"— falling back to greedy allocation."
    )
    HAS_SCIPY = False''')

ta_py = ta_py.replace('''        print("scipy not installed; falling back to greedy_allocate.")''', '') 

write_file('cd_version/task_allocator.py', ta_py)
print("task_allocator.py fixed")


# --- visualizer.py ---
viz_py = read_file('cd_version/visualizer.py')

viz_py = viz_py.replace('''    def _build_bg(self) -> np.ndarray:
        img = np.ones((self.wh.rows, self.wh.cols, 4))
        for r in range(self.wh.rows):
            for c in range(self.wh.cols):
                if self.wh.grid[r][c] == 1:
                    img[r, c] = _SHELF_RGBA
                else:
                    img[r, c] = _FLOOR_RGBA
        return img''', '''    def _build_bg(self) -> np.ndarray:
        grid_np = np.array(self.wh.grid, dtype=bool)
        img     = np.ones((self.wh.rows, self.wh.cols, 4))
        img[grid_np]  = _SHELF_RGBA
        img[~grid_np] = _FLOOR_RGBA
        return img''')

viz_py = viz_py.replace('''        for task in self.tasks:
            pr, pc = task.pickup
            dr, dc = task.dropoff
            pickup_pts[task.id].set_data([pc], [pr])
            dropoff_pts[task.id].set_data([dc], [dr])''', '''        for task in self.tasks:
            pr, pc = task.pickup
            dr, dc = task.dropoff
            pickup_pts[task.id].set_data([pc], [pr])
            dropoff_pts[task.id].set_data([dc], [dr])
            alpha = 0.25 if task.status == TaskStatus.COMPLETED else 0.9
            pickup_pts[task.id].set_alpha(alpha)
            dropoff_pts[task.id].set_alpha(alpha)''')

viz_py = viz_py.replace('''        for b, h in zip(bars_d, dists):
            b.set_height(h)
        for b, h in zip(bars_t, comps):
            b.set_height(h)''', '''        for b, h in zip(bars_d, dists):
            b.set_height(h)
        for b, h in zip(bars_t, comps):
            b.set_height(h)
        ax_dist.relim()
        ax_dist.autoscale_view()
        ax_tasks.relim()
        ax_tasks.autoscale_view()''')

viz_py = viz_py.replace('''def plot_metrics(self) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ids   = [r.id for r in self.robots]
        dists = [r.total_distance for r in self.robots]
        comps = [r.tasks_completed for r in self.robots]

        ax1.bar(ids, dists, color=[self.robots[i].color for i in ids])
        ax1.set_title("Distance Driven")
        ax1.set_xlabel("Robot ID")
        ax1.set_ylabel("Cells")

        ax2.bar(ids, comps, color=[self.robots[i].color for i in ids])
        ax2.set_title("Tasks Completed")
        ax2.set_xlabel("Robot ID")
        ax2.set_ylabel("Count")

        plt.tight_layout()
        plt.show()''', '''    def plot_metrics(self) -> None:
        has_waits = hasattr(self.robots[0], 'total_waits')
        n_plots = 3 if has_waits else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        
        if not has_waits:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes

        ids   = [r.id for r in self.robots]
        dists = [r.total_distance for r in self.robots]
        comps = [r.tasks_completed for r in self.robots]

        ax1.bar(ids, dists, color=[self.robots[i].color for i in ids])
        ax1.set_title("Distance Driven")
        ax1.set_xlabel("Robot ID")
        ax1.set_ylabel("Cells")

        ax2.bar(ids, comps, color=[self.robots[i].color for i in ids])
        ax2.set_title("Tasks Completed")
        ax2.set_xlabel("Robot ID")
        ax2.set_ylabel("Count")
        
        if has_waits:
            waits = [r.total_waits for r in self.robots]
            ax3.bar(ids, waits, color=[self.robots[i].color for i in ids])
            ax3.set_title("Wait Steps")
            ax3.set_xlabel("Robot ID")
            ax3.set_ylabel("Steps")

        plt.tight_layout()
        plt.show()''')

viz_py = viz_py.replace('''        # Add start/end labels to lines
        for task in self.tasks:
            ax.plot(task.pickup[1], task.pickup[0], "wo", markersize=3)
            ax.plot(task.dropoff[1], task.dropoff[0], "w*", markersize=4)''', '''        # Add start/end labels to lines
        for task in self.tasks:
            ax.plot(task.pickup[1], task.pickup[0], "wo", markersize=3)
            ax.plot(task.dropoff[1], task.dropoff[0], "w*", markersize=4)
            if getattr(task, "status", None) == "completed" or (hasattr(TaskStatus, "COMPLETED") and task.status == TaskStatus.COMPLETED):
                if hasattr(task, "makespan") and task.makespan is not None:
                    ax.text(
                        task.dropoff[1], task.dropoff[0],
                        str(task.makespan), color="white",
                        fontsize=6, ha="center", va="center", zorder=9,
                    )''')

write_file('cd_version/visualizer.py', viz_py)
print("visualizer.py fixed")
