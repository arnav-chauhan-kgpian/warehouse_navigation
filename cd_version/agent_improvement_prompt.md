# AI Agent Prompt — Multi-Robot Warehouse Simulation: Full Codebase Improvement

---

## YOUR ROLE

You are a senior Python engineer tasked with improving, bug-fixing, and
reorganising a complete multi-robot warehouse simulation codebase. You will
receive 9 source files. You must apply every fix and improvement listed below,
then output all 9 corrected files in full — no diffs, no partial edits, no
placeholders. Every file must be immediately runnable with `python main.py`.

---

## PROJECT OVERVIEW

A discrete-time simulation of autonomous warehouse robots that:
- Navigate a 2-D grid warehouse (shelves + aisles) using Space-Time A*
- Avoid collisions via a shared reservation table (prioritised planning)
- Accept task assignments via Greedy or Hungarian allocation
- Visualise results as a live Matplotlib animation + metric charts

### File Inventory

| Filename            | Role                                                   |
|---------------------|--------------------------------------------------------|
| `main.py`           | Entry point; CONFIG dict; factory functions            |
| `simulation.py`     | Discrete-time step loop; reservation table management  |
| `robot.py`          | `Robot`, `Task`, `TaskStatus`, `RobotStatus` dataclasses |
| `warehouse.py`      | 2-D grid generator with shelf/aisle layout             |
| `astar.py`          | Standard A* and Space-Time A* path planners            |
| `task_allocator.py` | Greedy and Hungarian task allocation strategies        |
| `visualizer.py`     | Matplotlib animation, metric charts, final heatmap     |
| `requirements.txt`  | `matplotlib>=3.5`, `numpy>=1.21`, `scipy>=1.7`         |
| `README.md`         | Project documentation                                  |

### Rename Rule
The attached files have version suffixes (e.g. `main-7.py`, `astar-2.py`).
Strip all suffixes. Output clean filenames: `main.py`, `astar.py`, etc.

---

## SECTION 1 — CONFIRMED BUGS (MUST ALL BE FIXED)

These are verified logic errors. Apply every fix exactly as specified.

---

### BUG 1 — `main.py` → `generate_tasks()`: Retry loop is dead code

**Location:** `generate_tasks()`, the inner `for _ in range(50):` loop.

**Problem:**
```python
for _ in range(50):
    pu = wh.random_free_cell(exclude=list(used))
    do = wh.random_free_cell(exclude=list(used) + [pu])
    break   # ← unconditional break — loop never retries
```
The `break` fires on every iteration. This is identical to no loop at all.
If `wh.random_free_cell` raises `ValueError` (no eligible cells left), the
exception propagates uncaught and crashes task generation silently.

**Required fix:**
```python
for _ in range(50):
    try:
        pu = wh.random_free_cell(exclude=list(used))
        do = wh.random_free_cell(exclude=list(used) + [pu])
        break                          # success → exit retry loop
    except ValueError:
        continue                       # pool temporarily exhausted → retry
else:
    raise RuntimeError(
        f"Could not place task {id_offset + i} after 50 attempts "
        f"— too many cells already occupied. Reduce num_robots or num_tasks."
    )
```
The `for/else` clause fires only when all 50 iterations exhaust without a
`break`, giving a clear, actionable error instead of a silent crash.

---

### BUG 2 — `main.py` → `generate_tasks()`: Dead `rng` variable

**Location:** `generate_tasks()`, top of function body.

**Problem:**
```python
rng = random.Random(seed + 200)   # created but never used
...
pu = wh.random_free_cell(...)     # uses wh._rng internally, not rng
```
The intent was to isolate task-generation randomness on a separate `seed+200`
stream. In practice, all sampling goes through `wh._rng` (seeded at warehouse
construction with the base seed), so task positions are not reproducibly
isolated from warehouse layout generation. The local `rng` is pure dead code.

**Required fix — save/restore pattern:**
```python
# Isolate task-placement RNG on seed+200 stream;
# restore warehouse RNG afterwards so other callers are unaffected.
saved_rng = wh._rng
wh._rng   = random.Random(seed + 200)
try:
    ...  # all wh.random_free_cell() calls now use the isolated stream
finally:
    wh._rng = saved_rng   # always restore, even on exception
```
Remove the now-unused `rng = random.Random(seed + 200)` line entirely.

---

### BUG 3 — `simulation.py`: Reservation table grows unbounded

**Location:** `Simulation` class; `_clear_reservations()` method and `run()`.

**Problem:**
`_clear_reservations(robot_id)` removes entries by robot ID, but never removes
entries whose timestep `t < self.step`. On a 700-step simulation with frequent
replanning, the dict accumulates tens of thousands of stale
`(row, col, t_past)` entries. Every call to `_clear_reservations` iterates the
entire dict — O(N) per call — causing quadratic slowdown by step 300+.

**Required fix — add a pruning method:**
```python
# Class constant: how often (in steps) to prune stale entries
_PRUNE_INTERVAL: int = 25

def _prune_old_reservations(self) -> None:
    """
    Remove reservation entries whose timestep has already passed.
    Keeps the dict bounded to at most ~(replan_horizon * n_robots) entries.
    Called every _PRUNE_INTERVAL steps from run().
    """
    stale = [k for k in self.reservation if k[2] < self.step]
    for k in stale:
        del self.reservation[k]
```

**Also required — call it in `run()`:**
```python
# Inside the main for-loop in run(), after _save_snapshot():
if step % self._PRUNE_INTERVAL == 0:
    self._prune_old_reservations()
```

---

### BUG 4 — `simulation.py`: No recovery when dropoff path planning fails

**Location:** `_check_completions()`, the "Arrived at pickup" branch.

**Problem:**
```python
path = self._plan(robot, task.dropoff)
if path:
    robot.assign_path(path)
    self._reserve(robot.id, path)
else:
    # Prints a warning — robot left in TO_DROPOFF with empty path
    print(f"[WARNING] Robot {robot.id}: no path to dropoff {task.dropoff}")
```
When `_plan()` returns `None`, the robot is left with:
- `status = RobotStatus.TO_DROPOFF`
- `at_path_end = True`
- `path = []` (empty)

On the **very next step**, `_check_completions` fires again (because
`robot.pos == task.pickup` and `at_path_end` is still `True`), tries the same
failing planner, prints another warning, and loops forever. This is an infinite
tight loop that stalls all task progress silently.

**Required fix — re-queue the task to PENDING:**
```python
path = self._plan(robot, task.dropoff)
if path:
    robot.assign_path(path)
    self._reserve(robot.id, path)
    if self.verbose:
        print(
            f"  [Step {self.step:4d}] 📦 Robot {robot.id} "
            f"picked up Task {task.id} → heading to {task.dropoff}"
        )
else:
    # Re-queue to PENDING — mirrors the rollback in _assign().
    # Prevents infinite retry loop. Allocator will retry on a future step.
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
```

---

## SECTION 2 — PER-FILE IMPROVEMENTS

Apply all of the following improvements alongside the bug fixes above.

---

### `main.py` — Improvements

1. **CONFIG validation function** — add `_validate_config(cfg: dict) -> None`
   that checks:
   - `num_robots >= 1`
   - `num_static_tasks >= 0`, `num_dynamic_tasks >= 0`
   - `max_steps >= 10`
   - `replan_horizon <= max_steps`
   - `allocation_method in {"greedy", "hungarian"}`
   - `seed` is an integer
   Raise `ValueError` with a descriptive message on any violation.
   Call it as the first line of `main()`.

2. **`generate_tasks` — remove `id_offset` default ambiguity** — the default
   `id_offset=0` causes duplicate task IDs when called multiple times. Change
   the signature so `id_offset` is a required positional argument (no default).
   Update all call sites accordingly.

3. **Progress header clarity** — the section `"5 / 5"` runs the simulation but
   visualisation is a separate phase. Rename to `"5 / 6"` and add a
   `"6 / 6 — Visualisation"` banner for the viz block.

4. **Type hints** — add return type `-> None` to `main()`, `_banner()`, and
   `_section()`. Add `-> List[Tuple[int, Task]]` to `schedule_dynamic_tasks`.

5. **`schedule_dynamic_tasks` — thread-safe seed isolation** — wrap the inner
   `generate_tasks` call in the same save/restore RNG pattern from Bug 2 so
   each dynamic task batch uses a fully isolated `seed + i * 13` stream.

---

### `simulation.py` — Improvements

1. **`_assign` — priority scheduling** — currently robots are planned in the
   order they appear in the `pairs` list from the allocator. Add a priority
   key: sort `pairs` by `manhattan(robot.pos, task.pickup)` ascending before
   planning, so robots closer to their target get reservations first (fewer
   waiting conflicts).

2. **`_refresh_idle_reservations` — avoid over-writing active robot entries** —
   the current loop checks `existing is None or existing == robot.id`, but
   should additionally skip cells already reserved by *other* robots:
   ```python
   if existing is not None and existing != robot.id:
       continue   # another robot already reserved this cell; don't overwrite
   ```

3. **`SimMetrics.__str__` — fix alignment** — the `║` box characters are
   misaligned for right-padded vs left-padded fields. Standardise all numeric
   fields to `{value:<13}` (left-aligned, width 13) and ensure every row is
   exactly 40 chars wide so the box closes flush.

4. **`run()` — early exit logging** — when `_all_done()` triggers early exit,
   log the number of steps saved: `f"  Finished {self.max_steps - step} steps early"`.

5. **`_save_snapshot` — exclude empty path robots** — in `robot_paths`, replace
   `list(r.path)` with `list(r.path) if r.path else []` (already fine), but
   also cap trail length at `trail_max=200` to keep snapshot memory bounded:
   ```python
   "robot_trails": {r.id: list(r.trail[-200:]) for r in self.robots},
   ```

---

### `astar.py` — Improvements

1. **`astar` — path reconstruction memory** — the current implementation stores
   the full `path` list inside each heap entry: `(f, tie, g, pos, path)`. This
   causes O(N²) memory on long paths as every node duplicates the prefix. Fix
   by storing a `came_from` dict instead and reconstructing the path at goal:
   ```python
   # Replace path-in-heap with came_from reconstruction
   came_from: Dict[Pos, Optional[Pos]] = {start: None}
   heap = [(h, counter, 0, start)]

   while heap:
       f, _, g, pos = heapq.heappop(heap)
       if pos == goal:
           # Reconstruct
           path, cur = [], pos
           while cur is not None:
               path.append(cur)
               cur = came_from[cur]
           return path[::-1]
       ...
       came_from[npos] = pos
   ```

2. **`space_time_astar` — same came_from optimisation** — apply the same
   came_from approach. State is `(r, c, t)` → previous `(r, c, t)`.
   Reconstruction at goal: walk `came_from` back to start, extract `(r, c)`,
   reverse.

3. **`space_time_astar` — wait explosion guard** — a robot can generate
   arbitrarily many wait-in-place states at the same cell across timesteps.
   Add a guard: if a robot has been waiting at the same `(r, c)` for more than
   `max_wait = 15` consecutive timesteps without moving, skip further waits
   from that state. This prevents the planner from exploring thousands of
   useless wait states in heavily congested grids.
   ```python
   # Track consecutive waits: store in state as (r, c, t, wait_count)
   # If wait_count >= 15: skip (0,0) direction for this state
   ```

4. **Add `euclidean_heuristic` option** — add an optional `heuristic` parameter
   (default `"manhattan"`) to both `astar` and `space_time_astar`. When
   `heuristic="octile"`, use the octile distance formula:
   `max(dr, dc) + (sqrt(2) - 1) * min(dr, dc)` (still admissible on 4-connected
   grids, tighter bound than Manhattan on diagonal-heavy layouts).

5. **Export `__all__`** — add at module top:
   ```python
   __all__ = ["astar", "space_time_astar", "manhattan"]
   ```

---

### `robot.py` — Improvements

1. **`Robot.move_one_step` — wait detection bug** — a wait step
   (`path[i] == path[i-1]`) does NOT increment `total_distance` — correct.
   But it also does NOT append to `trail` — also correct. However, there is no
   wait counter, so the `SimMetrics` cannot report total waits. Add:
   ```python
   total_waits: int = 0
   ```
   Increment it in `move_one_step` when `new_pos == old_pos`.

2. **`Task.makespan` — guard against negative values** — if a task is
   re-queued (Bug 4 fix), `assigned_time` may be set but `completed_time` may
   come later with a lower `created_time` relative to a re-queue cycle. Add:
   ```python
   @property
   def makespan(self) -> Optional[int]:
       if self.completed_time is not None and self.created_time is not None:
           return max(0, self.completed_time - self.created_time)
       return None
   ```

3. **`RobotStatus` — add `WAITING` state** — for future use and better
   diagnostics. Add `WAITING = "waiting"` to the enum. The simulation does not
   need to use it immediately, but the visualiser can check for it.

4. **`Robot` — add `__repr__`:**
   ```python
   def __repr__(self) -> str:
       return (
           f"Robot(id={self.id}, pos={self.pos}, "
           f"status={self.status.value}, task={self.current_task.id if self.current_task else None})"
       )
   ```

5. **`Task` — add `__repr__`:**
   ```python
   def __repr__(self) -> str:
       return (
           f"Task(id={self.id}, {self.pickup}→{self.dropoff}, "
           f"status={self.status.value})"
       )
   ```

---

### `warehouse.py` — Improvements

1. **`random_free_cell` — build exclusion set once** — currently
   `blocked = set(exclude or [])` re-builds the exclusion set from a list on
   every call. The caller in `generate_tasks` passes `list(used)` which is
   already a list. This is fine for small grids, but add a docstring note
   clarifying callers should pass a pre-built `set` for performance:
   ```python
   def random_free_cell(
       self,
       exclude: Optional[Collection[Tuple[int, int]]] = None,
   ) -> Tuple[int, int]:
   ```
   Change the type hint from `List` to `Collection` (import from `typing`).

2. **`_collect_free_cells` — cache as frozenset too** — add:
   ```python
   self.free_cells_set: frozenset = frozenset(self.free_cells)
   ```
   This lets `is_free` use O(1) lookup instead of the current O(rows*cols)
   attribute access pattern when called in tight loops.
   Update `is_free` to:
   ```python
   def is_free(self, pos: Tuple[int, int]) -> bool:
       return pos in self.free_cells_set
   ```

3. **`render_ascii` — add row/col indices** — prepend column index numbers
   every 5 columns and left-pad row numbers so the grid is easier to inspect:
   ```python
   def render_ascii(self) -> str:
       header = "    " + "".join(
           str(c % 10) if c % 5 != 0 else str(c // 10 % 10)
           for c in range(self.cols)
       )
       chars = {0: "·", 1: "█"}
       rows = [
           f"{r:3d} " + "".join(chars[cell] for cell in row)
           for r, row in enumerate(self.grid)
       ]
       return header + "\n" + "\n".join(rows)
   ```

4. **`summary` — add free-cell count and shelf count separately:**
   ```python
   def summary(self) -> str:
       total   = self.rows * self.cols
       free    = len(self.free_cells)
       shelves = total - free
       return (
           f"Warehouse {self.rows}×{self.cols} | "
           f"Free: {free} ({free/total*100:.1f}%) | "
           f"Shelves: {shelves} ({shelves/total*100:.1f}%)"
       )
   ```

---

### `task_allocator.py` — Improvements

1. **`greedy_allocate` — tie-breaking** — when two robots are equidistant from
   a task's pickup, the allocator picks whichever appears first in the `idle`
   list. This is non-deterministic across runs if robot order changes. Add
   tie-breaking by `robot.id`:
   ```python
   if d < best_dist or (d == best_dist and robot.id < best_robot.id):
       best_dist  = d
       best_robot = robot
   ```

2. **`hungarian_allocate` — add scipy import error logging** — currently just
   prints a bare string. Use the standard `[task_allocator]` prefix and add
   the Python version info for easier debugging:
   ```python
   import sys
   print(
       f"[task_allocator] scipy not available (Python {sys.version.split()[0]}) "
       f"— falling back to greedy allocation."
   )
   ```

3. **`_idle_and_pending` — expose sort key for FIFO** — rename the sort key
   comment to a named lambda for clarity:
   ```python
   pending = sorted(
       [t for t in tasks if t.status == TaskStatus.PENDING],
       key=lambda t: (t.created_time, t.id),   # FIFO; tie-break by id
   )
   ```
   Adding `t.id` as a secondary sort key makes ordering fully deterministic.

4. **Add `__all__`:**
   ```python
   __all__ = ["greedy_allocate", "hungarian_allocate"]
   ```

---

### `visualizer.py` — Improvements

1. **`animate` — task marker fade on completion** — when a task reaches
   `TaskStatus.COMPLETED`, set its pickup and dropoff markers to `alpha=0.25`
   so completed tasks visually recede without disappearing. Update markers
   inside the `update(frame)` function:
   ```python
   for task in self.tasks:
       alpha = 0.25 if task.status == TaskStatus.COMPLETED else 0.9
       pickup_pts[task.id].set_alpha(alpha)
       dropoff_pts[task.id].set_alpha(alpha)
   ```

2. **`animate` — y-axis auto-scaling for metric bars** — currently the bar
   charts' y-axes never rescale, so early frames show tiny bars. Fix by calling
   `ax_dist.relim(); ax_dist.autoscale_view()` and same for `ax_tasks` inside
   the `update` function after updating bar heights.

3. **`plot_metrics` — add wait-step breakdown** — if `Robot.total_waits` exists
   (from robot.py improvement #1), add a third subplot showing wait steps per
   robot alongside distance and tasks completed.

4. **`plot_final_map` — add task completion annotations** — annotate each
   completed task's dropoff cell with its makespan value as a small text label:
   ```python
   for task in self.tasks:
       if task.status == TaskStatus.COMPLETED and task.makespan is not None:
           ax.text(
               task.dropoff[1], task.dropoff[0],
               str(task.makespan), color="white",
               fontsize=6, ha="center", va="center", zorder=9,
           )
   ```

5. **`_build_bg` — vectorise with numpy** — the current double `for` loop is
   O(rows*cols) Python. Replace with numpy vectorisation:
   ```python
   def _build_bg(self) -> np.ndarray:
       grid_np = np.array(self.wh.grid, dtype=bool)
       img     = np.ones((self.wh.rows, self.wh.cols, 4))
       img[grid_np]  = _SHELF_RGBA
       img[~grid_np] = _FLOOR_RGBA
       return img
   ```

---

### `requirements.txt` — Improvements

Replace with a pinned, complete requirements file:
```
matplotlib>=3.7,<4.0
numpy>=1.24,<2.0
scipy>=1.10          # optional — required only for Hungarian allocation
```
Add a comment explaining the scipy optional status.

---

### `README.md` — Improvements

1. **Add a Bugs Fixed section** documenting all 4 bug fixes with a one-sentence
   description each (do not include full code — just the description and
   what broke before the fix).

2. **Update Project Structure** — add the clean filenames (no version suffixes).

3. **Add a Performance Notes section** explaining:
   - The reservation table is pruned every 25 steps (O(N) dict scan at bounded
     cost instead of unbounded growth).
   - The Space-Time A* uses came_from reconstruction (O(path length) memory
     instead of O(path length²)).
   - Hungarian allocation is O(n³) and recommended only for fleets > 6 robots.

4. **Add a Known Limitations section:**
   - Prioritised planning is incomplete (not guaranteed collision-free in all
     cases — CBS would be needed for completeness).
   - Standard A* fallback may produce conflicting paths in dense grids.
   - No real-time re-planning triggered by dynamic obstacles.

---

## SECTION 3 — CODE QUALITY STANDARDS

Apply these standards to every file you output.

### Type Hints
- Every function must have a complete signature: parameter types + return type.
- Use `from __future__ import annotations` at the top of every file.
- Use `Optional[X]` for nullable values; avoid bare `None` return types.

### Docstrings
- Every public class and function must have a docstring.
- Module-level docstring must describe the file's role in 2–4 sentences.
- Private methods (`_name`) need a one-line docstring minimum.

### Constants
- Move all magic numbers to named module-level constants:
  - `_PRUNE_INTERVAL = 25` (simulation.py)
  - `_MAX_RETRY = 50` (main.py generate_tasks)
  - `_MAX_WAIT = 15` (astar.py wait guard)
  - `BIG = 10_000.0` (task_allocator.py — already named, keep it)

### Imports
- Group imports: stdlib → third-party → local project, separated by blank lines.
- Remove any unused imports.
- `from __future__ import annotations` must be the very first non-comment line.

### Error Handling
- Any `ValueError` that can crash task generation must be caught or raised with
  a descriptive message (see Bug 1 fix).
- Never silently swallow exceptions. At minimum, re-raise or print + re-raise.

---

## SECTION 4 — OUTPUT REQUIREMENTS

1. Output all 9 files in full — no truncation, no `# ... rest unchanged ...`.
2. Use clean filenames: `main.py`, `simulation.py`, `robot.py`, `warehouse.py`,
   `astar.py`, `task_allocator.py`, `visualizer.py`, `requirements.txt`, `README.md`.
3. Preserve all existing functionality that is not explicitly changed above.
4. Do not add external dependencies beyond `matplotlib`, `numpy`, `scipy`.
5. Do not use `asyncio`, `threading`, or multiprocessing — keep single-threaded.
6. The simulation must produce identical output for `seed=42` before and after
   the improvements (except for the bug-fix changes which alter correctness).

---

## SECTION 5 — VERIFICATION CHECKLIST

After generating all files, mentally verify:

- [ ] Bug 1: `generate_tasks` retry loop uses `try/except` with `for/else`
- [ ] Bug 2: `wh._rng` is saved, replaced with `random.Random(seed+200)`, and
             restored in `finally`; old dead `rng = random.Random(...)` is removed
- [ ] Bug 3: `_prune_old_reservations()` method exists; called every 25 steps in `run()`
- [ ] Bug 4: Dropoff planning failure re-queues task to `PENDING` and resets robot to `IDLE`
- [ ] `astar.py`: both planners use `came_from` dict reconstruction (no path-in-heap)
- [ ] `robot.py`: `total_waits` counter exists and is incremented on wait steps
- [ ] `warehouse.py`: `free_cells_set` frozenset exists; `is_free` uses it
- [ ] `task_allocator.py`: greedy tie-breaks by `robot.id`; FIFO sort uses `(created_time, id)`
- [ ] `visualizer.py`: `_build_bg` uses numpy vectorisation; completed task markers fade
- [ ] All files: `from __future__ import annotations` present
- [ ] All files: no version-suffix filenames (`main-7.py` → `main.py`, etc.)
- [ ] `README.md`: Bugs Fixed section present; Performance Notes present

---

## SECTION 6 — QUICK EXECUTION TEST

After you output all files, confirm that the following commands would succeed
without errors (you do not need to run them — just verify the code is correct):

```bash
pip install matplotlib numpy scipy

python -c "
from warehouse import Warehouse
from robot import Robot, Task, RobotStatus, TaskStatus
from astar import astar, space_time_astar
from task_allocator import greedy_allocate, hungarian_allocate
from simulation import Simulation, SimMetrics
from visualizer import WarehouseVisualizer
print('All imports OK')
"

python main.py   # must complete without exception for seed=42
```

The `main.py` run must print:
- The 5-section banner
- Robot spawn positions
- Task assignments
- Simulation step progress
- Final `SimMetrics` box
- No `[WARNING]` lines for the default `seed=42` config (the default warehouse
  is not congested enough to trigger planning failures)

---

*End of prompt. Begin implementation now.*
