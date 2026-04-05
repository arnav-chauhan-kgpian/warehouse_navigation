"""
simulation.py — Premium Warehouse Simulation Visualization
=============================================================
Auto-scales to fit any screen resolution. Responsive layout with:
  • Large colour-coded robots with glow halos
  • Bright path trails & pulsating markers
  • Bottom HUD with stat cards & robot state indicators
  • Right legend panel
  • Interactive controls: Space, +/-, D, M, Esc
"""

import pygame
import sys
import os
import random
import math

from environment import (
    Warehouse, calculate_cell_size, draw_grid,
    SHELF, EMPTY, PICKUP, DROPOFF, CHARGING,
)
from robot_task_alocat import RobotTaskAllocator
from state_diagram import (
    RobotState, STATE_COLORS, print_state_diagram,
    generate_mermaid_fsm, generate_snapshot_diagram,
)

# ── Layout proportions (computed at runtime from screen size) ────────────
HUD_HEIGHT    = 160
LEGEND_WIDTH  = 200
FPS_DEFAULT   = 5

# ── Premium dark-theme colour tokens ─────────────────────────────────────
BG_DARK        = (15, 23, 42)
PANEL_BG       = (22, 33, 52)
PANEL_BORDER   = (51, 65, 85)
CARD_BG        = (30, 41, 59)
TEXT_WHITE      = (248, 250, 252)
TEXT_BRIGHT     = (226, 232, 240)
TEXT_DIM        = (148, 163, 184)
TEXT_ACCENT     = (129, 140, 248)
AMBER           = (251, 191, 36)
EMERALD         = (52, 211, 153)
RED_BRIGHT      = (248, 113, 113)
PURPLE          = (192, 132, 252)
BLUE_BRIGHT     = (96, 165, 250)
WHITE           = (255, 255, 255)
GREEN_BRIGHT    = (74, 222, 128)

# Robot colours per state
ROBOT_COLORS = {
    RobotState.IDLE:              (96, 165, 250),
    RobotState.MOVING_TO_PICKUP:  (74, 222, 128),
    RobotState.PICKING_UP:        (251, 191, 36),
    RobotState.MOVING_TO_DROPOFF: (192, 132, 252),
    RobotState.DROPPING_OFF:      (251, 113, 133),
}


# ═════════════════════════════════════════════════════════════════════════
#  DRAWING PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════

def draw_rounded_panel(surface, color, rect, radius=8, border_width=1,
                       border_color=None):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_width and border_color:
        pygame.draw.rect(surface, border_color, rect,
                         width=border_width, border_radius=radius)


def draw_glow_circle(surface, center, radius, color, intensity=3):
    for i in range(intensity, 0, -1):
        alpha = max(15, 50 // i)
        r = radius + i * 3
        glow = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*color[:3], alpha), (r + 2, r + 2), r)
        surface.blit(glow, (center[0] - r - 2, center[1] - r - 2))


def draw_marker_glow(surface, pos, cell_size, color, ox, oy, pulse_t=0):
    r, c = pos
    cx = ox + c * cell_size + cell_size // 2
    cy = oy + r * cell_size + cell_size // 2
    base_r = max(3, cell_size // 3)
    pulse = int(2 * math.sin(pulse_t * 0.12))
    draw_glow_circle(surface, (cx, cy), base_r + pulse, color, intensity=3)
    pygame.draw.circle(surface, color, (cx, cy), base_r)
    pygame.draw.circle(surface, WHITE, (cx, cy), base_r, width=2)


def draw_robot_premium(surface, pos, cell_size, color, rid, font,
                       ox, oy, pulse_t=0):
    r, c = pos
    cx = ox + c * cell_size + cell_size // 2
    cy = oy + r * cell_size + cell_size // 2
    radius = max(7, int(cell_size * 0.38))

    draw_glow_circle(surface, (cx, cy), radius, color, intensity=3)
    pygame.draw.circle(surface, (0, 0, 0), (cx + 2, cy + 2), radius)
    pygame.draw.circle(surface, color, (cx, cy), radius)
    pygame.draw.circle(surface, WHITE, (cx, cy), radius, width=2)

    id_text = font.render(f"R{rid}", True, (15, 23, 42))
    tr = id_text.get_rect(center=(cx, cy))
    surface.blit(id_text, tr)


def draw_path_trail(surface, path, step_idx, cell_size, color, ox, oy):
    if not path or step_idx >= len(path) - 1:
        return
    remaining = path[step_idx:]
    if len(remaining) < 2:
        return
    points = []
    for r, c in remaining:
        px = ox + c * cell_size + cell_size // 2
        py = oy + r * cell_size + cell_size // 2
        points.append((px, py))
    for i in range(len(points) - 1):
        w = 2 if i % 2 == 0 else 1
        c = color if i % 2 == 0 else tuple(max(0, v - 60) for v in color)
        pygame.draw.line(surface, c, points[i], points[i + 1], w)
    gx, gy = points[-1]
    pygame.draw.circle(surface, color, (gx, gy), 4, width=2)


# ═════════════════════════════════════════════════════════════════════════
#  HUD PANEL
# ═════════════════════════════════════════════════════════════════════════

def draw_hud(surface, fonts, allocator, step_count, fps, paused,
             hud_y, win_w):
    f_title, f_body, f_small = fonts

    hud_rect = pygame.Rect(0, hud_y, win_w, HUD_HEIGHT)
    pygame.draw.rect(surface, PANEL_BG, hud_rect)
    pygame.draw.line(surface, PANEL_BORDER, (0, hud_y), (win_w, hud_y), 2)

    x, y = 16, hud_y + 10

    # Title
    title = f_title.render("WAREHOUSE CONTROL CENTER", True, TEXT_ACCENT)
    surface.blit(title, (x, y))
    if paused:
        pw = title.get_width() + 14
        pause_bg = pygame.Rect(x + pw, y, 72, 20)
        pygame.draw.rect(surface, AMBER, pause_bg, border_radius=4)
        pt = f_small.render("PAUSED", True, (15, 23, 42))
        surface.blit(pt, (x + pw + 12, y + 2))
    y += 28

    # Stat cards
    total_tasks = (len(allocator.completed_tasks) +
                   len(allocator.pending_tasks) + sum(allocator.robot_busy))
    stats = [
        ("STEP",      str(step_count),                     TEXT_ACCENT),
        ("SPEED",     f"{fps} fps",                        BLUE_BRIGHT),
        ("DONE",      str(len(allocator.completed_tasks)), GREEN_BRIGHT),
        ("PENDING",   str(len(allocator.pending_tasks)),   RED_BRIGHT),
        ("ACTIVE",    str(sum(allocator.robot_busy)),      AMBER),
    ]

    n_cards = len(stats)
    available_w = win_w - LEGEND_WIDTH - 32
    card_gap = 8
    card_w = min(120, (available_w - card_gap * (n_cards - 1)) // n_cards)
    card_h = 42

    for i, (label, value, vcolor) in enumerate(stats):
        cx = x + i * (card_w + card_gap)
        card = pygame.Rect(cx, y, card_w, card_h)
        draw_rounded_panel(surface, CARD_BG, card, radius=6,
                           border_width=1, border_color=PANEL_BORDER)
        lt = f_small.render(label, True, TEXT_DIM)
        surface.blit(lt, (cx + 8, y + 4))
        vt = f_body.render(value, True, vcolor)
        surface.blit(vt, (cx + 8, y + 20))

    y += card_h + 10

    # Robot state indicators
    n = allocator.n_robots
    rcard_w = min(180, (available_w - (n - 1) * 8) // max(n, 1))
    for rid in range(n):
        fsm = allocator.fsm[rid]
        rc = ROBOT_COLORS.get(fsm.state, TEXT_DIM)
        rx = x + rid * (rcard_w + 8)
        rr = pygame.Rect(rx, y, rcard_w, 36)
        draw_rounded_panel(surface, CARD_BG, rr, radius=5,
                           border_width=1, border_color=PANEL_BORDER)
        pygame.draw.circle(surface, rc, (rx + 14, y + 12), 6)
        pygame.draw.circle(surface, WHITE, (rx + 14, y + 12), 6, width=1)
        nt = f_small.render(f"Robot {rid}", True, TEXT_WHITE)
        surface.blit(nt, (rx + 26, y + 2))
        state_str = fsm.state.name.replace("_", " ").title()
        st = f_small.render(state_str, True, rc)
        surface.blit(st, (rx + 26, y + 18))

    # Controls hint
    controls = "SPACE Pause · +/- Speed · D Tasks · M Metrics · ESC Quit"
    ct = f_small.render(controls, True, TEXT_DIM)
    surface.blit(ct, (x, hud_y + HUD_HEIGHT - 18))


# ═════════════════════════════════════════════════════════════════════════
#  LEGEND PANEL
# ═════════════════════════════════════════════════════════════════════════

def draw_legend(surface, fonts, lx, height):
    _, f_body, f_small = fonts

    rect = pygame.Rect(lx, 0, LEGEND_WIDTH, height)
    pygame.draw.rect(surface, PANEL_BG, rect)
    pygame.draw.line(surface, PANEL_BORDER, (lx, 0), (lx, height), 2)

    x = lx + 14
    y = 14

    t = f_body.render("LEGEND", True, TEXT_ACCENT)
    surface.blit(t, (x, y))
    y += 26

    def _section(label):
        nonlocal y
        st = f_small.render(label, True, TEXT_DIM)
        surface.blit(st, (x, y))
        y += 18

    def _circle_item(label, color):
        nonlocal y
        pygame.draw.circle(surface, color, (x + 7, y + 8), 6)
        pygame.draw.circle(surface, WHITE, (x + 7, y + 8), 6, width=1)
        lt = f_small.render(label, True, TEXT_BRIGHT)
        surface.blit(lt, (x + 20, y + 1))
        y += 22

    def _rect_item(label, color):
        nonlocal y
        pygame.draw.rect(surface, color,
                         pygame.Rect(x, y + 2, 14, 14), border_radius=3)
        lt = f_small.render(label, True, TEXT_BRIGHT)
        surface.blit(lt, (x + 20, y + 1))
        y += 22

    _section("ENVIRONMENT")
    _rect_item("Floor", (30, 41, 59))
    _rect_item("Shelf", (71, 85, 105))
    _circle_item("Pickup Zone", AMBER)
    _circle_item("Drop-off Zone", EMERALD)
    _rect_item("Charging", (129, 140, 248))
    y += 6

    _section("ROBOT STATES")
    _circle_item("Idle", ROBOT_COLORS[RobotState.IDLE])
    _circle_item("To Pickup", ROBOT_COLORS[RobotState.MOVING_TO_PICKUP])
    _circle_item("Picking Up", ROBOT_COLORS[RobotState.PICKING_UP])
    _circle_item("To Dropoff", ROBOT_COLORS[RobotState.MOVING_TO_DROPOFF])
    _circle_item("Dropping Off", ROBOT_COLORS[RobotState.DROPPING_OFF])
    y += 6

    _section("MARKERS")
    _circle_item("Pending Task", RED_BRIGHT)
    _circle_item("Active Pickup", AMBER)
    _circle_item("Active Dropoff", EMERALD)
    _circle_item("Planned Path", TEXT_DIM)


# ═════════════════════════════════════════════════════════════════════════
#  METRICS OVERLAY
# ═════════════════════════════════════════════════════════════════════════

def draw_metrics_overlay(surface, fonts, allocator, ox, oy, gw, gh):
    f_title, f_body, f_small = fonts

    overlay = pygame.Surface((gw, gh), pygame.SRCALPHA)
    overlay.fill((15, 23, 42, 210))
    surface.blit(overlay, (ox, oy))

    s = allocator.metrics.summary_dict()
    x, y = ox + 24, oy + 20

    title = f_title.render("PERFORMANCE METRICS", True, TEXT_ACCENT)
    surface.blit(title, (x, y))
    y += 30

    for line in [
        f"Total Steps:            {s['total_steps']}",
        f"Total Distance:         {s['total_distance']} cells",
        f"Tasks Completed:        {s['total_tasks_completed']}",
        f"Avg Completion Time:    {s['average_completion_time']} steps",
        f"Throughput:             {s['throughput_tasks_per_step']} tasks/step",
    ]:
        t = f_body.render(line, True, TEXT_BRIGHT)
        surface.blit(t, (x, y))
        y += 22
    y += 12

    bar_w = min(280, gw - 80)
    colors_list = list(ROBOT_COLORS.values())
    for r in s["per_robot"]:
        rid = r["robot_id"]
        util = r["utilization"]
        color = colors_list[rid % len(colors_list)]

        label = f_small.render(
            f"R{rid}: dist={r['distance']}  tasks={r['tasks_completed']}",
            True, TEXT_BRIGHT)
        surface.blit(label, (x, y))
        y += 16
        bar = pygame.Rect(x, y, bar_w, 12)
        pygame.draw.rect(surface, CARD_BG, bar, border_radius=4)
        fill = pygame.Rect(x, y, int(bar_w * util), 12)
        pygame.draw.rect(surface, color, fill, border_radius=4)
        pct = f_small.render(f"{util:.0%}", True, WHITE)
        surface.blit(pct, (x + bar_w + 6, y - 2))
        y += 22


# ═════════════════════════════════════════════════════════════════════════
#  BUILD DEMO WORLD
# ═════════════════════════════════════════════════════════════════════════

def build_demo_world(n_robots=4, n_tasks=8, seed=42):
    rng = random.Random(seed)
    rows, cols = 20, 30
    warehouse = Warehouse(rows, cols)
    warehouse.generate_realistic_layout()
    robots = warehouse.generate_robot_starts(n_robots, rng)
    tasks = warehouse.generate_random_tasks(n_tasks, rng)
    return warehouse, robots, tasks


# ═════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════

def main():
    pygame.init()

    # ── Auto-detect screen size and fit window ───────────────────────────
    info = pygame.display.Info()
    screen_w = info.current_w
    screen_h = info.current_h

    # Use 90% of screen, capped sensibly
    win_w = min(int(screen_w * 0.92), 1440)
    win_h = min(int(screen_h * 0.88), 900)

    # Ensure minimums
    win_w = max(win_w, 900)
    win_h = max(win_h, 600)

    screen = pygame.display.set_mode((win_w, win_h), pygame.RESIZABLE)
    pygame.display.set_caption(
        "Multi-Robot Warehouse — Navigation & Task Allocation")
    clock = pygame.time.Clock()

    # ── Fonts (scaled to window) ─────────────────────────────────────────
    font_scale = max(0.7, min(1.0, win_h / 900))
    try:
        f_title = pygame.font.SysFont("segoeui", int(18 * font_scale), bold=True)
        f_body  = pygame.font.SysFont("segoeui", int(14 * font_scale))
        f_small = pygame.font.SysFont("segoeui", int(12 * font_scale))
        f_robot = pygame.font.SysFont("segoeui", int(11 * font_scale), bold=True)
    except Exception:
        f_title = pygame.font.SysFont("arial", int(18 * font_scale), bold=True)
        f_body  = pygame.font.SysFont("arial", int(14 * font_scale))
        f_small = pygame.font.SysFont("arial", int(12 * font_scale))
        f_robot = pygame.font.SysFont("arial", int(11 * font_scale), bold=True)
    fonts = (f_title, f_body, f_small)

    print_state_diagram()

    # Build world
    warehouse, robots, tasks = build_demo_world(n_robots=4, n_tasks=8)
    allocator = RobotTaskAllocator(warehouse, robots, tasks)

    step_count = 0
    fps = FPS_DEFAULT
    paused = False
    running = True
    show_metrics = False
    frame_counter = 0

    while running:
        # Current window dimensions (support resize)
        win_w, win_h = screen.get_size()

        # ── Events ───────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                win_w, win_h = event.w, event.h
                screen = pygame.display.set_mode(
                    (win_w, win_h), pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                   pygame.K_KP_PLUS):
                    fps = min(fps + 2, 30)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    fps = max(fps - 2, 1)
                elif event.key == pygame.K_d:
                    new_tasks = warehouse.generate_random_tasks(
                        3, random.Random())
                    allocator.add_tasks(new_tasks)
                    print(f"[dynamic] Injected {len(new_tasks)} new tasks.")
                elif event.key == pygame.K_m:
                    show_metrics = not show_metrics
                    if show_metrics:
                        allocator.metrics.print_report()

        # ── Compute layout from current window size ──────────────────────
        legend_w = min(LEGEND_WIDTH, int(win_w * 0.16))
        hud_h = min(HUD_HEIGHT, int(win_h * 0.22))

        grid_area_w = win_w - legend_w
        grid_area_h = win_h - hud_h

        cell_size = calculate_cell_size(
            warehouse.warehouse.shape,
            grid_area_w - 20,
            grid_area_h - 10,
        )
        cell_size = max(14, cell_size)

        grid_px_w = cell_size * warehouse.col
        grid_px_h = cell_size * warehouse.row
        offset_x = max(4, (grid_area_w - grid_px_w) // 2)
        offset_y = max(4, (grid_area_h - grid_px_h) // 2)

        # ── Advance simulation ───────────────────────────────────────────
        if not paused and not allocator.is_done():
            allocator.step()
            step_count += 1
        frame_counter += 1

        # ── Render ───────────────────────────────────────────────────────
        screen.fill(BG_DARK)

        # Grid
        draw_grid(warehouse.warehouse, screen, cell_size, offset_x, offset_y)

        # Pending markers
        for pickup, dropoff in allocator.pending_tasks:
            draw_marker_glow(screen, pickup, cell_size, RED_BRIGHT,
                             offset_x, offset_y, frame_counter)
            draw_marker_glow(screen, dropoff, cell_size, (180, 80, 80),
                             offset_x, offset_y, frame_counter)

        # Active pickup markers
        for pos in allocator.active_pickups():
            draw_marker_glow(screen, pos, cell_size, AMBER,
                             offset_x, offset_y, frame_counter)

        # Active dropoff markers
        for pos in allocator.active_dropoffs():
            draw_marker_glow(screen, pos, cell_size, EMERALD,
                             offset_x, offset_y, frame_counter)

        # Path trails
        for rid in range(allocator.n_robots):
            if allocator.robot_busy[rid] and allocator.robot_path[rid]:
                rc = ROBOT_COLORS.get(allocator.fsm[rid].state, TEXT_DIM)
                draw_path_trail(screen, allocator.robot_path[rid],
                                allocator.robot_step_idx[rid],
                                cell_size, rc, offset_x, offset_y)

        # Robots
        for rid in range(allocator.n_robots):
            fsm = allocator.fsm[rid]
            color = ROBOT_COLORS.get(fsm.state, TEXT_DIM)
            draw_robot_premium(screen, allocator.robots[rid], cell_size,
                               color, rid, f_robot, offset_x, offset_y,
                               frame_counter)

        # Completion banner
        if allocator.is_done() and step_count > 0:
            bw, bh = min(360, grid_area_w - 40), 60
            bx = (grid_area_w - bw) // 2
            by = (grid_area_h - bh) // 2
            banner = pygame.Rect(bx, by, bw, bh)
            draw_rounded_panel(screen, (16, 185, 129), banner, radius=12,
                               border_width=3, border_color=WHITE)
            dt = f_title.render("ALL TASKS COMPLETED", True, WHITE)
            screen.blit(dt, dt.get_rect(
                center=(banner.centerx, banner.centery - 8)))
            sub = f_small.render(
                f"{len(allocator.completed_tasks)} tasks in {step_count} steps",
                True, (220, 255, 230))
            screen.blit(sub, sub.get_rect(
                center=(banner.centerx, banner.centery + 12)))

        # Legend
        draw_legend(screen, fonts, grid_area_w, grid_area_h)

        # HUD
        draw_hud(screen, fonts, allocator, step_count, fps, paused,
                 grid_area_h, win_w)

        # Metrics overlay
        if show_metrics:
            draw_metrics_overlay(screen, fonts, allocator,
                                 offset_x, offset_y,
                                 grid_px_w, grid_px_h)

        pygame.display.flip()
        clock.tick(fps)

    # ── Post-simulation ──────────────────────────────────────────────────
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print("=" * 60)
    allocator.metrics.print_report()
    allocator.metrics.plot_metrics(output_dir=output_dir)

    mermaid = generate_mermaid_fsm()
    snapshot = generate_snapshot_diagram(allocator.fsm, step_count)
    with open(os.path.join(output_dir, "fsm_diagram.md"), "w") as f:
        f.write("# Robot FSM State Diagram\n\n```mermaid\n")
        f.write(mermaid)
        f.write("\n```\n\n")
        f.write(f"# Final State Snapshot (Step {step_count})\n\n```mermaid\n")
        f.write(snapshot)
        f.write("\n```\n")
    print("[output] FSM diagrams saved to output/fsm_diagram.md")

    pygame.quit()


if __name__ == "__main__":
    main()