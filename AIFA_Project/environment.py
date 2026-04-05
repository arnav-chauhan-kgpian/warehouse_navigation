"""
environment.py — Warehouse Environment Module
================================================
Defines the warehouse grid, cell types, layout generation, and rendering utilities.
Supports realistic shelf/aisle layouts, staging areas, and charging stations.
"""

import numpy as np
import pygame
import random

# ── Cell type constants ──────────────────────────────────────────────────
EMPTY = 0
SHELF = 1
ROBOT = 2
PICKUP = 3
DROPOFF = 4
CHARGING = 5

# ── Premium Colour palette (RGB) ────────────────────────────────────────
CELL_COLORS = {
    EMPTY:    (30, 41, 59),      # Dark slate floor
    SHELF:    (71, 85, 105),     # Slate shelves with visible contrast
    PICKUP:   (251, 191, 36),    # Amber pickup markers
    DROPOFF:  (52, 211, 153),    # Emerald drop-off markers
    CHARGING: (99, 102, 241),    # Indigo charging pads
}

GRID_LINE_COLOR = (51, 65, 85)    # Subtle slate grid lines


def calculate_cell_size(array_shape, window_width, window_height=None):
    """Return the largest cell size that fits both dimensions."""
    cell_w = window_width // array_shape[1]
    if window_height is not None:
        cell_h = window_height // array_shape[0]
        return min(cell_w, cell_h)
    return cell_w


def draw_grid(array, surface, cell_size, offset_x=0, offset_y=0):
    """Render the warehouse grid with premium styling."""
    rows, cols = array.shape
    for r in range(rows):
        for c in range(cols):
            rect = pygame.Rect(
                offset_x + c * cell_size,
                offset_y + r * cell_size,
                cell_size, cell_size,
            )
            cell_val = int(array[r, c])
            color = CELL_COLORS.get(cell_val, CELL_COLORS[EMPTY])

            # Shelves get a subtle 3D raised look
            if cell_val == SHELF:
                # Shadow
                shadow = pygame.Rect(rect.x + 2, rect.y + 2, rect.w, rect.h)
                pygame.draw.rect(surface, (15, 23, 42), shadow, border_radius=3)
                # Main shelf block
                pygame.draw.rect(surface, color, rect, border_radius=3)
                # Highlight edge
                highlight = pygame.Rect(rect.x, rect.y, rect.w, 2)
                pygame.draw.rect(surface, (100, 116, 139), highlight)
            elif cell_val == PICKUP:
                pygame.draw.rect(surface, CELL_COLORS[EMPTY], rect)
                # Amber diamond marker
                cx = rect.x + cell_size // 2
                cy = rect.y + cell_size // 2
                s = cell_size // 4
                pygame.draw.polygon(surface, (251, 191, 36, 80),
                                    [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)])
                pygame.draw.rect(surface, GRID_LINE_COLOR, rect, width=1)
            elif cell_val == DROPOFF:
                pygame.draw.rect(surface, CELL_COLORS[EMPTY], rect)
                # Emerald diamond marker
                cx = rect.x + cell_size // 2
                cy = rect.y + cell_size // 2
                s = cell_size // 4
                pygame.draw.polygon(surface, (52, 211, 153, 80),
                                    [(cx, cy - s), (cx + s, cy), (cx, cy + s), (cx - s, cy)])
                pygame.draw.rect(surface, GRID_LINE_COLOR, rect, width=1)
            elif cell_val == CHARGING:
                pygame.draw.rect(surface, (20, 27, 46), rect)
                # Lightning bolt icon
                cx = rect.x + cell_size // 2
                cy = rect.y + cell_size // 2
                s = cell_size // 5
                pygame.draw.polygon(surface, (99, 102, 241),
                                    [(cx - s//2, cy - s), (cx + s//2, cy), (cx - s//3, cy),
                                     (cx + s//2, cy + s), (cx - s//2, cy), (cx + s//3, cy)])
                pygame.draw.rect(surface, GRID_LINE_COLOR, rect, width=1)
            else:
                # Floor tile — subtle checkerboard pattern for depth
                if (r + c) % 2 == 0:
                    pygame.draw.rect(surface, (30, 41, 59), rect)
                else:
                    pygame.draw.rect(surface, (33, 45, 64), rect)
                pygame.draw.rect(surface, GRID_LINE_COLOR, rect, width=1)


class Warehouse:
    """Manages the warehouse grid and provides layout generation utilities."""

    def __init__(self, rows, cols):
        self.row = rows
        self.col = cols
        self.warehouse = np.zeros((rows, cols), dtype=int)

    # ── Basic API (backward compatible) ──────────────────────────────────
    def create_environment(self):
        self.warehouse = np.zeros((self.row, self.col), dtype=int)
        return self.warehouse

    def add_obstacles(self, obstacles):
        """*obstacles* can be a tuple of arrays (row_indices, col_indices)."""
        self.warehouse[obstacles] = SHELF
        return self.warehouse

    def add_robot(self, robot):
        self.warehouse[robot] = ROBOT
        return self.warehouse

    def add_target(self, target):
        self.warehouse[target] = PICKUP
        return self.warehouse

    def is_valid_position(self, position):
        r, c = position
        return 0 <= r < self.row and 0 <= c < self.col

    def is_walkable(self, position):
        """True if position is in-bounds and not a shelf."""
        if not self.is_valid_position(position):
            return False
        return self.warehouse[position[0], position[1]] != SHELF

    # ── Layout generation ────────────────────────────────────────────────
    def generate_realistic_layout(self, shelf_rows=None, aisle_width=2,
                                   shelf_block_width=3, margin=2):
        """
        Generate a realistic warehouse with:
          • Rows of shelves separated by aisles
          • Left staging area (pickup zone)
          • Right staging area (drop-off zone)
          • Charging stations in corners
        """
        self.warehouse = np.zeros((self.row, self.col), dtype=int)

        # Determine shelf row positions
        if shelf_rows is None:
            shelf_rows = []
            r = margin + 1
            while r + shelf_block_width <= self.row - margin:
                shelf_rows.append(r)
                r += shelf_block_width + aisle_width

        # Place shelves (leaving left & right margins for staging)
        staging_left = margin + 2
        staging_right = self.col - margin - 3
        for sr in shelf_rows:
            for dr in range(shelf_block_width):
                row_idx = sr + dr
                if row_idx >= self.row - margin:
                    break
                # Place shelf segments with gaps for cross-aisles
                c = staging_left + 1
                while c < staging_right:
                    # shelf segment of length 4, then a gap of 2
                    for sc in range(min(4, staging_right - c)):
                        self.warehouse[row_idx, c + sc] = SHELF
                    c += 6  # 4 shelf + 2 gap

        # Mark staging areas
        for r in range(margin, self.row - margin):
            if self.warehouse[r, staging_left] == EMPTY:
                self.warehouse[r, staging_left] = PICKUP
            if self.warehouse[r, staging_right] == EMPTY:
                self.warehouse[r, staging_right] = DROPOFF

        # Charging stations in corners
        corners = [
            (1, 1), (1, self.col - 2),
            (self.row - 2, 1), (self.row - 2, self.col - 2),
        ]
        for cr, cc in corners:
            if self.is_valid_position((cr, cc)):
                self.warehouse[cr, cc] = CHARGING

        return self.warehouse

    # ── Utility helpers ──────────────────────────────────────────────────
    def get_free_positions(self):
        """Return list of (row, col) positions that are walkable (EMPTY)."""
        positions = []
        for r in range(self.row):
            for c in range(self.col):
                if self.warehouse[r, c] == EMPTY:
                    positions.append((r, c))
        return positions

    def get_positions_of_type(self, cell_type):
        """Return all positions matching *cell_type*."""
        positions = []
        for r in range(self.row):
            for c in range(self.col):
                if self.warehouse[r, c] == cell_type:
                    positions.append((r, c))
        return positions

    def generate_random_tasks(self, n_tasks, rng=None):
        """
        Generate *n_tasks* pick-and-place tasks as (pickup, dropoff) tuples.
        Pickup locations are chosen from PICKUP cells (or free cells near left).
        Dropoff locations are chosen from DROPOFF cells (or free cells near right).
        """
        if rng is None:
            rng = random.Random()

        pickups = self.get_positions_of_type(PICKUP)
        dropoffs = self.get_positions_of_type(DROPOFF)

        # Fall back to free positions if no marked zones
        if not pickups:
            free = self.get_free_positions()
            pickups = [p for p in free if p[1] < self.col // 3]
        if not dropoffs:
            free = self.get_free_positions()
            dropoffs = [p for p in free if p[1] > 2 * self.col // 3]

        tasks = []
        for _ in range(n_tasks):
            p = rng.choice(pickups) if pickups else (0, 0)
            d = rng.choice(dropoffs) if dropoffs else (self.row - 1, self.col - 1)
            tasks.append((p, d))
        return tasks

    def generate_robot_starts(self, n_robots, rng=None):
        """Place robots on free cells near the left wall."""
        if rng is None:
            rng = random.Random()
        free = [p for p in self.get_free_positions() if p[1] <= 3]
        if not free:
            free = self.get_free_positions()
        rng.shuffle(free)
        return free[:n_robots]
