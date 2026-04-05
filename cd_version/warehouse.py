"""
warehouse.py — Warehouse Grid Environment
==========================================
Generates a realistic warehouse grid with shelf rows and cross-aisles.
The grid is a 2-D list where:
    0 = free (aisle / open space)
    1 = obstacle (shelf rack)

Layout strategy
---------------
Shelves are arranged in horizontal bands separated by cross-aisles.
Within each band, shelf blocks of width ``shelf_width`` are separated
by narrow column-aisles of width ``aisle_gap``.
Wide receiving/staging margins are left at the top, bottom, left and right.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np


class Warehouse:
    """
    2-D grid warehouse with configurable shelf layout.

    Parameters
    ----------
    rows, cols   : Grid dimensions.
    shelf_height : Number of rows each shelf block occupies.
    shelf_width  : Number of columns each shelf block occupies.
    aisle_gap    : Free columns between consecutive shelf blocks (column-aisle)
                   and free rows between shelf bands (cross-aisle).
    margin       : Border of free cells around the warehouse perimeter.
    seed         : Random seed for reproducible free-cell sampling.
    """

    def __init__(
        self,
        rows:         int = 22,
        cols:         int = 24,
        shelf_height: int = 2,
        shelf_width:  int = 3,
        aisle_gap:    int = 2,
        margin:       int = 2,
        seed:         int = 42,
    ) -> None:
        self.rows         = rows
        self.cols         = cols
        self.shelf_height = shelf_height
        self.shelf_width  = shelf_width
        self.aisle_gap    = aisle_gap
        self.margin       = margin
        self.seed         = seed
        self._rng         = random.Random(seed)

        self.grid:       List[List[int]]       = self._build_grid()
        self.free_cells: List[Tuple[int, int]] = self._collect_free_cells()

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    def _build_grid(self) -> List[List[int]]:
        """Create the warehouse grid with shelf racks placed row by row."""
        grid  = [[0] * self.cols for _ in range(self.rows)]
        m     = self.margin
        step  = self.shelf_height + self.aisle_gap   # shelf band + cross-aisle

        row = m
        while row + self.shelf_height <= self.rows - m:
            col = m
            while col + self.shelf_width <= self.cols - m:
                # Fill one shelf block
                for dr in range(self.shelf_height):
                    for dc in range(self.shelf_width):
                        grid[row + dr][col + dc] = 1
                col += self.shelf_width + self.aisle_gap

            row += step

        return grid

    def _collect_free_cells(self) -> List[Tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == 0
        ]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rows, self.cols

    def is_free(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return (
            0 <= r < self.rows
            and 0 <= c < self.cols
            and self.grid[r][c] == 0
        )

    def random_free_cell(
        self,
        exclude: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        """
        Return a uniformly random free cell, optionally excluding a set.

        Raises ``ValueError`` if no eligible cells remain.
        """
        blocked = set(exclude or [])
        choices = [c for c in self.free_cells if c not in blocked]
        if not choices:
            raise ValueError(
                "No free cells left after exclusions — reduce robots/tasks."
            )
        return self._rng.choice(choices)

    def as_numpy(self) -> np.ndarray:
        """Return grid as a float32 NumPy array (0.0 free, 1.0 obstacle)."""
        return np.array(self.grid, dtype=np.float32)

    def render_ascii(self) -> str:
        """Return a human-readable ASCII representation of the grid."""
        chars = {0: "·", 1: "█"}
        return "\n".join("".join(chars[cell] for cell in row) for row in self.grid)

    def summary(self) -> str:
        total  = self.rows * self.cols
        free   = len(self.free_cells)
        blocked = total - free
        return (
            f"Warehouse {self.rows}×{self.cols}  |  "
            f"Free: {free}  |  Shelves: {blocked}  |  "
            f"Occupancy: {blocked/total*100:.1f}%"
        )
