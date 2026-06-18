"""
map_geometry.py  –  Lemon Heavy Metal Engine
DOOM-style vertex/linedef map representation.

All geometry is stored as flat, contiguous NumPy arrays so Numba @njit
functions can consume them directly without touching Python objects.

Array layout
------------
vertices  : float32 [N, 2]   – (x, y) world coordinates
linedefs  : int32   [M, 4]   – (v_start, v_end, front_sector, back_sector)
                               back_sector == -1 means solid wall (no sector behind)
sectors   : float32 [S, 2]   – (floor_height, ceil_height) per sector

Coordinate system
-----------------
+X = right, +Y = up (standard 2D math convention).
All units are world units (1.0 = one notional metre).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# Public types (thin named wrappers so call sites are readable)
# ---------------------------------------------------------------------------

class MapGeometry:
    """Immutable container for a loaded map's flat NumPy geometry arrays."""

    __slots__ = ("vertices", "linedefs", "sectors", "name", "spawn")

    def __init__(
        self,
        vertices: np.ndarray,                        # float32 [N, 2]
        linedefs: np.ndarray,                        # int32   [M, 4]
        sectors:  np.ndarray,                        # float32 [S, 2]
        name:     str = "unnamed",
        spawn:    tuple[float, float] = (1.5, 1.5),  # world-space player start
    ) -> None:
        self.vertices = vertices
        self.linedefs = linedefs
        self.sectors  = sectors
        self.name     = name
        self.spawn    = spawn

    # Convenience read-only properties
    @property
    def vertex_count(self) -> int:
        return self.vertices.shape[0]

    @property
    def linedef_count(self) -> int:
        return self.linedefs.shape[0]

    @property
    def sector_count(self) -> int:
        return self.sectors.shape[0]

    def __repr__(self) -> str:
        return (
            f"MapGeometry(name={self.name!r}, "
            f"vertices={self.vertex_count}, "
            f"linedefs={self.linedef_count}, "
            f"sectors={self.sector_count})"
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_map(source: Union[dict, str, Path]) -> MapGeometry:
    """
    Load a map from a dict or a JSON file path and return a MapGeometry.

    Expected dict schema
    --------------------
    {
        "name": "my_map",                       # optional
        "sectors": [
            {"floor_height": 0.0, "ceil_height": 2.0},
            ...
        ],
        "vertices": [
            {"x": 0.0, "y": 0.0},
            ...
        ],
        "linedefs": [
            {
                "v_start": 0,
                "v_end":   1,
                "front_sector": 0,
                "back_sector":  -1          # -1 = solid (no sector behind)
            },
            ...
        ]
    }
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as fh:
            data: dict = json.load(fh)
    else:
        data = source

    name: str = data.get("name", "unnamed")

    # --- sectors -----------------------------------------------------------
    raw_sectors = data["sectors"]
    sectors = np.array(
        [[s["floor_height"], s["ceil_height"]] for s in raw_sectors],
        dtype=np.float32,
    )                                                   # [S, 2]

    # --- vertices ----------------------------------------------------------
    raw_verts = data["vertices"]
    vertices = np.array(
        [[v["x"], v["y"]] for v in raw_verts],
        dtype=np.float32,
    )                                                   # [N, 2]

    # --- linedefs ----------------------------------------------------------
    raw_lines = data["linedefs"]
    linedefs = np.array(
        [
            [
                ld["v_start"],
                ld["v_end"],
                ld["front_sector"],
                ld.get("back_sector", -1),
            ]
            for ld in raw_lines
        ],
        dtype=np.int32,
    )                                                   # [M, 4]

    _validate(vertices, linedefs, sectors)

    return MapGeometry(vertices=vertices, linedefs=linedefs, sectors=sectors, name=name)


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _validate(
    vertices: np.ndarray,
    linedefs: np.ndarray,
    sectors:  np.ndarray,
) -> None:
    n_verts   = vertices.shape[0]
    n_sectors = sectors.shape[0]

    v_start = linedefs[:, 0]
    v_end   = linedefs[:, 1]
    front   = linedefs[:, 2]
    back    = linedefs[:, 3]

    if np.any(v_start < 0) or np.any(v_start >= n_verts):
        raise ValueError("linedef v_start index out of range")
    if np.any(v_end < 0) or np.any(v_end >= n_verts):
        raise ValueError("linedef v_end index out of range")
    if np.any(front < 0) or np.any(front >= n_sectors):
        raise ValueError("linedef front_sector index out of range")

    bad_back = back[(back != -1)]
    if len(bad_back) and (np.any(bad_back < 0) or np.any(bad_back >= n_sectors)):
        raise ValueError("linedef back_sector index out of range")


# ---------------------------------------------------------------------------
# Built-in test map  –  two connected rectangular rooms
# ---------------------------------------------------------------------------
#
#   Room A (sector 0):  vertices 0-3   (a 6×6 unit square, bottom-left at origin)
#   Room B (sector 1):  vertices 4-7   (a 6×6 unit square, to the right)
#   Portal opening:     shared edge between the two rooms, 2 units wide,
#                       centred on the shared wall (x=6, y=2 to y=4).
#
#   Y
#   ^
#   6 +--+--+--+
#     |  A  |  B  |
#   0 +--+--+--+
#     0     6    12  -> X
#
# Linedef winding convention: front sector is on the LEFT of the directed line
# (same as DOOM).  Outer walls face inward, so they are wound clockwise
# around the exterior, which puts the interior sector on the left.

_TWO_ROOM_MAP: dict = {
    "name": "two_room_test",
    "sectors": [
        {"floor_height": 0.0, "ceil_height": 2.5},   # 0 – Room A
        {"floor_height": 0.0, "ceil_height": 2.5},   # 1 – Room B
    ],
    "vertices": [
        # Room A corners
        {"x":  0.0, "y":  0.0},   # 0
        {"x":  6.0, "y":  0.0},   # 1
        {"x":  6.0, "y":  6.0},   # 2
        {"x":  0.0, "y":  6.0},   # 3
        # Room B corners  (shares the x=6 wall with Room A)
        # Note: vertices 1 and 2 are reused for the shared wall endpoints.
        # Extra vertices for the opening edges:
        {"x":  6.0, "y":  2.0},   # 4  – portal bottom  (on shared wall)
        {"x":  6.0, "y":  4.0},   # 5  – portal top     (on shared wall)
        # Room B far corners
        {"x": 12.0, "y":  0.0},   # 6
        {"x": 12.0, "y":  6.0},   # 7
    ],
    "linedefs": [
        # --- Room A outer walls (solid, back_sector = -1) ---
        # South wall  A: v0 -> v1  (front = sector 0, interior on left)
        {"v_start": 0, "v_end": 1, "front_sector": 0, "back_sector": -1},
        # Shared wall south segment: v1 -> v4  (solid, Room A side)
        {"v_start": 1, "v_end": 4, "front_sector": 0, "back_sector": -1},
        # Portal opening: v4 -> v5  (two-sided, connects Room A and Room B)
        {"v_start": 4, "v_end": 5, "front_sector": 0, "back_sector":  1},
        # Shared wall north segment: v5 -> v2  (solid, Room A side)
        {"v_start": 5, "v_end": 2, "front_sector": 0, "back_sector": -1},
        # North wall A: v2 -> v3  (solid)
        {"v_start": 2, "v_end": 3, "front_sector": 0, "back_sector": -1},
        # West wall A:  v3 -> v0  (solid)
        {"v_start": 3, "v_end": 0, "front_sector": 0, "back_sector": -1},

        # --- Room B outer walls (solid, back_sector = -1) ---
        # South wall B: v6 -> v1  (front = sector 1, interior on left)
        {"v_start": 6, "v_end": 1, "front_sector": 1, "back_sector": -1},
        # Shared wall south segment B side: v4 -> v6  ...wait, winding:
        # From B's perspective the shared wall is v4->v1 going south, but
        # we already have that linedef from A.  For the portal we already
        # emitted a two-sided linedef (v4->v5) above.  The remaining solid
        # shared segments are one-sided from the A side only (no duplicate
        # needed because the renderer reads both sides from one linedef).

        # East wall B:  v6 -> v7  (solid)
        {"v_start": 6, "v_end": 7, "front_sector": 1, "back_sector": -1},
        # North wall B: v7 -> v5  (solid)
        {"v_start": 7, "v_end": 5, "front_sector": 1, "back_sector": -1},
        # Shared wall north segment B side: v5 -> v2 already covered above.
        # South wall B bottom: v1 -> v6 already covered above (reversed winding).
        # Bottom of Room B from v1 to v6:
        {"v_start": 1, "v_end": 6, "front_sector": 1, "back_sector": -1},
        # North ceiling strip: v5 -> v7 ... covered by east/north above.
        # Fill missing north side of B ceiling strip
        {"v_start": 2, "v_end": 7, "front_sector": 1, "back_sector": -1},
    ],
}


def make_test_map() -> MapGeometry:
    """Return the built-in two-room test map as a MapGeometry."""
    return load_map(_TWO_ROOM_MAP)


# ---------------------------------------------------------------------------
# Grid → linedef geometry converter
# ---------------------------------------------------------------------------

def geometry_from_grid(
    grid: np.ndarray,
    name: str = "generated",
    spawn: tuple[float, float] = (1.5, 1.5),
) -> "MapGeometry":
    """
    Convert a uint8 tile grid (0 = floor, nonzero = wall) into a MapGeometry.

    Grid coordinate convention (matches Map / MapGenerator):
      grid[row, col]  –  row 0 is the top of the map
      World x = col,  world y = row  (+y is downward, matching the grid)

    Boundary linedefs are run-length merged along each axis, so a straight
    corridor wall emits one linedef rather than one per cell.  This keeps
    linedef counts low and produces better-balanced BSP trees.

    Winding convention (front sector on LEFT of directed line, y-down):
      North boundary → directed east
      South boundary → directed west
      West boundary  → directed north  (y decreasing)
      East boundary  → directed south  (y increasing)

    All floor space is sector 0.  Sector data: floor_height=0.0, ceil_height=2.5.
    """
    rows, cols = grid.shape

    # Vertex deduplication
    _verts: dict[tuple[int, int], int] = {}
    _vlist: list[list[float]] = []
    _ldefs: list[list[int]]   = []   # [v0, v1, front_sector=0, back_sector=-1]

    def _vid(x: int, y: int) -> int:
        k = (x, y)
        if k not in _verts:
            _verts[k] = len(_vlist)
            _vlist.append([float(x), float(y)])
        return _verts[k]

    def _emit(x1: int, y1: int, x2: int, y2: int) -> None:
        _ldefs.append([_vid(x1, y1), _vid(x2, y2), 0, -1])

    # ── North boundaries: floor(r,c) with wall/OOB above → y=r, going EAST ──
    for r in range(rows):
        in_run = False
        run_c  = 0
        for c in range(cols + 1):
            active = (c < cols
                      and grid[r, c] == 0
                      and (r == 0 or grid[r - 1, c] != 0))
            if active and not in_run:
                run_c  = c
                in_run = True
            elif not active and in_run:
                _emit(run_c, r, c, r)   # east: (run_c, r) → (c, r)
                in_run = False

    # ── South boundaries: floor(r,c) with wall/OOB below → y=r+1, going WEST ──
    for r in range(rows):
        in_run = False
        run_c  = 0
        for c in range(cols + 1):
            active = (c < cols
                      and grid[r, c] == 0
                      and (r == rows - 1 or grid[r + 1, c] != 0))
            if active and not in_run:
                run_c  = c
                in_run = True
            elif not active and in_run:
                _emit(c, r + 1, run_c, r + 1)  # west: (c, r+1) → (run_c, r+1)
                in_run = False

    # ── West boundaries: floor(r,c) with wall/OOB left → x=c, going NORTH ──
    for c in range(cols):
        in_run = False
        run_r  = 0
        for r in range(rows + 1):
            active = (r < rows
                      and grid[r, c] == 0
                      and (c == 0 or grid[r, c - 1] != 0))
            if active and not in_run:
                run_r  = r
                in_run = True
            elif not active and in_run:
                _emit(c, r, c, run_r)           # north: (c, r) → (c, run_r)
                in_run = False

    # ── East boundaries: floor(r,c) with wall/OOB right → x=c+1, going SOUTH ──
    for c in range(cols):
        in_run = False
        run_r  = 0
        for r in range(rows + 1):
            active = (r < rows
                      and grid[r, c] == 0
                      and (c == cols - 1 or grid[r, c + 1] != 0))
            if active and not in_run:
                run_r  = r
                in_run = True
            elif not active and in_run:
                _emit(c + 1, run_r, c + 1, r)  # south: (c+1, run_r) → (c+1, r)
                in_run = False

    if not _vlist or not _ldefs:
        raise ValueError("geometry_from_grid: grid contains no floor cells with wall boundaries")

    vertices = np.array(_vlist, dtype=np.float32)
    linedefs = np.array(_ldefs, dtype=np.int32)
    sectors  = np.array([[0.0, 2.5]], dtype=np.float32)

    return MapGeometry(
        vertices=vertices,
        linedefs=linedefs,
        sectors=sectors,
        name=name,
        spawn=spawn,
    )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    geo = make_test_map()
    print(geo)
    print(f"  vertices:\n{geo.vertices}")
    print(f"  linedefs:\n{geo.linedefs}")
    print(f"  sectors:\n{geo.sectors}")
    print("Validation passed.")
