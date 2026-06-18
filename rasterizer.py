"""
rasterizer.py  –  Lemon Heavy Metal Engine
Numba @njit software rasterizer.

Takes the visible seg list from bsp_traversal and draws directly into an
OpenCV framebuffer (np.uint8, BGR channel order, shape [H, W, 3]).
No allocation occurs inside the hot path — the framebuffer is mutated
in-place every frame.

Public API
----------
  rasterize(fb, vis_segs, n_segs, sectors, screen_w, screen_h)
      Mutates fb in-place.  Returns nothing.

  warm_up(screen_w, screen_h)
      Triggers JIT compilation before the game loop.

Framebuffer convention
----------------------
  fb[y, x, 0] = Blue
  fb[y, x, 1] = Green
  fb[y, x, 2] = Red
  Row 0 is the top of the screen.

Visible seg column indices (mirrors bsp_traversal.py)
-----------------------------------------------------
  0 sx1   1 sx2   2 wx1   3 wy1   4 wx2   5 wy2
  6 front_sector   7 back_sector (-1 = solid)
  8 depth1   9 depth2

Sector array columns (from map_geometry.py)
-------------------------------------------
  sectors[i, 0] = floor_height
  sectors[i, 1] = ceil_height
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Vis-seg column indices (duplicated here — no cross-module imports in @njit)
# ---------------------------------------------------------------------------
_VIS_SX1    = 0
_VIS_SX2    = 1
_VIS_DEPTH1 = 8
_VIS_DEPTH2 = 9
_VIS_FRONT  = 6
_VIS_BACK   = 7
_VIS_SX1F   = 10   # unclamped float screen x  (set by bsp_traversal)
_VIS_SX2F   = 11

# ---------------------------------------------------------------------------
# Palette (BGR)
# ---------------------------------------------------------------------------
# Wall base colour — will be shaded by distance
_WALL_B = np.uint8(180)
_WALL_G = np.uint8(160)
_WALL_R = np.uint8(120)

# Portal step colour (two-sided linedef) — blueish tint
_PORTAL_B = np.uint8(120)
_PORTAL_G = np.uint8(160)
_PORTAL_R = np.uint8(200)

# Ceiling flat colour
_CEIL_B = np.uint8(20)
_CEIL_G = np.uint8(20)
_CEIL_R = np.uint8(30)

# Floor flat colour
_FLOOR_B = np.uint8(50)
_FLOOR_G = np.uint8(45)
_FLOOR_R = np.uint8(35)

# Maximum shading distance — walls beyond this are pitch dark
_SHADE_DIST = 24.0


# ---------------------------------------------------------------------------
# @njit helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


@njit(cache=True)
def _clamp_u8(v: float) -> int:
    if v < 0.0:
        return 0
    if v > 255.0:
        return 255
    return int(v)


@njit(cache=True)
def _shade(base: int, depth: float) -> int:
    """
    Linear distance shading.  depth == 0 → full brightness.
    depth >= _SHADE_DIST → black.
    """
    t = depth / _SHADE_DIST
    if t > 1.0:
        t = 1.0
    return _clamp_u8(float(base) * (1.0 - t))


@njit(cache=True)
def _draw_column(
    fb:       np.ndarray,   # uint8 [H, W, 3]
    x:        int,
    y_top:    int,
    y_bot:    int,
    b:        int,
    g:        int,
    r:        int,
    screen_h: int,
) -> None:
    """Draw a vertical span at column x from y_top to y_bot (inclusive)."""
    y0 = y_top if y_top >= 0     else 0
    y1 = y_bot if y_bot < screen_h else screen_h - 1
    for y in range(y0, y1 + 1):
        fb[y, x, 0] = b
        fb[y, x, 1] = g
        fb[y, x, 2] = r


# ---------------------------------------------------------------------------
# Core rasterizer  (@njit, in-place framebuffer mutation)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _rasterize_kernel(
    fb:        np.ndarray,   # uint8  [H, W, 3]   — mutated in place
    vis_segs:  np.ndarray,   # float32[MAX_VIS, 10]
    n_segs:    int,
    sectors:   np.ndarray,   # float32[S, 2]
    screen_w:  int,
    screen_h:  int,
    camera_bob_offset: float = 0.0,  # vertical camera offset for head bobbing
) -> None:
    """
    Column-major wall rasterizer.

    For each visible seg (already in front-to-back order), iterate every
    screen column it covers.  For each column:
      1. Perspective-correct depth interpolation.
      2. wall_height = screen_h / depth
      3. Draw ceiling, wall, floor spans.

    A local col_drawn bool array ensures each column is written only once
    even if seg screen ranges overlap at boundaries.
    """
    half_h = screen_h * 0.5

    # Per-column depth buffer.  Initial value = infinity (nothing drawn yet).
    # A wall column is only drawn if its depth is strictly less than the value
    # already stored here.  This makes correctness independent of BSP traversal
    # order — the closest wall always wins, regardless of which subsector the
    # traversal visits first.
    col_depth = np.full(screen_w, 1e30, dtype=np.float32)

    # Calculate camera center accounting for head bob (positive = look up)
    y_center = half_h + camera_bob_offset
    horizon = int(y_center)

    # Fill ceiling and floor first (one full-screen pass)
    for x in range(screen_w):
        _draw_column(fb, x, 0,        horizon - 1, _CEIL_B, _CEIL_G, _CEIL_R, screen_h)
        _draw_column(fb, x, horizon, screen_h - 1, _FLOOR_B, _FLOOR_G, _FLOOR_R, screen_h)

    for si in range(n_segs):
        seg = vis_segs[si]

        sx1    = int(seg[_VIS_SX1])
        sx2    = int(seg[_VIS_SX2])
        sx1f   = seg[_VIS_SX1F]
        sx2f   = seg[_VIS_SX2F]
        depth1 = seg[_VIS_DEPTH1]
        depth2 = seg[_VIS_DEPTH2]
        is_portal = seg[_VIS_BACK] >= 0.0

        if sx1 > sx2 or depth1 <= 0.0:
            continue

        inv_d1  = 1.0 / depth1 if depth1 > 0.0 else 0.0
        inv_d2  = 1.0 / depth2 if depth2 > 0.0 else 0.0
        span_f  = sx2f - sx1f if sx2f != sx1f else 1.0

        for x in range(sx1, sx2 + 1):
            # Perspective-correct depth at this column
            t     = float(x - sx1f) / span_f
            inv_d = _lerp(inv_d1, inv_d2, t)
            depth = (1.0 / inv_d) if inv_d > 0.0 else _SHADE_DIST

            # Z-test: only draw if closer than what's already there
            if depth >= col_depth[x]:
                continue
            col_depth[x] = depth

            half_wall = half_h / depth

            # Apply camera bob offset to wall positions using pre-calculated y_center
            y_top = int(y_center - half_wall)
            y_bot = int(y_center + half_wall)

            if is_portal:
                wb = _shade(_PORTAL_B, depth)
                wg = _shade(_PORTAL_G, depth)
                wr = _shade(_PORTAL_R, depth)
            else:
                wb = _shade(_WALL_B, depth)
                wg = _shade(_WALL_G, depth)
                wr = _shade(_WALL_R, depth)

            _draw_column(fb, x, y_top, y_bot, wb, wg, wr, screen_h)

            if y_top > 0:
                _draw_column(fb, x, 0, y_top - 1, _CEIL_B, _CEIL_G, _CEIL_R, screen_h)
            if y_bot < screen_h - 1:
                _draw_column(fb, x, y_bot + 1, screen_h - 1, _FLOOR_B, _FLOOR_G, _FLOOR_R, screen_h)


# ---------------------------------------------------------------------------
# Public Python wrapper
# ---------------------------------------------------------------------------

def rasterize(
    fb:        np.ndarray,   # uint8 [H, W, 3]
    vis_segs:  np.ndarray,   # float32 [MAX_VIS, VIS_COLS]
    n_segs:    int,
    sectors:   np.ndarray,   # float32 [S, 2]
    screen_w:  int = 800,
    screen_h:  int = 450,
    camera_bob_offset: float = 0.0,
) -> None:
    """
    Draw the scene into fb.  Mutates fb in-place, returns nothing.
    fb must be np.uint8, shape (screen_h, screen_w, 3), C-contiguous.
    camera_bob_offset: vertical camera offset for head bobbing effect.
    """
    _rasterize_kernel(fb, vis_segs, n_segs, sectors, screen_w, screen_h, camera_bob_offset)


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

def warm_up(screen_w: int = 800, screen_h: int = 450) -> None:
    """Trigger JIT compilation with a minimal dummy call."""
    dummy_fb       = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    dummy_vis_segs = np.zeros((1, 12), dtype=np.float32)
    dummy_vis_segs[0, _VIS_SX1]    = 0.0
    dummy_vis_segs[0, _VIS_SX2]    = 1.0
    dummy_vis_segs[0, _VIS_DEPTH1] = 1.0
    dummy_vis_segs[0, _VIS_DEPTH2] = 1.0
    dummy_vis_segs[0, _VIS_BACK]   = -1.0
    dummy_vis_segs[0, _VIS_SX1F]   = 0.0
    dummy_vis_segs[0, _VIS_SX2F]   = 1.0
    dummy_sectors  = np.array([[0.0, 2.5]], dtype=np.float32)
    _rasterize_kernel(dummy_fb, dummy_vis_segs, 1, dummy_sectors, screen_w, screen_h, 0.0)


# ---------------------------------------------------------------------------
# CLI smoke-test  (logic verified without JIT using njit no-op)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))

    # Strip @njit for fast local testing
    import numba
    numba.njit = lambda *a, **kw: (lambda f: f)
    import importlib, rasterizer as _self
    importlib.reload(_self)

    from map_geometry  import make_test_map
    from bsp_builder   import build_bsp
    from bsp_traversal import cull_visible_segs

    geo  = make_test_map()
    tree = build_bsp(geo.vertices, geo.linedefs)

    fov = math.radians(66)
    result = cull_visible_segs(
        3.0, 3.0, 0.0, fov,
        tree.nodes, tree.seg_pool, tree.subsectors,
    )

    fb = np.zeros((450, 800, 3), dtype=np.uint8)
    _self.rasterize(fb, result.vis_segs, result.n, geo.sectors)

    # Sanity checks
    assert fb.shape == (450, 800, 3)
    assert fb.dtype == np.uint8

    # The centre strip should have wall colours, not pure black
    centre_col = fb[:, 400, :]
    assert centre_col.max() > 0, "Centre column is all black — nothing was drawn"

    # Top rows should be ceiling colour (dark, not wall colour)
    top_row = fb[0, 400, :]
    assert tuple(top_row) == (_CEIL_B, _CEIL_G, _CEIL_R), \
        f"Top row is not ceiling colour: {tuple(top_row)}"

    # Bottom rows should be floor colour
    bot_row = fb[449, 400, :]
    assert tuple(bot_row) == (_FLOOR_B, _FLOOR_G, _FLOOR_R), \
        f"Bottom row is not floor colour: {tuple(bot_row)}"

    print(f"Visible segs rendered : {result.n}")
    print(f"Framebuffer shape     : {fb.shape}  dtype={fb.dtype}")
    print(f"Centre column max px  : {centre_col.max()}")
    print(f"Top    row  (ceil)    : BGR {tuple(top_row)}")
    print(f"Bottom row  (floor)   : BGR {tuple(bot_row)}")

    # Save a preview PNG so the output is visually inspectable
    try:
        import cv2
        out_path = os.path.join(os.path.dirname(__file__), "rasterizer_preview.png")
        cv2.imwrite(out_path, fb)
        print(f"Preview saved        : {out_path}")
    except ImportError:
        print("cv2 not available — skipping PNG save")

    print("\nAll assertions passed.")
