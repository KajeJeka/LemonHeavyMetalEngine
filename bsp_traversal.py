"""
bsp_traversal.py  –  Lemon Heavy Metal Engine
Numba-accelerated BSP traversal with 1-D occlusion culling.

All hot-path functions are decorated @njit(cache=True).  They operate
exclusively on flat NumPy arrays and primitive scalars — no Python objects
enter any compiled function.

Public API
----------
  cull_visible_segs(px, py, angle, fov_rad, tree, screen_w=800)
      -> VisibleSet

  VisibleSet.vis_segs   float32 [n, VIS_COLS]  – projected visible segs
  VisibleSet.n          int                    – number of valid rows
  VisibleSet.occlusion  bool    [screen_w]     – final occlusion buffer

Visible seg column layout  (VIS_COLS = 10)
------------------------------------------
  0  sx1          left  screen column (float; cast to int by renderer)
  1  sx2          right screen column
  2  wx1          world x of start vertex
  3  wy1          world y of start vertex
  4  wx2          world x of end vertex
  5  wy2          world y of end vertex
  6  front_sector
  7  back_sector  (-1.0 == solid wall)
  8  depth1       camera-space depth of start vertex (used for perspective)
  9  depth2       camera-space depth of end vertex

BSP node / seg array layout is defined in bsp_builder.py.
Constants are duplicated here as plain int literals so @njit functions
never import from another module at compile time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Layout constants (mirrors bsp_builder.py — must stay in sync)
# ---------------------------------------------------------------------------

# Node columns
_N_PART_X  = 0
_N_PART_Y  = 1
_N_PART_DX = 2
_N_PART_DY = 3
_N_RIGHT   = 12
_N_LEFT    = 13

# Seg columns
_S_X1      = 0
_S_Y1      = 1
_S_X2      = 2
_S_Y2      = 3
_S_FRONT   = 4
_S_BACK    = 5

# Visible-seg output columns
VIS_SX1    = 0
VIS_SX2    = 1
VIS_WX1    = 2
VIS_WY1    = 3
VIS_WX2    = 4
VIS_WY2    = 5
VIS_FRONT  = 6
VIS_BACK   = 7
VIS_DEPTH1 = 8
VIS_DEPTH2 = 9
VIS_SX1F   = 10   # unclamped float screen x of left endpoint  (for correct depth interp)
VIS_SX2F   = 11   # unclamped float screen x of right endpoint
VIS_COLS   = 12

# Traversal limits
STACK_DEPTH  = 512    # max BSP tree depth (sufficient for any map <512 nodes)
MAX_VISIBLE  = 1024   # max segs returned per frame
NEAR_PLANE   = 0.1   # camera-space depth below which a vertex is behind player
                     # 0.001 causes depth contamination: inv_d_near=1000 bleeds into
                     # neighbouring columns even at t≈1, pulling depths toward 0 and
                     # producing a bright curved gradient on walls near the clip boundary.
                     # 0.1 keeps inv_d_near=10 — error < 0.003 units at t=0.999.


# ---------------------------------------------------------------------------
# @njit helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _cross2d(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


@njit(cache=True)
def _to_camera(
    wx: float, wy: float,
    px: float, py: float,
    cos_a: float, sin_a: float,
) -> tuple:
    """
    Transform world point (wx, wy) into camera space relative to player
    (px, py) facing angle whose cos/sin are pre-computed.

    Camera convention
    -----------------
      cam_x  =  forward depth  (+x is in front of player)
      cam_y  =  lateral offset (+y is to the right of player)

    Forward vector: (cos_a, sin_a)
    Right   vector: (sin_a, -cos_a)
    """
    dx = wx - px
    dy = wy - py
    cam_x =  dx * cos_a + dy * sin_a
    cam_y =  dx * sin_a - dy * cos_a
    return cam_x, cam_y


@njit(cache=True)
def _project_x(cam_x: float, cam_y: float, half_w: float, focal: float) -> float:
    """
    Project a camera-space point to a screen X column.
    focal = (screen_w / 2) / tan(fov / 2)
    Returns a float; the renderer rounds to int.
    """
    return half_w - (cam_y / cam_x) * focal


@njit(cache=True)
def _clip_to_near(
    cx1: float, cy1: float,
    cx2: float, cy2: float,
) -> tuple:
    """
    Clip a camera-space segment so both endpoints have cx >= NEAR_PLANE.
    Returns (cx1, cy1, cx2, cy2, valid).
    valid=0 means the whole segment is behind the camera.
    """
    near = NEAR_PLANE

    if cx1 < near and cx2 < near:
        return cx1, cy1, cx2, cy2, int(0)   # entirely behind

    if cx1 < near:
        # Interpolate: find t where cx == near
        t = (near - cx1) / (cx2 - cx1)
        cx1 = near
        cy1 = cy1 + t * (cy2 - cy1)

    if cx2 < near:
        t = (near - cx2) / (cx1 - cx2)
        cx2 = near
        cy2 = cy2 + t * (cy1 - cy2)

    return cx1, cy1, cx2, cy2, int(1)


@njit(cache=True)
def _occlusion_full(occlusion: np.ndarray, screen_w: int) -> bool:
    """Return True when every column in the occlusion buffer is set."""
    for i in range(screen_w):
        if not occlusion[i]:
            return False
    return True


@njit(cache=True)
def _mark_occlusion(
    occlusion: np.ndarray,
    sx1: int, sx2: int,
    screen_w: int,
    filled: int,
) -> int:
    """
    Mark columns [sx1, sx2] occupied.  Returns updated filled count.
    sx1 and sx2 are already clamped to [0, screen_w-1].
    """
    for x in range(sx1, sx2 + 1):
        if not occlusion[x]:
            occlusion[x] = True
            filled += 1
    return filled


@njit(cache=True)
def _range_fully_occluded(occlusion: np.ndarray, sx1: int, sx2: int) -> bool:
    """Return True when every column in [sx1, sx2] is already filled."""
    for x in range(sx1, sx2 + 1):
        if not occlusion[x]:
            return False
    return True


# ---------------------------------------------------------------------------
# Core @njit traversal
# ---------------------------------------------------------------------------

@njit(cache=True)
def _traverse(
    px:         float,
    py:         float,
    cos_a:      float,
    sin_a:      float,
    half_fov:   float,        # radians
    half_w:     float,        # screen_w / 2
    focal:      float,        # (screen_w/2) / tan(half_fov)
    screen_w:   int,
    screen_h:   int,          # needed to compute wall height for occlusion decision
    nodes:      np.ndarray,   # float32 [N, 14]
    seg_pool:   np.ndarray,   # float32 [M, 6]
    subsectors: np.ndarray,   # int32   [S, 2]
    vis_segs:   np.ndarray,   # float32 [MAX_VISIBLE, VIS_COLS]  — output
    occlusion:  np.ndarray,   # bool    [screen_w]               — output
) -> int:
    """
    Iterative front-to-back BSP traversal with 1-D occlusion culling.
    Returns the number of visible segs written into vis_segs.
    """
    n_nodes = nodes.shape[0]

    # Degenerate tree: single subsector, no nodes
    if n_nodes == 0:
        if subsectors.shape[0] == 0:
            return 0
        single_leaf = ~int(0)   # encodes subsector 0
        stack = np.empty(STACK_DEPTH, dtype=np.int32)
        stack[0] = single_leaf
        sp = int(1)
    else:
        stack = np.empty(STACK_DEPTH, dtype=np.int32)
        stack[0] = int(0)   # root node index
        sp = int(1)

    vis_count = int(0)
    filled    = int(0)

    while sp > 0:
        # Early exit: screen completely covered
        if filled >= screen_w:
            break

        sp -= 1
        entry = stack[sp]

        # ---- Leaf -----------------------------------------------------------
        if entry < 0:
            ss_idx  = ~entry                         # decode subsector index
            ss_off  = subsectors[ss_idx, 0]          # offset into seg_pool
            ss_cnt  = subsectors[ss_idx, 1]          # number of segs

            for si in range(ss_cnt):
                seg = seg_pool[ss_off + si]

                # Transform both endpoints to camera space
                cx1, cy1 = _to_camera(seg[_S_X1], seg[_S_Y1], px, py, cos_a, sin_a)
                cx2, cy2 = _to_camera(seg[_S_X2], seg[_S_Y2], px, py, cos_a, sin_a)

                # Near-plane clip
                cx1, cy1, cx2, cy2, valid = _clip_to_near(cx1, cy1, cx2, cy2)
                if not valid:
                    continue

                # Project to screen X
                sx1f = _project_x(cx1, cy1, half_w, focal)
                sx2f = _project_x(cx2, cy2, half_w, focal)

                # Ensure sx1 <= sx2
                if sx1f > sx2f:
                    sx1f, sx2f = sx2f, sx1f
                    cx1, cy1, cx2, cy2 = cx2, cy2, cx1, cy1

                # Integer screen columns, clamped
                sx1 = int(max(0.0, min(sx1f, float(screen_w - 1))))
                sx2 = int(max(0.0, min(sx2f, float(screen_w - 1))))

                if sx1 > sx2:
                    continue

                # Occlusion check: skip if already fully covered
                if _range_fully_occluded(occlusion, sx1, sx2):
                    continue

                # Record visible seg
                if vis_count < MAX_VISIBLE:
                    vis_segs[vis_count, VIS_SX1]    = float(sx1)
                    vis_segs[vis_count, VIS_SX2]    = float(sx2)
                    vis_segs[vis_count, VIS_WX1]    = seg[_S_X1]
                    vis_segs[vis_count, VIS_WY1]    = seg[_S_Y1]
                    vis_segs[vis_count, VIS_WX2]    = seg[_S_X2]
                    vis_segs[vis_count, VIS_WY2]    = seg[_S_Y2]
                    vis_segs[vis_count, VIS_FRONT]  = seg[_S_FRONT]
                    vis_segs[vis_count, VIS_BACK]   = seg[_S_BACK]
                    vis_segs[vis_count, VIS_DEPTH1] = cx1
                    vis_segs[vis_count, VIS_DEPTH2] = cx2
                    vis_segs[vis_count, VIS_SX1F]   = sx1f
                    vis_segs[vis_count, VIS_SX2F]   = sx2f
                    vis_count += 1

                # Only mark a column as fully occluded when the wall at that
                # column actually reaches floor-to-ceiling (half_wall >= half_h).
                # A short/distant wall that only covers the middle band of the
                # screen must NOT block columns for segs deeper in the scene —
                # e.g. a corridor wall further ahead is still visible above/below.
                half_h = float(screen_h) * 0.5
                span_f = sx2f - sx1f if sx2f != sx1f else 1.0
                inv_d1 = 1.0 / cx1 if cx1 > 0.0 else 0.0
                inv_d2 = 1.0 / cx2 if cx2 > 0.0 else 0.0
                for xc in range(sx1, sx2 + 1):
                    if not occlusion[xc]:
                        t      = float(xc - sx1f) / span_f
                        inv_d  = inv_d1 + t * (inv_d2 - inv_d1)
                        depth  = (1.0 / inv_d) if inv_d > 0.0 else 0.0
                        # Wall half-height in pixels at this column
                        hw     = half_h / depth if depth > 0.0 else half_h + 1.0
                        if hw >= half_h:          # floor-to-ceiling → fully occluded
                            occlusion[xc] = True
                            filled += 1
                if filled >= screen_w:
                    break

            continue

        # ---- Internal node --------------------------------------------------
        node_idx = entry

        px_n  = nodes[node_idx, _N_PART_X]
        py_n  = nodes[node_idx, _N_PART_Y]
        dx_n  = nodes[node_idx, _N_PART_DX]
        dy_n  = nodes[node_idx, _N_PART_DY]

        right = int(nodes[node_idx, _N_RIGHT])
        left  = int(nodes[node_idx, _N_LEFT])

        # Which side of the splitter is the player on?
        side = _cross2d(dx_n, dy_n, px - px_n, py - py_n)

        # Front-to-back: push the back child first so front is popped first.
        if side >= 0.0:
            if sp < STACK_DEPTH:
                stack[sp] = left    # back  – processed second
                sp += 1
            if sp < STACK_DEPTH:
                stack[sp] = right   # front – processed first
                sp += 1
        else:
            if sp < STACK_DEPTH:
                stack[sp] = right   # back  – processed second
                sp += 1
            if sp < STACK_DEPTH:
                stack[sp] = left    # front – processed first
                sp += 1

    return vis_count


# ---------------------------------------------------------------------------
# Python-level output container and public wrapper
# ---------------------------------------------------------------------------

@dataclass
class VisibleSet:
    """Result of one traversal call."""
    vis_segs:  np.ndarray   # float32 [n, VIS_COLS]  — only rows[:n] are valid
    n:         int          # number of visible segs written
    occlusion: np.ndarray   # bool [screen_w]


def cull_visible_segs(
    px:        float,
    py:        float,
    angle:     float,
    fov_rad:   float,
    nodes:     np.ndarray,
    seg_pool:  np.ndarray,
    subsectors: np.ndarray,
    screen_w:  int = 800,
    screen_h:  int = 450,
) -> VisibleSet:
    """
    Public entry point.  Allocates output arrays, calls the @njit kernel,
    returns a VisibleSet.

    Call this once per frame.  The @njit function is already compiled after
    the first call (cache=True), so subsequent calls have no JIT overhead.
    """
    cos_a    = math.cos(angle)
    sin_a    = math.sin(angle)
    half_fov = fov_rad * 0.5
    half_w   = screen_w * 0.5
    focal    = half_w / math.tan(half_fov)

    vis_segs  = np.empty((MAX_VISIBLE, VIS_COLS), dtype=np.float32)
    occlusion = np.zeros(screen_w, dtype=np.bool_)

    n = _traverse(
        px, py,
        cos_a, sin_a,
        half_fov, half_w, focal,
        screen_w, screen_h,
        nodes, seg_pool, subsectors,
        vis_segs, occlusion,
    )

    return VisibleSet(vis_segs=vis_segs, n=int(n), occlusion=occlusion)


# ---------------------------------------------------------------------------
# Warm-up helper  (call once at startup to trigger JIT compilation)
# ---------------------------------------------------------------------------

def warm_up() -> None:
    """
    Force Numba to compile all @njit functions with a minimal dummy call.
    dummy_nodes must be EMPTY (shape (0, 14)) so the traversal takes the
    degenerate n_nodes==0 path and exits immediately.
    """
    dummy_nodes      = np.empty((0, 14), dtype=np.float32)
    dummy_seg_pool   = np.zeros((1,  6), dtype=np.float32)
    dummy_subsectors = np.zeros((1,  2), dtype=np.int32)
    dummy_subsectors[0, 0] = 0
    dummy_subsectors[0, 1] = 1

    cull_visible_segs(
        0.0, 0.0, 0.0, math.radians(66),
        dummy_nodes, dummy_seg_pool, dummy_subsectors,
        screen_w=800, screen_h=450,
    )


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os, time
    sys.path.insert(0, os.path.dirname(__file__))
    from map_geometry import make_test_map
    from bsp_builder  import build_bsp

    geo  = make_test_map()
    tree = build_bsp(geo.vertices, geo.linedefs)

    print("Warming up Numba JIT...")
    t0 = time.perf_counter()
    warm_up()
    print(f"  JIT warm-up: {(time.perf_counter()-t0)*1000:.1f} ms")

    # Player standing in Room A, facing right (+X direction)
    px, py  = 3.0, 3.0
    angle   = 0.0                 # facing +X
    fov     = math.radians(66.0)

    print("\nRunning traversal (Room A, facing +X)...")
    t0 = time.perf_counter()
    for _ in range(1000):
        result = cull_visible_segs(
            px, py, angle, fov,
            tree.nodes, tree.seg_pool, tree.subsectors,
        )
    elapsed = (time.perf_counter() - t0) / 1000 * 1_000_000
    print(f"  {result.n} visible segs   |   {elapsed:.2f} µs/call (avg over 1000 runs)")

    print(f"  Occlusion filled: {result.occlusion.sum()} / 800 columns")
    print()
    print("  Visible segs (sx1, sx2, front_sector, back_sector, depth1, depth2):")
    for i in range(result.n):
        s = result.vis_segs[i]
        print(f"    [{i:02d}]  screen [{int(s[VIS_SX1]):4d},{int(s[VIS_SX2]):4d}]"
              f"  sector {int(s[VIS_FRONT]):2d}/{int(s[VIS_BACK]):2d}"
              f"  depth {s[VIS_DEPTH1]:.3f}–{s[VIS_DEPTH2]:.3f}")

    # Second position: inside Room A facing the portal toward Room B
    print("\nRunning traversal (Room A, facing portal)...")
    result2 = cull_visible_segs(
        3.0, 3.0, 0.0, fov,
        tree.nodes, tree.seg_pool, tree.subsectors,
    )
    has_portal = any(
        int(result2.vis_segs[i, VIS_BACK]) == 1
        for i in range(result2.n)
    )
    print(f"  Portal seg visible (back_sector==1): {has_portal}")
    assert has_portal, "Portal should be visible from Room A facing right"

    print("\nAll assertions passed.")
