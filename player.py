"""
player.py  –  Lemon Heavy Metal Engine
Player state, input, and linedef-based sliding collision.

Physics is driven entirely by the BSP seg_pool.  The legacy tile-grid
is_blocked() check has been removed.  All collision geometry comes from the
flat NumPy arrays produced by bsp_builder.build_bsp().

Hot-path functions are @njit compiled.  They take only NumPy arrays and
primitive scalars — no Python objects cross the JIT boundary.

Collision model
---------------
  The player is a circle of radius PLAYER_RADIUS in world space.
  Each frame the desired move vector (vx, vy) is computed from input.
  _collect_nearby_segs() walks the BSP to find seg_pool rows whose geometry
  is within SEARCH_RADIUS of the proposed new position.
  _slide_resolve() iterates up to COLLISION_ITERS times, pushing the player
  out of any penetrated solid linedef and projecting the velocity to slide
  along the wall surface.
  Only solid segs (back_sector == -1.0) block movement; portals are skipped.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

MOVE_SPEED      = 3.5
ROT_SPEED       = 2.0
MOUSE_SENS      = 0.0025
SPRINT_MULT     = 2.0

PLAYER_RADIUS   = 0.25        # world units
SEARCH_MULT     = 5.0         # collect segs within PLAYER_RADIUS * SEARCH_MULT
COLLISION_ITERS = 3           # penetration resolve iterations per frame
MAX_CANDIDATES  = 128         # hard cap on BSP-collected candidate segs
_STACK_DEPTH    = 256         # BSP traversal stack depth

BOB_FREQ_WALK   = 7.0
BOB_FREQ_SPRINT = 11.0
BOB_AMP_WALK    = 6.0
BOB_AMP_SPRINT  = 11.0
BOB_DECAY       = 12.0

# Seg column indices (mirrors bsp_builder.py — duplicated so @njit never
# imports from another module at compile time)
_S_X1   = 0
_S_Y1   = 1
_S_X2   = 2
_S_Y2   = 3
_S_BACK = 5   # -1.0 == solid wall sentinel

# Node column indices
_N_BBOX_F_XMIN = 4
_N_BBOX_F_XMAX = 5
_N_BBOX_F_YMIN = 6
_N_BBOX_F_YMAX = 7
_N_BBOX_B_XMIN = 8
_N_BBOX_B_XMAX = 9
_N_BBOX_B_YMIN = 10
_N_BBOX_B_YMAX = 11
_N_RIGHT       = 12
_N_LEFT        = 13


# ---------------------------------------------------------------------------
# @njit geometry primitives
# ---------------------------------------------------------------------------

@njit(cache=True)
def _closest_point_on_seg(
    px: float, py: float,
    x1: float, y1: float,
    x2: float, y2: float,
) -> tuple:
    """Return (cx, cy) — the closest point on segment [x1,y1]-[x2,y2] to (px,py)."""
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        return x1, y1
    t = ((px - x1) * dx + (py - y1) * dy) / len_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return x1 + t * dx, y1 + t * dy


@njit(cache=True)
def _circle_overlaps_aabb(
    px: float, py: float, r: float,
    xmin: float, xmax: float,
    ymin: float, ymax: float,
) -> bool:
    """True when circle (px,py,r) intersects the axis-aligned bounding box."""
    cx = px if px > xmin else xmin
    cx = cx if cx < xmax else xmax
    cy = py if py > ymin else ymin
    cy = cy if cy < ymax else ymax
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy <= r * r


# ---------------------------------------------------------------------------
# BSP-accelerated candidate collector
# ---------------------------------------------------------------------------

@njit(cache=True)
def _collect_nearby_segs(
    px:         float,
    py:         float,
    radius:     float,
    nodes:      np.ndarray,   # float32 [N, 14]
    seg_pool:   np.ndarray,   # float32 [M,  6]
    subsectors: np.ndarray,   # int32   [S,  2]
    out:        np.ndarray,   # int32   [MAX_CANDIDATES]
    max_n:      int,
) -> int:
    """
    Iterative BSP walk.  Collect seg_pool row indices for every seg whose
    parent subsector bounding box overlaps the circle (px, py, radius).
    Both children of each node are tested independently so segs on either
    side of a partition plane are never missed.
    Returns the number of indices written into out[].
    """
    n_out  = int(0)
    n_nodes = nodes.shape[0]

    if n_nodes == 0:
        if subsectors.shape[0] > 0:
            ss_off = subsectors[0, 0]
            ss_cnt = subsectors[0, 1]
            for i in range(ss_cnt):
                if n_out < max_n:
                    out[n_out] = ss_off + i
                    n_out += 1
        return n_out

    stack = np.empty(_STACK_DEPTH, dtype=np.int32)
    stack[0] = int(0)
    sp = int(1)

    while sp > 0 and n_out < max_n:
        sp -= 1
        entry = int(stack[sp])

        if entry < 0:
            ss_idx = ~entry
            ss_off = subsectors[ss_idx, 0]
            ss_cnt = subsectors[ss_idx, 1]
            for i in range(ss_cnt):
                if n_out < max_n:
                    out[n_out] = ss_off + i
                    n_out += 1
            continue

        ni = entry

        if _circle_overlaps_aabb(px, py, radius,
                                  nodes[ni, _N_BBOX_F_XMIN], nodes[ni, _N_BBOX_F_XMAX],
                                  nodes[ni, _N_BBOX_F_YMIN], nodes[ni, _N_BBOX_F_YMAX]):
            rc = int(nodes[ni, _N_RIGHT])
            if sp < _STACK_DEPTH:
                stack[sp] = rc
                sp += 1

        if _circle_overlaps_aabb(px, py, radius,
                                  nodes[ni, _N_BBOX_B_XMIN], nodes[ni, _N_BBOX_B_XMAX],
                                  nodes[ni, _N_BBOX_B_YMIN], nodes[ni, _N_BBOX_B_YMAX]):
            lc = int(nodes[ni, _N_LEFT])
            if sp < _STACK_DEPTH:
                stack[sp] = lc
                sp += 1

    return n_out


# ---------------------------------------------------------------------------
# Sliding collision resolver
# ---------------------------------------------------------------------------

@njit(cache=True)
def _slide_resolve(
    px:         float,
    py:         float,
    vx:         float,
    vy:         float,
    radius:     float,
    seg_pool:   np.ndarray,   # float32 [M, 6]
    candidates: np.ndarray,   # int32   [K]
    n_cands:    int,
) -> tuple:
    """
    Resolve penetrations against solid linedefs and slide velocity.
    Returns (new_px, new_py, collided: bool).

    Per pass:
      For each solid seg in candidates:
        1. Find closest point on seg to proposed position.
        2. If penetrating (dist < radius): push position out along the
           contact normal, then remove the into-wall velocity component
           so subsequent movement slides along the surface.
    """
    npx = px + vx
    npy = py + vy
    collided = False

    for _iter in range(COLLISION_ITERS):
        for i in range(n_cands):
            seg = seg_pool[candidates[i]]

            if seg[_S_BACK] >= 0.0:   # portal — walkable
                continue

            cx, cy = _closest_point_on_seg(
                npx, npy,
                seg[_S_X1], seg[_S_Y1],
                seg[_S_X2], seg[_S_Y2],
            )

            pen_x   = npx - cx
            pen_y   = npy - cy
            dist_sq = pen_x * pen_x + pen_y * pen_y

            if dist_sq >= radius * radius or dist_sq < 1e-10:
                continue

            dist = dist_sq ** 0.5
            nx   = pen_x / dist   # outward contact normal (wall → player)
            ny   = pen_y / dist

            # Push position clear of wall
            push = radius - dist
            npx += nx * push
            npy += ny * push

            # Project velocity to slide along wall (cancel into-wall component)
            dot = vx * nx + vy * ny
            if dot < 0.0:
                vx -= dot * nx
                vy -= dot * ny

            collided = True

    return npx, npy, collided


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

def collision_warm_up(
    nodes:      np.ndarray,
    seg_pool:   np.ndarray,
    subsectors: np.ndarray,
) -> None:
    """Pre-compile collision JIT kernels.  Call once at engine start-up."""
    dummy_out = np.empty(MAX_CANDIDATES, dtype=np.int32)
    n = _collect_nearby_segs(0.0, 0.0, 1.0, nodes, seg_pool, subsectors,
                              dummy_out, MAX_CANDIDATES)
    dummy_c = np.zeros(max(n, 1), dtype=np.int32)
    _slide_resolve(0.0, 0.0, 0.0, 0.0, PLAYER_RADIUS, seg_pool, dummy_c, 0)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

class Player:
    def __init__(self) -> None:
        self.x:          float = 1.5
        self.y:          float = 1.5
        self.angle:      float = 0.4
        self._bob_time:  float = 0.0
        self._bob_vel:   float = 0.0
        self.bob_offset: float = 0.0
        # Pre-allocated — never reallocated during the game loop
        self._candidate_buf: np.ndarray = np.empty(MAX_CANDIDATES, dtype=np.int32)

    @property
    def dir(self) -> tuple[float, float]:
        return math.cos(self.angle), math.sin(self.angle)

    def reset(self, x: float = 1.5, y: float = 1.5, angle: float = 0.4) -> None:
        self.x          = x
        self.y          = y
        self.angle      = angle
        self._bob_time  = 0.0
        self._bob_vel   = 0.0
        self.bob_offset = 0.0

    def update(
        self,
        keys:       set,
        mouse_dx:   float,
        dt:         float,
        nodes:      np.ndarray,   # BSPTree.nodes      float32 [N,14]
        seg_pool:   np.ndarray,   # BSPTree.seg_pool   float32 [M, 6]
        subsectors: np.ndarray,   # BSPTree.subsectors int32   [S, 2]
    ) -> bool:
        """
        Process input, resolve movement against BSP linedefs, update bob.
        Returns True if any wall collision was resolved this frame.
        """
        d_angle = mouse_dx * MOUSE_SENS
        if "LEFT"  in keys: d_angle -= ROT_SPEED * dt
        if "RIGHT" in keys: d_angle += ROT_SPEED * dt
        self.angle += d_angle

        sprinting = "SPRINT" in keys
        ms  = MOVE_SPEED * dt * (SPRINT_MULT if sprinting else 1.0)
        cos = math.cos(self.angle)
        sin = math.sin(self.angle)

        vx, vy  = 0.0, 0.0
        moving  = False
        if "FORWARD" in keys: vx += cos * ms;  vy += sin * ms;  moving = True
        if "BACK"    in keys: vx -= cos * ms;  vy -= sin * ms;  moving = True
        if "SLEFT"   in keys: vx += sin * ms;  vy -= cos * ms;  moving = True
        if "SRIGHT"  in keys: vx -= sin * ms;  vy += cos * ms;  moving = True

        collided = False
        if vx != 0.0 or vy != 0.0:
            npx      = self.x + vx
            npy      = self.y + vy
            search_r = PLAYER_RADIUS * SEARCH_MULT

            n_cands = _collect_nearby_segs(
                npx, npy, search_r,
                nodes, seg_pool, subsectors,
                self._candidate_buf, MAX_CANDIDATES,
            )

            new_x, new_y, collided = _slide_resolve(
                self.x, self.y, vx, vy, PLAYER_RADIUS,
                seg_pool, self._candidate_buf, n_cands,
            )

            self.x = new_x
            self.y = new_y

        self._update_bob(moving, sprinting, dt)
        return collided

    def _update_bob(self, moving: bool, sprinting: bool, dt: float) -> None:
        target = 1.0 if moving else 0.0
        self._bob_vel += (target - self._bob_vel) * min(1.0, BOB_DECAY * dt)
        if self._bob_vel > 0.001:
            freq = BOB_FREQ_SPRINT if sprinting else BOB_FREQ_WALK
            amp  = BOB_AMP_SPRINT  if sprinting else BOB_AMP_WALK
            self._bob_time  += dt * freq
            self.bob_offset  = math.sin(self._bob_time) * amp * self._bob_vel
        else:
            self._bob_time  += dt * BOB_FREQ_WALK
            self.bob_offset  = math.sin(self._bob_time) * BOB_AMP_WALK * self._bob_vel
