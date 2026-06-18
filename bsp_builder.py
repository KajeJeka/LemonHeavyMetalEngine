"""
bsp_builder.py  –  Lemon Heavy Metal Engine
Offline BSP compiler.  Runs once before the game loop; no Numba required.

Input
-----
  vertices : float32 [N, 2]
  linedefs : int32   [M, 4]   columns: (v_start, v_end, front_sector, back_sector)

Output  –  BSPTree dataclass
-----
  nodes      : float32 [num_nodes,    NODE_COLS]
  seg_pool   : float32 [num_segs,     SEG_COLS]
  subsectors : int32   [num_leaves,   2]          (seg_pool offset, seg count)

NODE_COLS layout  (indices are named constants below)
------------------------------------------------------
  0  part_x       splitter start x
  1  part_y       splitter start y
  2  part_dx      splitter direction x
  3  part_dy      splitter direction y
  4  bbox_front_x_min
  5  bbox_front_x_max
  6  bbox_front_y_min
  7  bbox_front_y_max
  8  bbox_back_x_min
  9  bbox_back_x_max
  10 bbox_back_y_min
  11 bbox_back_y_max
  12 right_child   positive = node index; negative = ~subsector index (leaf sentinel)
  13 left_child    same encoding

SEG_COLS layout
---------------
  0  x1, 1 y1       start vertex (world coords, float32)
  2  x2, 3 y2       end vertex
  4  front_sector   (float32 reinterpret of int, cast back when reading)
  5  back_sector    -1.0 == solid wall sentinel

Leaf encoding
-------------
  DOOM convention: child < 0  →  leaf.  Actual subsector index = ~child.
  child >= 0                  →  internal node index.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# Node columns
N_PART_X       = 0
N_PART_Y       = 1
N_PART_DX      = 2
N_PART_DY      = 3
N_BBOX_F_XMIN  = 4
N_BBOX_F_XMAX  = 5
N_BBOX_F_YMIN  = 6
N_BBOX_F_YMAX  = 7
N_BBOX_B_XMIN  = 8
N_BBOX_B_XMAX  = 9
N_BBOX_B_YMIN  = 10
N_BBOX_B_YMAX  = 11
N_RIGHT        = 12
N_LEFT         = 13
NODE_COLS      = 14

# Seg columns
S_X1           = 0
S_Y1           = 1
S_X2           = 2
S_Y2           = 3
S_FRONT        = 4
S_BACK         = 5
SEG_COLS       = 6

EPSILON        = 1e-5   # collinearity / zero-length guard


# ---------------------------------------------------------------------------
# Internal working representation during compilation
# ---------------------------------------------------------------------------

@dataclass
class _Seg:
    """Mutable segment used only during BSP construction."""
    x1: float
    y1: float
    x2: float
    y2: float
    front_sector: int
    back_sector:  int   # -1 == solid


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class BSPTree:
    nodes:      np.ndarray   # float32 [num_nodes,  NODE_COLS]
    seg_pool:   np.ndarray   # float32 [num_segs,   SEG_COLS]
    subsectors: np.ndarray   # int32   [num_leaves, 2]

    @property
    def node_count(self) -> int:
        return self.nodes.shape[0]

    @property
    def subsector_count(self) -> int:
        return self.subsectors.shape[0]

    @property
    def seg_count(self) -> int:
        return self.seg_pool.shape[0]

    def __repr__(self) -> str:
        return (
            f"BSPTree(nodes={self.node_count}, "
            f"subsectors={self.subsector_count}, "
            f"segs={self.seg_count})"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_bsp(vertices: np.ndarray, linedefs: np.ndarray) -> BSPTree:
    """
    Compile a BSP tree from MapGeometry arrays.

    Parameters
    ----------
    vertices  : float32 [N, 2]
    linedefs  : int32   [M, 4]   (v_start, v_end, front_sector, back_sector)

    Returns
    -------
    BSPTree with flat NumPy arrays ready for Numba traversal.
    """
    segs = _linedefs_to_segs(vertices, linedefs)

    # Accumulation lists — appended to during recursion, then frozen to arrays.
    node_rows:     List[List[float]]  = []
    subsector_segs: List[List[_Seg]] = []   # each entry = one leaf's seg list

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(8000)
    _build_node(segs, node_rows, subsector_segs)
    sys.setrecursionlimit(old_limit)

    return _pack(node_rows, subsector_segs)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _linedefs_to_segs(vertices: np.ndarray, linedefs: np.ndarray) -> List[_Seg]:
    segs: List[_Seg] = []
    for ld in linedefs:
        v_start, v_end, front, back = int(ld[0]), int(ld[1]), int(ld[2]), int(ld[3])
        x1, y1 = float(vertices[v_start, 0]), float(vertices[v_start, 1])
        x2, y2 = float(vertices[v_end,   0]), float(vertices[v_end,   1])
        segs.append(_Seg(x1, y1, x2, y2, front, back))
    return segs


def _cross_2d(ax: float, ay: float, bx: float, by: float) -> float:
    """2D cross product (scalar z of ax,ay × bx,by)."""
    return ax * by - ay * bx


def _classify_seg(seg: _Seg, px: float, py: float, dx: float, dy: float) -> float:
    """
    Signed distance of seg's midpoint from the splitter line.
    Positive  → front (right-hand side of directed line px+t*dx,py+t*dy).
    Negative  → back.
    ~0        → on the line.
    Uses the cross-product formulation for robustness.
    """
    mx = (seg.x1 + seg.x2) * 0.5 - px
    my = (seg.y1 + seg.y2) * 0.5 - py
    return _cross_2d(dx, dy, mx, my)


def _seg_side(seg: _Seg, px: float, py: float, dx: float, dy: float) -> Tuple[float, float]:
    """
    Return (d_start, d_end) — signed distances of each endpoint from the splitter.
    """
    d1 = _cross_2d(dx, dy, seg.x1 - px, seg.y1 - py)
    d2 = _cross_2d(dx, dy, seg.x2 - px, seg.y2 - py)
    return d1, d2


def _intersect(
    seg: _Seg,
    px: float, py: float,
    dx: float, dy: float,
) -> Tuple[float, float]:
    """
    Intersection of seg with the infinite splitter line.
    Returns (ix, iy).  Caller guarantees the lines are not parallel.
    t is clamped to [0, 1] so the result always lies on the seg itself,
    preventing phantom vertices outside the original geometry bounds.
    """
    sdx = seg.x2 - seg.x1
    sdy = seg.y2 - seg.y1
    denom = _cross_2d(dx, dy, sdx, sdy)
    if abs(denom) < EPSILON:
        return (seg.x1 + seg.x2) * 0.5, (seg.y1 + seg.y2) * 0.5
    t = _cross_2d(dx, dy, seg.x1 - px, seg.y1 - py) / denom
    t = max(0.0, min(1.0, t))   # clamp: result must lie on the seg
    ix = seg.x1 + t * sdx
    iy = seg.y1 + t * sdy
    return ix, iy


# ---------------------------------------------------------------------------
# Splitter selection
# ---------------------------------------------------------------------------

def _score_splitter(candidate: _Seg, segs: List[_Seg]) -> int:
    """
    Lower score = better.
    Scoring: splits cost 8 pts each, imbalance costs 1 pt per unit.
    An empty back or front set gets a large fixed penalty so the picker
    avoids degenerate splitters that leave everything on one side.
    """
    px, py = candidate.x1, candidate.y1
    dx = candidate.x2 - px
    dy = candidate.y2 - py
    if abs(dx) < EPSILON and abs(dy) < EPSILON:
        return 999999   # zero-length seg, never pick it

    front_count = back_count = split_count = 0
    for seg in segs:
        d1, d2 = _seg_side(seg, px, py, dx, dy)
        if abs(d1) <= EPSILON and abs(d2) <= EPSILON:
            front_count += 1   # collinear -> front, no split cost
        elif d1 > -EPSILON and d2 > -EPSILON:
            front_count += 1
        elif d1 < EPSILON and d2 < EPSILON:
            back_count += 1
        else:
            split_count += 1

    # Heavily penalise degenerate partitions
    if front_count == 0 or back_count == 0:
        return 999999

    return split_count * 8 + abs(front_count - back_count)


def _pick_splitter(segs: List[_Seg]) -> _Seg | None:
    """
    Evaluate every seg as a candidate splitter; return the one with the
    lowest score, or None if no valid split exists (all scores == 999999).
    """
    best_seg   = None
    best_score = 999999
    for candidate in segs:
        score = _score_splitter(candidate, segs)
        if score < best_score:
            best_score = score
            best_seg   = candidate
    return best_seg   # None when every candidate is degenerate


# ---------------------------------------------------------------------------
# Bounding box helper
# ---------------------------------------------------------------------------

def _bbox(segs: List[_Seg]) -> Tuple[float, float, float, float]:
    """Return (x_min, x_max, y_min, y_max) for a list of segs."""
    xs = [s.x1 for s in segs] + [s.x2 for s in segs]
    ys = [s.y1 for s in segs] + [s.y2 for s in segs]
    return min(xs), max(xs), min(ys), max(ys)


# ---------------------------------------------------------------------------
# Recursive BSP builder
# ---------------------------------------------------------------------------

def _build_node(
    segs:            List[_Seg],
    node_rows:       List[List[float]],
    subsector_segs:  List[List[_Seg]],
) -> int:
    """
    Recursively partition segs.  Returns the child index for the parent to store.
    Leaf  → returns ~subsector_index  (negative, DOOM convention).
    Node  → returns  node_index       (non-negative).
    """
    # ---- Base case: convex set → emit a subsector (leaf) ------------------
    if _is_convex(segs):
        idx = len(subsector_segs)
        subsector_segs.append(segs)
        return ~idx   # leaf sentinel: bitwise NOT makes it negative

    # ---- Pick splitter -----------------------------------------------------
    splitter = _pick_splitter(segs)
    if splitter is None:
        # No valid split possible — emit everything as one leaf
        idx = len(subsector_segs)
        subsector_segs.append(segs)
        return ~idx

    px, py   = splitter.x1, splitter.y1
    dx       = splitter.x2 - px
    dy       = splitter.y2 - py

    front_segs: List[_Seg] = []
    back_segs:  List[_Seg] = []

    for seg in segs:
        d1, d2 = _seg_side(seg, px, py, dx, dy)

        on1 = abs(d1) <= EPSILON
        on2 = abs(d2) <= EPSILON

        if on1 and on2:
            # Collinear with splitter: goes to front
            front_segs.append(seg)
        elif (d1 >= -EPSILON) and (d2 >= -EPSILON):
            # Both endpoints on front side (or on the line)
            front_segs.append(seg)
        elif (d1 <= EPSILON) and (d2 <= EPSILON):
            # Both endpoints on back side (or on the line)
            back_segs.append(seg)
        else:
            # Genuine straddle — split at intersection
            ix, iy = _intersect(seg, px, py, dx, dy)
            if d1 > 0:
                half_f = _Seg(seg.x1, seg.y1, ix, iy, seg.front_sector, seg.back_sector)
                half_b = _Seg(ix, iy, seg.x2, seg.y2, seg.front_sector, seg.back_sector)
            else:
                half_b = _Seg(seg.x1, seg.y1, ix, iy, seg.front_sector, seg.back_sector)
                half_f = _Seg(ix, iy, seg.x2, seg.y2, seg.front_sector, seg.back_sector)
            # Discard zero-length halves — these occur when clamped t=0 or t=1
            # produces an intersection exactly at an existing endpoint.
            front_len = (half_f.x2-half_f.x1)**2 + (half_f.y2-half_f.y1)**2
            back_len  = (half_b.x2-half_b.x1)**2 + (half_b.y2-half_b.y1)**2
            if front_len > EPSILON:
                front_segs.append(half_f)
            else:
                # Degenerate front half — the whole seg goes back
                back_segs.append(seg)
                continue
            if back_len > EPSILON:
                back_segs.append(half_b)
            # (if back half is zero-length, just drop it — seg is entirely front)

    # Guard against degenerate partitions (all segs on one side)
    if not front_segs:
        front_segs, back_segs = back_segs, []
    if not back_segs:
        # Partition produced nothing on the back side.
        # Recurse on front_segs so the tree keeps splitting rather than
        # creating a huge leaf with all segs.
        return _build_node(front_segs, node_rows, subsector_segs)

    # ---- Recurse -----------------------------------------------------------
    # Reserve a slot in node_rows BEFORE recursing so child indices are correct.
    node_idx = len(node_rows)
    node_rows.append([0.0] * NODE_COLS)   # placeholder

    right_child = _build_node(front_segs, node_rows, subsector_segs)
    left_child  = _build_node(back_segs,  node_rows, subsector_segs)

    # ---- Bounding boxes ----------------------------------------------------
    fx_min, fx_max, fy_min, fy_max = _bbox(front_segs)
    bx_min, bx_max, by_min, by_max = _bbox(back_segs)

    # ---- Write node row ----------------------------------------------------
    row = node_rows[node_idx]
    row[N_PART_X]      = px
    row[N_PART_Y]      = py
    row[N_PART_DX]     = dx
    row[N_PART_DY]     = dy
    row[N_BBOX_F_XMIN] = fx_min
    row[N_BBOX_F_XMAX] = fx_max
    row[N_BBOX_F_YMIN] = fy_min
    row[N_BBOX_F_YMAX] = fy_max
    row[N_BBOX_B_XMIN] = bx_min
    row[N_BBOX_B_XMAX] = bx_max
    row[N_BBOX_B_YMIN] = by_min
    row[N_BBOX_B_YMAX] = by_max
    row[N_RIGHT]       = float(right_child)
    row[N_LEFT]        = float(left_child)

    return node_idx


# ---------------------------------------------------------------------------
# Convexity test
# ---------------------------------------------------------------------------

def _is_convex(segs: List[_Seg]) -> bool:
    """
    A set of segs is convex (safe to emit as a leaf) when:
      1. All segs share the same front sector.
      2. The spatial bounding box of all seg endpoints is compact enough to
         belong to a single convex region.  For a grid-based maze where
         corridors are 1 world unit wide, any set of segs spanning more than
         ~2 world units in either axis cannot be a single convex subspace.
      3. Every seg's midpoint lies on the front side (or plane) of every
         other seg's directed line.

    The bbox check (criterion 2) is critical for single-sector maps (e.g. a
    maze where every wall has front_sector=0).  Without it, the cross-product
    test alone accepts spatially separated walls that happen to face the same
    direction, producing leaves that span the whole map.
    """
    if len(segs) <= 1:
        return True

    # Criterion 1: same front sector
    first_sector = segs[0].front_sector
    for seg in segs[1:]:
        if seg.front_sector != first_sector:
            return False

    # Criterion 2: spatial compactness
    # Collect all endpoint coordinates
    xs = [s.x1 for s in segs] + [s.x2 for s in segs]
    ys = [s.y1 for s in segs] + [s.y2 for s in segs]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    # Two parallel walls in the same corridor can be 2 units apart.
    # Use 3.0 as the threshold to allow slight BSP-split overshoot.
    if x_span > 2.0 or y_span > 2.0:
        return False

    # Criterion 3: mutual front-side test
    for splitter in segs:
        px = splitter.x1
        py = splitter.y1
        dx = splitter.x2 - px
        dy = splitter.y2 - py
        if abs(dx) < EPSILON and abs(dy) < EPSILON:
            continue
        for seg in segs:
            if seg is splitter:
                continue
            d = _classify_seg(seg, px, py, dx, dy)
            if d < -EPSILON:
                return False
    return True


# ---------------------------------------------------------------------------
# Pack into flat NumPy arrays
# ---------------------------------------------------------------------------

def _pack(
    node_rows:      List[List[float]],
    subsector_segs: List[List[_Seg]],
) -> BSPTree:
    # nodes
    if node_rows:
        nodes = np.array(node_rows, dtype=np.float32)
    else:
        nodes = np.empty((0, NODE_COLS), dtype=np.float32)

    # seg_pool and subsectors table
    all_segs_flat: List[List[float]] = []
    subsector_table: List[List[int]] = []

    for seg_list in subsector_segs:
        offset = len(all_segs_flat)
        for s in seg_list:
            all_segs_flat.append([
                s.x1, s.y1, s.x2, s.y2,
                float(s.front_sector),
                float(s.back_sector),
            ])
        subsector_table.append([offset, len(seg_list)])

    if all_segs_flat:
        seg_pool = np.array(all_segs_flat, dtype=np.float32)
    else:
        seg_pool = np.empty((0, SEG_COLS), dtype=np.float32)

    if subsector_table:
        subsectors = np.array(subsector_table, dtype=np.int32)
    else:
        subsectors = np.empty((0, 2), dtype=np.int32)

    return BSPTree(nodes=nodes, seg_pool=seg_pool, subsectors=subsectors)


# ---------------------------------------------------------------------------
# Convenience traversal for debug / validation
# ---------------------------------------------------------------------------

def point_in_subsector(
    tree: BSPTree,
    x: float,
    y: float,
) -> int:
    """
    Walk the BSP tree to find which subsector contains world point (x, y).
    Returns subsector index.  Pure Python — for debugging only.
    """
    if tree.node_count == 0:
        return 0   # degenerate tree with one implicit leaf

    node_idx = 0   # root is always the first node appended
    while node_idx >= 0:
        row  = tree.nodes[node_idx]
        px, py, dx, dy = row[N_PART_X], row[N_PART_Y], row[N_PART_DX], row[N_PART_DY]
        side = _cross_2d(dx, dy, x - px, y - py)

        if side >= 0:
            child = int(row[N_RIGHT])
        else:
            child = int(row[N_LEFT])

        if child < 0:
            return ~child   # leaf
        node_idx = child

    return 0   # unreachable


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from map_geometry import make_test_map

    geo  = make_test_map()
    tree = build_bsp(geo.vertices, geo.linedefs)
    print(tree)
    print(f"  nodes shape      : {tree.nodes.shape}")
    print(f"  seg_pool shape   : {tree.seg_pool.shape}")
    print(f"  subsectors shape : {tree.subsectors.shape}")
    print()

    # Sanity: points inside each room should land in different subsectors
    ss_a = point_in_subsector(tree, 3.0, 3.0)   # centre of Room A
    ss_b = point_in_subsector(tree, 9.0, 3.0)   # centre of Room B
    print(f"  Room A (3,3) → subsector {ss_a}")
    print(f"  Room B (9,3) → subsector {ss_b}")
    assert ss_a != ss_b, "Both rooms mapped to the same subsector — tree is broken"
    print("BSP validation passed.")
