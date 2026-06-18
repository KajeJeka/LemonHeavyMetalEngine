"""
raycaster.py  –  Lemon Heavy Metal Engine
Legacy module retained for screen dimension constants.

Raycaster and RaycastResult have been retired.  Rendering is now handled
by the BSP pipeline: bsp_traversal.cull_visible_segs → rasterizer.rasterize.
"""

W         = 800
H         = 450
HALF_H    = H // 2
MAX_DEPTH = 64    # retained for reference
