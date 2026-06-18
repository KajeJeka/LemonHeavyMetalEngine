import math

import numpy as np
import cv2
import pyglet

from raycaster import W, H, HALF_H
import rasterizer as _rasterizer


COLOR_CEILING = np.array([13,  13,  13],  dtype=np.uint8)
COLOR_FLOOR   = np.array([220, 64,  18],  dtype=np.uint8)

MINIMAP_WIDTH   = 280
MINIMAP_HEIGHT  = 240
MINIMAP_OX      = 10
MINIMAP_OY      = 10
MAP_WALL_COLOR  = (64,  64,  64)
MAP_FLOOR_COLOR = (13,  13,  13)
MAP_PLAYER_COL  = (0,   200, 240)
MAP_BORDER_COL  = (120, 120, 120)

MENU_BG         = (5,   5,   5)
MENU_TITLE_COL  = (200, 200, 200)
MENU_ITEM_COL   = (90,  90,  90)
MENU_SEL_COL    = (0,   200, 120)
MENU_HINT_COL   = (40,  40,  40)

_FONT      = cv2.FONT_HERSHEY_PLAIN
_FONT_MONO = cv2.FONT_HERSHEY_SIMPLEX


class Renderer:
    def __init__(self, game_map, texture_manager):
        self.map          = game_map
        self.textures     = texture_manager
        self.fb           = np.zeros((H, W, 3), dtype=np.uint8)
        self._vignette    = self._build_vignette()
        self.show_minimap = False
        self.crt_enabled  = True

    def update_map(self, game_map):
        self.map = game_map

    def _build_vignette(self) -> np.ndarray:
        ys = np.linspace(-1, 1, H, dtype=np.float32)
        xs = np.linspace(-1, 1, W, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        dist = np.sqrt(xx ** 2 + yy ** 2)
        mask = np.clip(1.0 - dist * 0.55, 0.0, 1.0)
        return np.stack([mask, mask, mask], axis=-1)

    # ── Game frame ────────────────────────────────────────────────

    def draw(self, fb: np.ndarray, vis, sectors: np.ndarray, player) -> np.ndarray:
        """
        Rasterize the scene into fb and apply post-effects.

        Parameters
        ----------
        fb      : uint8 [H, W, 3]  pre-allocated framebuffer, mutated in-place
        vis     : VisibleSet from bsp_traversal.cull_visible_segs
        sectors : float32 [S, 2]   from MapGeometry.sectors
        player  : Player

        Returns
        -------
        fb  (same object, for call-site convenience)
        """
        # BSP rasterizer fills walls, floor, and ceiling directly into fb.
        # Apply head bob offset to camera height
        _rasterizer.rasterize(fb, vis.vis_segs, vis.n, sectors, W, H, player.bob_offset)

        # CRT post-process (vignette + scanlines)
        self._apply_crt(fb)

        if self.show_minimap:
            self._draw_minimap(fb, player)

        return fb

    def _apply_crt(self, fb: np.ndarray) -> None:
        if not self.crt_enabled:
            return
        float_fb = fb.astype(np.float32)
        float_fb *= self._vignette
        float_fb[::2, :] *= 0.92
        np.clip(float_fb, 0, 255, out=float_fb)
        np.copyto(fb, float_fb.astype(np.uint8))

    def _draw_minimap(self, fb: np.ndarray, player) -> None:
        """
        Draws a top-down minimap from the linedef geometry.
        Works with MapGeometry which has vertices and linedefs attributes.
        """
        if not hasattr(self.map, "vertices") or not hasattr(self.map, "linedefs"):
            return

        vertices = self.map.vertices
        linedefs = self.map.linedefs

        if len(vertices) == 0 or len(linedefs) == 0:
            return

        # Calculate map bounds
        min_x = np.min(vertices[:, 0])
        max_x = np.max(vertices[:, 0])
        min_y = np.min(vertices[:, 1])
        max_y = np.max(vertices[:, 1])

        map_width = max_x - min_x
        map_height = max_y - min_y

        # Prevent division by zero
        if map_width < 0.1:
            map_width = 1.0
        if map_height < 0.1:
            map_height = 1.0

        # Calculate scale to fit minimap window with padding
        scale_x = (MINIMAP_WIDTH - 4) / map_width
        scale_y = (MINIMAP_HEIGHT - 4) / map_height
        scale = min(scale_x, scale_y)

        # Draw minimap background
        ox, oy = MINIMAP_OX, MINIMAP_OY
        cv2.rectangle(fb, (ox - 1, oy - 1),
                      (ox + MINIMAP_WIDTH, oy + MINIMAP_HEIGHT),
                      (0, 0, 0), -1)
        cv2.rectangle(fb, (ox - 1, oy - 1),
                      (ox + MINIMAP_WIDTH, oy + MINIMAP_HEIGHT),
                      MAP_BORDER_COL, 2)

        # Draw all linedefs
        for linedef in linedefs:
            v_start_idx = int(linedef[0])
            v_end_idx = int(linedef[1])

            if 0 <= v_start_idx < len(vertices) and 0 <= v_end_idx < len(vertices):
                v_start = vertices[v_start_idx]
                v_end = vertices[v_end_idx]

                # Transform to minimap coordinates
                x1 = int(ox + 2 + (v_start[0] - min_x) * scale)
                y1 = int(oy + 2 + (v_start[1] - min_y) * scale)
                x2 = int(ox + 2 + (v_end[0] - min_x) * scale)
                y2 = int(oy + 2 + (v_end[1] - min_y) * scale)

                # Determine if wall is solid (back_sector == -1)
                is_solid = int(linedef[3]) == -1
                color = MAP_WALL_COLOR if is_solid else (40, 40, 40)
                thickness = 2 if is_solid else 1

                cv2.line(fb, (x1, y1), (x2, y2), color, thickness)

        # Draw player position and direction indicator
        px = ox + 2 + (player.x - min_x) * scale
        py = oy + 2 + (player.y - min_y) * scale

        cv2.circle(fb, (int(px), int(py)), 4, MAP_PLAYER_COL, -1)

        # Draw direction indicator (small line from player center)
        direction_length = 15
        ex = px + math.cos(player.angle) * direction_length
        ey = py + math.sin(player.angle) * direction_length
        cv2.line(fb, (int(px), int(py)), (int(ex), int(ey)), MAP_PLAYER_COL, 2)

    # ── Menu frame ────────────────────────────────────────────────

    def draw_menu(self, state_mgr) -> np.ndarray:
        fb = self.fb
        fb[:] = MENU_BG

        cx = W // 2

        title = "L E M O N"
        (tw, th), _ = cv2.getTextSize(title, _FONT_MONO, 1.4, 2)
        cv2.putText(fb, title, (cx - tw // 2, 140),
                    _FONT_MONO, 1.4, MENU_TITLE_COL, 2, cv2.LINE_AA)

        subtitle = "HEAVY  METAL  v1.0"
        (sw, _), _ = cv2.getTextSize(subtitle, _FONT, 1.0, 1)
        cv2.putText(fb, subtitle, (cx - sw // 2, 165),
                    _FONT, 1.0, MENU_HINT_COL, 1, cv2.LINE_AA)

        cv2.line(fb, (cx - 80, 182), (cx + 80, 182), (30, 30, 30), 1)

        for i, option in enumerate(state_mgr.options):
            selected = (i == state_mgr.cursor)
            color    = MENU_SEL_COL if selected else MENU_ITEM_COL
            scale    = 0.75 if selected else 0.65
            prefix   = "> " if selected else "  "
            text     = prefix + option
            (tw, _), _ = cv2.getTextSize(text, _FONT_MONO, scale, 1)
            y = 230 + i * 36
            cv2.putText(fb, text, (cx - tw // 2, y),
                        _FONT_MONO, scale, color, 1, cv2.LINE_AA)

        hint = "W/S or UP/DOWN  SELECT    ENTER  CONFIRM"
        (hw, _), _ = cv2.getTextSize(hint, _FONT, 0.75, 1)
        cv2.putText(fb, hint, (cx - hw // 2, H - 28),
                    _FONT, 0.75, MENU_HINT_COL, 1, cv2.LINE_AA)

        float_fb = fb.astype(np.float32)
        float_fb[::2, :] *= 0.88
        np.clip(float_fb, 0, 255, out=float_fb)
        np.copyto(fb, float_fb.astype(np.uint8))

        return fb

    # ── Pyglet conversion ─────────────────────────────────────────

    def to_pyglet_image(self, fb: np.ndarray) -> pyglet.image.ImageData:
        rgb_flipped = cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)
        rgb_flipped = np.ascontiguousarray(np.flipud(rgb_flipped))
        return pyglet.image.ImageData(W, H, "RGB", rgb_flipped.tobytes())
