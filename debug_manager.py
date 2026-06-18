import math

import numpy as np
import cv2

from raycaster     import W, H
from bsp_traversal import VIS_DEPTH1


_FONT       = cv2.FONT_HERSHEY_PLAIN
_FONT_SCALE = 0.95
_THICKNESS  = 1
_LINE_H     = 14
_PAD        = 8
_BG_COLOR   = (10, 10, 10)
_TEXT_COLOR = (0, 200, 120)
_DIM_COLOR  = (60, 100, 80)
_ZBUF_H     = 32
_PANEL_W    = 260


class DebugManager:
    def __init__(self):
        self.enabled      = False
        self.frame_ms     = 0.0
        self.raycast_ms   = 0.0   # now measures BSP traversal time
        self.render_ms    = 0.0
        self.collided     = False
        self.game_state   = "MENU"
        self.map_mode     = "DEFAULT"
        self.map_seed: int | None = None
        self._frame_count = 0
        self._accum       = 0.0
        self._fps_display = 0.0

    def toggle(self):
        self.enabled = not self.enabled

    def update_timing(self, dt: float, bsp_ms: float, render_ms: float):
        self.frame_ms   = dt * 1000.0
        self.raycast_ms = bsp_ms
        self.render_ms  = render_ms
        self._accum    += dt
        self._frame_count += 1
        if self._accum >= 0.5:
            self._fps_display = self._frame_count / self._accum
            self._frame_count = 0
            self._accum       = 0.0

    def draw(self, fb: np.ndarray, player, vis) -> None:
        """
        vis  –  VisibleSet returned by bsp_traversal.cull_visible_segs.
        Replaces the legacy RaycastResult parameter.
        """
        if not self.enabled:
            return

        lines   = self._build_lines(player, vis)
        panel_h = _PAD + len(lines) * _LINE_H + _PAD + _ZBUF_H + _PAD
        panel_w = _PANEL_W

        x0, y0 = W - panel_w - 8, 8
        roi     = fb[y0:y0 + panel_h, x0:x0 + panel_w]
        overlay = roi.astype(np.float32)
        overlay = overlay * 0.25 + np.array(_BG_COLOR, dtype=np.float32) * 0.75
        np.copyto(roi, overlay.clip(0, 255).astype(np.uint8))

        for i, (label, value) in enumerate(lines):
            y = y0 + _PAD + i * _LINE_H + _LINE_H - 2
            cv2.putText(fb, label, (x0 + _PAD, y),
                        _FONT, _FONT_SCALE, _DIM_COLOR, _THICKNESS, cv2.LINE_AA)
            cv2.putText(fb, value, (x0 + 100, y),
                        _FONT, _FONT_SCALE, _TEXT_COLOR, _THICKNESS, cv2.LINE_AA)

        zbar_y = y0 + _PAD + len(lines) * _LINE_H + _PAD

        # Depth values for visible segs: vis_segs[:n, VIS_DEPTH1]
        depths = vis.vis_segs[:vis.n, VIS_DEPTH1] if vis.n > 0 else np.zeros(1, dtype=np.float32)
        self._draw_zbuffer(fb, depths, x0, zbar_y, panel_w - _PAD * 2, _ZBUF_H)

        cv2.rectangle(fb, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 80, 50), 1)

    def _build_lines(self, player, vis) -> list[tuple[str, str]]:
        depths   = vis.vis_segs[:vis.n, VIS_DEPTH1] if vis.n > 0 else np.zeros(1, dtype=np.float32)
        fb_bytes = vis.vis_segs.nbytes + vis.occlusion.nbytes
        seed_str = f"{self.map_seed:08X}" if self.map_seed is not None else "N/A"
        return [
            ("STATE",     self.game_state),
            ("MAP",       self.map_mode),
            ("SEED",      seed_str),
            ("FPS",       f"{self._fps_display:.1f}"),
            ("FRAME",     f"{self.frame_ms:.2f} ms"),
            ("BSP",       f"{self.raycast_ms:.2f} ms"),
            ("RENDER",    f"{self.render_ms:.2f} ms"),
            ("POS X",     f"{player.x:.4f}"),
            ("POS Y",     f"{player.y:.4f}"),
            ("ANGLE",     f"{math.degrees(player.angle) % 360:.1f}°"),
            ("BOB",       f"{player.bob_offset:.2f} px"),
            ("COLLIDE",   "YES" if self.collided else "NO"),
            ("VIS SEGS",  str(vis.n)),
            ("OCC COLS",  str(int(vis.occlusion.sum()))),
            ("DEPTH MIN", f"{float(depths.min()):.3f}"),
            ("DEPTH MAX", f"{float(depths.max()):.3f}"),
            ("FB MEM",    f"{fb_bytes / 1024:.1f} KB"),
        ]

    def _draw_zbuffer(self, fb, depths, x0, y0, width, height):
        """Bar chart of visible seg depths (replaces per-column Z-buffer display)."""
        if len(depths) == 0:
            return

        z_norm = np.clip(depths / 20.0, 0.0, 1.0)
        xs     = np.linspace(0, width - 1, len(z_norm)).astype(np.int32)
        bar_h  = (z_norm * height).astype(np.int32)

        for i in range(len(xs) - 1):
            bh = int(bar_h[i])
            if bh < 1:
                continue
            x = x0 + int(xs[i])
            cv2.line(fb, (x, y0 + height), (x, y0 + height - bh), (0, 140, 80), 1)

        cv2.rectangle(fb, (x0, y0), (x0 + width, y0 + height), (30, 80, 50), 1)
        cv2.putText(fb, "DEPTH", (x0 + 2, y0 + 10),
                    _FONT, 0.8, _DIM_COLOR, 1, cv2.LINE_AA)
