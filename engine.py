import math
import time

import numpy as np
import pyglet
from pyglet.window import key, mouse

from player        import Player, collision_warm_up
from renderer      import Renderer
from texture_manager import TextureManager
from debug_manager import DebugManager
from game_state    import GameStateManager, STATE_MENU, STATE_PLAYING
from map_generator import MapGenerator

from map_geometry  import make_test_map, geometry_from_grid
from bsp_builder   import build_bsp
from bsp_traversal import cull_visible_segs, warm_up as _bsp_warm_up
import rasterizer  as _rasterizer

from raycaster import W, H


FOV_RAD = math.radians(66.0)
FPS_CAP = 60
TITLE   = "LEMON HEAVY METAL"


class Engine(pyglet.window.Window):
    def __init__(self):
        super().__init__(width=W, height=H, caption=TITLE, vsync=False)
        self.set_mouse_visible(True)

        self.state_mgr = GameStateManager()
        self.map_gen   = MapGenerator()
        self.textures  = TextureManager()
        self.player    = Player()

        # Load default (hand-authored) geometry and build BSP
        self._geo = make_test_map()
        self._bsp = build_bsp(self._geo.vertices, self._geo.linedefs)

        # Single framebuffer — never reallocated during the game loop
        self._fb = np.zeros((H, W, 3), dtype=np.uint8)

        self.renderer = Renderer(self._geo, self.textures)
        self.debug    = DebugManager()

        self._keys           = key.KeyStateHandler()
        self.push_handlers(self._keys)
        self._mouse_captured = False
        self._mouse_dx       = 0.0

        self._prev_enter = False
        self._prev_up    = False
        self._prev_down  = False
        self._prev_m     = False
        self._prev_f2    = False
        self._prev_esc   = False

        self._pending_image = None

        # Compile all Numba kernels before first frame
        print("Warming up JIT kernels...")
        _bsp_warm_up()
        _rasterizer.warm_up(W, H)
        collision_warm_up(
            self._bsp.nodes,
            self._bsp.seg_pool,
            self._bsp.subsectors,
        )
        print("JIT ready.")

        pyglet.clock.schedule_interval(self._update, 1.0 / FPS_CAP)

    # ── Pyglet event handlers ─────────────────────────────────────

    def on_mouse_press(self, x, y, button, modifiers):
        if button == mouse.LEFT and self.state_mgr.in_game and not self._mouse_captured:
            self.set_exclusive_mouse(True)
            self.set_mouse_visible(False)
            self._mouse_captured = True

    def on_mouse_motion(self, x, y, dx, dy):
        if self._mouse_captured:
            self._mouse_dx += dx

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self._mouse_captured:
            self._mouse_dx += dx

    def on_key_press(self, symbol, modifiers):
        if symbol in (key.F1, key.F3):
            self.debug.toggle()

    def on_close(self):
        pyglet.app.exit()

    # ── Main update tick ──────────────────────────────────────────

    def _update(self, dt: float):
        dt = min(dt, 0.05)
        if self.state_mgr.in_menu:
            self._update_menu(dt)
        else:
            self._update_game(dt)

    def _update_menu(self, dt: float):
        k = self._keys

        up_now    = bool(k[key.W] or k[key.UP])
        down_now  = bool(k[key.S] or k[key.DOWN])
        enter_now = bool(k[key.RETURN] or k[key.SPACE])

        if up_now    and not self._prev_up:    self.state_mgr.select_up()
        if down_now  and not self._prev_down:  self.state_mgr.select_down()
        if enter_now and not self._prev_enter: self._launch_game()

        self._prev_up    = up_now
        self._prev_down  = down_now
        self._prev_enter = enter_now

        fb = self.renderer.draw_menu(self.state_mgr)
        self._pending_image = self.renderer.to_pyglet_image(fb)
        self.invalid = True

    def _update_game(self, dt: float):
        k = self._keys

        esc_now = bool(k[key.ESCAPE])
        if esc_now and not self._prev_esc:
            if self._mouse_captured:
                self._release_mouse()
            else:
                self._return_to_menu()
        self._prev_esc = esc_now

        keys = self._build_keyset()
        mdx  = self._mouse_dx
        self._mouse_dx = 0.0

        # BSP traversal
        t0 = time.perf_counter()
        vis = cull_visible_segs(
            self.player.x, self.player.y,
            self.player.angle, FOV_RAD,
            self._bsp.nodes,
            self._bsp.seg_pool,
            self._bsp.subsectors,
            screen_w=W,
            screen_h=H,
        )
        bsp_ms = (time.perf_counter() - t0) * 1000.0

        # Player physics — linedef-based, uses the same BSP arrays as rendering
        collided = self.player.update(
            keys, mdx, dt,
            self._bsp.nodes,
            self._bsp.seg_pool,
            self._bsp.subsectors,
        )

        self.debug.collided   = collided
        self.debug.game_state = self.state_mgr.state
        self.debug.map_mode   = self.state_mgr.map_mode
        self.debug.map_seed   = self.state_mgr.map_seed

        # Rasterize + post-process
        t1 = time.perf_counter()
        fb = self.renderer.draw(self._fb, vis, self._geo.sectors, self.player)
        render_ms = (time.perf_counter() - t1) * 1000.0

        self.debug.draw(fb, self.player, vis)
        self.debug.update_timing(dt, bsp_ms, render_ms)

        if not self._mouse_captured:
            self._draw_cv_overlay(fb)

        self._pending_image = self.renderer.to_pyglet_image(fb)
        self.invalid = True

    # ── Draw ─────────────────────────────────────────────────────

    def on_draw(self):
        self.clear()
        if self._pending_image is not None:
            self._pending_image.blit(0, 0)

    # ── Helpers ──────────────────────────────────────────────────

    def _launch_game(self):
        choice = self.state_mgr.confirm()

        if choice == "INFINITE":
            # Generate tile grid, then convert to linedef geometry in one step.
            # Both rendering and physics use the same MapGeometry / BSPTree —
            # the Frankenstein split is gone.
            self.map_gen.generate()
            tile_map = self.map_gen.build_map()
            self.state_mgr.map_seed = self.map_gen.seed

            spawn_x = float(tile_map.spawn_x)
            spawn_y = float(tile_map.spawn_y)

            self._geo = geometry_from_grid(
                tile_map.grid,
                name=f"generated_{self.map_gen.seed:08X}",
                spawn=(spawn_x, spawn_y),
            )
        else:
            self._geo = make_test_map()

        self._bsp = build_bsp(self._geo.vertices, self._geo.linedefs)

        spawn_x, spawn_y = self._geo.spawn
        self.player.reset(spawn_x, spawn_y)
        self.renderer.update_map(self._geo)

    def _return_to_menu(self):
        self.state_mgr.return_to_menu()
        self._release_mouse()

    def _release_mouse(self):
        self.set_exclusive_mouse(False)
        self.set_mouse_visible(True)
        self._mouse_captured = False

    def _build_keyset(self) -> set[str]:
        k = self._keys
        keys: set[str] = set()
        if k[key.W] or k[key.UP]:          keys.add("FORWARD")
        if k[key.S] or k[key.DOWN]:        keys.add("BACK")
        if k[key.A]:                        keys.add("SLEFT")
        if k[key.D]:                        keys.add("SRIGHT")
        if k[key.LEFT]:                     keys.add("LEFT")
        if k[key.RIGHT]:                    keys.add("RIGHT")
        if k[key.LSHIFT] or k[key.RSHIFT]: keys.add("SPRINT")

        m_now = bool(k[key.M])
        if m_now and not self._prev_m:
            self.renderer.show_minimap = not self.renderer.show_minimap
        self._prev_m = m_now

        f2_now = bool(k[key.F2])
        if f2_now and not self._prev_f2:
            self.renderer.crt_enabled = not self.renderer.crt_enabled
        self._prev_f2 = f2_now

        return keys

    def _draw_cv_overlay(self, fb: np.ndarray) -> None:
        import cv2
        roi     = fb[H // 2 - 12:H // 2 + 12, W // 2 - 100:W // 2 + 100]
        overlay = roi.astype(np.float32) * 0.5
        np.copyto(roi, overlay.clip(0, 255).astype(np.uint8))
        cv2.putText(fb, "CLICK TO ENABLE MOUSE LOOK",
                    (W // 2 - 98, H // 2 + 5),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (48, 48, 48), 1, cv2.LINE_AA)

    def run(self):
        pyglet.app.run()
