from pathlib import Path

import numpy as np
from PIL import Image


class TextureManager:
    """
    Pillow-backed texture store.

    Textures are loaded once and cached as (H, W, 3) uint8 NumPy arrays
    in BGR channel order (OpenCV convention) for zero-copy sampling.

    Usage
    -----
    tm = TextureManager()
    tm.load("wall_stone", "assets/stone.png")
    col = tm.sample_column("wall_stone", u=0.35, col_height=120)
    """

    def __init__(self):
        self._textures: dict[str, np.ndarray] = {}

    def load(self, name: str, path: str | Path) -> None:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        self._textures[name] = arr[:, :, ::-1]   # RGB → BGR

    def get(self, name: str) -> np.ndarray | None:
        return self._textures.get(name)

    def sample_column(
        self,
        name: str,
        u: float,
        col_height: int,
    ) -> np.ndarray:
        """
        Return a vertical BGR strip of `col_height` pixels for texture
        coordinate `u` (0.0–1.0 along the wall face width).

        The returned array has shape (col_height, 1, 3) for direct
        placement into an OpenCV framebuffer column.
        """
        tex = self._textures.get(name)
        if tex is None:
            return np.full((col_height, 1, 3), 128, dtype=np.uint8)

        tex_h, tex_w = tex.shape[:2]
        tx = min(int(u * tex_w), tex_w - 1)
        col = tex[:, tx, :]                    # (tex_h, 3)

        ys = np.linspace(0, tex_h - 1, col_height).astype(np.int32)
        return col[ys].reshape(col_height, 1, 3)

    def has(self, name: str) -> bool:
        return name in self._textures

    @property
    def loaded(self) -> list[str]:
        return list(self._textures.keys())
