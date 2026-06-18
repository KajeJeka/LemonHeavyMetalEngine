import random
import numpy as np


class MapGenerator:
    """
    Procedural maze using Recursive Backtracking (DFS).

    Grid size must be odd×odd so the algorithm's 2-step carving always
    lands on valid cell coordinates. The outer border is always wall.
    Player spawn is guaranteed at (1,1) which is always carved open.
    """

    ROWS = 31
    COLS = 41

    def __init__(self):
        self.seed: int = 0
        self.grid: np.ndarray = np.ones((self.ROWS, self.COLS), dtype=np.uint8)

    def generate(self, seed: int | None = None) -> "MapGenerator":
        self.seed = seed if seed is not None else random.randint(0, 0xFFFFFFFF)
        rng = random.Random(self.seed)

        grid = np.ones((self.ROWS, self.COLS), dtype=np.uint8)
        visited = np.zeros((self.ROWS, self.COLS), dtype=bool)

        def carve(r: int, c: int):
            visited[r, c] = True
            grid[r, c] = 0
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            rng.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.ROWS - 1 and 1 <= nc < self.COLS - 1 and not visited[nr, nc]:
                    grid[r + dr // 2, c + dc // 2] = 0   # knock out wall between
                    carve(nr, nc)

        # Increase recursion limit for large grids, then restore
        import sys
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(4000)
        carve(1, 1)
        sys.setrecursionlimit(old_limit)

        # Guarantee outer border is solid
        grid[0, :] = 1
        grid[-1, :] = 1
        grid[:, 0] = 1
        grid[:, -1] = 1

        self.grid = grid
        return self

    def build_map(self) -> "GeneratedMap":
        return GeneratedMap(self.grid.copy(), self.seed)


class GeneratedMap:
    """Drop-in replacement for Map that wraps a generated grid."""

    def __init__(self, grid: np.ndarray, seed: int):
        self.grid = grid
        self.rows, self.cols = grid.shape
        self.seed = seed
        # Spawn is always (1,1) which the carver guarantees empty
        self.spawn_x: float = 1.5
        self.spawn_y: float = 1.5

    def is_blocked(self, x: float, y: float) -> bool:
        col, row = int(x), int(y)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return True
        return self.grid[row, col] != 0

    def cell(self, x: float, y: float) -> int:
        col, row = int(x), int(y)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return 1
        return int(self.grid[row, col])
