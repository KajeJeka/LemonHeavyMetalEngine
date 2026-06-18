# Lemon Heavy Metal

A lightweight DOOM-style BSP engine written in Python.

Lemon Heavy Metal focuses on procedural map generation, BSP-based visibility determination, software rendering, and high-performance gameplay using Numba JIT acceleration. The project is designed as a foundation for experiments with classic 1990s FPS technology while remaining fully procedural and easy to extend.

---

## Features

* Procedural maze generation
* BSP tree construction
* BSP front-to-back traversal
* Software-rendered 3D walls
* Depth-buffered visibility
* Real-time collision detection
* Numba-accelerated critical systems
* Deterministic procedural generation
* Infinite map mode
* Debug visualization tools

---

## Technology Stack

### Python

Tested on:

```text
Python 3.12+
```

### Dependencies

```text
numpy
numba
pyglet
opencv-python
pillow
```

Recommended versions:

```text
numpy >= 2.0
numba >= 0.60
pyglet >= 2.1
opencv-python >= 4.10
pillow >= 11.0
```

Install:

```bash
pip install numpy numba pyglet opencv-python pillow
```

---

## Project Structure

```text
engine.py
player.py

map_generator.py
map_geometry.py

bsp_builder.py
bsp_traversal.py

rasterizer.py
renderer.py

debug_manager.py
raycaster.py
```

---

## Architecture Overview

The engine follows a traditional BSP rendering pipeline.

```text
Map Generation
        ↓
Geometry Conversion
        ↓
BSP Construction
        ↓
Visibility Traversal
        ↓
Rasterization
        ↓
Frame Output
```

Each subsystem has a single responsibility.

---

## Map Generation

File:

```text
map_generator.py
```

The engine generates maps procedurally at runtime.

The generator creates a two-dimensional grid containing walls and walkable cells. Every run can produce a different layout depending on the selected seed.

The resulting grid acts as the source representation of the world.

Example:

```text
#########
#.......#
#.###...#
#...#...#
#########
```

---

## Geometry System

File:

```text
map_geometry.py
```

The geometry layer converts the generated grid into renderable wall segments.

Responsibilities:

* Generate linedefs
* Build map boundaries
* Create wall geometry
* Store sector information
* Prepare data for BSP construction

The output is a collection of segments describing every visible wall in the map.

---

## BSP Builder

File:

```text
bsp_builder.py
```

The BSP builder converts map geometry into a Binary Space Partitioning tree.

Responsibilities:

* Select splitter segments
* Partition geometry
* Create BSP nodes
* Generate subsectors
* Build convex leaves

The BSP tree allows efficient visibility determination and reduces the amount of geometry that must be rendered each frame.

---

## BSP Traversal

File:

```text
bsp_traversal.py
```

The traversal system walks through the BSP tree relative to the player position.

Responsibilities:

* Determine visible subsectors
* Front-to-back traversal
* Frustum culling
* Occlusion management

Only potentially visible geometry is sent to the renderer.

---

## Rasterizer

File:

```text
rasterizer.py
```

The rasterizer converts visible wall segments into pixels.

Responsibilities:

* Perspective projection
* Screen-space clipping
* Column rendering
* Depth testing
* Framebuffer writes

The renderer uses a software rasterization approach inspired by classic FPS engines.

Walls are currently rendered using flat colors with distance-based shading.

---

## Rendering System

File:

```text
renderer.py
```

The renderer manages communication between the rasterizer and the display layer.

Responsibilities:

* Frame management
* Buffer handling
* Screen presentation
* Integration with Pyglet

This layer is intentionally lightweight.

---

## Player System

File:

```text
player.py
```

The player system handles movement and collision.

Responsibilities:

* Position updates
* Rotation
* Collision detection
* Movement constraints

Collision testing is accelerated using Numba.

The player cannot pass through solid walls.

---

## Game Loop

File:

```text
engine.py
```

The central controller of the engine.

Responsibilities:

* Window creation
* Input handling
* State management
* Update loop
* Render loop

The game loop continuously updates player state and renders the current frame.

---

## Infinite Mode

The engine includes an infinite exploration mode.

Instead of using a fixed map, new procedural areas can be generated as the player progresses.

This allows continuous exploration without predefined level boundaries.

---

## Performance

Performance-critical systems use Numba JIT compilation.

Accelerated components include:

* Collision detection
* BSP operations
* Geometry processing
* Rasterization

The goal is to keep most per-frame calculations inside compiled code rather than pure Python.

---

## Rendering Pipeline

The frame generation process follows these steps:

```text
1. Update player position
2. Traverse BSP tree
3. Collect visible segments
4. Project geometry
5. Rasterize walls
6. Perform depth testing
7. Present framebuffer
```

This process repeats every frame.

## COMMANDS
F1 = Debug Menu
F2 = Disable CRT Style
M = Map

