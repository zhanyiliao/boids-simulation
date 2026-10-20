# Boids Simulation

A Python/Pygame demonstration of flocking through separation, alignment and cohesion.
Each boid reacts only to local neighbors. The first boid is magenta; red, green and
blue circles show its separation, alignment and cohesion radii.

## Setup (Windows PowerShell)

Python 3.13 is the tested interpreter for the pinned dependencies.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe boid.py
```

No environment activation is required. In VS Code, choose
`.venv\Scripts\python.exe` under **Python: Select Interpreter**.

```powershell
.\.venv\Scripts\python.exe boid.py --smooth --boids_num 100
.\.venv\Scripts\python.exe boid.py --help
```

## Controls

Use the **Force weights** panel in the top-right corner to click or drag the red
separation, green alignment and blue cohesion sliders. Values update immediately
and are displayed to two decimal places. Each slider starts at a range of 0-2;
larger command-line or keyboard values automatically expand that slider's range.
Dragging beyond a track stops at its endpoints. Adjustments also work while paused
and affect movement when you resume. The panel stays visible when debug vectors
are hidden. Keyboard shortcuts change the same values shown by the sliders.

| Key | Action |
| --- | --- |
| Space | Pause/resume |
| R | Restore positions and velocities from the seed; retain current weights |
| D | Toggle behavior radii and acceleration vectors |
| 1 / 2 | Decrease/increase separation weight by 0.05 |
| 3 / 4 | Decrease/increase alignment weight by 0.05 |
| 5 / 6 | Decrease/increase cohesion weight by 0.05 |
| Esc | Exit |

Weights cannot go below zero. Current weights and rendering FPS appear in the window.
Colored vectors show each behavior before the combined acceleration cap and boundary
steering; the line in each boid's own color shows its final acceleration. Vectors
are scaled by 0.5 for visibility.

## Parameters

| Argument | Default | Meaning |
| --- | --- | --- |
| `--boids_num` | 50 | Positive number of boids |
| `--boid_size` | 8 | Circle radius in pixels |
| `--max_speed` | 100 | Speed limit in pixels/second |
| `--max_acceleration` | 150 | Combined acceleration limit in pixels/second squared |
| `--delta_time` | 0.0166667 | Fixed simulation step in seconds; greater than 0 and at most 0.05 |
| `--fps` | 60 | Rendering frame rate cap |
| `--sep_radius` | 22 | Separation radius |
| `--ali_radius` | 70 | Alignment radius |
| `--coh_radius` | 110 | Cohesion radius |
| `--sep_scalar` | 0.44 | Separation weight |
| `--ali_scalar` | 0.8 | Velocity matching gain |
| `--coh_scalar` | 0.09 | Attraction toward neighbors |
| `--smooth` | off | Linearly fade each neighbor's contribution near its radius |
| `--seed` | 603 | Nonnegative random seed |

Radii, speed and acceleration limits must be positive and finite; weights must be
finite and nonnegative. Size must fit within the 1080 x 720 window.

## Simulation design

- **Synchronous updates:** all behavior vectors are computed from the same positions
  and velocities before any boid moves.
- **Separation:** inverse-distance repulsion, capped per neighbor before averaging.
  Exactly overlapping pairs receive deterministic opposite horizontal forces.
- **Alignment:** use neighbor velocity minus the boid's own velocity, so matching
  velocities produce no alignment acceleration.
- **Cohesion:** average displacements toward neighbors.
- **Smoothing:** multiply contributions by `1 - distance / radius`, then divide by
  neighbor count. This fades the force at the radius rather than renormalizing weights.
- **Boundaries:** soft inward steering plus containment and inward reflection if a
  boid reaches an edge. Containment includes the circle radius.
- **Timing:** an accumulator advances fixed steps independently of rendering FPS.
  Catch-up is limited to eight steps per frame and frame time to 0.25 seconds. Under
  sustained overload the simulation slows rather than accumulating unlimited work.
- **Neighbor search:** rebuild a spatial grid each step and query nine nearby cells.
  This reduces work for dispersed flocks; dense flocks can still approach O(N^2).
- **Reproducibility:** resetting uses a local NumPy generator. Identical seeds,
  parameters and simulation step counts reproduce the state in the same environment.

The speed default changed from 20 to 100 and alignment weight from 0.08 to 0.8 when
switching to real seconds and velocity matching. Old presets do not reproduce the
previous motion. SciPy is no longer needed; two-dimensional distances use `math.hypot`.
Importing `boid` does not parse arguments or create a window.

## Verification

```powershell
.\.venv\Scripts\python.exe -B -m unittest -v
```

Tests run with SDL dummy video/audio drivers. They cover grid search against a full
scan, order-independent updates, alignment, separation, overlaps, limits, seeded
resets, argument validation, frame-rate independence, rendering, keyboard controls
and mouse sliders.

## Background

Originally developed for a Computer Animation and Special Effects course.
The flocking model is inspired by Craig Reynolds' 1987 paper,
[Flocks, Herds and Schools: A Distributed Behavioral Model](https://doi.org/10.1145/37402.37406).
