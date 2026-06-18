"""Example: build your own scene and drive the solver directly.

Run from the repo root:
    python examples/custom_scene.py            # live window
    python examples/custom_scene.py --headless # no window, prints stats
"""

import os
import sys

import numpy as np
import taichi as ti

# make `import swimmers` work when running from the repo without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swimmers import WATER, ELASTIC, MPMSolver, MATERIALS
from swimmers.scenes import fill_rect, fill_circle, combine

ti.init(arch=ti.cpu)

# --- 1. compose a scene: a shallow pool + an elastic ball dropped high up ----
rng = np.random.default_rng(0)
pos, mat, vel = combine(
    fill_rect((0.05, 0.05), (0.90, 0.15), WATER, rng=rng),    # pool at the bottom
    fill_circle((0.50, 0.80), 0.10, ELASTIC, rng=rng),        # ball up high
)

# --- 2. create the solver and load the particles ----------------------------
solver = MPMSolver(n_particles=pos.shape[0], n_grid=128, dt=1e-4)
solver.load_particles(pos, mat, vel)

# --- 3. step it ------------------------------------------------------------
headless = "--headless" in sys.argv
colors = np.array([m.color for m in MATERIALS], dtype=np.uint32)

if headless:
    for frame in range(150):
        solver.step(substeps=40)              # advance one rendered frame
        if frame % 30 == 0:
            x = solver.x.to_numpy()           # (N, 2) positions, read anytime
            print(f"frame {frame:3d}  top y = {x[:,1].max():.3f}")
    print("done")
else:
    gui = ti.GUI("custom scene", res=(700, 700), background_color=0x112F41)
    while gui.running:
        solver.step(substeps=40)
        x = solver.x.to_numpy()
        m = solver.material.to_numpy()
        gui.circles(x, radius=1.6, palette=colors, palette_indices=m)
        gui.show()
