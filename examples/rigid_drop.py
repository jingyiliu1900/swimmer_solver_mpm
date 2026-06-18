"""Example: a near-rigid, dense block dropped into a water pool.

The RIGID material uses the same elastic stress model as ELASTIC, but with a
much higher stiffness (so it barely deforms) and a higher density (so it sinks
instead of floating). True rigid bodies are not expressible in pure MPM; "very
stiff" is the standard stand-in.

Because RIGID is stiff, it needs a smaller timestep than the default or it goes
unstable -- this example uses dt=3e-5.

Run from the repo root:
    python examples/rigid_drop.py             # live window
    python examples/rigid_drop.py --headless  # no window, prints stats
"""

import os
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swimmers import WATER, RIGID, MPMSolver, MATERIALS
from swimmers.scenes import fill_rect, combine

ti.init(arch=ti.cpu)

# --- scene: a wide pool of water + a stiff, dense block falling in ----------
rng = np.random.default_rng(0)
pos, mat, vel = combine(
    fill_rect((0.04, 0.04), (0.92, 0.28), WATER, rng=rng),     # pool
    fill_rect((0.42, 0.70), (0.16, 0.16), RIGID, rng=rng),     # near-rigid block
)

# stiff material -> smaller timestep for stability
solver = MPMSolver(n_particles=pos.shape[0], n_grid=128, dt=3e-5)
solver.load_particles(pos, mat, vel)

headless = "--headless" in sys.argv
colors = np.array([m.color for m in MATERIALS], dtype=np.uint32)

# how much does the block deform? track the spread of its particles.
block = mat == RIGID
def block_size():
    x = solver.x.to_numpy()[block]
    return np.ptp(x[:, 0]), np.ptp(x[:, 1])   # width, height of the block's cloud

w0, h0 = block_size()

if headless:
    for frame in range(200):
        solver.step(substeps=40)
        if frame % 40 == 0:
            x = solver.x.to_numpy()
            w, h = block_size()
            cy = x[block][:, 1].mean()
            print(f"frame {frame:3d}  block center_y={cy:.3f}  "
                  f"deform: w {w/w0:.2f}x  h {h/h0:.2f}x")
    print("done (block stays ~1.0x its size = barely deforms, and sinks)")
else:
    gui = ti.GUI("rigid drop", res=(700, 700), background_color=0x112F41)
    while gui.running:
        solver.step(substeps=40)
        x = solver.x.to_numpy()
        m = solver.material.to_numpy()
        gui.circles(x, radius=1.6, palette=colors, palette_indices=m)
        gui.show()
