"""What does Poisson's ratio (nu) do to the elastic material?

Poisson's ratio controls how volume responds to deformation:
  * low  nu (~0.0)  -> compressible: squashing it shrinks its area.
  * high nu (~0.45) -> nearly incompressible: squashing it makes it bulge
                       sideways to keep its area roughly constant (like rubber).

This drops an elastic ball (fixed stiffness E) at three Poisson ratios, with a
downward kick, and tracks the ball's bounding-box AREA over time. If area dips
a lot on impact the material is compressible; if area is preserved it is
nearly incompressible.

Outputs a montage PNG -> poisson_sweep.png and a printed table.

Run from the repo root:
    python examples/poisson_sweep.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import taichi as ti  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import swimmers.materials as materials  # noqa: E402
from swimmers import ELASTIC, MPMSolver  # noqa: E402
from swimmers.scenes import fill_circle  # noqa: E402

ti.init(arch=ti.cpu)

POISSON = [0.0, 0.25, 0.45]   # compressible -> nearly incompressible
E = 1500.0
N_FRAMES = 180
SUBSTEPS = 30
SNAP_FRAMES = [0, 80, 115, 145, 179]


def extent(a):
    lo, hi = np.percentile(a, [2, 98], axis=0)
    return hi[0] - lo[0], hi[1] - lo[1]


fig, axes = plt.subplots(len(POISSON), len(SNAP_FRAMES),
                         figsize=(2.0 * len(SNAP_FRAMES), 2.0 * len(POISSON)),
                         facecolor="white")

print(f"{'nu':>5} | {'peak area lost':>14} | {'final area':>10} | meaning")
print("-" * 60)

for row, nu in enumerate(POISSON):
    base = materials.MATERIALS[ELASTIC]
    materials.MATERIALS[ELASTIC] = materials.Material(
        name="elastic", youngs=E, poisson=nu, rho=base.rho, color=base.color)

    pos, mat, _ = fill_circle((0.5, 0.45), 0.13, ELASTIC,
                              rng=np.random.default_rng(0))
    # near-incompressible (high nu) has a large bulk modulus -> needs small dt
    solver = MPMSolver(n_particles=pos.shape[0], n_grid=128, dt=2e-5, gravity=9.8)
    solver.load_particles(
        pos, mat, np.tile([0.0, -3.0], (pos.shape[0], 1)).astype(np.float32))

    w0, h0 = extent(pos)
    area0 = w0 * h0
    snaps = {}
    min_area, min_frame, final_area = 1.0, 0, 1.0
    for f in range(N_FRAMES):
        solver.step(SUBSTEPS)
        x = solver.x.to_numpy()
        w, h = extent(x)
        area = (w * h) / area0
        if area < min_area:
            min_area, min_frame = area, f
        final_area = area
        if f in SNAP_FRAMES:
            snaps[f] = x.copy()

    for col, f in enumerate(SNAP_FRAMES):
        ax = axes[row, col]
        ax.scatter(snaps[f][:, 0], snaps[f][:, 1], s=2, c="#3A7CA5", linewidths=0)
        ax.set_xlim(0.25, 0.75); ax.set_ylim(0, 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"frame {f}", fontsize=10)
        if col == 0:
            ax.set_ylabel(f"nu = {nu:.2f}", fontsize=11)

    lost = 100.0 * (1.0 - min_area)
    print(f"{nu:>5.2f} | {lost:>12.1f}% | {final_area:>10.2f} | "
          f"more nu = more volume-preserving")

fig.suptitle("Elastic ball impact at varying Poisson ratio: area loss = compressibility",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(os.path.dirname(__file__), "..", "poisson_sweep.png")
fig.savefig(out, dpi=110)
print(f"\nwrote {os.path.abspath(out)}")
