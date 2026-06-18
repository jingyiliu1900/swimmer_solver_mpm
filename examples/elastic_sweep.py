"""Is the 'elastic' material actually elastic?

Elastic = deforms under load, then RECOVERS its original shape. (Plastic would
stay deformed; rigid would not deform at all.)

This sweeps Young's modulus E over three values, drops an elastic ball onto the
floor with a downward kick, and tracks the ball's width/height aspect ratio
each frame. If the aspect ratio spikes on impact (squash) and then returns
toward 1.0 (round again), the material is elastic.

Outputs:
  * a montage PNG (rows = stiffness, columns = time) -> elastic_sweep.png
  * a printed table of aspect ratio over time

Run from the repo root:
    python examples/elastic_sweep.py
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

YOUNGS = [300.0, 1500.0, 8000.0]   # soft -> stiff
N_FRAMES = 180
SUBSTEPS = 30
SNAP_FRAMES = [0, 80, 115, 145, 179]  # columns: fall, impact, peak squash, recover

fig, axes = plt.subplots(len(YOUNGS), len(SNAP_FRAMES),
                         figsize=(2.0 * len(SNAP_FRAMES), 2.0 * len(YOUNGS)),
                         facecolor="white")

def extent(a):
    """Robust width/height of a point cloud (2nd-98th percentile span)."""
    lo, hi = np.percentile(a, [2, 98], axis=0)
    return hi[0] - lo[0], hi[1] - lo[1]


print(f"{'E':>7} | {'peak squash':>11} (frame) | {'final shape':>11} | verdict")
print("-" * 64)

for row, E in enumerate(YOUNGS):
    # override the elastic material's stiffness for this run, keep Poisson/rho
    base = materials.MATERIALS[ELASTIC]
    materials.MATERIALS[ELASTIC] = materials.Material(
        name="elastic", youngs=E, poisson=base.poisson, rho=base.rho,
        color=base.color)

    # one elastic ball, dropped with a downward kick onto the floor
    pos, mat, _ = fill_circle((0.5, 0.45), 0.13, ELASTIC,
                              rng=np.random.default_rng(0))
    solver = MPMSolver(n_particles=pos.shape[0], n_grid=128, dt=3e-5, gravity=9.8)
    solver.load_particles(pos, mat)
    solver.v.from_numpy(np.tile([0.0, -3.0], (pos.shape[0], 1)).astype(np.float32))

    snaps = {}
    w0, h0 = extent(pos)
    peak_ratio, peak_frame, final_ratio = 1.0, 0, 1.0
    for f in range(N_FRAMES):
        solver.step(SUBSTEPS)
        x = solver.x.to_numpy()
        w, h = extent(x)
        ratio = (w / w0) / (h / h0)   # >1 = squashed wide, <1 = stretched tall
        if ratio > peak_ratio:
            peak_ratio, peak_frame = ratio, f
        final_ratio = ratio
        if f in SNAP_FRAMES:
            snaps[f] = x.copy()

    for col, f in enumerate(SNAP_FRAMES):
        ax = axes[row, col]
        ax.scatter(snaps[f][:, 0], snaps[f][:, 1], s=2, c="#ED553B", linewidths=0)
        ax.set_xlim(0.25, 0.75)
        ax.set_ylim(0, 0.5)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f"frame {f}", fontsize=10)
        if col == 0:
            ax.set_ylabel(f"E = {E:.0f}", fontsize=11)

    recovered = abs(final_ratio - 1.0) < 0.5 * (peak_ratio - 1.0) + 0.05
    verdict = "elastic (recovered)" if recovered and peak_ratio > 1.05 \
        else ("elastic" if recovered else "still deformed")
    print(f"{E:>7.0f} | {peak_ratio:>10.2f} ({peak_frame:>3}) | "
          f"{final_ratio:>11.2f} | {verdict}")

fig.suptitle("Elastic ball dropped on floor: squash on impact, recover toward round",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(os.path.dirname(__file__), "..", "elastic_sweep.png")
fig.savefig(out, dpi=110)
print(f"\nwrote {os.path.abspath(out)}")
print("aspect ratio ~1.0 = round; >1.0 = squashed wide. "
      "Spike then return-to-~1 == elastic recovery.")
