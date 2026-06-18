"""Showcase: three fluid columns at different viscosities collapsing at once.

Same tall starting column of fluid, four viscosities side by side:
  * viscosity 0.0  -> water : slumps into a low, wide puddle
  * viscosity 0.2  -> syrup : collapses partway into a mound
  * viscosity 0.6  -> honey : slumps a little, stays a fat blob
  * viscosity 2.0  -> tar   : barely leans -- still a standing column

Renders a side-by-side animated GIF (no display needed) via matplotlib + Pillow,
so it embeds straight into the README.

Run from the repo root:
    python examples/viscosity_showcase.py                  # -> viscosity.gif
    python examples/viscosity_showcase.py --frames 300 --out visc.gif
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import taichi as ti  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import swimmers.materials as materials  # noqa: E402
from swimmers import WATER, MPMSolver  # noqa: E402
from swimmers.render import figure_to_rgb, save_gif  # noqa: E402
from swimmers.scenes import fill_rect  # noqa: E402

ti.init(arch=ti.cpu)

# Stiffer (more incompressible) water so the runny cases slump as a COHERENT
# sheet instead of exploding into spray -- the bodies keep their shape, only how
# far they slump changes with viscosity.
_w = materials.MATERIALS[WATER]
materials.MATERIALS[WATER] = materials.Material(
    name="water", youngs=2500.0, poisson=0.2, rho=1.0, color=_w.color)

# (viscosity, label, colour). A wide sweep, water -> tar. The high end stays a
# standing column while water slumps into a low puddle. Higher viscosity needs a
# smaller dt and/or coarser grid to stay stable (stability ~ dt*visc*n_grid^2).
FLUIDS = [
    (0.0, "viscosity 0.0  (water)", "#068587"),
    (0.2, "viscosity 0.2  (syrup)", "#3A7CA5"),
    (0.6, "viscosity 0.6  (honey)", "#C77DA0"),
    (2.0, "viscosity 2.0  (tar)",   "#F2A900"),
]
N_GRID = 96    # a touch coarser -> headroom for the very viscous case
DT = 1.0e-5    # small dt: stable for stiff water up to viscosity ~2.5
GRAVITY = 6.5  # moderate, gentle pull: coherent slump, water ends a wide puddle


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="viscosity.gif")
    ap.add_argument("--frames", type=int, default=300)   # longer simulation
    ap.add_argument("--substeps", type=int, default=100)
    ap.add_argument("--fps", type=int, default=42)        # snappier playback
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    # one identical tall column per fluid -- the runny ones slump into a low
    # puddle, the thick ones barely lean: the contrast is the whole point.
    solvers = []
    for eta, _, _ in FLUIDS:
        pos, mat, vel = fill_rect((0.43, 0.04), (0.14, 0.62), WATER,
                                  rng=np.random.default_rng(0))
        s = MPMSolver(n_particles=pos.shape[0], n_grid=N_GRID, dt=DT,
                      gravity=GRAVITY, viscosity=eta)
        s.load_particles(pos, mat, vel)
        solvers.append(s)

    print(f"fluids={[f[0] for f in FLUIDS]}  particles/each={solvers[0].n_particles}"
          f"  frames={args.frames}  dt={DT}  -> {args.out}")

    fig, axes = plt.subplots(1, len(FLUIDS), figsize=(4 * len(FLUIDS), 4),
                             dpi=args.dpi, facecolor="#112F41")
    scats = []
    for ax, s, (eta, label, color) in zip(axes, solvers, FLUIDS):
        ax.set_facecolor("#112F41")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, color="white", fontsize=13)
        scats.append(ax.scatter(s.x.to_numpy()[:, 0], s.x.to_numpy()[:, 1],
                                s=5, c=color, linewidths=0))
    fig.tight_layout()

    frames = []
    for f in range(args.frames):
        for s in solvers:
            s.step(args.substeps)
        for scat, s in zip(scats, solvers):
            scat.set_offsets(s.x.to_numpy())
        frames.append(figure_to_rgb(fig))
        if f % 40 == 0:
            print(f"  frame {f}/{args.frames}")
    plt.close(fig)

    save_gif(frames, args.out, fps=args.fps)

    # final spreads, for the record
    for s, (eta, label, _) in zip(solvers, FLUIDS):
        x = s.x.to_numpy()
        w = np.percentile(x[:, 0], 98) - np.percentile(x[:, 0], 2)
        print(f"  {label}: final spread width = {w:.3f}")
    print(f"wrote {args.out}  ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()