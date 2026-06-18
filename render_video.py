"""Render a scene to an animated GIF (no display needed).

Renders particle frames with matplotlib (Agg backend) and assembles them into a
GIF with Pillow, so the result embeds straight into the README. Water is teal,
elastic is red, rigid is yellow.

Usage:
    python render_video.py                          # showcase scene -> out.gif
    python render_video.py --scene splash --frames 200 --out splash.gif
    python render_video.py --scene showcase --dt 3e-5   # smaller dt for rigid
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import taichi as ti  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from swimmers.materials import MATERIALS  # noqa: E402
from swimmers.render import figure_to_rgb, save_gif  # noqa: E402
from swimmers.scenes import SCENES  # noqa: E402
from swimmers.solver import MPMSolver  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="showcase", choices=list(SCENES))
    ap.add_argument("--out", default="out.gif")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--dt", type=float, default=3e-5,
                    help="use 3e-5 for scenes with the rigid material")
    ap.add_argument("--gravity", type=float, default=9.8,
                    help="downward acceleration; use 0 for no gravity")
    ap.add_argument("--viscosity", type=float, default=0.0,
                    help="fluid viscosity; 0=water, ~0.5=honey (use --dt 3e-5)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--n-grid", type=int, default=128)
    ap.add_argument("--dpi", type=int, default=80)
    args = ap.parse_args()

    ti.init(arch=ti.cpu)
    pos, mat, vel = SCENES[args.scene]()
    solver = MPMSolver(n_particles=pos.shape[0], n_grid=args.n_grid, dt=args.dt,
                       gravity=args.gravity, viscosity=args.viscosity)
    solver.load_particles(pos, mat, vel)

    # matplotlib colours from each material's 0xRRGGBB
    colors = ["#%06X" % m.color for m in MATERIALS]
    point_colors = np.array(colors, dtype=object)[mat]

    print(f"scene={args.scene}  particles={solver.n_particles}  "
          f"frames={args.frames}  dt={args.dt}  -> {args.out}")

    fig = plt.figure(figsize=(6, 6), facecolor="#112F41", dpi=args.dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#112F41")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    scat = ax.scatter(pos[:, 0], pos[:, 1], s=4, c=point_colors, linewidths=0)

    frames = []
    for f in range(args.frames):
        solver.step(args.substeps)
        scat.set_offsets(solver.x.to_numpy())
        frames.append(figure_to_rgb(fig))
        if f % 25 == 0:
            print(f"  rendered frame {f}/{args.frames}")
    plt.close(fig)

    save_gif(frames, args.out, fps=args.fps)
    print(f"wrote {args.out}  ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
