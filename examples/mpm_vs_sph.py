"""Example: the SAME fluid scene solved two ways, side by side -- MPM vs SPH.

Both solvers are given the *identical* initial particle cloud (a dam-break: a
tall water column released against a wall) and advanced for the same amount of
simulated time per frame, so the panels stay in sync and you can watch where the
two methods agree and disagree.

* MPM (left)  -- hybrid: particles dump momentum onto a shared grid, the grid is
  solved, momentum is gathered back. Tends to look smooth and a little diffusive.
* SPH (right) -- purely particle-based: forces come from summing a smoothing
  kernel over each particle's neighbours. Tends to show sharper splashes and
  more surface detail (and a bit more pressure noise).

Run from the repo root:
    python examples/mpm_vs_sph.py                 # writes mpm_vs_sph.gif
    python examples/mpm_vs_sph.py --headless      # no render, prints stats
    python examples/mpm_vs_sph.py --frames 240
"""

import argparse
import os
import sys

import numpy as np
import taichi as ti

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swimmers import WATER, MATERIALS, MPMSolver, SPHSolver
from swimmers.scenes import fill_rect_lattice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--n-grid", type=int, default=128)
    parser.add_argument("--gravity", type=float, default=9.8)
    parser.add_argument("--out", default="mpm_vs_sph.gif")
    parser.add_argument("--headless", action="store_true",
                        help="run both solvers without rendering, print stats")
    args = parser.parse_args()

    ti.init(arch=ti.cpu)

    # --- one shared initial cloud: a dam-break water column ---------------
    # Sample on a lattice at the SPH particle spacing (dx/2) so SPH starts at
    # rest density instead of detonating from a random cloud; both solvers get
    # the identical cloud.
    spacing = 0.5 / args.n_grid
    pos, mat, vel = fill_rect_lattice((0.05, 0.05), (0.34, 0.62), WATER,
                                      spacing=spacing)
    n = pos.shape[0]

    # MPM with its usual settings.
    mpm = MPMSolver(n_particles=n, n_grid=args.n_grid, dt=1.0e-4,
                    gravity=args.gravity)
    mpm.load_particles(pos, mat, vel)
    mpm_dt, mpm_sub = mpm.dt, 40
    frame_time = mpm_dt * mpm_sub                       # simulated time per frame

    # SPH with a Courant-limited timestep; match the per-frame simulated time.
    sph = SPHSolver(n_particles=n, n_grid=args.n_grid, gravity=args.gravity)
    sph.load_particles(pos, mat, vel)
    sph_sub = max(1, round(frame_time / sph.dt))

    print(f"particles={n}  frame_time={frame_time:.2e}s  "
          f"MPM: {mpm_sub} substeps@{mpm_dt:.1e}  "
          f"SPH: {sph_sub} substeps@{sph.dt:.1e}")

    if args.headless:
        for f in range(args.frames):
            mpm.step(mpm_sub)
            sph.step(sph_sub)
            if f % 20 == 0 or f == args.frames - 1:
                xm = mpm.x.to_numpy()
                xs = sph.x.to_numpy()
                assert np.isfinite(xm).all() and np.isfinite(xs).all(), \
                    "NaN/Inf -- a solver blew up"
                print(f"frame {f:4d}  "
                      f"MPM y=[{xm[:,1].min():.3f},{xm[:,1].max():.3f}]  "
                      f"SPH y=[{xs[:,1].min():.3f},{xs[:,1].max():.3f}]")
        print("headless run OK (both solvers stayed finite)")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from swimmers.render import figure_to_rgb, save_gif

    bg = "#0E1B26"
    water = "#%06X" % MATERIALS[WATER].color
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=80, facecolor=bg)
    scats = []
    for ax, title in zip(axes, ("MPM", "SPH")):
        ax.set_facecolor(bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, color="white", fontsize=14, pad=8)
        scats.append(ax.scatter([], [], s=4, c=water, linewidths=0))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0.02)

    gif_frames = []
    for _ in range(args.frames):
        mpm.step(mpm_sub)
        sph.step(sph_sub)
        scats[0].set_offsets(mpm.x.to_numpy())
        scats[1].set_offsets(sph.x.to_numpy())
        gif_frames.append(figure_to_rgb(fig))
    plt.close(fig)
    out = save_gif(gif_frames, args.out, fps=30)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
