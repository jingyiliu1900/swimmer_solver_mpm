"""Showcase: a whole-body (non-particle) swimmer in a fluid-filled tank,
at three viscosities, side by side.

The swimmer is NOT made of particles. It is a single continuous solid -- an
analytic tapered strip whose midline undulates as a travelling wave. Because it
is defined by geometry, it can never fragment. It couples to the surrounding
fluid by direct forcing (the grid nodes it covers are driven to the body's
velocity), and its forward motion is *self-propelled*: it emerges from the
reaction of the fluid it pushes. Only the fluid viscosity differs between
panels, so you can see how a thicker medium changes the swimming.

The body is drawn as a filled shape; only the fluid is drawn as points.

Run from the repo root:
    python examples/whole_swimmer.py                 # -> whole_swimmer.gif
    python examples/whole_swimmer.py --frames 400 --density-div 3
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import numpy as np  # noqa: E402
import taichi as ti  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import swimmers.materials as materials  # noqa: E402
from swimmers import WATER, MPMSolver  # noqa: E402
from swimmers.render import figure_to_rgb, save_gif  # noqa: E402
from swimmers.scenes import DEFAULT_DENSITY, fill_rect, fish_halfwidth  # noqa: E402

ti.init(arch=ti.cpu)

# Stiffen the water (more incompressible) so it flows back in behind the body
# instead of leaving a cavitation void. Stable at DT below.
_w = materials.MATERIALS[WATER]
materials.MATERIALS[WATER] = materials.Material(
    name="water", youngs=8000.0, poisson=0.2, rho=1.0, color=_w.color)

# a wide viscosity sweep, water -> thick. Lower viscosities stay fully wetted;
# the thickest cases slow the body and show a small void near the fast tail.
FLUIDS = [
    (0.0, "viscosity 0.0", "#0B5563"),
    (0.3, "viscosity 0.3", "#274B6D"),
    (0.6, "viscosity 0.6", "#4A3A6B"),
    (0.9, "viscosity 0.9", "#6B3A5B"),
]
BODY_COLOR = "#7CFC9B"
DT = 1.5e-5
# slender-swimmer gait. A brisk-but-stable single travelling wave; the body is
# long and thin so it reads as a slender fish, rounded at both ends.
L, HALF_W, AMP0, N_WAVES, OMEGA = 0.40, 0.04, 0.09, 1.0, 45.0
BODY_K = 2.0 * np.pi * N_WAVES / L
START = (0.28, 0.5)   # start near the left so it can swim across


def body_shape(rx, t):
    """Slender body (mirror of MPMSolver._fish): smooth, rounded at both ends.
    Returns (centreline yc, half-thickness hw) for body-frame rx."""
    s = (rx + L / 2) / L                       # 0 = tail, 1 = head
    hw = fish_halfwidth(s, HALF_W)
    amp = AMP0 * (0.2 + 0.8 * (1.0 - s))
    yc = amp * np.sin(BODY_K * rx + OMEGA * t)
    return yc, hw


def build_tank(viscosity, density):
    rng = np.random.default_rng(0)
    cx, cy = START
    wb, wm, _ = fill_rect((0.03, 0.03), (0.94, 0.94), WATER, density=density, rng=rng)
    # carve out ONLY the body's thin silhouette at t=0 (not a big rectangle)
    rx, ry = wb[:, 0] - cx, wb[:, 1] - cy
    yc, hw = body_shape(rx, 0.0)
    inside = (np.abs(rx) < L / 2) & (np.abs(ry - yc) < hw)
    wb, wm = wb[~inside], wm[~inside]
    s = MPMSolver(n_particles=wb.shape[0], n_grid=128, dt=DT, gravity=0.0,
                  viscosity=viscosity)
    s.load_particles(wb, wm)
    s.configure_body(center=START, length=L, half_width=HALF_W, amp0=AMP0,
                     n_waves=N_WAVES, omega=OMEGA, density=1.0)
    return s


def body_polygon(s, nseg=56):
    """Outline of the analytic body in world space (top edge + bottom edge)."""
    cx, cy = s.body_pos.to_numpy()
    t = float(s.t[None])
    rx = np.linspace(-L / 2, L / 2, nseg)
    yc, hw = body_shape(rx, t)
    top = np.stack([cx + rx, cy + yc + hw], axis=1)
    bot = np.stack([cx + rx, cy + yc - hw], axis=1)
    return np.concatenate([top, bot[::-1]], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="whole_swimmer.gif")
    ap.add_argument("--frames", type=int, default=360)
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--density-div", type=int, default=3)
    args = ap.parse_args()

    density = DEFAULT_DENSITY // args.density_div
    solvers = [build_tank(eta, density) for eta, _, _ in FLUIDS]
    start = [s.body_pos.to_numpy().copy() for s in solvers]
    print(f"fluids={[f[0] for f in FLUIDS]}  water particles/each={solvers[0].n_particles}"
          f"  frames={args.frames}  dt={DT}  -> {args.out}")

    fig, axes = plt.subplots(1, len(FLUIDS), figsize=(4 * len(FLUIDS), 4),
                             dpi=args.dpi, facecolor="#0E1B26")
    water_sc, body_patch = [], []
    for ax, s, (eta, label, color) in zip(axes, solvers, FLUIDS):
        ax.set_facecolor("#0E1B26")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, color="white", fontsize=13)
        water_sc.append(ax.scatter(s.x.to_numpy()[:, 0], s.x.to_numpy()[:, 1],
                                   s=2, c=color, linewidths=0, alpha=0.45))
        patch = Polygon(body_polygon(s), closed=True, facecolor=BODY_COLOR,
                        edgecolor="white", linewidth=0.6, zorder=5)
        ax.add_patch(patch)
        body_patch.append(patch)
    fig.tight_layout()

    frames = []
    for f in range(args.frames):
        for s in solvers:
            s.step(args.substeps)
        for ws, bp, s in zip(water_sc, body_patch, solvers):
            ws.set_offsets(s.x.to_numpy())
            bp.set_xy(body_polygon(s))
        frames.append(figure_to_rgb(fig))
        if f % 40 == 0:
            print(f"  frame {f}/{args.frames}")
    plt.close(fig)

    save_gif(frames, args.out, fps=args.fps)

    for s, p0, (eta, label, _) in zip(solvers, start, FLUIDS):
        d = s.body_pos.to_numpy() - p0
        print(f"  {label}: body travelled ({d[0]:+.3f}, {d[1]:+.3f})  |d|={np.hypot(*d):.3f}")
    print(f"wrote {args.out}  ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()