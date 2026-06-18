"""Hero demo: a single fish swimming across a water-filled tank.

The fish is a whole-body (non-particle) swimmer -- one continuous analytic
solid with a blunt head, a tapered body, and a caudal fin. Its midline runs a
travelling bending wave (head -> tail), and it couples to the surrounding water
by direct forcing. Its forward glide is *self-propelled*: it emerges from the
reaction of the water the tail pushes back, not from any prescribed motion.

The fish is drawn as a filled shape; the water is drawn as points. Output is an
animated GIF so it embeds straight into the README.

Run from the repo root:
    python examples/fish_swim.py                  # -> fish.gif
    python examples/fish_swim.py --frames 320 --out fish.gif
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

# Stiffen the water (more incompressible) so it closes in behind the fish
# instead of leaving a cavitation void. Stable at the dt below.
_w = materials.MATERIALS[WATER]
materials.MATERIALS[WATER] = materials.Material(
    name="water", youngs=8000.0, poisson=0.2, rho=1.0, color=_w.color)

WATER_COLOR = "#0B5563"
BODY_COLOR = "#7CFC9B"
DT = 1.5e-5
# A brisk-but-stable gait: one travelling wave, peak amplitude at the tail.
# The body is long and thin so it reads as a slender fish, rounded at both ends.
L, HALF_W, AMP0, N_WAVES, OMEGA = 0.40, 0.04, 0.10, 1.0, 60.0
BODY_K = 2.0 * np.pi * N_WAVES / L
START = (0.24, 0.5)   # start near the left so it can swim across to the right


def body_shape(rx, t):
    """Slender body centreline yc and half-thickness hw for body coord rx."""
    s = (rx + L / 2) / L
    hw = fish_halfwidth(s, HALF_W)
    amp = AMP0 * (0.2 + 0.8 * (1.0 - s))          # undulation grows toward the tail
    yc = amp * np.sin(BODY_K * rx + OMEGA * t)
    return yc, hw


def build_tank(density):
    rng = np.random.default_rng(0)
    cx, cy = START
    wb, wm, _ = fill_rect((0.03, 0.03), (0.94, 0.94), WATER, density=density, rng=rng)
    # carve out the fish silhouette at t=0
    rx, ry = wb[:, 0] - cx, wb[:, 1] - cy
    yc, hw = body_shape(rx, 0.0)
    inside = (np.abs(rx) < L / 2) & (np.abs(ry - yc) < hw)
    wb, wm = wb[~inside], wm[~inside]
    s = MPMSolver(n_particles=wb.shape[0], n_grid=128, dt=DT, gravity=0.0)
    s.load_particles(wb, wm)
    s.configure_body(center=START, length=L, half_width=HALF_W, amp0=AMP0,
                     n_waves=N_WAVES, omega=OMEGA, density=1.0)
    return s


def body_polygon(s, nseg=64):
    """Outline of the fish in world space (top edge then bottom edge)."""
    cx, cy = s.body_pos.to_numpy()
    t = float(s.t[None])
    rx = np.linspace(-L / 2, L / 2, nseg)
    yc, hw = body_shape(rx, t)
    top = np.stack([cx + rx, cy + yc + hw], axis=1)
    bot = np.stack([cx + rx, cy + yc - hw], axis=1)
    return np.concatenate([top, bot[::-1]], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="fish.gif")
    ap.add_argument("--frames", type=int, default=260)   # long swim, compact file
    ap.add_argument("--substeps", type=int, default=90)  # more sim-time per frame
    ap.add_argument("--fps", type=int, default=40)       # snappier playback
    ap.add_argument("--dpi", type=int, default=64)
    ap.add_argument("--density-div", type=int, default=3)
    args = ap.parse_args()

    s = build_tank(DEFAULT_DENSITY // args.density_div)
    p0 = s.body_pos.to_numpy().copy()
    print(f"water particles={s.n_particles}  frames={args.frames}  dt={DT}  -> {args.out}")

    fig = plt.figure(figsize=(5, 5), dpi=args.dpi, facecolor="#0E1B26")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0E1B26")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    water_sc = ax.scatter(s.x.to_numpy()[:, 0], s.x.to_numpy()[:, 1],
                          s=2, c=WATER_COLOR, linewidths=0)
    patch = Polygon(body_polygon(s), closed=True, facecolor=BODY_COLOR,
                    edgecolor="white", linewidth=0.8, zorder=5)
    ax.add_patch(patch)

    frames = []
    for f in range(args.frames):
        s.step(args.substeps)
        water_sc.set_offsets(s.x.to_numpy())
        patch.set_xy(body_polygon(s))
        frames.append(figure_to_rgb(fig))
        if f % 40 == 0:
            print(f"  frame {f}/{args.frames}")
    plt.close(fig)

    save_gif(frames, args.out, fps=args.fps)
    d = s.body_pos.to_numpy() - p0
    print(f"fish swam ({d[0]:+.3f}, {d[1]:+.3f})  |d|={np.hypot(*d):.3f}  "
          f"wrote {args.out}  ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
