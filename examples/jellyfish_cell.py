"""Showcase: a pulsatile jellyfish cell that is ONE whole deformable body.

Unlike a particle blob (which can shed pieces), this cell is a single analytic
bell -- an upper half-annulus dome, open at the bottom -- so it can NEVER
fragment. It pulses by a prescribed radial contraction (rectified: contract,
then relax), jetting fluid out of its opening; the fluid reaction self-propels
it, exactly like the whole-body fish. It is drawn as one filled shape.

'Softness'/deformability is the pulse amplitude (how far the bell contracts):
  * soft (amp=0.50): big pulses, deforms and swims the most
  * medium (amp=0.35)
  * stiff (amp=0.20): small pulses, barely moves

Run from the repo root:
    python examples/jellyfish_cell.py                 # -> jellyfish.gif
    python examples/jellyfish_cell.py --frames 480 --density-div 2
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
from swimmers.scenes import DEFAULT_DENSITY, fill_rect  # noqa: E402

ti.init(arch=ti.cpu)

_w = materials.MATERIALS[WATER]
materials.MATERIALS[WATER] = materials.Material(
    name="water", youngs=8000.0, poisson=0.2, rho=1.0, color=_w.color)

CELLS = [  # (pulse amplitude = deformability, label, colour)
    (0.45, "soft  (amp=0.45)", "#FF6FA5"),
    (0.30, "medium (amp=0.30)", "#C75BAC"),
    (0.16, "stiff (amp=0.16)", "#7E4FB0"),
]
WATER_COLOR = "#16384A"
RIN, ROUT = 0.05, 0.105
START = (0.5, 0.30)   # start low; the bell jets downward and rises
DT = 1.5e-5
OMEGA = 60.0


def bell_scales(amp, t):
    """Anisotropic deformation factors (sx horizontal, sy vertical).

    Mirrors MPMSolver's bell: squeeze horizontally, stretch vertically, area
    preserved -- the bell deforms (prolate <-> oblate) instead of shrinking.
    """
    sx = 1.0 - amp * max(0.0, np.sin(OMEGA * t))
    return sx, 1.0 / sx


def build(amp, density):
    rng = np.random.default_rng(0)
    cx, cy = START
    wb, wm, _ = fill_rect((0.03, 0.03), (0.94, 0.94), WATER, density=density, rng=rng)
    dx, dy = wb[:, 0] - cx, wb[:, 1] - cy
    rho, ang = np.hypot(dx, dy), np.arctan2(dy, dx)
    inside = (ang > 0) & (ang < np.pi) & (rho > RIN) & (rho < ROUT)
    wb, wm = wb[~inside], wm[~inside]
    s = MPMSolver(n_particles=len(wb), n_grid=128, dt=DT, gravity=0.0, viscosity=0.0)
    s.load_particles(wb, wm)
    s.configure_bell(center=START, r_inner=RIN, r_outer=ROUT,
                     amp=amp, omega=OMEGA, density=1.0)
    return s


def bell_polygon(s, amp, nseg=48):
    """Filled outline of the deformable bell (exactly the physics shape)."""
    cx, cy = s.body_pos.to_numpy()
    sx, sy = bell_scales(amp, float(s.t[None]))
    th = np.linspace(0.0, np.pi, nseg)
    outer = np.stack([cx + sx * ROUT * np.cos(th), cy + sy * ROUT * np.sin(th)], 1)
    inner = np.stack([cx + sx * RIN * np.cos(th), cy + sy * RIN * np.sin(th)], 1)
    return np.concatenate([outer, inner[::-1]], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="jellyfish.gif")
    ap.add_argument("--frames", type=int, default=420)
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--density-div", type=int, default=3)
    args = ap.parse_args()

    density = DEFAULT_DENSITY // args.density_div
    solvers = [build(amp, density) for amp, _, _ in CELLS]
    print(f"amps={[c[0] for c in CELLS]}  water particles/each={solvers[0].n_particles}"
          f"  frames={args.frames}  dt={DT}  -> {args.out}")

    fig, axes = plt.subplots(1, len(CELLS), figsize=(4 * len(CELLS), 4),
                             dpi=args.dpi, facecolor="#0E1B26")
    water_sc, patches = [], []
    for ax, s, (amp, label, color) in zip(axes, solvers, CELLS):
        ax.set_facecolor("#0E1B26")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, color="white", fontsize=13)
        water_sc.append(ax.scatter(s.x.to_numpy()[:, 0], s.x.to_numpy()[:, 1],
                                   s=2, c=WATER_COLOR, linewidths=0))
        patch = Polygon(bell_polygon(s, amp), closed=True, facecolor=color,
                        edgecolor="white", linewidth=0.6, zorder=5)
        ax.add_patch(patch)
        patches.append(patch)
    fig.tight_layout()

    frames = []
    for f in range(args.frames):
        for s in solvers:
            s.step(args.substeps)
        for ws, pa, s, (amp, _, _) in zip(water_sc, patches, solvers, CELLS):
            ws.set_offsets(s.x.to_numpy())
            pa.set_xy(bell_polygon(s, amp))
        frames.append(figure_to_rgb(fig))
        if f % 40 == 0:
            print(f"  frame {f}/{args.frames}")
    plt.close(fig)

    save_gif(frames, args.out, fps=args.fps)
    for s, (amp, label, _) in zip(solvers, CELLS):
        d = s.body_pos.to_numpy()[1] - START[1]
        print(f"  {label}: rose {d:+.3f}")
    print(f"wrote {args.out}  ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()