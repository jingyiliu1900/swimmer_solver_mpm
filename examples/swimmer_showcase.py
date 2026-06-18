"""Showcase: a fish swimming inside a fully fluid-filled tank, at three
different viscosities, side by side.

Each panel is the SAME tank (domain completely filled with fluid) containing
one identical undulating fish. Only the fluid viscosity differs:
  * viscosity 0.0  -> water-like : the fish undulates and glides freely
  * viscosity 0.4  -> syrup-like : strokes damp out, progress is reduced
  * viscosity 0.8  -> honey-like : the dense medium resists the body strongly

The fish is an elastic body (blunt head, tapered tail, caudal fin) with a
muscle: an active stress along its long axis whose sign flips across its
thickness (dorsal vs ventral) and whose phase travels head -> tail, producing
an undulatory (anguilliform) gait.

Renders a side-by-side animated GIF (no display needed) via matplotlib + Pillow,
so it embeds straight into the README.

Run from the repo root:
    python examples/swimmer_showcase.py                 # -> swimmer.gif
    python examples/swimmer_showcase.py --frames 300 --density-div 3
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

from swimmers import SWIMMER, WATER, MPMSolver  # noqa: E402
from swimmers.render import figure_to_rgb, save_gif  # noqa: E402
from swimmers.scenes import DEFAULT_DENSITY, fill_fish, fill_rect, fish_inside  # noqa: E402

ti.init(arch=ti.cpu)

# (viscosity, label, water colour)
FLUIDS = [
    (0.0, "viscosity 0.0  (water)", "#068587"),
    (0.4, "viscosity 0.4  (syrup)", "#3A7CA5"),
    (0.8, "viscosity 0.8  (honey)", "#C77DA0"),
]
SWIMMER_COLOR = "#6BCB77"
DT = 1.5e-5         # small dt: stable for the thin tail + muscle stress (inviscid)
ACT_STRENGTH = 500.0
SWIM_OMEGA = 40.0
N_WAVES = 1.0


def build_tank(viscosity, density, center=(0.5, 0.5), length=0.38, width=0.03):
    """A domain fully filled with fluid, with one fish swimmer inside."""
    rng = np.random.default_rng(0)
    cx, cy = center
    x0 = cx - length / 2

    # fish body + its actuation (dorsal/ventral sign, travelling phase head->tail)
    sb, sm, _ = fill_fish(center, length, width, SWIMMER, density=density, rng=rng)
    amp = np.where(sb[:, 1] > cy, 1.0, -1.0)
    phase = 2.0 * np.pi * N_WAVES * (sb[:, 0] - x0) / length

    # fill the whole domain with water, then remove particles inside the fish
    wb, wm, _ = fill_rect((0.03, 0.03), (0.94, 0.94), WATER, density=density, rng=rng)
    inside = fish_inside(wb, center, length, width)
    wb, wm = wb[~inside], wm[~inside]

    pos = np.concatenate([wb, sb])
    mat = np.concatenate([wm, sm])
    swim_amp = np.concatenate([np.zeros(len(wb)), amp])
    swim_phase = np.concatenate([np.zeros(len(wb)), phase])

    s = MPMSolver(n_particles=pos.shape[0], n_grid=128, dt=DT, gravity=0.0,
                  viscosity=viscosity, act_strength=ACT_STRENGTH, swim_omega=SWIM_OMEGA)
    s.load_particles(pos, mat)
    s.set_actuation(swim_amp, swim_phase)
    return s, (mat == SWIMMER)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="swimmer.gif")
    ap.add_argument("--frames", type=int, default=280)
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--density-div", type=int, default=3,
                    help="divide particle density by this (higher = faster/coarser)")
    args = ap.parse_args()

    density = DEFAULT_DENSITY // args.density_div
    solvers, bodies, start_com = [], [], []
    for eta, _, _ in FLUIDS:
        s, body = build_tank(eta, density)
        solvers.append(s); bodies.append(body)
        start_com.append(s.x.to_numpy()[body].mean(axis=0))

    print(f"fluids={[f[0] for f in FLUIDS]}  particles/each={solvers[0].n_particles}"
          f"  frames={args.frames}  dt={DT}  -> {args.out}")

    fig, axes = plt.subplots(1, len(FLUIDS), figsize=(4 * len(FLUIDS), 4),
                             dpi=args.dpi, facecolor="#0E1B26")
    water_sc, body_sc = [], []
    for ax, s, body, (eta, label, color) in zip(axes, solvers, bodies, FLUIDS):
        ax.set_facecolor("#0E1B26")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, color="white", fontsize=13)
        x = s.x.to_numpy()
        water_sc.append(ax.scatter(x[~body][:, 0], x[~body][:, 1], s=2,
                                   c=color, linewidths=0, alpha=0.5))
        body_sc.append(ax.scatter(x[body][:, 0], x[body][:, 1], s=6,
                                  c=SWIMMER_COLOR, linewidths=0))
    fig.tight_layout()

    frames = []
    for f in range(args.frames):
        for s in solvers:
            s.step(args.substeps)
        for ws, bs, s, body in zip(water_sc, body_sc, solvers, bodies):
            x = s.x.to_numpy()
            ws.set_offsets(x[~body]); bs.set_offsets(x[body])
        frames.append(figure_to_rgb(fig))
        if f % 40 == 0:
            print(f"  frame {f}/{args.frames}")
    plt.close(fig)

    save_gif(frames, args.out, fps=args.fps)

    for s, body, c0, (eta, label, _) in zip(solvers, bodies, start_com, FLUIDS):
        c = s.x.to_numpy()[body].mean(axis=0)
        d = c - c0
        print(f"  {label}: swimmer net displacement = "
              f"({d[0]:+.3f}, {d[1]:+.3f})  |d|={np.hypot(*d):.3f}")
    print(f"wrote {args.out}  ({os.path.getsize(args.out)//1024} KB)")


if __name__ == "__main__":
    main()
