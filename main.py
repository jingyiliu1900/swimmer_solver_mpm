"""Run the 2D fluid <-> deformable-solid simulation with a live GUI.

Usage:
    python main.py                 # default 'splash' scene
    python main.py --scene dam_break
    python main.py --scene splash --arch gpu --headless --frames 300

Controls (windowed mode):
    r   reset the current scene
    q / Esc   quit
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import taichi as ti

# allow running straight from the repo without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from swimmers.materials import MATERIALS  # noqa: E402
from swimmers.scenes import SCENES  # noqa: E402
from swimmers.solver import MPMSolver  # noqa: E402


def build(scene_name: str, n_grid: int, dt: float, gravity: float,
          viscosity: float = 0.0):
    pos, mat, vel = SCENES[scene_name]()
    solver = MPMSolver(n_particles=pos.shape[0], n_grid=n_grid, dt=dt,
                       gravity=gravity, viscosity=viscosity)
    solver.load_particles(pos, mat, vel)
    return solver, pos, mat, vel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="splash", choices=list(SCENES))
    parser.add_argument("--arch", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--n-grid", type=int, default=128)
    parser.add_argument("--dt", type=float, default=1.0e-4)
    parser.add_argument("--substeps", type=int, default=40)
    parser.add_argument("--gravity", type=float, default=9.8,
                        help="downward acceleration; use 0 for no gravity")
    parser.add_argument("--viscosity", type=float, default=0.0,
                        help="fluid viscosity; 0=water, ~0.5=honey "
                             "(needs --dt 3e-5 for viscosity>0.1)")
    parser.add_argument("--res", type=int, default=720, help="GUI resolution")
    parser.add_argument("--headless", action="store_true",
                        help="run without a window (for CI / verification)")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 = run forever)")
    args = parser.parse_args()

    ti.init(arch=ti.gpu if args.arch == "gpu" else ti.cpu)

    solver, pos, mat, vel = build(args.scene, args.n_grid, args.dt, args.gravity,
                                  args.viscosity)
    colors = np.array([m.color for m in MATERIALS], dtype=np.uint32)

    print(f"scene={args.scene}  particles={solver.n_particles}  "
          f"grid={args.n_grid}  dt={args.dt}  substeps={args.substeps}")

    if args.headless:
        run_headless(solver, args.frames or 200)
        return

    gui = ti.GUI("swimmers - fluid x deformable solid",
                 res=(args.res, args.res), background_color=0x112F41)
    frame = 0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in (ti.GUI.ESCAPE, "q"):
                gui.running = False
            elif e.key == "r":
                solver.load_particles(pos, mat, vel)

        solver.step(args.substeps)

        x = solver.x.to_numpy()
        m = solver.material.to_numpy()
        gui.circles(x, radius=1.6, palette=colors, palette_indices=m)
        gui.show()

        frame += 1
        if args.frames and frame >= args.frames:
            break


def run_headless(solver: MPMSolver, frames: int) -> None:
    """Advance the sim with no window and print simple sanity stats."""
    for f in range(frames):
        solver.step(40)
        if f % 20 == 0 or f == frames - 1:
            x = solver.x.to_numpy()
            v = solver.v.to_numpy()
            speed = float(np.linalg.norm(v, axis=1).mean())
            assert np.isfinite(x).all(), "NaN/Inf in positions -- simulation blew up"
            assert (x >= -0.01).all() and (x <= 1.01).all(), "particles left domain"
            print(f"frame {f:4d}  mean|v|={speed:7.4f}  "
                  f"y_range=[{x[:,1].min():.3f}, {x[:,1].max():.3f}]")
    print("headless run OK (no NaNs, particles stayed in domain)")


if __name__ == "__main__":
    main()
