"""Bring your own shape: drop arbitrary polygons into a pool and watch them
splash and deform.

This is the "any shape or deformed object" demo. Each body is just a polygon
outline you supply -- a star, a hexagon, an arrow, your own list of (x, y)
points -- filled with deformable (elastic) material. They fall into a pool of
water, splash, jiggle, and settle, all through the same MPM coupling.

Run from the repo root:
    python examples/custom_shape.py                 # -> custom_shape.gif
    python examples/custom_shape.py --frames 300 --out shapes.gif
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from swimmers import Simulation  # noqa: E402
from swimmers.scenes import regular_polygon, star  # noqa: E402

# An arbitrary hand-written outline (a chunky down-arrow), to prove that ANY
# polygon works -- not just the built-in shape helpers. Placed on the right.
ARROW = np.array([
    (0.72, 0.92), (0.84, 0.92), (0.84, 0.80), (0.94, 0.80),
    (0.78, 0.66), (0.62, 0.80), (0.72, 0.80),
])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="custom_shape.gif")
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--substeps", type=int, default=40)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=70)
    ap.add_argument("--density-div", type=int, default=3)
    args = ap.parse_args()

    from swimmers.scenes import DEFAULT_DENSITY
    sim = Simulation(gravity=9.8, dt=4e-5, density=DEFAULT_DENSITY // args.density_div)
    sim.add_fluid(lower=(0.02, 0.02), size=(0.96, 0.24))            # a pool
    sim.add_body(star((0.22, 0.80), 0.10, 0.045, points=5), "elastic")
    sim.add_body(regular_polygon((0.50, 0.82), 0.09, sides=6), "elastic")
    sim.add_body(ARROW, "elastic")

    print(f"particles={sim.positions.shape[0]}  frames={args.frames}  -> {args.out}")
    out = sim.render_gif(args.out, frames=args.frames, substeps=args.substeps,
                         fps=args.fps, dpi=args.dpi, point_size=3.0)
    print(f"wrote {out}  ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()
