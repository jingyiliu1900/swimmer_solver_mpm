"""swimmers: a 2D multi-material particle simulator you drive with shapes.

Describe a scene with :class:`Simulation` -- add a pool of water and drop in any
polygon you like as a deformable (``elastic``), ``rigid``, or self-propelling
``swimmer`` body -- then render an animated GIF or open a live window:

    from swimmers import Simulation
    from swimmers.scenes import star

    sim = Simulation(gravity=9.8)
    sim.add_fluid(size=(0.96, 0.30))                       # a shallow pool
    sim.add_body(star((0.5, 0.7), 0.12, 0.05), "elastic")  # a soft star
    sim.render_gif("star.gif", frames=200)

The engine is MLS-MPM (Moving Least Squares Material Point Method). Every
particle, whatever its material, transfers momentum to one shared background
grid; the grid is solved and momentum is transferred back. Fluid<->solid
coupling and large deformation are therefore automatic -- properties of the
method, not special-cased code.
"""

from .materials import WATER, ELASTIC, RIGID, SWIMMER, Material, MATERIALS
from .solver import MPMSolver
from .sph_solver import SPHSolver
from .sim import Simulation

__all__ = [
    "Simulation",
    "MPMSolver",
    "SPHSolver",
    "WATER", "ELASTIC", "RIGID", "SWIMMER", "Material", "MATERIALS",
]
