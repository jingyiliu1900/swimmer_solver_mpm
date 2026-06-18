"""Material definitions.

Each material is just a small bundle of constitutive parameters plus a display
colour. The solver reads these per particle, so adding a new material (snow,
sand, near-rigid, ...) is a matter of appending an entry here and handling its
branch in ``solver.substep``.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- material ids (used as integer flags on the GPU) ------------------------
WATER = 0
ELASTIC = 1
RIGID = 2
SWIMMER = 3


@dataclass(frozen=True)
class Material:
    """Constitutive parameters for one material.

    Attributes
    ----------
    name:
        Human readable label.
    youngs:
        Young's modulus E. Stiffness. For water this acts as the bulk modulus
        of the pressure equation of state.
    poisson:
        Poisson ratio nu. Ignored for water (it carries no shear).
    rho:
        Rest density (mass per unit volume).
    color:
        0xRRGGBB colour for the GUI.
    """

    name: str
    youngs: float
    poisson: float
    rho: float
    color: int

    @property
    def mu(self) -> float:
        """Shear modulus (Lame's second parameter)."""
        return self.youngs / (2.0 * (1.0 + self.poisson))

    @property
    def lam(self) -> float:
        """Lame's first parameter."""
        return (
            self.youngs
            * self.poisson
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )


# Registry indexed by material id. Order must match the id constants above.
MATERIALS = [
    # Water: soft bulk stiffness, the solver zeroes its shear modulus so it
    # flows. A higher `youngs` makes it less compressible (and needs smaller dt).
    Material(name="water", youngs=400.0, poisson=0.2, rho=1.0, color=0x068587),
    # Elastic jelly: stiff enough to hold its shape, soft enough to wobble and
    # get deformed by the water it lands in / displaces.
    Material(name="elastic", youngs=2000.0, poisson=0.3, rho=1.0, color=0xED553B),
    # Near-rigid object: same constitutive model as elastic but very stiff, so
    # it barely deforms and behaves like a solid body. True rigidity is not
    # achievable in pure MPM; "very stiff" is the standard stand-in. Higher
    # `youngs` needs a smaller `dt` to stay stable -- see examples/rigid_drop.py.
    # A denser body (rho > 1) also sinks instead of floating.
    Material(name="rigid", youngs=30000.0, poisson=0.3, rho=2.0, color=0xF2C14E),
    # Active swimmer: an elastic body that additionally carries a time-varying
    # "muscle" stress along its fibers (see MPMSolver active-stress branch).
    # Soft enough to bend into a travelling wave, neutrally buoyant (rho = water)
    # so it neither sinks nor floats while swimming. See examples/swimmer_*.py.
    Material(name="swimmer", youngs=1500.0, poisson=0.3, rho=1.0, color=0x6BCB77),
]
