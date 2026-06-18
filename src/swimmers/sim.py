"""High-level API: assemble a scene from shapes, then run or render it.

This is the friendly front door to the simulator. You describe a scene by adding
fluid and solid bodies -- a body can be *any* polygon you supply -- pick how
each body should behave (deformable ``elastic`` by default, or ``rigid`` /
``swimmer``), then either render an animated GIF or open a live window.

    from swimmers import Simulation
    from swimmers.scenes import star

    sim = Simulation(gravity=9.8)
    sim.add_fluid(lower=(0.02, 0.02), size=(0.96, 0.30))   # a pool
    sim.add_body(star((0.5, 0.7), 0.12, 0.05, points=5))   # drop a star in
    sim.render_gif("star_splash.gif", frames=200)

Everything underneath is MLS-MPM, so fluid<->solid coupling and large
deformation come for free; you only describe shapes and materials.
"""

from __future__ import annotations

import numpy as np
import taichi as ti

from .materials import ELASTIC, MATERIALS, RIGID, SWIMMER, WATER
from .scenes import DEFAULT_DENSITY, fill_polygon, fill_rect
from .solver import MPMSolver

_NAME_TO_ID = {"water": WATER, "elastic": ELASTIC, "rigid": RIGID, "swimmer": SWIMMER}


def _material_id(material) -> int:
    if isinstance(material, str):
        try:
            return _NAME_TO_ID[material.lower()]
        except KeyError:
            raise ValueError(
                f"unknown material {material!r}; choose from {list(_NAME_TO_ID)}"
            ) from None
    return int(material)


class Simulation:
    """Build a 2D MPM scene from shapes, then ``render_gif`` or ``show`` it."""

    def __init__(
        self,
        *,
        n_grid: int = 128,
        dt: float = 1.0e-4,
        gravity: float = 9.8,
        viscosity: float = 0.0,
        act_strength: float = 500.0,
        swim_omega: float = 40.0,
        density: float | None = None,
        arch: str = "cpu",
    ):
        ti.init(arch=ti.gpu if arch == "gpu" else ti.cpu)
        self.n_grid = n_grid
        self.dt = dt
        self.gravity = gravity
        self.viscosity = viscosity
        self.act_strength = act_strength
        self.swim_omega = swim_omega
        self.density = DEFAULT_DENSITY if density is None else density
        self._pos: list[np.ndarray] = []
        self._mat: list[np.ndarray] = []
        self._vel: list[np.ndarray] = []
        self._amp: list[np.ndarray] = []
        self._phase: list[np.ndarray] = []
        self.solver: MPMSolver | None = None
        self.material: np.ndarray | None = None

    # -- scene assembly ----------------------------------------------------
    def _add(self, pos, mat, vel, amp=None, phase=None) -> "Simulation":
        n = len(pos)
        self._pos.append(pos)
        self._mat.append(mat)
        self._vel.append(vel)
        self._amp.append(np.zeros(n) if amp is None else amp)
        self._phase.append(np.zeros(n) if phase is None else phase)
        self.solver = None  # scene changed; force a rebuild
        return self

    def add_fluid(self, lower=(0.02, 0.02), size=(0.96, 0.50),
                  velocity=(0.0, 0.0), rng=None) -> "Simulation":
        """Add a rectangular pool/column of water."""
        return self._add(*fill_rect(lower, size, WATER, velocity, self.density, rng))

    def add_body(self, shape, material="elastic", velocity=(0.0, 0.0),
                 swim=False, n_waves=1.0, rng=None) -> "Simulation":
        """Add a solid body filling *any* polygon ``shape`` (an (N,2) outline).

        ``material`` is ``"elastic"`` (deformable, default), ``"rigid"``, or
        ``"swimmer"``. Pass ``swim=True`` to give a body an undulatory muscle
        (a travelling wave head->tail), which also makes it a swimmer.
        """
        mid = _material_id(material)
        pos, mat, vel = fill_polygon(np.asarray(shape, dtype=float), mid,
                                     velocity, self.density, rng)
        amp = phase = None
        if swim or mid == SWIMMER:
            mat[:] = SWIMMER
            cx, cy = pos[:, 0], pos[:, 1]
            span = max(1e-9, cx.max() - cx.min())
            amp = np.where(cy > cy.mean(), 1.0, -1.0)      # dorsal/ventral sign
            phase = 2.0 * np.pi * n_waves * (cx - cx.min()) / span  # head->tail
        return self._add(pos, mat, vel, amp, phase)

    # -- build & step ------------------------------------------------------
    def build(self) -> MPMSolver:
        """Materialise the accumulated shapes into an :class:`MPMSolver`."""
        if not self._pos:
            raise RuntimeError("nothing to simulate -- add_fluid()/add_body() first")
        pos = np.concatenate(self._pos)
        mat = np.concatenate(self._mat)
        vel = np.concatenate(self._vel)
        s = MPMSolver(n_particles=len(pos), n_grid=self.n_grid, dt=self.dt,
                      gravity=self.gravity, viscosity=self.viscosity,
                      act_strength=self.act_strength, swim_omega=self.swim_omega)
        s.load_particles(pos, mat, vel)
        if (mat == SWIMMER).any():
            s.set_actuation(np.concatenate(self._amp), np.concatenate(self._phase))
        self.solver = s
        self.material = mat
        return s

    def step(self, substeps: int = 40) -> None:
        if self.solver is None:
            self.build()
        self.solver.step(substeps)

    @property
    def positions(self) -> np.ndarray:
        if self.solver is None:
            self.build()
        return self.solver.x.to_numpy()

    # -- output ------------------------------------------------------------
    def _palette(self):
        return ["#%06X" % m.color for m in MATERIALS]

    def render_gif(self, path: str = "out.gif", frames: int = 200,
                   substeps: int = 40, fps: int = 30, dpi: int = 80,
                   point_size: float = 3.0, bg: str = "#0E1B26") -> str:
        """Run the scene and write an animated GIF (embeds in a README)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .render import figure_to_rgb, save_gif

        if self.solver is None:
            self.build()
        colors = np.array(self._palette(), dtype=object)[self.material]

        fig = plt.figure(figsize=(5, 5), dpi=dpi, facecolor=bg)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(bg)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        scat = ax.scatter(self.positions[:, 0], self.positions[:, 1],
                          s=point_size, c=colors, linewidths=0)

        gif_frames = []
        for _ in range(frames):
            self.solver.step(substeps)
            scat.set_offsets(self.solver.x.to_numpy())
            gif_frames.append(figure_to_rgb(fig))
        plt.close(fig)
        return save_gif(gif_frames, path, fps=fps)

    def show(self, substeps: int = 40, res: int = 720) -> None:
        """Open a live Taichi window (press q/Esc to quit)."""
        if self.solver is None:
            self.build()
        colors = np.array([m.color for m in MATERIALS], dtype=np.uint32)
        gui = ti.GUI("swimmers", res=(res, res), background_color=0x0E1B26)
        while gui.running:
            for e in gui.get_events(ti.GUI.PRESS):
                if e.key in (ti.GUI.ESCAPE, "q"):
                    gui.running = False
            self.solver.step(substeps)
            gui.circles(self.solver.x.to_numpy(), radius=2.0, palette=colors,
                        palette_indices=self.solver.material.to_numpy())
            gui.show()
