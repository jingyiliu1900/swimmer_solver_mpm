"""Weakly-compressible SPH (WCSPH) 2D fluid solver -- a foil for the MPM solver.

Where :class:`~swimmers.solver.MPMSolver` scatters momentum onto a shared
background grid (a hybrid Eulerian/Lagrangian method), SPH is *purely*
Lagrangian: every quantity -- density, pressure, the forces between particles --
is evaluated by summing a smoothing kernel over each particle's neighbours. No
grid stores physics; the grid here is only a bookkeeping device for finding
neighbours quickly.

The model is classic WCSPH (Monaghan 1994):

* density by summation        ``rho_i = sum_j m_j W(|x_i - x_j|, h)``
* pressure by a Tait EoS      ``p = B((rho/rho0)^gamma - 1)``, clamped to >= 0
* a symmetric pressure force  ``-sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2) grad W``
* Monaghan artificial viscosity for stability
* gravity + reflecting walls on the unit square

It is deliberately the same shape as :class:`MPMSolver` (``x``/``v``/
``material`` fields, ``load_particles``/``step``) so the two can be dropped into
the same render loop and compared frame-for-frame. SPH here is a *fluid* solver:
all particles are simulated as weakly-compressible water regardless of their
``material`` id (the id is carried through only so the renderer can colour them).
See ``examples/mpm_vs_sph.py`` for a side-by-side run.
"""

from __future__ import annotations

import math

import numpy as np
import taichi as ti

from .materials import MATERIALS, WATER


@ti.data_oriented
class SPHSolver:
    def __init__(
        self,
        n_particles: int,
        n_grid: int = 128,
        dt: float | None = None,
        gravity: float = 9.8,
        rho0: float | None = None,
        sound_speed: float = 40.0,
        gamma: float = 7.0,
        viscosity: float = 0.1,
        h_factor: float = 3.0,
    ):
        self.n_particles = n_particles
        self.gravity = gravity

        # Particle spacing is tied to the MPM grid so both solvers start from the
        # SAME cloud at the SAME density: MPM samples ~4 particles per cell, i.e.
        # one particle every dx/2. mass = rho0 * (area per particle).
        dx = 1.0 / n_grid
        self.spacing = 0.5 * dx
        self.rho0 = MATERIALS[WATER].rho if rho0 is None else float(rho0)
        self.mass = self.rho0 * self.spacing * self.spacing

        # Smoothing length (kernel support radius). h_factor * spacing keeps a
        # few dozen neighbours inside the support -- enough for a smooth field.
        self.h = h_factor * self.spacing
        self.inv_h = 1.0 / self.h
        # 2D cubic-spline normalisation, support radius == h.
        self.kernel_norm = 40.0 / (7.0 * math.pi * self.h * self.h)

        # Tait equation-of-state stiffness. A larger sound speed -> stiffer,
        # less compressible fluid (and a smaller stable timestep).
        self.sound_speed = sound_speed
        self.gamma = gamma
        self.B = self.rho0 * sound_speed * sound_speed / gamma
        # Monaghan artificial-viscosity coefficient (dimensionless).
        self.visc_alpha = viscosity

        # Courant-limited timestep unless the caller pins one explicitly.
        self.dt = 0.25 * self.h / sound_speed if dt is None else float(dt)

        # Reflecting walls sit one spacing in from the domain edge.
        self.pad = 2.0 * self.spacing

        # --- per-particle state -------------------------------------------
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
        self.rho = ti.field(dtype=ti.f32, shape=n_particles)
        self.pressure = ti.field(dtype=ti.f32, shape=n_particles)
        self.material = ti.field(dtype=ti.i32, shape=n_particles)

        # --- uniform grid for neighbour search ----------------------------
        # Cell size >= support radius so each particle only has to scan the 3x3
        # block of cells around its own to find every neighbour within h.
        self.grid_size = max(1, int(1.0 / self.h))
        self.cell_size = 1.0 / self.grid_size
        self.max_in_cell = 128
        self.cell_count = ti.field(dtype=ti.i32,
                                   shape=(self.grid_size, self.grid_size))
        self.cell_particles = ti.field(
            dtype=ti.i32,
            shape=(self.grid_size, self.grid_size, self.max_in_cell),
        )

    # ----------------------------------------------------------------------
    def load_particles(
        self,
        positions: np.ndarray,
        material: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Upload the initial cloud. Same signature as ``MPMSolver`` for parity.

        ``positions`` is (N, 2) in [0, 1]^2, ``material`` is (N,) of ids (kept
        only for colouring -- every particle is simulated as fluid), ``velocity``
        is an optional (N, 2) initial velocity.
        """
        assert positions.shape == (self.n_particles, 2)
        assert material.shape == (self.n_particles,)
        self.x.from_numpy(positions.astype(np.float32))
        self.material.from_numpy(material.astype(np.int32))
        if velocity is not None:
            assert velocity.shape == (self.n_particles, 2)
            self.v.from_numpy(velocity.astype(np.float32))
        else:
            self.v.fill(0.0)
        self.rho.fill(self.rho0)
        self.pressure.fill(0.0)

    # --- smoothing kernel (cubic spline, support radius h) ----------------
    @ti.func
    def _w(self, r):
        q = r * self.inv_h
        res = 0.0
        if q < 1.0:
            if q <= 0.5:
                res = self.kernel_norm * (6.0 * (q * q * q - q * q) + 1.0)
            else:
                t = 1.0 - q
                res = self.kernel_norm * 2.0 * t * t * t
        return res

    @ti.func
    def _dwdr(self, r):
        """Scalar radial derivative dW/dr; gradient is this times (x_i-x_j)/r."""
        q = r * self.inv_h
        res = 0.0
        if 1e-8 < q < 1.0:
            if q <= 0.5:
                res = self.kernel_norm * (18.0 * q * q - 12.0 * q) * self.inv_h
            else:
                t = 1.0 - q
                res = self.kernel_norm * (-6.0 * t * t) * self.inv_h
        return res

    @ti.func
    def _cell(self, p):
        c = (self.x[p] * (1.0 / self.cell_size)).cast(int)
        c[0] = ti.max(0, ti.min(self.grid_size - 1, c[0]))
        c[1] = ti.max(0, ti.min(self.grid_size - 1, c[1]))
        return c

    # ----------------------------------------------------------------------
    @ti.kernel
    def _build_grid(self):
        for I in ti.grouped(self.cell_count):
            self.cell_count[I] = 0
        for p in range(self.n_particles):
            c = self._cell(p)
            k = ti.atomic_add(self.cell_count[c[0], c[1]], 1)
            if k < self.max_in_cell:
                self.cell_particles[c[0], c[1], k] = p

    @ti.kernel
    def _compute_density(self):
        for p in range(self.n_particles):
            c = self._cell(p)
            rho = 0.0
            for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                ci, cj = c[0] + di, c[1] + dj
                if 0 <= ci < self.grid_size and 0 <= cj < self.grid_size:
                    n = ti.min(self.cell_count[ci, cj], self.max_in_cell)
                    for k in range(n):
                        j = self.cell_particles[ci, cj, k]
                        r = (self.x[p] - self.x[j]).norm()
                        rho += self.mass * self._w(r)
            self.rho[p] = ti.max(rho, 1e-6)
            # Tait EoS, clamped to >= 0: a free-surface fluid carries no tension,
            # and clamping also suppresses the SPH tensile-instability clumping.
            ratio = self.rho[p] / self.rho0
            self.pressure[p] = ti.max(
                0.0, self.B * (ratio ** self.gamma - 1.0))

    @ti.kernel
    def _integrate(self):
        eps = 0.01 * self.h * self.h
        for p in range(self.n_particles):
            c = self._cell(p)
            acc = ti.Vector([0.0, -self.gravity])
            pi_term = self.pressure[p] / (self.rho[p] * self.rho[p])
            for di, dj in ti.static(ti.ndrange((-1, 2), (-1, 2))):
                ci, cj = c[0] + di, c[1] + dj
                if 0 <= ci < self.grid_size and 0 <= cj < self.grid_size:
                    n = ti.min(self.cell_count[ci, cj], self.max_in_cell)
                    for k in range(n):
                        j = self.cell_particles[ci, cj, k]
                        if j == p:
                            continue
                        dx = self.x[p] - self.x[j]
                        r = dx.norm()
                        if r >= self.h or r < 1e-8:
                            continue
                        grad = self._dwdr(r) * dx / r
                        # symmetric pressure acceleration
                        pj_term = self.pressure[j] / (self.rho[j] * self.rho[j])
                        acc += -self.mass * (pi_term + pj_term) * grad
                        # Monaghan artificial viscosity (only when approaching)
                        dv = self.v[p] - self.v[j]
                        vr = dv.dot(dx)
                        if vr < 0.0:
                            rho_bar = 0.5 * (self.rho[p] + self.rho[j])
                            mu = self.h * vr / (r * r + eps)
                            pi_visc = -self.visc_alpha * self.sound_speed * mu \
                                / rho_bar
                            acc += -self.mass * pi_visc * grad
            # semi-implicit Euler
            self.v[p] += self.dt * acc
            self.x[p] += self.dt * self.v[p]
            # reflecting walls
            for d in ti.static(range(2)):
                if self.x[p][d] < self.pad:
                    self.x[p][d] = self.pad
                    if self.v[p][d] < 0.0:
                        self.v[p][d] = 0.0
                if self.x[p][d] > 1.0 - self.pad:
                    self.x[p][d] = 1.0 - self.pad
                    if self.v[p][d] > 0.0:
                        self.v[p][d] = 0.0

    def step(self, substeps: int = 50) -> None:
        """Advance the fluid by ``substeps`` WCSPH substeps."""
        for _ in range(substeps):
            self._build_grid()
            self._compute_density()
            self._integrate()
