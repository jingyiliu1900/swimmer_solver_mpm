"""MLS-MPM 2D solver for fluid <-> deformable-solid interaction.

The algorithm is the standard Moving Least Squares Material Point Method
(Hu et al. 2018), specialised to two materials:

* WATER   - shear modulus forced to zero; volume change drives an isotropic
            pressure (weakly compressible fluid).
* ELASTIC - fixed-corotated (neo-Hookean-like) stress; resists both shear and
            volume change, so it springs back into shape.

All particles live in the unit square [0, 1]^2 and deposit momentum onto one
shared background grid each substep. Because both materials read from and write
to that same grid, fluid and solid push on each other automatically -- the
two-way coupling is a property of the method, not special-cased code.
"""

from __future__ import annotations

import numpy as np
import taichi as ti

from .materials import MATERIALS, SWIMMER, WATER


@ti.data_oriented
class MPMSolver:
    def __init__(
        self,
        n_particles: int,
        n_grid: int = 128,
        dt: float = 1.0e-4,
        gravity: float = 9.8,
        viscosity: float = 0.0,
        act_strength: float = 0.0,
        swim_omega: float = 0.0,
    ):
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.dx = 1.0 / n_grid
        self.inv_dx = float(n_grid)
        self.dt = dt
        self.gravity = gravity
        # Dynamic viscosity of the WATER material. 0 = inviscid (default);
        # larger = thicker, honey-like flow that resists shearing.
        self.viscosity = viscosity
        # Active-swimmer muscle parameters (only affect SWIMMER particles):
        #   act_strength - peak active stress along the body fibre
        #   swim_omega   - angular frequency of the muscle oscillation
        self.act_strength = act_strength
        self.swim_omega = swim_omega

        # --- immersed whole-body swimmer (NOT made of particles) ----------
        # A single analytic solid (a slender undulating fish) coupled to the
        # fluid by direct forcing: grid nodes overlapping the body are forced
        # to the body's velocity, and the equal/opposite reaction propels the
        # body. Disabled unless configure_body() is called.
        self.body_on = False
        self.body_kind = 0        # 0 = fish (undulating strip), 1 = bell (jellyfish)
        self.body_L = 0.30        # body length
        self.body_W = 0.035       # body half-thickness
        self.body_amp0 = 0.06     # peak undulation amplitude (at the tail)
        self.body_k = 2.0 * 3.141592653589793 / 0.30  # wavenumber (1 wave)
        self.body_mass = 1.0      # set from area * density in configure_body
        # bell (jellyfish) parameters
        self.bell_rin = 0.05      # rest inner radius
        self.bell_rout = 0.10     # rest outer radius
        self.bell_amp = 0.3       # contraction fraction per pulse (deformability)
        self.body_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.body_vel = ti.Vector.field(2, dtype=ti.f32, shape=())
        self.body_force = ti.Vector.field(2, dtype=ti.f32, shape=())
        # diagnostic counters for boundary_void() (kernels can't return a value
        # here because `from __future__ import annotations` stringises the type)
        self._void_ring = ti.field(dtype=ti.i32, shape=())
        self._void_empty = ti.field(dtype=ti.i32, shape=())

        # Reference volume of one particle. Roughly 4 particles per cell.
        self.p_vol = (self.dx * 0.5) ** 2

        # --- per particle state -------------------------------------------
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # position
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # velocity
        self.C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # affine
        self.F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # def. grad
        self.J = ti.field(dtype=ti.f32, shape=n_particles)  # volume ratio (fluid)
        self.material = ti.field(dtype=ti.i32, shape=n_particles)
        self.mass = ti.field(dtype=ti.f32, shape=n_particles)  # per-particle mass

        # --- swimmer actuation (per particle) -----------------------------
        # swim_amp: signed muscle amplitude (>0 dorsal, <0 ventral, 0 = inert)
        # swim_phase: spatial phase (travelling wave for a fish; constant for a
        #             synchronously-pulsing jellyfish cell)
        # swim_dir:  rest fibre direction the active stress contracts along
        #            (body axis for a fish; circumferential for a jellyfish bell)
        self.swim_amp = ti.field(dtype=ti.f32, shape=n_particles)
        self.swim_phase = ti.field(dtype=ti.f32, shape=n_particles)
        self.swim_dir = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
        # rectify==True -> muscle only contracts (max(0, .)) and elasticity does
        # the recovery stroke; this time-asymmetry is what lets a pulsing bell
        # net-propel instead of just breathing in place.
        self.swim_rectify = False
        self.t = ti.field(dtype=ti.f32, shape=())  # simulation clock

        # --- background grid ----------------------------------------------
        self.grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))

        # --- per material Lame parameters (indexed by material id) --------
        n_mat = len(MATERIALS)
        self.mat_mu = ti.field(dtype=ti.f32, shape=n_mat)
        self.mat_lam = ti.field(dtype=ti.f32, shape=n_mat)
        self.mat_rho = ti.field(dtype=ti.f32, shape=n_mat)
        self.mat_mu.from_numpy(np.array([m.mu for m in MATERIALS], dtype=np.float32))
        self.mat_lam.from_numpy(
            np.array([m.lam for m in MATERIALS], dtype=np.float32)
        )
        self.mat_rho.from_numpy(
            np.array([m.rho for m in MATERIALS], dtype=np.float32)
        )

    # ----------------------------------------------------------------------
    def load_particles(
        self,
        positions: np.ndarray,
        material: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        """Upload initial particle positions, material ids, and velocities.

        ``positions`` is (N, 2) in [0, 1]^2, ``material`` is (N,) of ids, and
        ``velocity`` is an optional (N, 2) initial velocity (defaults to rest).
        """
        assert positions.shape == (self.n_particles, 2)
        assert material.shape == (self.n_particles,)
        self.x.from_numpy(positions.astype(np.float32))
        self.material.from_numpy(material.astype(np.int32))
        self._reset_kinematics()  # also sets per-particle mass; zeroes velocity
        self.t[None] = 0.0
        if velocity is not None:
            assert velocity.shape == (self.n_particles, 2)
            self.v.from_numpy(velocity.astype(np.float32))

    def set_actuation(self, swim_amp: np.ndarray, swim_phase: np.ndarray,
                      swim_dir: np.ndarray | None = None) -> None:
        """Upload per-particle muscle amplitude, phase, and fibre direction.

        Only SWIMMER particles feel it. The active stress on a swimmer particle
        is ``act_strength * a * (d (x) d)`` where ``d`` is the (deformed) fibre
        direction and ``a = swim_amp * sin(swim_omega * t - swim_phase)`` (or its
        positive part if ``swim_rectify`` is set).

        * Fish: ``swim_dir`` = body axis, phase increasing along the body and amp
          flipping sign across the thickness -> a travelling bending wave.
        * Jellyfish bell: ``swim_dir`` = circumferential, same phase everywhere,
          ``swim_rectify=True`` -> synchronous contraction pulses.
        """
        assert swim_amp.shape == (self.n_particles,)
        assert swim_phase.shape == (self.n_particles,)
        self.swim_amp.from_numpy(swim_amp.astype(np.float32))
        self.swim_phase.from_numpy(swim_phase.astype(np.float32))
        if swim_dir is not None:
            assert swim_dir.shape == (self.n_particles, 2)
            self.swim_dir.from_numpy(swim_dir.astype(np.float32))

    @ti.kernel
    def _reset_kinematics(self):
        for p in range(self.n_particles):
            self.v[p] = ti.Vector([0.0, 0.0])
            self.F[p] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
            self.C[p] = ti.Matrix.zero(ti.f32, 2, 2)
            self.J[p] = 1.0
            self.swim_amp[p] = 0.0
            self.swim_phase[p] = 0.0
            self.swim_dir[p] = ti.Vector([1.0, 0.0])
            # mass = reference volume * the material's rest density
            self.mass[p] = self.p_vol * self.mat_rho[self.material[p]]

    # ----------------------------------------------------------------------
    def configure_body(self, center, length, half_width, amp0, n_waves,
                       omega, density=1.0):
        """Enable the immersed whole-body swimmer.

        ``center`` is the initial centre of mass; ``length``/``half_width`` size
        the slender body; ``amp0`` is the peak (tail) undulation amplitude;
        ``n_waves`` body-wavelengths along the length; ``omega`` the beat
        frequency; ``density`` the body density relative to the fluid.

        The body is a continuous analytic shape (a tapered, undulating strip),
        so it can never fragment the way a particle body can. Its forward
        motion is not prescribed -- it emerges from the fluid reaction.
        """
        self.body_on = True
        self.body_L = float(length)
        self.body_W = float(half_width)
        self.body_amp0 = float(amp0)
        self.body_k = 2.0 * 3.141592653589793 * float(n_waves) / float(length)
        self.swim_omega = float(omega)
        # mass = area * density (tapered strip ~ 0.66 of the bounding box)
        area = length * (2.0 * half_width) * 0.66
        self.body_mass = area * float(density)
        self.body_kind = 0
        self.body_pos[None] = ti.Vector(list(center))
        self.body_vel[None] = ti.Vector([0.0, 0.0])
        self.body_force[None] = ti.Vector([0.0, 0.0])

    def configure_bell(self, center, r_inner, r_outer, amp, omega, density=1.0):
        """Enable the immersed whole-body JELLYFISH (a single analytic bell).

        The bell is an upper half-annulus (a dome open at the bottom) defined by
        geometry, so it is one whole object that can NEVER fragment. It pulses by
        a prescribed radial contraction of fraction ``amp`` at frequency
        ``omega`` (rectified: contract then relax). Each contraction jets fluid
        out of the opening; the reaction self-propels the bell. ``amp`` plays the
        role of 'deformability' -- a larger amp pulses/deforms more.
        """
        self.body_on = True
        self.body_kind = 1
        self.bell_rin = float(r_inner)
        self.bell_rout = float(r_outer)
        self.bell_amp = float(amp)
        self.swim_omega = float(omega)
        # mass = half-annulus area * density
        area = 0.5 * 3.141592653589793 * (r_outer ** 2 - r_inner ** 2)
        self.body_mass = area * float(density)
        self.body_pos[None] = ti.Vector(list(center))
        self.body_vel[None] = ti.Vector([0.0, 0.0])
        self.body_force[None] = ti.Vector([0.0, 0.0])

    @ti.func
    def _fish(self, rx):
        """Slender-swimmer geometry/kinematics at body-frame x = rx.

        Returns a vector [half_thickness, centreline_y, centreline_y_dot].
        Shared by the physics override, the wetting diagnostic, and (mirrored
        in Python) the renderer so all three use the identical shape: a smooth
        slender body rounded at both ends (no blunt head, no stuck-on tail fin).
        """
        s = (rx + 0.5 * self.body_L) / self.body_L     # 0 = tail, 1 = head
        u = 2.0 * s - 1.0                              # -1 at tail, +1 at head
        hw = self.body_W * ti.sqrt(ti.max(0.0, 1.0 - u * u * u * u))
        amp = self.body_amp0 * (0.2 + 0.8 * (1.0 - s))  # grows toward the tail
        ph = self.body_k * rx + self.swim_omega * self.t[None]
        return ti.Vector([hw, amp * ti.sin(ph), amp * self.swim_omega * ti.cos(ph)])

    @ti.kernel
    def _count_boundary_void(self):
        margin = 2.0 * self.dx
        self._void_ring[None] = 0
        self._void_empty[None] = 0
        if ti.static(self.body_on):
            for i, j in self.grid_m:
                rx = i * self.dx - self.body_pos[None][0]
                ry = j * self.dx - self.body_pos[None][1]
                if rx > -0.5 * self.body_L - margin and rx < 0.5 * self.body_L + margin:
                    f = self._fish(rx)
                    hw, yc = f[0], f[1]
                    dist = ti.abs(ry - yc)
                    inside = hw > 0.0 and dist < hw
                    near = dist < hw + margin
                    if near and not inside:        # the ring just outside the body
                        self._void_ring[None] += 1
                        if self.grid_m[i, j] == 0.0:
                            self._void_empty[None] += 1

    def boundary_void(self) -> float:
        """Fraction of the thin fluid ring just outside the body that is EMPTY.

        Direct forcing can only transfer force where fluid actually touches the
        body. Of the grid cells within ~2 cells outside the body silhouette,
        this returns the fraction with no fluid (grid_m == 0). 0.0 means the
        body is fully wetted (its reaction forces are valid); a large value
        means part of the boundary faces a void and that force is missing.
        """
        self._count_boundary_void()
        ring = int(self._void_ring[None])
        return float(self._void_empty[None]) / max(1, ring)

    @ti.kernel
    def substep(self):
        # 1. clear grid
        for i, j in self.grid_m:
            self.grid_v[i, j] = ti.Vector([0.0, 0.0])
            self.grid_m[i, j] = 0.0

        # 2. particles -> grid (P2G)
        for p in range(self.n_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            # quadratic B-spline weights
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            # advance deformation gradient
            self.F[p] = (ti.Matrix.identity(ti.f32, 2) + self.dt * self.C[p]) @ self.F[p]

            mat = self.material[p]
            mu = self.mat_mu[mat]
            lam = self.mat_lam[mat]
            if mat == WATER:
                mu = 0.0  # a fluid carries no shear stress

            U, sig, V = ti.svd(self.F[p])
            J = sig[0, 0] * sig[1, 1]

            if mat == WATER:
                # Reset the deformation gradient to a pure volumetric one. This
                # keeps the fluid numerically stable while preserving its
                # tracked volume change through J.
                self.J[p] *= (1.0 + self.dt * self.C[p].trace())
                self.F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(self.J[p])
                J = self.J[p]

            # fixed-corotated / neo-Hookean Cauchy stress
            R = U @ V.transpose()
            stress = 2.0 * mu * (self.F[p] - R) @ self.F[p].transpose() + \
                ti.Matrix.identity(ti.f32, 2) * lam * J * (J - 1.0)
            if mat == WATER:
                # Newtonian viscous stress: eta * (grad v + grad v^T).
                # The affine field C is the APIC estimate of grad v, so the
                # symmetric part C + C^T is (twice) the strain rate.
                stress += self.viscosity * (self.C[p] + self.C[p].transpose())
            if mat == SWIMMER:
                # Active muscle stress along the (deformed) fibre direction.
                # R rotates the rest fibre swim_dir so the muscle pulls along
                # the body as it deforms.
                a = self.swim_amp[p] * ti.sin(
                    self.swim_omega * self.t[None] - self.swim_phase[p])
                if ti.static(self.swim_rectify):
                    a = ti.max(0.0, a)  # contract-only; elasticity recovers
                d = R @ self.swim_dir[p]
                stress += self.act_strength * a * d.outer_product(d)
            stress = (-self.dt * self.p_vol * 4.0 * self.inv_dx * self.inv_dx) * stress
            affine = stress + self.mass[p] * self.C[p]

            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1]
                self.grid_v[base + offset] += weight * (
                    self.mass[p] * self.v[p] + affine @ dpos
                )
                self.grid_m[base + offset] += weight * self.mass[p]

        # 3. grid update: normalise, gravity, walls
        for i, j in self.grid_m:
            if self.grid_m[i, j] > 0.0:
                self.grid_v[i, j] = self.grid_v[i, j] / self.grid_m[i, j]
                self.grid_v[i, j][1] -= self.dt * self.gravity
                bound = 3
                if i < bound and self.grid_v[i, j][0] < 0:
                    self.grid_v[i, j][0] = 0.0
                if i > self.n_grid - bound and self.grid_v[i, j][0] > 0:
                    self.grid_v[i, j][0] = 0.0
                if j < bound and self.grid_v[i, j][1] < 0:
                    self.grid_v[i, j][1] = 0.0
                if j > self.n_grid - bound and self.grid_v[i, j][1] > 0:
                    self.grid_v[i, j][1] = 0.0

        # 3b. immersed whole-body swimmer: direct forcing.
        # For every fluid grid node covered by the analytic body, override the
        # node velocity with the body's velocity (rigid translation + the
        # prescribed undulation). Accumulate the momentum we injected; its
        # negative is the reaction impulse that propels the body.
        if ti.static(self.body_on and self.body_kind == 0):   # fish
            for i, j in self.grid_m:
                if self.grid_m[i, j] > 0.0:
                    rx = i * self.dx - self.body_pos[None][0]
                    ry = j * self.dx - self.body_pos[None][1]
                    if rx > -0.5 * self.body_L and rx < 0.5 * self.body_L:
                        f = self._fish(rx)
                        hw, yc, dyc_dt = f[0], f[1], f[2]
                        if hw > 0.0 and ry > yc - hw and ry < yc + hw:
                            u = self.body_vel[None] + ti.Vector([0.0, dyc_dt])
                            dp = self.grid_m[i, j] * (u - self.grid_v[i, j])
                            self.grid_v[i, j] = u
                            self.body_force[None] += -dp

        if ti.static(self.body_on and self.body_kind == 1):   # jellyfish bell
            # The bell DEFORMS (it does not shrink): a rectified pulse squeezes
            # it horizontally (sx < 1) and stretches it vertically (sy = 1/sx),
            # so it goes prolate ("closes") then flattens back, area preserved.
            # The inward side walls jet fluid out of the open bottom.
            phi = self.swim_omega * self.t[None]
            sinp = ti.sin(phi)
            pulse = ti.max(0.0, sinp)
            sx = 1.0 - self.bell_amp * pulse
            sy = 1.0 / sx            # area-preserving: narrows -> taller (propulsive)
            pulse_dot = 0.0
            if sinp > 0.0:
                pulse_dot = self.swim_omega * ti.cos(phi)
            sx_dot = -self.bell_amp * pulse_dot
            sy_dot = -sx_dot / (sx * sx)
            for i, j in self.grid_m:
                if self.grid_m[i, j] > 0.0:
                    rx = i * self.dx - self.body_pos[None][0]
                    ry = j * self.dx - self.body_pos[None][1]
                    rx0 = rx / sx          # undo the deformation -> rest coords
                    ry0 = ry / sy
                    rho0 = ti.sqrt(rx0 * rx0 + ry0 * ry0)
                    ang0 = ti.atan2(ry0, rx0)
                    if ang0 > 0.0 and ang0 < 3.14159265 and \
                            rho0 > self.bell_rin and rho0 < self.bell_rout:
                        u_def = ti.Vector([(sx_dot / sx) * rx, (sy_dot / sy) * ry])
                        u = self.body_vel[None] + u_def
                        dp = self.grid_m[i, j] * (u - self.grid_v[i, j])
                        self.grid_v[i, j] = u
                        self.body_force[None] += -dp

        # 4. grid -> particles (G2P)
        for p in range(self.n_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]
            new_v = ti.Vector.zero(ti.f32, 2)
            new_C = ti.Matrix.zero(ti.f32, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = offset.cast(float) - fx
                g_v = self.grid_v[base + offset]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4.0 * self.inv_dx * weight * g_v.outer_product(dpos)
            self.v[p] = new_v
            self.C[p] = new_C
            self.x[p] += self.dt * new_v

    @ti.kernel
    def update_body(self):
        """Advance the immersed body's centre of mass from the fluid reaction."""
        if ti.static(self.body_on):
            self.body_vel[None] += self.body_force[None] / self.body_mass
            self.body_pos[None] += self.dt * self.body_vel[None]
            self.body_force[None] = ti.Vector([0.0, 0.0])

    def step(self, substeps: int = 50) -> None:
        """Advance the simulation by ``substeps`` MLS-MPM substeps."""
        for _ in range(substeps):
            self.substep()
            self.update_body()
            self.t[None] += self.dt
