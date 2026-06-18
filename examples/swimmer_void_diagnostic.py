"""Diagnostic: is the immersed swimmer actually wetted by the fluid?

Direct-forcing coupling can only transfer force where fluid touches the body.
If a cavity opens next to the body, that part of the boundary exerts no force
and the swimming result is not trustworthy. This sweeps viscosity and reports,
per case, the swim displacement AND the worst/mean boundary-void fraction
(from MPMSolver.boundary_void). Void ~0 => fully wetted => valid.

Run from the repo root:  python examples/swimmer_void_diagnostic.py
"""

import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import taichi as ti
ti.init(arch=ti.cpu)
import swimmers.materials as materials
from swimmers import WATER, MPMSolver
from swimmers.scenes import fill_rect, fish_halfwidth, DEFAULT_DENSITY

# stiffer (more incompressible) water so it refills behind the body
_w = materials.MATERIALS[WATER]
materials.MATERIALS[WATER] = materials.Material(
    name="water", youngs=8000.0, poisson=0.2, rho=1.0, color=_w.color)

L, HALF_W, AMP0, NWAV, OMEGA = 0.40, 0.04, 0.09, 1.0, 45.0
START = (0.28, 0.5)
DT = 1.5e-5
DENS_DIV = 2
def body_shape(rx, t, body_k):
    s = (rx + L / 2) / L
    hw = fish_halfwidth(s, HALF_W)
    amp = AMP0 * (0.2 + 0.8 * (1 - s))
    yc = amp * np.sin(body_k * rx + OMEGA * t)
    return yc, hw


print("visc  | disp_x  disp_y |d|   | max_void%  mean_void%")
print("-" * 58)
for eta in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]:
    rng = np.random.default_rng(0)
    cx, cy = START
    body_k = 2 * np.pi * NWAV / L
    wb, wm, _ = fill_rect((0.03, 0.03), (0.94, 0.94), WATER,
                          density=DEFAULT_DENSITY // DENS_DIV, rng=rng)
    rx, ry = wb[:, 0] - cx, wb[:, 1] - cy
    yc, hw = body_shape(rx, 0.0, body_k)
    inside = (np.abs(rx) < L / 2) & (np.abs(ry - yc) < hw)
    wb, wm = wb[~inside], wm[~inside]
    s = MPMSolver(n_particles=wb.shape[0], n_grid=128, dt=DT, gravity=0.0,
                  viscosity=eta)
    s.load_particles(wb, wm)
    s.configure_body(center=START, length=L, half_width=HALF_W, amp0=AMP0,
                     n_waves=NWAV, omega=OMEGA, density=1.0)
    p0 = s.body_pos.to_numpy().copy()
    voids = []
    for f in range(220):
        s.step(40)
        voids.append(float(s.boundary_void()))
    p = s.body_pos.to_numpy()
    x = s.x.to_numpy()
    ok = bool(np.isfinite(x).all())
    d = p - p0
    print("%.2f  | %+.3f %+.3f %.3f | %7.1f%% %8.1f%%  %s"
          % (eta, d[0], d[1], np.hypot(*d), 100 * max(voids),
             100 * np.mean(voids), "" if ok else "UNSTABLE"), flush=True)