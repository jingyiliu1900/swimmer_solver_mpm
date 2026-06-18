"""Scene builders.

A scene returns ``(positions, material, velocity)`` numpy arrays describing the
initial particle cloud. Helpers sample filled shapes at a roughly uniform
density so fluid and solid have comparable mass per unit area, and each shape
can be given an initial velocity (useful for collisions, especially in zero-g).
"""

from __future__ import annotations

import numpy as np

# Particles per unit area. With the solver's default grid this is ~4 per cell.
DEFAULT_DENSITY = 4 * 128 * 128

Group = tuple[np.ndarray, np.ndarray, np.ndarray]  # (positions, material, velocity)


def _n_for_area(area: float, density: float) -> int:
    return max(1, int(area * density))


def _velocity(n: int, velocity: tuple[float, float]) -> np.ndarray:
    return np.tile(np.asarray(velocity, dtype=np.float64), (n, 1))


def fill_rect(
    lower: tuple[float, float],
    size: tuple[float, float],
    material: int,
    velocity: tuple[float, float] = (0.0, 0.0),
    density: float = DEFAULT_DENSITY,
    rng: np.random.Generator | None = None,
) -> Group:
    """Uniformly sample a rectangle ``[lower, lower+size]`` with initial velocity."""
    rng = rng or np.random.default_rng(0)
    n = _n_for_area(size[0] * size[1], density)
    pts = rng.random((n, 2)) * np.array(size) + np.array(lower)
    mat = np.full(n, material, dtype=np.int32)
    return pts, mat, _velocity(n, velocity)


def fill_circle(
    center: tuple[float, float],
    radius: float,
    material: int,
    velocity: tuple[float, float] = (0.0, 0.0),
    density: float = DEFAULT_DENSITY,
    rng: np.random.Generator | None = None,
) -> Group:
    """Uniformly sample a disk (rejection sampling) with initial velocity."""
    rng = rng or np.random.default_rng(0)
    n = _n_for_area(np.pi * radius * radius, density)
    pts = np.empty((n, 2), dtype=np.float64)
    filled = 0
    while filled < n:
        cand = (rng.random((n, 2)) * 2.0 - 1.0)
        cand = cand[(cand[:, 0] ** 2 + cand[:, 1] ** 2) <= 1.0]
        take = min(len(cand), n - filled)
        pts[filled : filled + take] = cand[:take]
        filled += take
    pts = pts * radius + np.array(center)
    mat = np.full(n, material, dtype=np.int32)
    return pts, mat, _velocity(n, velocity)


def polygon_area(vertices: np.ndarray) -> float:
    """Absolute area of a simple polygon via the shoelace formula."""
    v = np.asarray(vertices, dtype=float)
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def fill_polygon(
    vertices: np.ndarray,
    material: int,
    velocity: tuple[float, float] = (0.0, 0.0),
    density: float = DEFAULT_DENSITY,
    rng: np.random.Generator | None = None,
) -> Group:
    """Uniformly sample particles inside *any* simple polygon.

    ``vertices`` is an ``(N, 2)`` array of outline points in ``[0, 1]^2`` (the
    polygon may be convex or concave). This is the general "bring your own
    shape" sampler: hand it a star, a letter, an animal outline, anything, and
    it fills the interior with particles of the given ``material``.
    """
    from matplotlib.path import Path  # matplotlib is a runtime dependency

    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
        raise ValueError("vertices must be an (N>=3, 2) array of (x, y) points")
    lower = verts.min(axis=0)
    size = verts.max(axis=0) - lower
    n = _n_for_area(polygon_area(verts), density)
    path = Path(verts)
    rng = rng or np.random.default_rng(0)
    pts = np.empty((n, 2), dtype=np.float64)
    filled = 0
    while filled < n:
        cand = rng.random((n, 2)) * size + lower
        cand = cand[path.contains_points(cand)]
        take = min(len(cand), n - filled)
        pts[filled : filled + take] = cand[:take]
        filled += take
    mat = np.full(n, material, dtype=np.int32)
    return pts, mat, _velocity(n, velocity)


def regular_polygon(center: tuple[float, float], radius: float, sides: int,
                    rotation: float = 0.0) -> np.ndarray:
    """Vertices of a regular ``sides``-gon (triangle, pentagon, hexagon, ...)."""
    a = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False) + rotation
    return np.stack([center[0] + radius * np.cos(a),
                     center[1] + radius * np.sin(a)], axis=1)


def star(center: tuple[float, float], r_outer: float, r_inner: float,
         points: int = 5, rotation: float = np.pi / 2) -> np.ndarray:
    """Vertices of a ``points``-pointed star (a concave polygon)."""
    a = np.linspace(0.0, 2.0 * np.pi, 2 * points, endpoint=False) + rotation
    r = np.where(np.arange(2 * points) % 2 == 0, r_outer, r_inner)
    return np.stack([center[0] + r * np.cos(a),
                     center[1] + r * np.sin(a)], axis=1)


def fish_halfwidth(s: np.ndarray, width: float) -> np.ndarray:
    """Half-thickness of a slender swimmer at normalised coord ``s`` in ``[0, 1]``.

    A smooth body rounded at both ends (no abrupt head, no stuck-on tail fin):
    the half-thickness follows a flattened ellipse so the midsection stays full
    and the nose and tail taper to rounded points. The same silhouette is used
    by the immersed-body solver, so particle and analytic swimmers look alike.
    """
    u = np.clip(2.0 * s - 1.0, -1.0, 1.0)            # -1 at tail, +1 at head
    return width * np.sqrt(np.maximum(0.0, 1.0 - u ** 4))


def fish_inside(
    points: np.ndarray,
    center: tuple[float, float],
    length: float,
    width: float,
) -> np.ndarray:
    """Boolean mask of which ``points`` (N, 2) lie inside the fish silhouette.

    The fish is centred at ``center`` with its body axis along x (head at +x).
    Handy for carving the fish-shaped hole out of a fluid-filled tank.
    """
    rx = points[:, 0] - center[0]
    ry = points[:, 1] - center[1]
    s = (rx + length / 2) / length
    hw = fish_halfwidth(s, width)
    return (np.abs(rx) < length / 2) & (hw > 0.0) & (np.abs(ry) < hw)


def fill_fish(
    center: tuple[float, float],
    length: float,
    width: float,
    material: int,
    velocity: tuple[float, float] = (0.0, 0.0),
    density: float = DEFAULT_DENSITY,
    rng: np.random.Generator | None = None,
) -> Group:
    """Sample a fish-shaped particle cloud (blunt head, tapered tail, caudal fin).

    ``length`` is the head-to-tail extent, ``width`` the peak half-thickness, and
    the body axis runs along x with the head at +x.
    """
    rng = rng or np.random.default_rng(0)
    cx, cy = center
    n = _n_for_area(length * 2.0 * width, density)
    pts = np.empty((n, 2), dtype=np.float64)
    filled = 0
    while filled < n:
        cand = rng.random((n, 2))
        cand[:, 0] = (cand[:, 0] - 0.5) * length          # rx in [-L/2, L/2]
        cand[:, 1] = (cand[:, 1] - 0.5) * 2.0 * width      # ry in [-width, width]
        hw = fish_halfwidth((cand[:, 0] + length / 2) / length, width)
        cand = cand[np.abs(cand[:, 1]) < hw]
        take = min(len(cand), n - filled)
        pts[filled : filled + take] = cand[:take]
        filled += take
    pts[:, 0] += cx
    pts[:, 1] += cy
    mat = np.full(len(pts), material, dtype=np.int32)
    return pts, mat, _velocity(len(pts), velocity)


def combine(*groups: Group) -> Group:
    """Concatenate several ``(positions, material, velocity)`` groups."""
    pos = np.concatenate([g[0] for g in groups], axis=0)
    mat = np.concatenate([g[1] for g in groups], axis=0)
    vel = np.concatenate([g[2] for g in groups], axis=0)
    return pos, mat, vel


# --------------------------------------------------------------------------
def splash(rng: np.random.Generator | None = None) -> Group:
    """A pool of water with two elastic blobs dropping into it."""
    from .materials import ELASTIC, WATER

    rng = rng or np.random.default_rng(1)
    return combine(
        fill_rect((0.02, 0.02), (0.96, 0.22), WATER, rng=rng),       # pool
        fill_circle((0.32, 0.72), 0.09, ELASTIC, rng=rng),           # blob 1
        fill_rect((0.58, 0.62), (0.16, 0.16), ELASTIC, rng=rng),     # blob 2
    )


def dam_break(rng: np.random.Generator | None = None) -> Group:
    """A tall water column that collapses onto a resting elastic block."""
    from .materials import ELASTIC, WATER

    rng = rng or np.random.default_rng(2)
    return combine(
        fill_rect((0.04, 0.04), (0.30, 0.62), WATER, rng=rng),       # water column
        fill_rect((0.62, 0.04), (0.18, 0.18), ELASTIC, rng=rng),     # block to hit
    )


def showcase(rng: np.random.Generator | None = None) -> Group:
    """All three materials at once: water pool, an elastic blob, a rigid block."""
    from .materials import ELASTIC, RIGID, WATER

    rng = rng or np.random.default_rng(3)
    return combine(
        fill_rect((0.02, 0.02), (0.96, 0.24), WATER, rng=rng),       # pool
        fill_circle((0.30, 0.74), 0.10, ELASTIC, rng=rng),           # soft blob
        fill_rect((0.60, 0.70), (0.15, 0.15), RIGID, rng=rng),       # rigid block
    )


def collide(rng: np.random.Generator | None = None) -> Group:
    """Two elastic blobs fired at each other. Best viewed with --gravity 0.

    A clean demo of momentum transfer and elastic rebound with no gravity bias:
    the blobs squash against each other on contact, then spring apart.
    """
    from .materials import ELASTIC

    rng = rng or np.random.default_rng(4)
    return combine(
        fill_circle((0.22, 0.5), 0.10, ELASTIC, velocity=(2.5, 0.0), rng=rng),
        fill_circle((0.78, 0.5), 0.10, ELASTIC, velocity=(-2.5, 0.0), rng=rng),
    )


SCENES = {
    "splash": splash,
    "dam_break": dam_break,
    "showcase": showcase,
    "collide": collide,
}
