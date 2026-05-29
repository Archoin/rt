"""Flat-facet gem geometry: planar convex polygons and ray intersection.

Pure NumPy. A :class:`Gem` is a closed set of outward-facing :class:`Facet`s.
Intersection is brute-force over facets — a diamond has ~58, so no acceleration
structure is needed at prototype scale.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-7


class Facet:
    """A planar convex polygon with an outward unit normal."""

    def __init__(self, vertices, normal=None):
        self.v = np.asarray(vertices, dtype=float)        # (k, 3), ordered
        if normal is None:
            normal = np.cross(self.v[1] - self.v[0], self.v[2] - self.v[0])
        self.n = normal / np.linalg.norm(normal)          # outward unit normal
        self.p0 = self.v[0]
        self._drop = int(np.argmax(np.abs(self.n)))       # axis to drop for 2D
        self._keep = [i for i in range(3) if i != self._drop]

    def intersect(self, o, d):
        """Distance ``t > EPS`` to the polygon along ray ``o + t·d``, or None."""
        denom = float(np.dot(self.n, d))
        if abs(denom) < EPS:
            return None                                    # parallel to plane
        t = float(np.dot(self.n, self.p0 - o)) / denom
        if t <= EPS:
            return None
        if self._contains(o + t * d):
            return t
        return None

    def _contains(self, p):
        """Convex point-in-polygon in the dropped-axis 2D projection."""
        poly = self.v[:, self._keep]
        q = p[self._keep]
        sign = 0
        k = len(poly)
        for i in range(k):
            a = poly[i]
            e = poly[(i + 1) % k] - a
            w = q - a
            cross = e[0] * w[1] - e[1] * w[0]
            if abs(cross) < EPS:
                continue                                   # on the edge
            s = 1 if cross > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
        return True


class Gem:
    """A closed solid: a list of outward-facing facets."""

    def __init__(self, facets):
        self.facets = list(facets)

    def intersect(self, o, d):
        """Nearest facet hit. Returns ``(t, facet)`` or ``(None, None)``."""
        best_t, best_f = None, None
        for f in self.facets:
            t = f.intersect(o, d)
            if t is not None and (best_t is None or t < best_t):
                best_t, best_f = t, f
        return best_t, best_f


def _facet(verts, interior):
    """Build a facet whose normal is flipped to point away from ``interior``."""
    f = Facet(verts)
    if np.dot(f.n, f.v.mean(axis=0) - interior) < 0.0:
        f.n = -f.n
    return f


def triangular_prism(side: float = 2.0, depth: float = 2.0,
                     center=(0.0, 0.0, 0.0)) -> Gem:
    """Equilateral triangular prism, apex up, extruded along z.

    The cross-section lies in the xy-plane; rays travelling at constant z
    interact only with the three rectangular side faces (the z-caps are parallel
    to such rays).
    """
    s = side
    h = s * np.sqrt(3.0) / 2.0
    cx, cy, cz = center
    z0, z1 = cz - depth / 2.0, cz + depth / 2.0

    Lb = np.array([cx - s / 2.0, cy - h / 3.0, 0.0])      # left base
    Rb = np.array([cx + s / 2.0, cy - h / 3.0, 0.0])      # right base
    Ap = np.array([cx,           cy + 2.0 * h / 3.0, 0.0])  # apex
    interior = np.array([cx, cy, cz])

    def ext(p, z):
        return p + np.array([0.0, 0.0, z])

    def quad(a, b):
        return [ext(a, z0), ext(b, z0), ext(b, z1), ext(a, z1)]

    return Gem([
        _facet(quad(Lb, Rb), interior),                  # bottom
        _facet(quad(Rb, Ap), interior),                  # right
        _facet(quad(Ap, Lb), interior),                  # left
        _facet([ext(Lb, z0), ext(Rb, z0), ext(Ap, z0)], interior),  # z0 cap
        _facet([ext(Lb, z1), ext(Rb, z1), ext(Ap, z1)], interior),  # z1 cap
    ])


def intersect_plane(o, d, p0, n):
    """Ray/plane intersection distance, or None if parallel/behind."""
    denom = float(np.dot(n, d))
    if abs(denom) < EPS:
        return None
    t = float(np.dot(n, np.asarray(p0, float) - o)) / denom
    return t if t > EPS else None


__all__ = ["Facet", "Gem", "triangular_prism", "intersect_plane", "EPS"]
