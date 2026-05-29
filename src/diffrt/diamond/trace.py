"""Deterministic single-ray tracing through a flat-facet gem.

M1-level: no differentials yet — just follow one ray (one wavelength) through the
specular chain (refract / reflect / TIR) until it leaves the gem. The
beam-differential version comes in M2.
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.optics import reflect, refract


def trace_center_ray(gem, o, d, n_of_lambda, wavelength_um,
                     n_outside: float = 1.0, max_bounces: int = 16):
    """Trace one ray of a single wavelength through ``gem``.

    Args:
        gem: a :class:`~diffrt.diamond.geometry.Gem`.
        o, d: ray origin and (any-length) direction; ``d`` is normalized here.
        n_of_lambda: callable ``λ(µm) -> n`` for the gem interior.
        wavelength_um: the ray's wavelength in µm.

    Returns a dict with:
        path:    list of points [origin, surface hits...] (the in/inside chain),
        exit_o, exit_d: the outgoing ray after the gem (origin just past the
                        last surface, unit direction),
        events:  list of 'enter' / 'exit' / 'tir' per surface interaction,
        exited:  True if the ray left the gem,
        maxed:   True if it hit the bounce cap (likely trapped by TIR).
    """
    o = np.array(o, dtype=float)
    d = np.array(d, dtype=float)
    d = d / np.linalg.norm(d)
    n_in = float(n_of_lambda(wavelength_um))

    path = [o.copy()]
    events: list[str] = []

    for _ in range(max_bounces):
        t, f = gem.intersect(o, d)
        if t is None:                                # nothing ahead → left gem
            return {"path": path, "exit_o": o, "exit_d": d,
                    "events": events, "exited": len(path) > 1, "maxed": False}
        p = o + t * d
        path.append(p.copy())

        entering = np.dot(d, f.n) < 0.0             # ray opposes outward normal
        n_face = f.n if entering else -f.n          # orient against the ray
        eta = (n_outside / n_in) if entering else (n_in / n_outside)

        r = refract(d, n_face, eta)
        if r is None:                                # total internal reflection
            d = reflect(d, n_face)
            events.append("tir")
        else:
            d = r
            events.append("enter" if entering else "exit")
        o = p + d * 1e-7                             # nudge off the surface

    return {"path": path, "exit_o": o, "exit_d": d,
            "events": events, "exited": False, "maxed": True}


__all__ = ["trace_center_ray"]
