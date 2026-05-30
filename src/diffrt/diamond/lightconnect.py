"""Light-side sample-and-connect for diamond fire (MC-1, take 2).

Two passes:
  Pass 1 (sample): emit many rays from the point light through the diamond,
    measure each exit ray's perpendicular distance to the camera pinhole, sort.
  Pass 2 (MC connect): Newton-refine the top-K closest seeds (origin=light,
    target=camera, tuning e1,e2,wl) to minimize that distance; accept seeds whose
    final distance < accept_eps, reject the rest; splat accepted connections onto
    the film coloured by their wavelength wl.

Seeding from the light's closest rays puts the Newton solver inside connecting
facet topologies, which is what the camera-side pixel-grid seeding lacked.

``wl`` denotes wavelength (µm) throughout.
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.manifold import connect_newton_vec
from diffrt.diamond.render import wavelength_to_rgb
from diffrt.diamond.render_photon import (_camera_projector, _sample_cone,
                                          trace_beams_vec)


def ray_point_distance(P, D, C):
    """Perpendicular distance from point ``C`` to rays ``(P, D)`` (D unit).

    Returns ``inf`` where ``C`` is behind the ray (front check)."""
    v = np.asarray(C, float)[None, :] - P
    along = np.sum(v * D, axis=1)
    perp = v - along[:, None] * D
    dist = np.linalg.norm(perp, axis=1)
    return np.where(along > 0.0, dist, np.inf)


def sample_pass(gem, light_pos, camera_C, n_of, dn_of, n_samples, band,
                gem_radius=1.3, max_bounces=18, seed=0):
    """Emit rays from the light, trace, return per-ray distance-to-camera + sort.

    Returns dict: dirs (N,3), wl (N,), P (N,3), D (N,3), dist (N,), order (N,)."""
    rng = np.random.default_rng(seed)
    L = np.asarray(light_pos, float)
    axis = -L                                            # aim at the gem (~origin)
    cone_half = np.arctan(gem_radius / np.linalg.norm(L))

    dirs = _sample_cone(axis, cone_half, n_samples, rng)
    wl = rng.uniform(band[0], band[1], n_samples)
    O = np.broadcast_to(L, dirs.shape).copy()
    res = trace_beams_vec(gem, O, dirs, n_of(wl), dn_of(wl),
                          max_bounces=max_bounces)
    P, D, ex = res["P"], res["D"], res["exited"]
    dist = ray_point_distance(P, D, camera_C)
    dist = np.where(ex, dist, np.inf)
    order = np.argsort(dist)
    return {"dirs": dirs, "wl": wl, "P": P, "D": D, "dist": dist, "order": order}


def render_fire_connect(gem, camera, light_pos, n_of, dn_of, width, height,
                        n_samples=800_000, top_k=80_000, accept_eps=0.03,
                        band=(0.40, 0.70), newton_iters=40, max_bounces=18,
                        gem_radius=1.3, seed=0):
    """Render diamond fire by light-side sample → sort → Newton-connect → splat.

    Returns (img (H,W,3) linear RGB, stats)."""
    C, project = _camera_projector(camera, width, height)
    L = np.asarray(light_pos, float)

    sp = sample_pass(gem, L, C, n_of, dn_of, n_samples, band,
                     gem_radius=gem_radius, max_bounces=max_bounces, seed=seed)

    # top-K closest finite seeds
    top = sp["order"][:top_k]
    top = top[np.isfinite(sp["dist"][top])]

    res = connect_newton_vec(gem, L, sp["dirs"][top], sp["wl"][top], n_of, dn_of,
                             C, max_iter=newton_iters, tol=1e-7,
                             max_bounces=max_bounces, band=band, seed_resid=10.0)
    final_dist = ray_point_distance(res["P"], res["D"], C)
    accept = final_dist < accept_eps

    img = np.zeros((height * width, 3))
    if accept.any():
        Pa, wla = res["P"][accept], res["wl"][accept]
        col, row, valid = project(Pa)
        ci = np.round(col[valid]).astype(int)
        ri = np.round(row[valid]).astype(int)
        idx = ri * width + ci
        rgb = np.array([wavelength_to_rgb(w * 1000.0) for w in wla[valid]])
        np.add.at(img, idx, rgb)

    fd = final_dist[np.isfinite(final_dist)]
    stats = {
        "n_samples": n_samples,
        "n_exited": int(np.isfinite(sp["dist"]).sum()),
        "seeds_refined": int(top.size),
        "n_accepted": int(accept.sum()),
        "accept_rate": float(accept.mean()) if top.size else 0.0,
        "seed_min_dist": float(sp["dist"][sp["order"][0]]),
        "final_min_dist": float(fd.min()) if fd.size else float("inf"),
        "final_median_dist": float(np.median(fd)) if fd.size else float("inf"),
    }
    return img.reshape(height, width, 3), stats


__all__ = ["ray_point_distance", "sample_pass", "render_fire_connect"]
