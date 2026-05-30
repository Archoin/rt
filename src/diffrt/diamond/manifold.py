"""MC-1: Newton manifold projection — exact specular connection to the point light.

Camera-side, fixed-pinhole connection view. For a camera ray (direction) at a
fixed wavelength λ, Newton-iterate the **2-DOF direction** so that the beam's
exit ray passes exactly through the point light L. The connecting paths form the
1-DOF manifold swept by λ; this is the manifold-walk primitive the MCMC methods
need, and on its own it yields exact (un-blurred) fire curves.

Vectorized over many (seed direction, λ) pairs via trace_beams_vec.

Newton step: residual f(d) = (L − P(d)) × D(d) (rank 2). With the input-frame
footprint columns of the M2 Jacobian,
    Jf[:,k] = −Pj[:,k] × D + (L − P) × Dj[:,k],   k = e1, e2,
solve Jf Δ = −f (normal equations, 2×2) and update d += Δ·(e1,e2).
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.render import make_camera_rays, wavelength_to_rgb
from diffrt.diamond.render_photon import (_camera_projector, _make_frames,
                                          trace_beams_vec)


def _eval(gem, Ot, D, wl, n_of, dn_of, target, max_bounces):
    """Trace and return (res, residual rn, TP). rn = sin(miss to target), inf if
    not exited. ``wl`` = wavelength (µm)."""
    res = trace_beams_vec(gem, Ot, D, np.asarray(n_of(wl), float),
                          np.asarray(dn_of(wl), float), max_bounces=max_bounces)
    TP = target[None, :] - res["P"]
    dist = np.maximum(np.linalg.norm(TP, axis=1), 1e-12)
    rn = np.linalg.norm(np.cross(TP, res["D"]), axis=1) / dist
    rn = np.where(res["exited"], rn, np.inf)
    return res, rn, TP


def connect_newton_vec(gem, origin, dirs, wl, n_of, dn_of, target, max_iter=40,
                       tol=1e-6, max_bounces=18, band=(0.40, 0.70),
                       seed_resid=0.15, a_min=1e-3):
    """Project (direction, wavelength) seeds onto the 1-DOF connecting manifold.

    Generic: rays leave ``origin`` and we steer them to pass through ``target``
    (camera-side: origin=camera, target=light; light-side: origin=light,
    target=camera). Gauss–Newton with backtracking on the constraint
    ``f = (target − P) × D = 0``, solved over **all 3 DOF** (e1, e2, wl) via a
    min-norm (pseudo-inverse) step — never freezing two and sweeping one. A raw
    step can jump the ray onto different facets (path-topology change), so the
    step is line-searched: tried, re-traced, accepted only if the actual residual
    drops, else the per-seed step scale is halved. Only seeds within
    ``seed_resid`` are pursued.

    Returns dict: converged (N,), P, D (N,3), wl (N,), resid (N,), iters (N,).
    """
    N = dirs.shape[0]
    D = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).copy()
    wl = np.broadcast_to(wl, (N,)).astype(float).copy()
    target = np.asarray(target, float)
    Ot = np.broadcast_to(np.asarray(origin, float), (N, 3)).copy()

    res, rn, TP = _eval(gem, Ot, D, wl, n_of, dn_of, target, max_bounces)
    P, Dd, Pj, Dj = res["P"], res["D"], res["Pj"], res["Dj"]
    converged = np.zeros(N, bool)
    iters = np.zeros(N, int)
    alpha = np.ones(N)
    active = np.where(rn < seed_resid)[0]

    for it in range(max_iter):
        if active.size == 0:
            break
        a = active
        fa = np.cross(TP[a], Dd[a])
        Jf = np.empty((a.size, 3, 3))
        for k in range(3):
            Jf[:, :, k] = -np.cross(Pj[a, :, k], Dd[a]) + np.cross(TP[a], Dj[a, :, k])
        step = -np.einsum('nij,nj->ni', np.linalg.pinv(Jf), fa)   # min-norm

        ang = step[:, :2] * alpha[a, None]
        dl = step[:, 2] * alpha[a]
        e1, e2 = _make_frames(D[a])
        Dp = D[a] + ang[:, 0:1] * e1 + ang[:, 1:2] * e2
        Dp /= np.linalg.norm(Dp, axis=1, keepdims=True)
        wlp = np.clip(wl[a] + dl, band[0], band[1])

        rp, rnp, TPp = _eval(gem, Ot[a], Dp, wlp, n_of, dn_of, target, max_bounces)
        better = rnp < rn[a]
        ia = np.where(better)[0]
        aa = a[ia]
        D[aa], wl[aa], rn[aa], TP[aa] = Dp[ia], wlp[ia], rnp[ia], TPp[ia]
        P[aa], Dd[aa] = rp["P"][ia], rp["D"][ia]
        Pj[aa], Dj[aa] = rp["Pj"][ia], rp["Dj"][ia]
        alpha[aa] = np.minimum(alpha[aa] * 1.5, 4.0)
        alpha[a[np.where(~better)[0]]] *= 0.5
        iters[a] = it + 1

        conv_now = rn[a] < tol
        converged[a[np.where(conv_now)[0]]] = True
        cont = (~conv_now) & (alpha[a] >= a_min)
        active = a[np.where(cont)[0]]

    return {"converged": converged, "P": P, "D": Dd, "wl": wl,
            "resid": rn, "iters": iters}


def render_fire_manifold(gem, camera, light_pos, n_of, dn_of, width, height,
                         n_lambda=18, band=(0.40, 0.70), max_iter=12,
                         tol=1e-7, max_bounces=18):
    """Render exact fire curves: Newton-connect a (pixel × λ) seed grid, splat
    the converged exit points coloured by wavelength. Returns (img, stats)."""
    C, project = _camera_projector(camera, width, height)
    _, dirs_all, (H, W) = make_camera_rays(*camera, width, height)
    L = np.asarray(light_pos, float)

    lams = np.linspace(band[0], band[1], n_lambda)
    seeds_dir = np.repeat(dirs_all, n_lambda, axis=0)
    seeds_lam = np.tile(lams, H * W)

    res = connect_newton_vec(gem, C, seeds_dir, seeds_lam, n_of, dn_of, L,
                             max_iter=max_iter, tol=tol, max_bounces=max_bounces)
    conv = res["converged"]

    img = np.zeros((H * W, 3))
    if conv.any():
        Pc, lamc = res["P"][conv], seeds_lam[conv]
        col, row, valid = project(Pc)
        ci = np.round(col[valid]).astype(int)
        ri = np.round(row[valid]).astype(int)
        idx = ri * W + ci
        rgb = np.array([wavelength_to_rgb(l * 1000.0) for l in lamc[valid]])
        np.add.at(img, idx, rgb)

    stats = {
        "seeds": int(conv.size),
        "converged": int(conv.sum()),
        "success_frac": float(conv.mean()),
        "mean_iters": float(res["iters"][conv].mean()) if conv.any() else 0.0,
        "max_resid": float(res["resid"][conv].max()) if conv.any() else 0.0,
    }
    return img.reshape(H, W, 3), stats


__all__ = ["connect_newton_vec", "render_fire_manifold"]
