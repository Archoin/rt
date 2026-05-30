"""Fixed-facet-sequence model: a smooth specular chain for studying the solver.

A free re-trace can switch which facets are hit as parameters change (a
path-topology discontinuity), which breaks the Newton linearization. Here we
**fix** the chain — an ordered list of (facet_index, kind) with
kind ∈ {0=enter, 1=exit, 2=tir} — extracted once from a seed, then evaluate the
path by enforcing that exact sequence of plane refractions/reflections. The
result is a *smooth* map (e1, e2, wl) → exit ray with an exact analytic
Jacobian, so the solver can be studied without topology confounds.

``wl`` is wavelength (µm). Jacobian columns are ∂(P,D)/∂(e1, e2, wl).
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.optics import normalize
from diffrt.diamond.render_photon import _make_frames, _nearest_hit_vec

_NUDGE = 1e-7


def trace_record_sequence(gem, O, D0, wl, n_of, n_outside=1.0, max_bounces=18):
    """Free trace that records each ray's chain. Returns seq_fi, seq_kind
    (N, max_bounces) padded with -1, plus nint, exited (N,)."""
    N = O.shape[0]
    D = D0 / np.linalg.norm(D0, axis=1, keepdims=True)
    P = O.copy()
    ng = np.broadcast_to(np.asarray(n_of(wl), float), (N,)).copy()
    seq_fi = np.full((N, max_bounces), -1, dtype=int)
    seq_kind = np.full((N, max_bounces), -1, dtype=int)
    nint = np.zeros(N, dtype=int)
    exited = np.zeros(N, dtype=bool)
    alive = np.arange(N)
    facets = gem.facets

    for _ in range(max_bounces):
        if alive.size == 0:
            break
        Pa, Da = P[alive], D[alive]
        t, nrm, _p0, found, fi = _nearest_hit_vec(facets, Pa, Da)
        miss = ~found
        if miss.any():
            am = alive[miss]
            exited[am] = nint[am] > 0
        hit = found
        if not hit.any():
            break
        ah = alive[hit]
        Ph, Dh, nh, th, fih = Pa[hit], Da[hit], nrm[hit], t[hit], fi[hit]
        ngh = ng[ah]
        entering = np.sum(Dh * nh, axis=1) < 0.0
        nface = np.where(entering[:, None], nh, -nh)
        eta = np.where(entering, n_outside / ngh, ngh / n_outside)
        c_i = -np.sum(Dh * nface, axis=1)
        tir = eta * eta * (1.0 - c_i * c_i) > 1.0
        kind = np.where(tir, 2, np.where(entering, 0, 1))

        b = nint[ah]
        seq_fi[ah, b] = fih
        seq_kind[ah, b] = kind

        Hh = Ph + th[:, None] * Dh
        Dref = (eta[:, None] * Dh
                + (eta * c_i - np.sqrt(np.clip(1 - eta * eta * (1 - c_i * c_i), 0, None)))[:, None] * nface)
        Drfl = Dh + 2.0 * c_i[:, None] * nface
        newD = np.where(tir[:, None], Drfl, Dref)
        newD /= np.linalg.norm(newD, axis=1, keepdims=True)
        P[ah] = Hh + _NUDGE * newD
        D[ah] = newD
        nint[ah] += 1
        alive = ah

    return seq_fi, seq_kind, nint, exited


def trace_fixed_sequence_vec(facets, seq, O, D0, wl, n_of, dn_of,
                             n_outside=1.0, frame=None):
    """Trace a batch along a FIXED sequence ``seq`` = list of (facet_idx, kind).

    Smooth map; returns dict P, D (N,3), Pj, Dj (N,3,3) [cols ∂/∂(e1,e2,wl)]."""
    N = O.shape[0]
    D = D0 / np.linalg.norm(D0, axis=1, keepdims=True)
    e1, e2 = frame if frame is not None else _make_frames(D)
    P = np.asarray(O, float).copy()
    Pj = np.zeros((N, 3, 3))
    Dj = np.zeros((N, 3, 3))
    Dj[:, :, 0] = e1
    Dj[:, :, 1] = e2
    ng = np.broadcast_to(np.asarray(n_of(wl), float), (N,)).copy()
    dnv = np.broadcast_to(np.asarray(dn_of(wl), float), (N,)).copy()

    for fi, kind in seq:
        f = facets[fi]
        Nf, Q0 = f.n, f.p0
        # differential transfer to this facet's infinite plane
        NdotD = D @ Nf
        tau = ((Q0[None, :] - P) @ Nf) / NdotD
        nPj = np.einsum('k,mkp->mp', Nf, Pj)
        nDj = np.einsum('k,mkp->mp', Nf, Dj)
        tau_s = -(nPj + tau[:, None] * nDj) / NdotD[:, None]
        P = P + tau[:, None] * D
        Pj = Pj + D[:, :, None] * tau_s[:, None, :] + tau[:, None, None] * Dj

        nface = np.where((NdotD < 0)[:, None], Nf[None, :], -Nf[None, :])
        c_i = -np.sum(D * nface, axis=1)
        c_i_s = -np.einsum('mk,mkp->mp', nface, Dj)
        if kind == 2:                                    # reflection (TIR)
            newD = D + 2.0 * c_i[:, None] * nface
            newDj = Dj - 2.0 * nface[:, :, None] * (-c_i_s)[:, None, :]
        else:                                            # refraction enter/exit
            if kind == 0:
                eta = n_outside / ng
                deta = -n_outside * dnv / (ng * ng)
            else:
                eta = ng / n_outside
                deta = dnv / n_outside
            eta_s = np.zeros((N, 3)); eta_s[:, 2] = deta
            k = eta * eta * (1.0 - c_i * c_i)
            c_t = np.sqrt(np.clip(1.0 - k, 0.0, None))
            c_t_safe = np.where(c_t < 1e-8, 1e-8, c_t)
            k_s = (2.0 * eta[:, None] * eta_s * (1.0 - c_i * c_i)[:, None]
                   - 2.0 * (eta * eta)[:, None] * c_i[:, None] * c_i_s)
            c_t_s = -k_s / (2.0 * c_t_safe[:, None])
            newD = eta[:, None] * D + (eta * c_i - c_t)[:, None] * nface
            newDj = (D[:, :, None] * eta_s[:, None, :] + eta[:, None, None] * Dj
                     + nface[:, :, None]
                     * (eta_s * c_i[:, None] + eta[:, None] * c_i_s - c_t_s)[:, None, :])
        newD = newD / np.linalg.norm(newD, axis=1, keepdims=True)
        D, Dj = newD, newDj

    return {"P": P, "D": D, "Pj": Pj, "Dj": Dj}


def connect_fixed_newton(facets, seq, origin, dirs, wl, n_of, dn_of, target,
                         max_iter=40, tol=1e-9, band=(0.40, 0.70),
                         backtrack=True):
    """Gauss-Newton on the fixed-sequence model: steer (e1,e2,wl) so the exit
    ray passes through ``target``. Returns dict with P, D, wl, resid (|f|),
    plus per-iteration mean |f|, accepts, rejects (for studying convergence)."""
    N = dirs.shape[0]
    D = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).copy()
    wl = np.broadcast_to(wl, (N,)).astype(float).copy()
    target = np.asarray(target, float)
    O = np.broadcast_to(np.asarray(origin, float), (N, 3)).copy()

    def evalf(Dx, wlx):
        r = trace_fixed_sequence_vec(facets, seq, O, Dx, wlx, n_of, dn_of)
        TP = target[None, :] - r["P"]
        f = np.cross(TP, r["D"])
        return r, TP, f, np.linalg.norm(f, axis=1)

    r, TP, f, dist = evalf(D, wl)
    alpha = np.ones(N)
    hist = {"mean_dist": [], "accepts": [], "rejects": []}

    for _ in range(max_iter):
        Jf = np.empty((N, 3, 3))
        for k in range(3):
            Jf[:, :, k] = -np.cross(r["Pj"][:, :, k], r["D"]) + np.cross(TP, r["Dj"][:, :, k])
        step = -np.einsum('nij,nj->ni', np.linalg.pinv(Jf), f)
        sc = alpha if backtrack else np.ones(N)
        e1, e2 = _make_frames(D)
        Dp = D + (sc[:, None] * step[:, 0:1]) * e1 + (sc[:, None] * step[:, 1:2]) * e2
        Dp /= np.linalg.norm(Dp, axis=1, keepdims=True)
        wlp = np.clip(wl + sc * step[:, 2], band[0], band[1])
        rp, TPp, fp, distp = evalf(Dp, wlp)

        better = distp < dist if backtrack else np.ones(N, bool)
        D[better], wl[better] = Dp[better], wlp[better]
        r["P"][better], r["D"][better] = rp["P"][better], rp["D"][better]
        r["Pj"][better], r["Dj"][better] = rp["Pj"][better], rp["Dj"][better]
        TP[better], f[better], dist[better] = TPp[better], fp[better], distp[better]
        alpha[better] = np.minimum(alpha[better] * 1.5, 4.0)
        alpha[~better] *= 0.5
        hist["mean_dist"].append(float(np.median(dist)))
        hist["accepts"].append(int(better.sum()))
        hist["rejects"].append(int((~better).sum()))
        if np.all(dist < tol):
            break

    return {"P": r["P"], "D": r["D"], "wl": wl, "resid": dist, "hist": hist}


__all__ = ["trace_record_sequence", "trace_fixed_sequence_vec",
           "connect_fixed_newton"]
