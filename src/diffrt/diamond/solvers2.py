"""2nd-pass solver experiments on the fixed-sequence model.

Four methods, same interface ``solve(method, facets, seq, origin, dirs, wl,
n_of, dn_of, target) -> {converged, P, D, emit, wl, resid, iters}``:

  "gn_wl"        : current — joint 3-DOF Gauss-Newton (e1,e2,wl), per-ray column
                   equilibration + line search.
  "gn_freq"      : same, but the 3rd DOF is FREQUENCY ν=1/λ instead of wavelength.
  "gauss_seidel" : alternate — fix angle, 1-DOF update of wl; then fix wl, 2-DOF
                   update of angle. Decouples the incommensurate DOFs.
  "hessian"      : 2nd-order Newton on ½‖f‖² with the FULL Hessian
                   (JᵀJ + Σ fᵢ ∂²fᵢ), the ∂²f obtained by finite-differencing the
                   analytic Jacobian; Levenberg-damped + line search.

All share the backtracking line search (best step fraction per ray per iter).
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.fixedseq import trace_fixed_sequence_vec
from diffrt.diamond.render_photon import _make_frames

FRACS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def _eval(facets, seq, O, D, wl, n_of, dn_of, target, frame=None):
    r = trace_fixed_sequence_vec(facets, seq, O, D, wl, n_of, dn_of, frame=frame)
    TP = target[None, :] - r["P"]
    f = np.cross(TP, r["D"])
    return r, TP, f, np.linalg.norm(f, axis=1)


def _assemble_J(r, TP):
    """J (N,3,3): columns ∂f/∂(e1, e2, wl)."""
    J = np.empty((r["P"].shape[0], 3, 3))
    for k in range(3):
        J[:, :, k] = -np.cross(r["Pj"][:, :, k], r["D"]) + np.cross(TP, r["Dj"][:, :, k])
    return J


def _param(method, wl, band):
    """Return (p, to_l, dl_dp, pband) for the 3rd-DOF parametrization."""
    if method == "gn_freq":
        return (1.0 / wl, lambda p: 1.0 / p, lambda p: -1.0 / (p * p),
                (1.0 / band[1], 1.0 / band[0]))
    return wl.copy(), (lambda p: p), (lambda p: np.ones_like(p)), band


def _linesearch(facets, seq, O, D, p, ang_step, p_step, to_l, pband, dist,
                n_of, dn_of, target, fracs=FRACS):
    N = D.shape[0]
    best_dist = dist.copy(); best_frac = np.zeros(N)
    for fr in fracs:
        Dp = D + fr * ang_step
        Dp /= np.linalg.norm(Dp, axis=1, keepdims=True)
        pp = np.clip(p + fr * p_step, pband[0], pband[1])
        _, _, _, dp = _eval(facets, seq, O, Dp, to_l(pp), n_of, dn_of, target)
        better = dp < best_dist
        best_dist = np.where(better, dp, best_dist)
        best_frac = np.where(better, fr, best_frac)
    moved = best_frac > 0
    Dn = D + best_frac[:, None] * ang_step
    Dn /= np.linalg.norm(Dn, axis=1, keepdims=True)
    pn = np.clip(p + best_frac * p_step, pband[0], pband[1])
    D = np.where(moved[:, None], Dn, D)
    p = np.where(moved, pn, p)
    wl = to_l(p)
    r, TP, f, dist = _eval(facets, seq, O, D, wl, n_of, dn_of, target)
    return D, p, wl, r, TP, f, dist, moved


def solve(method, facets, seq, origin, dirs, wl0, n_of, dn_of, target,
          max_iter=30, tol=1e-9, band=(0.40, 0.70)):
    N = dirs.shape[0]
    D = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).copy()
    wl = np.broadcast_to(wl0, (N,)).astype(float).copy()
    target = np.asarray(target, float)
    O = np.broadcast_to(np.asarray(origin, float), (N, 3)).copy()
    p, to_l, dl_dp, pband = _param(method, wl, band)
    r, TP, f, dist = _eval(facets, seq, O, D, wl, n_of, dn_of, target)
    iters = 0

    for _ in range(max_iter):
        iters += 1
        if method in ("gn_wl", "gn_freq"):
            J = _assemble_J(r, TP)
            J[:, :, 2] = J[:, :, 2] * dl_dp(p)[:, None]          # ∂f/∂p
            cn = np.linalg.norm(J, axis=1); cn = np.where(cn < 1e-12, 1.0, cn)
            delta = np.nan_to_num(
                -np.einsum('nij,nj->ni', np.linalg.pinv(J / cn[:, None, :]), f) / cn)
            e1, e2 = _make_frames(D)
            ang = delta[:, 0:1] * e1 + delta[:, 1:2] * e2
            D, p, wl, r, TP, f, dist, moved = _linesearch(
                facets, seq, O, D, p, ang, delta[:, 2], to_l, pband, dist,
                n_of, dn_of, target)

        elif method == "gauss_seidel":
            # block A: wl (1-DOF), fix angle
            J = _assemble_J(r, TP)
            col = J[:, :, 2]
            dp = -np.sum(col * f, axis=1) / (np.sum(col * col, axis=1) + 1e-20)
            D, p, wl, r, TP, f, dist, _ = _linesearch(
                facets, seq, O, D, p, np.zeros_like(D), dp, to_l, pband, dist,
                n_of, dn_of, target)
            # block B: angle (2-DOF), fix wl
            J = _assemble_J(r, TP)
            d2 = -np.einsum('nij,nj->ni', np.linalg.pinv(J[:, :, :2]), f)
            e1, e2 = _make_frames(D)
            ang = d2[:, 0:1] * e1 + d2[:, 1:2] * e2
            D, p, wl, r, TP, f, dist, moved = _linesearch(
                facets, seq, O, D, p, ang, np.zeros(N), to_l, pband, dist,
                n_of, dn_of, target)

        else:  # hessian (param = wl)
            e1, e2 = _make_frames(D)
            J = _assemble_J(r, TP)                               # in (e1,e2) frame
            g = np.einsum('nij,ni->nj', J, f)                    # JᵀT f  (N,3)
            d = 1e-5
            dJ = np.zeros((N, 3, 3, 3))                          # ∂J_ij/∂Δ_k
            for k in range(3):
                if k < 2:
                    vec = e1 if k == 0 else e2
                    Dp = D + d * vec; Dp /= np.linalg.norm(Dp, axis=1, keepdims=True)
                    Dm = D - d * vec; Dm /= np.linalg.norm(Dm, axis=1, keepdims=True)
                    rp, TPp, *_ = _eval(facets, seq, O, Dp, wl, n_of, dn_of, target, frame=(e1, e2))
                    rm, TPm, *_ = _eval(facets, seq, O, Dm, wl, n_of, dn_of, target, frame=(e1, e2))
                else:
                    rp, TPp, *_ = _eval(facets, seq, O, D, wl + d, n_of, dn_of, target, frame=(e1, e2))
                    rm, TPm, *_ = _eval(facets, seq, O, D, wl - d, n_of, dn_of, target, frame=(e1, e2))
                dJ[:, :, :, k] = (_assemble_J(rp, TPp) - _assemble_J(rm, TPm)) / (2 * d)
            H = np.einsum('nki,nkj->nij', J, J) + np.einsum('ni,nijk->njk', f, dJ)
            H = 0.5 * (H + np.transpose(H, (0, 2, 1)))
            mu = 1e-6 + 1e-3 * np.einsum('nii->n', H)[:, None, None] / 3.0
            H = H + mu * np.eye(3)[None]
            delta = np.linalg.solve(H, -g[..., None])[..., 0]    # (de1,de2,dwl)
            ang = delta[:, 0:1] * e1 + delta[:, 1:2] * e2
            D, p, wl, r, TP, f, dist, moved = _linesearch(
                facets, seq, O, D, p, ang, delta[:, 2], to_l, pband, dist,
                n_of, dn_of, target)

        if not moved.any() or (dist < tol).all():
            break

    return {"converged": dist < tol, "P": r["P"], "D": r["D"], "emit": D,
            "wl": wl, "resid": dist, "iters": iters}


__all__ = ["solve"]
