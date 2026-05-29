"""Beam / ray differentials through a flat-facet specular chain.

Each beam carries, alongside its position ``P`` and unit direction ``D``, the
Jacobians of ``(P, D)`` with respect to the beam parameters ``(θ, φ, λ)``:

    Pj[:, k] = ∂P/∂param_k ,   Dj[:, k] = ∂D/∂param_k ,   k ∈ {θ, φ, λ}

θ, φ are small-angle offsets of the emission direction in an orthonormal frame
``(e1, e2)`` ⟂ the central direction; λ is the wavelength (µm). The novel piece
is the **wavelength column**: at refraction, the index ``n(λ)`` makes ``eta``
depend on λ, so ``∂D/∂λ`` becomes non-zero with magnitude ∝ ``dn/dλ`` — the
analytic spread that produces dispersion ("fire").

Surfaces are **flat** (constant normal), so the normal contributes no
derivative terms — the easy and exact case for faceted gems.

All formulas are validated against finite differences in
``scripts/diff_fd_check.py``.
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.optics import normalize

_EPS_NUDGE = 1e-7


def make_frame(d0):
    """Return an orthonormal ``(e1, e2)`` spanning the plane ⟂ ``d0``."""
    d0 = normalize(d0)
    helper = np.array([0.0, 0.0, 1.0]) if abs(d0[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    e1 = normalize(np.cross(helper, d0))
    e2 = np.cross(d0, e1)
    return e1, e2


def _transfer(P, Pj, D, Dj, N, Q0):
    """Free flight to the plane ``(N, Q0)``. Direction is unchanged.

    H = P + τ D,  τ = N·(Q0−P) / (N·D),
    τ derivative (flat plane, constant N):  τ_s = −(N·P_s + τ N·D_s) / (N·D).
    """
    NdotD = float(N @ D)
    tau = float(N @ (Q0 - P)) / NdotD
    tau_s = -(N @ Pj + tau * (N @ Dj)) / NdotD          # (3,)
    H = P + tau * D
    Hj = Pj + np.outer(D, tau_s) + tau * Dj
    return H, Hj, tau


def _refract_diff(D, Dj, n_face, eta, eta_s):
    """Differential Snell refraction. ``n_face`` faces against the ray.

    Returns ``(D', D'j)`` or ``(None, None)`` on total internal reflection.
    ``eta_s`` is ∂eta/∂param (non-zero only for the λ column → dispersion).
    """
    c_i = -float(D @ n_face)
    c_i_s = -(n_face @ Dj)                               # (3,)
    k = eta * eta * (1.0 - c_i * c_i)
    if k > 1.0:
        return None, None                                # TIR
    c_t = np.sqrt(1.0 - k)
    k_s = 2.0 * eta * eta_s * (1.0 - c_i * c_i) - 2.0 * eta * eta * c_i * c_i_s
    c_t_s = -k_s / (2.0 * c_t)
    Dp = eta * D + (eta * c_i - c_t) * n_face
    Dpj = (np.outer(D, eta_s) + eta * Dj
           + np.outer(n_face, eta_s * c_i + eta * c_i_s - c_t_s))
    return Dp, Dpj


def _reflect_diff(D, Dj, n_face):
    """Differential mirror reflection (used for TIR)."""
    a = float(D @ n_face)
    a_s = n_face @ Dj                                    # (3,)
    Dp = D - 2.0 * a * n_face
    Dpj = Dj - 2.0 * np.outer(n_face, a_s)
    return Dp, Dpj


def trace_beam(gem, P0, D0, n_of, dn_of, wavelength_um, frame=None,
               screen=None, n_outside=1.0, max_bounces=16):
    """Trace a beam with differentials through ``gem``.

    Args:
        n_of:  callable ``λ(µm) -> n``.
        dn_of: callable ``λ(µm) -> dn/dλ`` (per µm).
        frame: optional ``(e1, e2)``; if None, built from ``D0``.
        screen: optional ``(Q0, N)`` plane to land on (gives ``X, Xj``).

    Returns dict with D, Dj, P, Pj, optional X/Xj (screen landing + Jacobian),
    events, and ``exited``.
    """
    e1, e2 = frame if frame is not None else make_frame(D0)
    D = normalize(D0)
    Dj = np.column_stack([e1, e2, np.zeros(3)])          # ∂D/∂θ, ∂D/∂φ, ∂D/∂λ
    P = np.asarray(P0, dtype=float)
    Pj = np.zeros((3, 3))                                # point source: P fixed
    n_in = float(n_of(wavelength_um))
    dn = float(dn_of(wavelength_um))

    events: list[str] = []
    exited = False
    for _ in range(max_bounces):
        t, f = gem.intersect(P, D)
        if t is None:
            exited = len(events) > 0
            break
        P, Pj, _ = _transfer(P, Pj, D, Dj, f.n, f.p0)

        entering = (D @ f.n) < 0.0
        n_face = f.n if entering else -f.n
        if entering:
            eta = n_outside / n_in
            deta = -n_outside * dn / (n_in * n_in)       # ∂(n_out/n_in)/∂λ
        else:
            eta = n_in / n_outside
            deta = dn / n_outside                        # ∂(n_in/n_out)/∂λ
        eta_s = np.array([0.0, 0.0, deta])

        Dp, Dpj = _refract_diff(D, Dj, n_face, eta, eta_s)
        if Dp is None:
            D, Dj = _reflect_diff(D, Dj, n_face)
            events.append("tir")
        else:
            D, Dj = Dp, Dpj
            events.append("enter" if entering else "exit")

        P = P + _EPS_NUDGE * D                            # step off the surface
        Pj = Pj + _EPS_NUDGE * Dj

    out = {"D": D, "Dj": Dj, "P": P, "Pj": Pj,
           "events": events, "exited": exited}
    if screen is not None:
        Q0s, Ns = screen
        X, Xj, _ = _transfer(P, Pj, D, Dj, np.asarray(Ns, float),
                             np.asarray(Q0s, float))
        out["X"], out["Xj"] = X, Xj
    return out


__all__ = ["make_frame", "trace_beam"]
