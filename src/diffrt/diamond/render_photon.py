"""Light-side progressive photon-differential splatting (Direction 1).

Trace photon beams from the point light through the diamond carrying the
beam-differential footprint, then splat each exit onto the camera film, coloured
by wavelength (dn/dλ). Progressive (SPPM-style) radius reduction makes it
consistent. Requires a vectorized beam-differential tracer (this module), since
many beams are needed.

The vectorized tracer is the batched form of the scalar, M2-validated
``differentials.trace_beam`` (single path: transmit, or reflect on TIR). It is
checked against the scalar version in scripts/check_vec_tracer.py.
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.optics import normalize
from diffrt.diamond.render import wavelength_to_rgb

_EPS = 1e-9
_NUDGE = 1e-7


def _make_frames(D):
    """Per-row orthonormal (e1, e2) ⟂ D. D: (N,3) unit. Returns (N,3),(N,3)."""
    helper = np.where(np.abs(D[:, 2:3]) < 0.9,
                      np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    e1 = np.cross(helper, D)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(D, e1)
    return e1, e2


def _nearest_hit_vec(facets, O, D):
    """Nearest triangle hit per ray. Returns (t, normal, p0, found, facet_idx)."""
    K = O.shape[0]
    best_t = np.full(K, np.inf)
    best_n = np.zeros((K, 3))
    best_p0 = np.zeros((K, 3))
    best_fi = np.full(K, -1, dtype=int)
    found = np.zeros(K, dtype=bool)
    for fi, f in enumerate(facets):
        n = f.n
        v0, v1, v2 = f.v[0], f.v[1], f.v[2]
        denom = D @ n
        ok = np.abs(denom) > _EPS
        t = np.where(ok, ((v0 - O) @ n) / np.where(ok, denom, 1.0), -1.0)
        P = O + t[:, None] * D
        c0 = np.sum(np.cross(v1 - v0, P - v0) * n, axis=1)
        c1 = np.sum(np.cross(v2 - v1, P - v1) * n, axis=1)
        c2 = np.sum(np.cross(v0 - v2, P - v2) * n, axis=1)
        inside = ((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | \
                 ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))
        valid = ok & (t > _NUDGE) & inside & (t < best_t)
        best_t = np.where(valid, t, best_t)
        best_n = np.where(valid[:, None], n, best_n)
        best_p0 = np.where(valid[:, None], v0, best_p0)
        best_fi = np.where(valid, fi, best_fi)
        found |= valid
    return best_t, best_n, best_p0, found, best_fi


def trace_beams_vec(gem, O, D0, n_glass, dn, n_outside=1.0, max_bounces=18):
    """Batched beam-differential trace (single path: transmit / reflect on TIR).

    O, D0: (N,3). n_glass, dn: (N,) per-beam index and dn/dλ. Carries Jacobians
    Pj, Dj of shape (N,3,3), param axis = (e1, e2, λ). Returns dict with
    P, D (N,3), Pj, Dj (N,3,3), exited (N,) = entered the gem then left.
    """
    N = O.shape[0]
    D = D0 / np.linalg.norm(D0, axis=1, keepdims=True)
    e1, e2 = _make_frames(D)
    P = O.copy()
    Pj = np.zeros((N, 3, 3))
    Dj = np.zeros((N, 3, 3))
    Dj[:, :, 0] = e1
    Dj[:, :, 1] = e2
    n_glass = np.broadcast_to(np.asarray(n_glass, float), (N,)).copy()
    dn = np.broadcast_to(np.asarray(dn, float), (N,)).copy()

    exited = np.zeros(N, dtype=bool)
    nint = np.zeros(N, dtype=int)
    n_refr = np.zeros(N, dtype=int)
    n_refl = np.zeros(N, dtype=int)
    alive = np.arange(N)
    facets = gem.facets

    for _ in range(max_bounces):
        if alive.size == 0:
            break
        Pa, Da = P[alive], D[alive]
        Pja, Dja = Pj[alive], Dj[alive]
        t, nrm, p0, found, _fi = _nearest_hit_vec(facets, Pa, Da)

        miss = ~found
        if miss.any():
            am = alive[miss]
            exited[am] = nint[am] > 0                 # left after interacting
        hit = found
        if not hit.any():
            break
        ah = alive[hit]
        Ph, Dh, Pjh, Djh = Pa[hit], Da[hit], Pja[hit], Dja[hit]
        nh, p0h, th = nrm[hit], p0[hit], t[hit]
        ng, dnh = n_glass[ah], dn[ah]

        # --- differential transfer to the facet plane (flat: N constant) ----
        NdotD = np.sum(nh * Dh, axis=1)
        tau = np.sum(nh * (p0h - Ph), axis=1) / NdotD
        nPj = np.einsum('mk,mkp->mp', nh, Pjh)
        nDj = np.einsum('mk,mkp->mp', nh, Djh)
        tau_s = -(nPj + tau[:, None] * nDj) / NdotD[:, None]
        Hh = Ph + tau[:, None] * Dh
        Hjh = (Pjh + Dh[:, :, None] * tau_s[:, None, :]
               + tau[:, None, None] * Djh)

        entering = NdotD < 0.0
        nface = np.where(entering[:, None], nh, -nh)
        eta = np.where(entering, n_outside / ng, ng / n_outside)
        deta = np.where(entering, -n_outside * dnh / (ng * ng), dnh / n_outside)
        eta_s = np.zeros((ah.size, 3))
        eta_s[:, 2] = deta

        # --- differential refraction (and reflection for TIR) ---------------
        c_i = -np.sum(Dh * nface, axis=1)
        c_i_s = -np.einsum('mk,mkp->mp', nface, Djh)
        k = eta * eta * (1.0 - c_i * c_i)
        tir = k > 1.0
        c_t = np.sqrt(np.clip(1.0 - k, 0.0, None))
        c_t_safe = np.where(tir, 1.0, np.where(c_t < 1e-8, 1e-8, c_t))
        k_s = (2.0 * eta[:, None] * eta_s * (1.0 - c_i * c_i)[:, None]
               - 2.0 * (eta * eta)[:, None] * c_i[:, None] * c_i_s)
        c_t_s = -k_s / (2.0 * c_t_safe[:, None])

        Dref = eta[:, None] * Dh + (eta * c_i - c_t)[:, None] * nface
        Drefj = (Dh[:, :, None] * eta_s[:, None, :] + eta[:, None, None] * Djh
                 + nface[:, :, None]
                 * (eta_s * c_i[:, None] + eta[:, None] * c_i_s - c_t_s)[:, None, :])

        a_s = np.einsum('mk,mkp->mp', nface, Djh)        # reflection (= -c_i col)
        Drfl = Dh + 2.0 * c_i[:, None] * nface
        Drflj = Djh - 2.0 * nface[:, :, None] * a_s[:, None, :]

        newD = np.where(tir[:, None], Drfl, Dref)
        newDj = np.where(tir[:, None, None], Drflj, Drefj)
        newD /= np.linalg.norm(newD, axis=1, keepdims=True)

        P[ah] = Hh + _NUDGE * newD
        D[ah] = newD
        Pj[ah] = Hjh + _NUDGE * newDj
        Dj[ah] = newDj
        nint[ah] += 1
        n_refl[ah[tir]] += 1
        n_refr[ah[~tir]] += 1
        alive = ah

    return {"P": P, "D": D, "Pj": Pj, "Dj": Dj, "exited": exited,
            "nint": nint, "n_refr": n_refr, "n_refl": n_refl}


def trace_rays_vec(gem, O, D0, n_glass, n_outside=1.0, max_bounces=18):
    """Central-ray-only batched tracer — **no Jacobian**, memory-light.

    Same specular chain as ``trace_beams_vec`` (transmit / reflect on TIR) but
    carries only position and direction, so the sample pass can use far more
    rays. Returns P, D (N,3), exited, nint, n_refr, n_refl (N,).
    """
    N = O.shape[0]
    D = D0 / np.linalg.norm(D0, axis=1, keepdims=True)
    P = O.copy()
    n_glass = np.broadcast_to(np.asarray(n_glass, float), (N,)).copy()
    exited = np.zeros(N, dtype=bool)
    nint = np.zeros(N, dtype=int)
    n_refr = np.zeros(N, dtype=int)
    n_refl = np.zeros(N, dtype=int)
    alive = np.arange(N)
    facets = gem.facets

    for _ in range(max_bounces):
        if alive.size == 0:
            break
        Pa, Da = P[alive], D[alive]
        t, nrm, _p0, found, _fi = _nearest_hit_vec(facets, Pa, Da)
        miss = ~found
        if miss.any():
            am = alive[miss]
            exited[am] = nint[am] > 0
        hit = found
        if not hit.any():
            break
        ah = alive[hit]
        Ph, Dh, nh, th = Pa[hit], Da[hit], nrm[hit], t[hit]
        ng = n_glass[ah]

        Hh = Ph + th[:, None] * Dh
        entering = np.sum(Dh * nh, axis=1) < 0.0
        nface = np.where(entering[:, None], nh, -nh)
        eta = np.where(entering, n_outside / ng, ng / n_outside)
        c_i = -np.sum(Dh * nface, axis=1)
        k = eta * eta * (1.0 - c_i * c_i)
        tir = k > 1.0
        c_t = np.sqrt(np.clip(1.0 - k, 0.0, None))
        Dref = eta[:, None] * Dh + (eta * c_i - c_t)[:, None] * nface
        Drfl = Dh + 2.0 * c_i[:, None] * nface
        newD = np.where(tir[:, None], Drfl, Dref)
        newD /= np.linalg.norm(newD, axis=1, keepdims=True)

        P[ah] = Hh + _NUDGE * newD
        D[ah] = newD
        nint[ah] += 1
        n_refl[ah[tir]] += 1
        n_refr[ah[~tir]] += 1
        alive = ah

    return {"P": P, "D": D, "exited": exited,
            "nint": nint, "n_refr": n_refr, "n_refl": n_refl}


def _camera_projector(camera, W, H):
    """Return (C, project) where project(X (M,3)) -> (col, row, valid)."""
    C = np.asarray(camera[0], float)
    fwd = normalize(np.asarray(camera[1], float) - C)
    right = normalize(np.cross(fwd, np.asarray(camera[2], float)))
    upv = np.cross(right, fwd)
    half = np.tan(np.radians(camera[3]) / 2.0)
    aspect = W / H

    def project(X):
        v = X - C
        vf = v @ fwd
        valid = vf > 1e-6
        vfs = np.where(valid, vf, 1.0)
        gx = (v @ right) / vfs
        gy = -(v @ upv) / vfs
        col = ((gx / (half * aspect)) + 1.0) * 0.5 * W - 0.5
        row = ((gy / half) + 1.0) * 0.5 * H - 0.5
        valid &= (col >= 0) & (col < W) & (row >= 0) & (row < H)
        return col, row, valid

    return C, project


def _sample_cone(axis, half_angle, M, rng):
    """M uniformly-sampled unit directions within a cone about ``axis``."""
    a = normalize(axis)
    t = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = normalize(np.cross(t, a))
    e2 = np.cross(a, e1)
    cos_t = 1.0 - rng.random(M) * (1.0 - np.cos(half_angle))
    sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, None))
    phi = 2.0 * np.pi * rng.random(M)
    return (cos_t[:, None] * a + (sin_t * np.cos(phi))[:, None] * e1
            + (sin_t * np.sin(phi))[:, None] * e2)


def render_fire_photon(gem, camera, light_pos, n_of, dn_of, width, height,
                       band=(0.40, 0.70), beams_per_pass=200_000, passes=4,
                       gather_deg=2.5, gem_radius=1.3, max_bounces=18,
                       dens_clamp=50.0, seed=0):
    """Light-side progressive photon-differential splat → fire image.

    Photon beams from the point light refract through the diamond; those exiting
    toward the camera are splatted at their exit point, weighted by the
    photon-differential flux density (1/footprint-area, clamped) and a Gaussian
    of the angular miss, coloured by wavelength. The gather angle shrinks each
    pass (progressive). Returns (img (H,W,3), lit_count).
    """
    rng = np.random.default_rng(seed)
    C, project = _camera_projector(camera, width, height)
    L = np.asarray(light_pos, float)
    axis = -L                                            # aim at gem centre (~origin)
    dist_lc = np.linalg.norm(axis)
    cone_half = np.arctan(gem_radius / dist_lc)
    cell_solid = 2.0 * np.pi * (1.0 - np.cos(cone_half))

    img = np.zeros((height * width, 3))
    lit = 0
    for p in range(passes):
        gather = np.radians(gather_deg) * (0.6 ** p)     # shrink per pass
        cell_half = np.sqrt(cell_solid / beams_per_pass / np.pi)

        d = _sample_cone(axis, cone_half, beams_per_pass, rng)
        lam = rng.uniform(band[0], band[1], beams_per_pass)
        O = np.broadcast_to(L, d.shape).copy()
        res = trace_beams_vec(gem, O, d, n_of(lam), dn_of(lam),
                              max_bounces=max_bounces)
        ex = res["exited"]
        if not ex.any():
            continue
        P, D, Pj = res["P"][ex], res["D"][ex], res["Pj"][ex]
        lam_e = lam[ex]

        toC = C[None, :] - P
        dist = np.linalg.norm(toC, axis=1)
        cos_ang = np.sum(D * (toC / dist[:, None]), axis=1)
        conn = cos_ang > np.cos(gather)
        if not conn.any():
            continue
        Pc, lamc = P[conn], lam_e[conn]
        cosc = cos_ang[conn]
        # photon-differential flux density: 1 / exit footprint area
        area = np.linalg.norm(np.cross(Pj[conn, :, 0], Pj[conn, :, 1]), axis=1)
        area = area * cell_half * cell_half
        dens = np.minimum(1.0 / (area + 1e-12), dens_clamp)

        col, row, valid = project(Pc)
        if not valid.any():
            continue
        ci = np.round(col[valid]).astype(int)
        ri = np.round(row[valid]).astype(int)
        idx = ri * width + ci
        ang = np.arccos(np.clip(cosc[valid], -1.0, 1.0))
        w = dens[valid] * np.exp(-(ang / gather) ** 2)
        rgb = np.array([wavelength_to_rgb(l * 1000.0) for l in lamc[valid]])
        np.add.at(img, idx, w[:, None] * rgb)
        lit += int(valid.sum())

    img /= (beams_per_pass * passes)
    return img.reshape(height, width, 3), lit


def _projection_jacobian(project, P, eps=1e-4):
    """Central-difference Jacobian d(col,row)/dP of the camera projection.

    Returns (N,2,3). project()'s returned (col,row) are the raw projected
    coordinates (the in-frame clamp only affects its `valid` flag), so a tiny
    central difference about an in-front P is accurate.
    """
    n = P.shape[0]
    J = np.zeros((n, 2, 3))
    for k in range(3):
        e = np.zeros(3); e[k] = eps
        cp, rp, _ = project(P + e)
        cm, rm, _ = project(P - e)
        J[:, 0, k] = (cp - cm) / (2.0 * eps)
        J[:, 1, k] = (rp - rm) / (2.0 * eps)
    return J


def render_fire_differential(gem, camera, light_pos, n_of, dn_of, width, height,
                             band=(0.40, 0.70), n_samples=2_000_000, batches=8,
                             aperture=0.10, gem_radius=1.3, max_bounces=18,
                             min_foot=0.7, max_foot=4.0, win=13, seed=0):
    """Light-traced fire via beam-differential footprint splatting (finite aperture).

    Physically-correct replacement for the (measure-zero) point-light exact
    connection. Photons are emitted from `light_pos` over the gem cone, traced
    through the **bounded** gem (valid paths only). A photon contributes if its
    exit ray crosses the lens plane within `aperture` of the camera centre `C`
    (the finite aperture is what turns the measure-zero connection into positive
    measure). The lens is focused on the gem, so the film point is `project(P)`;
    each photon is splatted as a normalized Gaussian whose covariance is the beam
    differential footprint `Σ = JJᵀ`, `J = (∂U/∂P)·∂P/∂(e1,e2)·Δ`, coloured by
    wavelength. Dispersion (different wl → different exit ray → different U) fans
    each glint into a rainbow streak — the fire.

    Footprint is clamped to [`min_foot`, `max_foot`] px (regularize + uniform
    shrink) and splatted on a `win`×`win` window. Returns (img (H,W,3),
    n_accepted, aperture_distances (M,)).
    """
    rng = np.random.default_rng(seed)
    C, project = _camera_projector(camera, width, height)
    eye = np.asarray(camera[0], float)
    fwd = normalize(np.asarray(camera[1], float) - eye)
    L = np.asarray(light_pos, float)
    axis = -L
    cone = np.arctan(gem_radius / np.linalg.norm(axis))
    Omega = 2.0 * np.pi * (1.0 - np.cos(cone))
    Delta = np.sqrt(Omega / max(n_samples, 1))           # angular sample spacing (rad)

    img = np.zeros((height * width, 3))
    accepted = 0
    dists = []
    r = win // 2
    offs = np.arange(-r, r + 1)
    oy, ox = np.meshgrid(offs, offs, indexing="ij")
    ox = ox.ravel(); oy = oy.ravel()                     # (win²,)
    per = int(np.ceil(n_samples / batches))

    for _ in range(batches):
        d = _sample_cone(axis, cone, per, rng)
        wl = rng.uniform(band[0], band[1], per)
        O = np.broadcast_to(L, d.shape).copy()
        res = trace_beams_vec(gem, O, d, n_of(wl), dn_of(wl), max_bounces=max_bounces)
        ex = res["exited"]
        if not ex.any():
            continue
        P, D, Pj, wle = res["P"][ex], res["D"][ex], res["Pj"][ex], wl[ex]

        denom = D @ fwd
        ok = np.abs(denom) > _EPS
        tpl = np.where(ok, ((eye - P) @ fwd) / np.where(ok, denom, 1.0), -1.0)
        Q = P + tpl[:, None] * D                         # lens-plane crossing
        qd = np.linalg.norm(Q - C, axis=1)
        passes = ok & (tpl > 0.0) & (qd < aperture)
        if not passes.any():
            continue
        Pp, Pjp, wlp = P[passes], Pj[passes], wle[passes]

        col, row, _ = project(Pp)
        dUdP = _projection_jacobian(project, Pp)          # (n,2,3)
        J = np.einsum("nij,njk->nik", dUdP, Pjp[:, :, :2]) * Delta   # (n,2,2) px
        S = np.einsum("nik,njk->nij", J, J)              # J Jᵀ (n,2,2)
        S[:, 0, 0] += min_foot ** 2
        S[:, 1, 1] += min_foot ** 2
        tr = S[:, 0, 0] + S[:, 1, 1]
        det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2
        lmax = 0.5 * (tr + np.sqrt(np.clip(tr * tr - 4.0 * det, 0.0, None)))
        S *= np.minimum(1.0, max_foot ** 2 / np.maximum(lmax, 1e-12))[:, None, None]
        det = np.maximum(S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] ** 2, 1e-12)
        iS00, iS11, iS01 = S[:, 1, 1] / det, S[:, 0, 0] / det, -S[:, 0, 1] / det

        ci0 = np.round(col).astype(int); ri0 = np.round(row).astype(int)
        cc = ci0[:, None] + ox[None, :]; rr = ri0[:, None] + oy[None, :]   # (n,win²)
        dx = cc - col[:, None]; dy = rr - row[:, None]
        qf = iS00[:, None] * dx * dx + 2.0 * iS01[:, None] * dx * dy + iS11[:, None] * dy * dy
        inb = (cc >= 0) & (cc < width) & (rr >= 0) & (rr < height)
        wgt = np.where(inb, np.exp(-0.5 * qf), 0.0)
        wsum = wgt.sum(axis=1, keepdims=True)
        wgt = wgt / np.where(wsum > 0, wsum, 1.0)
        rgb = np.array([wavelength_to_rgb(l * 1000.0) for l in wlp])       # (n,3)
        contrib = wgt[:, :, None] * rgb[:, None, :]                        # (n,win²,3)
        idx = rr * width + cc
        np.add.at(img, idx[inb], contrib[inb])
        accepted += int(passes.sum())
        dists.append(qd[passes])

    img /= max(n_samples, 1)
    return (img.reshape(height, width, 3), accepted,
            np.concatenate(dists) if dists else np.zeros(0))


__all__ = ["trace_beams_vec", "trace_rays_vec", "render_fire_photon",
           "render_fire_differential"]

