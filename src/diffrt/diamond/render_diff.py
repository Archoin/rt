"""Differential (footprint) render: point light + diamond → fire.

Camera-side beam-differential connection — the dual of light-side photon
differentials, with a TRUE point light. For each pixel we trace a camera beam
through the diamond carrying the M2-validated footprint Jacobian
``∂(pos,dir)/∂(e1,e2,λ)``. Two differential quantities then turn the
zero-measure point connection into a finite one:

  * the **spatial footprint** ``∂D/∂(e1,e2)`` — magnified by refraction through
    the stone — gives the beam a finite angular spread (the point light, seen
    through 2+ refractions, is "spread into an area"); and
  * the **wavelength column** ``∂D/∂λ`` (∝ ``dn/dλ``) lets us solve analytically
    for the wavelength whose dispersed exit ray points at the light — the fire
    colour at that pixel.

A pixel lights up when, at its connecting wavelength, the exit ray points at the
point light within the beam's footprint. With the footprint shrunk to zero this
reduces to the black baseline (scripts/diamond_fire.py).
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.differentials import trace_beam
from diffrt.diamond.render import make_camera_rays, wavelength_to_rgb


def render_fire_diff(gem, camera, light_pos, n_of, dn_of, lambda0_um,
                     width, height, gather_px=18.0, band=(0.40, 0.70),
                     max_bounces=18):
    """Render the diamond's fire by connecting each pixel beam to the point light.

    For each pixel we solve (linearized) for the footprint offset
    ``Δ=(Δe1,Δe2,Δλ)`` whose exit ray passes through the point light, picking
    the wavelength (via ``∂D/∂λ ∝ dn/dλ``) that minimizes the needed spatial
    offset. A pixel within ``gather_px`` (in pixel-footprint units) of an exact
    connection lights up, weighted by a Gaussian of that offset and coloured by
    the connecting wavelength. ``gather_px → 0`` recovers the black baseline.

    Returns (img (H,W,3) linear RGB, lit_fraction).
    """
    C = np.asarray(camera[0], dtype=float)
    L = np.asarray(light_pos, dtype=float)
    _, dirs, (H, W) = make_camera_rays(*camera, width, height)
    img = np.zeros((H * W, 3))

    pix_half = np.radians(camera[3]) / height / 2.0     # pixel angular half-size
    lam_half = 0.5 * (band[1] - band[0])
    gather = gather_px * pix_half                       # offset tolerance (rad)

    lit = 0
    for i in range(H * W):
        res = trace_beam(gem, C, dirs[i], n_of, dn_of, lambda0_um,
                         max_bounces=max_bounces)
        if not res["exited"]:
            continue
        P0, D0, Pj, Dj = res["P"], res["D"], res["Pj"], res["Dj"]
        LP = L - P0
        if np.dot(LP, D0) <= 0.0:                       # light behind the beam
            continue

        # (L − P(Δ)) × D(Δ) = 0,  f0 + A·Δs + b·Δλ = 0.  Parametrize by Δλ:
        # Δs(Δλ) = p + q·Δλ; choose Δλ (within band) minimizing |Δs|.
        f0 = np.cross(LP, D0)
        Jf = np.empty((3, 3))
        for k in range(3):
            Jf[:, k] = -np.cross(Pj[:, k], D0) + np.cross(LP, Dj[:, k])
        Ap = np.linalg.pinv(Jf[:, :2])
        p = -Ap @ f0
        q = -Ap @ Jf[:, 2]
        qq = float(q @ q)
        dlam = 0.0 if qq < 1e-20 else -float(p @ q) / qq
        dlam = max(-lam_half, min(lam_half, dlam))
        ds = p + q * dlam

        # constraint must actually be satisfiable (rank-2 cross product)
        resid = np.linalg.norm(f0 + Jf[:, :2] @ ds + Jf[:, 2] * dlam)
        if resid > 0.02 * np.linalg.norm(LP):
            continue
        off = float(np.sqrt(ds @ ds))
        if off > gather:
            continue
        lam = lambda0_um + dlam
        if not (band[0] <= lam <= band[1]):
            continue

        weight = np.exp(-(off / gather) ** 2)
        img[i] += weight * np.array(wavelength_to_rgb(lam * 1000.0))
        lit += 1

    return img.reshape(H, W, 3), lit / (H * W)


__all__ = ["render_fire_diff"]
