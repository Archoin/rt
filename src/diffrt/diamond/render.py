"""Vectorized spectral camera renderer for the diamond.

Looks at the gem and, for each wavelength, traces camera rays through it
(refraction + total internal reflection) until they leave and sample a lit
environment. Because each wavelength refracts differently (dispersion), the
per-wavelength exit directions sample the environment at slightly different
places — that chromatic spread, accumulated to RGB, is the diamond's "fire".

Pure NumPy, vectorized over pixels (one wavelength at a time). No Mitsuba.
Facets are triangles (from the convex-hull gem), so intersection uses a fast
triangle test.
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.optics import normalize

_EPS = 1e-9
_NUDGE = 1e-6


def wavelength_to_rgb(nm: float):
    """Approximate visible wavelength → linear RGB weight (Bruton)."""
    if nm < 380 or nm > 750:
        return (0.0, 0.0, 0.0)
    if nm < 440:
        r, g, b = -(nm - 440) / 60, 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / 50, 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / 20
    elif nm < 580:
        r, g, b = (nm - 510) / 70, 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, -(nm - 645) / 65, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if nm < 420:
        s = 0.3 + 0.7 * (nm - 380) / 40
    elif nm > 700:
        s = 0.3 + 0.7 * (750 - nm) / 50
    else:
        s = 1.0
    return (r * s, g * s, b * s)


def make_camera_rays(cam_pos, look_at, up, fov_deg, width, height):
    """Pinhole camera. Returns (origins (M,3), dirs (M,3), (H, W))."""
    cam_pos = np.asarray(cam_pos, float)
    fwd = normalize(np.asarray(look_at, float) - cam_pos)
    right = normalize(np.cross(fwd, np.asarray(up, float)))
    upv = np.cross(right, fwd)
    half = np.tan(np.radians(fov_deg) / 2)
    aspect = width / height
    xs = ((np.arange(width) + 0.5) / width * 2 - 1) * half * aspect
    ys = ((np.arange(height) + 0.5) / height * 2 - 1) * half
    gx, gy = np.meshgrid(xs, ys)
    dirs = (fwd[None, None, :] + gx[..., None] * right[None, None, :]
            - gy[..., None] * upv[None, None, :]).reshape(-1, 3)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    o = np.broadcast_to(cam_pos, dirs.shape).copy()
    return o, dirs, (height, width)


def _nearest_hit(facets, o, d):
    """Nearest triangle hit for a batch of rays. Returns (t, normal, found)."""
    K = o.shape[0]
    best_t = np.full(K, np.inf)
    best_n = np.zeros((K, 3))
    found = np.zeros(K, dtype=bool)
    for f in facets:
        n = f.n
        v0, v1, v2 = f.v[0], f.v[1], f.v[2]
        denom = d @ n
        ok = np.abs(denom) > _EPS
        t = np.where(ok, ((v0 - o) @ n) / np.where(ok, denom, 1.0), -1.0)
        p = o + t[:, None] * d
        c0 = np.sum(np.cross(v1 - v0, p - v0) * n, axis=1)
        c1 = np.sum(np.cross(v2 - v1, p - v1) * n, axis=1)
        c2 = np.sum(np.cross(v0 - v2, p - v2) * n, axis=1)
        inside = ((c0 >= 0) & (c1 >= 0) & (c2 >= 0)) | \
                 ((c0 <= 0) & (c1 <= 0) & (c2 <= 0))
        valid = ok & (t > _NUDGE) & inside & (t < best_t)
        best_t = np.where(valid, t, best_t)
        best_n = np.where(valid[:, None], n, best_n)
        found |= valid
    return best_t, best_n, found


def trace_image_wavelength(gem, o0, d0, n_glass, env_fn, max_bounces=12,
                           n_outside=1.0):
    """Radiance per ray for one wavelength (scalar index ``n_glass``)."""
    M = o0.shape[0]
    rad = np.zeros(M)
    o = o0.copy()
    d = d0.copy()
    active = np.arange(M)
    facets = gem.facets

    for _ in range(max_bounces):
        if active.size == 0:
            break
        oo, dd = o[active], d[active]
        t, nrm, found = _nearest_hit(facets, oo, dd)

        miss = ~found
        if miss.any():                                   # left the gem
            rad[active[miss]] = env_fn(dd[miss])

        hit = found
        if not hit.any():
            active = active[hit]
            continue
        ah, dh, nh, th = active[hit], dd[hit], nrm[hit], t[hit]
        ph = oo[hit] + th[:, None] * dh

        entering = np.sum(dh * nh, axis=1) < 0.0
        n_face = np.where(entering[:, None], nh, -nh)
        eta = np.where(entering, n_outside / n_glass, n_glass / n_outside)
        cos_i = -np.sum(dh * n_face, axis=1)
        sin2_t = eta * eta * (1 - cos_i * cos_i)
        tir = sin2_t > 1.0
        cos_t = np.sqrt(np.clip(1 - sin2_t, 0.0, None))
        refr = eta[:, None] * dh + (eta * cos_i - cos_t)[:, None] * n_face
        refl = dh + 2 * cos_i[:, None] * n_face
        new_d = np.where(tir[:, None], refl, refr)
        new_d /= np.linalg.norm(new_d, axis=1, keepdims=True)

        o[ah] = ph + _NUDGE * new_d
        d[ah] = new_d
        active = ah                                      # hit rays continue

    if active.size:                                      # bounce cap → env
        rad[active] = env_fn(d[active])
    return rad


def render_spectral(gem, camera, env_fn, n_of_lambda, wavelengths_nm,
                    width, height, max_bounces=12):
    """Render an (H, W, 3) linear-RGB image by spectral accumulation."""
    o, d, (H, W) = make_camera_rays(*camera, width, height)
    img = np.zeros((H * W, 3))
    rgb_norm = np.zeros(3)
    for nm in wavelengths_nm:
        n_glass = float(n_of_lambda(nm / 1000.0))
        rad = trace_image_wavelength(gem, o, d, n_glass, env_fn, max_bounces)
        w = np.array(wavelength_to_rgb(nm))
        img += rad[:, None] * w[None, :]
        rgb_norm += w
    img /= np.maximum(rgb_norm, 1e-6)[None, :]            # uniform spectrum→white
    return img.reshape(H, W, 3)


__all__ = ["wavelength_to_rgb", "make_camera_rays", "render_spectral",
           "trace_image_wavelength"]
