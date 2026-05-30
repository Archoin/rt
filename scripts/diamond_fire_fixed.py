"""Render diamond fire via fixed-facet-sequence connections (post solver-fix).

Sample rays from the point light, record each ray's specular chain, group rays
by chain, and for each chain hold the ray's wavelength fixed and Newton-solve
(with a line search) the emission direction that connects light -> chain ->
camera pinhole. Splat every converged connection at its exit point, coloured by
wavelength. Connections span each chain's valid wavelength range -> colored fire
curves, not the sparse blobs the free-retrace solver produced.

    PYTHONPATH=src python scripts/diamond_fire_fixed.py
"""

from __future__ import annotations

import pathlib
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter

from diffrt.diamond.fixedseq import solve_fixed_2dof, trace_record_sequence
from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render_photon import (_camera_projector, _sample_cone,
                                          normalize)

WIDTH, HEIGHT = 240, 240
N_SAMPLES = 400_000
TOP_CHAINS = 80
MIN_RAYS = 25
TOL = 1e-7
MAX_ITER = 30
MAX_BOUNCES = 18
BAND = (0.40, 0.70)
GEM_RADIUS = 1.3
EXPOSURE, BLUR = 0.9, 0.7
CAMERA = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
LIGHT_POS = np.array([0.9, 3.0, 0.6])


def wl_to_rgb(nm):
    from diffrt.diamond.render import wavelength_to_rgb
    return np.array([wavelength_to_rgb(x) for x in nm])


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0, table_frac=0.55)
    facets = gem.facets
    n_of = lambda um: sellmeier_n(um, "diamond")
    dn_of = lambda um: sellmeier_dn_dlambda(um, "diamond")
    C, project = _camera_projector(CAMERA, WIDTH, HEIGHT)
    L = LIGHT_POS

    rng = np.random.default_rng(0)
    axis = normalize(-L)
    cone = np.arctan(GEM_RADIUS / np.linalg.norm(L))
    dirs = _sample_cone(axis, cone, N_SAMPLES, rng)
    wl = rng.uniform(BAND[0], BAND[1], N_SAMPLES)

    t0 = time.time()
    seq_fi, seq_kind, nint, exited = trace_record_sequence(
        gem, np.broadcast_to(L, dirs.shape).copy(), dirs, wl, n_of, max_bounces=MAX_BOUNCES)

    groups = defaultdict(list)
    for i in np.where(exited)[0]:
        Ln = nint[i]
        groups[(Ln, *seq_fi[i, :Ln], *seq_kind[i, :Ln])].append(i)
    chains = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    img = np.zeros((HEIGHT * WIDTH, 3))
    n_conn = 0
    used = 0
    for key, idxs in chains[:TOP_CHAINS]:
        if len(idxs) < MIN_RAYS:
            break
        used += 1
        idxs = np.array(idxs)
        Ln = key[0]
        seq = list(zip(key[1:1 + Ln], key[1 + Ln:1 + 2 * Ln]))
        res = solve_fixed_2dof(facets, seq, L, dirs[idxs], wl[idxs], n_of, dn_of,
                               C, max_iter=MAX_ITER, tol=TOL)
        conv = res["converged"]
        if not conv.any():
            continue
        col, row, valid = project(res["P"][conv])
        if valid.any():
            ci = np.round(col[valid]).astype(int)
            ri = np.round(row[valid]).astype(int)
            np.add.at(img, ri * WIDTH + ci, wl_to_rgb(res["wl"][conv][valid] * 1000.0))
            n_conn += int(valid.sum())
    dt = time.time() - t0

    img = img.reshape(HEIGHT, WIDTH, 3)
    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], BLUR)
    srgb = np.clip(img * (EXPOSURE / max(img.max(), 1e-12)), 0, 1) ** (1 / 2.2)
    out = pathlib.Path(__file__).resolve().parent.parent / "runs" / "diamond_fire_fixed.png"
    out.parent.mkdir(exist_ok=True)
    mpimg.imsave(str(out), srgb)

    print(f"samples {N_SAMPLES:,}, exited {int(exited.sum()):,}, "
          f"distinct chains {len(chains):,}")
    print(f"solved {used} chains (>= {MIN_RAYS} rays); connections splatted: {n_conn:,}")
    print(f"time {dt:.1f}s; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
