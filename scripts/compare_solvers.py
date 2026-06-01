"""Compare four 2nd-pass solvers: render an image + stats for each.

Same sampled facet-chains for all methods (apples-to-apples). For each method,
solve every chain, splat the converged connections (coloured by wavelength), save
an image, and tabulate connections / time / mean iterations.

    PYTHONPATH=src python scripts/compare_solvers.py
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

from diffrt.diamond.fixedseq import trace_record_sequence
from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import normalize, sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render import wavelength_to_rgb
from diffrt.diamond.render_photon import _camera_projector, _sample_cone
from diffrt.diamond.solvers2 import solve

WIDTH, HEIGHT = 240, 240
N_SAMPLES, TOP_CHAINS, MIN_RAYS = 400_000, 80, 25
TOL, MAX_ITER, MAX_BOUNCES = 1e-7, 30, 18
BAND, GEM_RADIUS = (0.40, 0.70), 1.3
EXPOSURE, BLUR = 0.9, 0.7
CAMERA = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
LIGHT_POS = np.array([0.9, 3.0, 0.6])
METHODS = ["gn_wl", "gn_freq", "gauss_seidel", "hessian"]


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
    seq_fi, seq_kind, nint, exited = trace_record_sequence(
        gem, np.broadcast_to(L, dirs.shape).copy(), dirs, wl, n_of, max_bounces=MAX_BOUNCES)

    groups = defaultdict(list)
    for i in np.where(exited)[0]:
        Ln = nint[i]
        groups[(Ln, *seq_fi[i, :Ln], *seq_kind[i, :Ln])].append(i)
    chains = [(k, np.array(v)) for k, v in
              sorted(groups.items(), key=lambda kv: -len(kv[1]))[:TOP_CHAINS]
              if len(v) >= MIN_RAYS]
    print(f"sampled {N_SAMPLES:,}, exited {int(exited.sum()):,}, "
          f"solving {len(chains)} chains\n")

    rundir = pathlib.Path(__file__).resolve().parent.parent / "runs"
    rundir.mkdir(exist_ok=True)
    print(f"{'method':>13} {'connections':>12} {'mean iters':>11} {'time(s)':>8}")
    for method in METHODS:
        img = np.zeros((HEIGHT * WIDTH, 3))
        n_conn, iters_sum = 0, 0
        t0 = time.time()
        for key, idxs in chains:
            Ln = key[0]
            seq = list(zip(key[1:1 + Ln], key[1 + Ln:1 + 2 * Ln]))
            res = solve(method, facets, seq, L, dirs[idxs], wl[idxs], n_of, dn_of,
                        C, max_iter=MAX_ITER, tol=TOL, band=BAND)
            iters_sum += res["iters"]
            m = res["converged"]
            if not m.any():
                continue
            col, row, valid = project(res["P"][m])
            if valid.any():
                ci = np.round(col[valid]).astype(int)
                ri = np.round(row[valid]).astype(int)
                rgb = np.array([wavelength_to_rgb(w * 1000.0) for w in res["wl"][m][valid]])
                np.add.at(img, ri * WIDTH + ci, rgb)
                n_conn += int(valid.sum())
        dt = time.time() - t0

        im = img.reshape(HEIGHT, WIDTH, 3)
        for c in range(3):
            im[:, :, c] = gaussian_filter(im[:, :, c], BLUR)
        srgb = np.clip(im * (EXPOSURE / max(im.max(), 1e-12)), 0, 1) ** (1 / 2.2)
        mpimg.imsave(str(rundir / f"solver_{method}.png"), srgb)
        print(f"{method:>13} {n_conn:>12,} {iters_sum/len(chains):>11.1f} {dt:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
