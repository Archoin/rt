"""Categorize converged connections: valid (in-view) / valid (out-of-view) /
geometrically invalid — per 2nd-pass solver. Plot the percentage breakdown, and
render gauss_seidel WITH vs WITHOUT geometrically-invalid connections.

Validity (rigorous): re-trace each converged connection's solved emission
direction with the REAL bounded-facet tracer; it is valid iff the free trace
follows the exact same chain (facets + kinds + length) and exits. Otherwise the
fixed-sequence solver connected via the facets' infinite-plane extensions
(off the real polygons) -> geometrically invalid.

    PYTHONPATH=src python scripts/connection_validity.py
"""

from __future__ import annotations

import pathlib
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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


def valid_mask(gem, seq, emit, wl, n_of):
    """True where the free (bounded-facet) trace of `emit` follows `seq` exactly."""
    O = np.broadcast_to(LIGHT_POS, emit.shape).copy()
    sfi, skd, ni, ex = trace_record_sequence(gem, O, emit, wl, n_of, max_bounces=MAX_BOUNCES)
    Ln = len(seq)
    v = ex & (ni == Ln)
    for k, (fi, kind) in enumerate(seq):
        v &= (sfi[:, k] == fi) & (skd[:, k] == kind)
    return v


def tonemap_save(splat, path):
    im = splat.reshape(HEIGHT, WIDTH, 3)
    for c in range(3):
        im[:, :, c] = gaussian_filter(im[:, :, c], BLUR)
    srgb = np.clip(im * (EXPOSURE / max(im.max(), 1e-12)), 0, 1) ** (1 / 2.2)
    mpimg.imsave(str(path), srgb)


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0, table_frac=0.55)
    facets = gem.facets
    n_of = lambda um: sellmeier_n(um, "diamond")
    dn_of = lambda um: sellmeier_dn_dlambda(um, "diamond")
    C, project = _camera_projector(CAMERA, WIDTH, HEIGHT)
    L = LIGHT_POS
    rundir = pathlib.Path(__file__).resolve().parent.parent / "runs"

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

    cats = {m: dict(good=0, oov=0, invalid=0) for m in METHODS}
    gs_with = []   # (col,row,wl) of on-frame converged (incl. invalid)
    gs_good = []   # whether each gs_with entry is geometrically valid

    for method in METHODS:
        t0 = time.time()
        for key, idxs in chains:
            Ln = key[0]
            seq = list(zip(key[1:1 + Ln], key[1 + Ln:1 + 2 * Ln]))
            res = solve(method, facets, seq, L, dirs[idxs], wl[idxs], n_of, dn_of,
                        C, max_iter=MAX_ITER, tol=TOL, band=BAND)
            m = res["converged"]
            if not m.any():
                continue
            emit, wlc, P = res["emit"][m], res["wl"][m], res["P"][m]
            valid = valid_mask(gem, seq, emit, wlc, n_of)
            col, row, view = project(P)
            cats[method]["invalid"] += int((~valid).sum())
            cats[method]["oov"] += int((valid & ~view).sum())
            cats[method]["good"] += int((valid & view).sum())
            if method == "gauss_seidel" and view.any():
                gs_with.append(np.column_stack([col[view], row[view], wlc[view]]))
                gs_good.append(valid[view])
        print(f"{method}: {cats[method]}  ({time.time()-t0:.1f}s)")

    # --- percentage breakdown diagram ------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    labels = METHODS
    tot = {m: max(sum(cats[m].values()), 1) for m in METHODS}
    good = [100 * cats[m]["good"] / tot[m] for m in METHODS]
    oov = [100 * cats[m]["oov"] / tot[m] for m in METHODS]
    inv = [100 * cats[m]["invalid"] / tot[m] for m in METHODS]
    ax.bar(labels, good, label="valid, in-view", color="tab:green")
    ax.bar(labels, oov, bottom=good, label="valid, out-of-view", color="tab:gray")
    ax.bar(labels, inv, bottom=np.array(good) + np.array(oov),
           label="geometrically invalid", color="tab:red")
    for i, m in enumerate(METHODS):
        ax.text(i, 50, f"n={tot[m]:,}", ha="center", rotation=90, color="white")
    ax.set_ylabel("% of converged connections"); ax.set_ylim(0, 100)
    ax.set_title("Converged-connection breakdown by 2nd-pass solver")
    ax.legend()
    fig.tight_layout(); fig.savefig(rundir / "connection_validity.png", dpi=120)

    print(f"\n{'method':>13} {'valid/in-view':>15} {'valid/out-view':>15} {'invalid':>15}")
    for m in METHODS:
        c = cats[m]; t = tot[m]
        print(f"{m:>13} {c['good']:>7,} ({100*c['good']/t:4.1f}%) "
              f"{c['oov']:>7,} ({100*c['oov']/t:4.1f}%) "
              f"{c['invalid']:>7,} ({100*c['invalid']/t:4.1f}%)")

    # --- gauss_seidel: with vs without invalid ---------------------------
    A = np.concatenate(gs_with) if gs_with else np.zeros((0, 3))
    G = np.concatenate(gs_good) if gs_good else np.zeros(0, bool)
    for tag, mask in (("with_invalid", np.ones(len(A), bool)), ("without_invalid", G)):
        sp = np.zeros((HEIGHT * WIDTH, 3))
        a = A[mask]
        if a.shape[0]:
            ci = np.round(a[:, 0]).astype(int); ri = np.round(a[:, 1]).astype(int)
            rgb = np.array([wavelength_to_rgb(w * 1000.0) for w in a[:, 2]])
            np.add.at(sp, ri * WIDTH + ci, rgb)
        tonemap_save(sp, rundir / f"fire_gs_{tag}.png")
    print(f"\ngauss_seidel on-frame: {len(A):,} total, "
          f"{int(G.sum()):,} valid, {int((~G).sum()):,} invalid (spurious)")
    print(f"wrote runs/connection_validity.png, runs/fire_gs_with_invalid.png, "
          f"runs/fire_gs_without_invalid.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
