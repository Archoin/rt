"""Histogram of #reflections (TIR) among the converged connections, per solver.

Every connection in a facet-chain shares that chain's reflection count
(= number of 'tir' interactions). For each of the four 2nd-pass solvers we tally
converged connections by reflection count, in absolute numbers and percentages,
and plot both.

    PYTHONPATH=src python scripts/reflection_hist.py
"""

from __future__ import annotations

import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffrt.diamond.fixedseq import trace_record_sequence
from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import normalize, sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render_photon import _camera_projector, _sample_cone
from diffrt.diamond.solvers2 import solve

N_SAMPLES, TOP_CHAINS, MIN_RAYS = 400_000, 80, 25
TOL, MAX_ITER, MAX_BOUNCES = 1e-7, 30, 18
BAND, GEM_RADIUS = (0.40, 0.70), 1.3
CAMERA = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
LIGHT_POS = np.array([0.9, 3.0, 0.6])
METHODS = ["gn_wl", "gn_freq", "gauss_seidel", "hessian"]


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0, table_frac=0.55)
    facets = gem.facets
    n_of = lambda um: sellmeier_n(um, "diamond")
    dn_of = lambda um: sellmeier_dn_dlambda(um, "diamond")
    C, _ = _camera_projector(CAMERA, 240, 240)
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

    hists = {m: defaultdict(int) for m in METHODS}
    for method in METHODS:
        for key, idxs in chains:
            Ln = key[0]
            kinds = key[1 + Ln:1 + 2 * Ln]
            n_refl = sum(1 for k in kinds if k == 2)
            res = solve(method, facets, list(zip(key[1:1 + Ln], kinds)), L,
                        dirs[idxs], wl[idxs], n_of, dn_of, C,
                        max_iter=MAX_ITER, tol=TOL, band=BAND)
            hists[method][n_refl] += int(res["converged"].sum())

    refls = sorted({r for m in METHODS for r in hists[m]})
    print(f"{'refl':>5} " + " ".join(f"{m:>22}" for m in METHODS))
    for r in refls:
        cells = []
        for m in METHODS:
            tot = sum(hists[m].values())
            c = hists[m].get(r, 0)
            cells.append(f"{c:>9,} ({100*c/max(tot,1):5.1f}%)")
        print(f"{r:>5} " + " ".join(f"{x:>22}" for x in cells))
    print(f"{'tot':>5} " + " ".join(f"{sum(hists[m].values()):>22,}" for m in METHODS))

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len(refls)); w = 0.2
    for j, m in enumerate(METHODS):
        tot = max(sum(hists[m].values()), 1)
        abs_v = [hists[m].get(r, 0) for r in refls]
        pct_v = [100 * v / tot for v in abs_v]
        axs[0].bar(x + (j - 1.5) * w, abs_v, w, label=m)
        axs[1].bar(x + (j - 1.5) * w, pct_v, w, label=m)
    for ax, t, yl in ((axs[0], "connections by #reflections (absolute)", "connections"),
                      (axs[1], "connections by #reflections (percentage)", "% of method's connections")):
        ax.set_xticks(x); ax.set_xticklabels(refls); ax.set_xlabel("# reflections (TIR)")
        ax.set_ylabel(yl); ax.set_title(t); ax.legend()
    fig.tight_layout()
    out = pathlib.Path("runs/reflection_hist.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
