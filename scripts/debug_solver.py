"""Debug the 2nd-pass solver: per-step trajectories for seeds with 2/3/4 TIRs.

Faithfully re-runs the connect_newton_vec Gauss-Newton + backtracking loop on a
few hand-picked seeds (closest-to-camera rays with exactly 2, 3, 4 reflections),
logging per iteration: residual |f| (perp distance to the pinhole), raw Newton
step vs applied step, alpha, accept/reject, and whether the proposal still exits.
Plots the trajectories for review.

    PYTHONPATH=src python scripts/debug_solver.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffrt.diamond.lightconnect import (build_gem, camera_from_meta, load_pass1)
from diffrt.diamond.manifold import _eval
from diffrt.diamond.optics import normalize, sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render_photon import _make_frames

MAX_ITER = 60
TOL = 1e-7
A_MIN = 1e-3


def debug_newton(gem, origin, dirs, wl, n_of, dn_of, target, max_bounces, band):
    """Same algorithm as connect_newton_vec, but logs per-seed per-iter history."""
    N = dirs.shape[0]
    D = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).copy()
    wl = np.asarray(wl, float).copy()
    target = np.asarray(target, float)
    Ot = np.broadcast_to(np.asarray(origin, float), (N, 3)).copy()

    res, rn, TP = _eval(gem, Ot, D, wl, n_of, dn_of, target, max_bounces)
    P, Dd, Pj, Dj = res["P"], res["D"], res["Pj"], res["Dj"]
    alpha = np.ones(N)
    hist = [[] for _ in range(N)]
    active = np.arange(N)

    for it in range(MAX_ITER):
        if active.size == 0:
            break
        a = active
        fa = np.cross(TP[a], Dd[a])
        dist_cur = np.linalg.norm(fa, axis=1)              # |f| = perp distance
        Jf = np.empty((a.size, 3, 3))
        for k in range(3):
            Jf[:, :, k] = -np.cross(Pj[a, :, k], Dd[a]) + np.cross(TP[a], Dj[a, :, k])
        step = -np.einsum('nij,nj->ni', np.linalg.pinv(Jf), fa)
        raw_ang = np.linalg.norm(step[:, :2], axis=1)
        raw_wl = np.abs(step[:, 2])
        ang = step[:, :2] * alpha[a, None]
        dl = step[:, 2] * alpha[a]

        e1, e2 = _make_frames(D[a])
        Dp = D[a] + ang[:, 0:1] * e1 + ang[:, 1:2] * e2
        Dp /= np.linalg.norm(Dp, axis=1, keepdims=True)
        wlp = np.clip(wl[a] + dl, band[0], band[1])
        rp, rnp, TPp = _eval(gem, Ot[a], Dp, wlp, n_of, dn_of, target, max_bounces)
        prop_dist = np.where(rp["exited"],
                             np.linalg.norm(np.cross(TPp, rp["D"]), axis=1), np.inf)
        better = rnp < rn[a]

        for i, si in enumerate(a):
            hist[si].append(dict(it=it, dist=float(dist_cur[i]), alpha=float(alpha[si]),
                                 raw_ang=float(raw_ang[i]), app_ang=float(np.linalg.norm(ang[i])),
                                 raw_wl=float(raw_wl[i] * 1000.0),       # nm
                                 app_wl=float(abs(dl[i]) * 1000.0),      # nm
                                 prop_dist=float(prop_dist[i]), accept=bool(better[i]),
                                 prop_exited=bool(rp["exited"][i])))

        ia = np.where(better)[0]; aa = a[ia]
        D[aa], wl[aa], rn[aa], TP[aa] = Dp[ia], wlp[ia], rnp[ia], TPp[ia]
        P[aa], Dd[aa] = rp["P"][ia], rp["D"][ia]
        Pj[aa], Dj[aa] = rp["Pj"][ia], rp["Dj"][ia]
        alpha[aa] = np.minimum(alpha[aa] * 1.5, 4.0)
        alpha[a[np.where(~better)[0]]] *= 0.5
        cont = (rn[a] >= TOL) & (alpha[a] >= A_MIN)
        active = a[np.where(cont)[0]]
    return hist


def main() -> int:
    samples, meta = load_pass1("runs/connect_pass1.npz")
    gem = build_gem(meta)
    cam = camera_from_meta(meta)
    C = np.asarray(cam[0], float)
    n_of = lambda um: sellmeier_n(um, meta["material"])
    dn_of = lambda um: sellmeier_dn_dlambda(um, meta["material"])

    ex = samples["exited"]; dist = samples["dist"]; nrefl = samples["n_refr"] * 0 + samples["n_refl"]
    picks = []
    for r in (2, 3, 4):
        idx = np.where(ex & (nrefl == r))[0]
        idx = idx[np.argsort(dist[idx])][:2]               # 2 closest with r reflections
        picks.extend((int(i), r) for i in idx)

    seed_idx = np.array([p[0] for p in picks])
    refls = [p[1] for p in picks]
    hist = debug_newton(gem, meta["light_pos"], samples["dirs"][seed_idx],
                        samples["wl"][seed_idx], n_of, dn_of, C,
                        meta["max_bounces"], tuple(meta["band"]))

    colors = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    print(f"{'seed':>8} {'refl':>4} {'init d':>9} {'final d':>9} {'iters':>6} "
          f"{'accept':>6} {'reject':>6}")
    for k, (si, r) in enumerate(picks):
        h = hist[k]
        if not h:
            continue
        its = np.array([x["it"] for x in h])
        d = np.array([x["dist"] for x in h])
        acc = np.array([x["accept"] for x in h])
        lbl = f"#{si} r={r} d0={d[0]:.3f}"
        c = colors[r]
        ls = "-" if k % 2 == 0 else "--"
        axs[0, 0].plot(its, d, ls, color=c, alpha=0.8, label=lbl)
        axs[0, 0].scatter(its[acc], d[acc], s=18, c=c, marker="o")
        axs[0, 0].scatter(its[~acc], d[~acc], s=22, facecolors="none", edgecolors=c)
        axs[0, 1].plot(its, [x["alpha"] for x in h], ls, color=c, alpha=0.8, label=lbl)
        axs[0, 2].plot(its, [x["app_ang"] for x in h], ls, color=c, alpha=0.8, label=lbl)
        axs[1, 0].plot(its, [x["app_wl"] for x in h], ls, color=c, alpha=0.8, label=lbl)
        axs[1, 1].plot(its, [x["raw_ang"] for x in h], ls, color=c, alpha=0.8, label=lbl)
        axs[1, 2].plot(its, [x["raw_wl"] for x in h], ls, color=c, alpha=0.8, label=lbl)
        n_acc = int(acc.sum())
        print(f"{si:>8} {r:>4} {d[0]:>9.4f} {min(d):>9.4f} {len(h):>6} "
              f"{n_acc:>6} {len(h)-n_acc:>6}")

    for ax in axs.ravel():
        ax.set_yscale("log"); ax.set_xlabel("iteration")
    axs[0, 0].set_title("residual |f| = distance to pinhole (● accept, ○ reject)")
    axs[0, 0].legend(fontsize=7)
    axs[0, 1].set_title("alpha (per-seed step scale)")
    axs[0, 2].set_title("applied step: angular (rad)")
    axs[1, 0].set_title("applied step: wavelength (nm)")
    axs[1, 1].set_title("raw Newton step: angular (rad)")
    axs[1, 2].set_title("raw Newton step: wavelength (nm)")
    fig.tight_layout()
    out = pathlib.Path("runs/solver_debug.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
