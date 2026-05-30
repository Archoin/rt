"""Diagnostics for light-side sample-and-connect (MC-1 take 2).

Reports: sample-pass count / time / peak memory; per-ray bounce, refraction, and
reflection histograms; whether Newton actually refines (seed vs final distance);
and three 2D density maps — accepted samples on the camera film, all outgoing
light directions, and accepted outgoing light directions.

    PYTHONPATH=src python scripts/diagnose_connect.py
"""

from __future__ import annotations

import pathlib
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.lightconnect import ray_point_distance
from diffrt.diamond.manifold import connect_newton_vec
from diffrt.diamond.optics import normalize, sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render_photon import (_camera_projector, _sample_cone,
                                          trace_beams_vec)

WIDTH, HEIGHT = 200, 200
N_SAMPLES = 1_000_000
TOP_K = 100_000
ACCEPT_EPS = 0.10
NEWTON_ITERS = 50
MAX_BOUNCES = 18
BAND = (0.40, 0.70)
GEM_RADIUS = 1.3
CAMERA = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
LIGHT_POS = np.array([0.9, 3.0, 0.6])


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # macOS: bytes


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0,
                          table_frac=0.55)
    n_of = lambda um: sellmeier_n(um, "diamond")
    dn_of = lambda um: sellmeier_dn_dlambda(um, "diamond")
    C, project = _camera_projector(CAMERA, WIDTH, HEIGHT)
    L = LIGHT_POS

    # cone frame about the light->gem axis (for emission-direction 2D maps)
    axis = normalize(-L)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = normalize(np.cross(helper, axis))
    e2 = np.cross(axis, e1)
    cone_half = np.arctan(GEM_RADIUS / np.linalg.norm(L))

    # ---- sample pass (instrumented) --------------------------------------
    rng = np.random.default_rng(0)
    rss0 = rss_mb()
    t0 = time.time()
    dirs = _sample_cone(axis, cone_half, N_SAMPLES, rng)
    wl = rng.uniform(BAND[0], BAND[1], N_SAMPLES)
    O = np.broadcast_to(L, dirs.shape).copy()
    res = trace_beams_vec(gem, O, dirs, n_of(wl), dn_of(wl), max_bounces=MAX_BOUNCES)
    t_sample = time.time() - t0
    rss_peak = rss_mb()

    P, D, ex = res["P"], res["D"], res["exited"]
    nint, nrefr, nrefl = res["nint"], res["n_refr"], res["n_refl"]
    dist = np.where(ex, ray_point_distance(P, D, C), np.inf)
    em_x, em_y = dirs @ e1, dirs @ e2          # emission tangent-plane coords

    # ---- connect pass on top-K closest -----------------------------------
    order = np.argsort(dist)
    top = order[:TOP_K]
    top = top[np.isfinite(dist[top])]
    seed_dist = dist[top]
    rc = connect_newton_vec(gem, L, dirs[top], wl[top], n_of, dn_of, C,
                            max_iter=NEWTON_ITERS, tol=1e-7,
                            max_bounces=MAX_BOUNCES, band=BAND, seed_resid=10.0)
    final_dist = ray_point_distance(rc["P"], rc["D"], C)
    accept = final_dist < ACCEPT_EPS
    acc_idx = top[accept]                       # sample indices of accepted

    # ---- stats -----------------------------------------------------------
    jac_mb = N_SAMPLES * (3 + 3 + 9 + 9) * 8 / 1e6
    improved = final_dist < 0.9 * seed_dist
    print(f"=== sample pass ===")
    print(f"samples generated : {N_SAMPLES:,}")
    print(f"exited gem        : {int(ex.sum()):,} ({ex.mean():.1%})")
    print(f"time              : {t_sample:.1f}s")
    print(f"peak RSS          : {rss_peak:.0f} MB (Δ {rss_peak-rss0:.0f}; "
          f"P/D/Pj/Dj arrays ≈ {jac_mb:.0f} MB)")
    exb = ex
    print(f"\n=== per-ray interactions (exited rays) ===")
    print(f"bounces (total)   : mean {nint[exb].mean():.2f}  median {int(np.median(nint[exb]))}  max {int(nint[exb].max())}")
    print(f"refractions       : mean {nrefr[exb].mean():.2f}  max {int(nrefr[exb].max())}")
    print(f"reflections (TIR) : mean {nrefl[exb].mean():.2f}  max {int(nrefl[exb].max())}")
    print(f"\n=== Newton refinement (top-{top.size:,} seeds) ===")
    print(f"seed dist  : min {seed_dist.min():.3e}  median {np.median(seed_dist):.3e}")
    print(f"final dist : min {final_dist.min():.3e}  median {np.median(final_dist):.3e}")
    print(f"seeds improved >10%: {improved.mean():.1%}   accepted (<{ACCEPT_EPS}): {int(accept.sum())}")

    # ---- plots -----------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))

    ax = axs[0, 0]
    mx = int(nint[exb].max())
    bins = np.arange(0, mx + 2) - 0.5
    ax.hist(nint[exb], bins=bins, alpha=0.5, label="total bounces")
    ax.hist(nrefr[exb], bins=bins, alpha=0.5, label="refractions")
    ax.hist(nrefl[exb], bins=bins, alpha=0.5, label="reflections (TIR)")
    ax.set_title("per-ray interactions (exited)")
    ax.set_xlabel("count"); ax.set_ylabel("rays"); ax.legend(); ax.set_yscale("log")

    ax = axs[0, 1]
    ax.scatter(seed_dist, final_dist, s=1, alpha=0.2)
    lim = [min(seed_dist.min(), final_dist.min()) * 0.5,
           max(seed_dist.max(), np.nanmax(final_dist[np.isfinite(final_dist)])) * 2]
    ax.plot(lim, lim, "r--", lw=1, label="y=x (no refinement)")
    ax.axhline(ACCEPT_EPS, color="g", lw=1, ls=":", label=f"accept_eps={ACCEPT_EPS}")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title("Newton refinement: seed vs final distance")
    ax.set_xlabel("seed distance to camera"); ax.set_ylabel("final distance"); ax.legend()

    ax = axs[0, 2]
    col, row, valid = project(rc["P"][accept])
    if valid.any():
        ax.hist2d(col[valid], row[valid], bins=80,
                  range=[[0, WIDTH], [0, HEIGHT]], cmap="magma")
    ax.set_title(f"accepted samples on camera film (n={int(accept.sum())})")
    ax.set_xlabel("col"); ax.set_ylabel("row"); ax.set_aspect("equal")

    ax = axs[1, 0]
    h = ax.hist2d(em_x[ex], em_y[ex], bins=160, cmap="viridis")
    fig.colorbar(h[3], ax=ax)
    ax.set_title("all outgoing light directions (exited)")
    ax.set_xlabel("· e1"); ax.set_ylabel("· e2"); ax.set_aspect("equal")

    ax = axs[1, 1]
    if acc_idx.size:
        ax.hist2d(em_x[acc_idx], em_y[acc_idx], bins=160,
                  range=[[em_x[ex].min(), em_x[ex].max()],
                         [em_y[ex].min(), em_y[ex].max()]], cmap="inferno")
    ax.set_title(f"accepted outgoing light directions (n={acc_idx.size})")
    ax.set_xlabel("· e1"); ax.set_ylabel("· e2"); ax.set_aspect("equal")

    ax = axs[1, 2]
    ax.hist(np.log10(seed_dist), bins=60, alpha=0.5, label="seed")
    ax.hist(np.log10(final_dist[np.isfinite(final_dist)]), bins=60, alpha=0.5, label="final")
    ax.axvline(np.log10(ACCEPT_EPS), color="g", ls=":", label="accept_eps")
    ax.set_title("distance-to-camera distribution"); ax.set_xlabel("log10(distance)")
    ax.set_ylabel("seeds"); ax.legend()

    fig.tight_layout()
    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    png = out / "connect_diagnostics.png"
    fig.savefig(png, dpi=110)
    print(f"\nwrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
