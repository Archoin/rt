"""Why is the fire concentrated in one small area?

Uses the cached stage-1 samples (no re-tracing). Builds the full
"near-connection landscape": for every exited light ray, where it images on the
camera film and how close its exit ray passes to the pinhole. Counts how many
distinct near regions exist (film + emission space) at several distance
thresholds — if there's essentially one, the fire is one facet-path channel.

    PYTHONPATH=src python scripts/diagnose_fire_location.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label

from diffrt.diamond.lightconnect import camera_from_meta, load_pass1
from diffrt.diamond.optics import normalize
from diffrt.diamond.render_photon import _camera_projector


def min_grid(ix, iy, val, nx, ny):
    g = np.full(nx * ny, np.inf)
    np.minimum.at(g, iy * nx + ix, val)
    return g.reshape(ny, nx)


def n_regions(grid, thr):
    lab, n = label(np.isfinite(grid) & (grid < thr))
    return n


def main() -> int:
    samples, meta = load_pass1("runs/connect_pass1.npz")
    cam = camera_from_meta(meta)
    W, H = meta["width"], meta["height"]
    C, project = _camera_projector(cam, W, H)

    ex = samples["exited"]
    P, dist = samples["P"][ex], samples["dist"][ex]
    dirs = samples["dirs"][ex]

    # --- film landscape: min distance-to-camera per imaged pixel ----------
    col, row, valid = project(P)
    ci = np.clip(np.round(col[valid]).astype(int), 0, W - 1)
    ri = np.clip(np.round(row[valid]).astype(int), 0, H - 1)
    film = min_grid(ci, ri, dist[valid], W, H)

    # --- emission landscape: min distance per emission-direction cell -----
    L = np.asarray(meta["light_pos"], float)
    axis = normalize(-L)
    helper = np.array([0, 0, 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0, 0])
    e1 = normalize(np.cross(helper, axis)); e2 = np.cross(axis, e1)
    ex1, ex2 = dirs @ e1, dirs @ e2
    NB = 200
    bx = np.clip(((ex1 - ex1.min()) / (np.ptp(ex1) + 1e-9) * NB).astype(int), 0, NB - 1)
    by = np.clip(((ex2 - ex2.min()) / (np.ptp(ex2) + 1e-9) * NB).astype(int), 0, NB - 1)
    emis = min_grid(bx, by, dist, NB, NB)

    print(f"exited rays: {ex.sum():,}")
    print(f"{'thr':>6} {'film regions':>13} {'film px<thr':>11} "
          f"{'emis regions':>13} {'emis cells<thr':>14}")
    for thr in (0.5, 0.2, 0.1, 0.05):
        fpx = int((np.isfinite(film) & (film < thr)).sum())
        ecl = int((np.isfinite(emis) & (emis < thr)).sum())
        print(f"{thr:6.2f} {n_regions(film, thr):13d} {fpx:11d} "
              f"{n_regions(emis, thr):13d} {ecl:14d}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    for ax, g, ttl, ext in (
        (axs[0], film, "film: min distance-to-camera (log)", None),
        (axs[1], emis, "emission dirs: min distance-to-camera (log)", None)):
        m = np.log10(np.where(np.isfinite(g), g, np.nan))
        im = ax.imshow(m, origin="lower", cmap="turbo", vmax=0.5, vmin=-1.5)
        fig.colorbar(im, ax=ax, label="log10(dist)")
        ax.set_title(ttl)
    out = pathlib.Path("runs/fire_location.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
