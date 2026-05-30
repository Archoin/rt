"""Density of samples that pass the connection criterion after the 2nd pass.

Runs the fixed-sequence pipeline (sample -> group by chain -> solve each chain),
collects the CONVERGED connections (resid < tol), and plots where they land:
  (1) on the camera film (2D density),
  (2) in emission-direction space (2D density, cone tangent coords),
  (3) on the film, coloured by connecting wavelength.

    PYTHONPATH=src python scripts/density_pass2.py [--solver 2dof|3dof]
"""

from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffrt.diamond.fixedseq import (solve_fixed_2dof, solve_fixed_3dof,
                                     trace_record_sequence)
from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import normalize, sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render import wavelength_to_rgb
from diffrt.diamond.render_photon import _camera_projector, _sample_cone

WIDTH, HEIGHT = 240, 240
N_SAMPLES = 400_000
TOP_CHAINS, MIN_RAYS = 80, 25
TOL, MAX_ITER, MAX_BOUNCES = 1e-7, 30, 18
BAND, GEM_RADIUS = (0.40, 0.70), 1.3
CAMERA = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
LIGHT_POS = np.array([0.9, 3.0, 0.6])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", choices=["2dof", "3dof"], default="3dof")
    args = ap.parse_args()
    solver = solve_fixed_3dof if args.solver == "3dof" else solve_fixed_2dof

    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0, table_frac=0.55)
    facets = gem.facets
    n_of = lambda um: sellmeier_n(um, "diamond")
    dn_of = lambda um: sellmeier_dn_dlambda(um, "diamond")
    C, project = _camera_projector(CAMERA, WIDTH, HEIGHT)
    L = LIGHT_POS

    # emission cone frame (for emission-space coords)
    axis = normalize(-L)
    helper = np.array([0, 0, 1.]) if abs(axis[2]) < 0.9 else np.array([1., 0, 0])
    e1 = normalize(np.cross(helper, axis)); e2 = np.cross(axis, e1)
    cone = np.arctan(GEM_RADIUS / np.linalg.norm(L))

    rng = np.random.default_rng(0)
    dirs = _sample_cone(axis, cone, N_SAMPLES, rng)
    wl = rng.uniform(BAND[0], BAND[1], N_SAMPLES)
    seq_fi, seq_kind, nint, exited = trace_record_sequence(
        gem, np.broadcast_to(L, dirs.shape).copy(), dirs, wl, n_of, max_bounces=MAX_BOUNCES)

    groups = defaultdict(list)
    for i in np.where(exited)[0]:
        Ln = nint[i]
        groups[(Ln, *seq_fi[i, :Ln], *seq_kind[i, :Ln])].append(i)
    chains = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    fcol, frow, em_x, em_y, wls = [], [], [], [], []
    for key, idxs in chains[:TOP_CHAINS]:
        if len(idxs) < MIN_RAYS:
            break
        idxs = np.array(idxs)
        Ln = key[0]
        seq = list(zip(key[1:1 + Ln], key[1 + Ln:1 + 2 * Ln]))
        res = solver(facets, seq, L, dirs[idxs], wl[idxs], n_of, dn_of, C,
                     max_iter=MAX_ITER, tol=TOL)
        m = res["converged"]
        if not m.any():
            continue
        col, row, valid = project(res["P"][m])
        emit = res["emit"][m]
        fcol.append(col[valid]); frow.append(row[valid])
        em_x.append(emit[valid] @ e1); em_y.append(emit[valid] @ e2)
        wls.append(res["wl"][m][valid])

    fcol = np.concatenate(fcol); frow = np.concatenate(frow)
    em_x = np.concatenate(em_x); em_y = np.concatenate(em_y)
    wls = np.concatenate(wls)
    print(f"solver {args.solver}: converged connections plotted: {fcol.size:,}")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    h0 = axs[0].hist2d(fcol, frow, bins=120, range=[[0, WIDTH], [0, HEIGHT]], cmap="magma")
    fig.colorbar(h0[3], ax=axs[0]); axs[0].set_title("connections on camera film (density)")
    axs[0].set_xlabel("col"); axs[0].set_ylabel("row"); axs[0].set_aspect("equal")

    h1 = axs[1].hist2d(em_x, em_y, bins=120, cmap="viridis")
    fig.colorbar(h1[3], ax=axs[1]); axs[1].set_title("emission directions (density)")
    axs[1].set_xlabel("· e1"); axs[1].set_ylabel("· e2"); axs[1].set_aspect("equal")

    rgb = np.clip([wavelength_to_rgb(w * 1000.0) for w in wls], 0, 1)
    axs[2].scatter(fcol, frow, s=4, c=rgb)
    axs[2].set_title("film connections coloured by wavelength")
    axs[2].set_xlim(0, WIDTH); axs[2].set_ylim(0, HEIGHT); axs[2].set_aspect("equal")
    axs[2].set_facecolor("black")

    fig.tight_layout()
    out = pathlib.Path(f"runs/density_pass2_{args.solver}.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
