"""Fixed-facet-sequence model: validate its Jacobian, then run the solver on it.

(1) Re-trace the closest near-connection rays to record their specular chains.
(2) Finite-difference-check the fixed-sequence Jacobian (must match analytic).
(3) Take the closest seed's chain, gather all rays sharing it, and run the
    Gauss-Newton solver on the SMOOTH fixed-sequence model — compare its
    accept/reject behaviour to the free-retrace rejection storm.

    PYTHONPATH=src python scripts/fixed_sequence_solver.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffrt.diamond.fixedseq import (connect_fixed_newton,
                                     trace_fixed_sequence_vec,
                                     trace_record_sequence)
from diffrt.diamond.lightconnect import (build_gem, camera_from_meta,
                                         load_pass1, ray_point_distance)
from diffrt.diamond.optics import normalize, sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render_photon import _make_frames

SUBSET = 200_000          # re-trace this many closest exited rays


def fd_check(facets, seq, O, D0, wl, n_of, dn_of):
    """Max error between analytic and central-FD Jacobian on the fixed seq."""
    e1, e2 = _make_frames(D0)
    ana = trace_fixed_sequence_vec(facets, seq, O, D0, wl, n_of, dn_of, frame=(e1, e2))
    d = 1e-5
    errs = []
    for k, pert in enumerate((("e1", e1), ("e2", e2), ("wl", None))):
        name, vec = pert
        if name == "wl":
            rp = trace_fixed_sequence_vec(facets, seq, O, D0, wl + d, n_of, dn_of)
            rm = trace_fixed_sequence_vec(facets, seq, O, D0, wl - d, n_of, dn_of)
        else:
            Dp = D0 + d * vec; Dp /= np.linalg.norm(Dp, axis=1, keepdims=True)
            Dm = D0 - d * vec; Dm /= np.linalg.norm(Dm, axis=1, keepdims=True)
            rp = trace_fixed_sequence_vec(facets, seq, O, Dp, wl, n_of, dn_of)
            rm = trace_fixed_sequence_vec(facets, seq, O, Dm, wl, n_of, dn_of)
        fdP = (rp["P"] - rm["P"]) / (2 * d)
        fdD = (rp["D"] - rm["D"]) / (2 * d)
        eP = np.max(np.abs(fdP - ana["Pj"][:, :, k]))
        eD = np.max(np.abs(fdD - ana["Dj"][:, :, k]))
        errs.append((name, eP, eD))
    return errs


def main() -> int:
    samples, meta = load_pass1("runs/connect_pass1.npz")
    gem = build_gem(meta); facets = gem.facets
    cam = camera_from_meta(meta); C = np.asarray(cam[0], float)
    L = np.asarray(meta["light_pos"], float)
    n_of = lambda um: sellmeier_n(um, meta["material"])
    dn_of = lambda um: sellmeier_dn_dlambda(um, meta["material"])

    ex = samples["exited"]
    order = np.argsort(np.where(ex, samples["dist"], np.inf))[:SUBSET]
    dirs, wl = samples["dirs"][order], samples["wl"][order]
    seq_fi, seq_kind, nint, exited = trace_record_sequence(gem, np.broadcast_to(L, dirs.shape).copy(),
                                                           dirs, wl, n_of, max_bounces=meta["max_bounces"])

    s0 = 0                                              # closest exited ray
    Ls = int(nint[s0])
    sfi, skd = seq_fi[s0, :Ls], seq_kind[s0, :Ls]
    seq = list(zip(sfi.tolist(), skd.tolist()))
    kind_name = {0: "enter", 1: "exit", 2: "tir"}
    print("closest seed chain:", " -> ".join(f"F{f}:{kind_name[k]}" for f, k in seq))

    same = (exited & (nint == Ls)
            & np.all(seq_fi[:, :Ls] == sfi, axis=1)
            & np.all(seq_kind[:, :Ls] == skd, axis=1))
    grp = np.where(same)[0]
    print(f"rays sharing this exact chain: {grp.size:,} of {SUBSET:,} closest")

    # (2) Jacobian validation on a slice of the group
    val = grp[:2000]
    errs = fd_check(facets, seq, np.broadcast_to(L, (val.size, 3)).copy(),
                    dirs[val], wl[val], n_of, dn_of)
    print("fixed-seq Jacobian vs finite differences (max abs err):")
    for name, eP, eD in errs:
        print(f"  d/d{name:>2}:  dP {eP:.2e}   dD {eD:.2e}")

    # (3) solver on the smooth fixed-sequence model
    res = connect_fixed_newton(facets, seq, L, dirs[grp], wl[grp], n_of, dn_of,
                               C, max_iter=40, tol=1e-9, band=tuple(meta["band"]))
    h = res["hist"]
    print(f"\nsolver on fixed sequence ({grp.size:,} rays):")
    print(f"  final resid: min {res['resid'].min():.3e}  median {np.median(res['resid']):.3e}")
    tot_acc = sum(h["accepts"]); tot_rej = sum(h["rejects"])
    print(f"  accepts {tot_acc:,}  rejects {tot_rej:,}  "
          f"accept-rate {tot_acc/(tot_acc+tot_rej):.1%}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    axs[0].plot(h["mean_dist"], "-o", ms=3)
    axs[0].set_yscale("log"); axs[0].set_title("median residual |f| vs iteration (fixed seq)")
    axs[0].set_xlabel("iteration"); axs[0].set_ylabel("median |f|")
    it = np.arange(len(h["accepts"]))
    axs[1].plot(it, h["accepts"], label="accepts")
    axs[1].plot(it, h["rejects"], label="rejects")
    axs[1].set_title("accepts / rejects per iteration"); axs[1].set_xlabel("iteration")
    axs[1].legend()
    fig.tight_layout()
    out = pathlib.Path("runs/fixed_seq_solver.png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
