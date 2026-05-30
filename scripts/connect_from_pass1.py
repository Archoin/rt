"""Functionality (a): load an exported sample pass and run a NEW 2nd pass.

Reuses the cached stage-1 samples (no re-tracing the light) and runs the connect
stage with possibly different parameters (--top-k, --newton-iters) or a different
solver, then exports a new pass2 and renders. To swap the algorithm, import a
different ``solver`` and pass it to ``connect_pass``.

    PYTHONPATH=src python scripts/connect_from_pass1.py \
        --pass1 runs/connect_pass1.npz --top-k 150000 --newton-iters 80 \
        --accept-eps 0.1 --out runs/connect_pass2_b.npz
"""

from __future__ import annotations

import argparse
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter

from diffrt.diamond.lightconnect import (build_gem, camera_from_meta,
                                         connect_pass, load_pass1,
                                         render_from_pass2, save_pass2)
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pass1", default="runs/connect_pass1.npz")
    ap.add_argument("--top-k", type=int, default=100_000)
    ap.add_argument("--newton-iters", type=int, default=50)
    ap.add_argument("--accept-eps", type=float, default=0.10)
    ap.add_argument("--out", default="runs/connect_pass2_replay.npz")
    ap.add_argument("--png", default="runs/diamond_fire_connect_replay.png")
    args = ap.parse_args()

    samples, meta = load_pass1(args.pass1)
    gem = build_gem(meta)
    camera = camera_from_meta(meta)
    n_of = lambda um: sellmeier_n(um, meta["material"])
    dn_of = lambda um: sellmeier_dn_dlambda(um, meta["material"])

    t0 = time.time()
    pass2 = connect_pass(gem, meta["light_pos"], camera[0], samples, n_of, dn_of,
                         top_k=args.top_k, newton_iters=args.newton_iters,
                         max_bounces=meta["max_bounces"], band=tuple(meta["band"]))
    dt = time.time() - t0
    save_pass2(args.out, pass2, meta)

    img, n_acc = render_from_pass2(pass2, camera, meta["width"], meta["height"],
                                   args.accept_eps)
    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], 1.2)
    srgb = np.clip(img * (0.9 / max(img.max(), 1e-12)), 0.0, 1.0) ** (1 / 2.2)
    mpimg.imsave(args.png, srgb)

    print(f"loaded {args.pass1}: {samples['dirs'].shape[0]:,} samples")
    print(f"connect: top_k={pass2['seed_idx'].size:,} newton_iters={args.newton_iters} "
          f"{dt:.1f}s  final min {pass2['dist'].min():.3e}")
    print(f"accepted @ eps={args.accept_eps}: {n_acc}")
    print(f"exported {args.out}; wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
