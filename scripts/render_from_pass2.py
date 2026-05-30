"""Functionality (b): load an exported connect pass and re-render at a NEW
accept_eps — no tracing, no Newton, just threshold + splat.

    PYTHONPATH=src python scripts/render_from_pass2.py \
        --pass2 runs/connect_pass2.npz --accept-eps 0.05 --png runs/fire_eps05.png
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter

from diffrt.diamond.lightconnect import (camera_from_meta, load_pass2,
                                         render_from_pass2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pass2", default="runs/connect_pass2.npz")
    ap.add_argument("--accept-eps", type=float, default=0.10)
    ap.add_argument("--exposure", type=float, default=0.9)
    ap.add_argument("--blur", type=float, default=1.2)
    ap.add_argument("--png", default="runs/diamond_fire_reeps.png")
    args = ap.parse_args()

    pass2, meta = load_pass2(args.pass2)
    camera = camera_from_meta(meta)
    img, n_acc = render_from_pass2(pass2, camera, meta["width"], meta["height"],
                                   args.accept_eps)
    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], args.blur)
    srgb = np.clip(img * (args.exposure / max(img.max(), 1e-12)), 0, 1) ** (1 / 2.2)
    mpimg.imsave(args.png, srgb)

    finite = pass2["dist"][np.isfinite(pass2["dist"])]
    print(f"loaded {args.pass2}: {pass2['dist'].size:,} connections "
          f"(final dist min {finite.min():.3e})")
    print(f"accepted @ eps={args.accept_eps}: {n_acc}")
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
