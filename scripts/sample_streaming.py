"""1st pass (improved): stream many samples, keep best-N per film bin, export.

Memory-bounded streaming sampler — total samples is compute-bound, not memory-
bound. Reports how the closest-seed distance improves with more samples and the
peak memory, then exports the kept best-N as runs/connect_pass1.npz for the 2nd
pass.

    PYTHONPATH=src python scripts/sample_streaming.py --total 20000000
"""

from __future__ import annotations

import argparse
import pathlib
import resource
import time

import numpy as np

from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.lightconnect import (make_meta, sample_pass_streaming,
                                         save_pass1)
from diffrt.diamond.optics import sellmeier_n

WIDTH, HEIGHT = 200, 200
BAND, GEM_RADIUS, MAX_BOUNCES = (0.40, 0.70), 1.3, 18
CROWN, PAVILION, TABLE, MATERIAL = 34.0, 41.0, 0.55, "diamond"
CAMERA = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
LIGHT_POS = (0.9, 3.0, 0.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=20_000_000)
    ap.add_argument("--batch", type=int, default=1_000_000)
    ap.add_argument("--film-bins", type=int, default=96)
    ap.add_argument("--per-bin", type=int, default=64)
    ap.add_argument("--out", default="runs/connect_pass1.npz")
    args = ap.parse_args()

    gem = round_brilliant(crown_angle_deg=CROWN, pavilion_angle_deg=PAVILION,
                          table_frac=TABLE)
    n_of = lambda um: sellmeier_n(um, MATERIAL)

    t0 = time.time()
    samples = sample_pass_streaming(
        gem, LIGHT_POS, CAMERA, n_of, args.total, batch_size=args.batch,
        band=BAND, gem_radius=GEM_RADIUS, max_bounces=MAX_BOUNCES,
        film_bins=(args.film_bins, args.film_bins), per_bin=args.per_bin)
    dt = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    meta = make_meta(LIGHT_POS, CAMERA, BAND, MAX_BOUNCES, CROWN, PAVILION,
                     TABLE, MATERIAL, args.total, 0, GEM_RADIUS, WIDTH, HEIGHT)
    pathlib.Path(args.out).parent.mkdir(exist_ok=True)
    save_pass1(args.out, samples, meta)

    d = samples["dist"]
    print(f"total sampled : {args.total:,} (batch {args.batch:,})")
    print(f"kept best-N   : {d.size:,}  (film bins {args.film_bins}^2, "
          f"per-bin {args.per_bin})")
    print(f"seed dist     : min {d.min():.4e}  median {np.median(d):.4e}")
    print(f"time {dt:.1f}s   peak RSS {rss:.0f} MB")
    print(f"exported {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
