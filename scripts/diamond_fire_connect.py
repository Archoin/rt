"""Full light-side sample-and-connect pipeline (point light, pinhole).

Stage 1 (sample, central-only → memory-light) → export runs/connect_pass1.npz
Stage 2 (Newton connect on top-K)            → export runs/connect_pass2.npz
Stage 3 (threshold by accept_eps + splat)    → runs/diamond_fire_connect.png

Re-run stage 2 with different params/algorithm: scripts/connect_from_pass1.py
Re-render with a different accept_eps:          scripts/render_from_pass2.py

    PYTHONPATH=src python scripts/diamond_fire_connect.py
"""

from __future__ import annotations

import pathlib
import resource
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter

from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.lightconnect import (connect_pass, make_meta,
                                         render_from_pass2, sample_pass,
                                         save_pass1, save_pass2)
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n

WIDTH, HEIGHT = 200, 200
N_SAMPLES = 1_000_000
TOP_K = 100_000
ACCEPT_EPS = 0.10
NEWTON_ITERS = 50
MAX_BOUNCES = 18
BAND = (0.40, 0.70)
GEM_RADIUS = 1.3
MATERIAL = "diamond"
CROWN, PAVILION, TABLE = 34.0, 41.0, 0.55
EXPOSURE, BLUR_SIGMA = 0.9, 1.2
SEED = 0

CAMERA = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
LIGHT_POS = (0.9, 3.0, 0.6)


def tonemap_save(img, path, exposure=EXPOSURE, blur=BLUR_SIGMA):
    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], blur)
    srgb = np.clip(img * (exposure / max(img.max(), 1e-12)), 0.0, 1.0) ** (1 / 2.2)
    mpimg.imsave(str(path), srgb)


def main() -> int:
    gem = round_brilliant(crown_angle_deg=CROWN, pavilion_angle_deg=PAVILION,
                          table_frac=TABLE)
    n_of = lambda um: sellmeier_n(um, MATERIAL)
    dn_of = lambda um: sellmeier_dn_dlambda(um, MATERIAL)
    C = np.asarray(CAMERA[0], float)
    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    meta = make_meta(LIGHT_POS, CAMERA, BAND, MAX_BOUNCES, CROWN, PAVILION,
                     TABLE, MATERIAL, N_SAMPLES, SEED, GEM_RADIUS, WIDTH, HEIGHT)

    t0 = time.time()
    samples = sample_pass(gem, LIGHT_POS, C, n_of, N_SAMPLES, BAND,
                          gem_radius=GEM_RADIUS, max_bounces=MAX_BOUNCES, seed=SEED)
    t_sample = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    save_pass1(out / "connect_pass1.npz", samples, meta)

    t1 = time.time()
    pass2 = connect_pass(gem, LIGHT_POS, C, samples, n_of, dn_of, top_k=TOP_K,
                         newton_iters=NEWTON_ITERS, max_bounces=MAX_BOUNCES,
                         band=BAND)
    t_connect = time.time() - t1
    save_pass2(out / "connect_pass2.npz", pass2, meta)

    img, n_acc = render_from_pass2(pass2, CAMERA, WIDTH, HEIGHT, ACCEPT_EPS)
    tonemap_save(img, out / "diamond_fire_connect.png")

    print(f"sample: {N_SAMPLES:,} rays, {int(samples['exited'].sum()):,} exited, "
          f"{t_sample:.1f}s, peak RSS {rss:.0f} MB")
    print(f"connect: {pass2['seed_idx'].size:,} seeds, {t_connect:.1f}s, "
          f"final min {pass2['dist'].min():.3e}")
    print(f"accepted @ eps={ACCEPT_EPS}: {n_acc}")
    print(f"exported: {out/'connect_pass1.npz'}  {out/'connect_pass2.npz'}")
    print(f"wrote {out/'diamond_fire_connect.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
