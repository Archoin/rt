"""Light-side sample-and-connect fire render (MC-1, take 2).

Same scene as scripts/diamond_fire.py (point light, pure-specular diamond,
pinhole) — black without help. Here we trace from the light, sort rays by
distance to the camera, and Newton-refine the closest seeds into exact
connections (accept if distance < accept_eps), then splat them coloured by
wavelength. Contrast with the black baseline runs/diamond_fire.png.

    PYTHONPATH=src python scripts/diamond_fire_connect.py
"""

from __future__ import annotations

import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter

from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.lightconnect import render_fire_connect
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n

WIDTH, HEIGHT = 200, 200
N_SAMPLES = 1_500_000
TOP_K = 200_000
ACCEPT_EPS = 0.10           # world-distance criterion to the pinhole (≈ aperture)
NEWTON_ITERS = 50
MAX_BOUNCES = 18
BAND = (0.40, 0.70)
EXPOSURE = 0.9
BLUR_SIGMA = 1.2

CAMERA = (                  # identical to the black point-light render
    (0.0, 2.05, 3.85),
    (0.0, -0.10, 0.0),
    (0.0, 1.0, 0.0),
    30.0,
)
LIGHT_POS = (0.9, 3.0, 0.6)


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0,
                          table_frac=0.55)
    n_of = lambda um: sellmeier_n(um, "diamond")
    dn_of = lambda um: sellmeier_dn_dlambda(um, "diamond")

    t0 = time.time()
    img, st = render_fire_connect(gem, CAMERA, LIGHT_POS, n_of, dn_of,
                                  WIDTH, HEIGHT, n_samples=N_SAMPLES,
                                  top_k=TOP_K, accept_eps=ACCEPT_EPS,
                                  band=BAND, newton_iters=NEWTON_ITERS,
                                  max_bounces=MAX_BOUNCES)
    dt = time.time() - t0

    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], BLUR_SIGMA)
    scale = EXPOSURE / max(img.max(), 1e-12)
    srgb = np.clip(img * scale, 0.0, 1.0) ** (1.0 / 2.2)

    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    png = out / "diamond_fire_connect.png"
    mpimg.imsave(str(png), srgb)

    print(f"rendered {WIDTH}x{HEIGHT} in {dt:.1f}s")
    print(f"samples: {st['n_samples']}  exited: {st['n_exited']}  "
          f"seeds refined: {st['seeds_refined']}")
    print(f"accepted: {st['n_accepted']}  (accept rate {st['accept_rate']:.2%})")
    print(f"seed min dist: {st['seed_min_dist']:.4f}  "
          f"final min: {st['final_min_dist']:.2e}  "
          f"median: {st['final_median_dist']:.2e}  (eps={ACCEPT_EPS})")
    print(f"wrote {png}")
    print("compare: scripts/diamond_fire.py (point light, no connect) = black")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
