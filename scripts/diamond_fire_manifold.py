"""MC-1: exact fire via Newton manifold projection (camera-side).

Same scene as scripts/diamond_fire.py (point light, pure-specular diamond,
pinhole) — black without help. Here a (pixel × λ) seed grid is Newton-projected
onto the connecting manifold (exit ray passes exactly through the point light),
and the converged exit points are splatted, coloured by wavelength. The 1-DOF
manifold is swept by λ, tracing exact fire curves (no gather blur).

    PYTHONPATH=src python scripts/diamond_fire_manifold.py
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
from diffrt.diamond.manifold import render_fire_manifold
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n

WIDTH, HEIGHT = 160, 160
N_LAMBDA = 24
MAX_ITER = 12
EXPOSURE = 0.8
BLUR_SIGMA = 0.7

CAMERA = (                 # identical to the black point-light render
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
    img, st = render_fire_manifold(gem, CAMERA, LIGHT_POS, n_of, dn_of,
                                   WIDTH, HEIGHT, n_lambda=N_LAMBDA,
                                   max_iter=MAX_ITER)
    dt = time.time() - t0

    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], BLUR_SIGMA)
    scale = EXPOSURE / max(img.max(), 1e-12)
    srgb = np.clip(img * scale, 0.0, 1.0) ** (1.0 / 2.2)

    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    png = out / "diamond_fire_manifold.png"
    mpimg.imsave(str(png), srgb)

    print(f"rendered {WIDTH}x{HEIGHT}, {N_LAMBDA} wavelengths in {dt:.1f}s")
    print(f"seeds: {st['seeds']}  converged: {st['converged']} "
          f"({st['success_frac']:.1%})")
    print(f"mean Newton iters: {st['mean_iters']:.2f}  "
          f"max residual: {st['max_resid']:.2e}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
