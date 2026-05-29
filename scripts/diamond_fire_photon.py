"""Direction 1: light-side progressive photon-differential splatting → fire.

Same scene as scripts/diamond_fire.py (point light, pure-specular diamond,
pinhole) — which renders BLACK. Here photon beams are traced FROM the point
light through the diamond and splatted onto the film, weighted by the
photon-differential flux density and coloured by wavelength. Multiple passes
with a shrinking gather angle make it progressive.

    PYTHONPATH=src python scripts/diamond_fire_photon.py
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
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render_photon import render_fire_photon

WIDTH, HEIGHT = 200, 200
BEAMS_PER_PASS = 400_000
PASSES = 4
GATHER_DEG = 3.0
MAX_BOUNCES = 18
EXPOSURE = 0.6
BLUR_SIGMA = 1.0           # splat footprint (px)

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
    img, lit = render_fire_photon(gem, CAMERA, LIGHT_POS, n_of, dn_of,
                                  WIDTH, HEIGHT, beams_per_pass=BEAMS_PER_PASS,
                                  passes=PASSES, gather_deg=GATHER_DEG,
                                  max_bounces=MAX_BOUNCES)
    dt = time.time() - t0

    for c in range(3):                                   # splat footprint
        img[:, :, c] = gaussian_filter(img[:, :, c], BLUR_SIGMA)
    scale = EXPOSURE / max(img.max(), 1e-12)
    srgb = np.clip(img * scale, 0.0, 1.0) ** (1.0 / 2.2)

    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    png = out / "diamond_fire_photon.png"
    mpimg.imsave(str(png), srgb)

    print(f"rendered {WIDTH}x{HEIGHT}, {PASSES}x{BEAMS_PER_PASS} beams in {dt:.1f}s")
    print(f"splat hits: {lit}")
    print(f"wrote {png}")
    print("compare: scripts/diamond_fire.py (point light, no differentials) = black")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
