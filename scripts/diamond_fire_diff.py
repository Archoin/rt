"""Differential render of diamond fire under a POINT light.

Same scene as scripts/diamond_fire.py (point light, pure-specular diamond,
pinhole camera) — which renders BLACK — but here each pixel beam carries the
beam-differential footprint and is connected to the point light within that
footprint. The point light's contribution now covers a finite image region, so
the black image becomes fire, coloured by the connecting wavelength (dn/dλ).

Run from the repo root:

    PYTHONPATH=src python scripts/diamond_fire_diff.py
"""

from __future__ import annotations

import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np

from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render_diff import render_fire_diff

WIDTH, HEIGHT = 120, 120
LAMBDA0_UM = 0.55
MAX_BOUNCES = 18
GATHER_PX = 22.0            # footprint gather radius (pixel-footprint units)
EXPOSURE = 1.4

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
    img, lit = render_fire_diff(gem, CAMERA, LIGHT_POS, n_of, dn_of, LAMBDA0_UM,
                                WIDTH, HEIGHT, gather_px=GATHER_PX,
                                max_bounces=MAX_BOUNCES)
    dt = time.time() - t0

    srgb = np.clip(img * EXPOSURE, 0.0, 1.0) ** (1.0 / 2.2)
    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    png = out / "diamond_fire_diff.png"
    mpimg.imsave(str(png), srgb)

    print(f"rendered {WIDTH}x{HEIGHT} in {dt:.1f}s")
    print(f"lit pixels: {lit:.3%}   max radiance: {img.max():.4g}")
    print(f"wrote {png}")
    print("compare: scripts/diamond_fire.py (point light, no differentials) = black")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
