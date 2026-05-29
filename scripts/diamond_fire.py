"""Point light + perfect reflection/refraction → (almost) black image.

Demonstrates the project's premise. With a POINT light, a pinhole camera, and a
purely specular dielectric diamond (perfect reflection + refraction, no
diffuse/glossy), camera rays form deterministic specular trees that can only
carry energy if a branch terminates EXACTLY on the zero-measure point light —
probability zero. The image is therefore black: naive ray tracing cannot
capture the diamond's fire. This black baseline is what the beam-differential
method must turn into fire.

Run from the repo root:

    PYTHONPATH=src python scripts/diamond_fire.py
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np

from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import sellmeier_n
from diffrt.diamond.render import render_spectral

WIDTH, HEIGHT = 200, 200
N_WAVELENGTHS = 14
MAX_BOUNCES = 16
EXPOSURE = 2.2

CAMERA = (
    (0.0, 2.05, 3.85),     # eye, front-above (pulled back for margin)
    (0.0, -0.10, 0.0),     # look-at
    (0.0, 1.0, 0.0),       # up
    30.0,                  # vertical fov (deg)
)

LIGHT_POS = np.array([0.9, 3.0, 0.6])   # point light, above/front of the stone
LIGHT_INTENSITY = 1.0
LIGHT_RADIUS = 0.0                       # TRUE point → zero-measure target


def point_light(L, intensity, radius):
    """Terminal radiance for a point light.

    Nonzero only if a terminal ray ``(o, d)`` actually passes through ``L``
    (perpendicular distance ≤ ``radius``). For ``radius=0`` this is a
    zero-measure event — essentially never — so the background is black.
    """
    L = np.asarray(L, dtype=float)

    def light_fn(o, d):
        if o.shape[0] == 0:
            return np.zeros(0)
        LO = L[None, :] - o
        tstar = np.sum(LO * d, axis=1)              # closest-approach param
        perp = LO - tstar[:, None] * d
        dist = np.linalg.norm(perp, axis=1)
        hit = (tstar > 0) & (dist <= radius)        # does the ray pass through L?
        return np.where(hit, intensity, 0.0)

    return light_fn


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0,
                          table_frac=0.55)
    light_fn = point_light(LIGHT_POS, LIGHT_INTENSITY, LIGHT_RADIUS)
    n_of = lambda um: sellmeier_n(um, "diamond")
    wavelengths = np.linspace(420.0, 680.0, N_WAVELENGTHS)

    img = render_spectral(gem, CAMERA, light_fn, n_of, wavelengths,
                          WIDTH, HEIGHT, max_bounces=MAX_BOUNCES)

    srgb = np.clip(img * EXPOSURE, 0.0, 1.0) ** (1.0 / 2.2)
    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    png = out / "diamond_point_light.png"
    mpimg.imsave(str(png), srgb)

    lit = float((img.max(axis=2) > 1e-9).mean())
    print(f"point light radius : {LIGHT_RADIUS}")
    print(f"lit pixels         : {lit:.4%}")
    print(f"max radiance       : {img.max():.4g}")
    print(f"wrote {png}")
    print("expected: black, ~0% lit  (point-light + pure specular = measure zero)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
