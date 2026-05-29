"""Render a brilliant-cut diamond showing dispersion ("fire") with our pipeline.

Spectral camera render: for each wavelength, camera rays refract/TIR through the
diamond (diamond Sellmeier n(λ)) and sample a lit environment of bright spots.
Per-wavelength dispersion makes the exit directions fan out, so the spots break
into spectral colors — the fire.

Run from the repo root:

    PYTHONPATH=src python scripts/diamond_fire.py
"""

from __future__ import annotations

import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np

from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import normalize, sellmeier_n
from diffrt.diamond.render import render_spectral

WIDTH, HEIGHT = 260, 260
N_WAVELENGTHS = 28
MAX_BOUNCES = 16
EXPOSURE = 2.2

CAMERA = (
    (0.0, 2.05, 3.85),     # eye, front-above (pulled back for margin)
    (0.0, -0.10, 0.0),     # look-at (toward the stone)
    (0.0, 1.0, 0.0),       # up
    30.0,                  # vertical fov (deg)
)


def build_environment():
    """A few bright spots on a dim sky → the sparkle that fire breaks up."""
    rng = np.random.default_rng(7)
    spots = []
    # one strong, tight key light, front-above
    spots.append((normalize(np.array([0.15, 0.9, 0.45])), 22.0, 0.025))
    # many tight fill lights over the upper hemisphere + some side/low ones,
    # so dispersed rays sweeping across their sharp edges break into color
    for _ in range(17):
        v = rng.normal(size=3)
        v[1] = abs(v[1]) * 1.2 + 0.05           # bias upward
        spots.append((normalize(v), rng.uniform(6.0, 14.0),
                      rng.uniform(0.018, 0.038)))

    dirs = np.array([s[0] for s in spots])
    inten = np.array([s[1] for s in spots])
    hw = np.array([s[2] for s in spots])

    def env_fn(d):                              # d: (K,3) unit
        cosang = np.clip(d @ dirs.T, -1, 1)     # (K, S)
        ang = np.arccos(cosang)
        contrib = inten[None, :] * np.exp(-(ang / hw[None, :]) ** 2)
        ambient = 0.015 + 0.05 * np.clip(d[:, 1], 0, 1)   # dark surround
        return contrib.sum(axis=1) + ambient

    return env_fn


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0,
                          table_frac=0.55)
    env_fn = build_environment()
    n_of = lambda um: sellmeier_n(um, "diamond")
    wavelengths = np.linspace(420.0, 680.0, N_WAVELENGTHS)

    t0 = time.time()
    img = render_spectral(gem, CAMERA, env_fn, n_of, wavelengths,
                          WIDTH, HEIGHT, max_bounces=MAX_BOUNCES)
    dt = time.time() - t0

    # tone map: exposure → clip → sRGB gamma
    srgb = np.clip(img * EXPOSURE, 0.0, 1.0) ** (1.0 / 2.2)

    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    png = out / "diamond_fire.png"
    mpimg.imsave(str(png), srgb)

    frac_lit = float((img.max(axis=2) > 0.01).mean())
    print(f"rendered {WIDTH}x{HEIGHT}, {N_WAVELENGTHS} wavelengths, "
          f"{MAX_BOUNCES} bounces in {dt:.1f}s")
    print(f"facets: {len(gem.facets)} | nonzero pixels: {frac_lit:.0%}")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
