"""Diamond fire via beam-differential footprint splatting through a FINITE
aperture — the physically-correct replacement for the (measure-zero) point-light
exact connection.

Point light + pinhole + pure specular = black (no connections exist). A finite
aperture turns the measure-zero connection into positive measure; the beam
differential gives each accepted photon a film footprint, and dispersion fans the
glints into rainbow streaks (the fire).

    PYTHONPATH=src python scripts/diamond_fire_differential.py
"""

from __future__ import annotations

import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import gaussian_filter

from diffrt.diamond.explog import log_run, run_dir
from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render_photon import render_fire_differential

WIDTH, HEIGHT = 320, 320
CAMERA = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
LIGHT_POS = (0.9, 3.0, 0.6)
N_SAMPLES, BATCHES, APERTURE = 12_000_000, 48, 0.18
EXPOSURE, BLUR = 1.0, 0.5


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0, table_frac=0.55)
    n_of = lambda um: sellmeier_n(um, "diamond")
    dn_of = lambda um: sellmeier_dn_dlambda(um, "diamond")

    img, accepted, dists = render_fire_differential(
        gem, CAMERA, LIGHT_POS, n_of, dn_of, WIDTH, HEIGHT,
        n_samples=N_SAMPLES, batches=BATCHES, aperture=APERTURE, seed=0)

    print(f"samples={N_SAMPLES:,}  aperture={APERTURE}  accepted={accepted:,} "
          f"({100*accepted/N_SAMPLES:.3f}%)")
    if dists.size:
        pct = np.percentile(dists, [0, 25, 50, 75, 100])
        print(f"aperture-crossing dist (min/25/50/75/max): "
              f"{pct[0]:.4f}/{pct[1]:.4f}/{pct[2]:.4f}/{pct[3]:.4f}/{pct[4]:.4f}")

    outdir = run_dir("fire_differential")
    np.save(outdir / "fire.npy", img)                     # raw HDR buffer for re-tonemapping
    for c in range(3):
        img[:, :, c] = gaussian_filter(img[:, :, c], BLUR)
    # HDR (Reinhard) tonemap: reveals faint glints without blowing out the bright one
    lum = img.sum(2)
    scale = EXPOSURE / max(np.mean(lum[lum > 0]) if (lum > 0).any() else 1.0, 1e-12)
    t = scale * img
    srgb = np.clip(t / (1.0 + t), 0, 1) ** (1 / 2.2)
    png = outdir / "fire.png"
    mpimg.imsave(str(png), srgb)

    log_run("fire_differential",
            params=dict(width=WIDTH, height=HEIGHT, n_samples=N_SAMPLES, batches=BATCHES,
                        aperture=APERTURE, camera=CAMERA, light=LIGHT_POS,
                        band=(0.40, 0.70), exposure=EXPOSURE, blur=BLUR),
            metrics=dict(accepted=accepted, accept_frac=accepted / N_SAMPLES,
                         dist_min=(float(dists.min()) if dists.size else None),
                         dist_median=(float(np.median(dists)) if dists.size else None)),
            outputs=[png, outdir / "fire.npy"],
            notes="differential-footprint fire: valid bounded paths + finite aperture")
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
