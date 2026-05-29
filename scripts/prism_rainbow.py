"""M1 — Newton's prism: disperse a white ray into a spectrum.

Traces one ray per wavelength through a triangular glass prism onto a screen,
producing a fan of colors. Validates the optics foundation (n(λ), Snell
refraction, TIR, facet geometry) before any differential machinery (M2).

Run from the repo root:

    PYTHONPATH=src python scripts/prism_rainbow.py

Writes runs/prism_rainbow.png and prints a per-wavelength deviation table plus a
physical sanity check (violet must deviate more than red).
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffrt.diamond.geometry import intersect_plane, triangular_prism
from diffrt.diamond.optics import sellmeier_n
from diffrt.diamond.trace import trace_center_ray

MATERIAL = "BK7"
SCREEN_X = 8.0                       # vertical screen plane at x = SCREEN_X
# Incidence chosen near MINIMUM DEVIATION for the center wavelength: this makes
# the interior ray ~horizontal and keeps internal angles (~30°) well below BK7's
# critical angle (~41°), so the whole visible band transmits without TIR.
RAY_D = np.array([0.9426, 0.3339, 0.0])   # from lower-left, up-right ~19.4°
RAY_O = np.array([-0.5, 0.289, 0.0]) - 4.0 * RAY_D  # aim at the left-face mid


def wavelength_to_rgb(nm: float):
    """Approximate visible-wavelength → linear RGB (Dan Bruton's algorithm)."""
    if nm < 380 or nm > 750:
        return (0.0, 0.0, 0.0)
    if nm < 440:
        r, g, b = -(nm - 440) / (440 - 380), 0.0, 1.0
    elif nm < 490:
        r, g, b = 0.0, (nm - 440) / (490 - 440), 1.0
    elif nm < 510:
        r, g, b = 0.0, 1.0, -(nm - 510) / (510 - 490)
    elif nm < 580:
        r, g, b = (nm - 510) / (580 - 510), 1.0, 0.0
    elif nm < 645:
        r, g, b = 1.0, -(nm - 645) / (645 - 580), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    if nm < 420:
        s = 0.3 + 0.7 * (nm - 380) / (420 - 380)
    elif nm > 700:
        s = 0.3 + 0.7 * (750 - nm) / (750 - 700)
    else:
        s = 1.0
    return (r * s, g * s, b * s)


def main() -> int:
    prism = triangular_prism(side=2.0, depth=2.0)
    n_of = lambda um: sellmeier_n(um, MATERIAL)
    screen_p0, screen_n = np.array([SCREEN_X, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])

    nm_values = np.linspace(400.0, 700.0, 24)

    fig, ax = plt.subplots(figsize=(11, 6))

    # prism outline (xy cross-section)
    s, h = 2.0, 2.0 * np.sqrt(3) / 2
    tri = np.array([[-1, -h / 3], [1, -h / 3], [0, 2 * h / 3], [-1, -h / 3]])
    ax.plot(tri[:, 0], tri[:, 1], "k-", lw=1.5, zorder=5)
    ax.axvline(SCREEN_X, color="0.4", lw=2)
    ax.text(SCREEN_X, -1.6, " screen", color="0.4", va="top")

    ray_d = RAY_D / np.linalg.norm(RAY_D)
    rows = []
    all_xy = [tri]
    for nm in nm_values:
        res = trace_center_ray(prism, RAY_O, ray_d, n_of, nm / 1000.0)
        rgb = wavelength_to_rgb(nm)

        pts = [p[:2] for p in res["path"]]
        screen_y = np.nan
        if res["exited"] and not res["maxed"]:
            ts = intersect_plane(res["exit_o"], res["exit_d"], screen_p0, screen_n)
            if ts is not None:
                hit = res["exit_o"] + ts * res["exit_d"]
                pts.append(hit[:2])
                screen_y = hit[1]
        poly = np.array(pts)
        all_xy.append(poly)
        ax.plot(poly[:, 0], poly[:, 1], "-", color=rgb, lw=1.0, alpha=0.9)

        dev = np.degrees(np.arccos(np.clip(np.dot(ray_d, res["exit_d"]), -1, 1)))
        rows.append((nm, float(n_of(nm / 1000.0)), dev, screen_y,
                     "".join(e[0] for e in res["events"]), res["maxed"]))

    pts_all = np.vstack(all_xy)
    ax.set_aspect("equal")
    ax.set_xlim(pts_all[:, 0].min() - 0.5, pts_all[:, 0].max() + 0.5)
    ax.set_ylim(pts_all[:, 1].min() - 0.5, pts_all[:, 1].max() + 0.5)
    ax.set_title(f"Newton's prism ({MATERIAL}) — dispersion of a white ray")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    out = pathlib.Path(__file__).resolve().parent.parent / "runs"
    out.mkdir(exist_ok=True)
    png = out / "prism_rainbow.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    print(f"wrote {png}")

    # --- per-wavelength table + physical sanity checks ---------------------
    print(f"\n{'λ(nm)':>7} {'n':>8} {'deviation°':>11} {'screen_y':>9}  events")
    for nm, n, dev, sy, ev, maxed in rows:
        flag = "  TRAPPED" if maxed else ""
        sy_s = f"{sy:9.4f}" if np.isfinite(sy) else "      n/a"
        print(f"{nm:7.1f} {n:8.5f} {dev:11.4f} {sy_s}  {ev}{flag}")

    all_exited = all(not r[5] for r in rows)
    n_mono = rows[0][1] > rows[-1][1]                  # n(violet) > n(red)
    dev_mono = rows[0][2] > rows[-1][2]                # violet deviates more
    print(f"\nall wavelengths exited cleanly : {all_exited}")
    print(f"n decreases with wavelength    : {n_mono}")
    print(f"violet deviates more than red  : {dev_mono}")
    ok = all_exited and n_mono and dev_mono
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
