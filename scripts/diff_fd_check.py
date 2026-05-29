"""M2 — validate beam differentials against finite differences.

The go/no-go test for the whole approach: the analytic Jacobians ∂(exit dir)/∂s
and ∂(screen landing)/∂s carried by trace_beam must match central finite
differences of full prism traces, for each beam parameter s ∈ {θ, φ, λ}.

The λ column is the dispersion term (∝ dn/dλ) — the project's contribution — so
its agreement is the headline result.

Run from the repo root:

    PYTHONPATH=src python scripts/diff_fd_check.py
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.differentials import make_frame, trace_beam
from diffrt.diamond.geometry import triangular_prism
from diffrt.diamond.optics import normalize, sellmeier_dn_dlambda, sellmeier_n

MATERIAL = "BK7"
LAMBDA0 = 0.55                                  # µm (center wavelength)
RAY_D = normalize(np.array([0.9426, 0.3339, 0.0]))     # min-deviation incidence
RAY_O = np.array([-0.5, 0.289, 0.0]) - 4.0 * RAY_D
SCREEN = (np.array([8.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]))  # (Q0, N)

DELTAS = {"theta": 1e-4, "phi": 1e-4, "lambda": 1e-4}
TOL = 1e-4


def main() -> int:
    prism = triangular_prism(side=2.0, depth=2.0)
    n_of = lambda um: sellmeier_n(um, MATERIAL)
    dn_of = lambda um: sellmeier_dn_dlambda(um, MATERIAL)
    e1, e2 = make_frame(RAY_D)
    frame = (e1, e2)

    # --- analytic differentials --------------------------------------------
    ana = trace_beam(prism, RAY_O, RAY_D, n_of, dn_of, LAMBDA0,
                     frame=frame, screen=SCREEN)
    print(f"central ray events: {'-'.join(ana['events'])}")
    print(f"exit direction:     {np.array2string(ana['D'], precision=5)}")
    print(f"screen landing:     {np.array2string(ana['X'], precision=5)}")
    print(f"dn/dλ @ {LAMBDA0}µm:   {dn_of(LAMBDA0):+.6f} /µm\n")

    # primal trace for a perturbed (direction, wavelength)
    def primal(dD, dlam):
        D0 = normalize(RAY_D + dD)
        r = trace_beam(prism, RAY_O, D0, n_of, dn_of, LAMBDA0 + dlam,
                       frame=frame, screen=SCREEN)
        return r["D"], r["X"]

    # finite-difference column for each parameter
    perturb = {
        "theta": (lambda d: (d * e1, 0.0)),
        "phi": (lambda d: (d * e2, 0.0)),
        "lambda": (lambda d: (np.zeros(3), d)),
    }
    col = {"theta": 0, "phi": 1, "lambda": 2}

    print(f"{'param':>7} {'quantity':>9} {'rel.err':>11}   status")
    worst = 0.0
    for name in ("theta", "phi", "lambda"):
        d = DELTAS[name]
        dD_p, dl_p = perturb[name](+d)
        dD_m, dl_m = perturb[name](-d)
        Dp, Xp = primal(dD_p, dl_p)
        Dm, Xm = primal(dD_m, dl_m)
        fd_D = (Dp - Dm) / (2 * d)
        fd_X = (Xp - Xm) / (2 * d)
        for qty, fd, ana_col in (("dDir", fd_D, ana["Dj"][:, col[name]]),
                                 ("dLand", fd_X, ana["Xj"][:, col[name]])):
            err = np.linalg.norm(fd - ana_col) / max(np.linalg.norm(fd), 1e-12)
            worst = max(worst, err)
            print(f"{name:>7} {qty:>9} {err:11.3e}   {'ok' if err <= TOL else 'FAIL'}")

    print(f"\nworst relative error: {worst:.3e}  (tol {TOL:.0e})")
    ok = worst <= TOL
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
