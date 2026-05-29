"""Validate the vectorized beam tracer against the scalar M2 trace_beam.

For a batch of camera rays through the diamond, compare exit direction, the
full footprint Jacobian Dj, and the 'exited' flag between trace_beams_vec and
the scalar differentials.trace_beam (which M2 finite-difference-validated).

    PYTHONPATH=src python scripts/check_vec_tracer.py
"""

from __future__ import annotations

import numpy as np

from diffrt.diamond.differentials import trace_beam
from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.optics import sellmeier_dn_dlambda, sellmeier_n
from diffrt.diamond.render import make_camera_rays
from diffrt.diamond.render_photon import trace_beams_vec

LAM = 0.55


def main() -> int:
    gem = round_brilliant(crown_angle_deg=34.0, pavilion_angle_deg=41.0,
                          table_frac=0.55)
    n_of = lambda um: sellmeier_n(um, "diamond")
    dn_of = lambda um: sellmeier_dn_dlambda(um, "diamond")
    cam = ((0.0, 2.05, 3.85), (0.0, -0.10, 0.0), (0.0, 1.0, 0.0), 30.0)
    C = np.array(cam[0])
    _, dirs, _ = make_camera_rays(*cam, 60, 60)

    ng = float(n_of(LAM))
    dn = float(dn_of(LAM))
    O = np.broadcast_to(C, dirs.shape).copy()
    vec = trace_beams_vec(gem, O, dirs, np.full(len(dirs), ng),
                          np.full(len(dirs), dn))

    dmax = djmax = 0.0
    exit_mismatch = 0
    compared = 0
    for i in range(len(dirs)):
        s = trace_beam(gem, C, dirs[i], n_of, dn_of, LAM, max_bounces=18)
        if bool(s["exited"]) != bool(vec["exited"][i]):
            exit_mismatch += 1
            continue
        if not s["exited"]:
            continue
        compared += 1
        dmax = max(dmax, float(np.max(np.abs(s["D"] - vec["D"][i]))))
        djmax = max(djmax, float(np.max(np.abs(s["Dj"] - vec["Dj"][i]))))

    print(f"rays compared (both exited): {compared}")
    print(f"exited-flag mismatches     : {exit_mismatch}")
    print(f"max |ΔD|                    : {dmax:.3e}")
    print(f"max |ΔDj| (footprint)       : {djmax:.3e}")
    ok = exit_mismatch == 0 and dmax < 1e-9 and djmax < 1e-7
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
