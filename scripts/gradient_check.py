"""Phase 4 milestone 2 — finite-difference vs. autodiff gradient sanity check.

Confirms that the gradients Dr.Jit/Mitsuba backpropagate through a render match
a central finite-difference estimate.  This is the green light before any
optimization work: it validates *correctness*, not just that autodiff runs.

The test parameter is the Cornell-box emitter radiance (a scalar applied to all
three channels).  Loss is the mean pixel value of the render.

Memory note: in autodiff mode Mitsuba records a backprop graph over the entire
sample wavefront, so peak memory scales with ``res * res * spp``.  The defaults
below (64x64, 32 spp) are deliberately small — this peaks around 200 MB; the
large finite-difference step keeps the FD signal above MC noise so accuracy
does not depend on a big sample count.  Scale up via --res / --spp on a roomier
machine (128x128 @ 64 spp peaks near 1 GB).

Run from the repo root:

    PYTHONPATH=src python scripts/gradient_check.py

(The LLVM library path is configured automatically by diffrt.variants.)
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res", type=int, default=64,
                    help="film resolution (square); peak memory ~ res^2 * spp")
    ap.add_argument("--spp", type=int, default=32,
                    help="samples per pixel (higher = lower MC noise, more mem)")
    ap.add_argument("--eps", type=float, default=1.0,
                    help="finite-difference step (radiance ~18, so O(1) "
                         "keeps the FD signal well above MC noise)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max acceptable relative error to PASS")
    args = ap.parse_args()

    from diffrt.variants import set_variant
    mi = set_variant()
    import drjit as dr

    print(f"variant: {mi.variant()} | res={args.res} spp={args.spp} "
          f"eps={args.eps} (~{args.res * args.res * args.spp / 1e6:.1f}M samples)")

    key = "light.emitter.radiance.value"

    def scene_dict():
        d = mi.cornell_box()
        d["sensor"]["film"]["width"] = args.res
        d["sensor"]["film"]["height"] = args.res
        return d

    def render_mean(delta: float = 0.0, grad: bool = False):
        """Render the Cornell box with the emitter radiance shifted by ``delta``.

        ``delta`` is added to every channel of the *true* base radiance, so both
        the autodiff and finite-difference probes move along the same (1,1,1)
        direction from the same point.  Returns (mean_loss_tensor, params).
        """
        scene = mi.load_dict(scene_dict())
        params = mi.traverse(scene)
        base = mi.Color3f(params[key])  # deterministic for a fixed scene

        params[key] = base + mi.Color3f(delta)
        params.update()

        if grad:
            dr.enable_grad(params[key])

        img = mi.render(scene, params, spp=args.spp, seed=args.seed)
        loss = dr.mean(img, axis=None)
        return loss, params

    # --- autodiff directional derivative along (1,1,1) ----------------------
    loss, params = render_mean(delta=0.0, grad=True)
    dr.backward(loss)
    # Directional derivative along (1,1,1) = sum of the per-channel gradients,
    # which is exactly what the +/- delta finite difference below measures.
    # params[key] is a Color3f; .x/.y/.z are its three (width-1) components.
    g = dr.grad(params[key])
    ad_grad = float(g.x[0] + g.y[0] + g.z[0])
    print(f"autodiff d(loss)/d(delta):    {ad_grad:.8f}")

    # Release the AD graph + sample buffers before the FD renders so the two
    # phases don't stack their peak allocations.
    del loss, params, g
    dr.flush_malloc_cache()

    # --- central finite difference along the same direction -----------------
    lo, _ = render_mean(delta=-args.eps)
    lo_val = float(lo.array[0])
    del lo
    dr.flush_malloc_cache()

    hi, _ = render_mean(delta=+args.eps)
    hi_val = float(hi.array[0])
    del hi
    dr.flush_malloc_cache()

    fd_grad = (hi_val - lo_val) / (2 * args.eps)
    print(f"finite-diff d(loss)/d(delta):  {fd_grad:.8f}")

    # --- compare ------------------------------------------------------------
    denom = max(abs(fd_grad), 1e-12)
    rel_err = abs(ad_grad - fd_grad) / denom
    print(f"relative error: {rel_err:.4%}")

    ok = rel_err <= args.tol
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
