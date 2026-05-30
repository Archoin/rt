"""Light-side sample-and-connect for diamond fire — staged & replayable.

Three decoupled stages, each cacheable to disk so it can be re-run without
repeating the expensive ones:

  1. sample_pass  : emit rays from the light (central-only tracer → memory-light),
                    measure each exit ray's distance to the camera pinhole.
                    -> save_pass1 / load_pass1
  2. connect_pass : Newton-refine the top-K closest seeds (origin=light,
                    target=camera); pluggable ``solver``. -> save_pass2 / load_pass2
  3. render       : render_from_pass2 thresholds by ``accept_eps`` and splats —
                    so a new image at a different accept_eps needs no recompute.

``wl`` denotes wavelength (µm) throughout.
"""

from __future__ import annotations

import json

import numpy as np

from diffrt.diamond.geometry import round_brilliant
from diffrt.diamond.manifold import connect_newton_vec
from diffrt.diamond.render import wavelength_to_rgb
from diffrt.diamond.render_photon import (_camera_projector, _sample_cone,
                                          trace_rays_vec, trace_beams_vec)


def ray_point_distance(P, D, C):
    """Perpendicular distance from point ``C`` to rays ``(P, D)`` (D unit).
    ``inf`` where ``C`` is behind the ray (front check)."""
    v = np.asarray(C, float)[None, :] - P
    along = np.sum(v * D, axis=1)
    perp = v - along[:, None] * D
    return np.where(along > 0.0, np.linalg.norm(perp, axis=1), np.inf)


def build_gem(meta):
    """Rebuild the gem from stored metadata."""
    return round_brilliant(crown_angle_deg=meta["crown_angle_deg"],
                           pavilion_angle_deg=meta["pavilion_angle_deg"],
                           table_frac=meta["table_frac"])


# --- stage 1: sample -------------------------------------------------------

def sample_pass(gem, light_pos, camera_C, n_of, n_samples, band,
                gem_radius=1.3, max_bounces=18, seed=0):
    """Emit rays from the light (central-only), return per-ray exit + distance.

    Returns dict: dirs, wl, P, D, dist, exited, nint, n_refr, n_refl."""
    rng = np.random.default_rng(seed)
    L = np.asarray(light_pos, float)
    axis = -L                                            # aim at the gem (~origin)
    cone_half = np.arctan(gem_radius / np.linalg.norm(L))

    dirs = _sample_cone(axis, cone_half, n_samples, rng)
    wl = rng.uniform(band[0], band[1], n_samples)
    O = np.broadcast_to(L, dirs.shape).copy()
    res = trace_rays_vec(gem, O, dirs, n_of(wl), max_bounces=max_bounces)
    dist = np.where(res["exited"], ray_point_distance(res["P"], res["D"], camera_C),
                    np.inf)
    return {"dirs": dirs, "wl": wl, "P": res["P"], "D": res["D"], "dist": dist,
            "exited": res["exited"], "nint": res["nint"],
            "n_refr": res["n_refr"], "n_refl": res["n_refl"]}


def _keep_best_per_bin(keep, cand, per_bin):
    """Merge candidate samples into the reservoir, retaining the ``per_bin``
    smallest-distance samples in each film bin. Vectorized grouped top-M."""
    comb = cand if keep is None else {k: np.concatenate([keep[k], cand[k]])
                                      for k in cand}
    order = np.lexsort((comb["dist"], comb["bin"]))   # by bin, then dist asc
    sb = comb["bin"][order]
    change = np.ones(sb.shape[0], dtype=bool)
    change[1:] = sb[1:] != sb[:-1]
    first = np.maximum.accumulate(np.where(change, np.arange(sb.shape[0]), 0))
    rank = np.arange(sb.shape[0]) - first              # within-bin rank (0-based)
    sel = order[rank < per_bin]
    return {k: comb[k][sel] for k in comb}


def sample_pass_streaming(gem, light_pos, camera, n_of, total_samples,
                          batch_size=1_000_000, band=(0.40, 0.70), gem_radius=1.3,
                          max_bounces=18, film_bins=(96, 96), per_bin=64, seed=0):
    """Stream ``total_samples`` light rays in batches, keeping only the
    ``per_bin`` closest-to-camera samples per coarse FILM bin.

    Memory is bounded (~ film_bins * per_bin kept samples + one batch), so
    ``total_samples`` is compute-bound, not memory-bound. Stratifying by film bin
    keeps every glint region represented as sampling grows (each glint images to
    its own bins), instead of a global top-N collapsing onto the nearest glint.

    NOTE: film-bin stratification is a cheap diversity proxy. A more precise
    alternative is to stratify by FACET-CHAIN — record each ray's chain with
    ``fixedseq.trace_record_sequence`` and keep best-M per chain (exact
    per-channel coverage, but it records the chain per ray, so it's heavier).
    To switch, replace the ``bin`` key below with a per-ray chain hash.

    Returns the same dict format as ``sample_pass`` (for the kept samples).
    """
    from diffrt.diamond.render_photon import trace_rays_vec

    rng = np.random.default_rng(seed)
    L = np.asarray(light_pos, float)
    C = np.asarray(camera[0], float)
    bnx, bny = film_bins
    _, project = _camera_projector(camera, bnx, bny)
    axis = -L
    cone_half = np.arctan(gem_radius / np.linalg.norm(L))

    keep = None
    done = 0
    while done < total_samples:
        B = min(batch_size, total_samples - done)
        done += B
        dirs = _sample_cone(axis, cone_half, B, rng)
        wl = rng.uniform(band[0], band[1], B)
        O = np.broadcast_to(L, dirs.shape).copy()
        res = trace_rays_vec(gem, O, dirs, n_of(wl), max_bounces=max_bounces)
        P, D = res["P"], res["D"]
        dist = ray_point_distance(P, D, C)
        col, row, valid = project(P)
        good = res["exited"] & np.isfinite(dist) & valid
        if not good.any():
            continue
        ci = np.clip(np.round(col[good]).astype(int), 0, bnx - 1)
        ri = np.clip(np.round(row[good]).astype(int), 0, bny - 1)
        cand = {"dirs": dirs[good], "wl": wl[good], "P": P[good], "D": D[good],
                "dist": dist[good], "nint": res["nint"][good],
                "n_refr": res["n_refr"][good], "n_refl": res["n_refl"][good],
                "bin": ri * bnx + ci}
        keep = _keep_best_per_bin(keep, cand, per_bin)

    keep.pop("bin")
    keep["exited"] = np.ones(keep["dist"].shape[0], dtype=bool)
    return keep


def save_pass1(path, samples, meta):
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **samples)


def load_pass1(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    samples = {k: d[k] for k in d.files if k != "meta"}
    return samples, meta


# --- stage 2: connect ------------------------------------------------------

def connect_pass(gem, light_pos, camera_C, samples, n_of, dn_of, top_k=100_000,
                 newton_iters=50, max_bounces=18, band=(0.40, 0.70),
                 seed_resid=10.0, solver=connect_newton_vec):
    """Newton-refine the top-K closest seeds. ``solver`` is pluggable: any
    callable ``(gem, origin, dirs, wl, n_of, dn_of, target, ...) -> {P,D,wl}``.

    Returns dict: seed_idx (into samples), P, D, wl, dist (final, to the camera)."""
    L = np.asarray(light_pos, float)
    order = np.argsort(samples["dist"])
    top = order[:top_k]
    top = top[np.isfinite(samples["dist"][top])]

    res = solver(gem, L, samples["dirs"][top], samples["wl"][top], n_of, dn_of,
                 camera_C, max_iter=newton_iters, tol=1e-7,
                 max_bounces=max_bounces, band=band, seed_resid=seed_resid)
    final_dist = ray_point_distance(res["P"], res["D"], camera_C)
    return {"seed_idx": top, "P": res["P"], "D": res["D"], "wl": res["wl"],
            "dist": final_dist, "seed_dist": samples["dist"][top]}


def save_pass2(path, pass2, meta):
    np.savez_compressed(path, meta=np.array(json.dumps(meta)), **pass2)


def load_pass2(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    pass2 = {k: d[k] for k in d.files if k != "meta"}
    return pass2, meta


# --- stage 3: render -------------------------------------------------------

def render_from_pass2(pass2, camera, width, height, accept_eps):
    """Threshold connections by ``accept_eps`` and splat → (img, n_accepted).

    Cheap: no tracing. Re-run with any accept_eps to re-render from a cached
    pass2."""
    _C, project = _camera_projector(camera, width, height)
    accept = pass2["dist"] < accept_eps
    img = np.zeros((height * width, 3))
    n = 0
    if accept.any():
        col, row, valid = project(pass2["P"][accept])
        ci = np.round(col[valid]).astype(int)
        ri = np.round(row[valid]).astype(int)
        rgb = np.array([wavelength_to_rgb(w * 1000.0)
                        for w in pass2["wl"][accept][valid]])
        np.add.at(img, ri * width + ci, rgb)
        n = int(valid.sum())
    return img.reshape(height, width, 3), n


# --- orchestrator ----------------------------------------------------------

def make_meta(light_pos, camera, band, max_bounces, crown_angle_deg,
              pavilion_angle_deg, table_frac, material, n_samples, seed,
              gem_radius, width, height):
    return {
        "light_pos": list(map(float, light_pos)),
        "camera": [list(map(float, camera[0])), list(map(float, camera[1])),
                   list(map(float, camera[2])), float(camera[3])],
        "band": [float(band[0]), float(band[1])],
        "max_bounces": int(max_bounces),
        "crown_angle_deg": float(crown_angle_deg),
        "pavilion_angle_deg": float(pavilion_angle_deg),
        "table_frac": float(table_frac),
        "material": str(material), "n_samples": int(n_samples),
        "seed": int(seed), "gem_radius": float(gem_radius),
        "width": int(width), "height": int(height),
    }


def camera_from_meta(meta):
    c = meta["camera"]
    return (tuple(c[0]), tuple(c[1]), tuple(c[2]), c[3])


__all__ = ["ray_point_distance", "build_gem", "sample_pass",
           "sample_pass_streaming", "save_pass1", "load_pass1", "connect_pass",
           "save_pass2", "load_pass2", "render_from_pass2", "make_meta",
           "camera_from_meta"]
