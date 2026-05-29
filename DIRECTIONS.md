# Research Directions — turning the sparse caustic into fire

Where we are: a point light + pure-specular diamond + pinhole renders **black**
(measure-zero connectivity); our beam-differential connection
(`render_diff.py`) recovers **sparse colored fire glints**. Turning that sparse,
between-pixels caustic into a clean image is exactly the problem most of the
rendering literature attacks. This file records the directions we're pursuing,
focused on **photon mapping** and **Metropolis–Hastings**.

## Unifying insight

Both families need the **same ingredient we already built and validated in M2**
— the beam/specular Jacobian `∂(exit pos,dir)/∂(emission θ,φ,λ)`:

- **Photon mapping** uses it as the **splat footprint** (this is literally
  "photon differentials", our lineage — Schjøth/Frisvad).
- **Metropolis / manifold exploration** uses it as the **manifold derivative** —
  how a specular path moves when an endpoint is nudged — to propose and
  re-project mutations.

Our `render_diff.py` connection (solve for the footprint offset that reaches the
light) is **one Newton/manifold step**. We are holding the engine; these
directions wrap it correctly.

---

## Direction 1 — Progressive photon-differential splatting  *(photon mapping)*  ⟵ IN PROGRESS

Light-side: trace photon beams from the point light through the diamond
(reuse the beam-differential tracer), and **splat each exit footprint onto the
camera film**, coloured by wavelength via `dn/dλ`. Then make it **progressive
(SPPM-style)**: many passes, shrinking the gather radius each pass
(`r ← r·√(N/(N+M))`).

- **Why:** removes the *arbitrary gather radius* that is v1's main flaw — the
  image becomes **consistent**, converging to the true (sparse) fire as photons
  accumulate, instead of a fixed blur.
- **Fit:** smallest leap from committed code; photon differentials *are* our M2
  footprint.
- **Caveats:** point-light caustics are photon-starved (expect blotches before
  convergence); connecting beams to a pinhole is sparse, so we need many beams
  → requires a **vectorized** beam-differential tracer.
- **Steps:** (a) vectorize the beam-differential tracer (validate vs the scalar
  M2 code); (b) light-side splat to the film with footprint-sized kernels;
  (c) progressive radius reduction; plot convergence vs. pass count.

## Direction 2 — Manifold-exploration Metropolis with the M2 Jacobian  *(MH)*  ★ deepest fit

MCMC over eye↔point-light **specular-chain** paths (Jakob & Marschner 2012, the
technique built for this exact hard case). Mutate a path (move pixel / vertex /
**wavelength**), Newton-iterate back onto the specular manifold using the path
Jacobian, accept/reject by Metropolis–Hastings. Once a fire glint is found, the
chain explores around it → samples concentrate on the sparse fire, **unbiased**.

- **Why best fit:** our M2 differential *is* the manifold derivative; our
  linearized connection is the first Newton step; wavelength becomes a mutation
  dimension and `dn/dλ` makes the manifold chromatic → spectral fire. Same
  lineage as the Yan/SMS/MNEE work in `LITERATURE_REVIEW.md`.
- **Caveats:** most complex; needs seeding (our connection solve can seed),
  MCMC tuning, careful acceptance ratio (manifold Jacobian determinant).
- **First step toward it:** upgrade the connection from one linear step to a
  **Newton iteration** (exact manifold projection — a small loop around current
  code); this also improves Direction 1's seeding.

## Direction 3 — Primary-Sample-Space MLT, a simpler MH warm-up  *(MH)*

Kelemen-style (2002): parametrize a path (camera-ray jitter, wavelength, a small
finite-light sample, Fresnel choices) as a point in `[0,1]ⁿ`; target = pixel
contribution; MH with small + large steps concentrates samples on the glints.

- **Why:** simplest MCMC, robust, easy to bolt on — a cheap test of whether MH
  concentrates samples on the fire before investing in manifold machinery.
- **Caveat:** needs a finite light (or our footprint) so the target isn't
  measure-zero; doesn't exploit our differential.

---

## Plan of attack

1. **Direction 1 now** — lowest-risk completion of the light-first plan; gives a
   *converging* fire image and exercises the photon-differential splat
   end-to-end. Requires vectorizing the beam tracer.
2. **One step toward Direction 2** — Newton-ize the connection (quality + the
   manifold-MLT foundation).
3. **Direction 3** — cheap MCMC sanity check, optional.

---

## References

- Jensen 1996 — *Global Illumination using Photon Maps.*
- Hachisuka, Ogaki, Jensen 2008 — *Progressive Photon Mapping.*
- Hachisuka, Jensen 2009 — *Stochastic Progressive Photon Mapping* —
  http://graphics.ucsd.edu/~henrik/papers/sppm/ ·
  https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
- Jarosz et al. — *Photon Beams* / beam radiance estimate.
- Schjøth, Frisvad et al. 2007–2014 — *Photon Differentials* (our lineage).
- Veach, Guibas 1997 — *Metropolis Light Transport.*
- Kelemen et al. 2002 — *A Simple and Robust Mutation Strategy for MLT*
  (primary sample space) — https://onlinelibrary.wiley.com/doi/10.1111/1467-8659.t01-1-00703
- Jakob, Marschner 2012 — *Manifold Exploration* —
  https://rgl.epfl.ch/publications/Jakob2012Manifold
- Pantaleoni 2017 — *Charted Metropolis Light Transport.*
- Wilkie et al. 2014 — *Hero Wavelength Spectral Sampling.*
