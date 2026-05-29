# Diamond "Fire" via Spectral Beam / Ray-Differential Tracing

**Goal.** Render the chromatic caustics ("fire") of a diamond under a **point
light** — the effect ordinary path tracing misses — by carrying **ray/beam
differentials** (footprints and their wavelength derivative) through specular
refraction/reflection, in a **photon-mapping-style two-pass** tracer.

Builds on the Mitsuba 3 + Dr.Jit setup in [PLAN.md](PLAN.md).

> **This supersedes the autodiff / inverse-rendering direction.** See
> "Relationship to the earlier plan" below.

---

## Scope (locked this phase)

| Item | Choice | Consequence |
|---|---|---|
| Which "differential" | **Ray/beam differentials** (Igehy sense): `∂(pos,dir)/∂(beam params, λ)` | Autodiff / gradient-based inverse rendering is **dropped** |
| Light model | **Point light** | This is the hard input we explicitly want to support |
| Architecture | **Two-pass, photon-mapping-like**; start **light-first** (Pass 1), add eye gather next | |
| Spectral | **Single wavelength per beam**; dispersion via `n(λ)` | Wavelength is a differential axis (the fire) |
| Inverse loop | **None** | Pure forward rendering |
| TIR / boundary discontinuity | **Decide after forward render** | Forward phase must surface where it bites |

Out of scope: autodiff differentiability, optimization, geometry optimization,
colored/absorbing stones, roughness, textures, volumetrics, subsurface.

---

## Relationship to the earlier plan

We are pivoting from **differentiable** ray tracing (autodiff gradients for
inverse rendering) to **differential** ray tracing (Igehy ray/beam differentials
for forward caustic connectivity).

> ⚠️ Naming hazard: "differenti**able**" (autodiff) ≠ "differenti**al**" (beam
> footprints). We mean the latter from here on.

- `src/diffrt/variants.py` is still used (spectral variant selection, LLVM path).
- `scripts/gradient_check.py` and the AD-variant work are **parked, not deleted**.
- Practical win: no autodiff graph ⇒ we use the lighter **non-AD spectral**
  variant (`scalar_spectral` / `llvm_spectral`), sidestepping the AD memory
  blow-up that OOM'd the Mac.

---

## The problem — why naive tracing is black

Point light + pinhole camera + purely specular diamond ⇒ the set of valid light
paths has **measure zero** (the classic **SDS / specular-caustic** problem). You
cannot do next-event estimation across a specular (delta) vertex, and you cannot
randomly *hit* a point light. You still see the light directly and the
**refracted background**, but the **fire** — concentrated, colored caustic
flashes — is lost or appears only as extreme noise.

---

## The mechanism — precise

A point light **stays a point**; a linear map of a zero-measure set is still
zero-measure. What gains finite measure is the **integration domain we carry as
differentials**:

- Parameterize emission by **emitted direction** over a small solid-angle bin
  `(θ, φ)` **× wavelength** `λ` over the sensor band.
- Each beam carries the Jacobian columns `∂(pos,dir)/∂θ, ∂/∂φ, ∂/∂λ`.
- Propagated through the specular chain, this 3-parameter family maps to a
  **finite footprint** on receivers — that footprint **is** the caustic.

**Flat-facet subtlety (load-bearing).** A single-λ beam through *flat* facets is
an affine map — its solid angle does **not** grow on its own. The chromatic fan
that makes *fire* comes from the **λ column**: differentiating Snell
`n₁ sinθ₁ = n₂(λ) sinθ₂` gives `∂(dir)/∂λ ∝ dn/dλ`. Hence:

> **fire angular width ∝ ∂(exit direction)/∂λ ∝ dn/dλ.**

This unifies the project's two original ideas: the **wavelength derivative is the
differential that spreads the spectrum into the visible fire**. Diamond's high
`dn/dλ` ⇒ strong fire.

---

## Architecture — two-pass, light-first

**Pass 1 (light → scene)** — *implement first.*
1. Emit beams from the point light. Each carries: origin, direction, a single `λ`
   sampled across the band, intensity, and the differential columns
   `∂(pos,dir)/∂(θ, φ, λ)`.
2. Propagate through the diamond: specular refraction/reflection updates **both
   the ray and its differentials** (Igehy update; flat facets ⇒ affine; add the
   `∂/∂λ` Snell term for dispersion; per-λ TIR threshold).
3. Deposit: where a beam lands on a receiver (or is stored as a beam), record the
   **footprint** = image of the parameter cube under the accumulated Jacobian,
   with its spectral spread.

**Pass 2 (eye gather)** — *add next.*
- Gather radiance from deposited beams/footprints (beam radiance estimate /
  footprint splat). **Bootstrap shortcut:** first deposit onto a diffuse receiver
  plane and ordinary-path-trace the eye to that plane (a classic surface caustic)
  before building true eye-side cones.

**Per-beam state (the "differential information"):** `λ`, intensity, position
`P`, direction `D`, and Jacobian columns `∂P/∂θ, ∂P/∂φ, ∂P/∂λ, ∂D/∂θ, ∂D/∂φ,
∂D/∂λ`.

---

## Differential propagation rules (technical core)

- **Free flight** distance `t`: `P += tD`; `∂P += t·∂D` (spatial footprint grows
  with distance).
- **Specular reflect/refract at a flat facet**: `D` by the (linearized) law; `∂D`
  by the law's Jacobian; flat ⇒ no curvature/shape-operator term. (Curved
  surfaces would add it; diamonds are flat-faceted — the easy case.)
- **Dispersion**: refraction depends on `λ`, so even at `δθ=δφ=0` the `∂D/∂λ`
  column is nonzero `∝ dn/dλ`; carry and accumulate it.
- **TIR**: per-λ critical angle; a beam may straddle the TIR boundary across its
  footprint — this is the discontinuity we **defer**. v0: reflect/refract the
  whole beam by its **center ray** (biased), and let the forward render show how
  bad that is.

**Validation:** compare propagated differentials against **finite differences of
neighboring rays** (trace `θ+δ`, `λ+δ` explicitly, compare footprints). Reuses
the finite-difference discipline from `gradient_check.py`.

Diamond `n(λ)`: 2-term Sellmeier (λ in µm)
`n² = 1 + 4.3356·λ²/(λ²−0.1060²) + 0.3306·λ²/(λ²−0.1750²)`
(n≈2.42 @589nm, dispersion `n_F−n_C`≈0.044, critical angle ≈24.4°). Cauchy
`n≈A+B/λ²` is a cheaper smooth stand-in.

---

## Compute cuts

Rule: **a path earns a long life only while it stays specular.**
- The point/sharp light is now the **intended** input, not a problem — cheap.
- **Starve diffuse** (direct only; Russian-roulette hard on diffuse), **feed
  specular** (≈10–30 internal bounces).
- **Minimal stage**: gem + receiver plane / simple env. No clutter.
- **Drop**: textures, roughness, depth-of-field, motion blur, volumetrics,
  subsurface; colorless **non-absorbing** stone.
- **Small res / ROI.**
- **Single wavelength per beam**; importance-sample `λ` toward sensor response.
  The `∂/∂λ` column lets us **reconstruct across λ from fewer samples** — the
  efficiency payoff that funds the extra spectral cost.
- **Non-AD spectral variant** ⇒ far lower memory than the AD path.

---

## Spikes & risks (cheap, do first)

- **Spike A — forward dispersion**: glass **prism** with `n(λ)` → visible
  rainbow. Resolve built-in `dielectric` vs a small custom BSDF.
- **Spike B — differential propagation (riskiest)**: footprint through one
  refraction vs finite-difference of neighbor rays.
- **Spike C — deposit/gather**: a finite-radiance caustic on a plane vs a
  brute-force (very noisy) path-traced reference.
- **Architecture fork — Mitsuba fit**: Mitsuba does not expose ray differentials
  through its high-level integrators. We will likely implement the
  beam-differential tracer as a **custom Python integrator on top of
  `scene.ray_intersect()`**, or roll a **minimal standalone tracer** for
  flat-faceted gems. Decide early; this shapes everything.

---

## Milestones

0. **Spectral env health** — forward render in the non-AD spectral variant.
1. **Prism rainbow** (Spike A) — forward dispersion works.
2. **Differential propagation validated** (Spike B) — one refraction vs finite
   differences.
3. **Single-bounce caustic on a plane** (Spike C) — light-side beam footprints
   vs brute-force reference.
4. **Diamond, first fire** — light-first beams → deposit/gather → fire image.
5. **Full two-pass + boundary decision** — add eye-side gather; decide TIR /
   boundary handling from M2–M4 evidence; quality pass.

---

## Mac → GPU split

- **Mac**: M0–M4 at small res / ROI, single gem, non-AD spectral.
- **NVIDIA**: scale resolution, beam count, and scene size once the method is
  trusted.

---

## Immediate next actions

- [ ] **Architecture fork**: build on Mitsuba `ray_intersect` vs minimal custom
      tracer.
- [ ] M0/M1: spectral variant health + prism rainbow.
- [ ] Draft the beam-differential **state + propagation equations**; set up the
      M2 finite-difference check.
- [ ] **Literature check** (this turn): position against SMS / MNEE / photon
      beams / ray differentials / spectral caustics.

---

## Literature positioning & novelty (from the literature check)

**Closest prior art — Photon Differentials (this *is* our light-first plan).**
Schjøth, Frisvad, Erleben, Sporring, *Photon Differentials* (GRAPHITE 2007),
extended in *Photon Differentials in Space and Time* (2010) and *Photon
Differential Splatting for Rendering Caustics* (Frisvad et al., CGF 2014). A
photon carries a **positional** Jacobian `Dx = [∂x/∂u, ∂x/∂v]` and a
**directional** Jacobian `Dω = [∂ω/∂u, ∂ω/∂v]`, where `u,v` parameterize the
**light-source emission**; the positional vectors span a **footprint** used for
anisotropic flux-density estimation. This is exactly the "light-first beam with
differentials" we planned.

> **The gap = the wavelength column.** Classic photon differentials are
> **monochromatic** — verified: the papers contain *zero* mentions of
> wavelength / dispersion / spectral. The 2010 paper's contribution was adding a
> **time** column `∂/∂t`. **Our novelty is the direct analogue: add a wavelength
> column `∂/∂λ`** so the footprint fans out chromatically — *spectral / chromatic
> photon differentials* for dispersion ("fire"). `∂(dir)/∂λ ∝ dn/dλ` is the
> column.

**Same problem, different mechanism — manifold methods.** Manifold Next-Event
Estimation (Hanika et al. 2015) and Specular Manifold Sampling (Zeltner,
Georgiev, Jakob, SIGGRAPH 2020; built on Mitsuba, code available) solve the same
point-light-through-specular connectivity via a Newton solve on the specular
manifold. Monochromatic, connection-based (eye-side). **Good comparison
baselines**, not our approach.

**Spectral caustics exist, but via sampling/clustering, not a `∂/∂λ`
differential.** "Adaptive Spectral Mapping for Real-Time Dispersive Refraction"
(real-time, screen-space, discrete wavelength samples) and spectral-caustic
**wavelength-clustering** methods represent the spectrum with discrete samples or
clusters — they do **not** carry an analytic wavelength derivative. Classic
dispersion model: Musgrave, *Prisms and Rainbows* (1989).

> ⚠️ **Confidence:** focused search, not exhaustive. Before any novelty claim in
> writing, read the 2014 splatting paper in full and sweep 2020–2026 spectral-
> caustic / differential-photon follow-ups (and SMS successors like *Batch SMS*
> 2025).

## References

- Schjøth, Frisvad, Erleben, Sporring 2007 — *Photon Differentials* (GRAPHITE) —
  https://orbit.dtu.dk/en/publications/f5e55cbd-3c46-40d4-846a-8f89075e7f69
- Schjøth et al. 2010 — *Photon Differentials in Space and Time* —
  https://erleben.github.io/pubs/2010/schoeth.ea10/schoeth.ea10.pdf
- Frisvad, Schjøth, Erleben, Sporring 2014 — *Photon Differential Splatting for
  Rendering Caustics* (CGF) — https://orbit.dtu.dk/en/publications/photon-differential-splatting-for-rendering-caustics/
- Igehy 1999 — *Tracing Ray Differentials*
- Hanika, Droske, Fascione 2015 — *Manifold Next Event Estimation* —
  https://onlinelibrary.wiley.com/doi/10.1111/cgf.12681
- Zeltner, Georgiev, Jakob 2020 — *Specular Manifold Sampling* —
  https://rgl.epfl.ch/publications/Zeltner2020Specular (code:
  https://github.com/tizian/specular-manifold-sampling)
- *Adaptive Spectral Mapping for Real-Time Dispersive Refraction* —
  http://web.cs.wpi.edu/~emmanuel/publications/PDFs/damon_dispersive_refraction.pdf
- Musgrave 1989 — *Prisms and Rainbows: a dispersion model for computer graphics*
- Jarosz et al. — *Photon Beams* / beam radiance estimate (light-side beams)
- Heckbert & Hanrahan 1984 — beam tracing; Amanatides 1984 — cone tracing
