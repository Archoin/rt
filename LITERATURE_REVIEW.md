# Literature Review — Spectral Caustics & "Diamond Fire" Rendering

*Scope:* prior art relevant to rendering the chromatic caustics ("fire") of a
diamond under a point light via **ray/beam differentials with a wavelength
dimension**. Compiled for the project in [PLAN_DIAMOND.md](PLAN_DIAMOND.md).
Last updated 2026-05-29.

> **Confidence note.** This is a focused review, not exhaustive. Claims about
> what a paper does/does not do are marked **[verified]** (read primary text),
> **[abstract]** (read abstract/project page), or **[secondary]** (from search
> summaries only). Verify **[secondary]** items before any written novelty claim.

---

## 1. The problem

Render the **fire** of a diamond: a **point light**, a pinhole camera, and a
**purely specular** faceted dielectric. The valid light paths (point light →
specular chain → pinhole) form a **measure-zero** set — the classic
**specular–diffuse–specular / specular-caustic connectivity problem**. Standard
unidirectional path tracing returns black (you see the light directly and the
refracted background, but not the concentrated colored caustics). Two sub-problems:

1. **Connectivity** — find the admissible specular paths at all.
2. **Dispersion** — the fire is *chromatic*: exit angle depends on wavelength
   via `n(λ)`, with angular spread `∝ ∂(dir)/∂λ ∝ dn/dλ`.

The literature splits cleanly along these two axes, and — importantly — **almost
nothing addresses them together**.

---

## 2. Ray / photon / beam differentials  *(our method's lineage)*

- **Igehy 1999, *Tracing Ray Differentials*** (SIGGRAPH). Propagates
  `∂(position, direction)/∂(image x, y)` through reflection/refraction to get a
  ray footprint; foundational for texture filtering. **[secondary]**
- **Schjøth, Frisvad, Erleben, Sporring 2007, *Photon Differentials*** (GRAPHITE).
  Applies ray-differential ideas to **photons** for anisotropic caustic flux
  estimation. Each photon carries positional `Dx = [∂x/∂u, ∂x/∂v]` and
  directional `Dω = [∂ω/∂u, ∂ω/∂v]` Jacobians, where `u,v` parameterize the
  **light emission**; positional vectors span the gather footprint. **[verified]**
- **Schjøth et al. 2010, *Photon Differentials in Space and Time***. Adds a
  **time** column `∂/∂t` to the Jacobians for animation/temporal coherence.
  **[verified]**
- **Frisvad, Schjøth, Erleben, Sporring 2014, *Photon Differential Splatting for
  Rendering Caustics*** (CGF). Splatting formulation; no map storage needed.
  **[abstract]**

> **Key finding [verified]:** photon differentials are **monochromatic** — the
> 2007/2010 papers contain *zero* mentions of wavelength, dispersion, or
> spectral. Their one demonstrated extension of the differential parameterization
> was **time**. **Adding a wavelength column `∂/∂λ` is the direct, untaken
> analogue** — and is the core novelty proposed in our plan.

---

## 3. Specular path finding for caustics & glints  *(connectivity)*

### 3a. Manifold (Newton-solver) methods
- **Hanika, Droske, Fascione 2015, *Manifold Next Event Estimation*** (EGSR/CGF).
  Deterministic manifold walk to connect a shading point to a light through
  specular interfaces; lightweight, no photon map. **[abstract]**
- **Zeltner, Georgiev, Jakob 2020, *Specular Manifold Sampling*** (SIGGRAPH).
  Stochastic Newton solver finding specular sub-paths between two endpoints;
  unifies caustics and glints; **implemented in Mitsuba, code public**. The
  natural **baseline** to compare against. **[abstract]**

### 3b. Ling-Qi Yan group — deterministic/algebraic program *(directly on-topic)*
A sustained line attacking admissible-specular-path finding:
- **Path Cuts: Efficient Rendering of Pure Specular Light Transport** (2020). **[secondary]**
- **Unbiased Caustics Rendering Guided by Representative Specular Paths**
  (SIGGRAPH Asia 2022). **[secondary]**
- **Manifold Path Guiding for Importance Sampling Specular Chains**
  (SIGGRAPH Asia 2023). **[secondary]**
- **Specular Polynomials** (SIGGRAPH 2024) — algebraic specular-path solving.
  **[secondary]**
- **Fan, Wang, Wang, Li, Guo, Yan, Guo, Guo 2025, *Bernstein Bounds for
  Caustics*** (ToG / SIGGRAPH; arXiv 2504.19163; code public). Bounds the
  irradiance of each triangle tuple via Bernstein coefficients of rational
  functions to sample high-contribution specular paths; **stochastic but
  unbiased**; uses implicit differentiation of constraints (not ray
  differentials). **[verified]**

> **Two findings that matter for diamonds [verified for Bernstein]:**
> 1. **All monochromatic** — Bernstein Bounds has *zero* spectral/dispersion
>    content; the program is geometry-only.
> 2. **Short chains only** — Bernstein Bounds is "only feasible for one or two
>    bounces"; complexity grows rapidly with chain length. The whole
>    algebraic/manifold family struggles as specular chains lengthen.
>
> Diamonds need **~10–30 internal TIR bounces**, where these methods are
> structurally weak. A **forward photon/beam trace is linear in depth** — the
> right tool for deep gem interiors (at the cost of being a biased density
> estimate rather than unbiased path-finding).

---

## 4. Spectral & dispersion rendering

- **Musgrave 1989, *Prisms and Rainbows: a dispersion model for computer
  graphics*** (Graphics Interface). Classic geometric dispersion model. **[secondary]**
- **Wilkie et al. 2014, *Hero Wavelength Spectral Sampling*** (EGSR). Efficient
  spectral MC; carries a hero wavelength + companions, decohering at dispersive
  events. The mechanism behind Mitsuba's spectral variants. **[secondary]**
- **Spectral caustics via wavelength clustering** (e.g., "Spectral caustic
  rendering … based on wavelength clustering and eye sensitivity"). Clusters
  wavelengths with similar refraction directions so one ray represents a cluster.
  **[secondary]**
- **Adaptive Spectral Mapping for Real-Time Dispersive Refraction** (WPI; Agu et
  al.). Screen-space, real-time extension of adaptive caustic mapping to
  dispersion; **discrete wavelength sampling**, an approximation. **[secondary]**

> **Finding:** spectral caustics are handled by **discrete wavelength samples or
> clusters**, never by an **analytic wavelength derivative** `∂/∂λ` carried on a
> differential/beam. This is the second half of our gap.

---

## 5. Wave optics  *(deeper physics — related / future, out of current scope)*

Yan group's physical-optics line, relevant if dispersion later needs diffraction:
- **Yan, Hašan, Walter, Marschner, Ramamoorthi 2018, *Rendering Specular
  Microgeometry with Wave Optics*** (SIGGRAPH) — wavelength-dependent specular
  highlights with color effects; code public. **[secondary]**
- **Towards Practical Physical-Optics Rendering** (SIGGRAPH 2022, Best Paper
  Hon. Mention). **[secondary]**
- **A Generalized Ray Formulation for Wave-Optical Light Transport**
  (SIGGRAPH Asia 2024). **[secondary]**

---

## 6. Gap analysis & positioning

| Method family | Long specular chains | Dispersion / spectral | Bias | Mechanism |
|---|---|---|---|---|
| Photon differentials (2007–2014) | ✅ (forward trace, linear in depth) | ❌ monochromatic | biased (density est.) | beam footprint, `∂/∂(u,v[,t])` |
| MNEE / SMS (2015/2020) | ⚠️ degrades with length | ❌ monochromatic | unbiased | Newton solve on specular manifold |
| Yan: Path Cuts → Bernstein (2020–2025) | ❌ 1–2 bounces | ❌ monochromatic | unbiased | algebraic path enumeration/bounds |
| Spectral caustics (clustering / adaptive) | varies | ✅ but discrete samples | approx. | wavelength clusters, no `∂/∂λ` |
| **This project (proposed)** | ✅ linear in depth | ✅ **analytic `∂/∂λ`** | biased | **photon differentials + wavelength column** |

**The gap (two independent confirmations):**
1. The connectivity literature — photon differentials *and* the strong Yan
   path-finding program *and* SMS/MNEE — is **monochromatic**.
2. The spectral-caustic literature handles dispersion only via **discrete
   wavelength sampling/clustering**, never an analytic wavelength differential.

**Our contribution =** extend photon differentials with a **wavelength column
`∂/∂λ`** (`∂dir/∂λ ∝ dn/dλ`), directly analogous to the 2010 *time* extension, to
render **chromatic caustics / fire** in **deep specular chains** where the
unbiased algebraic methods cannot reach. Tradeoff to own explicitly: ours is a
**biased density estimate** (like all photon mapping), not unbiased path-finding.

---

## 7. Open questions / to verify

- Read **Photon Differential Splatting 2014** in full (splat formulation; reuse
  vs. reimplement). **[abstract only so far]**
- Confirm none of **Path Cuts / Representative Specular Paths / Manifold Path
  Guiding / Specular Polynomials** sneak in any spectral handling. **[secondary]**
- Sweep **2020–2026** for differential-photon + spectral combinations and SMS
  successors (e.g. *Batch SMS* 2025) before claiming novelty in writing.
- Decide whether the wave-optics line is a future extension or out of scope.

---

## References

**Differentials**
- H. Igehy. *Tracing Ray Differentials.* SIGGRAPH 1999.
- L. Schjøth, J. R. Frisvad, K. Erleben, J. Sporring. *Photon Differentials.*
  GRAPHITE 2007. https://orbit.dtu.dk/en/publications/f5e55cbd-3c46-40d4-846a-8f89075e7f69
- L. Schjøth et al. *Photon Differentials in Space and Time.* 2010.
  https://erleben.github.io/pubs/2010/schoeth.ea10/schoeth.ea10.pdf
- J. R. Frisvad, L. Schjøth, K. Erleben, J. Sporring. *Photon Differential
  Splatting for Rendering Caustics.* Computer Graphics Forum 2014.
  https://orbit.dtu.dk/en/publications/photon-differential-splatting-for-rendering-caustics/

**Specular path finding**
- J. Hanika, M. Droske, L. Fascione. *Manifold Next Event Estimation.* CGF
  (EGSR) 2015. https://onlinelibrary.wiley.com/doi/10.1111/cgf.12681
- T. Zeltner, I. Georgiev, W. Jakob. *Specular Manifold Sampling for Rendering
  High-Frequency Caustics and Glints.* SIGGRAPH 2020.
  https://rgl.epfl.ch/publications/Zeltner2020Specular ·
  code: https://github.com/tizian/specular-manifold-sampling
- *Path Cuts: Efficient Rendering of Pure Specular Light Transport.* 2020. (Yan group)
- *Unbiased Caustics Rendering Guided by Representative Specular Paths.* SIGGRAPH Asia 2022. (Yan group)
- *Manifold Path Guiding for Importance Sampling Specular Chains.* SIGGRAPH Asia 2023. (Yan group)
- *Specular Polynomials.* SIGGRAPH 2024. (Yan group)
- Z. Fan, C. Wang, Y. Wang, B. Li, Y. Guo, L.-Q. Yan, Y. Guo, J. Guo. *Bernstein
  Bounds for Caustics.* ACM TOG (SIGGRAPH) 2025. https://arxiv.org/abs/2504.19163 ·
  code: https://github.com/mollnn/bound-caustics

**Spectral / dispersion**
- F. K. Musgrave. *Prisms and Rainbows: a dispersion model for computer
  graphics.* Graphics Interface 1989.
- A. Wilkie et al. *Hero Wavelength Spectral Sampling.* EGSR 2014.
- *Spectral caustic rendering … based on wavelength clustering and eye
  sensitivity.* (wavelength clustering)
- D. Blanchette, E. Agu et al. *Adaptive Spectral Mapping for Real-Time
  Dispersive Refraction.*
  http://web.cs.wpi.edu/~emmanuel/publications/PDFs/damon_dispersive_refraction.pdf

**Wave optics (related / future)**
- L.-Q. Yan, M. Hašan, B. Walter, S. Marschner, R. Ramamoorthi. *Rendering
  Specular Microgeometry with Wave Optics.* SIGGRAPH 2018.
  https://sites.cs.ucsb.edu/~lingqi/publications/paper_glints3.pdf
- *Towards Practical Physical-Optics Rendering.* SIGGRAPH 2022. (Yan group)
- *A Generalized Ray Formulation for Wave-Optical Light Transport.* SIGGRAPH
  Asia 2024. (Yan group)

**Author homepage:** Ling-Qi Yan, https://sites.cs.ucsb.edu/~lingqi/
