# Diamond Fire — Experiment Log

Curated lab notebook for the diamond-fire renderer (custom NumPy ray tracer with
ray/beam differentials). The machine-readable record of each run lives in
`runs/index.jsonl` (written by `diffrt.diamond.explog.log_run`); this file is the
human narrative: what each experiment *meant*, and — critically — **which results
are physically real vs. artifacts.**

## How logging works
- New/edited scripts write outputs into a timestamped, non-clobbering dir via
  `explog.run_dir(name)` → `runs/<YYYYMMDD-HHMMSS>_<name>/`, and call
  `explog.log_run(name, params, metrics, outputs, notes)` at the end.
- Legacy runs (before this convention) wrote fixed filenames directly in `runs/`;
  those paths are cited below as-is.

## Status legend
- **KEEPER** — physically correct, trust it.
- **ARTIFACT** — non-physical; kept only as a cautionary record. Do **not** treat as fire.
- **DIAGNOSTIC** — analysis/measurement, not a render.
- **VALIDATION** — confirms the engine against a known answer.
- **BASELINE / SUPERSEDED** — reference or replaced by later work.

---

## Standard scene (unless noted)
- **Gem**: `round_brilliant(crown_angle_deg=34, pavilion_angle_deg=41, table_frac=0.55)`
  (triangulated convex hull; every facet is a triangle).
- **Camera**: eye `(0, 2.05, 3.85)`, lookat `(0, -0.10, 0)`, up `(0, 1, 0)`, fov `30°`.
- **Light**: point at `(0.9, 3.0, 0.6)`.
- **Spectrum**: single-wavelength rays, band `0.40–0.70 µm`; diamond Sellmeier `n(λ)`
  with analytic `dn/dλ`.
- **Emission sampling**: cone aimed at gem, half-angle `arctan(gem_radius/‖L‖)`, `gem_radius=1.3`.

---

## Decisions & Assumptions (load-bearing)

1. **Point light + pinhole + pure specular ⇒ measure-zero connections.** For a fixed
   facet chain the connecting `(e1,e2,wl)` is generically a curve; for this scene it is
   **empty**. The correct point-light render is therefore **black**. *(Stated at the
   outset, then temporarily contradicted, then reconfirmed — see #3.)*

2. **Fixed facet sequence via INFINITE PLANES — the artifact source.** To get a smooth,
   differentiable connection problem we prescribed the facet chain and traced against
   each facet's *infinite plane* (no polygon-bound check) in
   `fixedseq.trace_fixed_sequence_vec`. **Consequence (was not flagged at the time):**
   the solver can move the emission direction off the real gem and "connect" through
   plane extensions out in empty space. Such solutions satisfy `f=0` but describe
   rays that never touch the diamond. **Any physical result must use bounded-facet
   tracing (`render_photon._nearest_hit_vec`), not the fixed-sequence planes.**

3. **The ~0.05 distance floor is REAL (geometric).** On the bounded diamond, point-light
   exit rays miss the pinhole by ≥ ~0.05 (world units). An earlier claim that "the floor
   is not geometric, the solver reaches resid 1e-15" was **WRONG** — that 1e-15 was an
   off-gem plane artifact (#2). `runs/fixed_seq_solver.png` shows the median residual
   flatlining at ~0.117 on the real gem.

4. **Physically-correct fire = finite aperture + valid bounded paths + beam-differential
   footprint splat.** (Chosen approach, 2026-05-31.) No connection solving.

5. **Finite-aperture model.** A photon contributes if its exit ray punctures a disk of
   radius `aperture` in the lens plane (⊥ camera axis), i.e. `‖Q−C‖ < aperture` where
   `Q = ray ∩ lens-plane`. Film position is `project(P)` — i.e. **lens focused on the
   gem, no depth-of-field blur** (a simplification). The earlier `accept_eps`
   (perpendicular distance from `C` to the ray) is a near-equivalent *isotropic* proxy;
   the two differ by a `1/cos γ` obliquity factor and agree for near-axial rays.

6. **Beam differentials are used ONLY for the splat footprint.** `Σ = JJᵀ`,
   `J = (∂U/∂P)·∂P/∂(e1,e2)·Δ`; each accepted photon is a unit-flux Gaussian of that
   covariance, colored by `wl`. They are **not** used to connect/solve. The wavelength
   column `∂P/∂wl` and the direction Jacobian `Dj` are computed but **unused** in the
   render (dispersion instead comes from sampling many `wl`).

7. **Terminology.** Wavelength is `wl` / `wl_um` (µm); avoid overloading `λ`.

8. **Engine is validated.** Tracer + Sellmeier dispersion reproduce Newton's prism
   (`runs/prism_rainbow.png`). The error was always in the *connection framing*, never
   the optics.

## Open questions
- **Densification** of the (correct but sparse) finite-aperture fire: importance
  sampling toward near-caustic emission directions (reuse pass-1 best-N) and/or a
  finite area light. *Decision parked with user (2026-05-31).*
- Uniform cone sampling **saturates** (~5M samples); coverage gain 5M→12M is ~3%.

---

## Experiment log (reverse chronological)

### 2026-05-31 — Differential-footprint fire render  ·  **KEEPER**
- **Script**: `scripts/diamond_fire_differential.py` → `render_photon.render_fire_differential`.
- **Run**: 12M beams, 48 batches, `aperture=0.18`, 320². Valid bounded-facet paths +
  finite aperture + beam-differential footprint splat + HDR (Reinhard) tonemap.
- **Result**: `runs/diamond_fire_differential.png` (+ raw `.npy`). **4,038 accepted
  (0.034%)**; aperture-crossing dist min 0.056 / median 0.146 / max 0.18.
  Three sharp colored glints (orange streak, green, violet) — genuine dispersion.
- **Conclusion**: first physically-honest fire. Correct but **sparse** (point light →
  few sparkles; low accept rate). Densification pending.

### 2026-05-31 — Connection validity audit  ·  **DIAGNOSTIC (critical)**
- **Script**: `scripts/connection_validity.py`.
- **Method**: re-trace each converged 2nd-pass connection's emission direction on the
  *real bounded* gem; valid iff it follows the exact chain and exits.
- **Result**: `runs/connection_validity.png` — **100% geometrically invalid, 0 valid**
  across all four solvers (~271k connections). Converged rays have `nint=0` (miss the
  gem entirely). Control: original sampled rays validate at 100%.
  `runs/fire_gs_with_invalid.png` = the spurious glints; `runs/fire_gs_without_invalid.png`
  = **black** (the truth).
- **Conclusion**: exposed Decision #2. The entire 2nd-pass "fire" was artifact.

### 2026-05-31 — Reflection-count histogram per solver  ·  **ARTIFACT (counts over artifacts)**
- **Script**: `scripts/reflection_hist.py`. `runs/reflection_hist.png`.
- 2-TIR chains dominate (94–99%), 3-TIR are 1–6%. Informative about chain structure,
  but the underlying connections are artifacts (see validity audit).

### 2026-05-31 — 2nd-pass solver comparison  ·  **ARTIFACT**
- **Script**: `scripts/compare_solvers.py` → `diamond/solvers2.py`.
- Four methods: `gn_wl` (current), `gn_freq` (1/λ), `gauss_seidel` (block-coordinate),
  `hessian` (FD-of-Jacobian, Levenberg). `runs/solver_{gn_wl,gn_freq,gauss_seidel,hessian}.png`.
- Film-valid connections: gn_wl 10,342 / gn_freq 14,036 / gauss_seidel 22,189 / hessian 21,051.
  Gauss-Seidel "won" — but all outputs are artifacts. **Numbers measure how efficiently
  each solver manufactures non-physical connections, not fire.**

### 2026-05-30 — Pass-2 connection density (3-DOF)  ·  **ARTIFACT (structure informative)**
- `runs/density_pass2_3dof.png`: ~5 rainbow streaks. Shows where glints would land and
  how dispersion fans — useful intuition — but the connections are fixed-seq artifacts.

### 2026-05-30 — Fire-location / closest-approach map  ·  **DIAGNOSTIC (keeper)**
- `runs/fire_location.png`: min distance-to-camera over film and over emission
  directions (log). Almost the whole gem is far; only small green/cyan patches reach
  ~0.05–0.1. Establishes the **geometric floor** and that fire is possible only in a few
  small regions.

### 2026-05-30 — Fixed-sequence solver (2-DOF, 3-DOF) renders  ·  **ARTIFACT**
- `runs/diamond_fire_fixed.png`, `runs/diamond_fire_fixed_3dof.png`: fire from the
  fixed-sequence solver (3-DOF adds column-equilibrated weighting). All off-gem artifacts.
- `runs/fixed_seq_solver.png` (**DIAGNOSTIC**): convergence — median residual drops then
  **flatlines at ~0.117**; only ~10–20% "accept" per iteration (the off-gem ones).
- `runs/solver_debug.png` (**DIAGNOSTIC**): per-step angular & wavelength steps, line-search
  α, raw vs applied Newton steps for sample 2/3/4-reflection chains.

### 2026-05-30 — Staged connect pipeline & accept_eps sweep  ·  **SUPERSEDED**
- `runs/diamond_fire_connect.png`, `..._replay.png`; `runs/fire_eps05.png`, `..._eps20.png`.
- `accept_eps` perpendicular-distance acceptance (precursor to the finite aperture).
  `runs/connect_diagnostics.png` (**DIAGNOSTIC**): bounce histogram, seed-vs-final distance,
  distance distribution (floor ~0.05). Superseded by the differential render.

### 2026-05-30 — MC-1 manifold connect from camera grid  ·  **SUPERSEDED**
- `runs/diamond_fire_manifold.png`: Newton manifold projection seeded from a
  pixel×wavelength grid → **0 connections** (seeds sit in the wrong facet topology;
  iteration stalls at the discontinuity). Motivated the light-side sampling and the
  (flawed) fixed-sequence model.

### 2026-05-30 — Early photon / differential splat attempts  ·  **SUPERSEDED**
- `runs/diamond_fire_photon.png`, `runs/diamond_fire_diff.png`: first photon-splat and
  differential experiments; replaced by `render_fire_differential`.

### 2026-05-29 — Point-light & area-light baselines  ·  **BASELINE**
- `runs/diamond_point_light.png`: point light, pure specular → ~black (Decision #1).
- `runs/diamond_fire.png`: area-light shaded diamond (realistic gray gem + highlight) —
  reference for extended-source shading.

### 2026-05-29 — Newton's prism dispersion  ·  **VALIDATION (M1)**
- `runs/prism_rainbow.png`: white ray dispersed into a clean spectrum through a BK7
  prism. Confirms tracer + Sellmeier `n(λ)` + `dn/dλ` are correct.

### 2026-05-20 — Cornell box  ·  **VALIDATION (M0, very early)**
- `runs/cornell.png`: sanity check of the base path tracer.
