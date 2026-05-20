# Differentiable Ray Tracing — Setup & Research Plan

## Goal

Stand up a working differentiable ray tracer to support cutting-edge research in
physics-based differentiable rendering. Primary development on macOS (laptop),
with the Windows + RTX machine reserved for scaled experiments.

---

## TL;DR

- **Stack:** Mitsuba 3 + Dr.Jit, installed via `pip`. Same Python API on Mac and
  Windows; only the active *variant* differs.
- **Mac (dev):** `llvm_ad_rgb` variant — CPU-only, no CUDA, no Metal. Good for
  iteration, debugging, small scenes.
- **Windows (RTX):** `cuda_ad_rgb` variant — OptiX/CUDA backend, 10–50× faster.
  Used for full-resolution runs and paper benchmarks.
- **Write platform-agnostic Python**, gate the variant on an environment
  variable so the same notebook runs on both machines.

---

## Why Mitsuba 3 (vs. redner / nvdiffrast / hand-rolled)

| Option | Verdict |
|---|---|
| **Mitsuba 3 + Dr.Jit** | ✅ Actively maintained by EPFL RGL. Implements PRB, reparameterization, projective sampling, SDF rendering. Prebuilt PyPI wheels on macOS, Windows, Linux. |
| **redner** | ❌ Unmaintained since ~2021, predates path replay backpropagation. Fine for teaching, not SOTA. |
| **nvdiffrast** | ❌ Rasterization, not ray tracing. CUDA-only — won't run on Mac. |
| **PSDR-JIT / PSDR-CUDA (UCI)** | 🔶 Add later as a "phase 2" — built on top of Dr.Jit; implements the path-space differential formulation with proper boundary handling. |
| **Hand-rolled CUDA/Metal** | ❌ Multi-month engineering tax with no research payoff. Build on Mitsuba; replace pieces only when they block a research idea. |

---

## Phase 0 — Decide the variant matrix

| Machine | Variant(s) to enable | Use case |
|---|---|---|
| Mac (Apple Silicon) | `llvm_ad_rgb`, `llvm_ad_spectral` | Develop, debug, prototype, small scenes |
| Windows (RTX) | `cuda_ad_rgb`, `cuda_ad_spectral`, plus `llvm_ad_rgb` as fallback | Full-scale experiments, paper benchmarks |

Convention in code:

```python
import os, mitsuba as mi
mi.set_variant(os.environ.get("MI_VARIANT", "llvm_ad_rgb"))
```

Set `MI_VARIANT=cuda_ad_rgb` on the Windows box; leave unset on the Mac.

---

## Phase 1 — Mac setup (do this first)

Target: laptop development, fast iteration, small scenes (≤ 512², ≤ 32 spp).

1. **Prereqs**
   - macOS 12+ (Apple Silicon or Intel)
   - Xcode Command Line Tools: `xcode-select --install`
   - Python 3.10–3.12 via `pyenv` or Miniforge (do NOT use system Python).
     On Apple Silicon, use a native `arm64` Miniforge — mixing arm64 and x86_64
     binaries breaks `dlopen`.

2. **Create an isolated env**

   ```bash
   conda create -n diffrt python=3.11
   conda activate diffrt
   pip install --upgrade pip
   ```

3. **Install Mitsuba 3**

   ```bash
   pip install mitsuba drjit
   ```

   PyPI ships an `aarch64` wheel for Apple Silicon and an `x86_64` wheel for
   Intel — no compilation needed.

4. **Smoke test 1 — variant loads**

   ```python
   import mitsuba as mi
   mi.set_variant("llvm_ad_rgb")
   print(mi.variants())  # should include 'llvm_ad_rgb'
   scene = mi.load_dict(mi.cornell_box())
   img = mi.render(scene, spp=16)
   mi.util.write_bitmap("cornell.exr", img)
   ```

5. **Smoke test 2 — gradients flow**

   Run the official ["Gradient-based optimization"][grad-opt] tutorial as a
   notebook. Expect a few minutes per optimization step at 256², 4–8 spp.

6. **(Optional) Build from source** — only if you need to modify C++
   integrators:

   ```bash
   git clone --recursive https://github.com/mitsuba-renderer/mitsuba3
   cd mitsuba3 && mkdir build && cd build
   cmake -GNinja ..
   ninja
   # Edit mitsuba.conf to add llvm_ad_rgb / llvm_ad_spectral
   # source setpath.sh
   ```

7. **Mac gotchas**
   - **No CUDA, no OptiX, no Metal backend.** Dr.Jit only has `llvm` and
     `cuda` backends. Don't waste time looking for an Apple GPU path.
   - Rosetta vs. native arm64: pick one Python, never mix.
   - The LLVM backend uses NEON on Apple Silicon; it's faster than you'd
     expect for small scenes, but still 10–50× slower than CUDA.

[grad-opt]: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/gradient_based_opt.html

---

## Phase 2 — Windows setup (for scaled experiments)

Target: full-resolution inverse-rendering runs, training loops, benchmarks.

1. **Hardware/driver check**
   - NVIDIA RTX 20-series or newer (Turing+). OptiX 7 needs RT cores.
   - NVIDIA driver **≥ 535**. Update via the NVIDIA app.
   - You do NOT need to install the CUDA Toolkit separately for the pip path —
     Dr.Jit loads CUDA libs dynamically from the driver. (Toolkit only needed
     when building Mitsuba from source.)

2. **Python env (Miniconda in "Anaconda Prompt")**

   ```powershell
   conda create -n diffrt python=3.11
   conda activate diffrt
   pip install --upgrade pip
   pip install mitsuba drjit
   ```

3. **Smoke test — CUDA variant loads**

   ```python
   import mitsuba as mi
   mi.set_variant("cuda_ad_rgb")
   print(mi.variants())  # must list cuda_ad_rgb
   ```

   If `cuda_ad_rgb` isn't listed, the driver is too old or the GPU isn't
   RTX-capable.

4. **Smoke test — PRB on GPU**

   Run the [forward/inverse rendering tutorial][fwdinv] with the `prb`
   integrator. Expect 1–2 orders of magnitude faster than the Mac.

5. **(Optional) Build from source on Windows**
   - Install Visual Studio 2022 Build Tools, CMake ≥ 3.9, CUDA Toolkit 12.x,
     Python 3.11.
   - Use the *x64 Native Tools Command Prompt for VS 2022*:

     ```
     git clone --recursive https://github.com/mitsuba-renderer/mitsuba3
     cd mitsuba3
     cmake -G "Visual Studio 17 2022" -A x64 -B build
     cmake --build build --config Release
     ```
   - Edit `mitsuba.conf` to enable `cuda_ad_rgb`, `prb`, `prb_volpath`, etc.

6. **Windows gotchas**
   - Long-path support: clone to `C:\src\` to dodge the 260-char `MAX_PATH`
     failures during recursive submodule clones.
   - Whitelist `%LOCALAPPDATA%\NVIDIA\OptixCache` in antivirus.
   - WSL2 works but adds friction. Native Windows is simpler for Mitsuba.

[fwdinv]: https://mitsuba.readthedocs.io/en/stable/src/inverse_rendering/forward_inverse_rendering.html

---

## Phase 3 — Repo layout

Initialize a clean Python project structure inside this repo:

```
rt/
├── PLAN.md                  # this file
├── README.md                # quick-start for collaborators
├── pyproject.toml           # dependency pinning
├── environment.yml          # conda env (cross-platform)
├── .gitignore               # __pycache__, *.exr, runs/, data/
├── src/
│   └── diffrt/
│       ├── __init__.py
│       ├── variants.py      # MI_VARIANT auto-select
│       ├── integrators/     # custom Python integrators
│       ├── scenes/          # scene factories
│       └── losses/          # image/feature losses
├── notebooks/
│   ├── 01_smoke_test.ipynb
│   ├── 02_gradient_check.ipynb
│   └── 03_cornell_inverse.ipynb
├── scripts/
│   └── reproduce_*.py       # one script per experiment
├── data/                    # reference images, meshes (gitignored, DVC later)
└── runs/                    # outputs, gitignored
```

`pyproject.toml` minimum:

```toml
[project]
name = "diffrt"
version = "0.0.1"
requires-python = ">=3.10"
dependencies = [
  "mitsuba>=3.5",
  "drjit>=1.0",
  "numpy",
  "matplotlib",
  "imageio",
  "tqdm",
]
```

---

## Phase 4 — First research milestones

In order, each milestone should be a single notebook or script that produces a
reproducible result.

1. **Smoke test parity** — Cornell box renders identically on Mac (`llvm_ad_rgb`)
   and Windows (`cuda_ad_rgb`) at low spp. Confirms env is healthy.
2. **Gradient sanity check** — finite-difference vs. autodiff for a single
   parameter (e.g. light intensity, BSDF albedo). Variance reasonable.
3. **Texture optimization** — recover an albedo texture from a target render.
   Standard Mitsuba tutorial. Confirms the optimization loop is wired up.
4. **Geometry optimization with PRB-reparam** — recover a mesh from multi-view
   targets. First taste of discontinuity gradients.
5. **Reproduce one recent SOTA paper** — pick from the table in
   `RELATED_WORK.md` (e.g. Bonnet et al. 2024 SDF rendering, or projective
   sampling 2023). Reproducing forces deep understanding and surfaces
   integration pain points.
6. **Identify the research gap** — by this point we'll know which subsystem
   (sampling, boundary handling, neural caching, materials, …) is the
   bottleneck. Choose a thesis direction from there.

---

## Phase 5 — Optional: add PSDR-JIT (path-space differential)

Once Mitsuba is solid and we're hitting limitations on boundary/silhouette
gradients, add UCI's PSDR-JIT:

```bash
pip install psdr-jit
```

Same Dr.Jit backend; gives proper path-space differential path integrals with
warped-area sampling. Useful for any project pushing on edge-sampling quality.

---

## Risks & decision points

- **Mac-only is slow.** Wall-clock will hurt as soon as scenes exceed 512² or
  optimization runs exceed ~100 iters. Mitigation: prototype on Mac, run on
  Windows.
- **Driver version drift on Windows.** Pin the driver version once a project
  is working; CUDA/OptiX behavior can change subtly between releases.
- **C++ integrator work.** If a research idea needs a new integrator in C++
  (not Python), Mac dev becomes painful because each rebuild is a full
  recompile of Dr.Jit + Mitsuba. Plan to do C++ changes on the Windows/Linux
  box.
- **No Metal backend.** If Apple ships one for Dr.Jit later, revisit. As of
  Mitsuba 3.6 there is no Apple-GPU path.

---

## Immediate next actions

- [ ] `cd /Users/yahanzhou/Projects/rt && conda create -n diffrt python=3.11`
- [ ] `pip install mitsuba drjit`
- [ ] Run smoke test 1 (Cornell box render with `llvm_ad_rgb`)
- [ ] Run smoke test 2 (gradient-based optimization tutorial)
- [ ] Create `src/diffrt/variants.py` with the `MI_VARIANT` selector
- [ ] Mirror the env on the Windows machine; confirm `cuda_ad_rgb` loads
- [ ] Draft `RELATED_WORK.md` from the SOTA table and pick a paper to reproduce

---

## References

- Mitsuba 3 — https://github.com/mitsuba-renderer/mitsuba3
- Mitsuba 3 docs — https://mitsuba.readthedocs.io/
- Dr.Jit — https://github.com/mitsuba-renderer/drjit
- PSDR-JIT — https://github.com/uci-rendering/psdr-jit
- A Survey on Physics-based Differentiable Rendering (2025) —
  https://arxiv.org/abs/2504.01402
