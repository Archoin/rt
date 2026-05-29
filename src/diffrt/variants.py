"""Variant selection for Mitsuba 3, with the macOS LLVM-library workaround.

On macOS the Dr.Jit LLVM backend does not auto-discover ``libLLVM`` shipped
inside the conda environment, so it has to be pointed at the dylib via the
``DRJIT_LIBLLVM_PATH`` environment variable *before* Mitsuba is imported.  This
module finds that library automatically (when needed) and selects the variant.

Usage::

    from diffrt.variants import set_variant
    mi = set_variant()              # honours $MI_VARIANT, defaults to llvm_ad_rgb

The active variant is chosen from, in order of precedence:
    1. the ``variant`` argument, if given
    2. the ``MI_VARIANT`` environment variable
    3. ``DEFAULT_VARIANT`` ("llvm_ad_rgb")
"""

from __future__ import annotations

import glob
import os
import sys

DEFAULT_VARIANT = "llvm_ad_rgb"


def _find_libllvm() -> str | None:
    """Locate a ``libLLVM`` shared library for the Dr.Jit LLVM backend.

    Returns the first match found, or ``None`` if none is located.  Honours an
    existing ``DRJIT_LIBLLVM_PATH`` if it points at a real file.
    """
    existing = os.environ.get("DRJIT_LIBLLVM_PATH")
    if existing and os.path.isfile(existing):
        return existing

    if sys.platform == "darwin":
        libname = "libLLVM*.dylib"
    elif sys.platform.startswith("linux"):
        libname = "libLLVM*.so*"
    else:  # windows: the cuda/llvm DLL is resolved by Dr.Jit itself
        return None

    # Search the directories most likely to hold the conda-provided LLVM.
    prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    candidate_dirs = [
        os.path.join(prefix, "lib"),
        os.path.join(sys.prefix, "lib"),
        "/opt/anaconda3/lib",
        "/opt/homebrew/lib",
        "/usr/local/lib",
    ]
    for d in candidate_dirs:
        matches = sorted(glob.glob(os.path.join(d, libname)))
        # Prefer the plain "libLLVM-NN.dylib" over "libLLVM-C.*" wrappers.
        matches = [m for m in matches if "libLLVM-C" not in os.path.basename(m)]
        if matches:
            return matches[0]
    return None


def set_variant(variant: str | None = None):
    """Configure the LLVM library path (if needed) and set the Mitsuba variant.

    Returns the imported ``mitsuba`` module so callers can do::

        mi = set_variant()
    """
    chosen = variant or os.environ.get("MI_VARIANT", DEFAULT_VARIANT)

    # The LLVM dylib must be discoverable before `import mitsuba` triggers the
    # backend init.  Only relevant for llvm_* variants.
    if chosen.startswith("llvm") and "DRJIT_LIBLLVM_PATH" not in os.environ:
        lib = _find_libllvm()
        if lib is not None:
            os.environ["DRJIT_LIBLLVM_PATH"] = lib

    import mitsuba as mi  # imported lazily, after the env var is set

    mi.set_variant(chosen)
    return mi


__all__ = ["set_variant", "DEFAULT_VARIANT"]
