"""Standalone (NumPy) spectral beam/ray-differential tracer for diamond fire.

Stage 1 of the architecture in PLAN_DIAMOND.md: a clear, dependency-light tracer
for flat-faceted gems, optimized for validating the differential math before any
port to Dr.Jit/Mitsuba. No Mitsuba dependency here.
"""

from diffrt.diamond import geometry, optics, trace

__all__ = ["optics", "geometry", "trace"]
