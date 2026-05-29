"""Geometric optics for the diamond tracer.

Dispersion ``n(λ)``, Snell refraction, mirror reflection, Fresnel reflectance,
and total internal reflection. Pure NumPy, no Mitsuba.

Conventions:
- Vectors are ``np.ndarray`` of shape ``(3,)``; directions are unit length.
- Wavelengths are in **micrometers (µm)** for the dispersion models.
"""

from __future__ import annotations

import numpy as np

# --- dispersion models n(λ) ------------------------------------------------

# Sellmeier:  n² = 1 + Σ Bᵢ λ² / (λ² − Cᵢ),   λ in µm.
SELLMEIER = {
    # Schott BK7 borosilicate crown glass — the classic prism glass.
    "BK7": (
        (1.03961212, 0.231792344, 1.01046945),
        (0.00600069867, 0.0200179144, 103.560653),
    ),
    # Schott SF10 dense flint — stronger dispersion (more vivid fan).
    "SF10": (
        (1.62153902, 0.256287842, 1.64447552),
        (0.0122241457, 0.0595736775, 147.468793),
    ),
    # Diamond (2-term). n≈2.42 @589nm, very strong dispersion.
    "diamond": (
        (0.3306, 4.3356),
        (0.1750 ** 2, 0.1060 ** 2),
    ),
}


def sellmeier_n(wavelength_um, material: str = "BK7"):
    """Refractive index from a Sellmeier model. Scalar or array in, same out."""
    B, C = SELLMEIER[material]
    l2 = np.asarray(wavelength_um, dtype=float) ** 2
    n2 = 1.0 + sum(Bi * l2 / (l2 - Ci) for Bi, Ci in zip(B, C))
    return np.sqrt(n2)


def cauchy_n(wavelength_um, A: float, B: float):
    """Cauchy two-term model ``n = A + B/λ²`` (cheap smooth stand-in)."""
    lam = np.asarray(wavelength_um, dtype=float)
    return A + B / (lam * lam)


def sellmeier_dn_dlambda(wavelength_um, material: str = "BK7"):
    """Analytic dispersion derivative ``dn/dλ`` (per µm) for a Sellmeier model.

    From ``n² = 1 + Σ Bᵢ λ²/(λ²−Cᵢ)``:
        d(n²)/dλ = Σ Bᵢ · (−2 λ Cᵢ)/(λ²−Cᵢ)²,   dn/dλ = d(n²)/dλ / (2n).
    This is the term that fans the spectrum into "fire": ∂dir/∂λ ∝ dn/dλ.
    """
    B, C = SELLMEIER[material]
    lam = float(wavelength_um)
    l2 = lam * lam
    dn2 = sum(Bi * (-2.0 * lam * Ci) / (l2 - Ci) ** 2 for Bi, Ci in zip(B, C))
    n = float(sellmeier_n(lam, material))
    return dn2 / (2.0 * n)


# --- reflection / refraction ----------------------------------------------

def normalize(v):
    return np.asarray(v, dtype=float) / np.linalg.norm(v)


def reflect(d, n):
    """Mirror-reflect unit direction ``d`` about unit normal ``n``."""
    return d - 2.0 * np.dot(d, n) * n


def refract(d, n, eta):
    """Snell refraction in vector form.

    Args:
        d:   incident unit direction.
        n:   unit surface normal oriented **against** the incident ray, i.e.
             ``dot(d, n) < 0``.
        eta: ratio ``n_incident / n_transmitted``.

    Returns:
        The unit refracted direction, or ``None`` on total internal reflection.
    """
    cos_i = -float(np.dot(d, n))                 # > 0 by the orientation rule
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        return None                              # total internal reflection
    cos_t = np.sqrt(1.0 - sin2_t)
    return eta * d + (eta * cos_i - cos_t) * n


def fresnel_dielectric(cos_i, eta):
    """Unpolarized Fresnel reflectance.

    ``cos_i`` is the cosine of the incidence angle (>0); ``eta = n_i / n_t``.
    Returns 1.0 under total internal reflection.
    """
    cos_i = float(cos_i)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t >= 1.0:
        return 1.0
    cos_t = np.sqrt(1.0 - sin2_t)
    rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
    rp = (eta * cos_t - cos_i) / (eta * cos_t + cos_i)
    return 0.5 * (rs * rs + rp * rp)


__all__ = [
    "SELLMEIER", "sellmeier_n", "sellmeier_dn_dlambda", "cauchy_n",
    "normalize", "reflect", "refract", "fresnel_dielectric",
]
