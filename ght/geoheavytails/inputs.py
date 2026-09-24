"""
Input-side samplers: global input families (Fig. 6b), the local shell measure
with prescribed nu (Fig. 6c), and embedded singular manifolds (Fig. 6d).

None of these existed in the previous archive, which is why the claims of
Sec. VI.C, VI.D and VI.E were unverifiable.
"""

from __future__ import annotations

import numpy as np

from .config import P0_TARGET

# --------------------------------------------------------------------------
# Global input families, all scaled to a common density at the singular point
# --------------------------------------------------------------------------
# Eq. (48) requires p_X(0) = 1/sqrt(2*pi) for every family, so that the *local*
# probability at the singular set is held fixed while the *global* shape varies.
# The tail exponent must then be identical across families -- that is the whole
# point of the test.  Scaling X by k maps p(0) -> p(0)/k, so once a family's
# natural p(0) is known analytically the correct scale factor is exact.


def _sample_gaussian(rng, n):
    # N(0,1): p(0) = 1/sqrt(2*pi) already.
    return rng.normal(0.0, 1.0, n)


def _sample_laplace(rng, n):
    # p(0) = 1/(2b)  =>  b = 1/(2*P0)
    b = 1.0 / (2.0 * P0_TARGET)
    return rng.laplace(0.0, b, n)


def _sample_logistic(rng, n):
    # p(0) = 1/(4s)  =>  s = 1/(4*P0)
    s = 1.0 / (4.0 * P0_TARGET)
    return rng.logistic(0.0, s, n)


def _sample_uniform(rng, n):
    # Uniform(-a, a): p(0) = 1/(2a)  =>  a = 1/(2*P0)
    a = 1.0 / (2.0 * P0_TARGET)
    return rng.uniform(-a, a, n)


def _sample_split_normal(rng, n):
    # Two-piece normal: f(x) = A exp(-x^2/(2 s1^2)) for x<0, s2 for x>=0,
    # with A = sqrt(2/pi)/(s1+s2).  Setting A = P0 = 1/sqrt(2*pi) gives
    # s1 + s2 = 2.  Choose a strongly skewed split.
    s1, s2 = 0.6, 1.4
    # Mass on each side is proportional to its scale.
    left = rng.random(n) < s1 / (s1 + s2)
    z = np.abs(rng.normal(0.0, 1.0, n))
    return np.where(left, -s1 * z, s2 * z)


def _sample_gaussian_mixture(rng, n):
    # Asymmetric, bimodal, finite and nonzero at the origin.
    w, mu1, s1, mu2, s2 = 0.7, -0.4, 0.5, 0.9, 0.8
    p0 = (
        w * np.exp(-0.5 * (mu1 / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
        + (1 - w) * np.exp(-0.5 * (mu2 / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))
    )
    comp = rng.random(n) < w
    x = np.where(comp, rng.normal(mu1, s1, n), rng.normal(mu2, s2, n))
    # Rescale so that p(0) matches the target exactly.
    return x * (p0 / P0_TARGET)


INPUT_FAMILIES = {
    "Gaussian": _sample_gaussian,
    "Laplace": _sample_laplace,
    "Logistic": _sample_logistic,
    "Uniform": _sample_uniform,
    "Split normal": _sample_split_normal,
    "Gaussian mixture": _sample_gaussian_mixture,
}


def density_at_zero_empirical(x, h: float = 0.02) -> float:
    """Kernel-free check of p_X(0) by counting mass in a narrow window."""
    x = np.asarray(x)
    return float(np.mean(np.abs(x) < h) / (2 * h))


# --------------------------------------------------------------------------
# Local shell measure with prescribed nu  (Eqs. 51-53)
# --------------------------------------------------------------------------

def sample_radius_nu(rng, n, c: float, nu: float):
    """Radial variable with p_R(r) ~ r^{c+nu-1} exp(-r^2/2), Eq. (51).

    Equivalently R^2 ~ Gamma(shape=(c+nu)/2, scale=2), Eq. (52).  Near r = 0
    this has exactly the shell measure assumed in Eq. (7), dP ~ r^{c-1+nu} dr,
    so it probes the nu dependence directly rather than only changing the
    global input shape.
    """
    shape = (c + nu) / 2.0
    if shape <= 0:
        raise ValueError("require c + nu > 0 for normalizability")
    return np.sqrt(rng.gamma(shape=shape, scale=2.0, size=n))


# --------------------------------------------------------------------------
# Embedded singular manifolds in R^3  (Eqs. 56-59)
# --------------------------------------------------------------------------

def manifold_distances(rng, n, R0: float = 1.0):
    """Distances to a sphere (c=1), a circle (c=2) and a point (c=3).

    Input is isotropic standard Gaussian in R^3, so the density is smooth and
    nonzero in a neighbourhood of each manifold and nu = 0.  Returns a dict of
    distance arrays keyed by name, together with the codimension.
    """
    x = rng.normal(size=(n, 3))
    x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
    rho_perp = np.hypot(x1, x2)
    norm = np.sqrt(x1 * x1 + x2 * x2 + x3 * x3)

    r_sph = np.abs(norm - R0)                       # Eq. (57), c = 1
    r_circ = np.hypot(rho_perp - R0, x3)            # Eq. (58), c = 2
    r_pt = norm                                     # Eq. (59), c = 3
    return {
        "Sphere": (r_sph, 1),
        "Circle": (r_circ, 2),
        "Point": (r_pt, 3),
    }
