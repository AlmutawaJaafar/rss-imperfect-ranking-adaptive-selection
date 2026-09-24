"""
Central configuration for all numerical experiments.

Every sample size, seed, threshold and physical constant used anywhere in this
repository is defined *here* and nowhere else.  This is deliberate: the referee
report on the previous archive noted that several sample sizes appeared only
inside figure functions and never in the manuscript, which made independent
reproduction impossible.  Any value used in a figure caption or in the
manuscript text must be traceable to a named constant below.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Random seeding
# --------------------------------------------------------------------------
# A single master seed.  Every experiment derives an independent generator from
# it via `spawn_rng(stream_id)`, which uses NumPy's SeedSequence spawning rather
# than the previous archive's `BASE_SEED + figure_number` offsetting.  Offsetting
# a counter produces generator states that are *not* guaranteed independent;
# SeedSequence spawning is.
MASTER_SEED = 20260915

# Named streams.  Integers are arbitrary but fixed; they must never be reused.
STREAM = {
    "fig01": 1,
    "fig04": 4,
    "fig05": 5,
    "fig06a": 61,
    "fig06b": 62,
    "fig06c": 63,
    "fig06d": 64,
    "fig07": 7,
    "fig08": 8,
    "fig09": 9,
    "fig10": 10,
    "fig11": 11,
    "table_exponent": 100,
    "table_robustness": 101,
    "table_cutoff": 102,
    "table_modelcomp": 103,
    "table_threshold": 104,
    "table_seed_sensitivity": 105,
}

# --------------------------------------------------------------------------
# Sample sizes  (these are the values that must appear in the manuscript)
# --------------------------------------------------------------------------
N_MAIN = 350_000        # Eq. (37): main exponent-validation sample size
N_RESOLUTION = 1_200_000  # Eq. (43): finite-resolution study
N_APPLICATION = 350_000   # all application figures (7-11); previously unstated
N_REALIZATIONS = 30       # Sec. VI.C: independent realizations per input family
N_SPECTRUM_FITS = 3_000   # genuine nonlinear refits of the pseudo-Voigt model

# --------------------------------------------------------------------------
# Tail-estimation defaults
# --------------------------------------------------------------------------
K_HILL = 4_000            # nominal fixed Hill threshold (kept for comparability)
K_GRID_MIN = 200          # threshold-stability sweep lower bound
K_GRID_MAX = 20_000       # threshold-stability sweep upper bound
K_GRID_POINTS = 60

# --------------------------------------------------------------------------
# Geometry / universality-class grids
# --------------------------------------------------------------------------
M_VALUES = (0.5, 1.0, 1.5, 2.0)      # Eq. (36)
C_VALUES = (1, 2, 3, 4)              # Sec. VI.A
NU_VALUES = (-0.5, 0.0, 1.0)         # Eq. (55)
C_VALUES_NU = (1, 2, 3)              # Eq. (55)
M_VALUES_NU = (1.0, 2.0)             # Eq. (55)

LAMBDA_VALUES = (0.30, 0.10, 0.03, 0.01)   # Eq. (44)/(46)
R0_MANIFOLD = 1.0                          # Sec. VI.E: sphere/circle radius

# Common density at the singular point imposed on every input family, Eq. (48).
P0_TARGET = 0.3989422804014327  # 1/sqrt(2*pi)

# --------------------------------------------------------------------------
# Critical exponents used in Fig. 11  (Sec. VII.E, Eqs. 95-98)
# --------------------------------------------------------------------------
# 3D Ising universality class.  The manuscript caption quotes m = 0.630 for the
# correlation length, m = 1.237 for the susceptibility and m = 1.260 for the
# critical relaxation time.  The first two are nu_c and gamma_c directly.  The
# third is *not* an independent number: by Eq. (98) the relaxation-time
# singularity order is m = z*nu_c.  With nu_c = 0.630 this requires z = 2.000.
# The previous archive hard-coded 1.89 for this curve, which corresponds to
# z = 3.0 and is inconsistent with the caption.  We therefore derive it:
NU_C = 0.630       # correlation-length exponent
GAMMA_C = 1.237    # susceptibility exponent
Z_C = 2.000        # dynamic critical exponent
M_TAU_C = Z_C * NU_C   # = 1.260, matching the manuscript caption

T_MIN_VALUES = (3e-3, 1e-2, 3e-2, 1e-1)   # control-resolution values

# --------------------------------------------------------------------------
# Figure style
# --------------------------------------------------------------------------
RC_PARAMS = {
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "axes.titlesize": 9,
    "legend.frameon": False,
}
