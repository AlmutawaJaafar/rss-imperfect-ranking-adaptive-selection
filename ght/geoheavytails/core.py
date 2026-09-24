"""
Core numerical machinery: reproducible RNGs, tail-index estimators with
threshold diagnostics, the regularized singular map, and empirical helpers.

Design decisions relevant to the referee report
-----------------------------------------------
1.  *No hard floors.*  The previous archive regularized near-zero denominators
    with `np.maximum(x, eps)`, using a different ad hoc eps in each figure.
    That is a hard-threshold protocol: it piles an atom of probability at the
    cutoff and produces the vertical CCDF drops seen in the old figures, which
    does not match the smooth crossover of Eq. (31)/(33).  Every map here uses
    the manuscript's own regularization, Eq. (28):
        Y_lambda = A (r^2 + lambda^2)^{-m/2}.
    `regularized_map` is the single implementation; nothing else regularizes.

2.  *Hill estimates are never reported at a single fixed k.*  `hill_curve`
    returns the whole Hill plot and `select_k_stability` picks a threshold from
    a stability plateau, so the fixed-k bias visible at steep tails in the old
    `figure6_exponent_validation.csv` is measured rather than inherited.
"""

from __future__ import annotations

import numpy as np

from .config import (
    MASTER_SEED,
    STREAM,
    K_GRID_MIN,
    K_GRID_MAX,
    K_GRID_POINTS,
)

# --------------------------------------------------------------------------
# Reproducible random number generation
# --------------------------------------------------------------------------

_ROOT_SEQ = np.random.SeedSequence(MASTER_SEED)


def spawn_rng(stream: str | int, realization: int = 0) -> np.random.Generator:
    """Return an independent Generator for a named stream and realization.

    Independence is guaranteed by SeedSequence spawning, not by offsetting a
    counter.  `realization` lets a single experiment draw many statistically
    independent repeats (used by the 30-realization robustness study).
    """
    sid = STREAM[stream] if isinstance(stream, str) else int(stream)
    child = _ROOT_SEQ.spawn(1)[0]
    # Derive deterministically from (master, stream, realization).
    seq = np.random.SeedSequence(
        entropy=child.entropy, spawn_key=(sid, int(realization))
    )
    return np.random.default_rng(seq)


# --------------------------------------------------------------------------
# The singular observation map
# --------------------------------------------------------------------------

def regularized_map(r, m: float, lam: float = 0.0, A: float = 1.0):
    """Eq. (28):  Y = A (r^2 + lambda^2)^{-m/2}.

    lam = 0 reproduces the bare singular map Y = A r^{-m}, Eq. (6).  For lam = 0
    an exactly-zero r would give inf; with continuous input distributions this
    is a probability-zero event, and we assert rather than silently floor it,
    so that any such event surfaces as an error instead of a fake atom.
    """
    r = np.asarray(r, dtype=float)
    if lam == 0.0:
        if np.any(r <= 0.0):
            raise ValueError(
                "bare singular map received r <= 0; use lam > 0 to regularize"
            )
        return A * r ** (-m)
    return A * (r * r + lam * lam) ** (-m / 2.0)


def y_max_theory(m: float, lam: float, A: float = 1.0) -> float:
    """Eq. (29): Y_max = A lambda^{-m}.  Used only for *comparison*, never to
    generate the data that tests it."""
    return A * lam ** (-m)


# --------------------------------------------------------------------------
# Empirical distribution helpers
# --------------------------------------------------------------------------

def positive_ccdf(x):
    """Empirical survival function of the positive, finite entries of x."""
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    return x, np.arange(x.size, 0, -1) / x.size


def log_density(x, bins: int = 90, qlo: float = 1e-3, qhi: float = 0.9995):
    """Logarithmically binned density estimate (geometric bin centres)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    edges = np.geomspace(np.quantile(x, qlo), np.quantile(x, qhi), bins + 1)
    h, e = np.histogram(x, bins=edges, density=True)
    return np.sqrt(e[:-1] * e[1:]), h


# --------------------------------------------------------------------------
# Tail index estimation
# --------------------------------------------------------------------------

def hill_gamma(x_sorted, k: int) -> float:
    """Hill estimator of the extreme-value index gamma = 1/kappa, Eq. (40).

    `x_sorted` must be ascending.  Uses the top k order statistics above the
    (k+1)-th largest as threshold.
    """
    k = int(min(k, x_sorted.size - 1))
    thr = x_sorted[-k - 1]
    return float(np.mean(np.log(x_sorted[-k:] / thr)))


def hill_alpha(x, k: int) -> float:
    """Density-tail exponent alpha = 1 + kappa = 1 + 1/gamma_Hill."""
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    return 1.0 + 1.0 / hill_gamma(x, k)


def hill_curve(x, k_min: int = K_GRID_MIN, k_max: int = K_GRID_MAX,
               n_k: int = K_GRID_POINTS):
    """Full Hill plot: arrays (k, alpha_hat) over a log-spaced grid of k.

    This is the threshold-stability diagnostic the manuscript demands of
    experimentalists (Sec. VI.E item 3) but which the previous archive never
    applied to its own synthetic data.
    """
    x = np.asarray(x, dtype=float)
    x = np.sort(x[np.isfinite(x) & (x > 0)])
    k_max = int(min(k_max, x.size - 2))
    ks = np.unique(np.geomspace(k_min, k_max, n_k).astype(int))
    alphas = np.array([1.0 + 1.0 / hill_gamma(x, int(k)) for k in ks])
    return ks, alphas


def select_k_stability(ks, alphas, window: int = 7):
    """Choose a threshold from the flattest plateau of the Hill plot.

    Resnick-Starica style: slide a window over the (log k, alpha) curve and
    pick the window with the smallest standard deviation of alpha; return the
    window's central k and the plateau mean.  This removes the arbitrariness of
    a single hard-wired k = 4000 applied to tails ranging from alpha = 1.5 to 9.
    """
    ks = np.asarray(ks)
    alphas = np.asarray(alphas)
    if ks.size < window:
        return int(ks[ks.size // 2]), float(np.mean(alphas))
    sds = np.array([
        np.std(alphas[i:i + window]) for i in range(alphas.size - window + 1)
    ])
    i = int(np.argmin(sds))
    centre = i + window // 2
    return int(ks[centre]), float(np.mean(alphas[i:i + window]))


def hill_adaptive(x):
    """Hill estimate at a stability-selected threshold. Returns (alpha, k)."""
    ks, alphas = hill_curve(x)
    k, alpha = select_k_stability(ks, alphas)
    return alpha, k


# --------------------------------------------------------------------------
# Empirical cutoff extraction (genuine Monte Carlo test of Eq. 29)
# --------------------------------------------------------------------------

def empirical_cutoff(y, q: float = 0.9999) -> float:
    """High empirical quantile used as the measured cutoff.

    A high quantile rather than the raw sample maximum: the maximum of a
    regularized sample converges to A*lambda^{-m} only as n -> inf and is
    itself a random variable with non-negligible spread.  The 99.99th
    percentile is a stable, reportable statistic.  The *ratio* to the true
    endpoint is a fixed constant independent of lambda, so the fitted slope of
    log(cutoff) against log(lambda) still estimates -m without bias.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    return float(np.quantile(y, q))


def loglog_slope(x, y):
    """Least-squares slope and standard error of log(y) vs log(x)."""
    lx = np.log(np.asarray(x, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    n = lx.size
    A = np.vstack([lx, np.ones(n)]).T
    coef, res, *_ = np.linalg.lstsq(A, ly, rcond=None)
    slope, intercept = coef
    resid = ly - A @ coef
    dof = max(n - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    return float(slope), float(intercept), float(np.sqrt(cov[0, 0]))


def hill_window_ratio(y, k: int, y_max: float) -> float:
    """Diagnostic: (Hill threshold) / (cutoff), which must be << 1.

    If the top-k observations used by the Hill estimator lie *above* the
    resolution-controlled cutoff of Eq. (29), the estimator is measuring the
    endpoint shape of Eq. (33), not the intermediate algebraic exponent, and it
    will return a spuriously steep alpha.

    The previous archive's ad hoc floors (2e-4, 0.002, 0.001, dw) put several
    application figures in exactly this regime, so their plotted CCDFs fell off
    a cliff instead of showing the predicted power law.  Every application
    figure here reports this ratio, and `check_manuscript_consistency.py`
    asserts it stays below WINDOW_MAX.
    """
    y = np.asarray(y, dtype=float)
    y = np.sort(y[np.isfinite(y) & (y > 0)])
    k = int(min(k, y.size - 1))
    return float(y[-k - 1] / y_max)


WINDOW_MAX = 0.05


def resolution_for_window(sigma_r: float, k: int, n: int,
                          p0_rel: float = 0.4, margin: float = 100.0) -> float:
    """Largest lambda that keeps the top-k order statistics in the power law.

    The k/n-th upper quantile of Y corresponds to a normal distance
    r_k ~ (k/n) / (2 p_R(0)); requiring lambda << r_k by a factor `margin`
    gives a principled resolution scale instead of an arbitrary floor.
    """
    r_k = (k / n) / (2.0 * p0_rel / sigma_r)
    return r_k / margin
