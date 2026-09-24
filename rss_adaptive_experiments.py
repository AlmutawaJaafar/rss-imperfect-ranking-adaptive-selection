#!/usr/bin/env python3
"""
Adaptive Robust RSS experiments
===============================

Purpose
-------
This script is designed for the revised RSS paper.  It tests the central
scientific question:

    Can two imperfect-ranking mechanisms having approximately the same
    scalar rank correlation rho produce different RSS efficiency?

It then studies:
  1. perfect vs random vs imperfect ranking;
  2. two different imperfect-ranking mechanisms at matched rho;
  3. sample-size / set-size effects;
  4. contamination effects;
  5. estimator dependence;
  6. a simple RSS-vs-SRS adaptive decision rule;
  7. an optional multivariate experiment.

IMPORTANT
---------
This is deliberately a simulation study, not a proof of any theorem.
The script does NOT assume that rho determines RSS efficiency.

Outputs are written to:
    rss_results/
        experiment_*.csv
        experiment_*.png

Dependencies:
    numpy, pandas, scipy, matplotlib

Optional:
    scikit-learn is NOT required.

Recommended first run:
    python rss_adaptive_experiments.py --reps 1000

For a quick test:
    python rss_adaptive_experiments.py --reps 200

For the full study:
    python rss_adaptive_experiments.py --reps 2000
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    seed: int = 20260915
    reps: int = 1000

    # RSS set size.  A cycle contains m sets, each of size m.
    m_values: Tuple[int, ...] = (3, 5, 7)

    # Number of RSS cycles. Total measured observations = m*c.
    cycles: int = 100

    # Ranking-quality targets.
    rho_targets: Tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 1.00)

    # Contamination fractions.
    eps_values: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)

    # Normal clean distribution.
    sigma: float = 1.0

    # Outlier magnitude.  Sign is randomized.
    outlier_scale: float = 8.0

    # Huber tuning constant.
    huber_c: float = 1.345

    # Tukey bisquare tuning constant.
    tukey_c: float = 4.685

    # Trim proportion for trimmed mean.
    trim: float = 0.10

    # Number of repetitions used to calibrate ranking mechanisms.
    calibration_reps: int = 120

    # Number of sets used in calibration.
    calibration_sets: int = 2000

    # Adaptive rule.
    # A small pilot is used to estimate variance before selecting RSS/SRS.
    pilot_cycles: int = 10
    adaptive_lambda: float = 0.0

    # Multivariate dimensions.
    p_values: Tuple[int, ...] = (4, 10, 50)


# ============================================================
# Basic robust location estimators
# ============================================================

def sample_mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def trimmed_mean(x: np.ndarray, trim: float = 0.10) -> float:
    x = np.sort(np.asarray(x))
    n = len(x)
    g = int(math.floor(trim * n))
    if 2 * g >= n:
        raise ValueError("trim is too large for the sample size")
    return float(np.mean(x[g:n-g]))


def huber_location(
    x: np.ndarray,
    c: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> float:
    """
    Huber M-location using iteratively reweighted updates.
    Scale is based on MAD and kept fixed during the iterations.
    """
    x = np.asarray(x, dtype=float)
    theta = float(np.median(x))
    mad = np.median(np.abs(x - theta))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = np.std(x, ddof=1)
    if scale <= 1e-12:
        return theta

    for _ in range(max_iter):
        u = (x - theta) / scale
        w = np.ones_like(u)
        mask = np.abs(u) > c
        w[mask] = c / np.abs(u[mask])
        theta_new = float(np.sum(w * x) / np.sum(w))
        if abs(theta_new - theta) <= tol * (1.0 + abs(theta)):
            return theta_new
        theta = theta_new
    return theta


def tukey_location(
    x: np.ndarray,
    c: float = 4.685,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> float:
    """
    Tukey bisquare M-location with MAD scale.
    """
    x = np.asarray(x, dtype=float)
    theta = float(np.median(x))
    mad = np.median(np.abs(x - theta))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = np.std(x, ddof=1)
    if scale <= 1e-12:
        return theta

    for _ in range(max_iter):
        u = (x - theta) / scale
        w = np.zeros_like(u)
        mask = np.abs(u) < c
        z = u[mask] / c
        w[mask] = (1.0 - z**2) ** 2

        if np.sum(w) <= 1e-12:
            return theta

        theta_new = float(np.sum(w * x) / np.sum(w))
        if abs(theta_new - theta) <= tol * (1.0 + abs(theta)):
            return theta_new
        theta = theta_new
    return theta


ESTIMATORS = {
    "mean": sample_mean,
    "trimmed": trimmed_mean,
    "huber": huber_location,
    "tukey": tukey_location,
}


def estimate(x: np.ndarray, name: str, cfg: Config) -> float:
    if name == "trimmed":
        return trimmed_mean(x, cfg.trim)
    if name == "huber":
        return huber_location(x, cfg.huber_c)
    if name == "tukey":
        return tukey_location(x, cfg.tukey_c)
    return ESTIMATORS[name](x)


# ============================================================
# Ranking mechanisms
# ============================================================

def true_ranks(x: np.ndarray) -> np.ndarray:
    """
    Returns ranks 1,...,m, where rank 1 is the smallest observation.
    """
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=int)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def rank_correlation(true_r: np.ndarray, observed_r: np.ndarray) -> float:
    """
    Spearman correlation of the rank labels.

    For permutations, this is exactly the usual rank correlation.
    For repeated / randomized assignments, it remains a convenient scalar
    summary, but it is NOT assumed to characterize the full ranking mechanism.
    """
    r = spearmanr(true_r, observed_r).statistic
    if not np.isfinite(r):
        return 0.0
    return float(r)


def noisy_score_ranking(
    x: np.ndarray,
    noise_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mechanism A:
        observed ranking is obtained by ranking x + Gaussian noise.

    This is a value-level ranking error model.
    """
    score = x + noise_sd * rng.normal(size=len(x))
    return true_ranks(score)


def adjacent_swap_ranking(
    true_r: np.ndarray,
    swap_prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mechanism B:
        start with the true rank ordering and independently swap adjacent
        pairs with probability swap_prob.

    This is a rank-level/local-error mechanism.

    The algorithm scans left-to-right.  If a pair is swapped, the scan jumps
    over the pair.
    """
    order = np.argsort(true_r).copy()
    i = 0
    while i < len(order) - 1:
        if rng.random() < swap_prob:
            order[i], order[i + 1] = order[i + 1], order[i]
            i += 2
        else:
            i += 1

    observed = np.empty_like(true_r)
    observed[order] = np.arange(1, len(order) + 1)
    return observed


def random_ranking(
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    perm = rng.permutation(m)
    observed = np.empty(m, dtype=int)
    observed[perm] = np.arange(1, m + 1)
    return observed


# ============================================================
# Calibration of ranking mechanisms
# ============================================================

def calibrate_noise_sd(
    target_rho: float,
    m: int,
    cfg: Config,
    rng: np.random.Generator,
) -> float:
    """
    Approximate calibration of Gaussian score noise so that the average
    Spearman rank correlation is close to target_rho.
    """
    if target_rho >= 0.999:
        return 0.0

    # Expand upper bound until correlation is below target.
    lo, hi = 0.0, 0.25
    for _ in range(30):
        vals = []
        for _ in range(cfg.calibration_reps):
            x = rng.normal(size=(cfg.calibration_sets, m))
            # Vectorized ranking by argsort.
            true = np.argsort(np.argsort(x, axis=1), axis=1) + 1
            y = x + hi * rng.normal(size=x.shape)
            obs = np.argsort(np.argsort(y, axis=1), axis=1) + 1
            # Spearman for ranks can be computed from squared differences.
            d2 = np.sum((true - obs) ** 2, axis=1)
            rho = 1.0 - 6.0 * d2 / (m * (m**2 - 1))
            vals.append(np.mean(rho))
        if np.mean(vals) <= target_rho:
            break
        hi *= 2.0

    for _ in range(35):
        mid = 0.5 * (lo + hi)
        vals = []
        for _ in range(max(10, cfg.calibration_reps // 4)):
            x = rng.normal(size=(cfg.calibration_sets, m))
            true = np.argsort(np.argsort(x, axis=1), axis=1) + 1
            y = x + mid * rng.normal(size=x.shape)
            obs = np.argsort(np.argsort(y, axis=1), axis=1) + 1
            d2 = np.sum((true - obs) ** 2, axis=1)
            rho = 1.0 - 6.0 * d2 / (m * (m**2 - 1))
            vals.append(np.mean(rho))
        current = float(np.mean(vals))
        if current > target_rho:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def calibrate_swap_prob(
    target_rho: float,
    m: int,
    cfg: Config,
    rng: np.random.Generator,
) -> float:
    """
    Calibrate adjacent-swap probability to approximately match target rho.
    """
    if target_rho >= 0.999:
        return 0.0

    lo, hi = 0.0, 1.0

    def mean_rho(prob: float) -> float:
        vals = []
        for _ in range(max(20, cfg.calibration_reps)):
            for _ in range(max(100, cfg.calibration_sets // 10)):
                tr = np.arange(1, m + 1)
                ob = adjacent_swap_ranking(tr, prob, rng)
                vals.append(rank_correlation(tr, ob))
        return float(np.mean(vals))

    # The map need not be perfectly monotone for very small m, so use a
    # coarse grid first and then local refinement.
    grid = np.linspace(0.0, 1.0, 21)
    grid_rho = np.array([mean_rho(q) for q in grid])
    idx = int(np.argmin(np.abs(grid_rho - target_rho)))
    best = float(grid[idx])

    if idx == 0 or idx == len(grid) - 1:
        return best

    lo = float(grid[idx - 1])
    hi = float(grid[idx + 1])

    for _ in range(20):
        mid = 0.5 * (lo + hi)
        rmid = mean_rho(mid)

        # Determine direction from nearby values.
        rlo = mean_rho(lo)
        if rlo > rmid:
            # correlation decreases with probability
            if rmid > target_rho:
                lo = mid
            else:
                hi = mid
        else:
            # fallback for unusual small-m behavior
            if abs(rmid - target_rho) < abs(rlo - target_rho):
                if rmid > target_rho:
                    lo = mid
                else:
                    hi = mid
            else:
                break

    return float(best if not np.isfinite(best) else 0.5 * (lo + hi))


# ============================================================
# Data generation
# ============================================================

def contaminate(
    x: np.ndarray,
    epsilon: float,
    outlier_scale: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Independent epsilon-contamination.

    Returns:
        contaminated sample,
        contamination indicator.
    """
    y = np.asarray(x, dtype=float).copy()
    n = len(y)
    mask = rng.random(n) < epsilon
    signs = rng.choice([-1.0, 1.0], size=n)
    y[mask] = signs[mask] * outlier_scale
    return y, mask


# ============================================================
# RSS and SRS
# ============================================================

def rss_sample(
    n_cycles: int,
    m: int,
    ranking_mechanism: str,
    mechanism_parameter: float,
    epsilon: float,
    rng: np.random.Generator,
    outlier_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Balanced RSS:
      - each cycle has m sets;
      - each set has m units;
      - rank 1,...,m is selected exactly once per cycle.

    Returns:
      selected observations,
      true ranks of selected observations,
      observed ranks assigned by the imperfect ranking mechanism.
    """
    selected = []
    selected_true_rank = []
    selected_obs_rank = []

    for _ in range(n_cycles):
        for desired_rank in range(1, m + 1):
            x = rng.normal(size=m)

            # True ranks are defined from the clean latent x BEFORE
            # contamination. This separates ranking error from contamination.
            tr = true_ranks(x)

            if ranking_mechanism == "perfect":
                obs_rank = tr.copy()

            elif ranking_mechanism == "random":
                obs_rank = random_ranking(m, rng)

            elif ranking_mechanism == "gaussian_score":
                obs_rank = noisy_score_ranking(x, mechanism_parameter, rng)

            elif ranking_mechanism == "adjacent_swap":
                obs_rank = adjacent_swap_ranking(tr, mechanism_parameter, rng)

            else:
                raise ValueError(f"Unknown mechanism: {ranking_mechanism}")

            # Select the unit believed to have desired_rank.
            idx = int(np.flatnonzero(obs_rank == desired_rank)[0])

            # Contamination occurs after ranking, so the ranking itself is
            # informative about the latent clean value.  This creates a
            # controlled separation between ranking quality and contamination.
            value = x[idx]
            if rng.random() < epsilon:
                value = rng.choice([-1.0, 1.0]) * outlier_scale

            selected.append(value)
            selected_true_rank.append(tr[idx])
            selected_obs_rank.append(obs_rank[idx])

    return (
        np.asarray(selected),
        np.asarray(selected_true_rank),
        np.asarray(selected_obs_rank),
    )


def srs_sample(
    n: int,
    epsilon: float,
    rng: np.random.Generator,
    outlier_scale: float,
) -> np.ndarray:
    x = rng.normal(size=n)
    y, _ = contaminate(x, epsilon, outlier_scale, rng)
    return y


# ============================================================
# Ranking transition matrix
# ============================================================

def estimate_transition_matrix(
    m: int,
    mechanism: str,
    parameter: float,
    n_sets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Estimate P_{rs}=Pr(observed rank=s | true rank=r).
    """
    counts = np.zeros((m, m), dtype=float)

    for _ in range(n_sets):
        x = rng.normal(size=m)
        tr = true_ranks(x)

        if mechanism == "perfect":
            ob = tr
        elif mechanism == "random":
            ob = random_ranking(m, rng)
        elif mechanism == "gaussian_score":
            ob = noisy_score_ranking(x, parameter, rng)
        elif mechanism == "adjacent_swap":
            ob = adjacent_swap_ranking(tr, parameter, rng)
        else:
            raise ValueError(mechanism)

        for r in range(1, m + 1):
            idx = int(np.flatnonzero(tr == r)[0])
            s = int(ob[idx])
            counts[r - 1, s - 1] += 1.0

    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / np.maximum(row_sums, 1.0)


# ============================================================
# Experiment 1: same rho, different ranking mechanisms
# ============================================================

def experiment_matched_rho(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    rows = []

    for m in cfg.m_values:
        print(f"\n[Calibration] m={m}")

        for rho_target in cfg.rho_targets:
            noise_sd = calibrate_noise_sd(
                rho_target, m, cfg, rng
            )
            swap_prob = calibrate_swap_prob(
                rho_target, m, cfg, rng
            )

            mechanisms = [
                ("perfect", 0.0),
                ("random", 0.0),
                ("gaussian_score", noise_sd),
                ("adjacent_swap", swap_prob),
            ]

            for mech, param in mechanisms:
                transition = estimate_transition_matrix(
                    m, mech, param, 20000, rng
                )

                for estimator in ["mean", "trimmed", "huber", "tukey"]:
                    rss_est = []
                    srs_est = []
                    realized_rho = []

                    n = cfg.cycles * m

                    for _ in range(cfg.reps):
                        xr, tr, orank = rss_sample(
                            cfg.cycles,
                            m,
                            mech,
                            param,
                            epsilon=0.0,
                            rng=rng,
                            outlier_scale=cfg.outlier_scale,
                        )
                        xs = srs_sample(
                            n,
                            epsilon=0.0,
                            rng=rng,
                            outlier_scale=cfg.outlier_scale,
                        )

                        rss_est.append(estimate(xr, estimator, cfg))
                        srs_est.append(estimate(xs, estimator, cfg))
                        realized_rho.append(
                            rank_correlation(tr, orank)
                        )

                    rss_est = np.asarray(rss_est)
                    srs_est = np.asarray(srs_est)

                    var_rss = float(np.var(rss_est, ddof=1))
                    var_srs = float(np.var(srs_est, ddof=1))

                    rows.append({
                        "experiment": "matched_rho",
                        "m": m,
                        "cycles": cfg.cycles,
                        "n": n,
                        "rho_target": rho_target,
                        "mechanism": mech,
                        "parameter": param,
                        "estimator": estimator,
                        "epsilon": 0.0,
                        "realized_rho_mean": float(np.mean(realized_rho)),
                        "realized_rho_sd": float(np.std(realized_rho, ddof=1)),
                        "var_rss": var_rss,
                        "var_srs": var_srs,
                        "RE_var": var_srs / var_rss if var_rss > 0 else np.nan,
                        "bias_rss": float(np.mean(rss_est)),
                        "bias_srs": float(np.mean(srs_est)),
                    })

    return pd.DataFrame(rows)


# ============================================================
# Experiment 2: contamination / estimator dependence
# ============================================================

def experiment_contamination(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 100)

    rows = []

    # Use mechanisms that are directly interpretable.
    for m in (5, 7):
        for rho_target in (0.50, 0.75, 0.90):
            noise_sd = calibrate_noise_sd(
                rho_target, m, cfg, rng
            )
            swap_prob = calibrate_swap_prob(
                rho_target, m, cfg, rng
            )

            mechanisms = [
                ("gaussian_score", noise_sd),
                ("adjacent_swap", swap_prob),
                ("random", 0.0),
            ]

            for mech, param in mechanisms:
                for eps in cfg.eps_values:
                    for estimator in ["mean", "trimmed", "huber", "tukey"]:
                        rss_est = []
                        srs_est = []

                        n = cfg.cycles * m

                        for _ in range(cfg.reps):
                            xr, _, _ = rss_sample(
                                cfg.cycles,
                                m,
                                mech,
                                param,
                                epsilon=eps,
                                rng=rng,
                                outlier_scale=cfg.outlier_scale,
                            )
                            xs = srs_sample(
                                n,
                                epsilon=eps,
                                rng=rng,
                                outlier_scale=cfg.outlier_scale,
                            )

                            rss_est.append(estimate(xr, estimator, cfg))
                            srs_est.append(estimate(xs, estimator, cfg))

                        rss_est = np.asarray(rss_est)
                        srs_est = np.asarray(srs_est)

                        mse_rss = float(np.mean(rss_est ** 2))
                        mse_srs = float(np.mean(srs_est ** 2))

                        rows.append({
                            "experiment": "contamination",
                            "m": m,
                            "cycles": cfg.cycles,
                            "n": n,
                            "rho_target": rho_target,
                            "mechanism": mech,
                            "parameter": param,
                            "estimator": estimator,
                            "epsilon": eps,
                            "MSE_rss": mse_rss,
                            "MSE_srs": mse_srs,
                            "RE_MSE": mse_srs / mse_rss if mse_rss > 0 else np.nan,
                        })

    return pd.DataFrame(rows)


# ============================================================
# Experiment 3: dimension / multivariate ranking
# ============================================================

def robust_componentwise_mean(X: np.ndarray) -> np.ndarray:
    return np.mean(X, axis=0)


def robust_componentwise_trimmed(X: np.ndarray, trim: float) -> np.ndarray:
    return np.array([trimmed_mean(X[:, j], trim) for j in range(X.shape[1])])


def mahalanobis_score(
    X: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    """
    Squared Mahalanobis score around the sample mean.
    This is a ranking score only.  It is not claimed to be a robust
    covariance estimator.
    """
    mu = np.mean(X, axis=0)
    Z = X - mu
    S = np.cov(Z, rowvar=False)

    if X.shape[1] == 1:
        S = np.array([[float(S)]])

    S = np.atleast_2d(S)
    S = S + regularization * np.eye(S.shape[0])

    Sinv = np.linalg.pinv(S)
    return np.einsum("ij,jk,ik->i", Z, Sinv, Z)


def multivariate_rss(
    cycles: int,
    m: int,
    p: int,
    ranking_noise: float,
    epsilon: float,
    rng: np.random.Generator,
    outlier_scale: float,
) -> np.ndarray:
    """
    Multivariate RSS using Mahalanobis-distance ranking.

    Ranking is performed on the latent clean vector, then contamination is
    introduced in the selected observation.
    """
    selected = []

    for _ in range(cycles):
        for desired_rank in range(1, m + 1):
            X = rng.normal(size=(m, p))

            score = mahalanobis_score(X)
            noisy_score = score + ranking_noise * rng.normal(size=m)

            observed_rank = np.argsort(np.argsort(noisy_score)) + 1

            idx = int(np.flatnonzero(observed_rank == desired_rank)[0])

            x = X[idx].copy()
            if rng.random() < epsilon:
                # Radial contamination.
                direction = rng.normal(size=p)
                direction /= max(np.linalg.norm(direction), 1e-12)
                x = outlier_scale * direction

            selected.append(x)

    return np.asarray(selected)


def multivariate_srs(
    n: int,
    p: int,
    epsilon: float,
    rng: np.random.Generator,
    outlier_scale: float,
) -> np.ndarray:
    X = rng.normal(size=(n, p))
    mask = rng.random(n) < epsilon
    if np.any(mask):
        directions = rng.normal(size=(np.sum(mask), p))
        directions /= np.maximum(
            np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
        )
        X[mask] = outlier_scale * directions
    return X


def experiment_multivariate(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 200)

    rows = []

    m = 5
    for p in cfg.p_values:
        # Ranking noise is scaled by the dispersion of Mahalanobis scores.
        # We calibrate it empirically to three useful quality levels.
        base = []
        for _ in range(200):
            X = rng.normal(size=(m, p))
            s = mahalanobis_score(X)
            base.append(np.std(s))
        score_sd = float(np.median(base))

        for noise_fraction in (0.0, 0.10, 0.30, 0.60):
            ranking_noise = noise_fraction * score_sd

            for eps in (0.0, 0.05, 0.10):
                for estimator in ("mean", "trimmed"):
                    rss_est = []
                    srs_est = []

                    n = cfg.cycles * m

                    for _ in range(max(100, cfg.reps // 2)):
                        XR = multivariate_rss(
                            cfg.cycles, m, p,
                            ranking_noise, eps, rng,
                            cfg.outlier_scale,
                        )
                        XS = multivariate_srs(
                            n, p, eps, rng,
                            cfg.outlier_scale,
                        )

                        if estimator == "mean":
                            er = robust_componentwise_mean(XR)
                            es = robust_componentwise_mean(XS)
                        else:
                            er = robust_componentwise_trimmed(XR, cfg.trim)
                            es = robust_componentwise_trimmed(XS, cfg.trim)

                        # MSE for the location vector.
                        rss_est.append(np.mean(er ** 2))
                        srs_est.append(np.mean(es ** 2))

                    rss_est = np.asarray(rss_est)
                    srs_est = np.asarray(srs_est)

                    rows.append({
                        "experiment": "multivariate",
                        "p": p,
                        "m": m,
                        "cycles": cfg.cycles,
                        "ranking_noise_fraction": noise_fraction,
                        "epsilon": eps,
                        "estimator": estimator,
                        "MSE_rss": float(np.mean(rss_est)),
                        "MSE_srs": float(np.mean(srs_est)),
                        "RE_MSE": float(np.mean(srs_est) / np.mean(rss_est)),
                    })

    return pd.DataFrame(rows)


# ============================================================
# Experiment 4: adaptive RSS vs SRS
# ============================================================

def pilot_variance_estimate(
    estimator: str,
    m: int,
    mechanism: str,
    parameter: float,
    epsilon: float,
    pilot_cycles: int,
    cfg: Config,
    rng: np.random.Generator,
    B: int = 25,
) -> Tuple[float, float]:
    """
    Estimate RSS and SRS variance from a small pilot.

    Returns:
        estimated RSS variance,
        estimated SRS variance.
    """
    rss_values = []
    srs_values = []
    n = pilot_cycles * m

    for _ in range(B):
        xr, _, _ = rss_sample(
            pilot_cycles, m, mechanism, parameter, epsilon,
            rng, cfg.outlier_scale
        )
        xs = srs_sample(
            n, epsilon, rng, cfg.outlier_scale
        )
        rss_values.append(estimate(xr, estimator, cfg))
        srs_values.append(estimate(xs, estimator, cfg))

    vr = float(np.var(rss_values, ddof=1))
    vs = float(np.var(srs_values, ddof=1))
    return vr, vs


def experiment_adaptive(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 300)
    rows = []

    m = 5

    scenarios = [
        ("perfect", 0.0),
        ("random", 0.0),
        ("gaussian_score", calibrate_noise_sd(0.50, m, cfg, rng)),
        ("gaussian_score", calibrate_noise_sd(0.90, m, cfg, rng)),
        ("adjacent_swap", calibrate_swap_prob(0.50, m, cfg, rng)),
        ("adjacent_swap", calibrate_swap_prob(0.90, m, cfg, rng)),
    ]

    for mechanism, parameter in scenarios:
        for eps in (0.0, 0.05, 0.10):
            for estimator in ("mean", "trimmed", "huber", "tukey"):
                adaptive_values = []
                rss_values = []
                srs_values = []
                decisions = []

                n = cfg.cycles * m

                for _ in range(cfg.reps):
                    vr, vs = pilot_variance_estimate(
                        estimator, m, mechanism, parameter, eps,
                        cfg.pilot_cycles, cfg, rng, B=15
                    )

                    # Cost term is currently zero.  Keep the expression
                    # explicit so that a future cost-aware version can use
                    # measured ranking / estimator costs.
                    rss_score = vr + cfg.adaptive_lambda * 0.0
                    srs_score = vs + cfg.adaptive_lambda * 0.0

                    use_rss = rss_score < srs_score
                    decisions.append(int(use_rss))

                    xr, _, _ = rss_sample(
                        cfg.cycles, m, mechanism, parameter,
                        eps, rng, cfg.outlier_scale
                    )
                    xs = srs_sample(
                        n, eps, rng, cfg.outlier_scale
                    )

                    er = estimate(xr, estimator, cfg)
                    es = estimate(xs, estimator, cfg)

                    rss_values.append(er)
                    srs_values.append(es)
                    adaptive_values.append(er if use_rss else es)

                adaptive_values = np.asarray(adaptive_values)
                rss_values = np.asarray(rss_values)
                srs_values = np.asarray(srs_values)

                rows.append({
                    "experiment": "adaptive",
                    "m": m,
                    "cycles": cfg.cycles,
                    "n": n,
                    "mechanism": mechanism,
                    "parameter": parameter,
                    "epsilon": eps,
                    "estimator": estimator,
                    "adaptive_MSE": float(np.mean(adaptive_values ** 2)),
                    "rss_MSE": float(np.mean(rss_values ** 2)),
                    "srs_MSE": float(np.mean(srs_values ** 2)),
                    "adaptive_minus_best_MSE": float(
                        np.mean(adaptive_values ** 2)
                        - min(np.mean(rss_values ** 2), np.mean(srs_values ** 2))
                    ),
                    "fraction_RSS_selected": float(np.mean(decisions)),
                })

    return pd.DataFrame(rows)


# ============================================================
# Diagnostics and plots
# ============================================================

def summarize_matched_rho(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each target rho and estimator, compare the two imperfect-ranking
    mechanisms.  The key diagnostic is the spread in RE_var at matched rho.
    """
    d = df[df["mechanism"].isin(["gaussian_score", "adjacent_swap"])].copy()

    out = (
        d.groupby(
            ["m", "rho_target", "estimator"], as_index=False
        )
        .agg(
            mean_realized_rho=("realized_rho_mean", "mean"),
            max_abs_rho_difference=(
                "realized_rho_mean",
                lambda z: float(np.max(z) - np.min(z)),
            ),
            min_RE_var=("RE_var", "min"),
            max_RE_var=("RE_var", "max"),
            mean_RE_var=("RE_var", "mean"),
        )
    )

    out["RE_spread"] = out["max_RE_var"] - out["min_RE_var"]
    return out


def make_plots(
    matched: pd.DataFrame,
    contamination: pd.DataFrame,
    adaptive: pd.DataFrame,
    outdir: str,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    # ---- Plot 1: matched rho, RE by mechanism ----
    for estimator in ["mean", "trimmed", "huber", "tukey"]:
        d = matched[
            (matched["estimator"] == estimator)
            & (matched["mechanism"].isin(
                ["gaussian_score", "adjacent_swap"]
            ))
            & (matched["m"] == 5)
        ].copy()

        if d.empty:
            continue

        plt.figure(figsize=(8, 5))
        for mech in ["gaussian_score", "adjacent_swap"]:
            z = d[d["mechanism"] == mech]
            plt.plot(
                z["rho_target"],
                z["RE_var"],
                marker="o",
                label=mech,
            )
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("Target rank correlation")
        plt.ylabel("Relative efficiency (variance)")
        plt.title(f"Matched-rho comparison: {estimator}, m=5")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"matched_rho_{estimator}.png"),
            dpi=180,
        )
        plt.close()

    # ---- Plot 2: contamination ----
    for estimator in ["mean", "trimmed", "huber", "tukey"]:
        d = contamination[
            (contamination["estimator"] == estimator)
            & (contamination["m"] == 5)
        ].copy()
        if d.empty:
            continue

        plt.figure(figsize=(8, 5))
        for mech in sorted(d["mechanism"].unique()):
            z = d[d["mechanism"] == mech]
            z = z.groupby("epsilon", as_index=False)["RE_MSE"].mean()
            plt.plot(
                z["epsilon"],
                z["RE_MSE"],
                marker="o",
                label=mech,
            )
        plt.axhline(1.0, linestyle="--", linewidth=1)
        plt.xlabel("Contamination fraction")
        plt.ylabel("Relative efficiency (MSE)")
        plt.title(f"Contamination experiment: {estimator}, m=5")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"contamination_{estimator}.png"),
            dpi=180,
        )
        plt.close()

    # ---- Plot 3: adaptive ----
    d = adaptive.copy()
    if not d.empty:
        for estimator in ["mean", "trimmed", "huber", "tukey"]:
            z = d[d["estimator"] == estimator]
            if z.empty:
                continue

            plt.figure(figsize=(9, 5))
            labels = []
            values = []
            for _, row in z.iterrows():
                label = f"{row['mechanism']}, eps={row['epsilon']}"
                labels.append(label)
                values.append(row["adaptive_MSE"])

            x = np.arange(len(values))
            plt.plot(x, values, marker="o", label="Adaptive")
            plt.xticks(x, labels, rotation=55, ha="right")
            plt.ylabel("MSE")
            plt.title(f"Adaptive RSS/SRS: {estimator}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(outdir, f"adaptive_{estimator}.png"),
                dpi=180,
            )
            plt.close()


def print_key_diagnostics(
    matched: pd.DataFrame,
    contamination: pd.DataFrame,
    adaptive: pd.DataFrame,
) -> None:
    print("\n" + "=" * 78)
    print("KEY DIAGNOSTICS")
    print("=" * 78)

    print("\n1. Matched-rho mechanism comparison")
    summary = summarize_matched_rho(matched)
    print(summary.to_string(index=False))

    print("\nInterpretation:")
    print(
        "  If two mechanisms have nearly identical realized rho but a "
        "non-negligible RE spread,\n"
        "  this is direct evidence that scalar rho does not determine RSS "
        "efficiency."
    )

    print("\n2. Cases where RSS is worse than SRS (RE < 1)")
    bad = matched[matched["RE_var"] < 1.0]
    if bad.empty:
        print("  None in this run.")
    else:
        print(
            bad[
                [
                    "m", "rho_target", "mechanism",
                    "estimator", "realized_rho_mean", "RE_var"
                ]
            ]
            .sort_values("RE_var")
            .head(30)
            .to_string(index=False)
        )

    print("\n3. Contamination: RSS worse in MSE")
    bad2 = contamination[contamination["RE_MSE"] < 1.0]
    if bad2.empty:
        print("  None in this run.")
    else:
        print(
            bad2[
                [
                    "m", "rho_target", "mechanism",
                    "estimator", "epsilon", "RE_MSE"
                ]
            ]
            .sort_values("RE_MSE")
            .head(30)
            .to_string(index=False)
        )

    print("\n4. Adaptive rule")
    if not adaptive.empty:
        print(
            adaptive[
                [
                    "mechanism", "epsilon", "estimator",
                    "adaptive_MSE", "rss_MSE", "srs_MSE",
                    "fraction_RSS_selected"
                ]
            ].to_string(index=False)
        )


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reps",
        type=int,
        default=1000,
        help="Monte Carlo repetitions (default: 1000)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="RSS cycles (default: 100)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="rss_results",
        help="output directory",
    )
    parser.add_argument(
        "--skip-multivariate",
        action="store_true",
        help="skip the optional multivariate experiment",
    )
    parser.add_argument(
        "--skip-adaptive",
        action="store_true",
        help="skip the adaptive experiment",
    )
    args = parser.parse_args()

    cfg = Config(
        reps=args.reps,
        cycles=args.cycles,
    )

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 78)
    print("ADAPTIVE ROBUST RSS EXPERIMENTS")
    print("=" * 78)
    print(f"seed   = {cfg.seed}")
    print(f"reps   = {cfg.reps}")
    print(f"cycles = {cfg.cycles}")
    print(f"m      = {cfg.m_values}")

    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------
    matched = experiment_matched_rho(cfg)
    matched_path = os.path.join(args.outdir, "experiment_matched_rho.csv")
    matched.to_csv(matched_path, index=False)

    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------
    contamination = experiment_contamination(cfg)
    contamination_path = os.path.join(
        args.outdir, "experiment_contamination.csv"
    )
    contamination.to_csv(contamination_path, index=False)

    # --------------------------------------------------------
    # Experiment 3
    # --------------------------------------------------------
    if args.skip_multivariate:
        multivariate = pd.DataFrame()
    else:
        multivariate = experiment_multivariate(cfg)
        multivariate.to_csv(
            os.path.join(args.outdir, "experiment_multivariate.csv"),
            index=False,
        )

    # --------------------------------------------------------
    # Experiment 4
    # --------------------------------------------------------
    if args.skip_adaptive:
        adaptive = pd.DataFrame()
    else:
        adaptive = experiment_adaptive(cfg)
        adaptive.to_csv(
            os.path.join(args.outdir, "experiment_adaptive.csv"),
            index=False,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    summary = summarize_matched_rho(matched)
    summary.to_csv(
        os.path.join(args.outdir, "summary_matched_rho.csv"),
        index=False,
    )

    make_plots(
        matched,
        contamination,
        adaptive,
        args.outdir,
    )

    print_key_diagnostics(
        matched,
        contamination,
        adaptive,
    )

    print("\n" + "=" * 78)
    print("FILES WRITTEN")
    print("=" * 78)
    for f in sorted(os.listdir(args.outdir)):
        print(os.path.join(args.outdir, f))

    print("\nDone.")


if __name__ == "__main__":
    main()
