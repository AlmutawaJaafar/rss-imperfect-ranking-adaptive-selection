#!/usr/bin/env python3
"""
rss_adaptive_experiments_v2.py
==============================

Version 2 of the RSS simulation study.

PRIMARY SCIENTIFIC QUESTION
----------------------------
Does a scalar rank-correlation measure rho determine RSS efficiency?

V2 is deliberately narrower and cleaner than the earlier experiment.  It
compares two different imperfect-ranking mechanisms at empirically matched
rank correlation:

    1. Gaussian rank-score noise
    2. Laplace rank-score noise

The mechanisms are calibrated by Monte Carlo so that their *expected* rank
correlations are as close as possible to the requested target.  The final
experiment reports the actually realized correlation, its Monte Carlo SE,
and whether the two mechanisms satisfy the requested matching tolerance.

CONTROLS
--------
    perfect ranking: rho = 1
    random ranking:  rho approximately 0

PRIMARY METRICS
---------------
    RE_Var = Var(theta_hat_SRS) / Var(theta_hat_RSS)
    RE_MSE = MSE(theta_hat_SRS) / MSE(theta_hat_RSS)

The two metrics are kept separate.  No claim is made that rho alone
characterizes either one.

CONTAMINATION
-------------
Two modes are available:
    post_ranking : ranking is determined from clean observations, then
                   response contamination is applied.
    pre_ranking  : contamination is applied before ranking, so contamination
                   can affect the ranking itself.

ADAPTIVE EXPERIMENT
-------------------
Optional.  A pilot simulation estimates RSS and SRS MSE and chooses the
lower estimated MSE design.  The pilot and evaluation samples are independent.

OUTPUTS
-------
rss_results_v2/
    calibration_v2.csv
    matched_rho_v2.csv
    matched_rho_summary_v2.csv
    transition_matrices_v2.csv
    transition_distance_v2.csv
    controls_v2.csv
    contamination_v2.csv
    adaptive_v2.csv                 (if --run-adaptive is used)
    *.png                            (if matplotlib is available)

Recommended quick run
---------------------
python3.10 rss_adaptive_experiments_v2.py \
    --reps 200 --cycles 50 --m 5 --targets 0.25 0.50 0.75 0.90 \
    --skip-contamination --skip-plots

Recommended first serious run
-----------------------------
python3.10 rss_adaptive_experiments_v2.py \
    --reps 1000 --cycles 100 --m 5 --targets 0.25 0.50 0.75 0.90 \
    --calibration-trials 20000

Then, only after inspecting the matched-rho results, run contamination/adaptive
experiments.

Dependencies
------------
numpy, pandas
matplotlib is optional and only used for plots.
No scipy or scikit-learn is required.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    seed: int = 20260915
    reps: int = 1000
    cycles: int = 100
    m: int = 5
    rho_targets: Tuple[float, ...] = (0.25, 0.50, 0.75, 0.90)

    # Pilot calibration for ranking-noise scales.
    calibration_trials: int = 20000
    calibration_grid_size: int = 120
    calibration_scale_min: float = 0.01
    calibration_scale_max: float = 100.0

    # Matching tolerance for the final empirical comparison.
    rho_match_tolerance: float = 0.02

    # Estimators.
    trim: float = 0.10
    huber_c: float = 1.345
    tukey_c: float = 4.685

    # Contamination.
    eps_values: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20)
    outlier_scale: float = 8.0

    # Bootstrap CI for ratios.
    bootstrap_reps: int = 400

    # Adaptive pilot.
    adaptive_trials: int = 100
    pilot_reps: int = 30
    pilot_cycles: int = 10

    outdir: str = "rss_results_v2"


ESTIMATOR_NAMES = ("mean", "trimmed", "huber", "tukey")
IMPERFECT_MECHANISMS = ("gaussian", "laplace")
ALL_MECHANISMS = ("perfect", "random", "gaussian", "laplace")


# ============================================================
# Robust location estimators
# ============================================================

def sample_mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def trimmed_mean(x: np.ndarray, trim: float = 0.10) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    g = int(math.floor(trim * n))
    if 2 * g >= n:
        raise ValueError("trim is too large for the sample size")
    return float(np.mean(x[g:n-g]))


def _mad_scale(x: np.ndarray, center: float) -> float:
    mad = float(np.median(np.abs(x - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return scale


def huber_location(x: np.ndarray, c: float = 1.345,
                   max_iter: int = 100, tol: float = 1e-9) -> float:
    x = np.asarray(x, dtype=float)
    theta = float(np.median(x))
    scale = _mad_scale(x, theta)
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


def tukey_location(x: np.ndarray, c: float = 4.685,
                   max_iter: int = 100, tol: float = 1e-9) -> float:
    x = np.asarray(x, dtype=float)
    theta = float(np.median(x))
    scale = _mad_scale(x, theta)
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


def estimate(x: np.ndarray, name: str, cfg: Config) -> float:
    if name == "mean":
        return sample_mean(x)
    if name == "trimmed":
        return trimmed_mean(x, cfg.trim)
    if name == "huber":
        return huber_location(x, cfg.huber_c)
    if name == "tukey":
        return tukey_location(x, cfg.tukey_c)
    raise ValueError(f"Unknown estimator: {name}")


# ============================================================
# Rank utilities
# ============================================================

def ranks_from_values(x: np.ndarray) -> np.ndarray:
    """Ranks 1,...,m; rank 1 is the smallest value."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=int)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def ranks_from_order(order: np.ndarray) -> np.ndarray:
    """Observed rank of each candidate, given candidate indices in rank order."""
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def pearson_rank_corr(true_rank: np.ndarray, observed_rank: np.ndarray) -> float:
    """Pearson correlation of rank labels (Spearman correlation)."""
    a = np.asarray(true_rank, dtype=float)
    b = np.asarray(observed_rank, dtype=float)
    da = a - np.mean(a)
    db = b - np.mean(b)
    den = math.sqrt(float(np.dot(da, da) * np.dot(db, db)))
    if den <= 0:
        return 0.0
    return float(np.dot(da, db) / den)


# ============================================================
# Ranking mechanisms
# ============================================================

def observed_order_from_true_rank(
    true_rank: np.ndarray,
    mechanism: str,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate the observed ranking order.

    The ranking error is deliberately specified on the rank scale.  This is
    useful here because it makes the ranking mechanism explicit and separates
    ranking error from the spacing of the response values.
    """
    m = len(true_rank)

    if mechanism == "perfect":
        return np.argsort(true_rank, kind="mergesort")

    if mechanism == "random":
        return rng.permutation(m)

    if mechanism == "gaussian":
        scores = true_rank.astype(float) + scale * rng.normal(size=m)
        return np.argsort(scores, kind="mergesort")

    if mechanism == "laplace":
        scores = true_rank.astype(float) + scale * rng.laplace(size=m)
        return np.argsort(scores, kind="mergesort")

    raise ValueError(f"Unknown ranking mechanism: {mechanism}")


# ============================================================
# Calibration
# ============================================================

def calibration_rho(
    m: int,
    mechanism: str,
    scale: float,
    trials: int,
    rng: np.random.Generator,
) -> float:
    """Estimate expected rank correlation for a mechanism/scale pair."""
    true = np.tile(np.arange(1, m + 1, dtype=float), (trials, 1))

    if mechanism == "gaussian":
        scores = true + scale * rng.normal(size=(trials, m))
    elif mechanism == "laplace":
        scores = true + scale * rng.laplace(size=(trials, m))
    else:
        raise ValueError("Calibration is only for imperfect mechanisms")

    # argsort(argsort(.)) converts scores into observed rank labels.
    obs = np.argsort(np.argsort(scores, axis=1, kind="mergesort"),
                     axis=1, kind="mergesort").astype(float) + 1.0

    a = true - true.mean(axis=1, keepdims=True)
    b = obs - obs.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a*a, axis=1) * np.sum(b*b, axis=1))
    rho = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return float(np.mean(rho))


def calibrate_mechanism(
    cfg: Config,
    mechanism: str,
    target: float,
    seed_offset: int,
) -> Tuple[float, float]:
    """
    Calibrate the rank-noise scale by bisection on the expected rho.

    For both Gaussian and Laplace additive rank-score noise, increasing the
    scale moves the ranking continuously from perfect toward random.  A
    deterministic calibration seed is reused at every bisection step so the
    Monte Carlo objective is much less noisy than a fresh random pilot at each
    candidate scale.
    """
    if not (0.0 <= target <= 1.0):
        raise ValueError("target rho must lie in [0, 1]")
    if target == 1.0:
        return 0.0, 1.0

    lo = 0.0
    hi = cfg.calibration_scale_max
    rng_seed = cfg.seed + seed_offset

    # Verify that the upper bound is sufficiently close to random ranking.
    rho_hi = calibration_rho(
        cfg.m, mechanism, hi, cfg.calibration_trials,
        np.random.default_rng(rng_seed),
    )
    if rho_hi > target:
        # Expand the upper bound if necessary.
        for _ in range(8):
            hi *= 2.0
            rho_hi = calibration_rho(
                cfg.m, mechanism, hi, cfg.calibration_trials,
                np.random.default_rng(rng_seed),
            )
            if rho_hi <= target:
                break

    for _ in range(28):
        mid = 0.5 * (lo + hi)
        rho_mid = calibration_rho(
            cfg.m, mechanism, mid, cfg.calibration_trials,
            np.random.default_rng(rng_seed),
        )
        # Expected rho is decreasing in the noise scale.
        if rho_mid > target:
            lo = mid
        else:
            hi = mid

    scale = 0.5 * (lo + hi)
    rho = calibration_rho(
        cfg.m, mechanism, scale, cfg.calibration_trials,
        np.random.default_rng(rng_seed),
    )
    return float(scale), float(rho)


def build_calibration_table(cfg: Config) -> pd.DataFrame:
    rows = []
    for j, target in enumerate(cfg.rho_targets):
        for i, mech in enumerate(IMPERFECT_MECHANISMS):
            scale, rho = calibrate_mechanism(
                cfg, mech, target,
                seed_offset=10000 + 100*j + i,
            )
            rows.append({
                "m": cfg.m,
                "target_rho": target,
                "mechanism": mech,
                "calibrated_scale": scale,
                "pilot_expected_rho": rho,
                "pilot_abs_error": abs(rho - target),
                "calibration_trials": cfg.calibration_trials,
            })
    return pd.DataFrame(rows)


# ============================================================
# One RSS/SRS dataset
# ============================================================

def contaminate_exact(
    x: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    outlier_scale: float,
) -> np.ndarray:
    """Replace exactly floor(epsilon*N) observations by gross errors."""
    y = np.array(x, copy=True, dtype=float)
    N = y.size
    q = int(math.floor(epsilon * N + 1e-12))
    if q <= 0:
        return y
    idx = rng.choice(N, size=q, replace=False)
    signs = rng.choice(np.array([-1.0, 1.0]), size=q)
    y.flat[idx] = signs * outlier_scale
    return y


def simulate_dataset(
    m: int,
    cycles: int,
    mechanism: str,
    scale: float,
    estimator: str,
    cfg: Config,
    rng: np.random.Generator,
    epsilon: float = 0.0,
    contamination_mode: str = "post_ranking",
) -> Dict[str, float]:
    """
    Generate one finite candidate population, one RSS sample and one SRS sample.

    There are k=cycles RSS cycles.  Each cycle has m sets of size m, hence
    n=cycles*m measured observations and N=cycles*m*m candidate observations.
    """
    if contamination_mode not in ("post_ranking", "pre_ranking"):
        raise ValueError("contamination_mode must be post_ranking or pre_ranking")

    X_clean = rng.normal(size=(cycles, m, m))

    # For pre-ranking contamination, ranking sees contaminated responses.
    if contamination_mode == "pre_ranking" and epsilon > 0:
        X_rank = contaminate_exact(
            X_clean, epsilon, rng, cfg.outlier_scale
        ).reshape(cycles, m, m)
    else:
        X_rank = X_clean.copy()

    selected_flat_indices: List[int] = []
    transition_counts = np.zeros((m, m), dtype=np.int64)
    rho_pairs_true: List[int] = []
    rho_pairs_obs: List[int] = []

    for c in range(cycles):
        for s in range(m):
            xset = X_rank[c, s, :]
            tr = ranks_from_values(xset)
            order = observed_order_from_true_rank(tr, mechanism, scale, rng)
            obs = ranks_from_order(order)

            # RSS cycle s selects the observation assigned observed rank s+1.
            selected_candidate = int(order[s])
            selected_flat_indices.append((c * m * m) + (s * m) + selected_candidate)

            # Record true-rank -> observed-rank transitions.
            # IMPORTANT: r is the candidate index, whereas tr[r] is its
            # true rank.  The transition matrix is indexed by ranks, not
            # candidate positions.
            for r in range(m):
                transition_counts[tr[r] - 1, obs[r] - 1] += 1
            rho_pairs_true.extend(tr.tolist())
            rho_pairs_obs.extend(obs.tolist())

    # Response contamination after ranking leaves the selection mechanism
    # untouched, whereas pre-ranking contamination changes X_rank itself.
    if contamination_mode == "post_ranking" and epsilon > 0:
        X_used = contaminate_exact(
            X_clean, epsilon, rng, cfg.outlier_scale
        ).reshape(cycles, m, m)
    else:
        X_used = X_rank

    flat = X_used.ravel()
    rss_sample = flat[np.asarray(selected_flat_indices, dtype=int)]

    n = len(selected_flat_indices)
    srs_indices = rng.choice(flat.size, size=n, replace=False)
    srs_sample = flat[srs_indices]

    rss_est = estimate(rss_sample, estimator, cfg)
    srs_est = estimate(srs_sample, estimator, cfg)

    realized_rho = pearson_rank_corr(
        np.asarray(rho_pairs_true), np.asarray(rho_pairs_obs)
    )

    return {
        "rss_est": rss_est,
        "srs_est": srs_est,
        "realized_rho": realized_rho,
        "transition_counts": transition_counts,
    }


# ============================================================
# Monte Carlo summaries
# ============================================================

def bootstrap_ratio_ci(
    rss: np.ndarray,
    srs: np.ndarray,
    metric: str,
    B: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Paired bootstrap CI for Var(SRS)/Var(RSS) or MSE(SRS)/MSE(RSS)."""
    R = len(rss)
    if R < 10 or B <= 0:
        return np.nan, np.nan

    idx = rng.integers(0, R, size=(B, R))
    rb = rss[idx]
    sb = srs[idx]

    if metric == "var":
        num = np.var(sb, axis=1, ddof=1)
        den = np.var(rb, axis=1, ddof=1)
    elif metric == "mse":
        num = np.mean(sb**2, axis=1)
        den = np.mean(rb**2, axis=1)
    else:
        raise ValueError(metric)

    ratios = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-15)
    ratios = ratios[np.isfinite(ratios)]
    if len(ratios) == 0:
        return np.nan, np.nan
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def summarize_replications(
    rep_df: pd.DataFrame,
    cfg: Config,
    seed: int,
) -> Dict[str, float]:
    rss = rep_df["rss_est"].to_numpy(float)
    srs = rep_df["srs_est"].to_numpy(float)
    R = len(rss)

    var_rss = float(np.var(rss, ddof=1))
    var_srs = float(np.var(srs, ddof=1))
    mse_rss = float(np.mean(rss**2))
    mse_srs = float(np.mean(srs**2))

    re_var = var_srs / var_rss if var_rss > 1e-15 else np.nan
    re_mse = mse_srs / mse_rss if mse_rss > 1e-15 else np.nan

    # Monte Carlo SE of the estimated mean rank correlation.
    rho = rep_df["realized_rho"].to_numpy(float)
    rho_mean = float(np.mean(rho))
    rho_se = float(np.std(rho, ddof=1) / math.sqrt(R)) if R > 1 else np.nan

    ci_rng = np.random.default_rng(seed)
    var_lo, var_hi = bootstrap_ratio_ci(
        rss, srs, "var", cfg.bootstrap_reps, ci_rng
    )
    mse_lo, mse_hi = bootstrap_ratio_ci(
        rss, srs, "mse", cfg.bootstrap_reps, ci_rng
    )

    return {
        "reps": R,
        "mean_realized_rho": rho_mean,
        "se_realized_rho": rho_se,
        "RE_Var": re_var,
        "RE_Var_CI_low": var_lo,
        "RE_Var_CI_high": var_hi,
        "RE_MSE": re_mse,
        "RE_MSE_CI_low": mse_lo,
        "RE_MSE_CI_high": mse_hi,
        "bias_RSS": float(np.mean(rss)),
        "bias_SRS": float(np.mean(srs)),
        "sd_RSS": math.sqrt(max(var_rss, 0.0)),
        "sd_SRS": math.sqrt(max(var_srs, 0.0)),
        "MSE_RSS": mse_rss,
        "MSE_SRS": mse_srs,
    }


# ============================================================
# Primary matched-rho experiment
# ============================================================

def run_matched_rho(
    cfg: Config,
    calibration: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    transition_rows = []

    for target in cfg.rho_targets:
        target = float(target)
        for mech in IMPERFECT_MECHANISMS:
            row_cal = calibration[
                (calibration.target_rho == target) &
                (calibration.mechanism == mech)
            ].iloc[0]
            scale = float(row_cal.calibrated_scale)

            for estimator in ESTIMATOR_NAMES:
                rep_rows = []
                total_transition = np.zeros((cfg.m, cfg.m), dtype=np.int64)

                for r in range(cfg.reps):
                    rng = np.random.default_rng(
                        cfg.seed + 1000000 + int(round(1000*target)) * 100
                        + (0 if mech == "gaussian" else 1) * 10
                        + r
                    )
                    out = simulate_dataset(
                        cfg.m, cfg.cycles, mech, scale, estimator, cfg, rng
                    )
                    rep_rows.append({
                        "rss_est": out["rss_est"],
                        "srs_est": out["srs_est"],
                        "realized_rho": out["realized_rho"],
                    })
                    total_transition += out["transition_counts"]

                rep_df = pd.DataFrame(rep_rows)
                summary = summarize_replications(
                    rep_df, cfg,
                    seed=cfg.seed + 500000 + int(round(1000*target))
                )

                rows.append({
                    "experiment": "matched_rho",
                    "m": cfg.m,
                    "cycles": cfg.cycles,
                    "n_measured": cfg.m * cfg.cycles,
                    "target_rho": target,
                    "mechanism": mech,
                    "estimator": estimator,
                    "calibrated_scale": scale,
                    "pilot_expected_rho": float(row_cal.pilot_expected_rho),
                    **summary,
                })

                # The ranking mechanism is independent of the location
                # estimator, so store the transition matrix only once per
                # (target_rho, mechanism).  The old code stored it inside the
                # estimator loop, which produced four duplicate 5x5 blocks.
                if estimator == ESTIMATOR_NAMES[0]:
                    row_totals = total_transition.sum(axis=1, keepdims=True)
                    P = np.divide(
                        total_transition,
                        row_totals,
                        out=np.zeros_like(total_transition, dtype=float),
                        where=row_totals > 0,
                    )

                    # Internal consistency check: each row is a conditional
                    # distribution P(observed rank | true rank).
                    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
                        raise RuntimeError(
                            "Transition-matrix rows do not sum to one."
                        )

                    rho_from_P = rho_from_transition_matrix(P)
                    for i in range(cfg.m):
                        for j in range(cfg.m):
                            transition_rows.append({
                                "m": cfg.m,
                                "target_rho": target,
                                "mechanism": mech,
                                "true_rank": i + 1,
                                "observed_rank": j + 1,
                                "transition_probability": P[i, j],
                                "count": int(total_transition[i, j]),
                                "rho_from_transition_matrix": rho_from_P,
                            })

    matched = pd.DataFrame(rows)
    transitions = pd.DataFrame(transition_rows)

    # Compare the two mechanisms directly at each target/estimator.
    g = matched[matched.mechanism == "gaussian"].copy()
    l = matched[matched.mechanism == "laplace"].copy()
    g = g.rename(columns={
        "mean_realized_rho": "rho_gaussian",
        "RE_Var": "RE_Var_gaussian",
        "RE_MSE": "RE_MSE_gaussian",
    })
    l = l.rename(columns={
        "mean_realized_rho": "rho_laplace",
        "RE_Var": "RE_Var_laplace",
        "RE_MSE": "RE_MSE_laplace",
    })
    summary = g[["m", "cycles", "target_rho", "estimator",
                 "rho_gaussian", "RE_Var_gaussian", "RE_MSE_gaussian"]].merge(
        l[["m", "cycles", "target_rho", "estimator",
           "rho_laplace", "RE_Var_laplace", "RE_MSE_laplace"]],
        on=["m", "cycles", "target_rho", "estimator"],
        how="inner",
    )
    summary["abs_rho_difference"] = (
        summary.rho_gaussian - summary.rho_laplace
    ).abs()
    summary["matched_within_tolerance"] = (
        summary.abs_rho_difference <= cfg.rho_match_tolerance
    )
    summary["abs_RE_Var_difference"] = (
        summary.RE_Var_gaussian - summary.RE_Var_laplace
    ).abs()
    summary["abs_RE_MSE_difference"] = (
        summary.RE_MSE_gaussian - summary.RE_MSE_laplace
    ).abs()
    summary["RE_Var_difference_gaussian_minus_laplace"] = (
        summary.RE_Var_gaussian - summary.RE_Var_laplace
    )
    summary["RE_MSE_difference_gaussian_minus_laplace"] = (
        summary.RE_MSE_gaussian - summary.RE_MSE_laplace
    )

    return matched, summary, transitions


# ============================================================
# Controls: perfect and random ranking
# ============================================================

def run_controls(cfg: Config) -> pd.DataFrame:
    rows = []
    for mech in ("perfect", "random"):
        for estimator in ESTIMATOR_NAMES:
            rep_rows = []
            for r in range(cfg.reps):
                rng = np.random.default_rng(
                    cfg.seed + 2000000 + (0 if mech == "perfect" else 100000) + r
                )
                out = simulate_dataset(
                    cfg.m, cfg.cycles, mech, 0.0, estimator, cfg, rng
                )
                rep_rows.append({
                    "rss_est": out["rss_est"],
                    "srs_est": out["srs_est"],
                    "realized_rho": out["realized_rho"],
                })
            rep_df = pd.DataFrame(rep_rows)
            summary = summarize_replications(
                rep_df, cfg, seed=cfg.seed + 2100000 + (0 if mech == "perfect" else 1)
            )
            rows.append({
                "experiment": "control",
                "m": cfg.m,
                "cycles": cfg.cycles,
                "n_measured": cfg.m * cfg.cycles,
                "mechanism": mech,
                "estimator": estimator,
                **summary,
            })
    return pd.DataFrame(rows)


# ============================================================
# Contamination experiment
# ============================================================

def run_contamination(
    cfg: Config,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    # Use representative ranking qualities.  Perfect and random are controls;
    # Gaussian and Laplace use calibrated scales.
    target_list = list(cfg.rho_targets)
    for target in target_list:
        for mech in ("gaussian", "laplace"):
            row_cal = calibration[
                (calibration.target_rho == target) &
                (calibration.mechanism == mech)
            ].iloc[0]
            scale = float(row_cal.calibrated_scale)

            for mode in ("post_ranking", "pre_ranking"):
                for eps in cfg.eps_values:
                    for estimator in ESTIMATOR_NAMES:
                        rep_rows = []
                        for r in range(cfg.reps):
                            rng = np.random.default_rng(
                                cfg.seed + 3000000
                                + int(round(1000*target))*1000
                                + (0 if mech == "gaussian" else 1)*100
                                + (0 if mode == "post_ranking" else 10)
                                + int(round(1000*eps))*20
                                + r
                            )
                            out = simulate_dataset(
                                cfg.m, cfg.cycles, mech, scale, estimator, cfg,
                                rng, epsilon=eps,
                                contamination_mode=mode,
                            )
                            rep_rows.append({
                                "rss_est": out["rss_est"],
                                "srs_est": out["srs_est"],
                                "realized_rho": out["realized_rho"],
                            })
                        rep_df = pd.DataFrame(rep_rows)
                        summary = summarize_replications(
                            rep_df, cfg,
                            seed=cfg.seed + 3100000 + int(round(1000*eps))
                        )
                        rows.append({
                            "experiment": "contamination",
                            "m": cfg.m,
                            "cycles": cfg.cycles,
                            "n_measured": cfg.m * cfg.cycles,
                            "target_rho": target,
                            "mechanism": mech,
                            "ranking_contamination_mode": mode,
                            "epsilon": eps,
                            "estimator": estimator,
                            "calibrated_scale": scale,
                            **summary,
                        })
    return pd.DataFrame(rows)


# ============================================================
# Optional adaptive experiment
# ============================================================

def pilot_choice(
    cfg: Config,
    mechanism: str,
    scale: float,
    estimator: str,
    epsilon: float,
    mode: str,
    rng: np.random.Generator,
) -> str:
    rss_vals = []
    srs_vals = []
    for _ in range(cfg.pilot_reps):
        out = simulate_dataset(
            cfg.m, cfg.pilot_cycles, mechanism, scale, estimator, cfg, rng,
            epsilon=epsilon, contamination_mode=mode,
        )
        rss_vals.append(out["rss_est"])
        srs_vals.append(out["srs_est"])
    rss_vals = np.asarray(rss_vals)
    srs_vals = np.asarray(srs_vals)
    rss_mse = float(np.mean(rss_vals**2))
    srs_mse = float(np.mean(srs_vals**2))
    return "RSS" if rss_mse <= srs_mse else "SRS"


def run_adaptive(
    cfg: Config,
    calibration: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    # Keep this experiment deliberately smaller than the primary study.
    targets = [t for t in cfg.rho_targets if 0.25 <= t <= 0.90]
    eps_list = [e for e in cfg.eps_values if e in (0.0, 0.10)]
    estimators = ("mean", "trimmed", "huber")

    for target in targets:
        for mech in IMPERFECT_MECHANISMS:
            scale = float(calibration[
                (calibration.target_rho == target) &
                (calibration.mechanism == mech)
            ].iloc[0].calibrated_scale)
            for mode in ("post_ranking", "pre_ranking"):
                for eps in eps_list:
                    for estimator in estimators:
                        selected = []
                        final_err = []
                        final_rss = []
                        final_srs = []

                        for a in range(cfg.adaptive_trials):
                            rng = np.random.default_rng(
                                cfg.seed + 4000000
                                + int(round(1000*target))*10000
                                + (0 if mech == "gaussian" else 1)*1000
                                + (0 if mode == "post_ranking" else 100)
                                + int(round(1000*eps))*10
                                + a
                            )
                            choice = pilot_choice(
                                cfg, mech, scale, estimator, eps, mode, rng
                            )
                            out = simulate_dataset(
                                cfg.m, cfg.cycles, mech, scale, estimator, cfg, rng,
                                epsilon=eps, contamination_mode=mode,
                            )
                            selected.append(choice)
                            final_rss.append(out["rss_est"])
                            final_srs.append(out["srs_est"])
                            final_err.append(
                                out["rss_est"] if choice == "RSS" else out["srs_est"]
                            )

                        selected = np.asarray(selected)
                        final_err = np.asarray(final_err)
                        final_rss = np.asarray(final_rss)
                        final_srs = np.asarray(final_srs)

                        rows.append({
                            "experiment": "adaptive",
                            "m": cfg.m,
                            "cycles": cfg.cycles,
                            "n_measured": cfg.m * cfg.cycles,
                            "target_rho": target,
                            "mechanism": mech,
                            "ranking_contamination_mode": mode,
                            "epsilon": eps,
                            "estimator": estimator,
                            "adaptive_trials": cfg.adaptive_trials,
                            "pilot_reps": cfg.pilot_reps,
                            "pilot_cycles": cfg.pilot_cycles,
                            "RSS_selection_fraction": float(np.mean(selected == "RSS")),
                            "adaptive_MSE": float(np.mean(final_err**2)),
                            "RSS_MSE_reference": float(np.mean(final_rss**2)),
                            "SRS_MSE_reference": float(np.mean(final_srs**2)),
                        })
    return pd.DataFrame(rows)


# ============================================================
# Transition-matrix comparison
# ============================================================

def rho_from_transition_matrix(P: np.ndarray) -> float:
    """Compute Corr(R, R_tilde) implied by a conditional transition matrix.

    The true rank R is uniform on {1,...,m}.  Because each observed ranking
    is a permutation, the observed rank is also uniform.
    """
    P = np.asarray(P, dtype=float)
    m = P.shape[0]
    if P.shape != (m, m):
        raise ValueError("P must be square")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Each row of the transition matrix must sum to 1")

    ranks = np.arange(1, m + 1, dtype=float)
    mean_rank = 0.5 * (m + 1.0)
    var_rank = (m**2 - 1.0) / 12.0
    cross_moment = float(
        np.sum(ranks[:, None] * ranks[None, :] * P) / m
    )
    covariance = cross_moment - mean_rank**2
    return float(covariance / var_rank)


def transition_distance(transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in sorted(transitions.target_rho.unique()):
        sub = transitions[transitions.target_rho == target]
        mats = {}
        for mech in IMPERFECT_MECHANISMS:
            q = sub[sub.mechanism == mech]
            m = int(q.m.iloc[0])
            P = np.zeros((m, m))
            for _, r in q.iterrows():
                P[int(r.true_rank)-1, int(r.observed_rank)-1] = r.transition_probability
            mats[mech] = P
        A = mats["gaussian"]
        B = mats["laplace"]
        rows.append({
            "m": A.shape[0],
            "target_rho": target,
            "Frobenius_distance": float(np.linalg.norm(A-B, ord="fro")),
            "max_entry_difference": float(np.max(np.abs(A-B))),
            "rho_from_gaussian_matrix": rho_from_transition_matrix(A),
            "rho_from_laplace_matrix": rho_from_transition_matrix(B),
        })
    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_matched(summary: pd.DataFrame, outdir: str) -> None:
    if not HAVE_MPL:
        return
    for estimator in ESTIMATOR_NAMES:
        q = summary[summary.estimator == estimator]
        if q.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(q.target_rho, q.RE_Var_gaussian, marker="o", label="Gaussian rank noise")
        ax.plot(q.target_rho, q.RE_Var_laplace, marker="s", label="Laplace rank noise")
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Target rank correlation")
        ax.set_ylabel("RE_Var = Var(SRS) / Var(RSS)")
        ax.set_title(f"Matched-rho RSS efficiency: {estimator}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"matched_RE_Var_{estimator}.png"), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(q.target_rho, q.RE_MSE_gaussian, marker="o", label="Gaussian rank noise")
        ax.plot(q.target_rho, q.RE_MSE_laplace, marker="s", label="Laplace rank noise")
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.set_xlabel("Target rank correlation")
        ax.set_ylabel("RE_MSE = MSE(SRS) / MSE(RSS)")
        ax.set_title(f"Matched-rho RSS MSE efficiency: {estimator}")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"matched_RE_MSE_{estimator}.png"), dpi=180)
        plt.close(fig)


def plot_transition_matrices(transitions: pd.DataFrame, outdir: str) -> None:
    if not HAVE_MPL:
        return
    for target in sorted(transitions.target_rho.unique()):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
        for ax, mech in zip(axes, IMPERFECT_MECHANISMS):
            q = transitions[(transitions.target_rho == target) &
                             (transitions.mechanism == mech)]
            m = int(q.m.iloc[0])
            P = np.zeros((m, m))
            for _, r in q.iterrows():
                P[int(r.true_rank)-1, int(r.observed_rank)-1] = r.transition_probability
            im = ax.imshow(P, origin="lower", vmin=0, vmax=max(0.2, P.max()))
            ax.set_title(mech.capitalize())
            ax.set_xlabel("Observed rank")
            ax.set_ylabel("True rank")
            ax.set_xticks(range(m))
            ax.set_yticks(range(m))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"Transition matrices at target rho={target:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"transition_matrices_rho_{target:.2f}.png"), dpi=180)
        plt.close(fig)


def plot_contamination(df: pd.DataFrame, outdir: str) -> None:
    if not HAVE_MPL or df.empty:
        return
    for estimator in sorted(df.estimator.unique()):
        q = df[(df.estimator == estimator) & (df.mechanism == "gaussian")]
        if q.empty:
            continue
        for mode in sorted(q.ranking_contamination_mode.unique()):
            qq = q[q.ranking_contamination_mode == mode]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for target in sorted(qq.target_rho.unique()):
                z = qq[qq.target_rho == target]
                ax.plot(z.epsilon, z.RE_MSE, marker="o", label=f"rho={target:.2f}")
            ax.axhline(1.0, linestyle="--", linewidth=1)
            ax.set_xlabel("Contamination fraction")
            ax.set_ylabel("RE_MSE")
            ax.set_title(f"Gaussian ranking: {estimator}, {mode}")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(os.path.join(
                outdir, f"contamination_{estimator}_{mode}.png"
            ), dpi=180)
            plt.close(fig)


# ============================================================
# CLI / main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 RSS matched-rho experiment")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--cycles", type=int, default=100)
    p.add_argument("--m", type=int, default=5)
    p.add_argument("--targets", type=float, nargs="+",
                    default=[0.25, 0.50, 0.75, 0.90])
    p.add_argument("--calibration-trials", type=int, default=20000)
    p.add_argument("--calibration-grid-size", type=int, default=120)
    p.add_argument("--rho-match-tolerance", type=float, default=0.02)
    p.add_argument("--bootstrap-reps", type=int, default=400)
    p.add_argument("--eps", type=float, nargs="+",
                    default=[0.0, 0.05, 0.10, 0.20])
    p.add_argument("--outdir", type=str, default="rss_results_v2")
    p.add_argument("--run-adaptive", action="store_true")
    p.add_argument("--adaptive-trials", type=int, default=100)
    p.add_argument("--pilot-reps", type=int, default=30)
    p.add_argument("--pilot-cycles", type=int, default=10)
    p.add_argument("--skip-controls", action="store_true")
    p.add_argument("--skip-contamination", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        reps=args.reps,
        cycles=args.cycles,
        m=args.m,
        rho_targets=tuple(float(x) for x in args.targets),
        calibration_trials=args.calibration_trials,
        calibration_grid_size=args.calibration_grid_size,
        rho_match_tolerance=args.rho_match_tolerance,
        bootstrap_reps=args.bootstrap_reps,
        eps_values=tuple(float(x) for x in args.eps),
        outdir=args.outdir,
        adaptive_trials=args.adaptive_trials,
        pilot_reps=args.pilot_reps,
        pilot_cycles=args.pilot_cycles,
    )

    if cfg.m < 3:
        raise ValueError("m must be at least 3")
    if cfg.reps < 20:
        print("WARNING: reps < 20; Monte Carlo summaries will be unstable.")

    os.makedirs(cfg.outdir, exist_ok=True)

    print("=" * 72)
    print("RSS ADAPTIVE EXPERIMENTS V2")
    print("=" * 72)
    print(f"seed={cfg.seed}, reps={cfg.reps}, cycles={cfg.cycles}, m={cfg.m}")
    print(f"n measured per replication = {cfg.m * cfg.cycles}")
    print(f"rho targets = {cfg.rho_targets}")
    print(f"matching tolerance = {cfg.rho_match_tolerance}")
    print()

    print("[1/5] Calibrating ranking mechanisms...")
    calibration = build_calibration_table(cfg)
    calibration.to_csv(
        os.path.join(cfg.outdir, "calibration_v2.csv"), index=False
    )
    print(calibration.to_string(index=False))
    print()

    print("[2/5] Running matched-rho experiment...")
    matched, matched_summary, transitions = run_matched_rho(cfg, calibration)
    matched.to_csv(
        os.path.join(cfg.outdir, "matched_rho_v2.csv"), index=False
    )
    matched_summary.to_csv(
        os.path.join(cfg.outdir, "matched_rho_summary_v2.csv"), index=False
    )
    transitions.to_csv(
        os.path.join(cfg.outdir, "transition_matrices_v2.csv"), index=False
    )
    distances = transition_distance(transitions)
    distances.to_csv(
        os.path.join(cfg.outdir, "transition_distance_v2.csv"), index=False
    )

    print("\nMatched-rho summary:")
    print(matched_summary[[
        "target_rho", "estimator", "rho_gaussian", "rho_laplace",
        "abs_rho_difference", "matched_within_tolerance",
        "RE_Var_gaussian", "RE_Var_laplace",
        "RE_MSE_gaussian", "RE_MSE_laplace",
    ]].to_string(index=False))
    print()

    if not cfg.rho_targets:
        raise RuntimeError("No rho targets supplied")

    if not args.skip_controls:
        print("[3/5] Running perfect/random controls...")
        controls = run_controls(cfg)
        controls.to_csv(
            os.path.join(cfg.outdir, "controls_v2.csv"), index=False
        )
        print(controls[[
            "mechanism", "estimator", "mean_realized_rho",
            "RE_Var", "RE_MSE",
        ]].to_string(index=False))
    else:
        print("[3/5] Controls skipped.")

    if not args.skip_contamination:
        print("[4/5] Running contamination experiment...")
        contamination = run_contamination(cfg, calibration)
        contamination.to_csv(
            os.path.join(cfg.outdir, "contamination_v2.csv"), index=False
        )
        print(
            contamination.groupby(
                ["target_rho", "mechanism", "ranking_contamination_mode"]
            )["RE_MSE"].mean().reset_index().to_string(index=False)
        )
    else:
        contamination = pd.DataFrame()
        print("[4/5] Contamination experiment skipped.")

    if args.run_adaptive:
        print("[5/5] Running adaptive experiment...")
        adaptive = run_adaptive(cfg, calibration)
        adaptive.to_csv(
            os.path.join(cfg.outdir, "adaptive_v2.csv"), index=False
        )
        print(adaptive.to_string(index=False))
    else:
        print("[5/5] Adaptive experiment skipped. Use --run-adaptive after validating matched-rho results.")

    if not args.skip_plots and HAVE_MPL:
        print("Generating plots...")
        plot_matched(matched_summary, cfg.outdir)
        plot_transition_matrices(transitions, cfg.outdir)
        if not contamination.empty:
            plot_contamination(contamination, cfg.outdir)
    elif not HAVE_MPL:
        print("matplotlib not available; CSV outputs were still written.")

    # A compact machine-readable decision aid.
    matched_ok = matched_summary[
        matched_summary.matched_within_tolerance
    ].copy()
    decision = {
        "m": cfg.m,
        "reps": cfg.reps,
        "cycles": cfg.cycles,
        "rho_match_tolerance": cfg.rho_match_tolerance,
        "n_matched_rows": int(len(matched_ok)),
        "n_total_rows": int(len(matched_summary)),
        "all_rows_matched": bool(len(matched_ok) == len(matched_summary)),
        "mean_abs_rho_difference": float(matched_summary.abs_rho_difference.mean()),
        "max_abs_rho_difference": float(matched_summary.abs_rho_difference.max()),
        "mean_abs_RE_Var_difference": float(matched_summary.abs_RE_Var_difference.mean()),
        "mean_abs_RE_MSE_difference": float(matched_summary.abs_RE_MSE_difference.mean()),
    }
    pd.DataFrame([decision]).to_csv(
        os.path.join(cfg.outdir, "v2_quality_check.csv"), index=False
    )

    print("\n" + "=" * 72)
    print("V2 COMPLETE")
    print("=" * 72)
    print(f"Results: {os.path.abspath(cfg.outdir)}")
    print("Key file: matched_rho_summary_v2.csv")
    print("Quality file: v2_quality_check.csv")
    print("Do NOT transfer numerical claims to the manuscript until the")
    print("matched-rho rows pass the tolerance check and are inspected.")


if __name__ == "__main__":
    main()
