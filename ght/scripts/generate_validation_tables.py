#!/usr/bin/env python3
"""
Machine-readable validation tables.

Each table addresses a specific gap identified in review of the previous
archive:

  exponent_validation.csv        fixed-k vs stability-selected Hill estimates
  threshold_stability.csv        the full Hill plot for every (c, m) pair
  robustness_raw.csv             ALL 30 per-realization estimates per family
  robustness_summary.csv         means and central 95% ranges
  nu_validation.csv              direct test of the local probability exponent
  manifold_validation.csv        embedded sphere/circle/point codimension test
  cutoff_scaling_monte_carlo.csv MEASURED cutoffs and regressed slopes
  exact_gaussian_moment_benchmark.csv  closed-form moments, Eq. (A5)
  model_comparison.csv           Pareto / truncated Pareto / lognormal /
                                 stretched exponential / q-Gaussian by AIC
  seed_sensitivity.csv           spread of the exponent across master seeds
"""

from __future__ import annotations

import csv
import sys
from math import gamma as gamma_fn
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geoheavytails.config import (  # noqa: E402
    N_MAIN, N_RESOLUTION, N_REALIZATIONS, K_HILL, M_VALUES, C_VALUES,
    NU_VALUES, C_VALUES_NU, M_VALUES_NU, R0_MANIFOLD,
)
from geoheavytails.core import (  # noqa: E402
    spawn_rng, regularized_map, hill_alpha, hill_curve, select_k_stability,
    empirical_cutoff, loglog_slope,
)
from geoheavytails.inputs import (  # noqa: E402
    INPUT_FAMILIES, sample_radius_nu, manifold_distances,
)
from geoheavytails.models import fit_all_models, MODEL_NAMES  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "data"
DATA.mkdir(exist_ok=True)


def _write(name, header, rows):
    with open(DATA / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {name}  ({len(rows)} rows)")


# --------------------------------------------------------------------------

def exponent_and_threshold_tables():
    """Fixed-k and adaptive Hill estimates, plus the complete Hill plots.

    The previous archive reported only a single fixed k = 4000 for every
    (c, m), and its own CSV showed a systematic -0.135 bias at alpha = 7 with
    no diagnostic to explain it.  Recording the whole threshold sweep makes the
    bias measurable and lets the adaptive estimate be compared against it.
    """
    rows, sweep_rows = [], []
    for c in C_VALUES:
        rng = spawn_rng("table_exponent", realization=c)
        x = rng.normal(size=(N_MAIN, c))
        r = np.linalg.norm(x, axis=1)
        for m in M_VALUES:
            y = regularized_map(r, m=m, lam=0.0)
            th = 1.0 + c / m
            fixed = hill_alpha(y, K_HILL)
            ks, alphas = hill_curve(y)
            k_ad, ad = select_k_stability(ks, alphas)
            rows.append([c, m, th, fixed, fixed - th, ad, ad - th, k_ad])
            for k_i, a_i in zip(ks, alphas):
                sweep_rows.append([c, m, th, int(k_i), float(a_i), a_i - th])

    _write("exponent_validation.csv",
           ["codimension_c", "singularity_order_m", "alpha_theory",
            "alpha_hill_fixed_k", "bias_fixed_k", "alpha_hill_adaptive",
            "bias_adaptive", "k_adaptive"],
           rows)
    _write("threshold_stability.csv",
           ["codimension_c", "singularity_order_m", "alpha_theory", "k",
            "alpha_hill", "bias"],
           sweep_rows)
    return rows


def robustness_tables():
    """30 independent realizations per input family, raw estimates exported.

    Reproduces the claim of Sec. VI.C that every central 95% simulation range
    contains alpha_th = 2 and that the largest relative deviation of a mean
    estimate is below 0.3%.
    """
    raw, summary = [], []
    for i, (name, sampler) in enumerate(INPUT_FAMILIES.items()):
        ests = []
        for rep in range(N_REALIZATIONS):
            rng = spawn_rng("table_robustness", realization=1000 * i + rep)
            x = sampler(rng, N_MAIN)
            y = regularized_map(np.abs(x), m=1.0, lam=0.0)
            a = hill_alpha(y, K_HILL)
            ests.append(a)
            raw.append([name, rep, N_MAIN, K_HILL, 2.0, a, a - 2.0])
        ests = np.array(ests)
        lo, hi = np.quantile(ests, [0.025, 0.975])
        summary.append([
            name, N_REALIZATIONS, float(ests.mean()), float(ests.std(ddof=1)),
            float(lo), float(hi), bool(lo <= 2.0 <= hi),
            100.0 * abs(ests.mean() - 2.0) / 2.0,
        ])

    _write("robustness_raw.csv",
           ["input_family", "realization", "N", "k", "alpha_theory",
            "alpha_hill", "difference"], raw)
    _write("robustness_summary.csv",
           ["input_family", "n_realizations", "mean_alpha", "sd_alpha",
            "ci95_low", "ci95_high", "contains_theory",
            "relative_deviation_percent"], summary)

    means = np.array([s[2] for s in summary])
    print(f"    mean estimates span [{means.min():.4f}, {means.max():.4f}]; "
          f"max relative deviation {100*np.max(np.abs(means-2))/2:.3f}%")
    return summary


def nu_table():
    """Direct test of the local probability exponent, Eqs. (51)-(55)."""
    rows = []
    idx = 0
    for c in C_VALUES_NU:
        for m in M_VALUES_NU:
            for nu in NU_VALUES:
                rng = spawn_rng("table_exponent", realization=500 + idx)
                idx += 1
                r = sample_radius_nu(rng, N_MAIN, c, nu)
                y = regularized_map(r, m=m, lam=0.0)
                th = 1.0 + (c + nu) / m
                a = hill_alpha(y, K_HILL)
                rows.append([c, m, nu, c + nu, th, a, a - th])
    _write("nu_validation.csv",
           ["c", "m", "nu", "c_plus_nu", "alpha_theory", "alpha_hill",
            "difference"], rows)
    return rows


def manifold_table():
    """Embedded sphere / circle / point test, Eqs. (56)-(61)."""
    rows = []
    for rep in range(8):
        rng = spawn_rng("table_exponent", realization=800 + rep)
        for name, (r, c) in manifold_distances(rng, N_MAIN, R0_MANIFOLD).items():
            y = regularized_map(r, m=1.0, lam=0.0)
            th = 1.0 + c
            a = hill_alpha(y, K_HILL)
            rows.append([name, c, rep, th, a, a - th])
    _write("manifold_validation.csv",
           ["manifold", "codimension_c", "realization", "alpha_theory",
            "alpha_hill", "difference"], rows)
    return rows


def cutoff_monte_carlo():
    """GENUINE Monte Carlo test of Eq. (29).

    For each (m, lambda) the cutoff is measured from a simulated sample as a
    high empirical quantile; log(Y_max) is then regressed on log(lambda) and
    the slope compared with -m.  Nothing here evaluates lambda**(-m) to
    produce the data being tested -- that circularity was the defect of the
    previous archive's `cutoff_scaling()`.
    """
    rng = spawn_rng("table_cutoff")
    r = np.abs(rng.normal(size=N_RESOLUTION))
    lam_grid = np.geomspace(3e-3, 0.3, 9)

    rows, fit_rows = [], []
    for m in M_VALUES:
        measured = []
        for lam in lam_grid:
            y = regularized_map(r, m=m, lam=lam)
            ym_meas = empirical_cutoff(y, 0.9999)
            ym_th = lam ** (-m)
            measured.append(ym_meas)
            rows.append([m, lam, ym_meas, ym_th, ym_meas / ym_th])
        slope, intercept, se = loglog_slope(lam_grid, np.array(measured))
        fit_rows.append([m, slope, se, -m, slope - (-m)])

    _write("cutoff_scaling_monte_carlo.csv",
           ["m", "lambda", "Ymax_measured", "Ymax_theory_A1",
            "measured_over_theory"], rows)
    _write("cutoff_scaling_fits.csv",
           ["m", "fitted_slope", "slope_stderr", "expected_slope",
            "difference"], fit_rows)
    for f in fit_rows:
        print(f"    m = {f[0]}: fitted slope {f[1]:+.4f} +/- {f[2]:.4f} "
              f"(expected {f[3]:+.1f})")
    return fit_rows


def moment_benchmark():
    """Exact isotropic-Gaussian moments, Eq. (A5), with a Monte Carlo check."""
    rows = []
    for c in C_VALUES:
        rng = spawn_rng("table_exponent", realization=900 + c)
        x = rng.normal(size=(N_MAIN, c))
        r = np.linalg.norm(x, axis=1)
        for m in (0.35, 0.5, 0.75, 1.0, 1.5, 2.0):
            y = regularized_map(r, m=m, lam=0.0)
            for s in (0.5, 1.0, 2.0):
                finite = (m * s < c)
                exact = (2 ** (-m * s / 2) * gamma_fn((c - m * s) / 2)
                         / gamma_fn(c / 2)) if finite else np.inf
                mc = float(np.mean(y ** s))
                rows.append([c, m, s, "finite" if finite else "diverges",
                             exact if finite else "inf", mc])
    _write("exact_gaussian_moment_benchmark.csv",
           ["c", "m", "moment_order_s", "regime", "exact_moment",
            "monte_carlo_moment"], rows)
    return rows


def model_comparison():
    """Likelihood comparison against the alternatives the manuscript names.

    Two samples are compared: the bare singular map (an untruncated power law)
    and the regularized map (a truncated one).  A correct analysis must prefer
    Pareto for the first and truncated Pareto for the second; if it does not,
    the paper's own diagnostic programme fails on its own data.
    """
    rng = spawn_rng("table_modelcomp")
    x = rng.normal(size=N_MAIN)

    cases = {
        "bare m=c=1": regularized_map(np.abs(x), m=1.0, lam=0.0),
        "regularized lambda=1e-2": regularized_map(np.abs(x), m=1.0, lam=1e-2),
        "regularized lambda=1e-3": regularized_map(np.abs(x), m=1.0, lam=1e-3),
    }

    rows = []
    for case, y in cases.items():
        ys = np.sort(y[np.isfinite(y) & (y > 0)])
        xmin = float(ys[-K_HILL - 1])
        fits, best, n_tail = fit_all_models(ys, xmin, subsample=3000, rng=rng)
        for name in MODEL_NAMES:
            f = fits[name]
            rows.append([
                case, xmin, n_tail, name, f.get("loglik", np.nan),
                f.get("k", np.nan), f.get("aic", np.nan),
                f.get("delta_aic", np.nan), name == best,
                "; ".join(f"{k}={v:.4g}" for k, v in f["params"].items()),
            ])
        print(f"    {case}: AIC-preferred model = {best}")
    _write("model_comparison.csv",
           ["case", "xmin", "n_tail_points", "model", "loglik", "n_params",
            "aic", "delta_aic", "preferred", "fitted_params"], rows)
    return rows


def seed_sensitivity():
    """Spread of the estimated exponent across independent master streams.

    The previous archive generated every figure from one fixed seed with no
    uncertainty quantification at all.
    """
    rows = []
    for c in C_VALUES:
        for m in M_VALUES:
            ests = []
            for rep in range(12):
                rng = spawn_rng("table_seed_sensitivity",
                                realization=100 * c + 10 * int(2 * m) + rep)
                xx = rng.normal(size=(N_MAIN, c))
                y = regularized_map(np.linalg.norm(xx, axis=1), m=m, lam=0.0)
                ests.append(hill_alpha(y, K_HILL))
            ests = np.array(ests)
            rows.append([c, m, 1.0 + c / m, float(ests.mean()),
                         float(ests.std(ddof=1)), float(ests.min()),
                         float(ests.max())])
    _write("seed_sensitivity.csv",
           ["c", "m", "alpha_theory", "mean_alpha", "sd_alpha", "min_alpha",
            "max_alpha"], rows)
    return rows


def main():
    print("Exponent and threshold-stability tables...")
    exponent_and_threshold_tables()
    print("Global-input robustness (30 realizations per family)...")
    robustness_tables()
    print("Local probability exponent nu...")
    nu_table()
    print("Embedded singular manifolds...")
    manifold_table()
    print("Monte Carlo cutoff scaling...")
    cutoff_monte_carlo()
    print("Exact moment benchmark...")
    moment_benchmark()
    print("Competing-model likelihood comparison...")
    model_comparison()
    print("Seed sensitivity...")
    seed_sensitivity()
    print(f"\nAll tables written to {DATA.resolve()}")


if __name__ == "__main__":
    main()
