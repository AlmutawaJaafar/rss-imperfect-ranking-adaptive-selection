#!/usr/bin/env python3
"""
Step 1.1 — Final Monte-Carlo uncertainty analysis for the V3 adaptive RSS study.

Expected directory layout:

~/Desktop/LTS_RSS/Tex_gpt/
├── rss_results_v3_adaptive_primary/
│   ├── adaptive_v3.csv
│   ├── adaptive_v3_decisions.csv
│   ├── adaptive_v3_reference_replicates.csv
│   ├── adaptive_v3_reference_risks.csv
│   └── calibration_v3.csv
└── rss_results_v3_paired_primary/
    ├── calibration_v3.csv
    ├── matched_rho_paired_replicates_v3.csv
    └── paired_delta_summary_v3.csv

The script uses the adaptive-primary directory for Step 1.1.
The paired-primary directory is accepted as an optional argument and is
reported for traceability; it is not required for the Step 1.1 calculations.

Run from ~/Desktop/LTS_RSS/Tex_gpt:

    python3.10 analyze_adaptive_v3_final.py

or explicitly:

    python3.10 analyze_adaptive_v3_final.py \
        --adaptive-dir ./rss_results_v3_adaptive_primary \
        --paired-dir ./rss_results_v3_paired_primary \
        --bootstrap 10000 \
        --seed 20260915

Outputs are written to:

    ./rss_results_v3_adaptive_primary/step1_1_uncertainty/
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm
except ImportError as exc:
    raise SystemExit(
        "ERROR: scipy is required.\n"
        "Install it with:\n"
        "  python3.10 -m pip install numpy pandas scipy"
    ) from exc


KEYS = [
    "target_rho",
    "mechanism",
    "ranking_contamination_mode",
    "epsilon",
    "estimator",
]

REQUIRED_DECISION_COLUMNS = KEYS + [
    "adaptive_trials",
    "reference_reps",
    "RSS_selection_fraction",
    "adaptive_variance",
    "adaptive_MSE",
    "RSS_variance_reference",
    "SRS_variance_reference",
    "RSS_MSE_reference",
    "SRS_MSE_reference",
    "oracle_best_variance",
    "oracle_best_MSE",
    "adaptive_excess_variance_vs_oracle",
    "adaptive_excess_MSE_vs_oracle",
    "adaptive_relative_oracle_regret_percent_variance",
    "adaptive_relative_oracle_regret_percent_MSE",
    "oracle_method_variance",
    "oracle_method_MSE",
    "decision_accuracy_vs_variance_oracle",
    "decision_accuracy_vs_MSE_oracle",
]

REQUIRED_REFERENCE_COLUMNS = KEYS + [
    "replicate",
    "rss_est",
    "srs_est",
]


def wilson_interval(k: int, n: int, alpha: float = 0.05):
    """Wilson confidence interval for a binomial proportion."""
    if n <= 0:
        return np.nan, np.nan

    z = norm.ppf(1.0 - alpha / 2.0)
    p = k / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = (
        z
        * math.sqrt(
            p * (1.0 - p) / n
            + z * z / (4.0 * n * n)
        )
        / denominator
    )

    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_statistic(
    x: np.ndarray,
    statistic,
    B: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
):
    """
    Nonparametric bootstrap CI for a scalar statistic.

    Chunked implementation prevents excessive memory usage.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < 2:
        return np.nan, np.nan

    values = np.empty(B, dtype=float)
    chunk = min(500, B)

    position = 0
    while position < B:
        b = min(chunk, B - position)
        indices = rng.integers(0, n, size=(b, n))
        for j in range(b):
            values[position + j] = statistic(x[indices[j]])
        position += b

    return tuple(np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0]))


def bootstrap_paired_ratio(
    srs: np.ndarray,
    rss: np.ndarray,
    B: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
):
    """
    Paired nonparametric bootstrap CI for

        RE = Var(SRS) / Var(RSS).

    The RSS and SRS estimates from the same reference replicate remain
    paired during resampling.
    """
    srs = np.asarray(srs, dtype=float)
    rss = np.asarray(rss, dtype=float)

    mask = np.isfinite(srs) & np.isfinite(rss)
    srs = srs[mask]
    rss = rss[mask]

    n = len(srs)
    if n < 2:
        return np.nan, np.nan

    values = np.empty(B, dtype=float)
    chunk = min(500, B)

    position = 0
    while position < B:
        b = min(chunk, B - position)
        indices = rng.integers(0, n, size=(b, n))

        for j in range(b):
            a = indices[j]
            numerator = np.var(srs[a], ddof=1)
            denominator = np.var(rss[a], ddof=1)
            values[position + j] = (
                numerator / denominator
                if denominator > 0
                else np.nan
            )

        position += b

    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan

    return tuple(np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0]))


def variance_stat(x):
    return np.var(x, ddof=1)


def mse_stat(x):
    """
    In this simulation the target parameter is zero, so MSE is E[estimate^2].
    """
    return np.mean(np.square(x))


def fmt(value, digits=6):
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def find_trial_file(adaptive_dir: Path):
    """
    Search for a raw adaptive trial-level file.

    The current V3 directory supplied by the user contains aggregate
    adaptive_v3.csv/adaptive_v3_decisions.csv and reference replicate files.
    These candidate names are included so the script automatically uses
    trial-level data if such a file is later supplied.
    """
    candidates = [
        "adaptive_v3_trial_results.csv",
        "adaptive_v3_trials.csv",
        "adaptive_v3_trial_replicates.csv",
        "adaptive_v3_adaptive_trials.csv",
        "adaptive_v3_trial_level.csv",
    ]

    for filename in candidates:
        candidate = adaptive_dir / filename
        if candidate.exists():
            return candidate

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Step 1.1 uncertainty analysis for V3 adaptive RSS results."
    )

    parser.add_argument(
        "--adaptive-dir",
        type=Path,
        default=Path.home()
        / "Desktop"
        / "LTS_RSS"
        / "Tex_gpt"
        / "rss_results_v3_adaptive_primary",
        help="Directory containing the adaptive V3 CSV files.",
    )

    parser.add_argument(
        "--paired-dir",
        type=Path,
        default=Path.home()
        / "Desktop"
        / "LTS_RSS"
        / "Tex_gpt"
        / "rss_results_v3_paired_primary",
        help="Directory containing the paired V3 CSV files.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Number of bootstrap replicates (default: 10000).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260915,
        help="Random seed for bootstrap resampling.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided confidence level parameter; default 0.05 gives 95%% CIs.",
    )

    args = parser.parse_args()

    if args.bootstrap < 1000:
        raise SystemExit(
            "ERROR: --bootstrap should be at least 1000 for the final analysis."
        )

    adaptive_dir = args.adaptive_dir.expanduser().resolve()
    paired_dir = args.paired_dir.expanduser().resolve()

    decision_path = adaptive_dir / "adaptive_v3_decisions.csv"
    reference_path = adaptive_dir / "adaptive_v3_reference_replicates.csv"
    adaptive_summary_path = adaptive_dir / "adaptive_v3.csv"
    reference_risks_path = adaptive_dir / "adaptive_v3_reference_risks.csv"
    calibration_path = adaptive_dir / "calibration_v3.csv"

    if not adaptive_dir.exists():
        raise SystemExit(f"ERROR: adaptive directory does not exist:\n{adaptive_dir}")

    for path in [decision_path, reference_path]:
        if not path.exists():
            raise SystemExit(f"ERROR: required file not found:\n{path}")

    output_dir = adaptive_dir / "step1_1_uncertainty"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 1.1 — V3 ADAPTIVE MONTE-CARLO UNCERTAINTY ANALYSIS")
    print("=" * 78)
    print(f"Adaptive directory : {adaptive_dir}")
    print(f"Paired directory   : {paired_dir}")
    print(f"Bootstrap replicates: {args.bootstrap}")
    print(f"Bootstrap seed      : {args.seed}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    decisions = pd.read_csv(decision_path)
    reference = pd.read_csv(reference_path)

    missing_decision = [
        c for c in REQUIRED_DECISION_COLUMNS
        if c not in decisions.columns
    ]

    missing_reference = [
        c for c in REQUIRED_REFERENCE_COLUMNS
        if c not in reference.columns
    ]

    if missing_decision:
        raise SystemExit(
            "ERROR: adaptive_v3_decisions.csv is missing columns:\n  "
            + "\n  ".join(missing_decision)
        )

    if missing_reference:
        raise SystemExit(
            "ERROR: adaptive_v3_reference_replicates.csv is missing columns:\n  "
            + "\n  ".join(missing_reference)
        )

    if adaptive_summary_path.exists():
        adaptive_summary = pd.read_csv(adaptive_summary_path)
    else:
        adaptive_summary = None

    if reference_risks_path.exists():
        reference_risks = pd.read_csv(reference_risks_path)
    else:
        reference_risks = None

    if calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
    else:
        calibration = None

    trial_path = find_trial_file(adaptive_dir)

    # ------------------------------------------------------------------
    # Check the V3 condition structure
    # ------------------------------------------------------------------

    if len(decisions) != 96:
        print(
            f"WARNING: decision file has {len(decisions)} rows; "
            f"expected 96 V3 conditions."
        )

    if len(reference) != 192000:
        print(
            f"WARNING: reference file has {len(reference)} rows; "
            f"expected 192000 rows for 96 conditions x 2000 reps."
        )

    rng = np.random.default_rng(args.seed)
    rows = []

    # ------------------------------------------------------------------
    # Condition-by-condition uncertainty
    # ------------------------------------------------------------------

    for _, decision_row in decisions.iterrows():

        key_values = {
            key: decision_row[key]
            for key in KEYS
        }

        mask = np.ones(len(reference), dtype=bool)

        for key, value in key_values.items():
            mask &= reference[key].eq(value)

        ref = reference.loc[mask]

        n_reference = len(ref)
        expected_reference = int(decision_row["reference_reps"])

        rss = ref["rss_est"].to_numpy(dtype=float)
        srs = ref["srs_est"].to_numpy(dtype=float)

        rss = rss[np.isfinite(rss)]
        srs = srs[np.isfinite(srs)]

        # Reference variance / MSE point estimates.
        rss_variance = np.var(rss, ddof=1)
        srs_variance = np.var(srs, ddof=1)

        rss_mse = np.mean(np.square(rss))
        srs_mse = np.mean(np.square(srs))

        re = srs_variance / rss_variance

        # Bootstrap reference uncertainty.
        rss_var_lo, rss_var_hi = bootstrap_statistic(
            rss,
            variance_stat,
            args.bootstrap,
            rng,
            args.alpha,
        )

        srs_var_lo, srs_var_hi = bootstrap_statistic(
            srs,
            variance_stat,
            args.bootstrap,
            rng,
            args.alpha,
        )

        rss_mse_lo, rss_mse_hi = bootstrap_statistic(
            rss,
            mse_stat,
            args.bootstrap,
            rng,
            args.alpha,
        )

        srs_mse_lo, srs_mse_hi = bootstrap_statistic(
            srs,
            mse_stat,
            args.bootstrap,
            rng,
            args.alpha,
        )

        re_lo, re_hi = bootstrap_paired_ratio(
            srs,
            rss,
            args.bootstrap,
            rng,
            args.alpha,
        )

        # Adaptive decision proportions.
        adaptive_trials = int(decision_row["adaptive_trials"])

        selection_fraction = float(
            decision_row["RSS_selection_fraction"]
        )

        selection_count = int(
            round(selection_fraction * adaptive_trials)
        )

        selection_lo, selection_hi = wilson_interval(
            selection_count,
            adaptive_trials,
            args.alpha,
        )

        agreement_fraction = float(
            decision_row["decision_accuracy_vs_variance_oracle"]
        )

        agreement_count = int(
            round(agreement_fraction * adaptive_trials)
        )

        agreement_lo, agreement_hi = wilson_interval(
            agreement_count,
            adaptive_trials,
            args.alpha,
        )

        # Adaptive risk CIs.
        #
        # These cannot legitimately be derived from the single aggregate
        # adaptive_variance/adaptive_MSE numbers. If raw adaptive trials
        # exist, use them. Otherwise leave them NA.
        adaptive_variance_lo = np.nan
        adaptive_variance_hi = np.nan
        adaptive_mse_lo = np.nan
        adaptive_mse_hi = np.nan

        adaptive_ci_status = (
            "NOT AVAILABLE: no adaptive trial-level file supplied"
        )

        if trial_path is not None:
            trial = pd.read_csv(trial_path)

            trial_mask = np.ones(len(trial), dtype=bool)

            for key, value in key_values.items():
                if key not in trial.columns:
                    trial_mask[:] = False
                    break
                trial_mask &= trial[key].eq(value)

            condition_trials = trial.loc[trial_mask]

            estimate_candidates = [
                "selected_estimate",
                "adaptive_estimate",
                "estimate",
                "selected_mean",
                "adaptive_mean",
            ]

            estimate_column = next(
                (
                    c for c in estimate_candidates
                    if c in condition_trials.columns
                ),
                None,
            )

            if estimate_column is not None:
                estimates = pd.to_numeric(
                    condition_trials[estimate_column],
                    errors="coerce",
                ).dropna().to_numpy(dtype=float)

                if len(estimates) >= 2:
                    adaptive_variance_lo, adaptive_variance_hi = (
                        bootstrap_statistic(
                            estimates,
                            variance_stat,
                            args.bootstrap,
                            rng,
                            args.alpha,
                        )
                    )

                    adaptive_mse_lo, adaptive_mse_hi = (
                        bootstrap_statistic(
                            estimates,
                            mse_stat,
                            args.bootstrap,
                            rng,
                            args.alpha,
                        )
                    )

                    adaptive_ci_status = (
                        f"COMPUTED from {len(estimates)} "
                        f"adaptive trial-level estimates"
                    )

        rows.append(
            {
                **key_values,

                "adaptive_trials": adaptive_trials,
                "reference_reps_expected": expected_reference,
                "reference_reps_observed": n_reference,

                "RSS_selection_fraction": selection_fraction,
                "RSS_selection_CI95_low": selection_lo,
                "RSS_selection_CI95_high": selection_hi,

                "oracle_agreement_fraction": agreement_fraction,
                "oracle_agreement_CI95_low": agreement_lo,
                "oracle_agreement_CI95_high": agreement_hi,

                "adaptive_variance_point": float(
                    decision_row["adaptive_variance"]
                ),
                "adaptive_variance_CI95_low": adaptive_variance_lo,
                "adaptive_variance_CI95_high": adaptive_variance_hi,

                "adaptive_MSE_point": float(
                    decision_row["adaptive_MSE"]
                ),
                "adaptive_MSE_CI95_low": adaptive_mse_lo,
                "adaptive_MSE_CI95_high": adaptive_mse_hi,

                "RSS_variance_reference": rss_variance,
                "RSS_variance_CI95_low": rss_var_lo,
                "RSS_variance_CI95_high": rss_var_hi,

                "SRS_variance_reference": srs_variance,
                "SRS_variance_CI95_low": srs_var_lo,
                "SRS_variance_CI95_high": srs_var_hi,

                "RSS_MSE_reference": rss_mse,
                "RSS_MSE_CI95_low": rss_mse_lo,
                "RSS_MSE_CI95_high": rss_mse_hi,

                "SRS_MSE_reference": srs_mse,
                "SRS_MSE_CI95_low": srs_mse_lo,
                "SRS_MSE_CI95_high": srs_mse_hi,

                "RE_reference": re,
                "RE_CI95_low": re_lo,
                "RE_CI95_high": re_hi,

                "relative_regret_variance_point_percent": float(
                    decision_row[
                        "adaptive_relative_oracle_regret_percent_variance"
                    ]
                ),

                "relative_regret_MSE_point_percent": float(
                    decision_row[
                        "adaptive_relative_oracle_regret_percent_MSE"
                    ]
                ),

                "adaptive_CI_status": adaptive_ci_status,
            }
        )

    result = pd.DataFrame(rows)

    condition_output = (
        output_dir / "adaptive_v3_uncertainty_by_condition.csv"
    )
    result.to_csv(condition_output, index=False)

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------

    summary_row = {
        "n_conditions": len(result),

        "mean_RSS_selection_fraction":
            result["RSS_selection_fraction"].mean(),

        "median_RSS_selection_fraction":
            result["RSS_selection_fraction"].median(),

        "mean_oracle_agreement_fraction":
            result["oracle_agreement_fraction"].mean(),

        "mean_reference_RE":
            result["RE_reference"].mean(),

        "median_reference_RE":
            result["RE_reference"].median(),

        "minimum_reference_RE":
            result["RE_reference"].min(),

        "maximum_reference_RE":
            result["RE_reference"].max(),

        "conditions_with_adaptive_variance_CI":
            result["adaptive_variance_CI95_low"].notna().sum(),

        "conditions_with_adaptive_MSE_CI":
            result["adaptive_MSE_CI95_low"].notna().sum(),

        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
    }

    summary = pd.DataFrame([summary_row])

    summary_output = (
        output_dir / "adaptive_v3_uncertainty_summary.csv"
    )
    summary.to_csv(summary_output, index=False)

    # ------------------------------------------------------------------
    # Human-readable report
    # ------------------------------------------------------------------

    report_lines = []

    report_lines.append(
        "STEP 1.1 — V3 ADAPTIVE MONTE-CARLO UNCERTAINTY ANALYSIS"
    )
    report_lines.append("=" * 78)
    report_lines.append("")
    report_lines.append(f"Adaptive directory: {adaptive_dir}")
    report_lines.append(f"Paired directory:   {paired_dir}")
    report_lines.append(f"Bootstrap B:        {args.bootstrap}")
    report_lines.append(f"Bootstrap seed:     {args.seed}")
    report_lines.append("Confidence level:   95%")
    report_lines.append("")

    report_lines.append("INPUT FILES")
    report_lines.append("-" * 78)
    report_lines.append(f"Decision file:      {decision_path}")
    report_lines.append(f"Reference file:     {reference_path}")
    report_lines.append(
        f"Adaptive summary:   "
        f"{adaptive_summary_path if adaptive_summary_path.exists() else 'not found'}"
    )
    report_lines.append(
        f"Reference risks:    "
        f"{reference_risks_path if reference_risks_path.exists() else 'not found'}"
    )
    report_lines.append(
        f"Calibration:        "
        f"{calibration_path if calibration_path.exists() else 'not found'}"
    )
    report_lines.append(
        f"Adaptive trial file:"
        f" {trial_path if trial_path is not None else 'not found'}"
    )
    report_lines.append("")

    report_lines.append("DESIGN CHECK")
    report_lines.append("-" * 78)
    report_lines.append(
        f"Decision conditions: {len(decisions)} "
        f"(expected 96)"
    )
    report_lines.append(
        f"Reference rows:      {len(reference)} "
        f"(expected 192000)"
    )
    report_lines.append("")

    report_lines.append("OVERALL RESULTS")
    report_lines.append("-" * 78)
    report_lines.append(
        f"Mean RSS selection fraction: "
        f"{fmt(summary_row['mean_RSS_selection_fraction'])}"
    )
    report_lines.append(
        f"Median RSS selection fraction: "
        f"{fmt(summary_row['median_RSS_selection_fraction'])}"
    )
    report_lines.append(
        f"Mean oracle agreement: "
        f"{fmt(summary_row['mean_oracle_agreement_fraction'])}"
    )
    report_lines.append(
        f"Mean reference RE: "
        f"{fmt(summary_row['mean_reference_RE'])}"
    )
    report_lines.append(
        f"Median reference RE: "
        f"{fmt(summary_row['median_reference_RE'])}"
    )
    report_lines.append(
        f"Reference RE range: "
        f"{fmt(summary_row['minimum_reference_RE'])} -- "
        f"{fmt(summary_row['maximum_reference_RE'])}"
    )
    report_lines.append("")

    report_lines.append("UNCERTAINTY METHOD")
    report_lines.append("-" * 78)
    report_lines.append(
        "RSS/SRS reference variance and MSE: "
        "nonparametric bootstrap over the 2,000 reference replicates "
        "within each condition."
    )
    report_lines.append(
        "Reference RE: paired nonparametric bootstrap, preserving the "
        "RSS/SRS pairing within each reference replicate."
    )
    report_lines.append(
        "RSS selection fraction: Wilson 95% interval from the 200 "
        "adaptive decisions."
    )
    report_lines.append(
        "Variance-oracle agreement: Wilson 95% interval from the 200 "
        "adaptive decisions."
    )
    report_lines.append("")

    report_lines.append("ADAPTIVE RISK LIMITATION")
    report_lines.append("-" * 78)

    if trial_path is None:
        report_lines.append(
            "The supplied V3 directory does not contain a recognized "
            "adaptive trial-level CSV."
        )
        report_lines.append(
            "Therefore no CI has been fabricated for adaptive variance, "
            "adaptive MSE, or relative regret."
        )
        report_lines.append(
            "The aggregate adaptive_v3_decisions.csv values are retained "
            "as point estimates only."
        )
        report_lines.append(
            "To obtain those CIs, the 200 adaptive trial-level estimates "
            "must be saved by the experiment runner."
        )
    else:
        report_lines.append(
            f"Trial-level file detected: {trial_path.name}"
        )
        report_lines.append(
            f"Conditions with adaptive variance CI: "
            f"{summary_row['conditions_with_adaptive_variance_CI']}"
        )
        report_lines.append(
            f"Conditions with adaptive MSE CI: "
            f"{summary_row['conditions_with_adaptive_MSE_CI']}"
        )

    report_lines.append("")
    report_lines.append("OUTPUT FILES")
    report_lines.append("-" * 78)
    report_lines.append(str(condition_output))
    report_lines.append(str(summary_output))
    report_lines.append(
        str(output_dir / "adaptive_v3_uncertainty_report.txt")
    )

    report_path = (
        output_dir / "adaptive_v3_uncertainty_report.txt"
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(report_lines))
    print("")
    print("DONE.")


if __name__ == "__main__":
    main()
