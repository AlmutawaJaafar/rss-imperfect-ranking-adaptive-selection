#!/usr/bin/env python3
"""
verify_point0.py

Point-0 reproducibility/freeze audit for the primary V3 adaptive RSS study.

This script DOES NOT rerun the Monte Carlo experiment.
It audits the already-generated CSV results, recomputes the headline
aggregates reported in the manuscript, checks internal consistency between
adaptive and fixed-strategy reference files, checks the paired V3 summary,
and records SHA-256 hashes of the primary result files.

Usage on the user's machine, for example:
    python3.10 verify_point0.py \
        --outdir "/home/jaafar/Desktop/LTS_RSS/Tex_gpt/rss_results_v3_adaptive_primary" \
        --paired-dir "/home/jaafar/Desktop/LTS_RSS/Tex_gpt/rss_results_v3_paired_primary"

The script exits with code 0 only when all required checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    "m": 5,
    "cycles": 100,
    "n_measured": 500,
    "adaptive_trials": 200,
    "pilot_reps": 30,
    "pilot_cycles": 20,
    "reference_reps": 2000,
    "calibration_trials": 20000,
    "seed": 20260915,
}

TARGET_RHOS = (0.25, 0.50, 0.75, 0.90)
MECHANISMS = ("gaussian", "laplace")
MODES = ("post_ranking", "pre_ranking")
EPS_VALUES = (0.0, 0.10)
ESTIMATORS = ("mean", "trimmed", "huber")

COND_COLS = [
    "target_rho",
    "mechanism",
    "ranking_contamination_mode",
    "epsilon",
    "estimator",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def assert_close(name: str, actual: float, expected: float, atol=1e-10, rtol=1e-8) -> None:
    if not math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol):
        fail(f"{name}: actual={actual:.15g}, expected={expected:.15g}")


def assert_equal(name: str, actual, expected) -> None:
    if actual != expected:
        fail(f"{name}: actual={actual!r}, expected={expected!r}")


def condition_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["target_rho"].round(10).astype(str)
        + "|"
        + df["mechanism"].astype(str)
        + "|"
        + df["ranking_contamination_mode"].astype(str)
        + "|"
        + df["epsilon"].round(10).astype(str)
        + "|"
        + df["estimator"].astype(str)
    )


def check_source_script(project_root: Path, report: list[str]) -> None:
    script = project_root / "adaptive_v3.py"
    if not script.exists():
        report.append("WARNING: adaptive_v3.py not found; source-setting audit skipped.")
        return

    text = script.read_text(encoding="utf-8", errors="replace")

    expected_literals = {
        r"SEED\s*=\s*20260915": "seed",
        r"M\s*=\s*5": "m",
        r"CYCLES\s*=\s*100": "cycles",
        r"CALIBRATION_TRIALS\s*=\s*20000": "calibration_trials",
        r"ADAPTIVE_TRIALS\s*=\s*200": "adaptive_trials",
        r"PILOT_REPS\s*=\s*30": "pilot_reps",
        r"PILOT_CYCLES\s*=\s*20": "pilot_cycles",
        r"REFERENCE_REPS\s*=\s*2000": "reference_reps",
        r"RHO_TARGETS\s*=\s*\(0\.25,\s*0\.50,\s*0\.75,\s*0\.90\)": "rho_targets",
        r"EPS_VALUES\s*=\s*\(0\.0,\s*0\.10\)": "eps_values",
        r'ESTIMATORS\s*=\s*\("mean",\s*"trimmed",\s*"huber"\)': "estimators",
        r'MECHANISMS\s*=\s*\("gaussian",\s*"laplace"\)': "mechanisms",
        r'MODES\s*=\s*\("post_ranking",\s*"pre_ranking"\)': "modes",
    }

    for pattern, label in expected_literals.items():
        if not re.search(pattern, text):
            fail(f"adaptive_v3.py source-setting check failed for: {label}")

    report.append("PASS: adaptive_v3.py source settings match the frozen V3 design.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--paired-dir", required=False, type=Path, default=None)
    args = parser.parse_args()

    outdir = args.outdir.expanduser().resolve()
    paired_dir = args.paired_dir.expanduser().resolve() if args.paired_dir else None

    required = {
        "adaptive": outdir / "adaptive_v3.csv",
        "decisions": outdir / "adaptive_v3_decisions.csv",
        "references": outdir / "adaptive_v3_reference_risks.csv",
        "calibration": outdir / "calibration_v3.csv",
    }

    for label, path in required.items():
        if not path.exists():
            fail(f"Missing required {label} file: {path}")

    report: list[str] = []
    report.append("POINT 0 — V3 PRIMARY EXPERIMENT FREEZE AUDIT")
    report.append("=" * 78)
    report.append(f"Adaptive result directory: {outdir}")
    report.append("No Monte Carlo simulation was rerun by this audit.")
    report.append("")

    # ------------------------------------------------------------------
    # Source-code freeze check
    # ------------------------------------------------------------------
    check_source_script(outdir.parent, report)

    # ------------------------------------------------------------------
    # Load files
    # ------------------------------------------------------------------
    adaptive = pd.read_csv(required["adaptive"])
    decisions = pd.read_csv(required["decisions"])
    references = pd.read_csv(required["references"])
    calibration = pd.read_csv(required["calibration"])

    # ------------------------------------------------------------------
    # Basic structural checks
    # ------------------------------------------------------------------
    assert_equal("adaptive rows", len(adaptive), 96)
    assert_equal("decision rows", len(decisions), 96 * EXPECTED["adaptive_trials"])
    assert_equal("reference condition rows", len(references), 96)
    assert_equal("calibration rows", len(calibration), 8)

    if len(set(condition_key(adaptive))) != 96:
        fail("Adaptive file does not contain exactly 96 unique conditions.")
    if len(set(condition_key(references))) != 96:
        fail("Reference file does not contain exactly 96 unique conditions.")

    report.append("PASS: 96 adaptive conditions, 19,200 decision trials, 96 references, 8 calibration rows.")

    # ------------------------------------------------------------------
    # V3 design fields
    # ------------------------------------------------------------------
    for col, expected in [
        ("m", EXPECTED["m"]),
        ("cycles", EXPECTED["cycles"]),
        ("n_measured", EXPECTED["n_measured"]),
        ("adaptive_trials", EXPECTED["adaptive_trials"]),
        ("pilot_reps", EXPECTED["pilot_reps"]),
        ("pilot_cycles", EXPECTED["pilot_cycles"]),
        ("reference_reps", EXPECTED["reference_reps"]),
    ]:
        vals = adaptive[col].dropna().unique().tolist()
        if vals != [expected]:
            fail(f"Adaptive column {col} has unexpected values: {vals}")

    assert_equal("adaptive rho levels",
                 tuple(sorted(adaptive["target_rho"].unique())),
                 TARGET_RHOS)
    assert_equal("adaptive mechanisms",
                 tuple(sorted(adaptive["mechanism"].unique())),
                 tuple(sorted(MECHANISMS)))
    assert_equal("adaptive modes",
                 tuple(sorted(adaptive["ranking_contamination_mode"].unique())),
                 tuple(sorted(MODES)))
    assert_equal("adaptive eps",
                 tuple(sorted(adaptive["epsilon"].unique())),
                 tuple(sorted(EPS_VALUES)))
    assert_equal("adaptive estimators",
                 tuple(sorted(adaptive["estimator"].unique())),
                 tuple(sorted(ESTIMATORS)))

    if not adaptive["RSS_selection_fraction"].between(0, 1).all():
        fail("RSS_selection_fraction is outside [0,1].")

    report.append("PASS: frozen design factors match Point-0 specification.")

    # ------------------------------------------------------------------
    # Reference oracle and RE
    # ------------------------------------------------------------------
    references["RE_Var_reference"] = (
        references["SRS_variance_reference"]
        / references["RSS_variance_reference"]
    )

    oracle_method = np.where(
        references["RSS_variance_reference"] <= references["SRS_variance_reference"],
        "RSS",
        "SRS",
    )
    references["oracle_method_variance_check"] = oracle_method

    rss_oracle_count = int(np.sum(oracle_method == "RSS"))
    srs_oracle_count = int(np.sum(oracle_method == "SRS"))

    assert_equal("fixed-strategy RSS oracle count", rss_oracle_count, 88)
    assert_equal("fixed-strategy SRS oracle count", srs_oracle_count, 8)

    re_all = references["RE_Var_reference"].to_numpy()
    overall_re_mean = float(np.mean(re_all))
    overall_re_median = float(np.median(re_all))
    overall_re_min = float(np.min(re_all))
    overall_re_max = float(np.max(re_all))

    assert_close("overall mean reference RE", overall_re_mean, 1.3873, atol=5e-5)
    assert_close("overall median reference RE", overall_re_median, 1.2611, atol=5e-5)
    assert_close("overall min reference RE", overall_re_min, 0.9289, atol=5e-5)
    assert_close("overall max reference RE", overall_re_max, 2.2006, atol=5e-5)

    report.append(
        f"PASS: fixed-strategy oracle = RSS {rss_oracle_count}/96, "
        f"SRS {srs_oracle_count}/96."
    )

    # ------------------------------------------------------------------
    # Adaptive selection aggregates
    # ------------------------------------------------------------------
    sel = adaptive["RSS_selection_fraction"].to_numpy()
    sel_mean = float(np.mean(sel))
    sel_median = float(np.median(sel))

    assert_close("overall RSS selection fraction", sel_mean, 0.7312, atol=5e-5)
    assert_close("median RSS selection fraction", sel_median, 0.7025, atol=5e-5)

    expected_by_rho = {
        0.25: 0.533,
        0.50: 0.656,
        0.75: 0.823,
        0.90: 0.913,
    }
    for rho, expected in expected_by_rho.items():
        actual = float(
            adaptive.loc[adaptive["target_rho"] == rho, "RSS_selection_fraction"].mean()
        )
        assert_close(f"mean RSS selection fraction rho={rho}", actual, expected, atol=5e-4)

    report.append("PASS: overall and per-rho RSS selection aggregates match Point-0 values.")

    # ------------------------------------------------------------------
    # Adaptive vs reference consistency
    # ------------------------------------------------------------------
    merged = adaptive.merge(
        references[
            COND_COLS
            + [
                "RSS_variance_reference",
                "SRS_variance_reference",
                "RSS_MSE_reference",
                "SRS_MSE_reference",
                "oracle_best_variance",
                "oracle_best_MSE",
            ]
        ],
        on=COND_COLS,
        how="outer",
        suffixes=("_adaptive", "_reference"),
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        fail("Adaptive/reference condition keys do not match exactly.")

    ref_cols = [
        "RSS_variance_reference",
        "SRS_variance_reference",
        "RSS_MSE_reference",
        "SRS_MSE_reference",
        "oracle_best_variance",
        "oracle_best_MSE",
    ]

    for c in ref_cols:
        a = merged[f"{c}_adaptive"].to_numpy()
        b = merged[f"{c}_reference"].to_numpy()
        if not np.allclose(a, b, rtol=0, atol=1e-14):
            fail(f"Adaptive/reference mismatch in {c}")

    report.append("PASS: adaptive_v3.csv reference-risk columns exactly match the independent reference file.")

    # ------------------------------------------------------------------
    # Trial-level agreement and condition-level majority agreement
    # ------------------------------------------------------------------
    ref_oracle = references[COND_COLS + ["RSS_variance_reference", "SRS_variance_reference"]].copy()
    ref_oracle["oracle_method"] = np.where(
        ref_oracle["RSS_variance_reference"] <= ref_oracle["SRS_variance_reference"],
        "RSS",
        "SRS",
    )

    d = decisions.merge(
        ref_oracle[COND_COLS + ["oracle_method"]],
        on=COND_COLS,
        how="left",
        validate="many_to_one",
    )

    if d["oracle_method"].isna().any():
        fail("Decision rows contain conditions with no reference oracle.")

    trial_agreement = float(np.mean(d["selected_method"] == d["oracle_method"]))
    assert_close(
        "trial-level variance-oracle agreement",
        trial_agreement,
        0.7264,
        atol=5e-5,
    )

    condition_sel = (
        d.groupby(COND_COLS, dropna=False)["selected_method"]
        .apply(lambda s: float(np.mean(s == "RSS")))
        .reset_index(name="rss_trial_fraction")
    )

    condition_sel = condition_sel.merge(
        ref_oracle[COND_COLS + ["oracle_method"]],
        on=COND_COLS,
        validate="one_to_one",
    )

    # Majority rule used for the manuscript audit:
    # RSS only when RSS is selected in strictly more than 50% of trials;
    # a 50/50 tie is counted as SRS.
    condition_sel["majority_method"] = np.where(
        condition_sel["rss_trial_fraction"] > 0.5, "RSS", "SRS"
    )
    majority_agreement = int(
        np.sum(condition_sel["majority_method"] == condition_sel["oracle_method"])
    )
    assert_equal("majority-choice agreement count", majority_agreement, 90)

    report.append(
        f"PASS: trial-level variance-oracle agreement = {trial_agreement:.4f}; "
        f"majority condition agreement = {majority_agreement}/96."
    )

    # ------------------------------------------------------------------
    # Calibration audit
    # ------------------------------------------------------------------
    if set(calibration["mechanism"].unique()) != set(MECHANISMS):
        fail("Calibration mechanisms are not Gaussian/Laplace.")
    if set(np.round(calibration["target_rho"], 10)) != set(np.round(TARGET_RHOS, 10)):
        fail("Calibration target rho levels are incorrect.")
    if not (calibration["trials"] == EXPECTED["calibration_trials"]).all():
        fail("Calibration trial count is not 20,000 for every row.")
    if "abs_rho_error" in calibration.columns:
        max_cal_err = float(calibration["abs_rho_error"].max())
        if max_cal_err > 1e-4:
            fail(f"Calibration error too large: max={max_cal_err}")
    else:
        max_cal_err = float(
            np.max(np.abs(calibration["pilot_expected_rho"] - calibration["target_rho"]))
        )
        if max_cal_err > 1e-4:
            fail(f"Calibration error too large: max={max_cal_err}")

    report.append(f"PASS: 8-row calibration table uses 20,000 trials per target/mechanism pair; max |rho-target| <= {max_cal_err:.6g}.")

    # ------------------------------------------------------------------
    # Paired Gaussian/Laplace supporting experiment
    # ------------------------------------------------------------------
    if paired_dir is not None:
        paired_path = paired_dir / "paired_delta_summary_v3.csv"
        if not paired_path.exists():
            fail(f"Paired summary not found: {paired_path}")

        paired = pd.read_csv(paired_path)
        assert_equal("paired summary rows", len(paired), 16)

        if "bootstrap_reps" not in paired.columns:
            fail("paired_delta_summary_v3.csv lacks bootstrap_reps.")
        if not (paired["bootstrap_reps"] == 400).all():
            fail("Paired summary does not use 400 bootstrap replicates in every row.")

        expected_est = set(ESTIMATORS) | {"tukey"}
        if set(paired["estimator"].unique()) != expected_est:
            fail("Paired summary estimator set is not mean/trimmed/huber/tukey.")

        # The paired file reports realized Gaussian/Laplace rho differences,
        # which should be small for the matched mechanism comparison.
        max_abs_rho_diff = float(paired["abs_rho_difference"].max())
        if max_abs_rho_diff > 0.01:
            fail(f"Paired rho matching is too loose: max abs difference={max_abs_rho_diff}")

        report.append(
            f"PASS: paired V3 mechanism summary has 16 rows, 400 bootstrap reps, "
            f"and max realized |rho_G-rho_L|={max_abs_rho_diff:.6f}."
        )

    # ------------------------------------------------------------------
    # Design-level summary
    # ------------------------------------------------------------------
    report.append("")
    report.append("Recomputed headline values:")
    report.append(f"  Mean RSS selection fraction: {sel_mean:.6f}")
    report.append(f"  Median RSS selection fraction: {sel_median:.6f}")
    report.append(f"  Fixed-strategy RSS oracle: {rss_oracle_count}/96")
    report.append(f"  Fixed-strategy SRS oracle: {srs_oracle_count}/96")
    report.append(f"  Trial-level oracle agreement: {trial_agreement:.6f}")
    report.append(f"  Majority condition-level agreement: {majority_agreement}/96")
    report.append(f"  Mean reference RE: {overall_re_mean:.6f}")
    report.append(f"  Median reference RE: {overall_re_median:.6f}")
    report.append(f"  Reference RE range: {overall_re_min:.6f}--{overall_re_max:.6f}")

    report.append("")
    report.append("Per-rho RSS selection fraction:")
    for rho in TARGET_RHOS:
        value = float(
            adaptive.loc[adaptive["target_rho"] == rho, "RSS_selection_fraction"].mean()
        )
        report.append(f"  rho={rho:.2f}: {value:.6f}")

    report.append("")
    report.append("Per-rho mean fixed-strategy RE:")
    for rho in TARGET_RHOS:
        value = float(
            references.loc[references["target_rho"] == rho, "RE_Var_reference"].mean()
        )
        report.append(f"  rho={rho:.2f}: {value:.6f}")

    # ------------------------------------------------------------------
    # SHA-256 freeze record
    # ------------------------------------------------------------------
    report.append("")
    report.append("SHA-256 hashes:")
    for label, path in required.items():
        report.append(f"  {label:10s} {sha256(path)}  {path.name}")

    if paired_dir is not None:
        paired_path = paired_dir / "paired_delta_summary_v3.csv"
        report.append(f"  {'paired':10s} {sha256(paired_path)}  {paired_path.name}")

    report.append("")
    report.append("STATUS: PASS — Point 0 is frozen and reproducible from the existing result files.")
    report_path = outdir / "point0_validation_report.txt"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nSaved audit report: {report_path}")


if __name__ == "__main__":
    main()
