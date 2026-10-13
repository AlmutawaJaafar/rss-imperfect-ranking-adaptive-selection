#!/usr/bin/env python3
"""
adaptive_v3.py

Standalone V3 adaptive RSS/SRS experiment.

This script runs ONLY the adaptive experiment.  It imports the validated V3
implementation from rss_adaptive_experiments_v3.py and fixes the problem in
the previous adaptive run: the RSS/SRS reference risks are estimated in a
separate, independent V3 Monte Carlo experiment rather than being taken from
the adaptive evaluation sample (or inherited from V2).

Design:
  rho       = {0.25, 0.50, 0.75, 0.90}
  mechanism = {gaussian, laplace}
  mode      = {post_ranking, pre_ranking}
  epsilon   = {0.0, 0.10}
  estimator = {mean, trimmed, huber}

Pilot and evaluation samples are independent. The adaptive decision uses
ONLY the pilot variance estimates. Independent reference risks are used
ONLY for ex-post oracle/regret/decision-accuracy diagnostics.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import rss_adaptive_experiments_v3 as v

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
SEED = 20260915
M = 5
CYCLES = 100
RHO_TARGETS = (0.25, 0.50, 0.75, 0.90)
EPS_VALUES = (0.0, 0.10)
ESTIMATORS = ("mean", "trimmed", "huber")
MECHANISMS = ("gaussian", "laplace")
MODES = ("post_ranking", "pre_ranking")

CALIBRATION_TRIALS = 20000
ADAPTIVE_TRIALS = 200
PILOT_REPS = 30
PILOT_CYCLES = 20

# Independent fixed-strategy reference Monte Carlo.
# Step 1.2 final independent fixed-strategy reference experiment.
REFERENCE_REPS = 5000

OUTDIR = "rss_results_v3_adaptive_primary"


def make_cfg() -> v.Config:
    return v.Config(
        seed=SEED,
        reps=REFERENCE_REPS,
        cycles=CYCLES,
        m=M,
        rho_targets=RHO_TARGETS,
        calibration_trials=CALIBRATION_TRIALS,
        bootstrap_reps=400,
        eps_values=EPS_VALUES,
        adaptive_trials=ADAPTIVE_TRIALS,
        pilot_reps=PILOT_REPS,
        pilot_cycles=PILOT_CYCLES,
        outdir=OUTDIR,
    )


def get_scale(
    calibration: pd.DataFrame,
    target: float,
    mechanism: str
) -> float:
    q = calibration[
        (calibration["target_rho"] == float(target))
        & (calibration["mechanism"] == mechanism)
    ]

    if len(q) != 1:
        raise RuntimeError(
            f"Expected one calibration row for rho={target}, "
            f"mechanism={mechanism}; found {len(q)}."
        )

    return float(q.iloc[0]["calibrated_scale"])


def reference_seed(
    target,
    mechanism,
    mode,
    eps,
    estimator,
    rep
):
    """Independent deterministic seed for the fixed-strategy reference run."""
    return (
        SEED
        + 10000000
        + int(round(1000 * target)) * 100000
        + (0 if mechanism == "gaussian" else 1) * 10000
        + (0 if mode == "post_ranking" else 1) * 1000
        + int(round(1000 * eps)) * 100
        + ESTIMATORS.index(estimator) * 10
        + rep
    )


def adaptive_seed(
    target,
    mechanism,
    mode,
    eps,
    estimator,
    trial
):
    """Independent seed stream for the adaptive pilot/evaluation experiment."""
    return (
        SEED
        + 40000000
        + int(round(1000 * target)) * 1000000
        + (0 if mechanism == "gaussian" else 1) * 100000
        + (0 if mode == "post_ranking" else 1) * 10000
        + int(round(1000 * eps)) * 1000
        + ESTIMATORS.index(estimator) * 100
        + trial
    )


def run_independent_references(cfg, calibration):
    """Estimate fixed RSS/SRS variance and MSE independently of adaptation."""
    summary = []
    replicate_rows = []

    for target in RHO_TARGETS:
        for mechanism in MECHANISMS:
            scale = get_scale(calibration, target, mechanism)

            for mode in MODES:
                for eps in EPS_VALUES:
                    for estimator in ESTIMATORS:

                        rss = np.empty(REFERENCE_REPS, dtype=float)
                        srs = np.empty(REFERENCE_REPS, dtype=float)

                        for r in range(REFERENCE_REPS):
                            rng = np.random.default_rng(
                                reference_seed(
                                    target,
                                    mechanism,
                                    mode,
                                    eps,
                                    estimator,
                                    r
                                )
                            )

                            out = v.simulate_dataset(
                                M,
                                CYCLES,
                                mechanism,
                                scale,
                                estimator,
                                cfg,
                                rng,
                                epsilon=eps,
                                contamination_mode=mode,
                            )

                            rss[r] = out["rss_est"]
                            srs[r] = out["srs_est"]

                        rv = float(np.var(rss, ddof=1))
                        sv = float(np.var(srs, ddof=1))
                        rm = float(np.mean(rss ** 2))
                        sm = float(np.mean(srs ** 2))

                        summary.append({
                            "target_rho": target,
                            "mechanism": mechanism,
                            "ranking_contamination_mode": mode,
                            "epsilon": eps,
                            "estimator": estimator,
                            "reference_reps": REFERENCE_REPS,
                            "RSS_variance_reference": rv,
                            "SRS_variance_reference": sv,
                            "RSS_MSE_reference": rm,
                            "SRS_MSE_reference": sm,
                            "oracle_best_variance": min(rv, sv),
                            "oracle_best_MSE": min(rm, sm),
                        })

                        # Save replicate-level reference values for auditability.
                        for r in range(REFERENCE_REPS):
                            replicate_rows.append({
                                "target_rho": target,
                                "mechanism": mechanism,
                                "ranking_contamination_mode": mode,
                                "epsilon": eps,
                                "estimator": estimator,
                                "replicate": r,
                                "rss_est": rss[r],
                                "srs_est": srs[r],
                            })

    return pd.DataFrame(summary), pd.DataFrame(replicate_rows)


def run_adaptive(cfg, calibration, references):
    """Run adaptive decisions using independent pilot and evaluation samples."""
    summary = []
    decisions = []

    for target in RHO_TARGETS:
        for mechanism in MECHANISMS:
            scale = get_scale(calibration, target, mechanism)

            for mode in MODES:
                for eps in EPS_VALUES:
                    for estimator in ESTIMATORS:

                        q = references[
                            (references.target_rho == float(target))
                            & (references.mechanism == mechanism)
                            & (
                                references.ranking_contamination_mode
                                == mode
                            )
                            & (references.epsilon == float(eps))
                            & (references.estimator == estimator)
                        ]

                        if len(q) != 1:
                            raise RuntimeError(
                                "Expected exactly one reference row for "
                                f"rho={target}, mechanism={mechanism}, "
                                f"mode={mode}, eps={eps}, "
                                f"estimator={estimator}; found {len(q)}."
                            )

                        ref = q.iloc[0]

                        oracle_method_var = (
                            "RSS"
                            if ref.RSS_variance_reference
                            <= ref.SRS_variance_reference
                            else "SRS"
                        )

                        oracle_method_mse = (
                            "RSS"
                            if ref.RSS_MSE_reference
                            <= ref.SRS_MSE_reference
                            else "SRS"
                        )

                        selected = []
                        chosen_values = []
                        eval_rss = []
                        eval_srs = []
                        pilot_rss = []
                        pilot_srs = []

                        for a in range(ADAPTIVE_TRIALS):

                            rng_pilot = np.random.default_rng(
                                adaptive_seed(
                                    target,
                                    mechanism,
                                    mode,
                                    eps,
                                    estimator,
                                    a
                                )
                            )

                            pilot_values_rss = []
                            pilot_values_srs = []

                            for _ in range(PILOT_REPS):
                                pilot = v.simulate_dataset(
                                    M,
                                    PILOT_CYCLES,
                                    mechanism,
                                    scale,
                                    estimator,
                                    cfg,
                                    rng_pilot,
                                    epsilon=eps,
                                    contamination_mode=mode,
                                )

                                pilot_values_rss.append(pilot["rss_est"])
                                pilot_values_srs.append(pilot["srs_est"])

                            rv = float(
                                np.var(
                                    pilot_values_rss,
                                    ddof=1
                                )
                            )

                            sv = float(
                                np.var(
                                    pilot_values_srs,
                                    ddof=1
                                )
                            )

                            choice = "RSS" if rv <= sv else "SRS"

                            rng_eval = np.random.default_rng(
                                adaptive_seed(
                                    target,
                                    mechanism,
                                    mode,
                                    eps,
                                    estimator,
                                    a + 1000000
                                )
                            )

                            out = v.simulate_dataset(
                                M,
                                CYCLES,
                                mechanism,
                                scale,
                                estimator,
                                cfg,
                                rng_eval,
                                epsilon=eps,
                                contamination_mode=mode,
                            )

                            rss_est = out["rss_est"]
                            srs_est = out["srs_est"]

                            chosen_value = (
                                rss_est
                                if choice == "RSS"
                                else srs_est
                            )

                            selected.append(choice)
                            chosen_values.append(chosen_value)
                            eval_rss.append(rss_est)
                            eval_srs.append(srs_est)
                            pilot_rss.append(rv)
                            pilot_srs.append(sv)

                            decisions.append({
                                "target_rho": target,
                                "mechanism": mechanism,
                                "ranking_contamination_mode": mode,
                                "epsilon": eps,
                                "estimator": estimator,
                                "adaptive_trial": a,
                                "selected_method": choice,
                                "pilot_RSS_variance": rv,
                                "pilot_SRS_variance": sv,
                                "evaluation_RSS_est": rss_est,
                                "evaluation_SRS_est": srs_est,
                                "reference_oracle_method_variance":
                                    oracle_method_var,
                                "reference_oracle_method_MSE":
                                    oracle_method_mse,
                            })

                        selected = np.asarray(selected)
                        chosen_values = np.asarray(chosen_values)
                        eval_rss = np.asarray(eval_rss)
                        eval_srs = np.asarray(eval_srs)

                        av = float(
                            np.var(
                                chosen_values,
                                ddof=1
                            )
                        )

                        am = float(
                            np.mean(
                                chosen_values ** 2
                            )
                        )

                        ov = float(
                            ref.oracle_best_variance
                        )

                        om = float(
                            ref.oracle_best_MSE
                        )

                        ev = av - ov
                        em = am - om

                        summary.append({
                            "experiment": "adaptive_v3",
                            "m": M,
                            "cycles": CYCLES,
                            "n_measured": M * CYCLES,
                            "target_rho": target,
                            "mechanism": mechanism,
                            "ranking_contamination_mode": mode,
                            "epsilon": eps,
                            "estimator": estimator,
                            "adaptive_trials": ADAPTIVE_TRIALS,
                            "pilot_reps": PILOT_REPS,
                            "pilot_cycles": PILOT_CYCLES,
                            "reference_reps": REFERENCE_REPS,
                            "RSS_selection_fraction":
                                float(
                                    np.mean(
                                        selected == "RSS"
                                    )
                                ),
                            "pilot_mean_RSS_variance":
                                float(
                                    np.mean(pilot_rss)
                                ),
                            "pilot_mean_SRS_variance":
                                float(
                                    np.mean(pilot_srs)
                                ),
                            "adaptive_variance": av,
                            "adaptive_MSE": am,
                            "RSS_variance_reference":
                                float(
                                    ref.RSS_variance_reference
                                ),
                            "SRS_variance_reference":
                                float(
                                    ref.SRS_variance_reference
                                ),
                            "RSS_MSE_reference":
                                float(
                                    ref.RSS_MSE_reference
                                ),
                            "SRS_MSE_reference":
                                float(
                                    ref.SRS_MSE_reference
                                ),
                            "oracle_best_variance": ov,
                            "oracle_best_MSE": om,
                            "adaptive_excess_variance_vs_oracle":
                                ev,
                            "adaptive_excess_MSE_vs_oracle":
                                em,
                            "adaptive_relative_oracle_regret_percent_variance":
                                100.0 * ev / ov,
                            "adaptive_relative_oracle_regret_percent_MSE":
                                100.0 * em / om,
                            "oracle_method_variance":
                                oracle_method_var,
                            "oracle_method_MSE":
                                oracle_method_mse,
                            "decision_accuracy_vs_variance_oracle":
                                float(
                                    np.mean(
                                        selected
                                        == oracle_method_var
                                    )
                                ),
                            "decision_accuracy_vs_MSE_oracle":
                                float(
                                    np.mean(
                                        selected
                                        == oracle_method_mse
                                    )
                                ),
                        })

    return pd.DataFrame(summary), pd.DataFrame(decisions)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    cfg = make_cfg()

    print("=" * 78)
    print("V3 ADAPTIVE RSS/SRS EXPERIMENT")
    print("=" * 78)

    print(
        f"seed={SEED}, m={M}, cycles={CYCLES}, n={M*CYCLES}"
    )

    print(f"rho={RHO_TARGETS}")

    print(
        f"adaptive_trials={ADAPTIVE_TRIALS}, "
        f"pilot_reps={PILOT_REPS}, "
        f"pilot_cycles={PILOT_CYCLES}"
    )

    print(
        f"reference_reps={REFERENCE_REPS}, "
        f"calibration_trials={CALIBRATION_TRIALS}"
    )

    print(
        f"output={os.path.abspath(OUTDIR)}"
    )

    print()

    print("[1/3] V3 calibration...")

    calibration = v.build_calibration_table(cfg)

    calibration.to_csv(
        os.path.join(
            OUTDIR,
            "calibration_v3.csv"
        ),
        index=False
    )

    print(
        "[2/3] Independent V3 RSS/SRS reference experiment..."
    )

    references, reference_reps = run_independent_references(
        cfg,
        calibration
    )

    references.to_csv(
        os.path.join(
            OUTDIR,
            "adaptive_v3_reference_risks.csv"
        ),
        index=False
    )

    # IMPORTANT:
    # Save the 5,000-replicate reference file separately so that the
    # preserved 2,000-replicate audit file remains untouched.
    reference_reps.to_csv(
        os.path.join(
            OUTDIR,
            "adaptive_v3_reference_final_5000.csv"
        ),
        index=False
    )

    print("[3/3] Adaptive V3 experiment...")

    adaptive, decisions = run_adaptive(
        cfg,
        calibration,
        references
    )

    adaptive.to_csv(
        os.path.join(
            OUTDIR,
            "adaptive_v3.csv"
        ),
        index=False
    )

    decisions.to_csv(
        os.path.join(
            OUTDIR,
            "adaptive_v3_decisions.csv"
        ),
        index=False
    )

    print()
    print("V3 adaptive experiment completed.")
    print(
        "5,000-replicate reference file:"
    )
    print(
        os.path.abspath(
            os.path.join(
                OUTDIR,
                "adaptive_v3_reference_final_5000.csv"
            )
        )
    )


if __name__ == "__main__":
    main()
