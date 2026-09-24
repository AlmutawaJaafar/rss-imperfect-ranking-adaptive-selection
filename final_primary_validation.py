import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("rss_results_v3_adaptive_primary")

adaptive_path = BASE / "adaptive_v3.csv"
trial_path = BASE / "adaptive_v3_trial_results.csv"
reference_path = BASE / "adaptive_v3_reference_final_5000.csv"

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------
adaptive = pd.read_csv(adaptive_path)
trials = pd.read_csv(trial_path)
reference = pd.read_csv(reference_path)

keys = [
    "target_rho",
    "mechanism",
    "ranking_contamination_mode",
    "epsilon",
    "estimator",
]

trial_keys = [
    "experiment",
    "m",
    "cycles",
    "n_measured",
] + keys

reference_keys = keys

print("=" * 80)
print("FINAL PRIMARY NUMERICAL VALIDATION")
print("=" * 80)

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
print("\n[1] CONFIGURATION")

print("Adaptive rows:", len(adaptive))
print("Trial rows:", len(trials))
print("Reference rows:", len(reference))

assert len(adaptive) == 96
assert len(trials) == 96 * 1000
assert len(reference) == 96 * 5000

assert set(adaptive["adaptive_trials"]) == {1000}
assert set(adaptive["pilot_reps"]) == {200}
assert set(adaptive["pilot_cycles"]) == {20}

print("Adaptive trials: 1000")
print("Pilot replications: 200")
print("Pilot cycles: 20")
print("Reference replications: 5000")
print("Conditions: 96")

# ------------------------------------------------------------
# 2. Adaptive summary structural validation
# ------------------------------------------------------------
print("\n[2] ADAPTIVE SUMMARY STRUCTURE")

n_conditions_adaptive = adaptive[keys].drop_duplicates().shape[0]
print("Unique adaptive conditions:", n_conditions_adaptive)

assert n_conditions_adaptive == 96

print("PASS: adaptive_v3.csv contains exactly 96 conditions.")

# ------------------------------------------------------------
# 3. Trial structure
# ------------------------------------------------------------
print("\n[3] ADAPTIVE TRIAL STRUCTURE")

n_conditions_trials = trials[trial_keys].drop_duplicates().shape[0]
trial_counts = trials.groupby(trial_keys, dropna=False).size()

print("Unique trial conditions:", n_conditions_trials)
print("Minimum trials/condition:", trial_counts.min())
print("Maximum trials/condition:", trial_counts.max())

assert n_conditions_trials == 96
assert trial_counts.min() == 1000
assert trial_counts.max() == 1000

assert set(trials["selected_method"].unique()) == {"RSS", "SRS"}

print("Selected methods:", sorted(trials["selected_method"].unique()))
print("PASS: 96 × 1000 adaptive trials.")

# ------------------------------------------------------------
# 4. Reference structure
# ------------------------------------------------------------
print("\n[4] 5,000-REPLICATE REFERENCE STRUCTURE")

n_conditions_ref = reference[reference_keys].drop_duplicates().shape[0]
ref_counts = reference.groupby(reference_keys, dropna=False).size()

print("Unique reference conditions:", n_conditions_ref)
print("Minimum replicates/condition:", ref_counts.min())
print("Maximum replicates/condition:", ref_counts.max())

assert n_conditions_ref == 96
assert ref_counts.min() == 5000
assert ref_counts.max() == 5000

dup_ref = reference.duplicated(reference_keys + ["replicate"]).sum()
print("Duplicate condition/replicate rows:", dup_ref)
assert dup_ref == 0

print("PASS: 96 × 5000 reference replicates.")

# ------------------------------------------------------------
# 5. Recompute fixed-strategy reference statistics from raw data
# ------------------------------------------------------------
print("\n[5] RECOMPUTING 5,000-REFERENCE ORACLE")

ref_rows = []

for condition, g in reference.groupby(reference_keys, sort=False):

    rss = g["rss_est"].to_numpy()
    srs = g["srs_est"].to_numpy()

    rss_var = np.var(rss, ddof=1)
    srs_var = np.var(srs, ddof=1)

    rss_mse = np.mean(rss ** 2)
    srs_mse = np.mean(srs ** 2)

    oracle_var = "RSS" if rss_var <= srs_var else "SRS"
    oracle_mse = "RSS" if rss_mse <= srs_mse else "SRS"

    re = srs_var / rss_var

    row = dict(zip(reference_keys, condition))
    row.update({
        "RSS_variance_recomputed": rss_var,
        "SRS_variance_recomputed": srs_var,
        "RSS_MSE_recomputed": rss_mse,
        "SRS_MSE_recomputed": srs_mse,
        "oracle_variance_recomputed": oracle_var,
        "oracle_MSE_recomputed": oracle_mse,
        "RE_recomputed": re,
    })

    ref_rows.append(row)

ref_stats = pd.DataFrame(ref_rows)

# ------------------------------------------------------------
# 6. Match recomputed reference statistics to adaptive summary
# ------------------------------------------------------------
print("\n[6] CROSS-CHECKING STORED REFERENCE VALUES")

merged = adaptive.merge(
    ref_stats,
    on=keys,
    how="left",
    validate="one_to_one",
)

assert len(merged) == 96
assert merged["RSS_variance_recomputed"].notna().all()

checks = {
    "RSS variance":
        np.max(np.abs(
            merged["RSS_variance_reference"]
            - merged["RSS_variance_recomputed"]
        )),

    "SRS variance":
        np.max(np.abs(
            merged["SRS_variance_reference"]
            - merged["SRS_variance_recomputed"]
        )),

    "RSS MSE":
        np.max(np.abs(
            merged["RSS_MSE_reference"]
            - merged["RSS_MSE_recomputed"]
        )),

    "SRS MSE":
        np.max(np.abs(
            merged["SRS_MSE_reference"]
            - merged["SRS_MSE_recomputed"]
        )),
}

for name, diff in checks.items():
    print(f"{name} max abs difference: {diff:.3e}")
    assert diff < 1e-12

print("PASS: Stored reference statistics exactly reproduce the raw 5,000-replicate file.")

# ------------------------------------------------------------
# 7. Verify oracle labels
# ------------------------------------------------------------
print("\n[7] ORACLE LABEL VALIDATION")

stored_var_oracle = np.where(
    merged["RSS_variance_reference"] <= merged["SRS_variance_reference"],
    "RSS",
    "SRS",
)

stored_mse_oracle = np.where(
    merged["RSS_MSE_reference"] <= merged["SRS_MSE_reference"],
    "RSS",
    "SRS",
)

# Compare with stored oracle-best values
oracle_var_value = merged[["RSS_variance_reference", "SRS_variance_reference"]].min(axis=1)
oracle_mse_value = merged[["RSS_MSE_reference", "SRS_MSE_reference"]].min(axis=1)

assert np.allclose(
    merged["oracle_best_variance"],
    oracle_var_value,
    rtol=0,
    atol=1e-12,
)

assert np.allclose(
    merged["oracle_best_MSE"],
    oracle_mse_value,
    rtol=0,
    atol=1e-12,
)

print("Variance oracle counts:")
print(pd.Series(stored_var_oracle).value_counts().to_string())

print("\nMSE oracle counts:")
print(pd.Series(stored_mse_oracle).value_counts().to_string())

# ------------------------------------------------------------
# 8. Recompute adaptive variance and MSE from the 96,000 trials
# ------------------------------------------------------------
print("\n[8] RECOMPUTING ADAPTIVE VARIANCE AND MSE")

adaptive_rows = []

for condition, g in trials.groupby(trial_keys, sort=False):

    estimates = g["adaptive_estimate"].to_numpy()

    var = np.var(estimates, ddof=1)
    mse = np.mean(estimates ** 2)

    row = dict(zip(trial_keys, condition))
    row.update({
        "adaptive_variance_recomputed": var,
        "adaptive_MSE_recomputed": mse,
    })

    adaptive_rows.append(row)

adaptive_recomputed = pd.DataFrame(adaptive_rows)

merged2 = adaptive.merge(
    adaptive_recomputed,
    on=trial_keys,
    how="left",
    validate="one_to_one",
)

assert len(merged2) == 96

var_diff = np.max(np.abs(
    merged2["adaptive_variance"]
    - merged2["adaptive_variance_recomputed"]
))

mse_diff = np.max(np.abs(
    merged2["adaptive_MSE"]
    - merged2["adaptive_MSE_recomputed"]
))

print(f"Adaptive variance max abs difference: {var_diff:.3e}")
print(f"Adaptive MSE max abs difference:      {mse_diff:.3e}")

assert var_diff < 1e-12
assert mse_diff < 1e-12

print("PASS: Adaptive variance and MSE exactly reproduce the 96,000 trial file.")

# ------------------------------------------------------------
# 9. Recompute trial-level oracle agreement
# ------------------------------------------------------------
print("\n[9] TRIAL-LEVEL ORACLE AGREEMENT")

oracle_lookup = ref_stats[
    keys + ["oracle_variance_recomputed", "oracle_MSE_recomputed"]
]

trials2 = trials.merge(
    oracle_lookup,
    on=keys,
    how="left",
    validate="many_to_one",
)

var_agreement = (
    trials2["selected_method"]
    == trials2["oracle_variance_recomputed"]
)

mse_agreement = (
    trials2["selected_method"]
    == trials2["oracle_MSE_recomputed"]
)

print(
    "Variance-oracle agreement:",
    f"{var_agreement.mean():.9f}",
    f"({var_agreement.sum():,}/{len(var_agreement):,})"
)

print(
    "MSE-oracle agreement:",
    f"{mse_agreement.mean():.9f}",
    f"({mse_agreement.sum():,}/{len(mse_agreement):,})"
)

# ------------------------------------------------------------
# 10. Recompute condition-level majority agreement
# ------------------------------------------------------------
print("\n[10] CONDITION-LEVEL MAJORITY AGREEMENT")

condition_results = []

for condition, g in trials2.groupby(keys, sort=False):

    oracle_v = g["oracle_variance_recomputed"].iloc[0]
    oracle_m = g["oracle_MSE_recomputed"].iloc[0]

    rss_fraction = (g["selected_method"] == "RSS").mean()

    majority_method = "RSS" if rss_fraction >= 0.5 else "SRS"

    row = dict(zip(keys, condition))
    row.update({
        "RSS_selection_fraction": rss_fraction,
        "majority_method": majority_method,
        "oracle_variance": oracle_v,
        "oracle_MSE": oracle_m,
        "variance_majority_agreement":
            majority_method == oracle_v,
        "MSE_majority_agreement":
            majority_method == oracle_m,
    })

    condition_results.append(row)

condition_results = pd.DataFrame(condition_results)

var_majority = condition_results["variance_majority_agreement"].mean()
mse_majority = condition_results["MSE_majority_agreement"].mean()

print(
    "Variance majority agreement:",
    f"{var_majority:.9f}",
    f"({condition_results['variance_majority_agreement'].sum()}/96)"
)

print(
    "MSE majority agreement:",
    f"{mse_majority:.9f}",
    f"({condition_results['MSE_majority_agreement'].sum()}/96)"
)

# ------------------------------------------------------------
# 11. Relative efficiency
# ------------------------------------------------------------
print("\n[11] REFERENCE RELATIVE EFFICIENCY")

re = ref_stats["RE_recomputed"]

print(f"Mean RE:   {re.mean():.9f}")
print(f"Median RE: {re.median():.9f}")
print(f"Min RE:    {re.min():.9f}")
print(f"Max RE:    {re.max():.9f}")

print("\nMean RE by rho:")
print(
    ref_stats.groupby("target_rho")["RE_recomputed"]
    .mean()
    .to_string()
)

# ------------------------------------------------------------
# 12. Aggregate adaptive selection
# ------------------------------------------------------------
print("\n[12] ADAPTIVE SELECTION")

overall_rss = (trials["selected_method"] == "RSS").mean()

condition_selection = (
    trials.groupby(keys)["selected_method"]
    .apply(lambda x: (x == "RSS").mean())
)

print(f"Overall RSS selection fraction: {overall_rss:.9f}")
print(f"Median condition RSS selection: {condition_selection.median():.9f}")

print("\nRSS selection by rho:")

rho_selection = (
    trials.assign(is_rss=trials["selected_method"].eq("RSS"))
    .groupby("target_rho")["is_rss"]
    .mean()
)

print(rho_selection.to_string())

# ------------------------------------------------------------
# 13. Excess variance / MSE verification
# ------------------------------------------------------------
print("\n[13] EXCESS VARIANCE / MSE")

# Recompute from actual adaptive and oracle values.
merged3 = merged2.merge(
    ref_stats[
        keys +
        [
            "oracle_variance_recomputed",
            "oracle_MSE_recomputed",
        ]
    ],
    on=keys,
    how="left",
    validate="one_to_one",
)

excess_var = (
    merged3["adaptive_variance"]
    - merged3["oracle_variance_recomputed"]
)

excess_mse = (
    merged3["adaptive_MSE"]
    - merged3["oracle_MSE_recomputed"]
)

print(
    "Max abs excess-variance difference from stored:",
    np.max(np.abs(
        excess_var - merged3["adaptive_excess_variance_vs_oracle"]
    ))
)

print(
    "Max abs excess-MSE difference from stored:",
    np.max(np.abs(
        excess_mse - merged3["adaptive_excess_MSE_vs_oracle"]
    ))
)

assert np.allclose(
    excess_var,
    merged3["adaptive_excess_variance_vs_oracle"],
    atol=1e-12,
    rtol=0,
)

assert np.allclose(
    excess_mse,
    merged3["adaptive_excess_MSE_vs_oracle"],
    atol=1e-12,
    rtol=0,
)

print("PASS: Stored excess variance and excess MSE exactly reproduce the final oracle.")

# ------------------------------------------------------------
# 14. Oracle condition list
# ------------------------------------------------------------
print("\n[14] CONDITIONS WHERE SRS IS THE 5,000-REFERENCE ORACLE")

srs_oracle = ref_stats[
    (ref_stats["oracle_variance_recomputed"] == "SRS")
    | (ref_stats["oracle_MSE_recomputed"] == "SRS")
]

print("Number of conditions with SRS variance or MSE oracle:",
      len(srs_oracle))

if len(srs_oracle):
    print(
        srs_oracle[
            keys +
            [
                "oracle_variance_recomputed",
                "oracle_MSE_recomputed",
                "RE_recomputed",
            ]
        ].to_string(index=False)
    )

# ------------------------------------------------------------
# FINAL
# ------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL VALIDATION STATUS: PASS")
print("=" * 80)
print("No simulations were rerun.")
print("All final numerical quantities were independently recomputed")
print("from the 96,000 adaptive trials and 480,000 reference replicates.")
