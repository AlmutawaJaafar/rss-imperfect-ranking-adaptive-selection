from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("rss_results_v3_adaptive_primary")

F200 = ROOT / "adaptive_v3_trial_results_200.csv"
F1000 = ROOT / "adaptive_v3_trial_results_1000.csv"

# Final 5,000-replicate fixed-strategy reference
REF = ROOT / "adaptive_v3_reference_final_5000.csv"

OUT = ROOT / "step1_3_adaptive_stability"
OUT.mkdir(exist_ok=True)

KEYS = [
    "target_rho",
    "mechanism",
    "ranking_contamination_mode",
    "epsilon",
    "estimator",
]

print("Loading 200-trial adaptive results...")
d200 = pd.read_csv(F200)

print("Loading 1000-trial adaptive results...")
d1000 = pd.read_csv(F1000)

print("Loading final 5,000-replicate reference...")
ref = pd.read_csv(REF)

# ------------------------------------------------------------
# Basic structural checks
# ------------------------------------------------------------
print("\n=== STRUCTURAL CHECKS ===")

for name, df, expected_trials in [
    ("200", d200, 200),
    ("1000", d1000, 1000),
]:
    n_conditions = df[KEYS].drop_duplicates().shape[0]
    counts = df.groupby(KEYS).size()

    print(
        f"{name}: rows={len(df)}, conditions={n_conditions}, "
        f"min_trials={counts.min()}, max_trials={counts.max()}"
    )

    assert n_conditions == 96
    assert counts.min() == expected_trials
    assert counts.max() == expected_trials

assert ref[KEYS].drop_duplicates().shape[0] == 96

# ------------------------------------------------------------
# Selection fraction summaries
# ------------------------------------------------------------
def selection_summary(df):
    x = (
        df.assign(rss_selected=(df["selected_method"] == "RSS").astype(int))
        .groupby(KEYS)["rss_selected"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "rss_selection_fraction"})
    )
    return x

s200 = selection_summary(d200)
s1000 = selection_summary(d1000)

merged = s200.merge(
    s1000,
    on=KEYS,
    suffixes=("_200", "_1000"),
)

merged["abs_change"] = (
    merged["rss_selection_fraction_1000"]
    - merged["rss_selection_fraction_200"]
).abs()

merged["signed_change"] = (
    merged["rss_selection_fraction_1000"]
    - merged["rss_selection_fraction_200"]
)

# ------------------------------------------------------------
# Overall / median / per-rho selection fractions
# ------------------------------------------------------------
overall200 = (d200["selected_method"] == "RSS").mean()
overall1000 = (d1000["selected_method"] == "RSS").mean()

median200 = s200["rss_selection_fraction"].median()
median1000 = s1000["rss_selection_fraction"].median()

rho200 = (
    d200.assign(rss_selected=(d200["selected_method"] == "RSS").astype(int))
    .groupby("target_rho")["rss_selected"]
    .mean()
)

rho1000 = (
    d1000.assign(rss_selected=(d1000["selected_method"] == "RSS").astype(int))
    .groupby("target_rho")["rss_selected"]
    .mean()
)

# ------------------------------------------------------------
# Trial-level oracle agreement using final 5k reference
# ------------------------------------------------------------
ref_oracle = (
    ref.groupby(KEYS)
    .agg(
        rss_variance=("rss_est", "var"),
        srs_variance=("srs_est", "var"),
        rss_mse=("rss_est", lambda x: np.mean((x - x.mean()) ** 2)),
        srs_mse=("srs_est", lambda x: np.mean((x - x.mean()) ** 2)),
    )
    .reset_index()
)

ref_oracle["oracle_variance"] = np.where(
    ref_oracle["rss_variance"] <= ref_oracle["srs_variance"],
    "RSS",
    "SRS",
)

ref_oracle["oracle_mse"] = np.where(
    ref_oracle["rss_mse"] <= ref_oracle["srs_mse"],
    "RSS",
    "SRS",
)

def oracle_agreement(df, oracle_col):
    x = df.merge(
        ref_oracle[KEYS + [oracle_col]],
        on=KEYS,
        how="left",
    )
    return (
        x["selected_method"].values
        == x[oracle_col].values
    ).mean()

agree200_var = oracle_agreement(d200, "oracle_variance")
agree1000_var = oracle_agreement(d1000, "oracle_variance")

agree200_mse = oracle_agreement(d200, "oracle_mse")
agree1000_mse = oracle_agreement(d1000, "oracle_mse")

# ------------------------------------------------------------
# Majority condition-level oracle agreement
# ------------------------------------------------------------
def majority_agreement(df, oracle_col):
    x = df.merge(
        ref_oracle[KEYS + [oracle_col]],
        on=KEYS,
        how="left",
    )

    majority = (
        x.groupby(KEYS)["selected_method"]
        .agg(lambda z: z.value_counts().idxmax())
        .reset_index(name="adaptive_majority")
    )

    oracle = ref_oracle[KEYS + [oracle_col]]

    m = majority.merge(oracle, on=KEYS)

    return (m["adaptive_majority"] == m[oracle_col]).mean(), m

maj200_var, majtab200 = majority_agreement(d200, "oracle_variance")
maj1000_var, majtab1000 = majority_agreement(d1000, "oracle_variance")

# ------------------------------------------------------------
# Majority-condition changes between 200 and 1000
# ------------------------------------------------------------
maj200 = (
    d200.groupby(KEYS)["selected_method"]
    .agg(lambda z: z.value_counts().idxmax())
    .reset_index(name="majority_200")
)

maj1000 = (
    d1000.groupby(KEYS)["selected_method"]
    .agg(lambda z: z.value_counts().idxmax())
    .reset_index(name="majority_1000")
)

maj_changes = maj200.merge(maj1000, on=KEYS)

maj_changes["changed"] = (
    maj_changes["majority_200"] != maj_changes["majority_1000"]
)

# ------------------------------------------------------------
# Save condition-level selection comparison
# ------------------------------------------------------------
merged.to_csv(
    OUT / "adaptive_selection_fraction_200_vs_1000.csv",
    index=False,
)

maj_changes.to_csv(
    OUT / "adaptive_majority_200_vs_1000.csv",
    index=False,
)

# ------------------------------------------------------------
# Print final comparison
# ------------------------------------------------------------
print("\n=== STEP 1.3 ADAPTIVE STABILITY ===")

print(f"Overall RSS selection:")
print(f"  200 trials : {overall200:.6f}")
print(f"  1000 trials: {overall1000:.6f}")
print(f"  absolute change: {abs(overall1000-overall200):.6f}")

print(f"\nMedian condition RSS selection:")
print(f"  200 trials : {median200:.6f}")
print(f"  1000 trials: {median1000:.6f}")
print(f"  absolute change: {abs(median1000-median200):.6f}")

print("\nRSS selection by rho:")
for rho in sorted(rho200.index):
    print(
        f"  rho={rho:.2f}: "
        f"200={rho200[rho]:.6f}, "
        f"1000={rho1000[rho]:.6f}, "
        f"change={abs(rho1000[rho]-rho200[rho]):.6f}"
    )

print("\nTrial-level variance-oracle agreement:")
print(f"  200 trials : {agree200_var:.6f}")
print(f"  1000 trials: {agree1000_var:.6f}")
print(f"  change     : {agree1000_var-agree200_var:+.6f}")

print("\nTrial-level MSE-oracle agreement:")
print(f"  200 trials : {agree200_mse:.6f}")
print(f"  1000 trials: {agree1000_mse:.6f}")
print(f"  change     : {agree1000_mse-agree200_mse:+.6f}")

print("\nMajority condition-level variance-oracle agreement:")
print(f"  200 trials : {maj200_var:.6f}")
print(f"  1000 trials: {maj1000_var:.6f}")

print(f"\nMajority condition changes:")
print(f"  changed conditions: {maj_changes['changed'].sum()} / 96")

print("\nMaximum condition-level RSS selection-fraction change:")
print(f"  {merged['abs_change'].max():.6f}")

largest = merged.loc[merged["abs_change"].idxmax()]
print("\nCondition with largest change:")
for k in KEYS:
    print(f"  {k}: {largest[k]}")
print(f"  200 trials : {largest['rss_selection_fraction_200']:.6f}")
print(f"  1000 trials: {largest['rss_selection_fraction_1000']:.6f}")

print("\nOutputs:")
print(OUT / "adaptive_selection_fraction_200_vs_1000.csv")
print(OUT / "adaptive_majority_200_vs_1000.csv")
