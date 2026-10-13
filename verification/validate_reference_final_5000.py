import pandas as pd
from pathlib import Path

p = Path("rss_results_v3_adaptive_primary/adaptive_v3_reference_final_5000.csv")

df = pd.read_csv(p)

expected_conditions = 96
expected_reps = 5000

keys = [
    "target_rho",
    "mechanism",
    "ranking_contamination_mode",
    "epsilon",
    "estimator",
]

required = keys + ["replicate", "rss_est", "srs_est"]

print("Columns:")
print(list(df.columns))
print()

missing = [c for c in required if c not in df.columns]
if missing:
    raise AssertionError(f"Missing required columns: {missing}")

print(f"Total rows: {len(df)}")

n_conditions = df[keys].drop_duplicates().shape[0]
print(f"Unique conditions: {n_conditions}")

counts = df.groupby(keys, dropna=False).size()
print(f"Minimum replicates/condition: {counts.min()}")
print(f"Maximum replicates/condition: {counts.max()}")

rep_min = df["replicate"].min()
rep_max = df["replicate"].max()
print(f"Replicate index range: {rep_min} to {rep_max}")

if len(df) != expected_conditions * expected_reps:
    raise AssertionError(
        f"Expected {expected_conditions * expected_reps} rows, got {len(df)}"
    )

if n_conditions != expected_conditions:
    raise AssertionError(
        f"Expected {expected_conditions} conditions, got {n_conditions}"
    )

if counts.min() != expected_reps or counts.max() != expected_reps:
    raise AssertionError(
        f"Not every condition has exactly {expected_reps} replicates"
    )

if rep_min != 0 or rep_max != expected_reps - 1:
    raise AssertionError(
        f"Unexpected replicate range: {rep_min}..{rep_max}"
    )

# Check uniqueness of condition + replicate
dup = df.duplicated(keys + ["replicate"]).sum()
print(f"Duplicate condition/replicate rows: {dup}")

if dup != 0:
    raise AssertionError("Duplicate condition/replicate combinations found")

# Check expected parameter values
expected_rho = {0.25, 0.50, 0.75, 0.90}
expected_mechanisms = {"gaussian", "laplace"}
expected_modes = {"pre_ranking", "post_ranking"}
expected_eps = {0.0, 0.10}
expected_estimators = {"mean", "trimmed", "huber"}

print()
print("Unique target_rho:", sorted(df["target_rho"].unique()))
print("Unique mechanisms:", sorted(df["mechanism"].unique()))
print("Unique ranking modes:", sorted(df["ranking_contamination_mode"].unique()))
print("Unique epsilon:", sorted(df["epsilon"].unique()))
print("Unique estimators:", sorted(df["estimator"].unique()))

if set(df["target_rho"].unique()) != expected_rho:
    raise AssertionError("Unexpected target_rho values")

if set(df["mechanism"].unique()) != expected_mechanisms:
    raise AssertionError("Unexpected mechanism values")

if set(df["ranking_contamination_mode"].unique()) != expected_modes:
    raise AssertionError("Unexpected ranking-contamination modes")

if set(df["epsilon"].unique()) != expected_eps:
    raise AssertionError("Unexpected epsilon values")

if set(df["estimator"].unique()) != expected_estimators:
    raise AssertionError("Unexpected estimator values")

# Numerical sanity checks
if df[["rss_est", "srs_est"]].isna().any().any():
    raise AssertionError("NaN values found in rss_est or srs_est")

if not df["rss_est"].map(lambda x: pd.api.types.is_number(x)).all():
    raise AssertionError("Non-numeric rss_est values found")

if not df["srs_est"].map(lambda x: pd.api.types.is_number(x)).all():
    raise AssertionError("Non-numeric srs_est values found")

print()
print("PASS: Final 5,000-reference file is structurally valid.")
print(f"PASS: {expected_conditions} conditions × {expected_reps} replicates = {len(df):,} rows.")
print("PASS: Replicates are exactly 0..4999 for every condition.")
print("PASS: No duplicate condition/replicate combinations.")
print("PASS: Parameter grid matches the intended 96-condition design.")
print("PASS: rss_est and srs_est contain no missing values.")
