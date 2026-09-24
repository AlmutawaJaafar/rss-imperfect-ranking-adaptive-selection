#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE = Path("rss_results_v3_adaptive_primary")

FILE_2000 = BASE / "adaptive_v3_reference_replicates_2000.csv"
FILE_5000 = BASE / "adaptive_v3_reference_final_5000.csv"

OUT_DIR = BASE / "step1_2_reference_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "adaptive_v3_reference_2000_vs_5000.csv"
OUT_SUMMARY = OUT_DIR / "adaptive_v3_reference_2000_vs_5000_summary.txt"


# ---------------------------------------------------------------------
# Condition definition
# ---------------------------------------------------------------------
CONDITION_COLS = [
    "target_rho",
    "mechanism",
    "ranking_contamination_mode",
    "epsilon",
    "estimator",
]


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
print("=" * 78)
print("STEP 1.2: 2,000 vs 5,000 REFERENCE COMPARISON")
print("=" * 78)

print("\n[1/6] Loading reference files...")

df2000 = pd.read_csv(FILE_2000)
df5000 = pd.read_csv(FILE_5000)

print(f"2,000-replicate rows: {len(df2000)}")
print(f"5,000-replicate rows: {len(df5000)}")


# ---------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------
print("\n[2/6] Validating condition structure...")

for name, df, expected_reps in [
    ("2,000", df2000, 2000),
    ("5,000", df5000, 5000),
]:
    n_conditions = df[CONDITION_COLS].drop_duplicates().shape[0]

    counts = df.groupby(CONDITION_COLS).size()

    print(
        f"{name}: conditions={n_conditions}, "
        f"min_reps={counts.min()}, "
        f"max_reps={counts.max()}, "
        f"bad_conditions={(counts != expected_reps).sum()}"
    )

    if n_conditions != 96:
        raise RuntimeError(
            f"{name}-replicate file does not contain exactly 96 conditions."
        )

    if (counts != expected_reps).any():
        raise RuntimeError(
            f"{name}-replicate file does not have exactly "
            f"{expected_reps} replicates per condition."
        )


# ---------------------------------------------------------------------
# Recompute reference statistics from replicate-level data
# ---------------------------------------------------------------------
print("\n[3/6] Computing reference statistics...")

def summarize(df):
    rows = []

    for keys, g in df.groupby(CONDITION_COLS, sort=True):

        rss = g["rss_est"].to_numpy(dtype=float)
        srs = g["srs_est"].to_numpy(dtype=float)

        rss_var = float(np.var(rss, ddof=1))
        srs_var = float(np.var(srs, ddof=1))

        rss_mse = float(np.mean(rss ** 2))
        srs_mse = float(np.mean(srs ** 2))

        rows.append({
            **dict(zip(CONDITION_COLS, keys)),
            "reference_reps": len(g),

            "RSS_variance": rss_var,
            "SRS_variance": srs_var,

            "RSS_MSE": rss_mse,
            "SRS_MSE": srs_mse,

            "oracle_variance":
                "RSS" if rss_var <= srs_var else "SRS",

            "oracle_MSE":
                "RSS" if rss_mse <= srs_mse else "SRS",
        })

    return pd.DataFrame(rows)


s2000 = summarize(df2000)
s5000 = summarize(df5000)

if len(s2000) != 96 or len(s5000) != 96:
    raise RuntimeError(
        "Unexpected number of summarized conditions."
    )


# ---------------------------------------------------------------------
# Merge and compare
# ---------------------------------------------------------------------
print("\n[4/6] Comparing 2,000 vs 5,000 results...")

left = s2000.rename(columns={
    "reference_reps": "reference_reps_2000",
    "RSS_variance": "RSS_variance_2000",
    "SRS_variance": "SRS_variance_2000",
    "RSS_MSE": "RSS_MSE_2000",
    "SRS_MSE": "SRS_MSE_2000",
    "oracle_variance": "oracle_variance_2000",
    "oracle_MSE": "oracle_MSE_2000",
})

right = s5000.rename(columns={
    "reference_reps": "reference_reps_5000",
    "RSS_variance": "RSS_variance_5000",
    "SRS_variance": "SRS_variance_5000",
    "RSS_MSE": "RSS_MSE_5000",
    "SRS_MSE": "SRS_MSE_5000",
    "oracle_variance": "oracle_variance_5000",
    "oracle_MSE": "oracle_MSE_5000",
})

comparison = left.merge(
    right,
    on=CONDITION_COLS,
    how="outer",
    validate="one_to_one",
)

if len(comparison) != 96:
    raise RuntimeError(
        f"Expected 96 merged conditions; found {len(comparison)}."
    )


# Absolute changes
comparison["delta_RSS_variance"] = (
    comparison["RSS_variance_5000"]
    - comparison["RSS_variance_2000"]
)

comparison["delta_SRS_variance"] = (
    comparison["SRS_variance_5000"]
    - comparison["SRS_variance_2000"]
)

comparison["delta_RSS_MSE"] = (
    comparison["RSS_MSE_5000"]
    - comparison["RSS_MSE_2000"]
)

comparison["delta_SRS_MSE"] = (
    comparison["SRS_MSE_5000"]
    - comparison["SRS_MSE_2000"]
)


# Relative changes (%)
def safe_relative(new, old):
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)

    out = np.full_like(old, np.nan, dtype=float)
    mask = old != 0

    out[mask] = 100.0 * (new[mask] - old[mask]) / old[mask]

    return out


comparison["relative_change_RSS_variance_percent"] = safe_relative(
    comparison["RSS_variance_5000"],
    comparison["RSS_variance_2000"],
)

comparison["relative_change_SRS_variance_percent"] = safe_relative(
    comparison["SRS_variance_5000"],
    comparison["SRS_variance_2000"],
)

comparison["relative_change_RSS_MSE_percent"] = safe_relative(
    comparison["RSS_MSE_5000"],
    comparison["RSS_MSE_2000"],
)

comparison["relative_change_SRS_MSE_percent"] = safe_relative(
    comparison["SRS_MSE_5000"],
    comparison["SRS_MSE_2000"],
)


# Oracle changes
comparison["variance_oracle_changed"] = (
    comparison["oracle_variance_2000"]
    != comparison["oracle_variance_5000"]
)

comparison["MSE_oracle_changed"] = (
    comparison["oracle_MSE_2000"]
    != comparison["oracle_MSE_5000"]
)


# Absolute gaps between RSS and SRS, useful for checking whether
# small numerical changes can alter the qualitative conclusion.
comparison["variance_gap_2000"] = (
    comparison["RSS_variance_2000"]
    - comparison["SRS_variance_2000"]
)

comparison["variance_gap_5000"] = (
    comparison["RSS_variance_5000"]
    - comparison["SRS_variance_5000"]
)

comparison["MSE_gap_2000"] = (
    comparison["RSS_MSE_2000"]
    - comparison["SRS_MSE_2000"]
)

comparison["MSE_gap_5000"] = (
    comparison["RSS_MSE_5000"]
    - comparison["SRS_MSE_5000"]
)


# ---------------------------------------------------------------------
# Aggregate diagnostics
# ---------------------------------------------------------------------
print("\n[5/6] Computing aggregate diagnostics...")

def max_abs(series):
    return float(np.nanmax(np.abs(series)))


n_variance_changes = int(
    comparison["variance_oracle_changed"].sum()
)

n_mse_changes = int(
    comparison["MSE_oracle_changed"].sum()
)

rss_variance_relative = comparison[
    "relative_change_RSS_variance_percent"
]

srs_variance_relative = comparison[
    "relative_change_SRS_variance_percent"
]

rss_mse_relative = comparison[
    "relative_change_RSS_MSE_percent"
]

srs_mse_relative = comparison[
    "relative_change_SRS_MSE_percent"
]

# Overall oracle counts
oracle_var_2000 = (
    comparison["oracle_variance_2000"]
    .value_counts()
    .to_dict()
)

oracle_var_5000 = (
    comparison["oracle_variance_5000"]
    .value_counts()
    .to_dict()
)

oracle_mse_2000 = (
    comparison["oracle_MSE_2000"]
    .value_counts()
    .to_dict()
)

oracle_mse_5000 = (
    comparison["oracle_MSE_5000"]
    .value_counts()
    .to_dict()
)


# ---------------------------------------------------------------------
# Write detailed comparison CSV
# ---------------------------------------------------------------------
comparison.to_csv(
    OUT_CSV,
    index=False
)


# ---------------------------------------------------------------------
# Write human-readable summary
# ---------------------------------------------------------------------
with open(OUT_SUMMARY, "w", encoding="utf-8") as f:

    f.write("=" * 78 + "\n")
    f.write("STEP 1.2: 2,000 vs 5,000 REFERENCE COMPARISON\n")
    f.write("=" * 78 + "\n\n")

    f.write("Input files\n")
    f.write(f"  2,000 reps: {FILE_2000}\n")
    f.write(f"  5,000 reps: {FILE_5000}\n\n")

    f.write("Experiment structure\n")
    f.write("  Conditions: 96\n")
    f.write("  2,000-replicate file: 2,000 per condition\n")
    f.write("  5,000-replicate file: 5,000 per condition\n")
    f.write("  Total 2,000-replicate rows: 192,000\n")
    f.write("  Total 5,000-replicate rows: 480,000\n\n")

    f.write("Oracle-method stability\n")
    f.write(
        f"  Variance-oracle changes: {n_variance_changes}/96\n"
    )
    f.write(
        f"  MSE-oracle changes: {n_mse_changes}/96\n\n"
    )

    f.write("Variance-oracle counts\n")
    f.write(
        f"  2,000 reps: RSS={oracle_var_2000.get('RSS', 0)}, "
        f"SRS={oracle_var_2000.get('SRS', 0)}\n"
    )
    f.write(
        f"  5,000 reps: RSS={oracle_var_5000.get('RSS', 0)}, "
        f"SRS={oracle_var_5000.get('SRS', 0)}\n\n"
    )

    f.write("MSE-oracle counts\n")
    f.write(
        f"  2,000 reps: RSS={oracle_mse_2000.get('RSS', 0)}, "
        f"SRS={oracle_mse_2000.get('SRS', 0)}\n"
    )
    f.write(
        f"  5,000 reps: RSS={oracle_mse_5000.get('RSS', 0)}, "
        f"SRS={oracle_mse_5000.get('SRS', 0)}\n\n"
    )

    f.write("Maximum absolute relative changes\n")
    f.write(
        "  RSS variance: "
        f"{max_abs(rss_variance_relative):.6f}%\n"
    )
    f.write(
        "  SRS variance: "
        f"{max_abs(srs_variance_relative):.6f}%\n"
    )
    f.write(
        "  RSS MSE: "
        f"{max_abs(rss_mse_relative):.6f}%\n"
    )
    f.write(
        "  SRS MSE: "
        f"{max_abs(srs_mse_relative):.6f}%\n\n"
    )

    f.write("Mean absolute relative changes\n")
    f.write(
        "  RSS variance: "
        f"{np.nanmean(np.abs(rss_variance_relative)):.6f}%\n"
    )
    f.write(
        "  SRS variance: "
        f"{np.nanmean(np.abs(srs_variance_relative)):.6f}%\n"
    )
    f.write(
        "  RSS MSE: "
        f"{np.nanmean(np.abs(rss_mse_relative)):.6f}%\n"
    )
    f.write(
        "  SRS MSE: "
        f"{np.nanmean(np.abs(srs_mse_relative)):.6f}%\n\n"
    )

    f.write("Conditions with changed oracle conclusions\n")

    changed_variance = comparison[
        comparison["variance_oracle_changed"]
    ]

    if len(changed_variance) == 0:
        f.write("  Variance oracle: NONE\n")
    else:
        f.write(
            f"  Variance oracle: {len(changed_variance)} conditions\n"
        )
        f.write(
            changed_variance[
                CONDITION_COLS
                + [
                    "oracle_variance_2000",
                    "oracle_variance_5000",
                    "variance_gap_2000",
                    "variance_gap_5000",
                ]
            ].to_string(index=False)
        )
        f.write("\n")

    changed_mse = comparison[
        comparison["MSE_oracle_changed"]
    ]

    if len(changed_mse) == 0:
        f.write("  MSE oracle: NONE\n")
    else:
        f.write(
            f"  MSE oracle: {len(changed_mse)} conditions\n"
        )
        f.write(
            changed_mse[
                CONDITION_COLS
                + [
                    "oracle_MSE_2000",
                    "oracle_MSE_5000",
                    "MSE_gap_2000",
                    "MSE_gap_5000",
                ]
            ].to_string(index=False)
        )
        f.write("\n")

    f.write("\n")
    f.write("Interpretation status\n")

    if n_variance_changes == 0 and n_mse_changes == 0:
        f.write(
            "  No condition changes its variance or MSE oracle method "
            "when increasing the reference experiment from 2,000 to "
            "5,000 replicates.\n"
        )
        f.write(
            "  Therefore the qualitative oracle conclusions are stable "
            "under this replication increase.\n"
        )
    else:
        f.write(
            "  At least one condition changes its oracle method.\n"
        )
        f.write(
            "  Manuscript conclusions depending on the affected oracle "
            "conditions require inspection before being updated.\n"
        )


# ---------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------
print("\n[6/6] Comparison complete.")
print()
print(f"Detailed CSV:")
print(f"  {OUT_CSV}")
print()
print(f"Summary report:")
print(f"  {OUT_SUMMARY}")
print()

print("Oracle-method changes:")
print(f"  Variance oracle: {n_variance_changes}/96")
print(f"  MSE oracle:      {n_mse_changes}/96")

print()
print("Maximum absolute relative changes:")
print(
    f"  RSS variance: {max_abs(rss_variance_relative):.6f}%"
)
print(
    f"  SRS variance: {max_abs(srs_variance_relative):.6f}%"
)
print(
    f"  RSS MSE:      {max_abs(rss_mse_relative):.6f}%"
)
print(
    f"  SRS MSE:      {max_abs(srs_mse_relative):.6f}%"
)

print()
print("=" * 78)
