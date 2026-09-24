#!/usr/bin/env python3
"""
Rebuild the final 96-condition adaptive summary using the existing:
  adaptive_v3.csv
  adaptive_v3_trial_results.csv
  adaptive_v3_reference_final_5000.csv

No simulations are rerun. The original adaptive_v3.csv is preserved.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

ROOT = Path.home() / "Desktop" / "LTS_RSS" / "Tex_gpt"
OUT = ROOT / "rss_results_v3_adaptive_primary"

ADAPTIVE = OUT / "adaptive_v3.csv"
TRIALS = OUT / "adaptive_v3_trial_results.csv"
REFERENCE = OUT / "adaptive_v3_reference_final_5000.csv"
BACKUP = OUT / "adaptive_v3_before_5k_summary_fix.csv"
FINAL = OUT / "adaptive_v3_final_5k_summary.csv"
DIAG = OUT / "final_5k_summary_validation.txt"

COND = ["target_rho","mechanism","ranking_contamination_mode","epsilon","estimator"]

def fail(msg):
    raise RuntimeError(msg)

def main():
    for p in (ADAPTIVE, TRIALS, REFERENCE):
        if not p.exists():
            fail(f"Missing required file: {p}")

    if not BACKUP.exists():
        shutil.copy2(ADAPTIVE, BACKUP)

    adaptive = pd.read_csv(ADAPTIVE)
    trials = pd.read_csv(TRIALS)
    ref = pd.read_csv(REFERENCE)

    if len(adaptive) != 96:
        fail(f"Expected 96 adaptive summary rows, found {len(adaptive)}")
    if len(trials) != 96000:
        fail(f"Expected 96000 trial rows, found {len(trials)}")
    if len(ref) != 480000:
        fail(f"Expected 480000 reference rows, found {len(ref)}")

    for c in COND:
        if c not in adaptive.columns or c not in trials.columns or c not in ref.columns:
            fail(f"Missing condition column: {c}")

    if adaptive[COND].drop_duplicates().shape[0] != 96:
        fail("Adaptive summary does not contain exactly 96 conditions.")
    if trials[COND].drop_duplicates().shape[0] != 96:
        fail("Trial file does not contain exactly 96 conditions.")
    if ref[COND].drop_duplicates().shape[0] != 96:
        fail("Reference file does not contain exactly 96 conditions.")

    tc = trials.groupby(COND).size()
    if tc.min() != 1000 or tc.max() != 1000:
        fail("Trial counts are not exactly 1000 for every condition.")

    rc = ref.groupby(COND).size()
    if rc.min() != 5000 or rc.max() != 5000:
        fail("Reference counts are not exactly 5000 for every condition.")

    reps = ref.groupby(COND)["replicate"].agg(["min","max","nunique"])
    if not ((reps["min"] == 0).all() and (reps["max"] == 4999).all()
            and (reps["nunique"] == 5000).all()):
        fail("Reference replicates are not exactly 0..4999 per condition.")
    if ref.duplicated(COND + ["replicate"]).any():
        fail("Duplicate reference condition/replicate rows found.")

    # Authoritative fixed-strategy 5000-replicate statistics.
    rows = []
    for key, g in ref.groupby(COND, sort=False):
        rss = g["rss_est"].to_numpy(float)
        srs = g["srs_est"].to_numpy(float)
        rv = float(np.var(rss, ddof=1))
        sv = float(np.var(srs, ddof=1))
        rm = float(np.mean(rss**2))
        sm = float(np.mean(srs**2))
        rows.append({
            **dict(zip(COND, key)),
            "reference_reps": 5000,
            "RSS_variance_reference": rv,
            "SRS_variance_reference": sv,
            "RSS_MSE_reference": rm,
            "SRS_MSE_reference": sm,
            "oracle_best_variance": min(rv, sv),
            "oracle_best_MSE": min(rm, sm),
            "oracle_method_variance": "RSS" if rv <= sv else "SRS",
            "oracle_method_MSE": "RSS" if rm <= sm else "SRS",
            "reference_RE_variance": sv / rv,
            "reference_RE_MSE": sm / rm,
        })
    rs = pd.DataFrame(rows)

    # Trial-level agreement against the 5000-reference oracle.
    t = trials.merge(
        rs[COND + ["oracle_method_variance","oracle_method_MSE"]],
        on=COND, how="left", validate="many_to_one"
    )
    t["agreement_variance_oracle"] = (
        t["selected_method"] == t["oracle_method_variance"]
    )
    t["agreement_MSE_oracle"] = (
        t["selected_method"] == t["oracle_method_MSE"]
    )
    agreement = (
        t.groupby(COND)
         .agg(
             decision_accuracy_vs_variance_oracle=("agreement_variance_oracle","mean"),
             decision_accuracy_vs_MSE_oracle=("agreement_MSE_oracle","mean"),
         ).reset_index()
    )

    # Majority decision per condition.
    def majority_method(s):
        return "RSS" if (s == "RSS").sum() >= (s == "SRS").sum() else "SRS"

    maj = trials.groupby(COND)["selected_method"].agg(majority_selected_method=majority_method).reset_index()
    maj = maj.merge(
        rs[COND + ["oracle_method_variance","oracle_method_MSE"]],
        on=COND, how="left", validate="one_to_one"
    )
    maj["majority_agreement_variance_oracle"] = (
        maj["majority_selected_method"] == maj["oracle_method_variance"]
    )
    maj["majority_agreement_MSE_oracle"] = (
        maj["majority_selected_method"] == maj["oracle_method_MSE"]
    )

    # Preserve all adaptive-run quantities; replace only reference/oracle fields.
    dropcols = [
        "reference_reps","RSS_variance_reference","SRS_variance_reference",
        "RSS_MSE_reference","SRS_MSE_reference","oracle_best_variance",
        "oracle_best_MSE","adaptive_excess_variance_vs_oracle",
        "adaptive_excess_MSE_vs_oracle","oracle_method_variance",
        "oracle_method_MSE","reference_RE_variance","reference_RE_MSE",
        "decision_accuracy_vs_variance_oracle","decision_accuracy_vs_MSE_oracle",
        "majority_selected_method","majority_agreement_variance_oracle",
        "majority_agreement_MSE_oracle",
    ]
    final = adaptive.drop(columns=[c for c in dropcols if c in adaptive.columns])
    final = final.merge(rs, on=COND, how="left", validate="one_to_one")
    final = final.merge(agreement, on=COND, how="left", validate="one_to_one")
    final = final.merge(
        maj[COND + ["majority_selected_method",
                    "majority_agreement_variance_oracle",
                    "majority_agreement_MSE_oracle"]],
        on=COND, how="left", validate="one_to_one"
    )

    final["adaptive_excess_variance_vs_oracle"] = (
        final["adaptive_variance"] - final["oracle_best_variance"]
    )
    final["adaptive_excess_MSE_vs_oracle"] = (
        final["adaptive_MSE"] - final["oracle_best_MSE"]
    )
    final["adaptive_relative_oracle_regret_percent_variance"] = (
        100 * final["adaptive_excess_variance_vs_oracle"] / final["oracle_best_variance"]
    )
    final["adaptive_relative_oracle_regret_percent_MSE"] = (
        100 * final["adaptive_excess_MSE_vs_oracle"] / final["oracle_best_MSE"]
    )
    final = final.sort_values(COND).reset_index(drop=True)

    # Preservation audit.
    original = adaptive.set_index(COND)
    rebuilt = final.set_index(COND)
    preserve = [
        "RSS_selection_fraction","pilot_mean_RSS_variance",
        "pilot_mean_SRS_variance","adaptive_variance","adaptive_MSE",
        "adaptive_trials","pilot_reps","pilot_cycles"
    ]
    diffs = {}
    for c in preserve:
        if c in original.columns:
            diffs[c] = float(np.nanmax(np.abs(
                pd.to_numeric(original[c]) - pd.to_numeric(rebuilt[c])
            )))

    final.to_csv(FINAL, index=False)

    var_oracle = rs["oracle_method_variance"].value_counts().to_dict()
    mse_oracle = rs["oracle_method_MSE"].value_counts().to_dict()
    overall_rss = float((trials["selected_method"] == "RSS").mean())
    trial_var = float(t["agreement_variance_oracle"].mean())
    trial_mse = float(t["agreement_MSE_oracle"].mean())
    maj_var = int(maj["majority_agreement_variance_oracle"].sum())
    maj_mse = int(maj["majority_agreement_MSE_oracle"].sum())

    rv_re = rs["reference_RE_variance"]
    rm_re = rs["reference_RE_MSE"]

    lines = [
        "FINAL 5000-REFERENCE SUMMARY VALIDATION",
        "========================================",
        "",
        "PASS: 96 adaptive conditions.",
        "PASS: 96 x 1000 adaptive trials.",
        "PASS: 96 x 5000 fixed-strategy reference replicates.",
        "PASS: reference replicates are exactly 0..4999 per condition.",
        "PASS: no duplicate reference condition/replicate rows.",
        "",
        "5000-REFERENCE ORACLE",
        f"Variance oracle: RSS={var_oracle.get('RSS',0)}, SRS={var_oracle.get('SRS',0)}",
        f"MSE oracle: RSS={mse_oracle.get('RSS',0)}, SRS={mse_oracle.get('SRS',0)}",
        f"Mean RE (variance): {rv_re.mean():.6f}",
        f"Median RE (variance): {rv_re.median():.6f}",
        f"Range RE (variance): {rv_re.min():.6f}--{rv_re.max():.6f}",
        f"Mean RE (MSE): {rm_re.mean():.6f}",
        f"Median RE (MSE): {rm_re.median():.6f}",
        f"Range RE (MSE): {rm_re.min():.6f}--{rm_re.max():.6f}",
        "",
        "ADAPTIVE 1000-TRIAL RESULTS",
        f"Overall RSS selection fraction: {overall_rss:.6f}",
        f"Trial variance-oracle agreement: {trial_var:.6f}",
        f"Trial MSE-oracle agreement: {trial_mse:.6f}",
        f"Majority variance-oracle agreement: {maj_var}/96 = {maj_var/96:.6f}",
        f"Majority MSE-oracle agreement: {maj_mse}/96 = {maj_mse/96:.6f}",
        "",
        "PRESERVATION AUDIT",
    ]
    for c, d in diffs.items():
        lines.append(f"{c}: max abs difference = {d:.3e}")
    if all(d <= 1e-14 for d in diffs.values()):
        lines.append("PASS: all adaptive-run quantities are unchanged.")
    else:
        lines.append("FAIL: at least one adaptive-run quantity changed.")

    lines += [
        "",
        f"Backup: {BACKUP}",
        f"Final:  {FINAL}",
        f"Diagnostics: {DIAG}",
        "",
        "The original adaptive_v3.csv was not overwritten."
    ]
    DIAG.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nWrote: {FINAL}")

if __name__ == "__main__":
    main()
