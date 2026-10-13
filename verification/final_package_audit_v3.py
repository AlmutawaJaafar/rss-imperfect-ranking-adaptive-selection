#!/usr/bin/env python3
"""Package-level audit for the final RSS/LTS V3 manuscript numbers.

This audit does not rerun simulations. It checks the authoritative final CSVs,
recomputes the principal reported values, and checks the manuscript for stale
exploratory sections/phrases. Run from the project root.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path.home()/"Desktop/LTS_RSS/Tex_gpt"
OUT = ROOT/"rss_results_v3_adaptive_primary"
TEX = ROOT/"main_stage2_adaptive_v3.tex"
SUMMARY = OUT/"adaptive_v3_final_5k_summary.csv"
TRIALS = OUT/"adaptive_v3_trial_results.csv"
REF = OUT/"adaptive_v3_reference_final_5000.csv"
UNC = OUT/"step1_2_final_uncertainty/final_uncertainty_summary.csv"
KEYS=["target_rho","mechanism","ranking_contamination_mode","epsilon","estimator"]

def fail(msg): raise SystemExit("FAIL: "+msg)
def ok(msg): print("PASS: "+msg)

def main():
    for p in [TEX,SUMMARY,TRIALS,REF,UNC]:
        if not p.exists(): fail(f"missing {p}")
    s=pd.read_csv(SUMMARY); t=pd.read_csv(TRIALS); r=pd.read_csv(REF); u=pd.read_csv(UNC)
    if len(s)!=96: fail(f"summary rows={len(s)}")
    if len(t)!=96000: fail(f"trial rows={len(t)}")
    if len(r)!=480000: fail(f"reference rows={len(r)}")
    if t.groupby(KEYS).size().nunique()!=1 or t.groupby(KEYS).size().iloc[0]!=1000: fail("not exactly 1000 trials/condition")
    if r.groupby(KEYS).size().nunique()!=1 or r.groupby(KEYS).size().iloc[0]!=5000: fail("not exactly 5000 references/condition")
    if r.duplicated(KEYS+["replicate"]).any(): fail("duplicate reference rows")
    reps=r.groupby(KEYS)["replicate"].agg(["min","max","nunique"])
    if not ((reps["min"]==0).all() and (reps["max"]==4999).all() and (reps["nunique"]==5000).all()): fail("reference replicate indices are not 0..4999")
    ok("final data structure: 96 conditions, 96,000 trials, 480,000 references")

    sel=(t.selected_method.astype(str)=="RSS")
    overall=float(sel.mean())
    # Oracle map from authoritative reference risks.
    oracle={}
    for k,g in r.groupby(KEYS,sort=False):
        rv=np.var(g.rss_est.to_numpy(float),ddof=1); sv=np.var(g.srs_est.to_numpy(float),ddof=1)
        oracle[k]="RSS" if rv<=sv else "SRS"
    agree=np.array([str(m)==oracle[tuple(row[k] for k in KEYS)] for m,(_,row) in zip(t.selected_method,t.iterrows())])
    if abs(overall-0.839729)>5e-7: fail(f"RSS selection {overall} != 0.839729")
    if abs(float(agree.mean())-0.835375)>5e-7: fail(f"oracle agreement {agree.mean()} != 0.835375")
    maj=[]
    for k,g in t.groupby(KEYS,sort=False):
        chosen="RSS" if (g.selected_method.astype(str)=="RSS").mean()>=0.5 else "SRS"
        maj.append(chosen==oracle[k])
    if sum(maj)!=90: fail(f"majority oracle agreement {sum(maj)}/96 != 90/96")
    ok("primary adaptive values match the final trial/reference data")

    revals=[]
    for k,g in r.groupby(KEYS,sort=False):
        rv=np.var(g.rss_est.to_numpy(float),ddof=1); sv=np.var(g.srs_est.to_numpy(float),ddof=1)
        revals.append(sv/rv)
    mean_re=float(np.mean(revals)); med_re=float(np.median(revals))
    if abs(mean_re-1.389747)>5e-6: fail(f"mean RE {mean_re}")
    if abs(med_re-1.225325)>5e-6: fail(f"median RE {med_re}")
    ok("reference mean/median RE match the authoritative 5,000-reference file")

    txt=TEX.read_text()
    if "Exploratory Multivariate Simulation Study" in txt or "Exploratory Benchmark-Data Applications" in txt:
        fail("exploratory empirical sections still present")
    forbidden=["RSS offers a way forward", "without sacrificing robustness", "computational advantage is established", "MSE-optimal adaptive rule"]
    for phrase in forbidden:
        if phrase.lower() in txt.lower(): fail(f"overclaim remains: {phrase}")
    ok("manuscript has no retained exploratory empirical sections or audited computational overclaims")
    print("FINAL PACKAGE AUDIT PASSED")

if __name__=="__main__": main()
