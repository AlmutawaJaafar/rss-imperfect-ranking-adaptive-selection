#!/usr/bin/env python3
"""Step 1.4: adaptive pilot-budget sensitivity for V3.

Runs pilot budgets 10,20,30,50,100,200 using the *same deterministic
trial seed construction* as rss_adaptive_experiments_v3.py. For each
adaptive trial, 200 pilot datasets are generated once; all six prefix
variance estimates are then evaluated from those same 200 observations.
The independent final evaluation sample is generated once per trial.

This is intentionally a diagnostic run: it does not modify the primary
V3 outputs.
"""
from __future__ import annotations

import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# The user should run this file from the project root or place it there.
PROJECT = Path.home() / "Desktop" / "LTS_RSS" / "Tex_gpt"
if not (PROJECT / "rss_adaptive_experiments_v3.py").exists():
    PROJECT = ROOT
sys.path.insert(0, str(PROJECT))
import rss_adaptive_experiments_v3 as v

SEED = 20260915
M = 5
CYCLES = 100
PILOT_CYCLES = 20
ADAPTIVE_TRIALS = 1000
MAX_PILOT_REPS = 200
BUDGETS = (10, 20, 30, 50, 100, 200)
TARGETS = (0.25, 0.50, 0.75, 0.90)
MECHS = ("gaussian", "laplace")
MODES = ("post_ranking", "pre_ranking")
EPS = (0.0, 0.10)
ESTIMATORS = ("mean", "trimmed", "huber")
SCALES = {
    (0.25, "gaussian"): 5.56036,
    (0.25, "laplace"): 4.76666,
    (0.50, "gaussian"): 2.50672,
    (0.50, "laplace"): 2.07187,
    (0.75, "gaussian"): 1.33957,
    (0.75, "laplace"): 1.03538,
    (0.90, "gaussian"): 0.795978,
    (0.90, "laplace"): 0.579369,
}

OUTDIR = PROJECT / "rss_results_v3_adaptive_primary" / "step1_4_pilot_sensitivity"


def condition_jobs():
    for target in TARGETS:
        for mech in MECHS:
            for mode in MODES:
                for eps in EPS:
                    for estimator in ESTIMATORS:
                        yield (target, mech, mode, eps, estimator)


def base_seed(target, mech, mode, eps, estimator, a):
    return (
        SEED + 4000000
        + int(round(1000 * target)) * 10000
        + (0 if mech == "gaussian" else 1) * 1000
        + (0 if mode == "post_ranking" else 100)
        + int(round(1000 * eps)) * 10
        + v.ESTIMATOR_NAMES.index(estimator) * 2
        + a
    )


def run_condition(job):
    target, mech, mode, eps, estimator = job
    scale = SCALES[(target, mech)]
    cfg = v.Config(
        seed=SEED, reps=1000, cycles=CYCLES, m=M,
        adaptive_trials=ADAPTIVE_TRIALS,
        pilot_reps=MAX_PILOT_REPS,
        pilot_cycles=PILOT_CYCLES,
        eps_values=EPS,
    )
    rows = []
    for a in range(ADAPTIVE_TRIALS):
        bs = base_seed(target, mech, mode, eps, estimator, a)
        rng = np.random.default_rng(bs)
        rss_vals = np.empty(MAX_PILOT_REPS, dtype=float)
        srs_vals = np.empty(MAX_PILOT_REPS, dtype=float)
        for j in range(MAX_PILOT_REPS):
            out = v.simulate_dataset(
                M, PILOT_CYCLES, mech, scale, estimator, cfg, rng,
                epsilon=eps, contamination_mode=mode,
            )
            rss_vals[j] = out["rss_est"]
            srs_vals[j] = out["srs_est"]
        # Independent final evaluation, exactly as the primary experiment.
        eval_rng = np.random.default_rng(bs + 1)
        final = v.simulate_dataset(
            M, CYCLES, mech, scale, estimator, cfg, eval_rng,
            epsilon=eps, contamination_mode=mode,
        )
        row = dict(target_rho=target, mechanism=mech,
                   ranking_contamination_mode=mode, epsilon=eps,
                   estimator=estimator, adaptive_trial=a,
                   evaluation_seed=bs+1,
                   rss_estimate=final["rss_est"], srs_estimate=final["srs_est"])
        for b in BUDGETS:
            rv = float(np.var(rss_vals[:b], ddof=1))
            sv = float(np.var(srs_vals[:b], ddof=1))
            row[f"pilot_rss_var_{b}"] = rv
            row[f"pilot_srs_var_{b}"] = sv
            row[f"selected_{b}"] = "RSS" if rv <= sv else "SRS"
        rows.append(row)
    return rows


def oracle_from_reference(ref_path):
    df = pd.read_csv(ref_path)
    keys = ["target_rho","mechanism","ranking_contamination_mode","epsilon","estimator"]
    out=[]
    for key, g in df.groupby(keys, sort=False):
        rv=float(np.var(g.rss_est.to_numpy(float), ddof=1))
        sv=float(np.var(g.srs_est.to_numpy(float), ddof=1))
        rm=float(np.mean(g.rss_est.to_numpy(float)**2))
        sm=float(np.mean(g.srs_est.to_numpy(float)**2))
        out.append(dict(zip(keys,key), oracle_variance="RSS" if rv<=sv else "SRS",
                           oracle_mse="RSS" if rm<=sm else "SRS",
                           rss_var=rv,srs_var=sv,rss_mse=rm,srs_mse=sm))
    return pd.DataFrame(out)


def main():
    t0=time.time(); OUTDIR.mkdir(parents=True, exist_ok=True)
    ref = PROJECT / "rss_results_v3_adaptive_primary" / "adaptive_v3_reference_final_5000.csv"
    if not ref.exists():
        raise FileNotFoundError(f"Missing final 5,000-reference file: {ref}")
    print(f"Project: {PROJECT}")
    print(f"Output:  {OUTDIR}")
    print(f"Budgets: {BUDGETS}; conditions={96}; trials/condition={ADAPTIVE_TRIALS}")
    jobs=list(condition_jobs())
    all_rows=[]
    workers=min(3, os.cpu_count() or 1)
    print(f"Workers: {workers}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs={ex.submit(run_condition,j):j for j in jobs}
        done=0
        for fut in as_completed(futs):
            job=futs[fut]
            rows=fut.result(); all_rows.extend(rows); done+=1
            print(f"[{done:02d}/96] {job}", flush=True)
    df=pd.DataFrame(all_rows)
    df=df.sort_values(["target_rho","mechanism","ranking_contamination_mode","epsilon","estimator","adaptive_trial"])
    raw=OUTDIR/"adaptive_v3_pilot_sensitivity_trials.csv"
    df.to_csv(raw,index=False)

    keys=["target_rho","mechanism","ranking_contamination_mode","epsilon","estimator"]
    oracle=oracle_from_reference(ref)
    df=df.merge(oracle[keys+['oracle_variance','oracle_mse','rss_var','srs_var','rss_mse','srs_mse']],on=keys,how='left',validate='many_to_one')
    summaries=[]; cond=[]
    for b in BUDGETS:
        sel=df[f"selected_{b}"]
        oracle_agree=(sel==df.oracle_variance).astype(float)
        for k,g in df.assign(_agree=oracle_agree).groupby(keys,sort=False):
            s=g[f"selected_{b}"]
            majority="RSS" if (s=="RSS").sum() >= (s=="SRS").sum() else "SRS"
            cond.append(dict(zip(keys,k),pilot_reps=b,
                RSS_selection_fraction=float((s=="RSS").mean()),
                oracle_agreement=float((s==g.oracle_variance).mean()),
                majority_choice=majority,majority_oracle_agreement=int(majority==g.oracle_variance.iloc[0]),
                n_RSS=int((s=="RSS").sum()),n_SRS=int((s=="SRS").sum())))
        summaries.append(dict(pilot_reps=b,
            overall_RSS_selection_fraction=float((sel=="RSS").mean()),
            median_condition_RSS_selection_fraction=float(pd.DataFrame(cond).query("pilot_reps==@b").RSS_selection_fraction.median()),
            trial_level_oracle_agreement=float(oracle_agree.mean()),
            majority_condition_oracle_agreement=float(pd.DataFrame(cond).query("pilot_reps==@b").majority_oracle_agreement.mean()),
            majority_conditions_agree=int(pd.DataFrame(cond).query("pilot_reps==@b").majority_oracle_agreement.sum()),
            majority_conditions_total=96,
            elapsed_seconds=time.time()-t0))
    summary=pd.DataFrame(summaries)
    cond_df=pd.DataFrame(cond)
    summary.to_csv(OUTDIR/"adaptive_v3_pilot_sensitivity_summary.csv",index=False)
    cond_df.to_csv(OUTDIR/"adaptive_v3_pilot_sensitivity_by_condition.csv",index=False)

    # Exact reproducibility check at the existing 30-pilot baseline.
    baseline=PROJECT/"rss_results_v3_adaptive_primary"/"adaptive_v3_trial_results_1000.csv"
    if baseline.exists():
        old=pd.read_csv(baseline)
        old=old[keys+["adaptive_trial","evaluation_seed","selected_method"]].sort_values(keys+["adaptive_trial"]).reset_index(drop=True)
        new=df[keys+["adaptive_trial","evaluation_seed","selected_30"]].rename(columns={"selected_30":"selected_method"}).sort_values(keys+["adaptive_trial"]).reset_index(drop=True)
        ok=old.equals(new)
        print(f"30-pilot exact baseline selection/seed check: {ok}")
        if not ok:
            raise RuntimeError("30-pilot prefix does not exactly reproduce the existing 1000-trial run")
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print(f"\nPASS: Step 1.4 sensitivity outputs written to {OUTDIR}")

if __name__ == '__main__': main()
