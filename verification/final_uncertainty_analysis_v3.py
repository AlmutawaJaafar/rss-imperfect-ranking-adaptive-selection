#!/usr/bin/env python3
"""Final uncertainty analysis for the RSS/LTS V3 primary experiment.

Uses EXISTING final data only; no simulations are rerun.
Expected files under rss_results_v3_adaptive_primary/:
  adaptive_v3_final_5k_summary.csv
  adaptive_v3_trial_results.csv
  adaptive_v3_reference_final_5000.csv

Outputs step1_2_final_uncertainty/.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

KEYS=["target_rho","mechanism","ranking_contamination_mode","epsilon","estimator"]

def wilson(k,n,alpha=.05):
    if n<=0:return (np.nan,np.nan)
    z=norm.ppf(1-alpha/2); p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)

def boot_indices(n,B,rng,chunk=250):
    for pos in range(0,B,chunk):
        b=min(chunk,B-pos)
        yield rng.integers(0,n,size=(b,n))

def boot_scalar(x,stat,B,rng,alpha=.05):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n<2:return (np.nan,np.nan)
    vals=[]
    for idx in boot_indices(n,B,rng):
        vals.extend(stat(x[j]) for j in idx)
    return tuple(np.quantile(vals,[alpha/2,1-alpha/2]))

def var(x): return np.var(x,ddof=1)
def mse(x): return np.mean(x*x)

def paired_re(rss,srs,B,rng,alpha=.05):
    rss=np.asarray(rss,float); srs=np.asarray(srs,float)
    n=min(len(rss),len(srs)); rss=rss[:n]; srs=srs[:n]
    vals=[]
    for idx in boot_indices(n,B,rng):
        rv=np.var(rss[idx],axis=1,ddof=1); sv=np.var(srs[idx],axis=1,ddof=1)
        z=sv/rv; vals.extend(z[np.isfinite(z)])
    return tuple(np.quantile(vals,[alpha/2,1-alpha/2]))

def adaptive_risk_boot(est,B,rng,alpha=.05):
    return boot_scalar(est,var,B,rng,alpha), boot_scalar(est,mse,B,rng,alpha)

def joint_excess_boot(est,rss,srs,B,rng,alpha=.05):
    est=np.asarray(est,float); rss=np.asarray(rss,float); srs=np.asarray(srs,float)
    ne,nr=len(est),min(len(rss),len(srs)); rss=rss[:nr]; srs=srs[:nr]
    dv=[]; dm=[]; rv=[]
    # Independent resampling of adaptive evaluation trials and fixed reference reps.
    for pos in range(0,B,100):
        b=min(100,B-pos)
        ie=rng.integers(0,ne,size=(b,ne)); ir=rng.integers(0,nr,size=(b,nr))
        ae=est[ie]; rr=rss[ir]; ss=srs[ir]
        av=np.var(ae,axis=1,ddof=1); am=np.mean(ae*ae,axis=1)
        rvv=np.minimum(np.var(rr,axis=1,ddof=1),np.var(ss,axis=1,ddof=1))
        rm=np.minimum(np.mean(rr*rr,axis=1),np.mean(ss*ss,axis=1))
        dv.extend(av-rvv); dm.extend(am-rm)
        rv.extend(100*(av-rvv)/rvv)
    return (tuple(np.quantile(dv,[alpha/2,1-alpha/2])),
            tuple(np.quantile(dm,[alpha/2,1-alpha/2])),
            tuple(np.quantile(rv,[alpha/2,1-alpha/2])))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--adaptive-dir',type=Path,default=Path.home()/ 'Desktop/LTS_RSS/Tex_gpt/rss_results_v3_adaptive_primary')
    ap.add_argument('--bootstrap',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=20260915)
    ap.add_argument('--alpha',type=float,default=.05)
    a=ap.parse_args(); d=a.adaptive_dir.expanduser().resolve(); out=d/'step1_2_final_uncertainty'; out.mkdir(parents=True,exist_ok=True)
    summary=d/'adaptive_v3_final_5k_summary.csv'; trials=d/'adaptive_v3_trial_results.csv'; refp=d/'adaptive_v3_reference_final_5000.csv'
    for p in (summary,trials,refp):
        if not p.exists(): raise SystemExit(f'MISSING: {p}')
    dec=pd.read_csv(summary); tr=pd.read_csv(trials); ref=pd.read_csv(refp)
    if len(dec)!=96 or len(tr)!=96000 or len(ref)!=480000: raise SystemExit(f'Unexpected sizes: summary={len(dec)}, trials={len(tr)}, reference={len(ref)}')
    if tr.groupby(KEYS).size().nunique()!=1 or tr.groupby(KEYS).size().iloc[0]!=1000: raise SystemExit('Adaptive trials are not exactly 1000 per condition.')
    if ref.groupby(KEYS).size().nunique()!=1 or ref.groupby(KEYS).size().iloc[0]!=5000: raise SystemExit('Reference reps are not exactly 5000 per condition.')
    rng=np.random.default_rng(a.seed); rows=[]
    for _,drow in dec.iterrows():
        mask=np.ones(len(tr),bool); rmask=np.ones(len(ref),bool)
        for k in KEYS: mask &= tr[k].eq(drow[k]); rmask &= ref[k].eq(drow[k])
        tg=tr.loc[mask].copy(); rg=ref.loc[rmask].copy()
        sel=(tg.selected_method.astype(str).values=='RSS'); agr=sel if drow.get('oracle_method_variance','RSS')=='RSS' else ~sel
        ae=tg.adaptive_estimate.astype(float).values; rss=tg.rss_estimate.astype(float).values; srs=tg.srs_estimate.astype(float).values
        rr=rg.rss_est.astype(float).values; ss=rg.srs_est.astype(float).values
        rv,sv=np.var(rr,ddof=1),np.var(ss,ddof=1); rm,sm=np.mean(rr*rr),np.mean(ss*ss)
        re=sv/rv; oracle='RSS' if rv<=sv else 'SRS'; om='RSS' if rm<=sm else 'SRS'
        sl,sh=wilson(int(sel.sum()),len(sel),a.alpha); al,ah=wilson(int(agr.sum()),len(agr),a.alpha)
        avci,mci=adaptive_risk_boot(ae,a.bootstrap,rng,a.alpha)
        rvc=boot_scalar(rr,var,a.bootstrap,rng,a.alpha); svc=boot_scalar(ss,var,a.bootstrap,rng,a.alpha)
        rmc=boot_scalar(rr,mse,a.bootstrap,rng,a.alpha); smc=boot_scalar(ss,mse,a.bootstrap,rng,a.alpha)
        reci=paired_re(rr,ss,a.bootstrap,rng,a.alpha)
        exv,exm,regr=joint_excess_boot(ae,rr,ss,a.bootstrap,rng,a.alpha)
        rows.append({**{k:drow[k] for k in KEYS},'n_trials':len(tg),'n_reference':len(rg),
            'RSS_selection_fraction':sel.mean(),'RSS_selection_CI95_low':sl,'RSS_selection_CI95_high':sh,
            'oracle_agreement_fraction':agr.mean(),'oracle_agreement_CI95_low':al,'oracle_agreement_CI95_high':ah,
            'adaptive_variance':var(ae),'adaptive_variance_CI95_low':avci[0],'adaptive_variance_CI95_high':avci[1],
            'adaptive_MSE':mse(ae),'adaptive_MSE_CI95_low':mci[0],'adaptive_MSE_CI95_high':mci[1],
            'RSS_variance_reference':rv,'RSS_variance_CI95_low':rvc[0],'RSS_variance_CI95_high':rvc[1],
            'SRS_variance_reference':sv,'SRS_variance_CI95_low':svc[0],'SRS_variance_CI95_high':svc[1],
            'RSS_MSE_reference':rm,'RSS_MSE_CI95_low':rmc[0],'RSS_MSE_CI95_high':rmc[1],
            'SRS_MSE_reference':sm,'SRS_MSE_CI95_low':smc[0],'SRS_MSE_CI95_high':smc[1],
            'RE_variance':re,'RE_variance_CI95_low':reci[0],'RE_variance_CI95_high':reci[1],
            'oracle_method_variance':oracle,'oracle_method_MSE':om,
            'adaptive_excess_variance':var(ae)-min(rv,sv),'adaptive_excess_variance_CI95_low':exv[0],'adaptive_excess_variance_CI95_high':exv[1],
            'adaptive_excess_MSE':mse(ae)-min(rm,sm),'adaptive_excess_MSE_CI95_low':exm[0],'adaptive_excess_MSE_CI95_high':exm[1],
            'adaptive_relative_oracle_regret_percent':100*(var(ae)-min(rv,sv))/min(rv,sv),
            'adaptive_relative_oracle_regret_percent_CI95_low':regr[0],'adaptive_relative_oracle_regret_percent_CI95_high':regr[1]})
    R=pd.DataFrame(rows); R.to_csv(out/'final_uncertainty_by_condition.csv',index=False)
    # Aggregate uncertainty: exact 96k trial-level Wilson intervals + condition-level reference bootstrap.
    allsel=(tr.selected_method.astype(str).values=='RSS'); oracle_map=R.set_index(KEYS).oracle_method_variance.to_dict(); agree=np.array([str(row.selected_method)==str(oracle_map[tuple(row[k] for k in KEYS)]) for _,row in tr.iterrows()])
    sl=wilson(int(allsel.sum()),len(allsel),a.alpha); ag=wilson(int(agree.sum()),len(agree),a.alpha)
    # Aggregate fixed-strategy mean/median RE via condition bootstrap draws already implicit in per-condition CIs; recompute 10k joint draws.
    rng2=np.random.default_rng(a.seed+1); cond_groups=[ref.loc[np.logical_and.reduce([ref[k].eq(v) for k,v in zip(KEYS,key)])] for key in dec[KEYS].itertuples(index=False,name=None)]
    mean_re_draw=[]; med_re_draw=[]
    for _ in range(a.bootstrap):
        vals=[]
        for g in cond_groups:
            ii=rng2.integers(0,len(g),len(g)); rr=g.rss_est.to_numpy(float)[ii]; ss=g.srs_est.to_numpy(float)[ii]; vals.append(np.var(ss,ddof=1)/np.var(rr,ddof=1))
        mean_re_draw.append(np.mean(vals)); med_re_draw.append(np.median(vals))
    summary_out=pd.DataFrame([{'conditions':96,'adaptive_trials_total':len(tr),'reference_reps_total':len(ref),
        'overall_RSS_selection_fraction':allsel.mean(),'overall_RSS_selection_CI95_low':sl[0],'overall_RSS_selection_CI95_high':sl[1],
        'overall_oracle_agreement_fraction':agree.mean(),'overall_oracle_agreement_CI95_low':ag[0],'overall_oracle_agreement_CI95_high':ag[1],
        'mean_reference_RE':R.RE_variance.mean(),'mean_reference_RE_CI95_low':np.quantile(mean_re_draw,.025),'mean_reference_RE_CI95_high':np.quantile(mean_re_draw,.975),
        'median_reference_RE':R.RE_variance.median(),'median_reference_RE_CI95_low':np.quantile(med_re_draw,.025),'median_reference_RE_CI95_high':np.quantile(med_re_draw,.975),
        'bootstrap_replicates':a.bootstrap,'bootstrap_seed':a.seed}])
    summary_out.to_csv(out/'final_uncertainty_summary.csv',index=False)
    report=out/'final_uncertainty_report.txt'
    lines=['FINAL V3 UNCERTAINTY ANALYSIS','='*78,'','PASS: 96 conditions; 96 x 1000 adaptive trials; 96 x 5000 reference reps.','',
      f'Overall RSS selection: {allsel.mean():.6f}  95% Wilson CI [{sl[0]:.6f}, {sl[1]:.6f}]',
      f'Overall oracle agreement: {agree.mean():.6f}  95% Wilson CI [{ag[0]:.6f}, {ag[1]:.6f}]',
      f'Mean reference RE: {R.RE_variance.mean():.6f}  bootstrap 95% CI [{np.quantile(mean_re_draw,.025):.6f}, {np.quantile(mean_re_draw,.975):.6f}]',
      f'Median reference RE: {R.RE_variance.median():.6f}  bootstrap 95% CI [{np.quantile(med_re_draw,.025):.6f}, {np.quantile(med_re_draw,.975):.6f}]','',
      'METHODS: Wilson intervals for trial proportions; nonparametric bootstrap over 5,000 reference replicates for fixed risks; paired bootstrap for RE; independent two-sample bootstrap for adaptive excess risk/regret, preserving independence between adaptive evaluation trials and fixed reference replicates.','',
      'OUTPUTS:',str(out/'final_uncertainty_by_condition.csv'),str(out/'final_uncertainty_summary.csv'),str(report)]
    report.write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__': main()
