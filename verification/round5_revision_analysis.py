#!/usr/bin/env python3
"""Deterministic post-processing for Round-3/Round-5 manuscript tables.

This script DOES NOT rerun the simulation. It consumes the final V3 data products
and generates the placement-by-estimator factorial breakdown requested by the
referee, plus a compact archive validation report.

Expected files in --adaptive-dir:
  adaptive_v3_reference_final_5000.csv
  adaptive_v3_final_5k_summary.csv
  adaptive_v3_trial_results.csv

The reported relative efficiency is RE_V = Var(SRS) / Var(RSS), consistent
with the manuscript. The default factorial table focuses on the contaminated
high-ranking regime rho_R=0.90, epsilon=0.10, averaging Gaussian/Laplace
mechanisms within each contamination-placement x estimator cell.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEYS=["target_rho","mechanism","ranking_contamination_mode","epsilon","estimator"]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--adaptive-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=None)
    ap.add_argument('--rho',type=float,default=0.90)
    ap.add_argument('--epsilon',type=float,default=0.10)
    args=ap.parse_args()
    d=args.adaptive_dir.expanduser().resolve()
    refp=d/'adaptive_v3_reference_final_5000.csv'
    summp=d/'adaptive_v3_final_5k_summary.csv'
    trialp=d/'adaptive_v3_trial_results.csv'
    for p in (refp,summp,trialp):
        if not p.exists():
            raise SystemExit(f'Missing required final data product: {p}')
    ref=pd.read_csv(refp)
    if len(ref)!=480000:
        raise SystemExit(f'Expected 480000 final reference rows, found {len(ref)}')
    if ref.duplicated(KEYS+['replicate']).any():
        raise SystemExit('Duplicate condition/replicate rows detected.')
    counts=ref.groupby(KEYS).size()
    if len(counts)!=96 or counts.min()!=5000 or counts.max()!=5000:
        raise SystemExit('Final reference grid is not exactly 96 x 5000.')

    rows=[]
    mask=(ref.target_rho==args.rho)&(ref.epsilon==args.epsilon)
    sub=ref.loc[mask]
    for (mode,est),g in sub.groupby(['ranking_contamination_mode','estimator']):
        rss=g.rss_est.to_numpy(float); srs=g.srs_est.to_numpy(float)
        # Equal mechanism weighting: average the two mechanism-specific REs,
        # rather than pooling observations across mechanisms.
        by_mech=[]
        for mech,gm in g.groupby('mechanism'):
            rv=np.var(gm.rss_est,ddof=1); sv=np.var(gm.srs_est,ddof=1)
            by_mech.append((mech,sv/rv))
        re=np.mean([x[1] for x in by_mech])
        rows.append({'rho_R':args.rho,'epsilon':args.epsilon,
                     'placement':mode,'estimator':est,
                     'Gaussian_RE':dict(by_mech).get('gaussian',np.nan),
                     'Laplace_RE':dict(by_mech).get('laplace',np.nan),
                     'mean_RE_across_mechanisms':re})
    out=pd.DataFrame(rows).sort_values(['placement','estimator'])
    outpath=args.out or (d/'placement_by_estimator_factorial_round5.csv')
    out.to_csv(outpath,index=False)
    print(out.to_string(index=False))
    print(f'Wrote {outpath}')

if __name__=='__main__':
    main()
