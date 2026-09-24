import os
import pandas as pd
import rss_adaptive_experiments_v3 as v

outdir='rss_results_v3_paired_primary'
os.makedirs(outdir, exist_ok=True)
cfg=v.Config(seed=20260915,reps=2000,cycles=100,m=5,
             rho_targets=(0.25,0.50,0.75,0.90),
             calibration_trials=20000,bootstrap_reps=400,outdir=outdir)
cal=v.build_calibration_table(cfg)
cal.to_csv(os.path.join(outdir,'calibration_v3.csv'),index=False)
reps,delta=v.run_paired_mechanism_comparison(cfg,cal)
reps.to_csv(os.path.join(outdir,'matched_rho_paired_replicates_v3.csv'),index=False)
delta.to_csv(os.path.join(outdir,'paired_delta_summary_v3.csv'),index=False)
print(delta.to_string(index=False))
