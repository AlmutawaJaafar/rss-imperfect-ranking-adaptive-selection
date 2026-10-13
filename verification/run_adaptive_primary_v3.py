import os
import pandas as pd
import rss_adaptive_experiments_v3 as v

outdir='rss_results_v3_adaptive_primary'
os.makedirs(outdir, exist_ok=True)
cfg=v.Config(seed=20260915,reps=5000,cycles=100,m=5,
             rho_targets=(0.25,0.50,0.75,0.90),
             calibration_trials=20000,bootstrap_reps=400,
             eps_values=(0.0,0.10),
             adaptive_trials=1000,pilot_reps=200,pilot_cycles=20,outdir=outdir)
cal=v.build_calibration_table(cfg)
cal.to_csv(os.path.join(outdir,'calibration_v3.csv'),index=False)
adaptive=v.run_adaptive(cfg,cal)
adaptive.to_csv(os.path.join(outdir,'adaptive_v3.csv'),index=False)
print(adaptive.to_string(index=False))
