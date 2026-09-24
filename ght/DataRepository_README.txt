DATA REPOSITORY / SUPPLEMENTARY CODE DESCRIPTION

Manuscript:
  Geometric Origins of Heavy Tails in Physical Systems

Purpose:
  Complete numerical-generation code reproducing every figure, every panel and
  every quoted number in the manuscript.

Reproducibility:
  A single MASTER_SEED = 20260915 in geoheavytails/config.py.  Independent
  streams are derived by NumPy SeedSequence spawning.  Output is byte-identical
  across repeated runs.

One-command reproduction:
  python -m pip install -r requirements.txt
  python run_all.py

  run_all.py regenerates all figures and tables and then runs
  scripts/check_manuscript_consistency.py, which asserts every number quoted in
  the manuscript against the number the code produces.  It exits nonzero on any
  disagreement.

Core relations implemented:
  alpha   = 1 + (c + nu)/m                 (Eq. 4)
  q_tail  = 1 + 2m/(m + c + nu)            (Eq. 19)
  Y       = A (r^2 + lambda^2)^(-m/2)      (Eq. 28)
  Y_max   = A lambda^(-m)                  (Eq. 29; MEASURED, then regressed)

Contents:
  geoheavytails/config.py    all seeds, sample sizes, thresholds, constants
  geoheavytails/core.py      RNG, Hill estimator, threshold diagnostics, maps
  geoheavytails/inputs.py    input families, nu-shell sampler, manifolds
  geoheavytails/models.py    competing-model MLE and AIC comparison
  geoheavytails/figures.py   Figures 1-11 with complete panel sets
  scripts/                   launchers, table generation, consistency check
  figures/                   generated PNG files
  data/                      generated CSV tables and figure_summary.json
  EQUATION_MAP.md            manuscript equation -> implementing code
  README.md                  full description of the rebuild

Data provenance:
  No experimental data are used.  All results are analytical calculations or
  synthetic Monte Carlo realizations generated from the equations and parameter
  values encoded in the scripts.
