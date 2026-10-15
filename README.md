# Ranked Set Sampling under Imperfect Ranking

Reproducibility package accompanying:

**Ranked Set Sampling under Imperfect Ranking: Design-Based Theory, Robust Estimation, and Theory--Simulation Alignment**

Author: Jaafar Almutawa  
Affiliation: Bahrain Polytechnic

## Contents

- `verification/` — scripts for theoretical and numerical verification and simulation experiments.
- `data/` — data used in the real-data illustration.
- `matrices/` — transition matrices used by the computational experiments.
- `results/` — saved outputs from the experiments.
- `supplementary/` — supplementary material accompanying the manuscript.

## Requirements

Python 3.10 or newer and the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Main experiment settings: m = 5, 96 experimental conditions, 200 pilot replications, 20 pilot RSS cycles, 1,000 adaptive trials and 5,000 fixed-strategy reference replications per condition, seed = 20260915.

## Reproducibility

The scripts in `verification/` reproduce the numerical checks, simulation experiments, tables, and real-data analysis reported in the manuscript.

The main verification scripts include:

- `verify_score_bound.py`
- `verify_ordering.py`
- `verify_monotone_range.py`
- `verify_atoms.py`
- `verify_pilot_design.py`
- `verify_cost_adaptive.py`
- `verify_cost_tables.py`
- `verify_cost_adaptive_minimax.py`
- `verify_minimax_tables.py`
- `verify_extra_mechanisms.py`
- `verify_iid_alignment.py`
- `verify_extra_experiments.py`
- `verify_extra_tables.py`
- `verify_real_data_abalone.py`
- `verify_review_checks.py`

The saved results in `results/` provide the numerical outputs used for the reported analyses.

## Data

The `data/` directory contains the data used for the real-data illustration.

## Supplementary material

`Online_Resource_1.pdf` contains supplementary theoretical and numerical material referenced by the manuscript.

## Citation

Please cite the associated manuscript and the archived Zenodo release of this reproducibility package.
