# Ranked Set Sampling under Imperfect Ranking:
# Design-Based Analysis and Adaptive Selection

This repository contains the computational code and reproducibility materials for:

> J. Almutawa, "Ranked Set Sampling under Imperfect Ranking: Design-Based Analysis and Adaptive Selection"

## Computational Environment

- Python 3.10
- NumPy
- Pandas
- SciPy
- Matplotlib

## Main Experiment Parameters

The primary simulation uses:
- m = 5
- 96 experimental conditions
- 200 pilot replications
- 20 pilot RSS cycles
- 1,000 adaptive trials per condition
- 5,000 fixed-strategy reference replications per condition
- seed = 20260915

## Running the Code

### Main Experiment
python3.10 code/rss_adaptive_experiments_v3_complete.py

### Parallel Implementation
python3.10 code/rss_adaptive_experiments_v3_parallel.py

### Pilot-Budget Sensitivity
python3.10 code/step14_pilot_budget_sensitivity_v3.py

### Uncertainty Analysis
python3.10 code/final_uncertainty_analysis.py

## Reproducibility

The scripts use deterministic random-number seeds.
The adaptive evaluation data and fixed-strategy reference data
are generated independently.

## Data

The complete simulation outputs are archived in Zenodo.

## Citation

Please cite the associated paper and Zenodo record.
=======
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

The computational experiments use Python 3 and the packages listed in `requirements.txt`.

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
