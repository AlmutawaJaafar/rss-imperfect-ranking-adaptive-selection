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
