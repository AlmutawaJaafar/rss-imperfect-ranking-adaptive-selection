# Ranked Set Sampling under Imperfect Ranking

Reproducibility package for

**Ranked Set Sampling under Imperfect Ranking: An Efficiency Ordering of Ranking Mechanisms and a Cost-Aware Adaptive Design**
Jaafar Almutawa, Department of Mathematics, Bahrain Polytechnic (submitted to *Statistical Papers*).

## Requirements

Python 3.10 or newer: `pip install -r requirements.txt`

## Layout

| Folder | Contents |
|---|---|
| `verification/` | All scripts (see the two tables below) |
| `results/` | Saved outputs of the cost-aware experiments and the real-data illustration |
| `matrices/` | Transition matrices used by the scripts (`P_mech.npz`, `P_minRE.npy`) |
| `data/` | Abalone data (UCI Machine Learning Repository, https://doi.org/10.24432/C55C7W) |
| `rss_results_v3_adaptive_primary/`, `rss_results_v3_paired_primary/` | Outputs of the fixed-count reference experiment |

To run a script, copy the files it needs from `results/`, `matrices/` and `data/` into `verification/` and run it there, for example `python verify_cost_tables.py`.

## Main paper: theory checks, cost-aware adaptive design, real data

40,000 replications per condition, budget C = 500, set sizes 3, 5 and 7.

| Script | Produces or checks |
|---|---|
| `verify_score_bound.py` | Score-correlation bound (Theorem 2.1) and score columns of the mechanism table |
| `verify_ordering.py` | Ordering theorem and spectrum of P P^T (writes `P_mech.npz`) |
| `verify_monotone_range.py` | Attainable efficiency range at a fixed rank correlation |
| `verify_atoms.py` | Necessity condition for populations with atoms |
| `verify_pilot_design.py` | Approximately minimax pilot sizes |
| `verify_cost_adaptive.py` | Adaptive experiment with a 20-cycle pilot (writes `cost_adaptive_results.json`) |
| `verify_cost_tables.py` | Tables from `cost_adaptive_results.json`, including the pilot-size and budget sensitivity |
| `verify_cost_adaptive_minimax.py` | Adaptive experiment with minimax pilots (writes `cost_adaptive_minimax.json`) |
| `verify_minimax_tables.py` | Headline regret table from `cost_adaptive_minimax.json` |
| `verify_extra_mechanisms.py` | Permutation and minimum-efficiency mechanisms (needs `P_minRE.npy`, `rev_exp.json`) |
| `verify_iid_alignment.py` | Predicted versus simulated efficiency under i.i.d. contamination |
| `verify_extra_experiments.py <part>` | Set sizes 3 and 7, lognormal, asymmetric contamination, misspecified cost (`part` = `setsize3`, `setsize7`, `lognormal`, `asym`, `misspec`) |
| `verify_extra_tables.py` | Tables from the `extra_<part>.json` files |
| `verify_real_data_abalone.py` | Abalone illustration (writes `abalone_results.json`) |
| `verify_review_checks.py` | Further numerical values quoted in the text |

## Online Resource 1, Section S4: fixed-count reference experiment

96 labeled conditions, m = 5, n = 500, 5,000 fixed-strategy replications per condition, seed 20260915; details in `README_V3_REPRODUCIBILITY.md`.

| Script | Role |
|---|---|
| `rss_adaptive_experiments_v3.py` | Simulation module imported by the scripts below |
| `run_adaptive_primary_v3.py`, `run_paired_primary_v3.py` | Simulation drivers |
| `validate_reference_final_5000.py`, `rebuild_final_5k_adaptive_summary.py` | Reference data validation and summary |
| `final_uncertainty_analysis_v3.py`, `analyze_adaptive_v3_final.py` | Bootstrap intervals and summaries |
| `step14_pilot_budget_sensitivity_v3.py` | Cost-free selection rule discussed in Section S4 |

## Licence and citation

MIT licence. Please cite the article and this archive (see `CITATION.cff`).
