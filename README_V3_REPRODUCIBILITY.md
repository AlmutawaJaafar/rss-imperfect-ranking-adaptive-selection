# V3 reproducibility and audit package

## Primary manuscript scope
The confirmatory numerical evidence in the manuscript is restricted to the final V3 univariate adaptive experiment. Earlier multivariate and benchmark-data analyses are not part of the confirmatory evidence chain.

## Final primary protocol
- Conditions: 96 = 4 target ranking qualities × 2 mechanisms × 2 contamination-placement modes × 2 contamination levels × 3 estimators.
- Set size: `m=5`.
- Cycles: 100.
- Measured observations per trial: `n=500`.
- Ranking targets: 0.25, 0.50, 0.75, 0.90.
- Ranking mechanisms: Gaussian, Laplace.
- Contamination: epsilon 0 and 0.10; pre-ranking or post-ranking.
- Pilot: 200 independent replications, 20 cycles per pilot replication.
- Adaptive evaluation: 1,000 trials per condition.
- Fixed-strategy reference: 5,000 independent replications per condition.
- Simulation seed: `20260915`.
- Uncertainty bootstrap: 10,000 replicates, bootstrap seed `20260915`.

## Expected final data products
Under `rss_results_v3_adaptive_primary/`:

- `adaptive_v3_trial_results.csv` — 96,000 adaptive trial rows.
- `adaptive_v3_reference_final_5000.csv` — 480,000 fixed-reference rows.
- `adaptive_v3_final_5k_summary.csv` — authoritative 96-condition summary rebuilt from the 5,000-reference file.
- `final_5k_summary_validation.txt` — structural/provenance validation.
- `step1_2_final_uncertainty/final_uncertainty_by_condition.csv` — condition-level uncertainty output.
- `step1_2_final_uncertainty/final_uncertainty_summary.csv` — aggregate uncertainty output.
- `step1_2_final_uncertainty/final_uncertainty_report.txt` — uncertainty-analysis report.

## Source mapping for the final manuscript
| Manuscript item | Authoritative source | Derivation / check |
|---|---|---|
| Table `v3_calibration` | `calibration_v3.csv` | V3 calibration output; 20,000 trials per mechanism-target pair |
| Table `v3_reference_summary` | `adaptive_v3_final_5k_summary.csv` | Rebuilt from `adaptive_v3_reference_final_5000.csv` |
| Table `v3_pilot_sensitivity` | preserved V3 pilot-sensitivity runs/results | 1,000 adaptive trials per condition at each pilot budget |
| Table `v3_adaptive_final_summary` | `final_uncertainty_summary.csv` + `adaptive_v3_final_5k_summary.csv` | Wilson intervals and final 5,000-reference oracle |
| Matched-mechanism table `v3_paired_mechanism` | final paired-mechanism analysis output | Paired Gaussian/Laplace V3 experiment |
| Theorem/proposition equations | `main_stage2_adaptive_v3.tex` | Mathematical audit; no numerical substitution |

## Primary reported values
- Overall RSS selection: `0.839729`, Wilson 95% CI `[0.837395, 0.842036]`.
- Trial-level variance-oracle agreement: `0.835375`, Wilson 95% CI `[0.833016, 0.837707]`.
- Majority condition-level oracle agreement: `90/96 = 93.75%`.
- Mean reference variance RE: `1.389747`, bootstrap 95% CI `[1.382332, 1.398196]`.
- Median reference variance RE: `1.225325`, bootstrap 95% CI `[1.213007, 1.258669]`.
- Reference variance oracle: RSS in 90/96 conditions; SRS in 6/96.

## Required final package contents
Include the exact simulation drivers used for the final run, the two raw final CSV files, the rebuilt summary, uncertainty-analysis scripts and outputs, seed information, and the execution environment/package versions. The environment file should be generated on the machine used for submission (for example with `python3.10 -m pip freeze > requirements-freeze.txt`).

## Audit principle
No manuscript table or scalar should be manually updated from an earlier exploratory run. Every confirmatory numerical value must be traceable to one of the final V3 source files above or to a deterministic analysis of those files.
