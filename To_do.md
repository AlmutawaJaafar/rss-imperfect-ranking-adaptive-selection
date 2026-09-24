# To-do: RSS Robust Estimation Manuscript

## 0. Current status after the V3 adaptive run

The primary adaptive V3 experiment is now complete and the main LaTeX draft has been updated to use it as the primary numerical evidence.

Current validated design:

- `m = 5`, `cycles = 100`, `n_measured = 500`.
- Target ranking quality: `rho = {0.25, 0.50, 0.75, 0.90}`.
- Ranking mechanisms: Gaussian and Laplace rank perturbations.
- Contamination: `epsilon = 0, 0.10`.
- Contamination placement: pre-ranking and post-ranking.
- Estimators: mean, 10% trimmed mean, Huber M with `c = 1.345`.
- 96 adaptive conditions.
- Ranking calibration: 20,000 trials per mechanism/target pair.
- Adaptive pilot: 30 pilot replications, 20 cycles per pilot replication.
- Adaptive evaluation: 200 independent pilot/decision/evaluation trials per condition.
- Independent fixed-strategy reference: 2,000 RSS/SRS replications per condition.
- Seed: `20260915`.

Validated aggregate results used in the draft:

- Fixed-strategy oracle: RSS in 88/96 conditions; SRS in 8/96.
- Overall RSS selection fraction: 0.7312; median 0.7025.
- Trial-level agreement with the fixed-strategy variance oracle: 0.7264.
- Majority-choice condition-level agreement with the oracle: 90/96 = 93.75%.
- Mean fixed-strategy reference RE across all 96 conditions: 1.3873; median 1.2611; range 0.9289--2.2006.
- Mean RSS selection fraction by target rho: 0.533, 0.656, 0.823, 0.913 for rho = 0.25, 0.50, 0.75, 0.90.
- Mean reference RE by target rho: 1.022, 1.188, 1.487, 1.852.

A separate paired Gaussian/Laplace V3 mechanism-comparison experiment is also available (`paired_delta_summary_v3.csv`, 16 rows, 400 bootstrap replicates). It has already been used in the revised draft as supporting evidence that similar scalar rank correlation can still produce different empirical efficiency.

---

# 1. Immediate numerical work required before submission

## 1.1 Add Monte Carlo uncertainty to the adaptive results

**Priority: CRITICAL**

The present adaptive file reports point estimates based on 200 adaptive trials per condition. This is enough for a development result, but the manuscript should not rely on point estimates alone.

Create a new Python analysis script, preferably:

`analyze_adaptive_v3_final.py`

It should compute, for every condition:

- SE/CI for RSS selection fraction using a binomial or Wilson interval.
- CI for the adaptive selected-design variance.
- CI for adaptive MSE.
- CI for oracle-agreement fraction.
- CI for relative regret.
- A clearly defined uncertainty measure for fixed-strategy RE and MSE ratios.

Do **not** compare adaptive and oracle point estimates row-by-row without reporting their different Monte Carlo uncertainty.

## 1.2 Increase the fixed-strategy reference replication count

**Priority: CRITICAL**

The current reference experiment uses 2,000 replications. For the final paper, rerun the fixed RSS/SRS reference risks with at least:

- 5,000 replications for the final main table, preferably 10,000 if runtime is acceptable.
- Same deterministic seed construction and same 96-condition grid.
- Preserve the existing 2,000-replicate results for auditability.

Save as, for example:

`adaptive_v3_reference_final_5000.csv`

and, if run,

`adaptive_v3_reference_final_10000.csv`.

Update the paper only after checking that the qualitative conclusions are stable.

## 1.3 Increase adaptive trial count

**Priority: HIGH**

The current 200 adaptive trials are useful but noisy for a decision probability. Rerun with:

- 500 adaptive trials as the minimum final check.
- 1,000 adaptive trials if computationally feasible.

Keep pilot/evaluation independence exactly as in `adaptive_v3.py`.

The final paper should report whether the 200-trial conclusions are stable at 500/1,000 trials.

## 1.4 Pilot-budget sensitivity analysis

**Priority: CRITICAL for the adaptive contribution**

The current pilot uses only 30 replications. This is the main finite-pilot tuning parameter and must be stress-tested.

Write:

`adaptive_v3_pilot_sensitivity.py`

Run pilot sizes such as:

- 10
- 20
- 30
- 50
- 100
- 200

Keep pilot cycles fixed initially at 20, then optionally vary pilot cycles separately.

For each pilot budget report:

- RSS selection fraction.
- Oracle agreement.
- Majority-choice condition accuracy.
- Regret.
- Monte Carlo uncertainty.

This experiment should answer whether the adaptive rule is genuinely useful or simply highly sensitive to the 30-replication pilot.

## 1.5 Repeated-seed stability

**Priority: HIGH**

Run the complete adaptive experiment under at least 3--5 independent master seeds.

Do not change the design between seeds.

Report:

- aggregate RSS selection fraction,
- oracle agreement,
- reference RE stability,
- condition-level majority decision stability.

If conclusions move materially across seeds, state this explicitly and enlarge the final Monte Carlo experiment.

---

# 2. Complete the negative controls

## 2.1 Random-ranking control

**Priority: CRITICAL**

The theory explicitly identifies random ranking as the SRS limit, but the current primary adaptive V3 paper section does not yet contain the final random-ranking experiment.

Add a run with:

- exact same `m`, cycles, n, estimators, contamination settings,
- `mechanism = random`,
- same reference and adaptive budgets.

Verify:

- empirical RSS/SRS risk ratio is approximately 1,
- no systematic RSS advantage appears,
- the adaptive rule does not systematically select RSS when risks are indistinguishable.

This is an essential negative control.

## 2.2 Perfect-ranking control

**Priority: HIGH**

Add the exact perfect-ranking case:

- `mechanism = perfect`.

Use it to verify the known balanced-RSS limiting case and the implementation of the ideal ranking benchmark.

Do not infer a nonlinear universal rho-to-RE formula from this control alone.

## 2.3 Implementation-limit checks

**Priority: HIGH**

Numerically verify the transition-mechanism limits:

- `P = I` / perfect ranking.
- `P_{rs} = 1/m` / random ranking.
- intermediate Gaussian/Laplace mechanisms.

Record the empirical transition matrix and the realized rank correlation.

---

# 3. Strengthen the matched-mechanism evidence

## 3.1 Extend the Gaussian-vs-Laplace matched-rho experiment

**Priority: HIGH**

Current paired results cover clean data and four estimators. Extend the paired mechanism comparison to:

- epsilon = 0 and 0.10,
- pre-ranking and post-ranking contamination,
- mean, trimmed, Huber, and Tukey,
- preferably larger replication counts than the current 2,000 if feasible.

Use paired random seeds so that the Gaussian/Laplace comparison remains low-noise.

## 3.2 Store and report the transition matrices

**Priority: HIGH**

For every matched-rho condition, save:

- empirical transition matrix `P_hat`,
- row sums,
- column sums,
- realized rho,
- a matrix-distance measure between Gaussian and Laplace mechanisms.

This is important because the paper's main theoretical point is that `rho` is not sufficient.

## 3.3 Construct an explicit counterexample for the theory

**Priority: CRITICAL THEORETICAL SUPPORT**

Construct two admissible transition matrices `P1` and `P2` with:

- identical or nearly identical scalar rank correlation,
- different selected-stratum distributions,
- different exact RSS variance.

For `m=3` or `m=5`, search numerically first and then formulate the construction analytically.

This would turn the statement “scalar rho is not sufficient” into a concrete proposition/example rather than relying only on simulations.

---

# 4. Theory that still needs to be developed

## 4.1 General unbiasedness condition beyond doubly stochastic P

**Priority: CRITICAL**

The current proposition uses doubly stochastic `P` as a sufficient condition. It is intentionally not claimed to be necessary.

Develop the exact general condition.

With

`q_{r|s} = P_{rs} / sum_u P_{us}`

for the uniform prior on true ranks, derive the condition under which

`(1/m) sum_s g_s(x) = f(x)`.

This should produce a proposition giving necessary/sufficient conditions for distributional unbiasedness, and a weaker condition for mean unbiasedness.

Then prove the proposition.

## 4.2 General exact variance formula

**Priority: HIGH**

Extend the current exact variance formula from the doubly stochastic case to a general admissible transition matrix.

Separate clearly:

- exact variance of the balanced RSS mean,
- mean-unbiasedness condition,
- distributional equality with SRS,
- perfect-ranking limit.

Do not conflate these four properties.

## 4.3 Formal “rho is not sufficient” proposition

**Priority: CRITICAL**

Add a proposition of the form:

> There exist transition mechanisms with the same scalar rank correlation but different RSS efficiencies.

Provide either:

- an explicit analytic construction, or
- a finite-dimensional constructive proof for a specified `m`.

Then connect the proposition to the empirical matched-rho experiment.

## 4.4 Stronger estimator-specific robustness theory

**Priority: CRITICAL if breakdown preservation remains a contribution**

The current conditional robustness proposition is deliberately generic. It is logically safe but close to the definition of robustness transfer.

For the final paper, develop at least one nontrivial estimator-specific theorem, preferably for:

- trimmed mean, or
- Huber M-estimator,

and only claim MCD/S-estimator results if they can be proved correctly.

The theorem should state exactly:

- contamination model,
- ranking contamination model,
- selection mechanism,
- admissible selected sample class,
- finite-sample breakdown/bias bound,
- what changes under pre-ranking contamination.

Do not claim unconditional preservation of breakdown.

## 4.5 Influence-function theory under RSS strata

**Priority: HIGH**

Formalize the asymptotic linear representation for a stratified/RSS sample with independent but non-identically distributed selected observations.

Derive:

- asymptotic variance,
- conditions for a CLT,
- estimator-specific influence-function expression,
- comparison with SRS.

This would provide a rigorous bridge between the design-based mean theory and nonlinear robust estimators.

## 4.6 Adaptive selection theorem

**Priority: CRITICAL for the adaptive contribution**

The current adaptive rule is empirical. Develop a finite-pilot decision result.

Desired structure:

- Let `V_RSS < V_SRS` with gap `Delta > 0`.
- Let pilot variance estimators be consistent under finite fourth moments or a stronger moment/tail condition.
- Derive a bound on
  `P(select SRS | RSS is truly better)`.
- Show that the error probability decreases with pilot size.
- State the corresponding result when SRS is truly better.

A Bernstein/Hoeffding/sub-Gaussian type argument may be possible depending on the pilot variance estimator used. The exact assumptions need to be derived, not guessed.

## 4.7 Regret bound for adaptive RSS/SRS

**Priority: HIGH**

Once the selection-probability result is established, derive an expected regret bound of the form

`E[adaptive risk - oracle risk] <= risk-gap * P(wrong selection)`

plus any estimator of the selected risk that introduces additional finite-sample error.

This would give the adaptive section a genuine theoretical result.

## 4.8 Contamination-specific MSE theory

**Priority: HIGH**

The V3 simulation shows that the adaptive rule is variance-based while contamination is present. Develop the theory needed to distinguish:

- variance-optimal design,
- MSE-optimal design,
- pilot estimation of variance,
- pilot estimation of MSE.

Do not describe the current variance-based pilot as MSE-optimal.

---

# 5. Adaptive-rule extensions that should be implemented

## 5.1 MSE-aware adaptive rule

**Priority: HIGH**

Write a second implementation that selects by pilot MSE rather than variance.

Important: because the true parameter is not available in real data, the simulation version should explicitly distinguish:

- simulation-truth MSE oracle,
- implementable pilot criterion,
- any bias-corrected or replicated pilot estimator of MSE.

Do not use known simulation truth in a proposed practical decision rule.

## 5.2 Cost-aware adaptive rule

**Priority: HIGH**

Implement

`RSS if V_RSS + lambda*C_RSS < V_SRS + lambda*C_SRS`.

Run sensitivity over several `lambda` values.

Measure separately:

- ranking cost,
- sorting cost,
- robust-estimation cost,
- total wall-clock cost.

## 5.3 Adaptive set-size selection

**Priority: MEDIUM**

A later extension can choose `m` adaptively, but this should only be done after the fixed-`m=5` rule is validated.

Possible grid:

`m in {3,5,7,9}`

at fixed measured sample size.

This is not necessary for the first submission unless set-size adaptation is claimed as a contribution.

---

# 6. Multivariate simulation must be redone under the new framework

**Priority: CRITICAL before keeping the old multivariate tables as confirmatory evidence**

The existing multivariate section in the draft comes from an earlier protocol. It should not be treated as equivalent to the V3 adaptive study.

## 6.1 Rewrite the multivariate simulation code

Create a new file, for example:

`multivariate_v3_primary.py`

Use a single coherent design with:

- dimensions `p = 4, 10, 50`,
- fixed measured sample size,
- set sizes explicitly controlled,
- robust Mahalanobis ranking or another precisely defined ranking score,
- Gaussian/Laplace ranking mechanisms where appropriate,
- contamination before/after ranking,
- mean, trimmed, Huber, and MCD only where mathematically and numerically well-defined.

## 6.2 Calibrate multivariate ranking quality correctly

Do not insert a univariate rho into a multivariate theorem.

Report:

- realized ranking metric,
- transition matrix or an appropriate multivariate analogue,
- dimension,
- score definition,
- contamination state.

## 6.3 Remove or justify current old multivariate “theoretical RE” columns

The current exploratory tables contain theoretical RE values that are not supported by the revised transition-matrix theory.

Either:

- derive the corresponding multivariate theory, or
- remove those theoretical columns and report empirical results only.

Do not preserve unsupported theoretical benchmarks merely for presentation continuity.

---

# 7. Decide what to do with the benchmark-data section

**Priority: HIGH**

The current IRIS/Wine/Air-Quality section is based on an older protocol and contains strong narrative claims.

There are two acceptable paths:

### Path A: keep the applications

Rerun all applications under the final RSS framework and document:

- exact ranking variable,
- contamination generation,
- RSS design,
- SRS control,
- replication count,
- estimator definitions,
- uncertainty intervals,
- whether ranking uses any information that would be unavailable before measurement.

### Path B: remove the applications

Remove the benchmark-data section entirely and keep the paper focused on the design theory and controlled simulation study.

Do not retain old application tables merely to increase the number of experiments.

---

# 8. Monte Carlo reporting and statistical inference

**Priority: CRITICAL**

For every main table, add enough uncertainty reporting to make the numerical claims auditable.

At minimum report:

- replication count,
- point estimate,
- Monte Carlo SE or CI,
- exact definition of the ratio/metric,
- seed or reproducibility information.

For RE ratios, be careful with naive ratio-of-independent-estimates intervals. Prefer an appropriate paired bootstrap or a log-ratio bootstrap when justified.

For adaptive selection fractions, Wilson/binomial intervals are natural.

For paired mechanism differences, retain the current paired bootstrap design and report the number of bootstrap replicates.

---

# 9. Figures to add after the final simulation runs

**Priority: HIGH**

Recommended final figures:

1. **Reference RE versus ranking quality**
   - separate variance and MSE panels only if both are needed;
   - show mechanism/contamination variation without implying a universal curve.

2. **Adaptive RSS-selection probability versus ranking quality**
   - show uncertainty bars;
   - clearly distinguish pilot-decision probability from oracle-optimality.

3. **Oracle regret versus pilot size**
   - directly illustrates the adaptive theorem/sensitivity study.

4. **Matched-rho mechanism comparison**
   - Gaussian vs Laplace efficiency at matched realized rho.

5. **Transition-matrix heatmaps**
   - at least one low-rho and one high-rho example.

6. **Multivariate dimension sensitivity**
   - only after the multivariate V3 study is rerun.

---

# 10. Manuscript corrections still required

## 10.1 Remove obsolete claims from old drafts

Search the whole project for language such as:

- “RSS consistently provides …”
- “RSS preserves the breakdown point”
- “efficiency increases with rho” as a universal theorem,
- “RSS overhead is negligible” without timing,
- “theoretical RE” values not derived from the current theory,
- “RSS is computationally cheaper” based only on fewer measured observations.

These claims should either be proved, experimentally verified under the final protocol, or removed.

## 10.2 Keep variance and MSE claims separate

The adaptive rule currently minimizes pilot variance.

The paper must state clearly:

- variance is the implemented decision criterion,
- MSE is an ex-post evaluation metric,
- an MSE-aware adaptive rule is future work unless it is implemented and analyzed.

## 10.3 Make the contamination mechanism explicit

The paper must define precisely what “pre-ranking” and “post-ranking” mean.

State whether contamination changes:

- ranking inputs,
- measured responses,
- both,
- the selected candidate indices.

## 10.4 Replace “oracle beats adaptive” language

Negative adaptive excess values in some 200-replicate rows are sampling noise relative to the independent 2,000-replicate oracle estimates.

Do not interpret them as systematic oracle violations.

## 10.5 Clarify what oracle means

Use “fixed-strategy reference oracle” or similar terminology.

The oracle is not available to the actual adaptive procedure; it is an ex-post simulation benchmark.

---

# 11. Reproducibility package

**Priority: CRITICAL before submission**

Create a clean reproducibility directory containing:

- `rss_adaptive_experiments_v3.py`
- `adaptive_v3.py`
- the final simulation drivers,
- calibration CSV,
- reference-risk CSV,
- adaptive-result CSV,
- paired-mechanism CSV,
- final analysis scripts,
- exact random seeds,
- environment/package versions,
- CPU/RAM/OS information,
- a README explaining how to reproduce each table and figure.

Run every final table directly from the saved CSV files rather than manually copying numbers into LaTeX.

Ideally create a single script such as:

`reproduce_all_results.py`

that regenerates all final summaries/figures from raw simulation outputs.

---

# 12. Bibliography and reference verification

**Priority: HIGH**

Verify every cited reference in the final TeX source.

In particular:

- confirm the classical RSS references,
- confirm robust-estimator references,
- remove unused references,
- verify publication year, journal, volume, pages, and DOI where available,
- ensure no placeholder/fabricated reference remains.

The isolated LaTeX compilation performed during this update used an empty temporary bibliography only to test TeX syntax. The actual manuscript still needs its real `bibliography.bib` during the final compilation.

---

# 13. Final mathematical review

**Priority: CRITICAL**

Before submission, perform one full proof audit from the beginning without relying on previous reviews.

Check every theorem/proposition for:

- assumptions stated before use,
- dimensions and indexing,
- probability conditioning,
- independence assumptions,
- finite versus asymptotic statements,
- contamination model,
- whether a result is design-based or estimator-specific,
- whether a claim concerns variance, MSE, bias, or breakdown.

In particular, audit:

- `Proposition: Balanced RSS under a doubly stochastic ranking mechanism`.
- `Proposition: Conditional robustness transfer`.
- all influence-function claims,
- every efficiency formula,
- every limiting-case statement.

Then add estimator-specific propositions/theorems only where the assumptions and proofs are genuinely available.

---

# 14. Final coding audit

**Priority: CRITICAL**

Even though some earlier manuscript-review tasks explicitly ignored code verification, the final submission package should be checked against the paper.

Cross-check every final equation and table against the actual implementation:

- ranking score,
- rank perturbation equation,
- contamination construction,
- RSS selection index,
- SRS sampling,
- estimator tuning constants,
- pilot variance estimator,
- adaptive decision rule,
- reference oracle calculation,
- MSE definition,
- bootstrap procedure.

Run automated consistency tests where possible.

---

# 15. Final paper structure decision

The current `main_stage2_adaptive_v3.tex` intentionally keeps the old multivariate and benchmark-data material but labels it exploratory in the primary simulation discussion.

Before submission, choose one of these structures:

### Option A: theory + univariate adaptive V3 as the main paper

Keep the paper focused, remove unsupported/obsolete multivariate and real-data sections, and use the V3 study plus matched-rho mechanism experiment as the main empirical contribution.

### Option B: theory + adaptive V3 + fully rebuilt multivariate study

Best only if the multivariate experiment is rerun under the same transition-mechanism framework and uncertainty reporting.

Do not use the current old multivariate tables as final confirmatory evidence without this rebuild.

---

# 16. Exact files currently available

Primary draft:

- `main_stage1_revised.tex` — previous revised draft.
- `main_stage2_adaptive_v3.tex` — updated draft produced in this pass.

Primary code:

- `rss_adaptive_experiments_v3.py` — core V3 implementation.
- `adaptive_v3.py` — standalone corrected adaptive runner.

Existing results:

- `adaptive_v3_decisions.csv` — complete 96-condition adaptive result, including reference risks and oracle diagnostics.
- `adaptive_v3_reference_replicates.csv` — independent reference replicate-level data.
- `paired_delta_summary_v3.csv` — paired Gaussian/Laplace mechanism comparison.
- `matched_rho_paired_replicates_v3.csv` — paired replicate-level mechanism data.

---

# 17. Recommended order of work

1. Run reference reps = 5,000 or 10,000.
2. Run adaptive trials = 500 or 1,000.
3. Run pilot-size sensitivity.
4. Run 3--5 master seeds.
5. Add perfect-ranking and random-ranking controls.
6. Extend matched-rho comparison to contamination modes.
7. Add final Monte Carlo CIs.
8. Complete the general transition-matrix theorem and the explicit same-rho counterexample.
9. Develop and prove the adaptive selection-error/regret theorem.
10. Prove at least one genuinely estimator-specific robustness result.
11. Rebuild multivariate simulations or remove the old exploratory section.
12. Decide whether to retain the benchmark-data section; rerun if retained.
13. Add final figures.
14. Perform full equation/code/table cross-check.
15. Verify bibliography.
16. Compile the final manuscript with the real bibliography and all final CSV-derived tables.
17. Perform one final strict reviewer audit and only then call the paper submission-ready.

---

# 18. Submission-readiness criterion

The manuscript should not be called “submission-ready” until all of the following are true:

- the primary adaptive result is stable under larger Monte Carlo budgets and multiple seeds;
- random and perfect ranking controls pass;
- uncertainty intervals are reported;
- the adaptive criterion is stated exactly as implemented;
- at least one nontrivial adaptive-theory result is proved;
- any estimator-specific breakdown claim is actually proved under explicit assumptions;
- the multivariate and real-data sections either use the final protocol or are removed;
- all main tables/figures are generated from reproducible scripts;
- the real bibliography is present and verified;
- the final TeX compiles without unresolved internal references or citation problems.
