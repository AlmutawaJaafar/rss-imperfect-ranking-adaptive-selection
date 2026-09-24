# Geometric Origins of Heavy Tails — numerical archive (rebuilt)

Reproducible code for **"Geometric Origins of Heavy Tails in Physical Systems"**.

This archive is a ground-up rewrite. It exists because the previous archive did
not reproduce the manuscript: several figure panels had no implementation at
all, one figure used a numerical value that contradicted its own caption, and
one "validation" evaluated the formula it claimed to be testing. Every such
defect is addressed below and is now guarded by an automated check.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_all.py          # figures + tables + consistency check
```

`run_all.py` exits nonzero if any manuscript claim disagrees with the numbers
the code produces. **Treat a nonzero exit as a block on submission.**

Single figure, single table, or check alone:

```bash
python scripts/figure_06.py
python scripts/generate_validation_tables.py
python scripts/check_manuscript_consistency.py
```

Full run takes roughly 30 s on a laptop.

## Layout

```
geoheavytails/
  config.py     every seed, sample size, threshold and constant (single source)
  core.py       RNG, Hill estimator + threshold diagnostics, regularized map
  inputs.py     input families, nu-shell sampler, embedded manifolds
  models.py     competing-model MLE and AIC comparison
  figures.py    Figures 1-11, full panel sets
scripts/        figure launchers, table generation, consistency check
figures/        generated PNGs
data/           generated CSVs and figure_summary.json
EQUATION_MAP.md every manuscript equation -> the code that implements it
```

## What changed, and why

### 1. Figure 6 now has all four panels
Panels (b), (c) and (d) previously had no code anywhere, which made the claims
of Secs. VI.C, VI.D and VI.E unverifiable. Now implemented:

- **(b)** six input families — Gaussian, Laplace, logistic, uniform, skewed
  split-normal, Gaussian mixture — each analytically rescaled so that
  `p_X(0) = 1/sqrt(2 pi)` exactly (Eq. 48), with **30 independent realizations
  each**. Per-realization estimates are exported raw to `robustness_raw.csv`,
  not just summarized.
- **(c)** the local probability exponent, using `R^2 ~ Gamma((c+nu)/2, 2)`
  (Eqs. 51-52) over all 18 `(c, m, nu)` classes of Eq. (55).
- **(d)** embedded sphere / circle / point in three-dimensional Gaussian input
  space (Eqs. 57-59), with error bars from 8 repeats.

### 2. The relaxation-time exponent is derived, not typed
The Fig. 11 caption quotes `m = 1.260` for the critical relaxation time; the
previous code hard-coded `1.89`. By Eq. (98) this quantity is not free: it is
`m = z * nu_c`. `config.py` therefore defines `NU_C = 0.630`, `Z_C = 2.000` and
computes `M_TAU_C = Z_C * NU_C = 1.260`. The consistency checker asserts it.
Note the physical consequence: `chi` (1.237) and `tau_c` (1.260) nearly
coincide, so their curves in Fig. 11 almost overlap. The old value made them
look comfortably separated — which is exactly why the error survived.

### 3. Cutoff scaling is now measured, not asserted
The previous `cutoff_scaling()` computed `lambda**(-m)` from the formula and
wrote it to a CSV labelled as a validation. Here, for each `(m, lambda)` a
sample is simulated, the cutoff is extracted as a high empirical quantile, and
`log Y_max` is regressed on `log lambda`. Recovered slopes:

| m | fitted slope | expected |
|---|---|---|
| 0.5 | −0.4999 | −0.5 |
| 1.0 | −0.9999 | −1.0 |
| 1.5 | −1.4998 | −1.5 |
| 2.0 | −1.9997 | −2.0 |

### 4. One regularization, applied everywhere
The old code used a different ad hoc floor in each figure (`2e-4`, `0.002`,
`0.001`, `dw`), each producing an artificial atom of probability and the sharp
vertical CCDF drops visible in the old Figs. 7-9. All maps now use the
manuscript's own Eq. (28), `Y = A(r^2 + lambda^2)^{-m/2}`, and the resolution
scale is chosen by `core.resolution_for_window`, which guarantees the Hill
window sits inside the algebraic regime rather than on the cutoff. Each
application figure reports its window ratio; the checker requires it to stay
small. Fig. 5(a) now overlays the **exact** analytic crossover density on the
simulated histogram, including the endpoint accumulation of Eq. (33) — the
upturn at the cutoff for `c + nu = 1` is predicted behaviour, not an artefact.

### 5. Threshold stability, seed sensitivity, and competing models
- `threshold_stability.csv` records the entire Hill plot for every `(c, m)`,
  so the finite-threshold bias of Eq. (14) is measured rather than inherited
  from a single hard-wired `k = 4000`. `core.select_k_stability` additionally
  provides a plateau-selected threshold.
- `seed_sensitivity.csv` gives the spread of each exponent over 12 independent
  streams. The previous archive quantified no uncertainty anywhere.
- `model_comparison.csv` fits Pareto, truncated Pareto, lognormal, stretched
  exponential and q-Gaussian by maximum likelihood and ranks them by AIC —
  the procedure the manuscript demands of experimentalists (Sec. VI.E item 5)
  but never applied to its own data. Results: the bare singular map prefers
  Pareto, and every regularized map prefers **truncated** Pareto by a large
  margin, as the theory requires. Note honestly that on the bare map all five
  models sit within `Delta AIC < 3` at 3000 tail points — a power law is not
  strongly identified at that sample size, which is worth stating in the text.

### 6. Other panel repairs
- **Fig. 1** now has three panels and its "regular map" really is `Y = X^2`, as
  captioned (the old code used `exp(0.55 x)`).
- **Fig. 4(b)** uses a log ordinate so all three moment regimes are legible;
  previously the infinite-mean curve was a single vertical jump.
- **Fig. 8(c)** and **Fig. 9(c)** exist for the first time: a measured
  `Q_max` vs resolution law, and a measured collapse of nominal 95% interval
  coverage as `u_delta/delta` grows (Eq. 83).
- **Fig. 9(a)** shows the force–deflection synthetic data with the true
  response, per the caption, instead of a deterministic `E(delta)` curve.
- **Fig. 10** shows the four systems named in the caption (the old code added
  an undocumented fifth) and includes the missing collapse panel.
- **Fig. 7** performs 3000 genuine nonlinear pseudo-Voigt refits (Eq. 67)
  rather than asserting a linewidth distribution by hand.

### 7. Reproducibility mechanics
- A single `MASTER_SEED`, with independent streams derived by `SeedSequence`
  spawning rather than by adding an integer to a seed (added offsets do not
  guarantee independent generator states).
- All sample sizes are named constants in `config.py`, including the ones the
  manuscript never stated. Any value appearing in the text must be traceable to
  a constant there.
- Output verified byte-identical across repeated runs.

## Known limitations, stated plainly

- Eqs. (100)–(105) — near-degenerate inverse problems, competing singular sets
  and the numerator-cancellation criterion — are **not** simulated anywhere.
  They are theory only. See `EQUATION_MAP.md`.
- The Sec. VI.C claim that the largest relative deviation is "below 0.3%" is
  not reproduced: this archive obtains **0.434%** (driven by the
  bounded-uniform family), with all six 95% ranges still containing
  `alpha = 2`. `check_manuscript_consistency.py` therefore **FAILS this check
  deliberately**, and `run_all.py` exits 1. This is the intended behaviour: it
  is a real disagreement between the text and the code, and it should block
  submission until the manuscript sentence is corrected.

  To clear it: change the Sec. VI.C sentence to the reproduced bound (e.g.
  "below 0.5%"), then set `robustness_max_rel_deviation_pct` in
  `MANUSCRIPT_CLAIMS` at the top of `scripts/check_manuscript_consistency.py`
  to the same number. Do **not** clear it by widening a tolerance or by
  changing the check kind from `"below"` to `"approx"` — an earlier version of
  the script did exactly that and silently marked this discrepancy as PASS.
- Fig. 7 draws the rendered tail from the *measured* sampling distribution of
  the fitted linewidth rather than from 350,000 individual nonlinear fits,
  which would be computationally prohibitive. This is documented in the code
  and should be stated in the caption.

## Citation of tooling

If generative tooling assisted in producing any code, say so in the
acknowledgments **and** state that `check_manuscript_consistency.py` was run
and passed. The failure mode that produced the previous archive's defects was
plausible-looking code that was never checked against the manuscript.
