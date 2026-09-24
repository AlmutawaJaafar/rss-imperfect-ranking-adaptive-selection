# Equation-to-code map

Every equation in the manuscript is listed below with the code object that
implements or tests it. "Analytic only" means the equation is a derivation step
with no separate numerical counterpart.

| Eq. | Statement | Implemented in | Tested by |
|-----|-----------|----------------|-----------|
| (1) | `Y = f(X)` | `core.regularized_map` | all figures |
| (2) | regular variation of the survival function | analytic only | `models.fit_pareto` (regular-variation index via MLE) |
| (3) | `tau=hbar/Gamma`, `Q=f0/df`, `E=FL^3/(3 I d)`, `RI=I1/I2` | `figures.fig07`, `fig08`, `fig09` | Figs. 7-9 |
| (4) | **`alpha = 1 + (c+nu)/m`** | `config` grids + `core.hill_alpha` | Fig. 6(a)-(d); `exponent_validation.csv`, `nu_validation.csv`, `manifold_validation.csv` |
| (5) | pushforward integral | analytic only | — |
| (6) | `f(x) = A r^{-m}[1+O(r^w)]` | `core.regularized_map(..., lam=0)` | Fig. 6(a) |
| (7) | shell measure `dP ~ B r^{c-1+nu} dr` | `inputs.sample_radius_nu` | Fig. 6(c) |
| (8)-(9) | slowly varying factor `L` | analytic only | — |
| (10) | `r(y)`, `|dr/dy|` | inverse of `core.regularized_map` | Fig. 5(a) analytic overlay |
| (11) | `p_Y(y) ~ (B/m) A^{(c+nu)/m} y^{-[1+(c+nu)/m]}` | `core.log_density` + theory line | Fig. 5(a); Fig. 6 |
| (12) | survival law | `core.positive_ccdf` | Figs. 1(c), 7(b,c), 8(b), 9(b), 10, 11(b) |
| (13) | `Y>y  <=>  r < (A/y)^{1/m}` | analytic only | — |
| (14) | correction to scaling, `Delta = min(w,eta)/m` | — | `threshold_stability.csv` (drifting local slope is measured) |
| (15)-(16) | ratio limit `Y=A/X`, `p_Y ~ A p_X(0) y^{-2}` | `m=c=1` cases | Fig. 6(b); `robustness_summary.csv` |
| (17) | class `C=(m,c,nu)` | `config.M_VALUES`, `C_VALUES`, `NU_VALUES` | Fig. 6 |
| (18)-(19) | `q_tail = 1 + 2m/(m+c+nu)` | `figures.fig03` | `models.fit_q_gaussian` in `model_comparison.csv` |
| (20)-(21) | reparameterization stability | analytic only | — |
| (22)-(23) | `kappa = (c+nu)/m`; moment finite for `s<kappa` | `generate_validation_tables.moment_benchmark` | `exact_gaussian_moment_benchmark.csv` |
| (24)-(25) | mean/variance existence boundaries | `figures.fig04` panel (a) | Fig. 4(a) |
| (26)-(27) | largest-event and running-mean scaling | `figures.fig04` panel (b) | Fig. 4(b) |
| (28) | **`Y_lambda = A(r^2+lambda^2)^{-m/2}`** | `core.regularized_map` | Fig. 5; every application figure |
| (29) | **`Y_max = A lambda^{-m}`** | `core.y_max_theory` (comparison only) | `core.empirical_cutoff` + `core.loglog_slope` -> Fig. 5(c), `cutoff_scaling_fits.csv` |
| (30) | `rho = lambda/sigma_r` | `core.resolution_for_window` | window ratios in `figure_summary.json` |
| (31) | scaling form `Phi(y/Ymax, rho)` | — | Fig. 5(b) collapse |
| (32)-(33) | inversion and crossover density | analytic overlay in `figures.fig05` | Fig. 5(a) dotted curve |
| (34) | local survival slope `kappa_eff` | `core.hill_curve` | `threshold_stability.csv` |
| (35) | `Y=(X_1^2+...+X_c^2)^{-m/2}` | `figures._fig6a` | Fig. 6(a) |
| (36)-(38) | parameter grid, `N`, `alpha_th=1+c/m` | `config.M_VALUES`, `N_MAIN` | `check_manuscript_consistency.py` |
| (39)-(40) | Hill estimator | `core.hill_gamma`, `core.hill_alpha` | all exponent tables |
| (41) | primary test `alpha_hat =? 1+(c+nu)/m` | `generate_validation_tables` | `exponent_validation.csv` |
| (42) | `Y=|X|^{-m}` | `figures.fig04` | Fig. 4(b) |
| (43)-(45) | resolution test, `N`, `lambda` set, `Ymax=lambda^{-1}` | `config.N_RESOLUTION`, `LAMBDA_VALUES` | Fig. 5 |
| (46)-(47) | multi-`m` cutoff law, `ln Ymax = ln A - m ln lambda` | `figures.fig05` panel (c) | `cutoff_scaling_fits.csv` (regressed slopes) |
| (48) | `p_X(0)=1/sqrt(2 pi)` for all families | `config.P0_TARGET`, `inputs.INPUT_FAMILIES` | `inputs.density_at_zero_empirical` |
| (49)-(50) | `Y=|X|^{-1}`, `alpha_th=2` | `figures._fig6b` | Fig. 6(b), `robustness_*.csv` |
| (51)-(52) | radial density; `R^2 ~ Gamma((c+nu)/2, 2)` | `inputs.sample_radius_nu` | Fig. 6(c) |
| (53)-(55) | `Y=R^{-m}`; the 18 `(c,m,nu)` classes | `figures._fig6c` | `nu_validation.csv` |
| (56)-(59) | embedded manifold distances `r_sph`, `r_circ`, `r_pt` | `inputs.manifold_distances` | Fig. 6(d) |
| (60)-(61) | `Y=r^{-1}`; `alpha = 2, 3, 4` | `figures._fig6d` | `manifold_validation.csv` |
| (62)-(65) | `tau=hbar/Gamma`; `tau_max ~ hbar/Gamma_min` | `figures.fig07` | Fig. 7(b) |
| (66) | intensity ratio `RI=I1/I2` | `figures.fig07` | Fig. 7(c) |
| (67)-(68) | pseudo-Voigt spectrum and fitted parameters | `figures.pseudo_voigt`, `_spectrum_model`, `_refit_spectra` | Fig. 7(a); 3000 genuine nonlinear refits |
| (69)-(70) | `tau_max = hbar/Gamma_min`; joint test | `figures.fig07` | `figure_summary.json` (`tau_max_theory` vs `tau_max_measured`) |
| (71)-(72) | `H(w)`; `Q = w0/dw ~ 1/(2 zeta)` | `figures._frequency_response` | Fig. 8(a) |
| (73) | `p_Q(Q) ~ Q^{-2}` | `figures.fig08` | Fig. 8(b), `alpha_Q` |
| (74) | `tau_r = 1/gamma` | same class as (73) | — |
| (75) | noisy response sampling | `figures.fig08` | Fig. 8(a) |
| (76)-(77) | `p_Q ~ Q^{-2}`; `Qmax ∝ (dw_eff)^{-1}` | `figures.fig08` panel (c) | measured slope in `figure_summary.json` |
| (78) | generalized susceptibility | analytic only | — |
| (79)-(80) | `E = FL^3/(3 I d)`; first-order propagation | `figures.fig09` | Fig. 9(a) |
| (81)-(82) | `p_E ~ E^{-2}`; `Emax ∝ d_min^{-1}` | `figures.fig09` | Fig. 9(b), `alpha_E` |
| (83) | `Pi_d = |d|/u_d` | `figures.fig09` panel (c) | measured coverage collapse |
| (84)-(87) | `R=1/G`, `n=1/(e R_H)`, `mu=sigma/(ne)`, `Rth` | `figures.fig10` | Fig. 10 |
| (88) | `Pr(Y>y) ∝ y^{-1}` collapse | `figures.fig10` panel (b) | Fig. 10(b) plateau |
| (89)-(90) | `Y ~ |g-gc|^{-m}`; `p_Y ~ y^{-(1+1/m)}` | `figures.fig11` | Fig. 11(b) |
| (91)-(94) | reduced distance `t`; resolved `t_hat`; density | `figures.fig11` panel (a) | Fig. 11(a) |
| (95)-(98) | `chi,xi,tau_c` exponents; **`alpha_tau = 1+(z nu_c)^{-1}`** | `config.NU_C`, `GAMMA_C`, `Z_C`, `M_TAU_C` | `check_manuscript_consistency.py` (this is the check that catches the 1.89 error) |
| (99) | `Ymax = A t_min^{-m}` | `figures.fig11` panel (c) | regressed slopes in `figure_summary.json` |
| (100) | `||x_inv|| ~ s_min^{-m}` | — | not simulated; flagged in README |
| (101) | competing singularities sum | — | not simulated; flagged in README |
| (102)-(104) | numerator cancellation, `m_eff = m - l` | — | not simulated; flagged in README |
| (105) | `kappa_out = min(kappa_remote, (c+nu)/m)` | — | not simulated; flagged in README |
| (106)-(109) | joint diagnostic `alpha` and `Ymax ∝ lambda^{-m}` | `check_manuscript_consistency.py` | full check table |
| (110)-(111) | conclusion restatements of (4) and (19) | see (4), (19) | — |
| (A1)-(A5) | chi distribution; exact `p_Y`; exact moments | `generate_validation_tables.moment_benchmark` | `exact_gaussian_moment_benchmark.csv` (exact vs Monte Carlo) |
| (B1)-(B5) | two-branch one-dimensional transformation | `figures._fig6b` (both branches via `|X|`) | Fig. 6(b) |

## Equations with no numerical counterpart

Eqs. (100)-(105) — near-degenerate inverse problems, competing singular sets,
and the numerator-cancellation criterion — are stated in the manuscript but are
**not** simulated anywhere in this archive. They are theoretical claims only.
Either add simulations or state plainly in the text that these are not
numerically tested; do not let a reader infer that Fig. 6 validates them.
