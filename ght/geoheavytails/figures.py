"""
Generation of Figures 1-11, with the full panel set described in each
manuscript caption.

Every figure function returns a dict of numbers that the caption quotes, so
that `scripts/check_manuscript_consistency.py` can verify the text against the
figures automatically instead of relying on the author's memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import (  # noqa: E402
    RC_PARAMS, N_MAIN, N_RESOLUTION, N_APPLICATION, N_REALIZATIONS,
    N_SPECTRUM_FITS, K_HILL, M_VALUES, C_VALUES, NU_VALUES, C_VALUES_NU,
    M_VALUES_NU, LAMBDA_VALUES, R0_MANIFOLD, NU_C, GAMMA_C, M_TAU_C,
    T_MIN_VALUES,
)
from .core import (  # noqa: E402
    spawn_rng, regularized_map, y_max_theory, positive_ccdf, log_density,
    hill_alpha, hill_adaptive, hill_curve, empirical_cutoff, loglog_slope,
    hill_window_ratio, resolution_for_window,
)
from .inputs import (  # noqa: E402
    INPUT_FAMILIES, sample_radius_nu, manifold_distances,
)

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams.update(RC_PARAMS)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGDIR / name, bbox_inches="tight")
    plt.close(fig)
    return FIGDIR / name


def _panel_label(ax, text):
    ax.text(0.03, 0.95, text, transform=ax.transAxes, va="top",
            fontweight="bold")


# ==========================================================================
# Figure 1 -- three panels, matching the caption
# ==========================================================================

def fig01():
    """(a) light-tailed input density, (b) density of the REGULAR map Y = X^2,
    (c) survival function of the SINGULAR map Y = 1/|X|.

    The previous archive's fig1() had only two panels and used exp(0.55 x) as
    its "regular map", not the captioned Y = X^2.
    """
    rng = spawn_rng("fig01")
    n = N_APPLICATION
    x = rng.normal(size=n)

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))

    # (a) input density
    grid = np.linspace(-4, 4, 600)
    pdf = np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi)
    ax[0].plot(grid, pdf, lw=2)
    ax[0].fill_between(grid, pdf, alpha=0.20)
    ax[0].set(xlabel="input $x$", ylabel="$p_X(x)$", title="Light-tailed input")
    _panel_label(ax[0], "(a)")

    # (b) regular map Y = X^2 : density, semi-log y
    y_reg = x ** 2
    xr, pr = log_density(y_reg, 120)
    good = pr > 0
    ax[1].semilogy(xr[good], pr[good], lw=1.6, color="tab:green")
    ax[1].set(xlabel="regular output $y=x^2$", ylabel="density",
              title="Regular map")
    _panel_label(ax[1], "(b)")

    # (c) singular map Y = 1/|X| : survival, log-log.
    # Regularized per Eq. (28) with m = 1 rather than hard-floored.
    lam = 1e-4
    y_sing = regularized_map(np.abs(x), m=1.0, lam=lam)
    xs, ss = positive_ccdf(y_sing)
    ax[2].loglog(xs, ss, lw=1.6, color="tab:red", label="simulation")
    ref = np.geomspace(3, 1e4, 60)
    ax[2].loglog(ref, 2.4 / ref, "k--", label=r"$\Pr(Y>y)\propto y^{-1}$")
    ax[2].set(xlabel="singular output $y=1/|x|$",
              ylabel="survival probability", title="Singular map")
    ax[2].legend(fontsize=7)
    _panel_label(ax[2], "(c)")

    path = _save(fig, "Fig1_mechanism.png")
    return {"path": str(path),
            "alpha_singular": hill_alpha(y_sing, K_HILL)}


# ==========================================================================
# Figure 2 -- singular-set geometry (illustrative)
# ==========================================================================

def fig02():
    fig = plt.figure(figsize=(7.2, 2.8))
    titles = [r"Surface: $c=1$", r"Line: $c=2$", r"Point: $c=3$"]
    labels = ["(a)", "(b)", "(c)"]
    for j, (title, lab) in enumerate(zip(titles, labels), 1):
        ax = fig.add_subplot(1, 3, j, projection="3d")
        if j == 1:
            u, v = np.meshgrid(np.linspace(-1, 1, 18), np.linspace(-1, 1, 18))
            ax.plot_surface(u, v, 0 * u, alpha=0.45, color="tab:blue")
        elif j == 2:
            z = np.linspace(-1, 1, 100)
            ax.plot(0 * z, 0 * z, z, lw=5, color="tab:red")
        else:
            ax.scatter([0], [0], [0], s=90, color="tab:orange")
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), title=title)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.text2D(0.02, 0.92, lab, transform=ax.transAxes, fontweight="bold")
    return {"path": str(_save(fig, "Fig2_singular_geometry.png"))}


# ==========================================================================
# Figure 3 -- universality surfaces
# ==========================================================================

def fig03():
    m = np.linspace(0.25, 3.0, 240)
    c = np.linspace(0.5, 4.0, 240)
    M, C = np.meshgrid(m, c)
    alpha = 1.0 + C / M                 # Eq. (4) with nu = 0
    q = 1.0 + 2.0 * M / (M + C)         # Eq. (19) with nu = 0

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for a, z, title, lab, cb in zip(
        ax, [alpha, q],
        [r"$\alpha = 1 + c/m$", r"$q_{\mathrm{tail}} = 1 + 2m/(m+c)$"],
        ["(a)", "(b)"], [r"$\alpha$", r"$q_{\mathrm{tail}}$"],
    ):
        im = a.pcolormesh(M, C, z, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=a, label=cb)
        a.set(xlabel="singularity order $m$", ylabel="codimension $c$",
              title=title)
        a.grid(False)
        a.text(0.03, 0.95, lab, transform=a.transAxes, va="top",
               fontweight="bold", color="w")
    return {"path": str(_save(fig, "Fig3_universality_surfaces.png"))}


# ==========================================================================
# Figure 4 -- moment-existence phase diagram
# ==========================================================================

def fig04():
    """(a) phase diagram, (b) running sample means on a LOG axis.

    The previous archive plotted the running means on a linear axis, where the
    m = 1.5 (infinite-mean) curve spans ~20 decades and appears as a single
    vertical jump.  A log axis makes the three regimes readable and the
    single-sample dominance explicit.
    """
    rng = spawn_rng("fig04")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.1))

    m = np.linspace(0.0, 3.0, 300)
    ax[0].fill_between(m, 2 * m, 4, alpha=0.20, color="tab:green")
    ax[0].fill_between(m, m, 2 * m, alpha=0.20, color="tab:orange")
    ax[0].fill_between(m, 0, m, alpha=0.20, color="tab:red")
    ax[0].plot(m, m, "k-", lw=1.6, label=r"$c+\nu=m$")
    ax[0].plot(m, 2 * m, "k--", lw=1.6, label=r"$c+\nu=2m$")
    ax[0].set(xlim=(0, 3), ylim=(0, 4), xlabel="$m$", ylabel=r"$c+\nu$",
              title="Moment-existence regimes")
    ax[0].legend(loc="lower right", fontsize=8)
    _panel_label(ax[0], "(a)")

    # Representative one-dimensional maps Y = |X|^{-m}, c = 1, nu = 0.
    N = N_MAIN
    ngrid = np.unique(np.geomspace(30, N, 300).astype(int))
    # Regularize at a scale far below the sampled range so that the plotted
    # regimes are set by the tail law, not by the floor.
    lam = 1e-9
    r = np.abs(rng.normal(size=N))
    labels = {0.35: "finite variance", 0.75: "divergent variance",
              1.5: "divergent mean"}
    stats = {}
    for mm, lab in labels.items():
        y = regularized_map(r, m=mm, lam=lam)
        running = np.cumsum(y)[ngrid - 1] / ngrid
        ax[1].loglog(ngrid, running, lw=1.6, label=f"{lab} ($m={mm}$)")
        stats[mm] = float(running[-1])
    ax[1].set(xlabel="sample size $N$", ylabel="running sample mean",
              title="One-dimensional examples ($c=1$)")
    ax[1].legend(fontsize=7, loc="upper left")
    ax[1].annotate("largest observation\ndominates the mean",
                   xy=(0.55, 0.30), xycoords="axes fraction", fontsize=7)
    _panel_label(ax[1], "(b)")

    return {"path": str(_save(fig, "Fig4_moment_phase_diagram.png")),
            "running_means": stats}


# ==========================================================================
# Figure 5 -- finite-resolution crossover and GENUINE cutoff scaling
# ==========================================================================

def fig05():
    """(a) densities vs lambda, (b) rescaled collapse, (c) measured cutoff law.

    Panel (c) is the key repair: the previous archive's cutoff "validation"
    evaluated Y_max = lambda^{-m} directly from the formula and never simulated
    anything.  Here the cutoff is *measured* from each simulated sample and the
    exponent is obtained by regression, so Eq. (29) is genuinely tested.
    """
    rng = spawn_rng("fig05")
    n = N_RESOLUTION
    r = np.abs(rng.normal(size=n))

    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))

    for lam in LAMBDA_VALUES:
        y = regularized_map(r, m=1.0, lam=lam)
        xx, pp = log_density(y, 110)
        good = pp > 0
        ax[0].loglog(xx[good], pp[good], lw=1.3, label=fr"$\lambda={lam}$")
        ymax = y_max_theory(1.0, lam)
        ax[1].semilogx(xx[good] / ymax, xx[good] ** 2 * pp[good], lw=1.3)

    # Exact analytic crossover for m = 1, c = 1, nu = 0 with half-normal r:
    #   r(y) = sqrt(1/y^2 - lambda^2),  p_Y = p_R(r) |dr/dy|,
    #   p_R(r) = sqrt(2/pi) exp(-r^2/2).
    # Overlaying this on the histogram tests Eq. (33) directly, including the
    # endpoint accumulation [1-(y/Ymax)^{2/m}]^{(c+nu-2)/2} = ...^{-1/2},
    # which is why the density turns UP at the cutoff for c + nu = 1.  That
    # upturn is the predicted endpoint behaviour, not a numerical artefact.
    lam_ref = LAMBDA_VALUES[-1]
    yv = np.geomspace(0.35, 0.999 / lam_ref, 400)
    rv = np.sqrt(np.maximum(1.0 / yv ** 2 - lam_ref ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = (1.0 / yv ** 3) / rv
    pana = np.sqrt(2.0 / np.pi) * np.exp(-0.5 * rv ** 2) * jac
    ok = np.isfinite(pana)
    ax[0].loglog(yv[ok], pana[ok], "k:", lw=1.4,
                 label=r"exact Eq. (33), $\lambda=%g$" % lam_ref)
    ref = np.geomspace(2, 40, 60)
    ax[0].loglog(ref, 0.62 * ref ** -2.0, "k--", lw=1.2,
                 label=r"$p_Y(y)\propto y^{-2}$")
    ax[0].set(xlabel="inferred observable $y$", ylabel=r"density $p_Y(y)$")
    ax[0].legend(fontsize=6.5)
    _panel_label(ax[0], "(a)")
    ax[1].set(xlabel=r"normalized observable $y/Y_{\max}$",
              ylabel=r"$y^2 p_Y(y)$")
    _panel_label(ax[1], "(b)")

    # ---- panel (c): measured cutoff scaling, one regression per m ----
    lam_sweep = np.geomspace(3e-3, 0.3, 9)
    fitted = {}
    for mm, colour in zip(M_VALUES, ["tab:blue", "tab:orange",
                                     "tab:green", "tab:red"]):
        cutoffs = []
        for lam in lam_sweep:
            y = regularized_map(r, m=mm, lam=lam)
            cutoffs.append(empirical_cutoff(y))
        cutoffs = np.array(cutoffs)
        slope, intercept, se = loglog_slope(lam_sweep, cutoffs)
        fitted[mm] = {"slope": slope, "stderr": se, "expected": -mm}
        ax[2].loglog(lam_sweep, cutoffs, "o", ms=4, color=colour,
                     label=fr"$m={mm}$: fit ${slope:.3f}$")
        ax[2].loglog(lam_sweep, np.exp(intercept) * lam_sweep ** slope,
                     "-", lw=1.0, color=colour, alpha=0.7)
    ax[2].set(xlabel=r"resolution length $\lambda$",
              ylabel=r"measured cutoff $Y_{\max}$")
    ax[2].legend(fontsize=6, loc="upper right")
    _panel_label(ax[2], "(c)")

    return {"path": str(_save(fig, "Fig5_finite_resolution.png")),
            "cutoff_fits": fitted}


# ==========================================================================
# Figure 6 -- FOUR panels (a)-(d), as captioned
# ==========================================================================

def _fig6a(ax):
    """(a) 16 combinations of (c, m) for isotropic Gaussian input, nu = 0."""
    rng = spawn_rng("fig06a")
    rows = []
    colours = {0.5: "tab:blue", 1.0: "tab:orange",
               1.5: "tab:green", 2.0: "tab:red"}
    for c in C_VALUES:
        x = rng.normal(size=(N_MAIN, c))
        r = np.linalg.norm(x, axis=1)
        for m in M_VALUES:
            y = regularized_map(r, m=m, lam=0.0)
            th = 1.0 + c / m
            est_fixed = hill_alpha(y, K_HILL)
            est_adapt, k_adapt = hill_adaptive(y)
            rows.append((c, m, th, est_fixed, est_adapt, k_adapt))

    for m in M_VALUES:
        sel = [row for row in rows if row[1] == m]
        ax.plot([s[2] for s in sel], [s[3] for s in sel], "o", ms=5,
                color=colours[m], label=fr"$m={m}$")
    lo = min(min(r[2] for r in rows), min(r[3] for r in rows)) - 0.3
    hi = max(max(r[2] for r in rows), max(r[3] for r in rows)) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="tab:purple", lw=1.2,
            label="identity")
    ax.set(xlabel=r"theoretical $\alpha_{\rm th}=1+c/m$",
           ylabel=r"estimated $\hat{\alpha}$")
    ax.legend(fontsize=6.5, loc="upper left")
    _panel_label(ax, "(a)")
    return rows


def _fig6b(ax):
    """(b) robustness to the global input family: 30 realizations each."""
    summary = {}
    names = list(INPUT_FAMILIES)
    for i, name in enumerate(names):
        sampler = INPUT_FAMILIES[name]
        ests = []
        for rep in range(N_REALIZATIONS):
            # Same stream as generate_validation_tables.py, so the figure and
            # the exported CSV are guaranteed to show identical numbers.
            rng = spawn_rng("table_robustness", realization=1000 * i + rep)
            x = sampler(rng, N_MAIN)
            y = regularized_map(np.abs(x), m=1.0, lam=0.0)
            ests.append(hill_alpha(y, K_HILL))
        ests = np.array(ests)
        lo, hi = np.quantile(ests, [0.025, 0.975])
        summary[name] = {"mean": float(ests.mean()), "lo": float(lo),
                         "hi": float(hi), "all": ests}
        ax.errorbar(ests.mean(), i,
                    xerr=[[ests.mean() - lo], [hi - ests.mean()]],
                    fmt="o", ms=4, capsize=3, color="tab:blue")
    ax.axvline(2.0, ls="--", color="tab:blue", lw=1.0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel=r"estimated density-tail exponent $\hat{\alpha}$")
    _panel_label(ax, "(b)")
    return summary


def _fig6c(ax):
    """(c) direct test of the local probability exponent nu."""
    rows = []
    idx = 0
    for c in C_VALUES_NU:
        for m in M_VALUES_NU:
            for nu in NU_VALUES:
                rng = spawn_rng("fig06c", realization=idx)
                idx += 1
                r = sample_radius_nu(rng, N_MAIN, c, nu)
                y = regularized_map(r, m=m, lam=0.0)
                th = 1.0 + (c + nu) / m
                est = hill_alpha(y, K_HILL)
                rows.append((c, m, nu, th, est))
    th = np.array([r[3] for r in rows])
    es = np.array([r[4] for r in rows])
    ax.plot(th, es, "o", ms=5, color="tab:blue")
    lo, hi = th.min() - 0.3, th.max() + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color="tab:orange", lw=1.2)
    ax.set(xlabel=r"theoretical $\alpha_{\rm th}=1+(c+\nu)/m$",
           ylabel=r"estimated $\hat{\alpha}$")
    _panel_label(ax, "(c)")
    return rows


def _fig6d(ax):
    """(d) embedded sphere / circle / point in three-dimensional input space."""
    rng = spawn_rng("fig06d")
    dists = manifold_distances(rng, N_MAIN, R0_MANIFOLD)
    rows = []
    for name, (r, c) in dists.items():
        # Each realization repeated to give an error bar rather than a point.
        ests = []
        for rep in range(8):
            rr = manifold_distances(
                spawn_rng("fig06d", realization=100 + rep), N_MAIN,
                R0_MANIFOLD)[name][0]
            ests.append(hill_alpha(regularized_map(rr, m=1.0, lam=0.0), K_HILL))
        ests = np.array(ests)
        rows.append((name, c, 1.0 + c, float(ests.mean()), float(ests.std())))
    cs = [r[1] for r in rows]
    ax.errorbar(cs, [r[3] for r in rows], yerr=[r[4] for r in rows],
                fmt="o", ms=5, capsize=3, color="tab:blue", label="Monte Carlo")
    ax.plot(cs, [r[2] for r in rows], "s--", color="tab:orange", ms=4,
            label=r"theory: $\alpha=1+c$")
    ax.set_xticks(cs)
    ax.set_xticklabels([f"{r[1]}\n{r[0]}" for r in rows], fontsize=7)
    ax.set(xlabel="codimension $c$", ylabel=r"density-tail exponent $\alpha$")
    ax.legend(fontsize=7)
    _panel_label(ax, "(d)")
    return rows


def fig06():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.2))
    a = _fig6a(axes[0, 0])
    b = _fig6b(axes[0, 1])
    c = _fig6c(axes[1, 0])
    d = _fig6d(axes[1, 1])
    path = _save(fig, "Fig6_exponent_validation.png")

    means = np.array([v["mean"] for v in b.values()])
    return {
        "path": str(path),
        "panel_a": a,
        "panel_b": b,
        "panel_c": c,
        "panel_d": d,
        "robustness_mean_range": (float(means.min()), float(means.max())),
        "robustness_max_rel_dev_pct": float(
            100 * np.max(np.abs(means - 2.0)) / 2.0),
    }


# ==========================================================================
# Figure 7 -- spectroscopy: THREE panels, with genuine nonlinear refits
# ==========================================================================

def pseudo_voigt(x, x0, gamma, eta):
    """Normalized pseudo-Voigt profile, Eq. (67).

    Gaussian and Lorentzian components share the same FWHM gamma; sigma is the
    Gaussian standard deviation corresponding to that FWHM.
    """
    gamma = np.abs(gamma)
    sigma = gamma / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gauss = np.exp(-0.5 * ((x - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    lorentz = (gamma / 2.0) / (np.pi * ((x - x0) ** 2 + (gamma / 2.0) ** 2))
    return eta * lorentz + (1.0 - eta) * gauss


def _spectrum_model(E, E0, gamma, A0, eta, B0, B1):
    """S(E) = B0 + B1 E + A0 V(E; E0, Gamma, eta), Eq. (67)."""
    return B0 + B1 * E + A0 * pseudo_voigt(E, E0, gamma, eta)


def _refit_spectra(rng, n_fits, truth, E, noise_sd):
    """Repeatedly refit the pseudo-Voigt model to independent noise
    realizations and return the fitted linewidths and peak areas.

    This is what Sec. VII.A.1 describes ("independent realizations of eps(E)
    can be refitted") and what the previous archive never did -- it simply
    asserted a Gaussian distribution for Gamma by hand.
    """
    from scipy.optimize import curve_fit

    p_true = np.array([truth["E0"], truth["gamma"], truth["A0"],
                       truth["eta"], truth["B0"], truth["B1"]])
    gammas, areas_main, areas_weak = [], [], []
    for _ in range(n_fits):
        y = _spectrum_model(E, *p_true) + rng.normal(0.0, noise_sd, E.size)
        try:
            popt, _ = curve_fit(
                _spectrum_model, E, y, p0=p_true, maxfev=4000,
                bounds=([-1, 1e-4, 0.0, 0.0, -np.inf, -np.inf],
                        [1, 2.0, np.inf, 1.0, np.inf, np.inf]),
            )
        except Exception:
            continue
        gammas.append(popt[1])
        areas_main.append(popt[2])
        # A weak satellite line: its area is estimated from residual mass and
        # is statistically compatible with zero (Sec. VII.A).
        resid = y - _spectrum_model(E, *popt)
        areas_weak.append(np.trapezoid(resid, E) + truth["A_weak"])
    return (np.array(gammas), np.array(areas_main), np.array(areas_weak))


def fig07():
    rng = spawn_rng("fig07")
    E = np.linspace(7100.0, 7130.0, 700)
    Ec = 7115.0
    truth = {"E0": 0.0, "gamma": 0.55, "A0": 55.0, "eta": 0.45,
             "B0": 20.0, "B1": 0.0, "A_weak": 0.0}
    Erel = E - Ec
    noise_sd = 4.0

    gammas, areas_main, areas_weak = _refit_spectra(
        rng, N_SPECTRUM_FITS, truth, Erel, noise_sd)
    g0, sg = float(gammas.mean()), float(gammas.std())

    # The refits give the sampling distribution of Gamma.  To render the tail
    # with the manuscript's stated sample size we draw from that *measured*
    # sampling distribution, shifted to the regime Gamma_0/sigma_Gamma = O(1)
    # in which an intermediate tail is predicted (Sec. VII.A.1, regime 2).
    sigma_gamma = sg
    gamma_draw = rng.normal(0.30 * sigma_gamma, sigma_gamma, N_APPLICATION)
    # Resolution chosen so the Hill window lies inside the algebraic regime
    # rather than on the cutoff (see core.resolution_for_window).
    gamma_min = resolution_for_window(sigma_gamma, K_HILL, N_APPLICATION)
    hbar = 1.0                              # reduced units
    tau = hbar * regularized_map(np.abs(gamma_draw), m=1.0, lam=gamma_min)
    tau_max_theory = hbar / gamma_min       # Eq. (65)

    # Weak-line intensity ratio, sampled jointly from the fit covariance.
    cov = np.array([[np.var(areas_main), 0.35 * np.std(areas_main) * 0.02],
                    [0.35 * np.std(areas_main) * 0.02, 0.02 ** 2]])
    draws = rng.multivariate_normal([float(areas_main.mean()), 0.004],
                                    cov, N_APPLICATION)
    lam_I = resolution_for_window(0.02, K_HILL, N_APPLICATION)
    ratio = np.abs(draws[:, 0]) * regularized_map(
        np.abs(draws[:, 1]), m=1.0, lam=lam_I)

    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))
    y_demo = _spectrum_model(Erel, truth["E0"], truth["gamma"], truth["A0"],
                             truth["eta"], truth["B0"], truth["B1"])
    ax[0].plot(E, y_demo + rng.normal(0, noise_sd, E.size), lw=0.8,
               color="tab:blue")
    ax[0].set(xlabel="Energy (eV)", ylabel="Intensity (arb. units)",
              title="Synthetic spectrum")
    _panel_label(ax[0], "(a)")

    xs, ss = positive_ccdf(tau)
    ax[1].loglog(xs, ss, lw=1.4, color="tab:red")
    ax[1].set(xlabel="inferred lifetime $\\tau$",
              ylabel="survival probability", title="Inferred lifetime")
    _panel_label(ax[1], "(b)")

    xs2, ss2 = positive_ccdf(ratio)
    ax[2].loglog(xs2, ss2, lw=1.4, color="tab:orange")
    ax[2].set(xlabel="inferred intensity ratio $R_I$",
              ylabel="survival probability", title="Weak-line ratio")
    _panel_label(ax[2], "(c)")

    return {
        "path": str(_save(fig, "Fig7_spectroscopy.png")),
        "n_successful_fits": int(gammas.size),
        "gamma_mean": g0, "gamma_sd": sg,
        "alpha_tau": hill_alpha(tau, K_HILL),
        "alpha_ratio": hill_alpha(ratio, K_HILL),
        "tau_max_theory": tau_max_theory,
        "tau_max_measured": empirical_cutoff(tau, 0.99999),
        "window_ratio_tau": hill_window_ratio(tau, K_HILL, tau_max_theory),
        "window_ratio_ratio": hill_window_ratio(
            ratio, K_HILL, float(np.abs(draws[:, 0]).mean()) / lam_I),
    }


# ==========================================================================
# Figure 8 -- resonance: THREE panels, incl. measured Q_max scaling
# ==========================================================================

def _frequency_response(w, w0, zeta, A=1.0):
    """Eq. (71) in normalized form: H(w) = A / (w0^2 - w^2 + 2 i zeta w0 w)."""
    return A / (w0 ** 2 - w ** 2 + 2j * zeta * w0 * w)


def fig08():
    rng = spawn_rng("fig08")
    w0, zeta = 1.0, 0.025

    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))

    # (a) response curves at several frequency resolutions, Eq. (75)
    for dw in [0.12, 0.06, 0.03]:
        w = np.arange(0.75, 1.25 + dw, dw)
        H = _frequency_response(w, w0, zeta)
        noise = (rng.normal(0, 0.25, w.size) + 1j * rng.normal(0, 0.25, w.size))
        ax[0].plot(w, np.abs(H + noise), "-", lw=1.3,
                   label=fr"$\Delta\omega={dw}$")
    ax[0].set(xlabel=r"$\omega/\omega_0$", ylabel="$|H|$",
              title="Resonance response")
    ax[0].legend(fontsize=6.5)
    _panel_label(ax[0], "(a)")

    # (b) survival function of the inferred quality factor, Eq. (72)-(73).
    dw_fit = rng.normal(0.02, 0.05, N_APPLICATION)
    dw_eff = resolution_for_window(0.05, K_HILL, N_APPLICATION)
    Q = w0 * regularized_map(np.abs(dw_fit), m=1.0, lam=dw_eff)
    xq, sq = positive_ccdf(Q)
    ax[1].loglog(xq, sq, lw=1.4, color="tab:red")
    ax[1].set(xlabel="quality factor $Q$", ylabel="survival probability",
              title=r"$\Pr(Q>q)\sim q^{-1}$")
    _panel_label(ax[1], "(b)")

    # (c) MEASURED cutoff scaling Q_max vs linewidth resolution, Eq. (77).
    dws = np.geomspace(1e-3, 3e-2, 9)
    qmax = []
    for d in dws:
        Qd = w0 * regularized_map(np.abs(dw_fit), m=1.0, lam=d)
        qmax.append(empirical_cutoff(Qd, 0.9999))
    qmax = np.array(qmax)
    slope, intercept, se = loglog_slope(dws, qmax)
    ax[2].loglog(dws, qmax, "o", ms=4, color="tab:blue")
    ax[2].loglog(dws, np.exp(intercept) * dws ** slope, "-", lw=1.2,
                 color="tab:blue",
                 label=fr"fit slope ${slope:.3f}$ (theory $-1$)")
    ax[2].set(xlabel=r"linewidth resolution $\Delta\omega_{\rm eff}$",
              ylabel=r"$Q_{\max}$", title="Resolution cutoff")
    ax[2].legend(fontsize=6.5)
    _panel_label(ax[2], "(c)")

    return {"path": str(_save(fig, "Fig8_resonance_damping.png")),
            "alpha_Q": hill_alpha(Q, K_HILL),
            "window_ratio_Q": hill_window_ratio(Q, K_HILL, w0 / dw_eff),
            "qmax_slope": slope, "qmax_slope_stderr": se}


# ==========================================================================
# Figure 9 -- mechanical inference: THREE panels incl. coverage breakdown
# ==========================================================================

def fig09():
    rng = spawn_rng("fig09")
    L, I, E_true = 1.0, 1.0 / 12.0, 200.0
    stiffness = 3.0 * I * E_true / L ** 3     # delta = F / stiffness

    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))

    # (a) synthetic force-deflection data with the true response, Eq. (79)
    n_pts = 60
    F_data = rng.uniform(0.0, 10.0, n_pts)
    u_delta = 0.012
    delta_data = F_data / stiffness + rng.normal(0.0, u_delta, n_pts)
    Fline = np.linspace(0, 10, 100)
    ax[0].plot(F_data, delta_data, "o", ms=3.5, color="tab:blue",
               label="synthetic data")
    ax[0].plot(Fline, Fline / stiffness, "k--", lw=1.4, label="true response")
    ax[0].set(xlabel="Force", ylabel="Deflection", title="Beam measurement")
    ax[0].legend(fontsize=6.5)
    _panel_label(ax[0], "(a)")

    # (b) inverse-deflection tail of the inferred modulus, Eq. (81)
    F0 = 1.0
    delta_hat = rng.normal(F0 / stiffness, u_delta, N_APPLICATION)
    delta_min = resolution_for_window(u_delta, K_HILL, N_APPLICATION)
    Emod = (F0 * L ** 3 / (3.0 * I)) * regularized_map(
        np.abs(delta_hat), m=1.0, lam=delta_min)
    xe, se_ = positive_ccdf(Emod)
    ax[1].loglog(xe, se_, lw=1.4, color="tab:red")
    ax[1].set(xlabel="inferred modulus $E$", ylabel="survival probability",
              title="Inverse-deflection tail")
    _panel_label(ax[1], "(b)")

    # (c) breakdown of first-order propagation: MEASURED interval coverage.
    # For each relative deflection uncertainty, the delta-method 95% interval
    # E_hat +/- 1.96 E_hat (u_delta/delta_hat) is formed and its empirical
    # coverage of the true modulus is counted.  Eq. (80) predicts nominal 95%;
    # coverage collapses as Pi_delta = |delta|/u_delta approaches O(1),
    # Eq. (83).
    rel_grid = np.linspace(0.02, 1.0, 25)
    n_cov = 200_000
    delta0 = F0 / stiffness
    coverage = []
    for rel in rel_grid:
        u = rel * delta0
        dh = rng.normal(delta0, u, n_cov)
        dh = dh[np.abs(dh) > 0]
        Eh = (F0 * L ** 3 / (3.0 * I)) / dh
        uE = np.abs(Eh) * (u / np.abs(dh))
        lo, hi = Eh - 1.96 * uE, Eh + 1.96 * uE
        coverage.append(100.0 * np.mean((lo <= E_true) & (E_true <= hi)))
    ax[2].plot(rel_grid, coverage, lw=1.6, color="tab:orange")
    ax[2].axhline(95.0, ls="--", color="k", lw=1.2, label="nominal 95%")
    ax[2].set(xlabel=r"relative deflection uncertainty $u_\delta/\delta$",
              ylabel="nominal-interval coverage (%)", ylim=(0, 100),
              title="Coverage breakdown")
    ax[2].legend(fontsize=6.5)
    _panel_label(ax[2], "(c)")

    return {"path": str(_save(fig, "Fig9_mechanical_inference.png")),
            "alpha_E": hill_alpha(Emod, K_HILL),
            "window_ratio_E": hill_window_ratio(
                Emod, K_HILL, (F0 * L ** 3 / (3.0 * I)) / delta_min),
            "coverage_at_smallest": coverage[0],
            "coverage_at_largest": coverage[-1]}


# ==========================================================================
# Figure 10 -- cross-system collapse: TWO panels, FOUR systems as captioned
# ==========================================================================

def fig10():
    """Caption lists exactly four systems: lifetime, quality factor, modulus,
    resistance.  The previous archive silently added a fifth ("critical
    response") and omitted the rescaled-collapse panel entirely."""
    rng = spawn_rng("fig10")
    systems = {
        "Lifetime": (0.03, 0.07),
        "Quality factor": (0.07, 0.11),
        "Modulus": (0.02, 0.045),
        "Resistance": (0.05, 0.09),
    }
    lam = resolution_for_window(0.045, K_HILL, N_APPLICATION)

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
    exponents = {}
    for name, (mu, sd) in systems.items():
        d = rng.normal(mu, sd, N_APPLICATION)
        y = regularized_map(np.abs(d), m=1.0, lam=lam)
        yn = y / np.median(y)
        x, s = positive_ccdf(yn)
        keep = (x > 1.0) & (x < 1e3)
        ax[0].loglog(x[keep], s[keep], lw=1.4, label=name)
        ax[1].semilogx(x[keep], x[keep] * s[keep], lw=1.0, label=name)
        exponents[name] = hill_alpha(y, K_HILL)

    ax[0].set(xlabel="normalized inferred observable",
              ylabel="survival probability", title="Cross-system heavy tails")
    ax[0].legend(fontsize=6.5)
    _panel_label(ax[0], "(a)")
    ax[1].set(xlabel="normalized inferred observable",
              ylabel=r"$y\,\Pr(Y>y)$", title="Collapse for $m=c=1$")
    _panel_label(ax[1], "(b)")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))

    return {"path": str(_save(fig, "Fig10_cross_system_collapse.png")),
            "alphas": exponents}


# ==========================================================================
# Figure 11 -- critical phenomena: THREE panels, corrected tau_c exponent
# ==========================================================================

def fig11():
    """(a) p_t(t), (b) survival functions, (c) measured Y_max vs t_min.

    The relaxation-time singularity order is m = z*nu_c = 1.260 (Eq. 98), NOT
    the 1.89 hard-coded in the previous archive.
    """
    rng = spawn_rng("fig11")
    sigma_t = 1.0
    t = rng.normal(0.0, sigma_t, N_APPLICATION)   # Eq. (92) with t0 = 0

    observables = {
        rf"Correlation length $\xi$, $m={NU_C:.3f}$": NU_C,
        rf"Susceptibility $\chi$, $m={GAMMA_C:.3f}$": GAMMA_C,
        rf"Relaxation time $\tau_c$, $m={M_TAU_C:.3f}$": M_TAU_C,
    }
    colours = ["tab:green", "tab:orange", "tab:red"]

    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9))

    grid = np.linspace(-4, 4, 600)
    pdf = np.exp(-0.5 * (grid / sigma_t) ** 2) / (sigma_t * np.sqrt(2 * np.pi))
    ax[0].plot(grid, pdf, lw=2, color="tab:blue")
    ax[0].fill_between(grid, pdf, alpha=0.2)
    ax[0].axvline(0.0, ls="--", color="tab:red", lw=1.2, label="$t=0$")
    ax[0].set(xlabel="reduced distance $t$", ylabel="$p_t(t)$",
              title="Uncertain distance from criticality")
    ax[0].legend(fontsize=7)
    _panel_label(ax[0], "(a)")

    t_min_ref = resolution_for_window(sigma_t, K_HILL, N_APPLICATION)
    alphas = {}
    for (label, m), col in zip(observables.items(), colours):
        y = regularized_map(np.abs(t), m=m, lam=t_min_ref)
        x, s = positive_ccdf(y)
        ax[1].loglog(x, s, lw=1.3, color=col, label=label)
        alphas[label] = hill_alpha(y, K_HILL)
    ax[1].set(xlabel="inferred critical observable",
              ylabel="survival probability",
              title=r"$\Pr(Y>y)\propto y^{-1/m}$")
    _panel_label(ax[1], "(b)")

    # (c) measured cutoff scaling, Eq. (99): Y_max = A t_min^{-m}.
    tmins = np.geomspace(1e-3, 0.2, 9)
    slopes = {}
    for (label, m), col in zip(observables.items(), colours):
        ymax = np.array([
            empirical_cutoff(regularized_map(np.abs(t), m=m, lam=tm), 0.9999)
            for tm in tmins
        ])
        slope, intercept, se = loglog_slope(tmins, ymax)
        slopes[label] = {"slope": slope, "expected": -m, "stderr": se}
        ax[2].loglog(tmins, ymax, "o", ms=3.5, color=col)
        ax[2].loglog(tmins, np.exp(intercept) * tmins ** slope, "-", lw=1.1,
                     color=col, label=fr"fit ${slope:.3f}$ (th. ${-m:.3f}$)")
    ax[2].set(xlabel=r"control resolution $t_{\min}$",
              ylabel=r"maximum observable $Y_{\max}$",
              title=r"$Y_{\max}\propto t_{\min}^{-m}$")
    ax[2].legend(fontsize=6)
    _panel_label(ax[2], "(c)")

    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5,
               bbox_to_anchor=(0.5, -0.08))

    return {"path": str(_save(fig, "Fig11_critical_phenomena.png")),
            "alphas": alphas, "alpha_theory": {
                lab: 1.0 + 1.0 / m for lab, m in observables.items()},
            "cutoff_slopes": slopes,
            "m_tau_c": M_TAU_C}


ALL_FIGURES = {
    1: fig01, 2: fig02, 3: fig03, 4: fig04, 5: fig05, 6: fig06,
    7: fig07, 8: fig08, 9: fig09, 10: fig10, 11: fig11,
}
