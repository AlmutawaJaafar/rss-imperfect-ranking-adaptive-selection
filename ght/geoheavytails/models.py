"""
Likelihood-based comparison of competing tail models.

Sec. VI.E item 5 and Sec. IX item 6 of the manuscript require that any claimed
power law be compared against Pareto, truncated-Pareto, lognormal,
stretched-exponential and q-Gaussian alternatives.  The previous archive
demanded this of experimentalists but never performed it on its own synthetic
data, so every "validation" was the paper's own Hill estimator checked against
the paper's own formula.  This module supplies the missing comparison.

All models are fitted by maximum likelihood *conditional on X > xmin*, i.e. to
the same tail sample that the Hill estimator uses, so the likelihoods are
directly comparable.  Model selection is by AIC.
"""

from __future__ import annotations

import numpy as np
import warnings

from scipy import optimize, special, integrate

__all__ = ["fit_all_models", "MODEL_NAMES"]

MODEL_NAMES = (
    "Pareto",
    "Truncated Pareto",
    "Lognormal",
    "Stretched exponential",
    "q-Gaussian",
)


def _tail(x, xmin):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > xmin)]
    return x


# --------------------------------------------------------------------------
# Pareto
# --------------------------------------------------------------------------

def fit_pareto(x, xmin):
    """f(x) = (a-1) xmin^{a-1} x^{-a},  x > xmin.  Closed-form MLE (Hill)."""
    x = _tail(x, xmin)
    n = x.size
    s = np.sum(np.log(x / xmin))
    a = 1.0 + n / s
    ll = n * np.log(a - 1.0) + (a - 1.0) * n * np.log(xmin) - a * np.sum(np.log(x))
    return {"params": {"alpha": a}, "loglik": float(ll), "k": 1}


# --------------------------------------------------------------------------
# Truncated Pareto
# --------------------------------------------------------------------------

def fit_truncated_pareto(x, xmin):
    """f(x) = (a-1) x^{-a} / (xmin^{1-a} - xt^{1-a}),  xmin < x < xt.

    The MLE of the upper truncation point is the sample maximum; a is then
    obtained numerically.  This is the model that a genuinely
    resolution-truncated observable (Eq. 33) should favour.
    """
    x = _tail(x, xmin)
    n = x.size
    xt = float(np.max(x)) * (1.0 + 1e-12)
    slog = np.sum(np.log(x))

    def nll(p):
        a = p[0]
        if a <= 1.0 + 1e-9:
            return np.inf
        denom = xmin ** (1.0 - a) - xt ** (1.0 - a)
        if denom <= 0:
            return np.inf
        return -(n * np.log(a - 1.0) - n * np.log(denom) - a * slog)

    a0 = 1.0 + n / np.sum(np.log(x / xmin))
    res = optimize.minimize(nll, [a0], method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-8})
    return {
        "params": {"alpha": float(res.x[0]), "xt": xt},
        "loglik": float(-res.fun),
        "k": 2,
    }


# --------------------------------------------------------------------------
# Lognormal (left-truncated at xmin)
# --------------------------------------------------------------------------

def fit_lognormal(x, xmin):
    x = _tail(x, xmin)
    n = x.size
    lx = np.log(x)
    lmin = np.log(xmin)

    def nll(p):
        mu, sigma = p
        if sigma <= 1e-8:
            return np.inf
        z = (lmin - mu) / sigma
        surv = 0.5 * special.erfc(z / np.sqrt(2.0))
        if surv <= 1e-300:
            return np.inf
        ll = (
            -np.sum(lx)
            - n * np.log(sigma * np.sqrt(2 * np.pi))
            - np.sum((lx - mu) ** 2) / (2 * sigma ** 2)
            - n * np.log(surv)
        )
        return -ll

    res = optimize.minimize(nll, [np.mean(lx), max(np.std(lx), 0.1)],
                            method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-8,
                                     "maxiter": 5000})
    return {
        "params": {"mu": float(res.x[0]), "sigma": float(res.x[1])},
        "loglik": float(-res.fun),
        "k": 2,
    }


# --------------------------------------------------------------------------
# Stretched exponential / Weibull (left-truncated at xmin)
# --------------------------------------------------------------------------

def fit_stretched_exponential(x, xmin):
    """f(x) ∝ (b/s)(x/s)^{b-1} exp[-((x/s)^b - (xmin/s)^b)],  x > xmin."""
    x = _tail(x, xmin)
    n = x.size

    def nll(p):
        logb, logs = p
        if abs(logb) > 20 or abs(logs) > 40:
            return np.inf
        b, s = np.exp(logb), np.exp(logs)
        if not np.isfinite(b) or not np.isfinite(s):
            return np.inf
        with np.errstate(over="ignore", invalid="ignore"):
            z = (x / s) ** b
            zmin = (xmin / s) ** b
            ll = (
                n * np.log(b / s)
                + (b - 1.0) * np.sum(np.log(x / s))
                - np.sum(z)
                + n * zmin
            )
        return -ll if np.isfinite(ll) else np.inf

    res = optimize.minimize(nll, [np.log(0.5), np.log(np.median(x))],
                            method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-8,
                                     "maxiter": 8000})
    return {
        "params": {"beta": float(np.exp(res.x[0])),
                   "scale": float(np.exp(res.x[1]))},
        "loglik": float(-res.fun),
        "k": 2,
    }


# --------------------------------------------------------------------------
# q-Gaussian tail (one-sided, left-truncated at xmin)
# --------------------------------------------------------------------------

def fit_q_gaussian(x, xmin):
    """f(x) ∝ [1 + (q-1) b x^2]^{-1/(q-1)},  x > xmin, 1 < q < 3.

    This is the model implicitly invoked whenever an empirical q_tail > 1 is
    read as evidence of nonextensivity (Eq. 19 and Sec. X).  Including it here
    lets us state whether the observation-induced sample is actually
    distinguishable from a q-Gaussian on the fitted range.
    """
    x = _tail(x, xmin)
    n = x.size

    def log_norm(q, b):
        expo = 1.0 / (q - 1.0)

        def integrand(t):
            return (1.0 + (q - 1.0) * b * t * t) ** (-expo)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val, _ = integrate.quad(integrand, xmin, np.inf, limit=200)
        return np.log(val) if val > 0 else np.inf

    def nll(p):
        q = 1.0 + np.exp(p[0])       # q > 1
        b = np.exp(p[1])
        if q >= 3.0 - 1e-6:
            return np.inf
        try:
            lz = log_norm(q, b)
        except Exception:
            return np.inf
        if not np.isfinite(lz):
            return np.inf
        ll = -(1.0 / (q - 1.0)) * np.sum(
            np.log1p((q - 1.0) * b * x * x)
        ) - n * lz
        return -ll if np.isfinite(ll) else np.inf

    res = optimize.minimize(nll, [np.log(1.0), np.log(1.0)],
                            method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-6,
                                     "maxiter": 2000})
    q = 1.0 + float(np.exp(res.x[0]))
    return {
        "params": {"q": q, "beta": float(np.exp(res.x[1]))},
        "loglik": float(-res.fun),
        "k": 2,
    }


_FITTERS = {
    "Pareto": fit_pareto,
    "Truncated Pareto": fit_truncated_pareto,
    "Lognormal": fit_lognormal,
    "Stretched exponential": fit_stretched_exponential,
    "q-Gaussian": fit_q_gaussian,
}


def fit_all_models(x, xmin, subsample=None, rng=None):
    """Fit every competing model to the tail x > xmin and rank by AIC.

    `subsample` caps the number of tail points used, because the q-Gaussian
    normalization requires a quadrature per likelihood evaluation and is the
    rate-limiting step.  AIC differences are reported per model.
    """
    x = _tail(x, xmin)
    if subsample is not None and x.size > subsample:
        rng = rng or np.random.default_rng(0)
        x = rng.choice(x, size=subsample, replace=False)

    out = {}
    for name, fitter in _FITTERS.items():
        try:
            r = fitter(x, xmin)
            r["aic"] = 2 * r["k"] - 2 * r["loglik"]
        except Exception as exc:  # pragma: no cover - diagnostic path
            r = {"params": {}, "loglik": np.nan, "k": np.nan,
                 "aic": np.inf, "error": str(exc)}
        out[name] = r

    best = min(out, key=lambda k: out[k]["aic"])
    for name in out:
        out[name]["delta_aic"] = out[name]["aic"] - out[best]["aic"]
    return out, best, x.size
