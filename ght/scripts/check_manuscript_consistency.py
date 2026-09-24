#!/usr/bin/env python3
"""
Automated check of every number the manuscript quotes against the numbers this
code actually produces.

Rationale
---------
The defects found in the previous archive were all of one kind: a value stated
in the text or a caption had drifted away from the value in the code, and
nothing checked.  The clearest case was the relaxation-time singularity order,
quoted as m = 1.260 in the Fig. 11 caption but hard-coded as 1.89.  A referee
found it by reading; it should have been impossible to submit.

Every manuscript claim is encoded below as an assertion with an explicit
comparison rule.  Exit status is nonzero if any check fails, so this can gate
submission and be wired into CI.

Comparison kinds
----------------
  "exact"   value must equal the claim exactly (integers: sample sizes, k)
  "approx"  |value - claim| <= tol            (a target the code should hit)
  "below"   value < claim                     (an upper bound asserted in text)
  "above"   value > claim                     (a lower bound asserted in text)
  "bool"    a qualitative statement, true or false
  "info"    reported but never fails          (context for the reader)

A "below" claim must NEVER be written as "approx": an upper bound of 0.3 with
a tolerance of 0.3 silently accepts 0.6.  That bug was present in the first
version of this script and marked a real discrepancy as PASS.

When a check fails, fix the manuscript or fix the code.  Do not widen the
tolerance, and do not downgrade a "below" check to an "approx" check.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geoheavytails.config import (  # noqa: E402
    NU_C, GAMMA_C, Z_C, M_TAU_C, N_MAIN, N_RESOLUTION, N_REALIZATIONS, K_HILL,
)

DATA = ROOT / "data"


# ==========================================================================
# EVERY NUMBER THE MANUSCRIPT STATES.  Edit here when you edit the paper.
# ==========================================================================
MANUSCRIPT_CLAIMS = {
    # Fig. 11 caption / Eqs. (95)-(98)
    "m_xi": 0.630,
    "m_chi": 1.237,
    "m_tau_c": 1.260,

    # Sec. VI.A, Eqs. (37), (43)
    "N_main": 350_000,
    "N_resolution": 1_200_000,
    "n_realizations": 30,
    "hill_k": 4_000,

    # Sec. VI.C -- an UPPER BOUND stated in the text ("below 0.3%").
    # This archive reproduces 0.434%, so the check FAILS by design until the
    # manuscript sentence is updated.  See README, "Known limitations".
    "robustness_max_rel_deviation_pct": 0.30,

    # Eq. (61): embedded-manifold exponents
    "alpha_sphere": 2.0,
    "alpha_circle": 3.0,
    "alpha_point": 4.0,

    # Tolerances that are themselves methodological choices, not free knobs.
    # The Hill estimator has asymptotic standard error alpha/sqrt(k); a
    # deviation is acceptable only within this many standard errors.
    "hill_bias_max_standard_errors": 3.0,
    "nu_test_max_abs_bias": 0.25,
    "manifold_alpha_tol": 0.15,
    "cutoff_slope_tol": 0.01,
    "robustness_mean_tol": 0.02,
}


# ==========================================================================
# Check framework
# ==========================================================================
RESULTS = []


def check(name, kind, claim, value, tol=None, note=""):
    """Record one check.  `kind` selects the comparison rule (see docstring)."""
    if kind == "exact":
        ok, crit = (value == claim), "=="
    elif kind == "approx":
        ok, crit = (abs(value - claim) <= tol), f"+/-{tol:g}"
    elif kind == "below":
        ok, crit = (value < claim), "<"
    elif kind == "above":
        ok, crit = (value > claim), ">"
    elif kind == "bool":
        ok, crit = bool(value), "is true"
    elif kind == "info":
        ok, crit = True, "(info)"
    else:
        raise ValueError(f"unknown check kind: {kind}")
    RESULTS.append({"name": name, "kind": kind, "claim": claim,
                    "value": value, "criterion": crit, "ok": ok, "note": note})
    return ok


def read_csv(name):
    with open(DATA / name) as f:
        return list(csv.DictReader(f))


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return f"{v:d}"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)[:14]


# ==========================================================================
# The checks
# ==========================================================================

def run_checks():
    C = MANUSCRIPT_CLAIMS

    # ---- Sec. VII.E / Fig. 11: the exponent that was wrong before --------
    # Eq. (98): the relaxation-time singularity order is not a free number,
    # it is m = z * nu_c.  This is the check that catches the old 1.89.
    check("Eq. (98)  m_tau = z * nu_c", "approx", Z_C * NU_C, M_TAU_C, 1e-12)
    check("Fig. 11 caption  m_xi", "approx", C["m_xi"], NU_C, 5e-4)
    check("Fig. 11 caption  m_chi", "approx", C["m_chi"], GAMMA_C, 5e-4)
    check("Fig. 11 caption  m_tau_c", "approx", C["m_tau_c"], M_TAU_C, 5e-4)

    # ---- Sec. VI.A: sample sizes and threshold quoted in the text --------
    check("Eq. (37)  N (main)", "exact", C["N_main"], N_MAIN)
    check("Eq. (43)  N (resolution)", "exact", C["N_resolution"], N_RESOLUTION)
    check("Sec. VI.C  realizations", "exact", C["n_realizations"],
          N_REALIZATIONS)
    check("Sec. VI.A  Hill k", "exact", C["hill_k"], K_HILL)

    # ---- Fig. 6(a): the exponent law alpha = 1 + c/m ---------------------
    # Tolerance is not chosen by eye.  The Hill estimator has asymptotic
    # standard error alpha/sqrt(k), so a deviation counts as acceptable only
    # if it is within a few standard errors of that scale.  This makes the
    # residual bias at steep tails a quantified finite-threshold effect
    # (Eq. 14) rather than an unexplained discrepancy.
    rows = read_csv("exponent_validation.csv")
    ratios, abs_biases = [], []
    for r in rows:
        th = float(r["alpha_theory"])
        bias = float(r["alpha_hill_fixed_k"]) - th
        ratios.append(abs(bias) / (th / np.sqrt(K_HILL)))
        abs_biases.append(abs(bias))
    check("Fig. 6(a)  max |bias| / (alpha/sqrt(k))", "below",
          C["hill_bias_max_standard_errors"], max(ratios))
    check("Fig. 6(a)  max absolute bias", "info", "-", max(abs_biases),
          note="finite-threshold bias, Eq. (14)")

    # ---- Sec. VI.C: robustness to the global input distribution ----------
    rows = read_csv("robustness_summary.csv")
    means = np.array([float(r["mean_alpha"]) for r in rows])
    check("Sec. VI.C  every 95% range contains alpha = 2", "bool", "all six",
          all(r["contains_theory"] == "True" for r in rows))

    # ONE-SIDED.  The manuscript states an upper bound, so anything above it
    # must fail.  Writing this as "approx" with tol = claim would accept twice
    # the stated bound -- that was the bug in the first version of this file.
    max_rel = max(float(r["relative_deviation_percent"]) for r in rows)
    check("Sec. VI.C  max relative deviation (%)", "below",
          C["robustness_max_rel_deviation_pct"], max_rel,
          note="text states an upper bound ('below 0.3%')")

    check("Sec. VI.C  mean-estimate span, low", "approx", 2.0,
          float(means.min()), C["robustness_mean_tol"])
    check("Sec. VI.C  mean-estimate span, high", "approx", 2.0,
          float(means.max()), C["robustness_mean_tol"])

    # ---- Sec. VI.D: the local probability exponent nu, Eqs. (51)-(55) ----
    rows = read_csv("nu_validation.csv")
    check("Fig. 6(c)  max |bias| over 18 nu-classes", "below",
          C["nu_test_max_abs_bias"],
          max(abs(float(r["difference"])) for r in rows))

    # ---- Sec. VI.E: embedded manifolds, Eq. (61) -------------------------
    rows = read_csv("manifold_validation.csv")
    for name, key in (("Sphere", "alpha_sphere"), ("Circle", "alpha_circle"),
                      ("Point", "alpha_point")):
        vals = [float(r["alpha_hill"]) for r in rows if r["manifold"] == name]
        check(f"Eq. (61)  alpha ({name})", "approx", C[key],
              float(np.mean(vals)), C["manifold_alpha_tol"])

    # ---- Eq. (29)/(47): cutoff scaling, MEASURED then regressed ----------
    for r in read_csv("cutoff_scaling_fits.csv"):
        m = float(r["m"])
        check(f"Eq. (47)  cutoff slope (m={m})", "approx", -m,
              float(r["fitted_slope"]), C["cutoff_slope_tol"])

    # ---- Sec. VI.E item 5: competing models ------------------------------
    rows = read_csv("model_comparison.csv")
    bare_best = [r["model"] for r in rows
                 if r["case"].startswith("bare") and r["preferred"] == "True"]
    check("Model comparison: bare map prefers Pareto", "bool",
          "Pareto", bare_best == ["Pareto"])
    reg_best = {r["case"]: r["model"] for r in rows
                if r["case"].startswith("regularized")
                and r["preferred"] == "True"}
    check("Model comparison: regularized prefer trunc. Pareto", "bool",
          "Trunc Pareto",
          bool(reg_best) and all(v == "Truncated Pareto"
                                 for v in reg_best.values()))


# ==========================================================================
# Report
# ==========================================================================

def report():
    w = 52
    print("=" * (w + 38))
    print(f"{'check':<{w}}{'claim':>12}{'crit':>9}{'value':>12}   result")
    print("=" * (w + 38))
    for r in RESULTS:
        verdict = "INFO" if r["kind"] == "info" else (
            "PASS" if r["ok"] else "FAIL")
        print(f"{r['name']:<{w}}{_fmt(r['claim']):>12}"
              f"{r['criterion']:>9}{_fmt(r['value']):>12}   {verdict}")
    print("=" * (w + 38))

    failures = [r for r in RESULTS if not r["ok"] and r["kind"] != "info"]
    n_info = sum(1 for r in RESULTS if r["kind"] == "info")
    n_real = len(RESULTS) - n_info
    print(f"{n_real - len(failures)}/{n_real} checks passed"
          f"   ({n_info} informational)")

    if failures:
        print("\nFAILED CHECKS -- the manuscript and the code disagree:")
        for r in failures:
            print(f"  * {r['name']}")
            print(f"      manuscript states : {r['criterion']} "
                  f"{_fmt(r['claim'])}")
            print(f"      code reproduces   : {_fmt(r['value'])}")
            if r["note"]:
                print(f"      note              : {r['note']}")
        print("\nResolve each by changing the manuscript text or the code.")
        print("Do NOT widen a tolerance and do NOT turn a 'below' check into")
        print("an 'approx' check -- that is how the discrepancy hid before.")
    return len(failures)


def main():
    required = ("exponent_validation.csv", "robustness_summary.csv",
                "nu_validation.csv", "manifold_validation.csv",
                "cutoff_scaling_fits.csv", "model_comparison.csv")
    missing = [n for n in required if not (DATA / n).exists()]
    if missing:
        print("Missing tables: " + ", ".join(missing))
        print("Run scripts/generate_validation_tables.py first.")
        return 2

    run_checks()
    return 1 if report() else 0


if __name__ == "__main__":
    sys.exit(main())
