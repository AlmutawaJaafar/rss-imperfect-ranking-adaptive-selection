#!/usr/bin/env python3
# First-order contaminated-law predictions for RE under imperfect ranking.
#
# IMPORTANT:
# The manuscript uses fixed-count replacement contamination. These calculations
# therefore are NOT an exact finite-sample theorem for that simulation design.
# They use the i.i.d. contaminated-law approximation
#     F_eps = (1-eps) F + eps H
# and are intended as first-order theory--simulation diagnostics.

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(11)
m = 5
eps = 0.10
c = 1.345

# 90% quantile of the symmetric contaminated-law approximation used for
# the 10% trimmed-mean influence function.
q = norm.ppf((0.9 - 0.05) / 0.9)

# Monte Carlo sizes.
M = 4_000_000
N = 2_000_000

# Influence/estimating functions used in the first-order calculation.
psi = {
    "mean": lambda x: x,
    "trimmed": lambda x: np.clip(x, -q, q),
    "huber": lambda x: np.clip(x, -c, c),
}

# Clean order statistics.
Xc = np.sort(rng.standard_normal((M, m)), axis=1)

# Contaminated order statistics for the pre-ranking approximation.
Xe = rng.standard_normal((M, m))
cm = rng.random((M, m)) < eps
Xe[cm] = 8.0 * rng.choice([-1, 1], size=cm.sum())
Xe = np.sort(Xe, axis=1)

# One-dimensional contaminated law for the influence-function second moment.
Fe = rng.standard_normal(M)
ce = rng.random(M) < eps
Fe[ce] = 8.0 * rng.choice([-1, 1], size=ce.sum())

Ec = {k: f(Xc).mean(axis=0) for k, f in psi.items()}
Ee = {k: f(Xe).mean(axis=0) for k, f in psi.items()}
E2 = {k: (f(Fe) ** 2).mean() for k, f in psi.items()}

# Calibrated ranking-error scales used in the manuscript.
calibration = [
    ("gaussian", 0.25, 5.56036327034235),
    ("laplace",  0.25, 4.7666555270552635),
    ("gaussian", 0.50, 2.5067249312996864),
    ("laplace",  0.50, 2.071872539818287),
    ("gaussian", 0.75, 1.3395747169852257),
    ("laplace",  0.75, 1.0353794321417809),
    ("gaussian", 0.90, 0.795978493988514),
    ("laplace",  0.90, 0.579368881881237),
]

# Generate transition matrices P and the first-order predictions.
R = np.tile(np.arange(1, m + 1), (N, 1)).astype(float)
res = {}

for mech, rho_target, tau in calibration:
    if mech == "gaussian":
        Zr = rng.standard_normal((N, m))
    else:
        Zr = rng.laplace(0.0, 1.0, (N, m))

    observed = (R + tau * Zr).argsort(axis=1).argsort(axis=1) + 1

    # P[r,s] = Pr(observed rank = s | true rank = r).
    P = np.zeros((m, m))
    for r in range(m):
        P[r] = np.bincount(
            observed[:, r] - 1,
            minlength=m
        ) / N

    for estimator in psi:
        # Post-ranking contamination.
        post = E2[estimator] / (
            E2[estimator]
            - np.mean(((1.0 - eps) * (P.T @ Ec[estimator])) ** 2)
        )

        # Pre-ranking contamination.
        pre = E2[estimator] / (
            E2[estimator]
            - np.mean((P.T @ Ee[estimator]) ** 2)
        )

        res.setdefault((estimator, rho_target), []).append((post, pre))

print("First-order contaminated-law predictions averaged over")
print("Gaussian and Laplace ranking mechanisms (post / pre):")
print()

for rho_target in (0.25, 0.50, 0.75, 0.90):
    print(f"rho = {rho_target:.2f}")
    for estimator in psi:
        post, pre = np.mean(
            res[(estimator, rho_target)], axis=0
        )
        print(
            f"  {estimator:8s}: "
            f"post = {post:.3f}, pre = {pre:.3f}"
        )
    print()
