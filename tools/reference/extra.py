"""High-precision values and gradients for the rows of stan_reference_extra.csv where Stan Math
and stanmath-rs disagree: the shape derivatives of the chi-square family CDFs, the beta negative
binomial CDFs, one exp_mod_normal CDF in the tail, the Poisson binomial CCDF at its upper
bound, the ordered probit gradients, the shape derivatives of the probability scale CDFs built on
the incomplete gamma and beta functions, and the Wiener first passage CDFs.

Run from the repository root:

    uv run --with mpmath python tools/reference/extra.py

Writes crates/stanmath-rs-distributions/tests/data/mpmath_reference_extra.csv in the fixture's
format.
"""

import csv
import os
from mpmath import mp, mpf, diff, log, exp, erfc, sqrt, gammainc, loggamma, beta, betainc, inf, ncdf, pi, quad

mp.dps = 40
DATA = "crates/stanmath-rs-distributions/tests/data"


def P(a, x):
    return gammainc(a, 0, x, regularized=True)


def Q(a, x):
    return gammainc(a, x, inf, regularized=True)


def Phi(x):
    return erfc(-x / sqrt(2)) / 2


def emn_cdf(y, mu, s, l):
    z = (y - mu) / s
    return Phi(z) - exp(-l * (y - mu) + l * l * s * s / 2) * Phi(z - l * s)


def bnb_pmf(k, r, a, b):
    return exp(loggamma(k + r) - loggamma(k + 1) - loggamma(r)) * beta(k + b, r + a) / beta(a, b)


def bnb_cdf(n, r, a, b):
    return sum(bnb_pmf(k, r, a, b) for k in range(n + 1))


def pb_lccdf(n, p):
    # Probability of more than n successes among len(p) independent trials.
    dist = [mpf(1)]
    for q in p:
        nxt = [mpf(0)] * (len(dist) + 1)
        for k, v in enumerate(dist):
            nxt[k] += v * (1 - q)
            nxt[k + 1] += v * q
        dist = nxt
    tail = sum(dist[n + 1:])
    return log(tail) if tail > 0 else -inf


def ordered_probit(k, eta, c):
    if k == 1:
        return log(1 - ncdf(eta - c[0]))
    if k == len(c) + 1:
        return log(ncdf(eta - c[-1]))
    return log(ncdf(eta - c[k - 2]) - ncdf(eta - c[k - 1]))


def I(a, b, x):
    return betainc(a, b, 0, x, regularized=True)


def wiener_upper(y, a, t0, w, v):
    """The defective CDF of first passage through the upper boundary by time y, and the
    probability of ever passing through it (Stan's wiener_lcdf_unnorm and wiener_lccdf_unnorm are
    the logs of the CDF and of the difference). The upper boundary is the lower one with w and v
    reflected; the density is the small time series, which is exact at every time."""
    w, v = 1 - w, -v

    def dens(t):
        u = t / a**2
        s = sum((w + 2 * k) * exp(-(w + 2 * k) ** 2 / (2 * u)) for k in range(-30, 31))
        return exp(-v * a * w - v * v * t / 2) / a**2 * u ** mpf(-1.5) / sqrt(2 * pi) * s

    cdf = quad(dens, [0, y - t0])
    hit = (exp(-2 * v * a * w) - exp(-2 * v * a)) / (1 - exp(-2 * v * a))
    return cdf, hit


def nb_cdf(n, alpha, beta_):
    return I(alpha, n + 1, beta_ / (1 + beta_))


F = {
    "gamma_cdf": lambda n, x: P(x[1], x[2] * x[0]),
    "inv_gamma_cdf": lambda n, x: Q(x[1], x[2] / x[0]),
    "beta_cdf": lambda n, x: I(x[1], x[2], x[0]),
    "scaled_inv_chi_square_cdf": lambda n, x: Q(x[1] / 2, x[1] * x[2] ** 2 / (2 * x[0])),
    "neg_binomial_cdf": lambda n, x: nb_cdf(n[0], x[0], x[1]),
    "neg_binomial_2_cdf": lambda n, x: nb_cdf(n[0], x[1], x[1] / x[0]),
    "beta_neg_binomial_cdf": lambda n, x: bnb_cdf(n[0], *x),
    "wiener_lcdf_unnorm": lambda n, x: log(wiener_upper(*x)[0]),
    "wiener_lccdf_unnorm": lambda n, x: (lambda c, h: log(h - c))(*wiener_upper(*x)),
    "chi_square_lcdf": lambda n, x: log(P(x[1] / 2, x[0] / 2)),
    "chi_square_lccdf": lambda n, x: log(Q(x[1] / 2, x[0] / 2)),
    "inv_chi_square_lcdf": lambda n, x: log(Q(x[1] / 2, 1 / (2 * x[0]))),
    "inv_chi_square_lccdf": lambda n, x: log(P(x[1] / 2, 1 / (2 * x[0]))),
    "scaled_inv_chi_square_lcdf": lambda n, x: log(Q(x[1] / 2, x[1] * x[2] ** 2 / (2 * x[0]))),
    "scaled_inv_chi_square_lccdf": lambda n, x: log(P(x[1] / 2, x[1] * x[2] ** 2 / (2 * x[0]))),
    "exp_mod_normal_lcdf": lambda n, x: log(emn_cdf(*x)),
    "beta_neg_binomial_lcdf": lambda n, x: log(bnb_cdf(n[0], *x)),
    "beta_neg_binomial_lccdf": lambda n, x: log(1 - bnb_cdf(n[0], *x)),
    "poisson_binomial_lccdf": lambda n, x: pb_lccdf(n[0], x),
    "ordered_probit_lpmf": lambda n, x: ordered_probit(n[0], x[0], x[1:]),
}


def fmt(v):
    return "-inf" if v == -inf else mp.nstr(v, 20, strip_zeros=False)


def main():
    rows = []
    with open(os.path.join(DATA, "stan_reference_extra.csv")) as f:
        for line in f:
            if line.startswith("#"):
                continue
            name, ints, reals = line.strip().split(",")[:3]
            if name not in F:
                continue
            n = [int(v) for v in ints.split(";")] if ints else []
            x = [mpf(v) for v in reals.split(";")]
            g = F[name]
            value = g(n, x)
            if value == -inf:
                # The gradient of a log probability of zero: every partial is zero.
                grad = [mpf(0)] * len(x)
            else:
                grad = [
                    diff(lambda t, k=k: g(n, x[:k] + [t] + x[k + 1:]), x[k]) for k in range(len(x))
                ]
            rows.append([name, ints, reals, fmt(value), ";".join(fmt(v) for v in grad)])
    with open(os.path.join(DATA, "mpmath_reference_extra.csv"), "w", newline="") as f:
        f.write("# Generated by tools/reference/extra.py with mpmath at 40 digits\n")
        csv.writer(f, lineterminator="\n").writerows(rows)


main()
