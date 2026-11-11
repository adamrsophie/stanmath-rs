"""High-precision values and gradients for every case in the Stan Math reference fixture.

Run from the repository root after regenerating stan_reference.csv:

    uv run --with mpmath python tools/reference/distributions.py

Reads crates/probgrad-distributions/tests/data/stan_reference.csv and writes mpmath_reference.csv
next to it, with the same columns. Values and gradients are computed at 320 significant digits, enough for
tails such as log Phi(30) = -4.9e-198; gradients are central differences with a 1e-80 step.
"""

import csv
import os
from mpmath import mp, mpf, diff, log, log1p, exp, ncdf, gammainc, betainc, loggamma, pi, atan, inf

mp.dps = 320

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "probgrad-distributions", "tests", "data")


def P(a, x):
    return gammainc(a, 0, x, regularized=True)


def Q(a, x):
    return gammainc(a, x, inf, regularized=True)


def I(x, a, b):
    return betainc(a, b, 0, x, regularized=True)


def inv_logit(x):
    return 1 / (1 + exp(-x))


def lchoose(n, k):
    return loggamma(n + 1) - loggamma(k + 1) - loggamma(n - k + 1)


def t_cdf(t, nu):
    h = I(nu / (nu + t * t), nu / 2, mpf(1) / 2) / 2
    return h if t < 0 else 1 - h


def normal_lpdf(y, m, s):
    z = (y - m) / s
    return -z * z / 2 - log(s) - log(2 * pi) / 2


def t_lpdf(y, nu, m, s):
    z = (y - m) / s
    return loggamma((nu + 1) / 2) - loggamma(nu / 2) - log(nu) / 2 - log(pi) / 2 - log(s) - (nu + 1) / 2 * log1p(z * z / nu)


def logistic_lpdf(y, m, s):
    z = (y - m) / s
    return -z - log(s) - 2 * log1p(exp(-z))


def nb_lpmf(n, mu, phi):
    return loggamma(n + phi) - loggamma(n + 1) - loggamma(phi) + n * (log(mu) - log(mu + phi)) + phi * (log(phi) - log(mu + phi))


def weibull_lpdf(y, a, s):
    return log(a) - log(s) + (a - 1) * (log(y) - log(s)) - (y / s) ** a


REAL = {
    "normal_lpdf": normal_lpdf,
    "normal_lcdf": lambda y, m, s: log(ncdf((y - m) / s)),
    "normal_lccdf": lambda y, m, s: log(ncdf((m - y) / s)),
    "std_normal_lpdf": lambda y: normal_lpdf(y, 0, 1),
    "std_normal_lcdf": lambda y: log(ncdf(y)),
    "std_normal_lccdf": lambda y: log(ncdf(-y)),
    "student_t_lpdf": t_lpdf,
    "student_t_lcdf": lambda y, nu, m, s: log(t_cdf((y - m) / s, nu)),
    "student_t_lccdf": lambda y, nu, m, s: log(t_cdf((m - y) / s, nu)),
    "cauchy_lpdf": lambda y, m, s: -log(pi) - log(s) - log1p(((y - m) / s) ** 2),
    "cauchy_lcdf": lambda y, m, s: log(mpf(1) / 2 + atan((y - m) / s) / pi),
    "cauchy_lccdf": lambda y, m, s: log(mpf(1) / 2 - atan((y - m) / s) / pi),
    "logistic_lpdf": logistic_lpdf,
    "logistic_lcdf": lambda y, m, s: log(inv_logit((y - m) / s)),
    "logistic_lccdf": lambda y, m, s: log(inv_logit((m - y) / s)),
    "exponential_lpdf": lambda y, b: log(b) - b * y,
    "exponential_lcdf": lambda y, b: log(1 - exp(-b * y)),
    "exponential_lccdf": lambda y, b: -b * y,
    "gamma_lpdf": lambda y, a, b: a * log(b) - loggamma(a) + (a - 1) * log(y) - b * y,
    "gamma_lcdf": lambda y, a, b: log(P(a, b * y)),
    "gamma_lccdf": lambda y, a, b: log(Q(a, b * y)),
    "inv_gamma_lpdf": lambda y, a, b: a * log(b) - loggamma(a) - (a + 1) * log(y) - b / y,
    "inv_gamma_lcdf": lambda y, a, b: log(Q(a, b / y)),
    "inv_gamma_lccdf": lambda y, a, b: log(P(a, b / y)),
    "beta_lpdf": lambda y, a, b: (a - 1) * log(y) + (b - 1) * log(1 - y) - (loggamma(a) + loggamma(b) - loggamma(a + b)),
    "beta_lcdf": lambda y, a, b: log(I(y, a, b)),
    "beta_lccdf": lambda y, a, b: log(I(1 - y, b, a)),
    "lognormal_lpdf": lambda y, m, s: normal_lpdf(log(y), m, s) - log(y),
    "lognormal_lcdf": lambda y, m, s: log(ncdf((log(y) - m) / s)),
    "lognormal_lccdf": lambda y, m, s: log(ncdf((m - log(y)) / s)),
    "uniform_lpdf": lambda y, a, b: -log(b - a),
    "uniform_lcdf": lambda y, a, b: log((y - a) / (b - a)),
    "uniform_lccdf": lambda y, a, b: log((b - y) / (b - a)),
    "weibull_lpdf": weibull_lpdf,
    "weibull_lcdf": lambda y, a, s: log(1 - exp(-((y / s) ** a))),
    "weibull_lccdf": lambda y, a, s: -((y / s) ** a),
}

def lower(v):
    m = mp.zeros(3, 3)
    m[0, 0], m[1, 0], m[1, 1], m[2, 0], m[2, 1], m[2, 2] = v
    return m


def symmetric(v):
    m = lower(v)
    for i in range(3):
        for j in range(i):
            m[j, i] = m[i, j]
    return m


def mvn_cholesky(*v):
    y, mu, L = mp.matrix(v[0:3]), mp.matrix(v[3:6]), lower(v[6:12])
    z = mp.lu_solve(L, y - mu)
    return -1.5 * log(2 * pi) - sum(log(L[i, i]) for i in range(3)) - (z.T * z)[0] / 2


def mvn(*v):
    y, mu, S = mp.matrix(v[0:3]), mp.matrix(v[3:6]), symmetric(v[6:12])
    d = y - mu
    return -1.5 * log(2 * pi) - log(mp.det(S)) / 2 - (d.T * mp.inverse(S) * d)[0] / 2


def lkj_cholesky(*v):
    L, eta, K = lower(v[0:6]), v[6], 3
    # Normalizing constant of Lewandowski, Kurowicka, and Joe (2009), general-eta form.
    c = (K - 1) * loggamma(eta + mpf(K - 1) / 2)
    for k in range(1, K):
        c -= k * log(pi) / 2 + loggamma(eta + mpf(K - 1 - k) / 2)
    return c + sum((K - i - 1 + 2 * eta - 2) * log(L[i, i]) for i in range(1, K))


def dirichlet(*v):
    theta, alpha = v[0:3], v[3:6]
    return loggamma(sum(alpha)) - sum(loggamma(a) for a in alpha) + sum((a - 1) * log(t) for t, a in zip(theta, alpha))


REAL.update({
    "multi_normal_cholesky_lpdf": mvn_cholesky,
    "multi_normal_lpdf": mvn,
    "lkj_corr_cholesky_lpdf": lkj_cholesky,
    "dirichlet_lpdf": dirichlet,
})

INT = {
    "bernoulli_lpmf": lambda k, t: log(t) if k[0] == 1 else log(1 - t),
    "bernoulli_lcdf": lambda k, t: log(1 - t) if k[0] == 0 else mpf(0),
    "bernoulli_lccdf": lambda k, t: log(t),
    "bernoulli_logit_lpmf": lambda k, a: log(inv_logit(a)) if k[0] == 1 else log(inv_logit(-a)),
    "binomial_lpmf": lambda k, t: lchoose(k[1], k[0]) + k[0] * log(t) + (k[1] - k[0]) * log(1 - t),
    "binomial_lcdf": lambda k, t: log(I(1 - t, k[1] - k[0], k[0] + 1)),
    "binomial_lccdf": lambda k, t: log(I(t, k[0] + 1, k[1] - k[0])),
    "binomial_logit_lpmf": lambda k, a: lchoose(k[1], k[0]) + k[0] * log(inv_logit(a)) + (k[1] - k[0]) * log(inv_logit(-a)),
    "poisson_lpmf": lambda k, l: k[0] * log(l) - l - loggamma(k[0] + 1),
    "poisson_lcdf": lambda k, l: log(Q(k[0] + 1, l)),
    "poisson_lccdf": lambda k, l: log(P(k[0] + 1, l)),
    "poisson_log_lpmf": lambda k, a: k[0] * a - exp(a) - loggamma(k[0] + 1),
    "neg_binomial_2_lpmf": lambda k, mu, phi: nb_lpmf(k[0], mu, phi),
    "neg_binomial_2_lcdf": lambda k, mu, phi: log(I(phi / (mu + phi), phi, k[0] + 1)),
    "neg_binomial_2_lccdf": lambda k, mu, phi: log(I(mu / (mu + phi), k[0] + 1, phi)),
    "neg_binomial_2_log_lpmf": lambda k, eta, phi: nb_lpmf(k[0], exp(eta), phi),
    "categorical_lpmf": lambda k, *theta: log(theta[k[0] - 1]),
    "categorical_logit_lpmf": lambda k, *beta: beta[k[0] - 1] - log(sum(exp(b) for b in beta)),
}


def parse(s):
    return [] if s == "" else s.split(";")


def fmt(v):
    return repr(float(v))


def main():
    rows = []
    with open(os.path.join(DATA, "stan_reference.csv")) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split(",")
            name, ints, reals, value = f[0], [int(v) for v in parse(f[1])], [mpf(v) for v in parse(f[2])], f[3]
            if value == "error":
                rows.append([name, f[1], f[2], "error", ""])
                continue
            if name in REAL:
                g = REAL[name]
            else:
                base = INT[name]
                g = lambda *x, base=base, ints=ints: base(ints, *x)
            v = g(*reals)
            h = mpf(10) ** -80
            grad = [diff(g, tuple(reals), tuple(1 if j == i else 0 for j in range(len(reals))), h=h) for i in range(len(reals))]
            rows.append([name, f[1], f[2], fmt(v), ";".join(fmt(x) for x in grad)])
    with open(os.path.join(DATA, "mpmath_reference.csv"), "w", newline="") as fh:
        fh.write("# Generated by tools/reference/distributions.py (mpmath, 320 digits) for the cases in stan_reference.csv\n")
        csv.writer(fh, lineterminator="\n").writerows(rows)


if __name__ == "__main__":
    main()
