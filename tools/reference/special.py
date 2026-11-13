"""High-precision reference values for stanmath-rs's special functions.

Run from the repository root:

    uv run --with mpmath python tools/reference/special.py

Writes special_unary_reference.csv, special_binary_reference.csv, and beta_inc_reference.csv into
crates/stanmath-rs-autodiff/tests/data. Derivatives are central differences evaluated at 60 digits
with a step relative to the argument, which keeps their error far below double precision.
"""

import csv
import os
from mpmath import mp, mpf, diff

mp.dps = 60

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "crates", "stanmath-rs-autodiff", "tests", "data")


def phi_cdf(x):
    return mp.ncdf(x)


def inv_phi(p):
    # 2p - 1 needs about -log10(p) extra digits near p = 0.
    with mp.workdps(mp.dps + 350):
        return mp.sqrt(2) * mp.erfinv(2 * p - 1)


def stirling_diff(x):
    return mp.loggamma(x) - ((x - mpf(1) / 2) * mp.log(x) - x + mp.log(2 * mp.pi) / 2)


UNARY = {
    "lgamma": (mp.loggamma, [1e-8, 0.5, 1.4616, 3.7, 10.5, 100, 1e5, 2.5e15]),
    "tgamma": (mp.gamma, [0.1, 2.5, 7.3, 50, -1.5]),
    "digamma": (mp.digamma, [1e-6, 0.3, 3.5, 9.99, 10.01, 50, 1e6, -0.5, -3.7]),
    "trigamma": (lambda x: mp.polygamma(1, x), [1e-6, 0.3, 2.5, 9.9, 20, 1e4, -0.5, -3.7]),
    "polygamma2": (lambda x: mp.polygamma(2, x), [0.2, 1.5, 7, 30]),
    "polygamma3": (lambda x: mp.polygamma(3, x), [0.2, 1.5, 7, 30]),
    "erf": (mp.erf, [-3, -0.5, 1e-10, 0.3, 2, 5.9]),
    "erfc": (mp.erfc, [-2, 0.5, 3, 10, 26.5]),
    "erfcx": (lambda x: mp.exp(x * x) * mp.erfc(x), [-5, -0.5, 0.1, 1, 2.5]),
    "Phi": (phi_cdf, [-38, -10, -1, 0.3, 4, 8]),
    "log_Phi": (lambda x: mp.log(phi_cdf(x)), [-1e3, -40, -20.5, -19.5, -5, 0.0, 2, 4.9, 5.5, 10]),
    "inv_Phi": (inv_phi, [1e-300, 1e-100, 1e-10, 0.01, 0.3, 0.6, 0.99]),
    "log1pmx": (lambda u: mp.log1p(u) - u, [-0.9, -0.5, -1e-5, 1e-8, 0.3, 0.6, 10]),
    "lgamma_stirling_diff": (stirling_diff, [0.5, 3, 9.9, 10.1, 100, 1e8]),
}

# Points where the value is large enough that second derivatives by differences are meaningful;
# erfcx and inv_Phi at the extreme points are checked for value only.
VALUE_ONLY = {
    "erfcx": [25.9, 26.1, 100, 1e5],
    "inv_Phi": [1 - mpf(2) ** -40],
}


def gamma_p(a, x):
    return mp.gammainc(a, 0, x, regularized=True)


def gamma_q(a, x):
    return mp.gammainc(a, x, mp.inf, regularized=True)


def lbeta(a, b):
    return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(a + b)


GAMMA_POINTS = [
    (0.5, 0.1), (0.5, 3.0), (2.0, 0.5), (2.0, 5.0), (10.0, 3.0), (10.0, 15.0),
    (100.0, 90.0), (100.0, 110.0), (1e-3, 0.5), (30.0, 1e-3), (5.0, 60.0),
]

BINARY = {
    "lbeta": (lbeta, [(0.5, 0.5), (1e-8, 3.0), (2.0, 3.0), (0.3, 15.0), (12.0, 15.0),
                      (5.0, 1e6), (1e6, 1e7), (100.0, 100.0)]),
    "gamma_p": (gamma_p, GAMMA_POINTS),
    "gamma_q": (gamma_q, GAMMA_POINTS),
}

BETA_POINTS = [
    (0.5, 0.5, 0.3), (2.0, 3.0, 0.4), (2.0, 3.0, 0.9), (10.0, 20.0, 0.3), (0.1, 5.0, 0.01),
    (50.0, 40.0, 0.55), (1.0, 1.0, 0.5), (3.5, 0.7, 0.999), (0.8, 30.0, 0.05),
]


def betainc(a, b, x):
    return mp.betainc(a, b, 0, x, regularized=True)


def fmt(v):
    return repr(float(v))


def step(*xs):
    m = min((abs(x) for x in xs if x != 0), default=mpf(1))
    return min(m, mpf(1)) * mpf(10) ** -18


def unary_rows():
    for name, (f, xs) in UNARY.items():
        for x in xs:
            x = mpf(x)
            h = step(x)
            yield [name, fmt(x), fmt(f(x)), fmt(diff(f, x, 1, h=h)), fmt(diff(f, x, 2, h=h))]
        for x in VALUE_ONLY.get(name, []):
            x = mpf(x)
            yield [name, fmt(x), fmt(f(x)), "nan", "nan"]


def binary_rows():
    for name, (f, pts) in BINARY.items():
        for a, b in pts:
            # Q(a, x) can sit within 1e-120 of one; give the differences enough digits.
            with mp.workdps(260):
                yield binary_row(name, f, mpf(a), mpf(b))


def binary_row(name, f, a, b):
    h = step(a, b)
    row = [name, fmt(a), fmt(b), fmt(f(a, b))]
    row += [fmt(diff(f, (a, b), (1, 0), h=h)), fmt(diff(f, (a, b), (0, 1), h=h))]
    row += [
        fmt(diff(f, (a, b), (2, 0), h=h)),
        fmt(diff(f, (a, b), (1, 1), h=h)),
        fmt(diff(f, (a, b), (0, 2), h=h)),
    ]
    return row


def beta_rows():
    for a, b, x in BETA_POINTS:
        a, b, x = mpf(a), mpf(b), mpf(x)
        h = step(a, b, x)
        row = [fmt(a), fmt(b), fmt(x), fmt(betainc(a, b, x))]
        for order in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            row.append(fmt(diff(betainc, (a, b, x), order, h=h)))
        yield row


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "special_unary_reference.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "x", "value", "d1", "d2"])
        w.writerows(unary_rows())
    with open(os.path.join(OUT, "special_binary_reference.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "a", "b", "value", "da", "db", "daa", "dab", "dbb"])
        w.writerows(binary_rows())
    with open(os.path.join(OUT, "beta_inc_reference.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "x", "value", "da", "db", "dx"])
        w.writerows(beta_rows())


if __name__ == "__main__":
    main()
