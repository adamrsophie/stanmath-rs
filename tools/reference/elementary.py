"""High-precision reference values for stanmath-rs's elementary functions.

Run from the repository root:

    uv run --with mpmath python tools/reference/elementary.py

Writes crates/stanmath-rs-autodiff/tests/data/unary_reference.csv and binary_reference.csv with the
value, first derivatives, and second derivatives computed by mpmath at 50 significant digits and
rounded once to the nearest double.
"""

import csv
import os
from mpmath import mp, mpf, diff

mp.dps = 60

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
OUT = os.path.join(ROOT, "crates", "stanmath-rs-autodiff", "tests", "data")


def inv_logit(x):
    return 1 / (1 + mp.exp(-x))


UNARY = {
    "exp": (mp.exp, [-30, -1.5, 0.0, 0.7, 25]),
    "exp2": (lambda x: mp.power(2, x), [-20, 0.3, 10]),
    "expm1": (mp.expm1, [-1e-10, 1e-12, -0.3, 5]),
    "log": (mp.log, [1e-300, 1e-5, 0.5, 3, 1e10]),
    "log1p": (mp.log1p, [1e-12, -0.5, 10, -0.999999]),
    "log1m": (lambda x: mp.log1p(-x), [1e-12, 0.5, -10, 0.999999]),
    "log2": (lambda x: mp.log(x, 2), [0.1, 3, 1e6]),
    "log10": (mp.log10, [0.1, 3, 1e6]),
    "sqrt": (mp.sqrt, [1e-10, 0.25, 2, 1e8]),
    "cbrt": (lambda x: mp.sign(x) * mp.cbrt(abs(x)), [-8, 1e-9, 0.5, 27]),
    "square": (lambda x: x * x, [-3, 0.5, 1e5]),
    "inv": (lambda x: 1 / x, [-3, 0.5, 1e-5]),
    "inv_sqrt": (lambda x: 1 / mp.sqrt(x), [0.01, 2, 1e6]),
    "inv_square": (lambda x: 1 / (x * x), [-3, 0.5, 1e3]),
    "abs": (abs, [-2.5, 3.5]),
    "sin": (mp.sin, [-2, 1e-8, 0.5, 100]),
    "cos": (mp.cos, [-2, 1e-8, 0.5, 100]),
    "tan": (mp.tan, [-1, 0.5, 1.5]),
    "asin": (mp.asin, [-0.3, 1e-8, 0.5, 0.999]),
    "acos": (mp.acos, [-0.3, 1e-8, 0.5, 0.999]),
    "atan": (mp.atan, [-5, 0.5, 1e5]),
    "sinh": (mp.sinh, [-3, 1e-8, 0.5, 20]),
    "cosh": (mp.cosh, [-3, 1e-8, 0.5, 20]),
    "tanh": (mp.tanh, [-3, 1e-8, 0.5, 10]),
    "asinh": (mp.asinh, [-3, 1e-8, 0.5, 1e10]),
    "acosh": (mp.acosh, [1.0001, 3, 1e10]),
    "atanh": (mp.atanh, [-0.5, 1e-8, 0.999]),
    "log1p_exp": (lambda x: mp.log1p(mp.exp(x)), [-40, -1, 0.0, 0.5, 30, 700]),
    "log1m_exp": (lambda x: mp.log(-mp.expm1(x)), [-1e-10, -0.5, -0.6931, -5, -50]),
    "inv_logit": (inv_logit, [-40, -1, 0.0, 3, 40]),
    "logit": (lambda p: mp.log(p / (1 - p)), [1e-10, 0.3, 0.999]),
    "log_inv_logit": (lambda x: mp.log(inv_logit(x)), [-40, -1, 0.0, 3, 40]),
    "log1m_inv_logit": (lambda x: mp.log(1 - inv_logit(x)), [-40, -1, 0.0, 3, 40]),
}

BINARY = {
    "pow": (mp.power, [(2.0, 3.0), (0.5, -1.5), (3.0, 0.5), (1.5, 1e-8)]),
    "atan2": (mp.atan2, [(1.0, 2.0), (-1.0, -0.5), (1e-5, 3.0)]),
    "hypot": (lambda a, b: mp.sqrt(a * a + b * b), [(3.0, 4.0), (-1e-3, 2.0), (1e5, 1e-3), (1e-3, 1e5)]),
    "log_sum_exp": (lambda a, b: mp.log(mp.exp(a) + mp.exp(b)), [(1.0, 2.0), (-50.0, 3.0), (700.0, 699.0)]),
    "log_diff_exp": (lambda a, b: mp.log(mp.exp(a) - mp.exp(b)), [(2.0, 1.0), (0.0, -30.0), (5.0, 4.999)]),
}


def fmt(v):
    return repr(float(v))


def step(*xs):
    # Central differences at 60 digits with a step relative to the argument keep both truncation
    # and rounding error far below double precision, and the step stays inside the domain.
    m = min(abs(x) for x in xs if x != 0) if any(x != 0 for x in xs) else mpf(1)
    return min(m, mpf(1)) * mpf(10) ** -15


def unary_rows():
    for name, (f, xs) in UNARY.items():
        for x in xs:
            x = mpf(x)
            h = step(x)
            yield [name, fmt(x), fmt(f(x)), fmt(diff(f, x, 1, h=h)), fmt(diff(f, x, 2, h=h))]


def binary_rows():
    for name, (f, pts) in BINARY.items():
        for a, b in pts:
            a, b = mpf(a), mpf(b)
            row = [name, fmt(a), fmt(b), fmt(f(a, b))]
            h = step(a, b)
            row += [fmt(diff(f, (a, b), (1, 0), h=h)), fmt(diff(f, (a, b), (0, 1), h=h))]
            row += [
                fmt(diff(f, (a, b), (2, 0), h=h)),
                fmt(diff(f, (a, b), (1, 1), h=h)),
                fmt(diff(f, (a, b), (0, 2), h=h)),
            ]
            yield row


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "unary_reference.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "x", "value", "d1", "d2"])
        w.writerows(unary_rows())
    with open(os.path.join(OUT, "binary_reference.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function", "a", "b", "value", "da", "db", "daa", "dab", "dbb"])
        w.writerows(binary_rows())


if __name__ == "__main__":
    main()
