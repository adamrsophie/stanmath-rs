"""Writes crates/stanmath-rs/tests/data/c99_reference_complex.csv: for each row of
stan_reference_complex.csv, the value under C99 Annex G (Python's cmath, which keeps the sign of
zero and so the side of each branch cut) and the gradient on that side of the cut (mpmath, 40
digits, differentiating along the real axis, which stays on the side the signed zero picks).

    uv run -q --with mpmath python -I tools/reference/complex.py

Where Stan and stanmath-rs disagree, stanmath-rs must match this file.
"""

import cmath
import math

import mpmath as mp

mp.mp.dps = 40
DATA = "crates/stanmath-rs/tests/data"

HOLO = {
    "exp": (cmath.exp, mp.exp), "log": (cmath.log, mp.log), "log10": (cmath.log10, lambda z: mp.log(z, 10)),
    "sqrt": (cmath.sqrt, mp.sqrt), "sin": (cmath.sin, mp.sin), "cos": (cmath.cos, mp.cos),
    "tan": (cmath.tan, mp.tan), "sinh": (cmath.sinh, mp.sinh), "cosh": (cmath.cosh, mp.cosh),
    "tanh": (cmath.tanh, mp.tanh), "asin": (cmath.asin, mp.asin), "acos": (cmath.acos, mp.acos),
    "atan": (cmath.atan, mp.atan), "asinh": (cmath.asinh, mp.asinh), "acosh": (cmath.acosh, mp.acosh),
    "atanh": (cmath.atanh, mp.atanh), "minus": (lambda z: -z, lambda z: -z),
}


def side(x, y):
    """The point mpmath evaluates for (x, y): on a cut, nudged to the side the zero's sign picks."""
    if y == 0:
        y = math.copysign(mp.mpf("1e-35"), y)
    return mp.mpc(x, y)


def deriv(f, z):
    h = mp.mpf("1e-18")
    return (f(z + h) - f(z - h)) / (2 * h)


def fmt(v):
    return repr(float(v))


def row(name, inp):
    if name in HOLO:
        c, m = HOLO[name]
        w = c(complex(inp[0], inp[1]))
        d = deriv(m, side(*inp))
        # Cauchy-Riemann: d Re/dx = Re f', d Re/dy = -Im f', d Im/dx = Im f', d Im/dy = Re f'.
        return [w.real, w.imag], [d.real, -d.imag, d.imag, d.real]
    x, y = inp[:2]
    if name == "conj":
        return [x, -y], [1, 0, 0, -1]
    if name == "proj":
        w = complex(x, y) if math.isfinite(x) and math.isfinite(y) else complex(math.inf, math.copysign(0.0, y))
        return [w.real, w.imag], [1, 0, 0, 1]
    r = math.hypot(x, y)
    if name == "abs":
        return [r], [x / r, y / r]
    if name == "arg":
        return [math.atan2(y, x)], [-y / r ** 2, x / r ** 2]
    if name == "norm":
        return [x * x + y * y], [2 * x, 2 * y]
    z, w = complex(x, y), complex(inp[2], inp[3])
    zm, wm = side(x, y), side(inp[2], inp[3])
    if name == "pow":
        v = cmath.exp(w * cmath.log(z))
        fz, fw = (lambda a: mp.exp(wm * mp.log(a))), (lambda b: mp.exp(b * mp.log(zm)))
    elif name == "divide":
        v = z / w
        fz, fw = (lambda a: a / wm), (lambda b: zm / b)
    else:
        v = z * w
        fz, fw = (lambda a: a * wm), (lambda b: zm * b)
    dz, dw = deriv(fz, zm), deriv(fw, wm)
    grad = [dz.real, -dz.imag, dw.real, -dw.imag, dz.imag, dz.real, dw.imag, dw.real]
    return [v.real, v.imag], grad


def main():
    out = ["# name,inputs,outputs,gradient: C99 values (cmath) and one-sided gradients (mpmath)"]
    for line in open(f"{DATA}/stan_reference_complex.csv"):
        if line.startswith("#"):
            continue
        name, inputs = line.split(",")[:2]
        inp = [float(v) for v in inputs.split(";")]
        vals, grad = row(name, inp)
        out.append(f"{name},{inputs},{';'.join(fmt(v) for v in vals)},{';'.join(fmt(g) for g in grad)}")
    open(f"{DATA}/c99_reference_complex.csv", "w").write("\n".join(out) + "\n")


main()
