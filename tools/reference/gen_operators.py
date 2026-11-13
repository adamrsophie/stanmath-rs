"""Writes crates/stanmath-rs/tests/signatures_operators.rs: every container signature of the
elementwise Stan operators add, subtract, elt_multiply, and elt_divide.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_operators.py sigs.txt && cargo fmt

Two real matrices go through the library functions (add, subtract, elt_multiply, elt_divide);
every other form (vectors, complex containers, a scalar with a container) is the Rust operator
applied through apply_scalar_unary or apply_scalar_binary. Each result must equal the operator
applied to the elements in order.
"""

import re
import sys

import ledger

OPS = {"add": "+", "subtract": "-", "elt_multiply": "*", "elt_divide": "/"}


def rust(t):
    lt = "Complex<f64>" if t.startswith("complex") else "f64"
    core = t.replace("complex_", "").replace("complex", "real")
    return {"real": lt, "vector": f"Vec<{lt}>", "row_vector": f"Vec<{lt}>",
            "matrix": f"Matrix<{lt}>"}[core]


def value(ty, start):
    lt = "Complex<f64>" if "Complex" in ty else "f64"
    one = (lambda v: f"Complex::new({v / 8!r}, {-v / 16!r})") if lt != "f64" else (lambda v: f"{v / 8!r}")
    if ty == lt:
        return one(start + 1)
    if ty.startswith("Matrix"):
        return f"Matrix::from_column_major(2, 3, vec![{', '.join(one(start + i) for i in range(1, 7))}])"
    return f"vec![{', '.join(one(start + i) for i in range(1, 4))}]"


def main():
    body, total = [], 0
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"(add|subtract|elt_multiply|elt_divide)\((\w+), (\w+)\) => (.*)", l)
        if not m:
            continue
        name, a, b = m.group(1), m.group(2), m.group(3)
        if a in ("int", "real", "complex") and b in ("int", "real", "complex"):
            continue
        op = OPS[name]
        ta, tb = rust(a), rust(b)
        lets = f"        let x: {ta} = {value(ta, 0)};\n        let y: {tb} = {value(tb, 10)};\n"
        ca, cb = ta.startswith(("Vec", "Matrix")), tb.startswith(("Vec", "Matrix"))
        if ca and cb and ta == "Matrix<f64>":
            got = f"{name}(&x, &y).unwrap()"
        elif ca and cb:
            got = f"apply_scalar_binary(&x, &y, |a, b| a {op} b).unwrap()"
        elif ca:
            got = f"apply_scalar_unary(&x, |a| a {op} y)"
        else:
            got = f"apply_scalar_unary(&y, |b| x {op} b)"
        xs = "flat(&x)" if ca else "vec![x; 6]"
        ys = "flat(&y)" if cb else "vec![y; 6]"
        label = f"{name}({a}, {b})"
        body.append(f"    {{\n{lets}        let got = flat(&{got});\n"
                    f"        let want: Vec<_> = {xs}.into_iter().zip({ys}).map(|(a, b)| a {op} b).collect();\n"
                    f"        assert_eq!(got, want[..got.len()], \"{label}\");\n    }}\n")
        total += 1
        ledger.record("called", name, [a, b])
    out = HEADER.format(total=total) + "#[test]\nfn operator_signatures() {\n" + "".join(body) + "}\n"
    open("crates/stanmath-rs/tests/signatures_operators.rs", "w").write(out)
    print(total, "signatures")


HEADER = """//! Every container signature of the elementwise Stan operators `add`, `subtract`,
//! `elt_multiply`, and `elt_divide`.
//!
//! Written by `tools/reference/gen_operators.py` from the CmdStan 2.40.0 signature dump: {total}
//! Stan signatures. Two real matrices go through the library functions; the other forms are the
//! Rust operator through `apply_scalar_unary` or `apply_scalar_binary`.
#![allow(clippy::too_many_lines, clippy::unreadable_literal, clippy::float_cmp)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

fn flat<C: ScalarContainer>(c: &C) -> Vec<C::Scalar> {{
    to_array_1d_nested(c)
}}

"""

main()
