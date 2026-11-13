"""Writes crates/stanmath-rs/tests/signatures_complex.rs: every signature of to_complex, get_real,
get_imag, and conj in the Stan language, through apply_scalar_unary and apply_scalar_binary.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_complex.py sigs.txt && cargo fmt

Stan types map to Rust as in gen_elementwise.py, with complex leaves Complex<f64>; calls stop at
three Vec levels (deeper arrays share the same recursive impl). Each result is checked against the
definition: to_complex(a, b) is a + bi, get_real and get_imag return the parts, and conj negates
the imaginary part.
"""

import collections
import re
import sys

import ledger

MAX_DEPTH = 3


def rust(t):
    m = re.fullmatch(r"array\[(,*)\] (.*)", t)
    dims = len(m.group(1)) + 1 if m else 0
    base = m.group(2) if m else t
    ty = {"real": "f64", "int": "i64", "complex": "Complex<f64>", "vector": "Vec<f64>",
          "row_vector": "Vec<f64>", "matrix": "Matrix<f64>", "complex_vector": "Vec<Complex<f64>>",
          "complex_row_vector": "Vec<Complex<f64>>", "complex_matrix": "Matrix<Complex<f64>>"}.get(base)
    if ty is None:
        return None
    for _ in range(dims):
        ty = f"Vec<{ty}>"
    return ty


def leaf(ty):
    return "Complex<f64>" if "Complex" in ty else "i64" if "i64" in ty else "f64"


def value(ty, start):
    k = [start]
    lt = leaf(ty)

    def scalar():
        k[0] += 1
        v = k[0]
        return {"f64": f"{v}.25", "i64": f"{v}_i64", "Complex<f64>": f"Complex::new({v}.25, -{v}.5)"}[lt]

    def build(t):
        if t == lt:
            return scalar()
        if t.startswith("Matrix<"):
            return f"Matrix::from_column_major(2, 2, vec![{', '.join(scalar() for _ in range(4))}])"
        return f"vec![{build(t[4:-1])}, {build(t[4:-1])}]"

    return build(ty)


def main():
    sigs = collections.defaultdict(list)
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"(to_complex|get_real|get_imag|conj)\((.*)\) => (.*)", l)
        if m:
            sigs[m.group(1)].append(m.group(2).split(", ") if m.group(2) else [])
    body, total, deep = [], 0, 0
    for name in sorted(sigs):
        for args in sigs[name]:
            ra = [rust(t) for t in args]
            if any(r.count("Vec<") > MAX_DEPTH for r in ra):
                deep += 1
                ledger.record("deep", name, args)
                continue
            total += 1
            ledger.record("called", name, args)
            body.append(call(name, args, ra))
    out = HEADER.format(total=total, deep=deep) + "#[test]\nfn complex_signatures() {\n" + "".join(body) + "}\n"
    open("crates/stanmath-rs/tests/signatures_complex.rs", "w").write(out)
    print(total, "signatures;", deep, "deeper")


def as_f64(expr, ty):
    return f"({expr} as f64)" if leaf(ty) == "i64" else expr


def call(name, args, ra):
    label = f"{name}({', '.join(args)})"
    if name == "to_complex":
        if not ra:
            return f"    assert_eq!(to_complex(0.0, 0.0), Complex::new(0.0, 0.0), \"{label}\");\n"
        if len(ra) == 1:
            return (f"    {{\n        let a: {ra[0]} = {value(ra[0], 0)};\n"
                    f"        assert_eq!(to_complex({as_f64('a', ra[0])}, 0.0), Complex::new({as_f64('a', ra[0])}, 0.0), \"{label}\");\n    }}\n")
        a, b = ra
        lines = f"    {{\n        let a: {a} = {value(a, 0)};\n        let b: {b} = {value(b, 10)};\n"
        ca, cb = a not in ("f64", "i64"), b not in ("f64", "i64")
        fa, fb = as_f64("x", a), as_f64("y", b)
        if ca and cb and a == b:
            lines += f"        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex({fa}, {fb})).unwrap());\n"
        elif ca and cb:
            lines += f"        let bl = leaves(&b);\n        let mut i = 0;\n"
            lines += f"        let got = leaves(&apply_scalar_unary(&a, |x| {{ let y = bl[i]; i += 1; to_complex({fa}, {fb}) }}));\n"
        elif ca:
            lines += f"        let y = b;\n        let got = leaves(&apply_scalar_unary(&a, |x| to_complex({fa}, {fb})));\n"
        elif cb:
            lines += f"        let x = a;\n        let got = leaves(&apply_scalar_unary(&b, |y| to_complex({fa}, {fb})));\n"
        else:
            lines += f"        let (x, y) = (a, b);\n        let got = vec![to_complex({fa}, {fb})];\n"
        al = "leaves(&a)" if ca else "vec![a; got.len()]"
        bl = "leaves(&b)" if cb else "vec![b; got.len()]"
        lines += (f"        let want: Vec<Complex<f64>> = {al}.into_iter().zip({bl})"
                  f".map(|(x, y)| Complex::new({fa}, {fb})).collect();\n"
                  f"        assert_eq!(got, want, \"{label}\");\n    }}\n")
        return lines
    (x,) = ra
    spec = {"get_real": "z.re", "get_imag": "z.im", "conj": "Complex::new(z.re, -z.im)"}[name]
    return (f"    {{\n        let x: {x} = {value(x, 0)};\n"
            f"        let got = leaves(&apply_scalar_unary(&x, {name}));\n"
            f"        let want: Vec<_> = leaves(&x).into_iter().map(|z| {spec}).collect();\n"
            f"        assert_eq!(got, want, \"{label}\");\n    }}\n")


HEADER = """//! Every signature of `to_complex`, `get_real`, `get_imag`, and `conj` in the Stan language,
//! through `apply_scalar_unary` and `apply_scalar_binary`.
//!
//! Written by `tools/reference/gen_complex.py` from the CmdStan 2.40.0 signature dump: {total} Stan
//! signatures, each checked against the definition. The other {deep} signatures nest arrays more
//! than three levels deep and share the same recursive impl.
#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::redundant_closure
)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

fn leaves<C: ScalarContainer>(c: &C) -> Vec<C::Scalar> {{
    let mut out = Vec::new();
    apply_scalar_unary(c, |v| out.push(v));
    out
}}

"""

main()
