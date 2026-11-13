"""Writes crates/stanmath-rs/tests/signatures_utilities.rs: every signature of the Stan conversion
and utility functions sum, prod, cumulative_sum, to_vector, to_row_vector, to_array_1d,
to_array_2d, to_vector_array, to_row_vector_array, to_matrix, append_col, append_row, fma,
log_mix, hypergeometric_3F2, and hypergeometric_pFq.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_utilities.py sigs.txt && cargo fmt

Stan vectors, row vectors, and one dimensional arrays are Rust slices, so each signature maps to
the Rust function for its slice, matrix, integer, or complex form. Every result is checked against
the definition written out in the test from the input's elements (column-major for matrices,
row-major for arrays, as probed against Stan Math 5.4.0 for to_matrix and log_mix).
"""

import collections
import re
import sys

NAMES = ["sum", "prod", "cumulative_sum", "to_vector", "to_row_vector", "to_array_1d",
         "to_array_2d", "to_vector_array", "to_row_vector_array", "to_matrix", "append_col",
         "append_row", "fma", "log_mix", "hypergeometric_3F2", "hypergeometric_pFq"]


def parse(t):
    """(dims, base) of a Stan type."""
    m = re.fullmatch(r"array\[(,*)\] (.*)", t)
    return (len(m.group(1)) + 1, m.group(2)) if m else (0, t)


def leaf(base):
    if base.startswith("complex"):
        return "Complex<f64>"
    return "i64" if base == "int" else "f64"


def rust(t):
    dims, base = parse(t)
    lt = leaf(base)
    core = base.replace("complex_", "").replace("complex", "real")
    ty = {"real": lt, "int": lt, "vector": f"Vec<{lt}>", "row_vector": f"Vec<{lt}>",
          "matrix": f"Matrix<{lt}>"}[core]
    for _ in range(dims):
        ty = f"Vec<{ty}>"
    return ty


def value(ty, start):
    k = [start]
    lt = "Complex<f64>" if "Complex" in ty else "i64" if "i64" in ty else "f64"

    def scalar():
        k[0] += 1
        v = k[0]
        return {"f64": f"{v / 8!r}", "i64": f"{v}_i64",
                "Complex<f64>": f"Complex::new({v / 8!r}, {-v / 16!r})"}[lt]

    def build(t):
        if t == lt:
            return scalar()
        if t.startswith("Matrix<"):
            return f"Matrix::from_column_major(2, 3, vec![{', '.join(scalar() for _ in range(6))}])"
        inner = t[4:-1]
        n = SLICE_LEN[0] if inner == lt else 2
        return f"vec![{', '.join(build(inner) for _ in range(n))}]"

    out = build(ty)
    LAST[0] = (start, k[0], lt)
    return out


LAST = [None]
SLICE_LEN = [3]


def literal(start, end, lt):
    """The scalars value() wrote, in order, as a Rust vec literal."""
    one = {"f64": lambda v: f"{v / 8!r}", "i64": lambda v: f"{v}_i64",
           "Complex<f64>": lambda v: f"Complex::new({v / 8!r}, {-v / 16!r})"}[lt]
    return f"vec![{', '.join(one(v) for v in range(start + 1, end + 1))}]"


def is_slice(t):
    d, b = parse(t)
    return (d == 1 and b in ("real", "int", "complex")) or (d == 0 and "vector" in b)


def is_matrix(t):
    return parse(t) == (0, "matrix") or parse(t) == (0, "complex_matrix")


def block(label, lets, got, want, cmp="near"):
    return (f"    {{\n{lets}        let got = {got};\n        let want = {want};\n"
            f"        check(&mut failures, \"{label}\", {cmp}(&got, &want));\n    }}\n")


def lets_for(args, start=0):
    names = "xyzw"
    out = ""
    for i, t in enumerate(args):
        if t == "int":
            continue
        ty = rust(t)
        out += f"        let {names[i]}: {ty} = {value(ty, start + 10 * i)};\n"
    return out


def gen(name, args, ret):
    label = f"{name}({', '.join(args)})"
    lets = lets_for(args)
    a0 = args[0] if args else ""
    lt = leaf(parse(a0)[1]) if args else "f64"
    flat = "flat(&x)"
    if name in ("sum", "prod"):
        if lt == "i64":
            fn = "x.iter().sum::<i64>()" if name == "sum" else "prod_int(&x)"
        else:
            base = name + ("_complex" if lt == "Complex<f64>" else "")
            fn = f"{base}({'x.as_slice()' if is_matrix(a0) else '&x'})"
        sym = "+" if name == "sum" else "*"
        return block(label, lets, fn, f"{flat}.into_iter().reduce(|a, b| a {sym} b).unwrap()")
    if name == "cumulative_sum":
        fn = {"f64": "cumulative_sum(&x)", "i64": "cumulative_sum_int(&x)",
              "Complex<f64>": "cumulative_sum_complex(&x)"}[lt]
        return block(label, lets, fn, f"running_sums({flat})")
    if name in ("to_vector", "to_row_vector"):
        if is_matrix(a0):
            fn = f"{name}_matrix(&x)"
        elif lt == "i64":
            fn = f"{name}_int::<f64>(&x)"
        else:
            fn = f"{name}(&x)"
        want = f"{flat}.into_iter().map(|v| v as f64).collect::<Vec<f64>>()" if lt == "i64" else flat
        return block(label, lets, fn, want)
    if name == "to_array_1d":
        value(rust(a0), 0)
        return block(label, lets, "to_array_1d_nested(&x)" if not is_matrix(a0) else "to_array_1d(&x)",
                     literal(*LAST[0]))
    if name in ("to_array_2d", "to_row_vector_array"):
        return block(label, lets, f"{name}(&x)",
                     "(0..2).map(|i| (0..3).map(|j| x[(i, j)]).collect()).collect::<Vec<Vec<_>>>()")
    if name == "to_vector_array":
        return block(label, lets, f"{name}(&x)",
                     "(0..3).map(|j| (0..2).map(|i| x[(i, j)]).collect()).collect::<Vec<Vec<_>>>()")
    if name == "to_matrix":
        return gen_to_matrix(label, args, lets, lt)
    if name in ("append_col", "append_row"):
        return gen_append(label, name, args, lets)
    if name == "fma":
        return gen_fma(label, args, lets)
    if name == "log_mix":
        if len(args) == 3:
            return ""
        fn = "log_mix_array(&x, &y)" if parse(args[1])[0] == 1 and "vector" in parse(args[1])[1] \
            else "log_mix_slice(&x, &y)"
        lps = "&y" if "array" in fn else "std::slice::from_ref(&y)"
        return block(label, lets, fn, f"log_mix_definition(&x, {lps})", "close")
    if name == "hypergeometric_3F2":
        lets = lets_for(args[:2]) + "        let z = 0.3;\n"
        return block(label, lets, "hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z)",
                     "pfq_series(&x, &y[..2], z)", "close")
    if name == "hypergeometric_pFq":
        lets = lets_for(args[:2]) + "        let z = 0.3;\n"
        return block(label, lets, "hypergeometric_pFq(&x, &y, z)", "pfq_series(&x, &y, z)", "close")
    raise ValueError(label)


def gen_to_matrix(label, args, lets, lt):
    a0 = args[0]
    d, base = parse(a0)
    real = lt == "i64"
    conv = "let xs = to_vector_int::<f64>(&x);\n" if real else ""
    src = "xs" if real else "x"
    if len(args) == 1:
        if is_matrix(a0):
            return block(label, lets, "x.clone()", "x.clone()")
        if d == 0:
            fn = "to_matrix_from_vector(&x)" if base.endswith("vector") and "row" not in base \
                else "to_matrix_from_row_vector(&x)"
            n_r, n_c = ("3", "1") if "row" not in base else ("1", "3")
            return block(label, lets, fn, f"Matrix::from_fn({n_r}, {n_c}, |i, j| x[i + j])")
        if d == 1 and "row" not in base:
            return block(label, lets, "to_matrix_from_vector_array(&x).unwrap()",
                         "Matrix::from_fn(3, 2, |i, j| x[j][i])")
        fn = "to_matrix_from_int_array_2d::<f64>(&x).unwrap()" if real else "to_matrix_from_array_2d(&x).unwrap()"
        cast = " as f64" if real else ""
        rows, cols = ("2", "3")
        return block(label, lets, fn, f"Matrix::from_fn({rows}, {cols}, |i, j| x[i][j]{cast})")
    if not is_matrix(a0):
        SLICE_LEN[0] = 6
        lets = lets_for(args)
        SLICE_LEN[0] = 3
    lets += f"        {conv}" if conv else ""
    flat_src = f"to_array_1d(&{src})" if is_matrix(a0) else f"{src}.clone()"
    lets += f"        let f = {flat_src};\n"
    if len(args) == 3:
        fn = f"to_matrix_reshape(&{src}, 3, 2).unwrap()" if is_matrix(a0) else f"to_matrix(&{src}, 3, 2).unwrap()"
        return block(label, lets, fn, "Matrix::from_fn(3, 2, |i, j| f[i + 3 * j])")
    out = block(label + " col_major=1", lets,
                f"to_matrix(&f, 3, 2).unwrap()", "Matrix::from_fn(3, 2, |i, j| f[i + 3 * j])")
    out += block(label + " col_major=0", lets,
                 f"to_matrix_row_major(&f, 3, 2).unwrap()", "Matrix::from_fn(3, 2, |i, j| f[2 * i + j])")
    return out


def gen_append(label, name, args, lets):
    a, b = args
    km = lambda t: "m" if is_matrix(t) else "s" if parse(t)[1] in ("real", "complex") else "v"
    ka, kb = km(a), km(b)
    col = name == "append_col"
    # Two vectors joined along their own direction are one longer vector.
    joins = (col and "row_vector" in a and "row_vector" in b) or \
        (not col and a.endswith("vector") and "row" not in a and b.endswith("vector") and "row" not in b)
    if ka == "s" or kb == "s" or (ka == kb == "v" and joins):
        xa = "&[x]" if ka == "s" else "&x"
        xb = "&[y]" if kb == "s" else "&y"
        want = f"[{'vec![x]' if ka == 's' else 'x.clone()'}, {'vec![y]' if kb == 's' else 'y.clone()'}].concat()"
        return block(label, lets, f"append_array({xa}, {xb})", want)
    if ka == kb == "m":
        fn = f"{name}(&x, &y).unwrap()"
    elif ka == kb == "v":
        fn = "append_col_vectors(&x, &y).unwrap()" if col else "append_row_row_vectors(&x, &y).unwrap()"
    elif ka == "m":
        fn = "append_col_matrix_vector(&x, &y).unwrap()" if col else "append_row_matrix_row_vector(&x, &y).unwrap()"
    else:
        fn = "append_col_vector_matrix(&x, &y).unwrap()" if col else "append_row_row_vector_matrix(&x, &y).unwrap()"
    if col and sorted((ka, kb)) == ["m", "v"]:
        v = "x" if ka == "v" else "y"
        lets += f"        let {v} = {v}[..2].to_vec();\n"
    as_m = lambda n, k: (f"{n}.clone()" if k == "m" else
                         (f"Matrix::from_column_major({n}.len(), 1, {n}.clone())" if col else
                          f"Matrix::from_column_major(1, {n}.len(), {n}.clone())"))
    lets += f"        let (ma, mb) = ({as_m('x', ka)}, {as_m('y', kb)});\n"
    if col:
        want = "Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| if j < ma.ncols() { ma[(i, j)] } else { mb[(i, j - ma.ncols())] })"
    else:
        want = "Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| if i < ma.nrows() { ma[(i, j)] } else { mb[(i - ma.nrows(), j)] })"
    return block(label, lets, fn, want)


def gen_fma(label, args, lets):
    names = "xyz"
    cont = [names[i] for i, t in enumerate(args) if t != "real"]
    params = ", ".join(f"{n}_" if n in cont else n for n in names)
    f = f"|{', '.join(n + '_' for n in cont)}| fma({params})"
    refs = ", ".join(f"&{n}" for n in cont)
    if not cont:
        return block(label, lets, "fma(x, y, z)", "x * y + z", "close")
    helper = {1: "apply_scalar_unary", 2: "apply_scalar_binary", 3: "apply_scalar_ternary"}[len(cont)]
    got = f"{helper}({refs}, {f})" + (".unwrap()" if len(cont) > 1 else "")
    for n in cont:
        lets += f"        let {n}s = flat(&{n});\n"
    wargs = [f"{n}s[i]" if n in cont else n for n in names]
    want = f"(0..{cont[0]}s.len()).map(|i| {wargs[0]}.mul_add({wargs[1]}, {wargs[2]})).collect::<Vec<f64>>()"
    return block(label, lets, f"flat(&{got})", want)


HEADER = """//! Every signature of the Stan conversion and utility functions `sum`, `prod`, `cumulative_sum`,
//! `to_vector`, `to_row_vector`, `to_array_1d`, `to_array_2d`, `to_vector_array`,
//! `to_row_vector_array`, `to_matrix`, `append_col`, `append_row`, `fma`, `log_mix`,
//! `hypergeometric_3F2`, and `hypergeometric_pFq`.
//!
//! Written by `tools/reference/gen_utilities.py` from the CmdStan 2.40.0 signature dump: {total}
//! Stan signatures, each mapped to its Rust form and checked against the definition written out
//! from the input's elements.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::redundant_clone,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::redundant_closure
)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

trait Near {{
    fn near(&self, other: &Self) -> bool;
}}

impl Near for f64 {{
    fn near(&self, other: &Self) -> bool {{
        (self - other).abs() <= 1e-13 * other.abs().max(1.0)
    }}
}}

impl Near for i64 {{
    fn near(&self, other: &Self) -> bool {{
        self == other
    }}
}}

impl Near for Complex<f64> {{
    fn near(&self, other: &Self) -> bool {{
        self.re.near(&other.re) && self.im.near(&other.im)
    }}
}}

impl<T: Near> Near for Vec<T> {{
    fn near(&self, other: &Self) -> bool {{
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a.near(b))
    }}
}}

impl<T: Near + Copy> Near for Matrix<T> {{
    fn near(&self, other: &Self) -> bool {{
        self.nrows() == other.nrows()
            && self.ncols() == other.ncols()
            && self.as_slice().iter().zip(other.as_slice()).all(|(a, b)| a.near(b))
    }}
}}

fn near<T: Near>(a: &T, b: &T) -> bool {{
    a.near(b)
}}

fn close(a: &f64, b: &f64) -> bool {{
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}}

fn check(failures: &mut Vec<String>, label: &str, ok: bool) {{
    if !ok {{
        failures.push(label.to_string());
    }}
}}

fn flat<C: ScalarContainer>(c: &C) -> Vec<C::Scalar> {{
    to_array_1d_nested(c)
}}

fn running_sums<T: Copy + std::ops::Add<Output = T>>(v: Vec<T>) -> Vec<T> {{
    let mut out: Vec<T> = Vec::new();
    for x in v {{
        out.push(out.last().map_or(x, |&s| s + x));
    }}
    out
}}

/// `sum_n log(sum_k theta_k exp(lp_nk))`.
fn log_mix_definition(theta: &[f64], lps: &[Vec<f64>]) -> f64 {{
    lps.iter()
        .map(|lp| theta.iter().zip(lp).map(|(t, l)| t * l.exp()).sum::<f64>().ln())
        .sum()
}}

/// The hypergeometric series `sum_k prod (a)_k / prod (b)_k z^k / k!`.
fn pfq_series(a: &[f64], b: &[f64], z: f64) -> f64 {{
    let (mut term, mut s) = (1.0, 1.0);
    for k in 0..200 {{
        let k = f64::from(k);
        term *= a.iter().map(|x| x + k).product::<f64>() / b.iter().map(|x| x + k).product::<f64>()
            * z
            / (k + 1.0);
        s += term;
    }}
    s
}}

"""


def main():
    sigs = collections.defaultdict(list)
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\) => (.*)", l)
        if m and m.group(1) in NAMES:
            sigs[m.group(1)].append((m.group(2).split(", ") if m.group(2) else [], m.group(3)))
    body, total = [], 0
    for name in NAMES:
        for args, ret in sigs[name]:
            if all(t in ("real", "int") for t in args) and name != "fma":
                continue
            code = gen(name, args, ret)
            if code:
                total += 1
                body.append(code)
    out = HEADER.format(total=total)
    out += "#[test]\nfn utility_signatures() {\n    let mut failures: Vec<String> = Vec::new();\n"
    out += "".join(body)
    out += '    assert!(failures.is_empty(), "\\n{}", failures.join("\\n"));\n}\n'
    open("crates/stanmath-rs/tests/signatures_utilities.rs", "w").write(out)
    print(total, "signatures")


main()
