"""Writes crates/stanmath-rs/tests/signatures_elementwise.rs: every vectorized form of a scalar
function in the Stan language, as its Rust call through apply_scalar_unary or apply_scalar_binary.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_elementwise.py sigs.txt && cargo fmt

A signature is elementwise when the function also has a scalar signature and every container
argument has the shape of the result. Stan types map to Rust containers: vector, row_vector, and
array[] real are Vec<f64>; matrix is Matrix<f64>; each further array dimension adds a Vec; int
leaves are i64. Signatures whose Stan types map to the same Rust types share one call. Each call
must return the scalar function of every element (NaN where the scalar function rejects it); the
scalar functions are compared with Stan Math by stan_reference_scalar.rs.
"""

import collections
import re
import sys

SCALAR = {"int", "real"}
# Rust operators (+, -, *, /) or, for squared_distance, a reduction rather than an elementwise map.
OPERATORS = {"add", "subtract", "multiply", "elt_multiply", "elt_divide", "squared_distance"}
# Arrays nest through one recursive impl (Vec<C> for any container C), so calls stop at this many
# Vec levels; deep_arrays() checks one function to the deepest level Stan has.
MAX_DEPTH = 3


def rust(t):
    """(Rust type, depth, leaf) for a Stan type, or None for types outside this test."""
    m = re.fullmatch(r"array\[(,*)\] (.*)", t)
    dims = len(m.group(1)) + 1 if m else 0
    base = m.group(2) if m else t
    if base in ("int", "real"):
        leaf = "i64" if base == "int" else "f64"
        ty = leaf
    elif base in ("vector", "row_vector"):
        leaf, ty = "f64", "Vec<f64>"
    elif base == "matrix":
        leaf, ty = "f64", "Matrix<f64>"
    else:
        return None
    for _ in range(dims):
        ty = f"Vec<{ty}>"
    return ty, leaf


def value(ty, leaf, base, ints):
    """A literal of Rust type ty whose scalars start at base (or the int ints)."""
    counter = [0]

    def nxt():
        counter[0] += 1
        if leaf == "i64":
            return f"{ints}_i64"
        return repr(round(base * (1 + 0.03 * counter[0]), 12))

    def build(t):
        if t in ("f64", "i64"):
            return nxt()
        if t == "Matrix<f64>":
            return f"Matrix::from_column_major(2, 2, vec![{', '.join(nxt() for _ in range(4))}])"
        inner = t[4:-1]
        return f"vec![{build(inner)}, {build(inner)}]"

    return build(ty)


def scalar_points(path):
    pts = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.split(",")
        if f[3] != "error" and f[0] not in pts:
            pts[f[0]] = [float(v) for v in f[2].split(";")]
    return pts


def main():
    sigs = collections.defaultdict(list)
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\) => (.*)", l)
        if m:
            sigs[m.group(1)].append((m.group(2).split(", ") if m.group(2) else [], m.group(3)))
    pts = scalar_points("crates/stanmath-rs/tests/data/stan_reference_scalar.csv")
    body = []
    total = 0
    names = 0
    deep = [0]
    for name in sorted(sigs):
        if name.endswith("_rng") or name.startswith(("lower_bound", "upper_bound")) or name in OPERATORS:
            continue
        scalar = [a for a, r in sigs[name] if all(t in SCALAR for t in a) and r in SCALAR]
        if not scalar or len(scalar[0]) not in (1, 2):
            continue
        k = len(scalar[0])
        is_int = [all(a[p] == "int" for a in scalar) for p in range(k)]
        if name not in pts and not all(is_int):
            continue
        reals = list(pts.get(name, []))
        calls = {}
        for a, r in sigs[name]:
            if len(a) != k or all(t in SCALAR for t in a):
                continue
            ra = [rust(t) for t in a]
            if None in ra or rust(r) is None:
                continue
            cont = [x for x in ra if x[0] not in ("f64", "i64")]
            if any(c[0].replace("i64", "f64") != cont[0][0].replace("i64", "f64") for c in cont):
                continue
            if max(x[0].count("Vec<") for x in ra) > MAX_DEPTH:
                deep[0] += 1
                continue
            calls.setdefault(tuple(x[0] for x in ra), []).append(a)
        if not calls:
            continue
        names += 1
        for shape, stan in sorted(calls.items()):
            total += len(stan)
            body.append(call(name, shape, is_int, reals))
    out = HEADER.format(total=total, names=names, shapes=len(body), deep=deep[0], depth=MAX_DEPTH)
    out += "#[test]\nfn elementwise_signatures() {\n    let mut failures = Vec::new();\n"
    out += "".join(body)
    out += '    assert!(failures.is_empty(), "\\n{}", failures.join("\\n"));\n}\n'
    out += DEEP
    open("crates/stanmath-rs/tests/signatures_elementwise.rs", "w").write(out)
    print(total, "signatures,", names, "functions,", len(body), "Rust shapes;", deep[0], "deeper")


def arg(p, leaf, is_int):
    """The scalar call's argument from a leaf variable."""
    if is_int[p]:
        return f"{leaf} as _"
    return f"{leaf} as f64"


def call(name, shape, is_int, reals):
    k = len(shape)
    label = f"{name}({', '.join(shape)})"
    ri = iter(reals)
    lits = []
    for p, ty in enumerate(shape):
        leaf = "i64" if "i64" in ty else "f64"
        if is_int[p]:
            lits.append(value(ty, "i64", 0, 2))
        else:
            v = next(ri)
            lits.append(value(ty, leaf, v, max(1, round(abs(v)))))
    if k == 1:
        f = f"|v| eval({name}({arg(0, 'v', is_int)}))"
        return (
            f"    {{\n        let x: {shape[0]} = {lits[0]};\n"
            f"        let got = apply_scalar_unary(&x, {f});\n"
            f"        let want = apply_scalar_unary(&x, {f});\n"
            f"        compare({label!r}, leaves(&got), leaves(&want), &mut failures);\n    }}\n"
        ).replace("'", '"')
    a_c = shape[0] not in ("f64", "i64")
    b_c = shape[1] not in ("f64", "i64")
    a_arg, b_arg = arg(0, "a", is_int), arg(1, "b", is_int)
    lines = [f"    {{\n        let a: {shape[0]} = {lits[0]};\n        let b: {shape[1]} = {lits[1]};\n"]
    if a_c and b_c and shape[0] == shape[1]:
        lines.append(f"        let got = apply_scalar_binary(&a, &b, |a, b| eval({name}({a_arg}, {b_arg}))).unwrap();\n")
        want = f"apply_scalar_binary(&a, &b, |a, b| eval({name}({a_arg}, {b_arg}))).unwrap()"
    elif a_c and b_c:
        # Containers of the same shape with int and real leaves: pair them by position.
        lines.append(f"        let bl = leaves(&apply_scalar_unary(&b, |b| b as f64));\n        let mut i = 0;\n")
        lines.append(f"        let got = apply_scalar_unary(&a, |a| {{ let b = bl[i]; i += 1; eval({name}({a_arg}, {b_arg})) }});\n")
        want = None
    elif a_c:
        lines.append(f"        let got = apply_scalar_unary(&a, |a| eval({name}({a_arg}, {b_arg})));\n")
        want = None
    else:
        lines.append(f"        let got = apply_scalar_unary(&b, |b| eval({name}({a_arg}, {b_arg})));\n")
        want = None
    lines.append(
        f"        let n = leaves(&got).len();\n"
        f"        compare({label!r}, leaves(&got), {'leaves(&' + want + ')' if want else 'leaves(&got)'}, &mut failures);\n"
        f"        assert!(n > 0);\n    }}\n"
    )
    return "".join(lines).replace("'", '"')


DEEP = """
#[test]
fn deep_arrays() {
    // array[,,,,,,,] real, the deepest array in the signature dump.
    let mut x = vec![0.5, -1.0];
    let mut depth8: Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>> = Vec::new();
    for _ in 0..2 {
        depth8 = vec![vec![vec![vec![vec![vec![vec![x.clone()]]]]]]];
        x.push(2.0);
    }
    let got = apply_scalar_unary(&depth8, exp);
    let want: Vec<f64> = leaves(&depth8).into_iter().map(f64::exp).collect();
    assert_eq!(leaves(&got), want);
    let pairs = apply_scalar_binary(&depth8, &got, |a, b| pow(b, a)).unwrap();
    assert_eq!(leaves(&pairs).len(), want.len());
}
"""

HEADER = """//! Every vectorized form of a scalar function of one or two arguments in the Stan language,
//! called through `apply_scalar_unary` and `apply_scalar_binary`.
//!
//! Written by `tools/reference/gen_elementwise.py` from the CmdStan 2.40.0 signature dump: {total}
//! Stan signatures of {names} functions in {shapes} Rust shapes. Each call must keep the container's
//! shape and return the scalar function of every element; the scalar functions are compared with
//! Stan Math by `stan_reference_scalar.rs`. The other {deep} signatures nest arrays more than {depth}
//! `Vec` levels deep; they go through the same recursive impl, which `deep_arrays` checks to depth 8.
//! `add`, `subtract`, `multiply`, `elt_multiply`, and `elt_divide` are Rust's operators.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::unnecessary_cast,
    clippy::approx_constant,
    clippy::redundant_closure,
    clippy::type_complexity
)]

use stanmath_rs::prelude::*;

/// The scalar functions return a value or a `Result`; a rejected argument becomes NaN.
trait Eval {{
    fn eval(self) -> f64;
}}

impl Eval for f64 {{
    fn eval(self) -> f64 {{
        self
    }}
}}

impl Eval for i64 {{
    fn eval(self) -> f64 {{
        self as f64
    }}
}}

impl Eval for i32 {{
    fn eval(self) -> f64 {{
        f64::from(self)
    }}
}}

impl<E: Eval> Eval for Result<E> {{
    fn eval(self) -> f64 {{
        self.map_or(f64::NAN, Eval::eval)
    }}
}}

fn eval(v: impl Eval) -> f64 {{
    v.eval()
}}

/// The scalars of a container in order.
fn leaves<C: ScalarContainer<Scalar = f64>>(c: &C) -> Vec<f64> {{
    let mut out = Vec::new();
    apply_scalar_unary(c, |v| out.push(v));
    out
}}

fn compare(label: &str, got: Vec<f64>, want: Vec<f64>, failures: &mut Vec<String>) {{
    let same = got.len() == want.len()
        && got.iter().zip(&want).all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
    if !same {{
        failures.push(format!("{{label}}: {{got:?}} vs {{want:?}}"));
    }}
}}

"""

main()
