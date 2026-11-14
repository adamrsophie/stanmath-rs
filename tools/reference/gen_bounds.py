"""Writes crates/stanmath-rs/tests/signatures_bounds.rs: every container signature of the bound
transforms in the Stan language (lower_bound, upper_bound, lower_upper_bound, and
offset_multiplier, each with _constrain, _jacobian, and _unconstrain).

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_bounds.py sigs.txt && cargo fmt

A container argument goes through apply_scalar_unary, apply_scalar_binary, or
apply_scalar_ternary, with scalar bounds captured by the closure; a _jacobian call adds every
element's log Jacobian to one lp. Stan types map to Rust as in gen_elementwise.py; literals branch
in two up to three Vec levels and outer levels repeat their element (up()). Each call must equal the scalar
transform of every element in order, and its lp the sum of theirs; the scalar transforms are
compared with Stan Math by stan_reference_matrix.rs and the transforms tests.
"""

import collections
import re
import sys

import ledger

MAX_DEPTH = 3
BASES = {"lower_bound": [-1.0], "upper_bound": [2.0], "lower_upper_bound": [-1.0, 2.0],
         "offset_multiplier": [0.5, 1.5]}


def rust(t):
    m = re.fullmatch(r"array\[(,*)\] (.*)", t)
    dims = len(m.group(1)) + 1 if m else 0
    base = m.group(2) if m else t
    ty = {"real": "f64", "vector": "Vec<f64>", "row_vector": "Vec<f64>", "matrix": "Matrix<f64>"}.get(base)
    if ty is None:
        return None
    for _ in range(dims):
        ty = f"Vec<{ty}>"
    return ty


def value(ty, base):
    k = [0]

    def nxt():
        k[0] += 1
        return repr(round(base * (1 + 0.03 * k[0]), 12))

    def build(t):
        if t == "f64":
            return nxt()
        if t == "Matrix<f64>":
            return f"Matrix::from_column_major(2, 2, vec![{', '.join(nxt() for _ in range(4))}])"
        inner = t[4:-1]
        if t.count("Vec<") > MAX_DEPTH:
            return f"up({build(inner)})"
        return f"vec![{build(inner)}, {build(inner)}]"

    return build(ty)


def main():
    sigs = collections.defaultdict(list)
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"((lower_bound|upper_bound|lower_upper_bound|offset_multiplier)_"
                         r"(constrain|jacobian|unconstrain))\((.*)\) => (.*)", l)
        if m:
            sigs[m.group(1)].append((m.group(2), m.group(3), m.group(4).split(", ")))
    body, total, deep, skipped = [], 0, 0, 0
    for name in sorted(sigs):
        shapes = collections.defaultdict(int)
        for fam, kind, args in sigs[name]:
            ra = [rust(t) for t in args]
            if None in ra or ra[0] == "f64":
                skipped += ra[0] == "f64"
                continue
            if any(r not in ("f64", ra[0]) for r in ra[1:]):
                skipped += 1
                continue
            shapes[(fam, kind, tuple(ra))] += 1
            ledger.record("called", name, args)
        blocks = []
        for (fam, kind, ra), n in sorted(shapes.items()):
            total += n
            blocks.append(call(name, fam, kind, ra))
        body.append((name, blocks))
    shapes = sum(len(b) for _, b in body)
    out = HEADER.format(total=total, shapes=shapes)
    for name, blocks in body:
        out += f"#[test]\nfn {name}_signatures() {{\n    let mut failures = Vec::new();\n"
        out += "".join(blocks)
        out += '    assert!(failures.is_empty(), "\\n{}", failures.join("\\n"));\n}\n\n'
    open("crates/stanmath-rs/tests/signatures_bounds.rs", "w").write(out)
    print(total, "signatures,", shapes, "Rust shapes;", skipped, "scalar")


def call(name, fam, kind, ra):
    label = f"{name}({', '.join(ra)})"
    bases = BASES[fam]
    names = ["x"] + ["b", "c"][: len(bases)]
    lets = [f"        let x: {ra[0]} = {value(ra[0], 0.3)};\n"]
    for nm, ty, base in zip(names[1:], ra[1:], bases):
        lets.append(f"        let {nm}: {ty} = {value(ty, base)};\n")
    cont = [nm for nm, ty in zip(names, ra) if ty != "f64"]
    lp = kind == "jacobian"
    params = ", ".join(f"{n}_" if n in cont else n for n in names)
    closure_args = ", ".join(f"{n}_" for n in cont)
    tail = ", &mut lp" if lp else ""
    f = f"|{closure_args}| eval({name}({params}{tail}))"
    refs = ", ".join(f"&{n}" for n in cont)
    if len(cont) == 1:
        got = f"un({refs}, &mut {f})"
    elif len(cont) == 2:
        got = f"bin({refs}, &mut {f}).unwrap()"
    else:
        got = f"ter({refs}, &mut {f}).unwrap()"
    wargs = ", ".join(f"{n}s[i]" if n in cont else n for n in names)
    lines = "".join(lets)
    if lp:
        lines += "        let mut lp = 0.0;\n"
    lines += f"        let got = leaves(&{got});\n"
    for n in cont:
        lines += f"        let {n}s = leaves(&{n});\n"
    if lp:
        lines += "        let got_lp = lp;\n        let mut lp = 0.0;\n"
    lines += f"        let want: Vec<f64> = (0..xs.len()).map(|i| eval({name}({wargs}{tail}))).collect();\n"
    check = "got == want" + (" && close(got_lp, lp)" if lp else "")
    lines += (f"        if !({check}) {{\n"
              f"            failures.push(format!(\"{label}: {{got:?}} vs {{want:?}}\"));\n        }}\n")
    return "    {\n" + lines + "    }\n"


HEADER = """//! Every container signature of the bound transforms in the Stan language, through
//! `apply_scalar_unary`, `apply_scalar_binary`, and `apply_scalar_ternary`.
//!
//! Written by `tools/reference/gen_bounds.py` from the CmdStan 2.40.0 signature dump: {total} Stan
//! signatures in {shapes} Rust shapes. Each call must equal the scalar transform of every element,
//! and a `_jacobian` call's lp the sum of theirs. Literals branch in two up to three `Vec` levels;
//! outer levels repeat their element (`up`), so every depth Stan has is called.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::approx_constant,
    clippy::type_complexity,
    clippy::float_cmp
)]

use stanmath_rs::prelude::*;

/// The transforms return a value or a `Result`; a rejected argument becomes NaN.
trait Eval {{
    fn eval(self) -> f64;
}}

impl Eval for f64 {{
    fn eval(self) -> f64 {{
        self
    }}
}}

impl Eval for Result<f64> {{
    fn eval(self) -> f64 {{
        self.unwrap_or(f64::NAN)
    }}
}}

fn eval(v: impl Eval) -> f64 {{
    v.eval()
}}

fn leaves<C: ScalarContainer<Scalar = f64>>(c: &C) -> Vec<f64> {{
    let mut out = Vec::new();
    apply_scalar_unary(c, |v| out.push(v));
    out
}}

/// The apply functions with trait objects, so each container type compiles once rather than once
/// per closure.
fn un<C: ScalarContainer>(x: &C, f: &mut dyn FnMut(C::Scalar) -> f64) -> C::Mapped<f64> {{
    apply_scalar_unary(x, f)
}}

fn bin<C: ScalarContainer>(
    a: &C,
    b: &C,
    f: &mut dyn FnMut(C::Scalar, C::Scalar) -> f64,
) -> Result<C::Mapped<f64>> {{
    apply_scalar_binary(a, b, f)
}}

fn ter<C: ScalarContainer>(
    a: &C,
    b: &C,
    c: &C,
    f: &mut dyn FnMut(C::Scalar, C::Scalar, C::Scalar) -> f64,
) -> Result<C::Mapped<f64>> {{
    apply_scalar_ternary(a, b, c, f)
}}

/// Two copies of `x`: one more array level.
fn up<T: Clone>(x: T) -> Vec<T> {{
    vec![x.clone(), x]
}}

fn close(a: f64, b: f64) -> bool {{
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}}

"""

main()
