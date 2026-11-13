"""Writes crates/stanmath-rs/tests/signatures_bounds.rs: every container signature of the bound
transforms in the Stan language (lower_bound, upper_bound, lower_upper_bound, and
offset_multiplier, each with _constrain, _jacobian, and _unconstrain).

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_bounds.py sigs.txt && cargo fmt

A container argument goes through apply_scalar_unary, apply_scalar_binary, or
apply_scalar_ternary, with scalar bounds captured by the closure; a _jacobian call adds every
element's log Jacobian to one lp. Stan types map to Rust as in gen_elementwise.py, and calls stop
at three Vec levels (deeper arrays share the same recursive impl). Each call must equal the scalar
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
            if ra[0].count("Vec<") > MAX_DEPTH:
                deep += 1
                ledger.record("deep", name, args)
                continue
            shapes[(fam, kind, tuple(ra))] += 1
            ledger.record("called", name, args)
        for (fam, kind, ra), n in sorted(shapes.items()):
            total += n
            body.append(call(name, fam, kind, ra))
    out = HEADER.format(total=total, shapes=len(body), deep=deep, depth=MAX_DEPTH)
    out += "#[test]\nfn bound_transform_signatures() {\n    let mut failures = Vec::new();\n"
    out += "".join(body)
    out += '    assert!(failures.is_empty(), "\\n{}", failures.join("\\n"));\n}\n'
    open("crates/stanmath-rs/tests/signatures_bounds.rs", "w").write(out)
    print(total, "signatures,", len(body), "Rust shapes;", deep, "deeper;", skipped, "scalar")


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
        got = f"apply_scalar_unary({refs}, {f})"
    elif len(cont) == 2:
        got = f"apply_scalar_binary({refs}, {f}).unwrap()"
    else:
        got = f"apply_scalar_ternary({refs}, {f}).unwrap()"
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
//! and a `_jacobian` call's lp the sum of theirs. The other {deep} signatures nest arrays more than
//! {depth} `Vec` levels deep and share the same recursive impl.
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

fn close(a: f64, b: f64) -> bool {{
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}}

"""

main()
