"""Writes crates/stanmath-rs-distributions/tests/signatures.rs, one call per distribution signature
shape in the Stan language.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_signatures.py sigs.txt && cargo fmt

Every vectorized `_lpdf`, `_lpmf`, `_lcdf`, `_lccdf`, `_cdf`, `_lupdf`, and `_lupmf` signature whose
arguments are int, real, vector, row_vector, or one dimensional arrays is mapped to its Rust call:
a scalar argument is a scalar, and a container is a slice of three elements. Stan's vector,
row_vector, and array forms of one argument collapse to the same Rust slice, so each Rust shape
stands for several Stan signatures. For each shape the test requires the broadcast call to equal
the sum (the product for `_cdf`) of the scalar calls at each index, in value and gradient. The
scalar calls themselves are compared with Stan Math by stan_reference.rs and stan_reference_extra.rs,
whose fixtures give each family its base point.

Signatures with matrix or array of vector arguments, with a container that never broadcasts
(categorical, multinomial, Poisson binomial), and the 6 to 9 argument Wiener forms are left to
the Stan reference tests.
"""

import collections
import itertools
import re
import sys

import ledger

DIST = re.compile(r"_(lpdf|lpmf|lcdf|lccdf|cdf|lupdf|lupmf)$")
SIMPLE = {"int", "real", "vector", "row_vector", "array[] int", "array[] real"}
DATA = "crates/stanmath-rs-distributions/tests/data"
# Base points for families the Stan fixtures do not cover: (ints, reals).
EXTRA = {
    "discrete_range_lpmf": ([3, 1, 5], []),
    "discrete_range_lcdf": ([3, 1, 5], []),
    "hypergeometric_lpmf": ([2, 5, 4, 6], []),
    "yule_simon_lpmf": ([3], [2.5]),
    "wiener_lpdf": ([], [1.2, 2.0, 0.3, 0.5, 0.8]),
}
N = 3


def base_name(n):
    n = re.sub(r"_lup([dm])f$", r"_lp\1f", n)
    if n.endswith("_cdf"):
        n = n[: -len("_cdf")] + "_lcdf"
    if n.endswith("_lccdf") or n.endswith("_lcdf"):
        pass
    return n


def points():
    out = {}
    for f in ("stan_reference.csv", "stan_reference_extra.csv"):
        for line in open(f"{DATA}/{f}"):
            if line.startswith("#"):
                continue
            name, ints, reals = line.split(",")[:3]
            if name not in out:
                out[name] = ([int(v) for v in ints.split(";") if v], [float(v) for v in reals.split(";") if v])
    out.update(EXTRA)
    return out


def base_point(name, pts):
    b = base_name(name)
    for cand in (b, re.sub(r"_l?c?cdf$", "_lpdf", b), re.sub(r"_l?c?cdf$", "_lpmf", b), re.sub(r"_l?c?cdf$", "_lcdf", b)):
        if cand in pts:
            return pts[cand]
    return None


def lit(v):
    s = repr(float(v))
    return s


def main():
    fam = collections.defaultdict(list)
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\) => (.*)", l)
        if m and DIST.search(m.group(1)):
            fam[m.group(1)].append(m.group(2).split(", "))
    pts = points()
    tests = []
    skipped = []
    total = 0
    for name in sorted(fam):
        sigs = [s for s in fam[name] if all(t in SIMPLE for t in s)]
        if not sigs or (name.startswith("wiener") and len(sigs[0]) != 5):
            skipped.append(name)
            continue
        sigs = [s for s in sigs if len(s) == len(sigs[0])]
        k = len(sigs[0])
        cont = [[t not in ("int", "real") for t in (s[p] for s in sigs)] for p in range(k)]
        if any(all(c) for c in cont):
            skipped.append(name)
            continue
        is_int = [all(s[p] in ("int", "array[] int") for s in sigs) for p in range(k)]
        bp = base_point(name, pts)
        if bp is None:
            skipped.append(name)
            continue
        ints, reals = list(bp[0]), list(bp[1])
        if len(ints) != sum(is_int) or len(reals) != k - sum(is_int):
            skipped.append(name)
            continue
        shapes = collections.Counter(tuple(t not in ("int", "real") for t in s) for s in sigs)
        total += len(sigs)
        for s in sigs:
            ledger.record("called", name, s)
        body = []
        for shape in sorted(shapes):
            body.append(call(name, shape, is_int, ints, reals))
        tests.append((name, len(sigs), body))
    out = [HEADER.format(total=total, shapes=sum(len(b) for _, _, b in tests), skipped=", ".join(skipped))]
    for name, nsig, body in tests:
        out.append(f"#[test]\nfn {name}_signatures() {{\n    let mut failures = Vec::new();\n")
        out.extend(body)
        out.append("    assert!(failures.is_empty(), \"\\n{}\", failures.join(\"\\n\"));\n}\n")
    open("crates/stanmath-rs-distributions/tests/signatures.rs", "w").write("\n".join(out))
    print(total, "signatures,", sum(len(b) for _, _, b in tests), "shapes; skipped", skipped)


def call(name, shape, is_int, ints, reals):
    prod = name.endswith("_cdf")
    label = name + "(" + ", ".join("vec" if c else "scalar" for c in shape) + ")"
    ii = iter(ints)
    ri = iter(reals)
    params = []
    vec_args, sc_args = [], []
    for p, c in enumerate(shape):
        if is_int[p]:
            v = next(ii)
            vec_args.append(f"&[{v}i64; {N}][..]" if c else f"{v}i64")
            sc_args.append(f"{v}i64")
        else:
            v = next(ri)
            off = len(params)
            if c:
                params.extend(v * (1 + 0.02 * i) for i in range(N))
                vec_args.append(f"&x[{off}..{off + N}]")
                sc_args.append(f"x[{off} + i]")
            else:
                params.append(v)
                vec_args.append(f"x[{off}]")
                sc_args.append(f"x[{off}]")
    n = N if any(shape) else 1
    va, sa = ", ".join(vec_args), ", ".join(sc_args)
    iv = "i" if "+ i]" in sa else "_i"
    op = "*=" if prod else "+="
    if not params:
        va = va.replace("x[", "X[")
        return (
            f"    {{\n        let v = {name}({va});\n"
            f"        let s = (0..{n}).try_fold({1.0 if prod else 0.0}, |acc, _i| {name}({sa}).map(|r| acc {'*' if prod else '+'} r));\n"
            f"        compare_data({label!r}, v, s, &mut failures);\n    }}\n"
        ).replace("'", '"')
    p = ", ".join(lit(v) for v in params)
    init = "x[0] * 0.0 + 1.0" if prod else "x[0] * 0.0"
    return (
        f"    {{\n        let p = [{p}];\n"
        f"        let v = try_value_and_grad(|x| {name}({va}), &p);\n"
        f"        let s = try_value_and_grad(\n            |x| {{\n                let mut acc = {init};\n"
        f"                for {iv} in 0..{n} {{\n                    acc {op} {name}({sa})?;\n                }}\n"
        f"                Ok(acc)\n            }},\n            &p,\n        );\n"
        f"        compare({label!r}, v, s, &mut failures);\n    }}\n"
    ).replace("'", '"')


HEADER = """//! Every vectorized distribution signature of the Stan language, mapped to its Rust call.
//!
//! Written by `tools/reference/gen_signatures.py` from the CmdStan 2.40.0 signature dump: {total}
//! Stan signatures in {shapes} Rust shapes (vector, row_vector, and array forms of one argument
//! are the same Rust slice). Each broadcast call must equal the sum (the product for `_cdf`) of
//! the scalar calls at each index, in value and gradient; the scalar calls are compared with Stan
//! Math by `stan_reference.rs` and `stan_reference_extra.rs`.
//!
//! Left to the Stan reference tests: {skipped}.
#![allow(clippy::too_many_lines, clippy::unreadable_literal, clippy::identity_op)]

use stanmath_rs_autodiff::{{ValueAndGrad, try_value_and_grad}};
use stanmath_rs_core::Result;
use stanmath_rs_distributions::*;

fn close(a: f64, b: f64) -> bool {{
    a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}}

fn compare(
    label: &str,
    v: Result<ValueAndGrad>,
    s: Result<ValueAndGrad>,
    failures: &mut Vec<String>,
) {{
    match (v, s) {{
        (Err(_), Err(_)) => {{}}
        (Ok(v), Ok(s)) => {{
            let same = close(v.value, s.value)
                && v.gradient.iter().zip(&s.gradient).all(|(a, b)| close(*a, *b));
            if !same {{
                failures.push(format!("{{label}}: {{v:?}} vs {{s:?}}"));
            }}
        }}
        (v, s) => failures.push(format!("{{label}}: {{v:?}} vs {{s:?}}")),
    }}
}}

fn compare_data(label: &str, v: Result<f64>, s: Result<f64>, failures: &mut Vec<String>) {{
    match (v, s) {{
        (Err(_), Err(_)) => {{}}
        (Ok(v), Ok(s)) if close(v, s) => {{}}
        (v, s) => failures.push(format!("{{label}}: {{v:?}} vs {{s:?}}")),
    }}
}}
"""

main()
