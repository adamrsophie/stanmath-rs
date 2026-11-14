"""Writes crates/stanmath-rs-distributions/tests/signatures_glm.rs: every `_glm_lpdf` and
`_glm_lpmf` signature in the Stan language, each compared with the sum of the response
distribution over the observations, in value and gradient.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_glm.py sigs.txt && cargo fmt

Stan types map to Rust: `matrix x` is a 3 by 2 matrix, `row_vector x` a 1 by 2 matrix that every
observation shares, and `vector x` a 3 by 1 matrix (one predictor); a real response is a one
element slice that every observation shares, an int response an `i64`, and an `array[] int` a
slice with one entry per observation. `alpha`, `sigma`, and `phi` are scalars or one value per
observation (with a one row x and a real response there is one observation). The response
distributions are compared with Stan Math by stan_reference.rs.

Stan Math 5.4.0 differs in one case: poisson_log_glm_lpmf with an int response and several rows
of x adds -lgamma(y + 1) once rather than once per observation. stanmath-rs returns the sum.
"""

import re
import sys

import ledger

N = 3
INTS = {
    "bernoulli_logit": [1, 0, 1], "poisson_log": [2, 0, 3], "neg_binomial_2_log": [2, 0, 3],
    "binomial_logit": [1, 3, 2], "ordered_logistic": [1, 3, 2], "categorical_logit": [1, 3, 2],
}
TRIALS = [4, 5, 6]
CATS = 3


class Params:
    """The real parameter vector: each request takes the next values."""

    def __init__(self):
        self.vals = []

    def take(self, k):
        o = len(self.vals)
        self.vals += [round(0.37 * ((i + o) % 5) - 0.6 + 0.05 * (i + o), 4) for i in range(k)]
        return o


def main():
    blocks, labels, total = [], [], 0
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"([a-z_0-9]+)_glm_(lpdf|lpmf)\((.*)\) => real", l)
        if not m:
            continue
        name, args = f"{m.group(1)}_glm_{m.group(2)}", m.group(3).split(", ")
        ledger.record("called", name, args)
        total += 1
        blocks.append(call(m.group(1), name, args))
    out = HEADER.format(total=total)
    out += "#[test]\nfn glm_signatures() {\n    let mut failures = Vec::new();\n"
    out += "".join(blocks)
    out += '    assert!(failures.is_empty(), "\\n{}", failures.join("\\n"));\n}\n'
    open("crates/stanmath-rs-distributions/tests/signatures_glm.rs", "w").write(out)
    print(total, "signatures")


def call(fam, name, args):
    p = Params()
    label = f"{name}({', '.join(args)})"
    lets, call_args, scal = [], [], {}
    # The response(s).
    xpos = 2 if fam == "binomial_logit" else 1
    resp = args[:xpos]
    xt = args[xpos]
    k = 1 if xt == "vector" else 2
    rows = 1 if xt == "row_vector" else N
    nobs = N if (rows == N or any(t.startswith("array") or t == "vector" for t in resp)) else 1
    if fam == "normal_id":
        if resp[0] == "real":
            o = p.take(1)
            call_args.append(f"&x[{o}..{o + 1}]")
            scal["y"] = f"x[{o}]"
        else:
            o = p.take(N)
            call_args.append(f"&x[{o}..{o + N}]")
            scal["y"] = f"x[{o} + i]"
    else:
        for j, t in enumerate(resp):
            vals = TRIALS if j == 1 else INTS[fam]
            if t == "int":
                call_args.append(f"{vals[0]}_i64")
                scal[f"y{j}"] = f"{vals[0]}_i64"
            else:
                lets.append(f"        let y{j} = {vals[:N]}_i64;\n".replace("[", "[").replace(", ", "_i64, ").replace("]_i64", "_i64]"))
                call_args.append(f"&y{j}[..]")
                scal[f"y{j}"] = f"y{j}[i]"
    o = p.take(rows * k)
    xm = f"Matrix::from_column_major({rows}, {k}, x[{o}..{o + rows * k}].to_vec())"
    call_args.append("&xm")
    rest = args[xpos + 1:]
    if fam == "categorical_logit":
        oa = p.take(CATS)
        ob = p.take(k * CATS)
        call_args += [f"&x[{oa}..{oa + CATS}]", "&bm"]
        lets_x = (f"                let bm = Matrix::from_column_major({k}, {CATS}, x[{ob}..{ob + k * CATS}].to_vec());\n")
        eta = (f"let eta: Vec<_> = (0..{CATS}).map(|c| x[{oa} + c] + (0..{k}).map(|j| xm[(r, j)] * bm[(j, c)]).sum::<Var>()).collect();")
        term = f"categorical_logit_lpmf({scal['y0']}, &eta)?"
    elif fam == "ordered_logistic":
        ob = p.take(k)
        oc = p.take(2)
        p.vals[oc + 1] = p.vals[oc] + 1.1
        call_args += [f"&x[{ob}..{ob + k}]", f"&x[{oc}..{oc + 2}]"]
        lets_x = ""
        eta = f"let eta = (0..{k}).map(|j| xm[(r, j)] * x[{ob} + j]).sum::<Var>();"
        term = f"ordered_logistic_lpmf({scal['y0']}, eta, &x[{oc}..{oc + 2}])?"
    else:
        alpha = rest[0]
        if alpha == "real":
            oa = p.take(1)
            call_args.append(f"x[{oa}]")
            a_i = f"x[{oa}]"
        else:
            oa = p.take(nobs)
            call_args.append(f"&x[{oa}..{oa + nobs}]")
            a_i = f"x[{oa} + i]"
        ob = p.take(k)
        call_args.append(f"&x[{ob}..{ob + k}]")
        lets_x = ""
        eta = f"let eta = {a_i} + (0..{k}).map(|j| xm[(r, j)] * x[{ob} + j]).sum::<Var>();"
        if fam == "normal_id":
            if rest[2] == "real":
                os_ = p.take(1)
                p.vals[os_] = 1.3
                call_args.append(f"x[{os_}]")
                s_i = f"x[{os_}]"
            else:
                os_ = p.take(nobs)
                for i in range(nobs):
                    p.vals[os_ + i] = 0.8 + 0.3 * i
                call_args.append(f"&x[{os_}..{os_ + nobs}]")
                s_i = f"x[{os_} + i]"
            term = f"normal_lpdf({scal['y']}, eta, {s_i})?"
        elif fam == "neg_binomial_2_log":
            op = p.take(1)
            p.vals[op] = 2.5
            call_args.append(f"x[{op}]")
            term = f"neg_binomial_2_log_lpmf({scal['y0']}, eta, x[{op}])?"
        elif fam == "binomial_logit":
            term = f"binomial_logit_lpmf({scal['y0']}, {scal['y1']}, eta)?"
        else:
            term = f"{fam}_lpmf({scal['y0']}, eta)?"
    vals = ", ".join(repr(v) for v in p.vals)
    uses_i = "i]" in term or "+ i]" in eta
    idx = "i" if uses_i else "_i"
    row = "i" if rows == N else "0"
    if row == "i":
        idx = "i"
    return (
        f"    // stan: {label}\n    {{\n" + "".join(lets) + f"        let p = [{vals}];\n"
        f"        let v = try_value_and_grad(|x| {{\n            let xm = {xm};\n{lets_x.replace('                ', '            ')}"
        f"            {name}({', '.join(call_args)})\n        }}, &p);\n"
        f"        let s = try_value_and_grad(|x| {{\n            let xm = {xm};\n{lets_x.replace('                ', '            ')}"
        f"            let mut acc = x[0] * 0.0;\n"
        f"            for {idx} in 0..{nobs} {{\n                let r = {row};\n                {eta}\n                acc += {term};\n            }}\n"
        f"            Ok(acc)\n        }}, &p);\n"
        f"        compare(\"{label}\", v, s, &mut failures);\n    }}\n"
    )


HEADER = """//! Every `_glm_lpdf` and `_glm_lpmf` signature of the Stan language, compared with the sum of the
//! response distribution over the observations.
//!
//! Written by `tools/reference/gen_glm.py` from the CmdStan 2.40.0 signature dump: {total} Stan
//! signatures. `matrix x` is 3 by 2; `row_vector x` is 1 by 2 and shared by every observation;
//! `vector x` is 3 by 1. A real response is a one element slice shared by every observation. Each
//! GLM must equal the sum over the observations of its response distribution at
//! `x[i] beta + alpha[i]`, in value and gradient with respect to every real argument; the response
//! distributions are compared with Stan Math by `stan_reference.rs`.
//!
//! Stan Math 5.4.0 differs in one case: `poisson_log_glm_lpmf` with an int response and several
//! rows of `x` adds `-lgamma(y + 1)` once rather than once per observation. stanmath-rs returns the
//! sum.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::identity_op,
    clippy::erasing_op,
    clippy::needless_range_loop
)]

use stanmath_rs_autodiff::{{ValueAndGrad, Var, try_value_and_grad}};
use stanmath_rs_core::{{Matrix, Result}};
use stanmath_rs_distributions::*;

fn close(a: f64, b: f64) -> bool {{
    a == b || (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}}

fn compare(label: &str, v: Result<ValueAndGrad>, s: Result<ValueAndGrad>, failures: &mut Vec<String>) {{
    match (v, s) {{
        (Ok(v), Ok(s))
            if close(v.value, s.value)
                && v.gradient.len() == s.gradient.len()
                && v.gradient.iter().zip(&s.gradient).all(|(a, b)| close(*a, *b)) => {{}}
        (v, s) => failures.push(format!("{{label}}: {{v:?}} vs {{s:?}}")),
    }}
}}

"""

main()
