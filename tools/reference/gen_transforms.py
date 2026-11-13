"""Writes crates/stanmath-rs/tests/signatures_transforms.rs: every array signature of the
structural transforms in the Stan language (simplex, sum_to_zero, ordered, positive_ordered,
unit_vector, stochastic_row, stochastic_column, cov_matrix, corr_matrix, cholesky_factor_corr,
cholesky_factor_cov; each with _constrain, _jacobian, and _unconstrain).

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_transforms.py sigs.txt && cargo fmt

Stan's array forms apply the transform to each element; in Rust they are an iterator map over the
array (a _jacobian map adds every element's log Jacobian to one lp). Each call must keep the
array's shape and equal the transform of every element in order, and its lp the sum of theirs.
Calls stop at three array levels; deeper arrays are the same nested map. The single transforms
are compared with Stan Math by stan_reference_matrix.rs.
"""

import collections
import re
import sys

import ledger

MAX_DEPTH = 3
# Rust name stem, free length, extra size arguments, input kind ("vec" or "mat"), and whether
# constrain, jacobian, and unconstrain return a Result.
FAMILIES = {
    "simplex": ("simplex", 3, "", "vec", (False, False, True)),
    "sum_to_zero": ("sum_to_zero", 3, "", "vec", (False, False, True)),
    "sum_to_zero_matrix": ("sum_to_zero_matrix", 4, "", "mat", (False, False, True)),
    "ordered": ("ordered", 3, "", "vec", (False, False, True)),
    "positive_ordered": ("positive_ordered", 3, "", "vec", (False, False, True)),
    "unit_vector": ("unit_vector", 3, "", "vec", (True, True, True)),
    "stochastic_row": ("stochastic_row", 4, "", "mat", (False, False, True)),
    "stochastic_column": ("stochastic_column", 4, "", "mat", (False, False, True)),
    "cov_matrix": ("cov_matrix", 6, ", 3", "vec", (True, True, True)),
    "corr_matrix": ("corr_matrix", 3, ", 3", "vec", (True, True, True)),
    "cholesky_factor_corr": ("cholesky_factor_corr", 3, ", 3", "vec", (True, True, True)),
    "cholesky_factor_cov": ("cholesky_factor_cov", 5, ", 3, 2", "vec", (True, True, True)),
}
NAMES = re.compile(r"(%s)_(constrain|jacobian|unconstrain)" % "|".join(
    sorted((k for k in FAMILIES if k != "sum_to_zero_matrix"), key=len, reverse=True)))


def family(stem, args):
    """The FAMILIES key: sum_to_zero over matrices is sum_to_zero_matrix in Rust."""
    if stem == "sum_to_zero" and "matrix" in args[0]:
        return "sum_to_zero_matrix"
    return stem


def depth(t):
    m = re.fullmatch(r"array\[(,*)\] .*", t)
    return len(m.group(1)) + 1 if m else 0


def main():
    calls = collections.defaultdict(list)
    deep = 0
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\) => (.*)", l)
        if not m:
            continue
        n = NAMES.fullmatch(m.group(1))
        args = m.group(2).split(", ")
        if not n or depth(args[0]) == 0:
            continue
        d = depth(args[0])
        if d > MAX_DEPTH:
            deep += 1
            ledger.record("deep", m.group(1), args)
            continue
        ledger.record("called", m.group(1), args)
        calls[(family(n.group(1), args), n.group(2), d)].append(m.group(1))
    body = [call(f, op, d) for (f, op, d) in sorted(calls)]
    total = sum(len(v) for v in calls.values())
    out = HEADER.format(total=total, shapes=len(body), deep=deep, depth=MAX_DEPTH)
    out += "#[test]\nfn transform_array_signatures() {\n    let mut failures = Vec::new();\n"
    out += "".join(body)
    out += '    assert!(failures.is_empty(), "\\n{}", failures.join("\\n"));\n}\n'
    open("crates/stanmath-rs/tests/signatures_transforms.rs", "w").write(out)
    print(total, "signatures,", len(body), "Rust shapes;", deep, "deeper")


def nest(d, leaf):
    """A nested vec! of depth d whose leaves are leaf(i) for i = 0, 1, ..."""
    k = [0]

    def build(level):
        if level == 0:
            k[0] += 1
            return leaf(k[0] - 1)
        return f"vec![{build(level - 1)}, {build(level - 1)}]"

    return build(d)


def mapped(d, f):
    """The nested map of f over an array of depth d named x."""
    inner = f("v")
    for level in range(d):
        var = "x" if level == d - 1 else f"a{d - level - 1}"
        arg = "v" if level == 0 else f"a{d - level}"
        inner = f"{var}.iter().map(|{arg}| {inner}).collect::<Vec<_>>()"
    return inner


def call(fam, op, d):
    stem, n, sizes, kind, result = FAMILIES[fam]
    unwrap = [".unwrap()" if r else "" for r in result]

    def free(i):
        vals = ", ".join(repr(round(0.3 * (j + 1) * (-1) ** (i + j) + 0.1 * i, 6)) for j in range(n))
        return f"Matrix::from_column_major(2, 2, vec![{vals}])" if kind == "mat" else f"vec![{vals}]"

    con = lambda a: f"{stem}_constrain({a}{sizes}){unwrap[0]}"
    if op == "constrain":
        f = lambda a: con(a)
    elif op == "jacobian":
        f = lambda a: f"{stem}_jacobian({a}{sizes}, &mut lp){unwrap[1]}"
    else:
        f = lambda a: f"{stem}_unconstrain({a}){unwrap[2]}"
    leaf = (lambda i: con("&" + free(i))) if op == "unconstrain" else free
    label = f"{fam}_{op}({'array ' * d}{kind})"
    flat = "got.into_iter()" + ".flatten()" * (d - 1)
    flat = f"{flat}.collect::<Vec<_>>()" if d > 1 else flat
    lines = [
        "    {\n",
        f"        let x = {nest(d, leaf)};\n",
        f"        let flat_x: Vec<_> = vec![{', '.join(leaf(i) for i in range(2 ** d))}];\n",
    ]
    if op == "jacobian":
        lines += [
            "        let mut lp = 0.0;\n",
            f"        let got = {mapped(d, f)};\n",
            "        let got_lp = lp;\n        let mut lp = 0.0;\n",
            f"        let want: Vec<_> = flat_x.iter().map(|v| {f('v')}).collect();\n",
            f"        if (got_lp - lp).abs() > 1e-12 {{\n            failures.push(format!(\"{label} lp: {{got_lp}} vs {{lp}}\"));\n        }}\n",
        ]
    else:
        lines += [
            f"        let got = {mapped(d, f)};\n",
            f"        let want: Vec<_> = flat_x.iter().map(|v| {f('v')}).collect();\n",
        ]
    lines += [
        f"        let got = {flat};\n" if d > 1 else "",
        f"        if format!(\"{{got:?}}\") != format!(\"{{want:?}}\") {{\n",
        f"            failures.push(format!(\"{label}: {{got:?}} vs {{want:?}}\"));\n        }}\n",
        "    }\n",
    ]
    return "".join(lines)


HEADER = """//! Every array signature of the structural transforms in the Stan language (`_constrain`,
//! `_jacobian`, and `_unconstrain` of simplex, sum_to_zero, ordered, positive_ordered,
//! unit_vector, stochastic_row, stochastic_column, cov_matrix, corr_matrix, cholesky_factor_corr,
//! and cholesky_factor_cov).
//!
//! Written by `tools/reference/gen_transforms.py` from the CmdStan 2.40.0 signature dump: {total}
//! Stan signatures in {shapes} Rust shapes. Stan's array forms apply the transform to each element;
//! in Rust they are an iterator map over the array, and a `_jacobian` map adds every element's log
//! Jacobian to one `lp`. Each call must keep the array's shape and equal the transform of every
//! element in order, and its `lp` the sum of theirs; the single transforms are compared with Stan
//! Math by `stan_reference_matrix.rs`. The other {deep} signatures nest arrays more than {depth}
//! deep and are the same nested map.
#![allow(clippy::too_many_lines, clippy::unreadable_literal, clippy::redundant_closure, clippy::useless_vec)]

use stanmath_rs::prelude::*;

"""

main()
