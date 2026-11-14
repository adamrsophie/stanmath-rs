"""Writes crates/stanmath-rs-distributions/tests/signatures_rng.rs: every vectorized `_rng`
signature in the Stan language, as its Rust call through apply_broadcast1 to apply_broadcast4.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_rng.py sigs.txt && cargo fmt

Arguments of type vector, row_vector, or a one dimensional array are slices of three elements,
so the vector, row vector, and array forms of one argument share one Rust call. Each call must
return the draws of the scalar sampler at each index, drawn in order from the same seed. The scalar
samplers are checked against their CDFs or moments by samplers.rs; their parameters here are the
base points of the Stan reference fixtures.
"""

import collections
import re
import sys

import ledger

SIMPLE = {"int", "real", "vector", "row_vector", "array[] int", "array[] real"}
DATA = "crates/stanmath-rs-distributions/tests/data"
EXTRA = {"discrete_range": ([1, 5], []), "hypergeometric": ([5, 4, 6], []), "yule_simon": ([], [2.5]),
         "wiener": ([], [2.0, 0.3, 0.5, 0.8])}
N = 3


def points():
    out = {}
    for f in ("stan_reference.csv", "stan_reference_extra.csv"):
        for line in open(f"{DATA}/{f}"):
            if line.startswith("#"):
                continue
            name, ints, reals = line.split(",")[:3]
            m = re.fullmatch(r"(.*)_lp([dm])f", name)
            if not m or m.group(1) in out:
                continue
            ints = [int(v) for v in ints.split(";") if v]
            reals = [float(v) for v in reals.split(";") if v]
            # Drop the outcome, the first argument.
            if m.group(2) == "m":
                ints = ints[1:]
            else:
                reals = reals[1:]
            out[m.group(1)] = (ints, reals)
    out.update(EXTRA)
    return out


def main():
    fam = collections.defaultdict(list)
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"([A-Za-z_0-9]+)_rng\((.*)\) => (.*)", l)
        if m:
            fam[m.group(1)].append(m.group(2).split(", ") if m.group(2) else [])
    pts = points()
    body, skipped, total = [], [], 0
    for base in sorted(fam):
        sigs = fam[base]
        if not all(t in SIMPLE for s in sigs for t in s) or not sigs[0]:
            skipped.append(base + "_rng")
            continue
        k = len(sigs[0])
        cont = [[t not in ("int", "real") for t in (s[p] for s in sigs)] for p in range(k)]
        vec = [s for s in sigs if any(t not in ("int", "real") for t in s)]
        if not vec or any(all(c) for c in cont) or base not in pts:
            skipped.append(base + "_rng")
            continue
        is_int = [all(s[p] in ("int", "array[] int") for s in sigs) for p in range(k)]
        ints, reals = pts[base]
        if len(ints) != sum(is_int) or len(reals) != k - sum(is_int):
            skipped.append(base + "_rng")
            continue
        shapes = collections.Counter(tuple(t not in ("int", "real") for t in s) for s in vec)
        total += len(vec)
        for s in sigs:
            ledger.record("called", base + "_rng", s)
        for shape in sorted(shapes):
            body.append(call(base + "_rng", shape, is_int, ints, reals))
    used = sorted({int(m) for m in re.findall(r"apply_broadcast(\d)\(", "".join(body))})
    imports = ", ".join(f"apply_broadcast{k}" for k in used)
    out = HEADER.format(total=total, shapes=len(body), skipped=", ".join(skipped), imports=imports)
    out += "#[test]\nfn vectorized_rng_signatures() {\n    let mut failures = Vec::new();\n"
    out += "".join(body)
    out += '    assert!(failures.is_empty(), "\\n{}", failures.join("\\n"));\n}\n'
    open("crates/stanmath-rs-distributions/tests/signatures_rng.rs", "w").write(out)
    print(total, "signatures,", len(body), "Rust shapes; skipped", skipped)


def call(name, shape, is_int, ints, reals):
    k = len(shape)
    ii, ri = iter(ints), iter(reals)
    lets, args, scal, vars_ = [], [], [], []
    for p, c in enumerate(shape):
        v = next(ii) if is_int[p] else next(ri)
        var = "abcd"[p]
        vars_.append(var)
        if is_int[p]:
            elems = [f"{v}_i64"] * N
            scalar = f"{v}_i64"
        else:
            elems = [repr(round(v * (1 + 0.02 * i), 12)) for i in range(N)]
            scalar = repr(float(v))
        if c:
            lets.append(f"        let {var} = [{', '.join(elems)}];\n")
            args.append(f"&{var}[..]")
            scal.append(f"{var}[i]")
        else:
            args.append(scalar)
            scal.append(scalar)
    n = N if any(shape) else 1
    params = ", ".join(vars_)
    cargs = ", ".join(f"{v} as _" for v in vars_)
    sargs = ", ".join(f"{s} as _" for s in scal)
    label = f"{name}({', '.join('vec' if c else 'scalar' for c in shape)})"
    return (
        "    {\n" + "".join(lets)
        + "        let mut rng = StdRng::seed_from_u64(7);\n"
        + f"        let got = apply_broadcast{k}(\"{name}\", {', '.join(args)}, |{params}| {name}({cargs}, &mut rng));\n"
        + "        let mut rng = StdRng::seed_from_u64(7);\n"
        + f"        let want: Result<Vec<_>> = (0..{n}).map(|i| {name}({sargs}, &mut rng)).collect();\n"
        + f"        if format!(\"{{got:?}}\") != format!(\"{{want:?}}\") {{\n"
        + f"            failures.push(format!(\"{label}: {{got:?}} vs {{want:?}}\"));\n        }}\n"
        + "    }\n"
    )


HEADER = """//! Every vectorized `_rng` signature of the Stan language, through `apply_broadcast1` to
//! `apply_broadcast4`.
//!
//! Written by `tools/reference/gen_rng.py` from the CmdStan 2.40.0 signature dump: {total} Stan
//! signatures in {shapes} Rust shapes (vector, row vector, and array forms of one argument are the
//! same Rust slice). Each call must return the scalar sampler's draws at each index, in order from
//! the same seed; the scalar samplers are checked against their CDFs or moments by `samplers.rs`.
//!
//! Not vectorized in Stan, or with container parameters: {skipped}.
#![cfg(feature = "rand")]
#![allow(clippy::too_many_lines, clippy::unreadable_literal, clippy::unnecessary_cast)]

use rand::SeedableRng;
use rand::rngs::StdRng;
use stanmath_rs_autodiff::functions::{{{imports}}};
use stanmath_rs_core::Result;
use stanmath_rs_distributions::*;

"""

main()
