"""Writes crates/stanmath-rs/tests/signatures_gp.rs: every signature of the stationary Gaussian
process kernels in the Stan language (gp_exp_quad_cov, gp_matern32_cov, gp_matern52_cov,
gp_exponential_cov, gp_periodic_cov), each compared with the kernel formula.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_gp.py sigs.txt && cargo fmt

Stan's forms map to Rust: `array[] real` points go through gp_points_1d, a second point set is the
`_cross` function, and an `array[] real` length scale (one per dimension) is the `_ard` function.
Each entry must equal sigma^2 k(d) with d the Euclidean distance between the points after dividing
each coordinate by its length scale.
"""

import re
import sys

import ledger

KERNELS = {
    "gp_exp_quad_cov": "(-d * d / 2.0).exp()",
    "gp_matern32_cov": "(1.0 + 3f64.sqrt() * d) * (-(3f64.sqrt()) * d).exp()",
    "gp_matern52_cov": "(1.0 + 5f64.sqrt() * d + 5.0 * d * d / 3.0) * (-(5f64.sqrt()) * d).exp()",
    "gp_exponential_cov": "(-d).exp()",
}


def main():
    blocks, total = [], 0
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"(gp_(?:exp_quad|matern32|matern52|exponential|periodic)_cov)\((.*)\) => matrix", l)
        if not m:
            continue
        name, args = m.group(1), m.group(2).split(", ")
        ledger.record("called", name, args)
        total += 1
        blocks.append(call(name, args))
    out = HEADER.format(total=total) + "#[test]\nfn gp_signatures() {\n" + "".join(blocks) + "}\n"
    open("crates/stanmath-rs/tests/signatures_gp.rs", "w").write(out)
    print(total, "signatures")


def call(name, args):
    label = f"{name}({', '.join(args)})"
    one_d = args[0] == "array[] real"
    cross = len(args) > 1 and args[1] == args[0]
    rest = args[2:] if cross else args[1:]
    ard = name != "gp_periodic_cov" and rest[1] == "array[] real"
    if one_d:
        pts = ["[0.3, -0.8, 1.4]", "[0.9, -0.1]"]
        conv = lambda v: f"gp_points_1d(&{v})"
    else:
        pts = ["vec![vec![0.3, -0.8], vec![1.4, 0.2], vec![-0.5, 0.6]]", "vec![vec![0.9, -0.1], vec![0.0, 1.1]]"]
        conv = lambda v: v
    lines = [f"    // stan: {label}\n    {{\n", f"        let x1 = {conv(pts[0])};\n"]
    lines.append(f"        let x2 = {conv(pts[1]) if cross else 'x1.clone()'};\n")
    sigma = "1.3"
    if name == "gp_periodic_cov":
        f = f"{name}_cross(&x1, &x2, {sigma}, 0.7, 2.5)" if cross else f"{name}(&x1, {sigma}, 0.7, 2.5)"
        kern = "(-2.0 * (std::f64::consts::PI * d / 2.5).sin().powi(2) / (0.7 * 0.7)).exp()"
        scale = "[1.0, 1.0]"
    elif ard:
        f = f"{name}_ard_cross(&x1, &x2, {sigma}, &l)" if cross else f"{name}_ard(&x1, {sigma}, &l)"
        kern = KERNELS[name]
        scale = "l"
        lines.append(f"        let l = {'[0.7]' if one_d else '[0.7, 1.6]'};\n")
    else:
        f = f"{name}_cross(&x1, &x2, {sigma}, 0.7)" if cross else f"{name}(&x1, {sigma}, 0.7)"
        kern = KERNELS[name]
        scale = "[0.7, 0.7]"
    lines += [
        f"        let got = {f}.unwrap();\n",
        f"        let scale = {scale};\n",
        "        assert_eq!((got.nrows(), got.ncols()), (x1.len(), x2.len()));\n",
        "        for i in 0..x1.len() {\n            for j in 0..x2.len() {\n",
        "                let d = x1[i].iter().zip(&x2[j]).zip(&scale).map(|((a, b), s)| ((a - b) / s).powi(2)).sum::<f64>().sqrt();\n",
        f"                let want = {sigma} * {sigma} * {kern};\n",
        f"                assert!((got[(i, j)] - want).abs() <= 1e-12, \"{label} ({{i}}, {{j}}): {{}} vs {{want}}\", got[(i, j)]);\n",
        "            }\n        }\n    }\n",
    ]
    return "".join(lines)


HEADER = """//! Every signature of the stationary Gaussian process kernels in the Stan language, compared with
//! the kernel formula.
//!
//! Written by `tools/reference/gen_gp.py` from the CmdStan 2.40.0 signature dump: {total} Stan
//! signatures. `array[] real` points go through `gp_points_1d`, a second point set is the `_cross`
//! function, and one length scale per dimension is the `_ard` function. Each entry must equal
//! `sigma^2 k(d)`, with `d` the distance between the points after dividing each coordinate by its
//! length scale (the periodic kernel uses the plain distance).
#![allow(clippy::too_many_lines, clippy::unreadable_literal, clippy::needless_range_loop)]

use stanmath_rs::prelude::*;

"""

main()
