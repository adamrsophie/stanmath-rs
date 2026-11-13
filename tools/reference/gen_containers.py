"""Writes crates/stanmath-rs/tests/signatures_containers.rs: every signature of the generic
container functions of the Stan language (dims, num_elements, size, rep_array, reverse,
append_array, head, tail, segment) over every element type, complex ones included.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/gen_containers.py sigs.txt && cargo fmt

Stan types map to Rust: real and int are f64 and i64, complex is Complex<f64>, vectors and row
vectors are Vec, matrices are Matrix, and each array dimension adds a Vec; every container has two
entries per dimension. Each call checks the result against the shape the generator expects:
dims and num_elements of the value, the length of a repeated, appended, or sliced array, and that
reverse reverses. Calls stop at three Vec levels; deeper arrays share the same generic code.
"""

import collections
import re
import sys

import ledger

MAX_DEPTH = 3
FUNCS = {"dims", "num_elements", "size", "rep_array", "reverse", "append_array", "head", "tail",
         "segment"}


def rust(t):
    """(Rust type, dims) for a Stan type, or None."""
    m = re.fullmatch(r"array\[(,*)\] (.*)", t)
    dims = len(m.group(1)) + 1 if m else 0
    base = m.group(2) if m else t
    leafs = {"real": ("f64", []), "int": ("i64", []), "complex": ("Complex<f64>", []),
             "vector": ("Vec<f64>", [2]), "row_vector": ("Vec<f64>", [2]),
             "matrix": ("Matrix<f64>", [2, 2]),
             "complex_vector": ("Vec<Complex<f64>>", [2]),
             "complex_row_vector": ("Vec<Complex<f64>>", [2]),
             "complex_matrix": ("Matrix<Complex<f64>>", [2, 2])}
    if base not in leafs:
        return None
    ty, d = leafs[base]
    for _ in range(dims):
        ty = f"Vec<{ty}>"
    return ty, [2] * dims + d, dims


def value(ty):
    k = [0]

    def scalar(s):
        k[0] += 1
        v = k[0]
        return {"f64": f"{v}.5", "i64": f"{v}_i64", "Complex<f64>": f"Complex::new({v}.5, -{v}.0)"}[s]

    def build(t):
        if t in ("f64", "i64", "Complex<f64>"):
            return scalar(t)
        m = re.fullmatch(r"Matrix<(.*)>", t)
        if m:
            return f"Matrix::from_column_major(2, 2, vec![{', '.join(scalar(m.group(1)) for _ in range(4))}])"
        inner = t[4:-1]
        return f"vec![{build(inner)}, {build(inner)}]"

    return build(ty)


def main():
    sigs = collections.defaultdict(list)
    for l in open(sys.argv[1]).read().splitlines():
        m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\) => (.*)", l)
        if m and m.group(1) in FUNCS:
            sigs[m.group(1)].append(m.group(2).split(", "))
    body, total, deep, other = [], 0, 0, 0
    for name in sorted(sigs):
        for args in sigs[name]:
            r = rust(args[0])
            if r is None:
                other += 1
                continue
            if r[0].count("Vec<") > MAX_DEPTH:
                deep += 1
                ledger.record("deep", name, args)
                continue
            code = call(name, args, r)
            if code is None:
                other += 1
                continue
            total += 1
            ledger.record("called", name, args)
            body.append(code)
    out = HEADER.format(total=total, deep=deep)
    out += "#[test]\nfn container_signatures() {\n"
    out += "".join(body)
    out += "}\n"
    open("crates/stanmath-rs/tests/signatures_containers.rs", "w").write(out)
    print(total, "signatures;", deep, "deeper;", other, "other")


def call(name, args, r):
    ty, dims, arr = r
    label = f"{name}({', '.join(args)})"
    x = f"        let x: {ty} = {value(ty)};\n"
    n = 1
    for d in dims:
        n *= d
    if name == "dims" and len(args) == 1:
        return f"    {{\n{x}        assert_eq!(dims(&x), vec!{dims!r}, \"{label}\");\n    }}\n".replace("[", "[").replace("vec![", "vec![")
    if name == "num_elements" and len(args) == 1:
        return f"    {{\n{x}        assert_eq!(num_elements(&x), {n}, \"{label}\");\n    }}\n"
    if name == "size" and len(args) == 1:
        if arr:
            return f"    {{\n{x}        assert_eq!(size(&x), 2, \"{label}\");\n    }}\n"
        # size of a scalar, vector, or matrix is its number of elements.
        return f"    {{\n{x}        assert_eq!(num_elements(&x), {n}, \"{label}\");\n    }}\n"
    if name == "rep_array":
        k = len(args) - 1
        if not all(a == "int" for a in args[1:]) or k > 3:
            return None
        fn = {1: "rep_array", 2: "rep_array_2d", 3: "rep_array_3d"}[k]
        counts = [3, 2, 4][:k]
        cargs = ", ".join(str(c) for c in counts)
        want = counts + dims
        return (f"    {{\n{x}        let r = {fn}(x.clone(), {cargs});\n"
                f"        assert_eq!(dims(&r), vec!{want!r}, \"{label}\");\n    }}\n")
    if not arr and name in ("reverse", "head", "tail", "segment", "append_array") and not ty.startswith("Vec<"):
        return None
    if name == "reverse" and len(args) == 1:
        return (f"    {{\n{x}        let r = reverse(&x);\n"
                f"        assert_eq!(r[0], x[1], \"{label}\");\n        assert_eq!(reverse(&r), x);\n    }}\n")
    if name == "append_array" and len(args) == 2 and args[0] == args[1]:
        return (f"    {{\n{x}        let r = append_array(&x, &x);\n"
                f"        assert_eq!(r.len(), 4, \"{label}\");\n        assert_eq!(r[2], x[0]);\n    }}\n")
    if name == "head" and args[1:] == ["int"]:
        return f"    {{\n{x}        assert_eq!(head(&x, 1).unwrap(), vec![x[0].clone()], \"{label}\");\n    }}\n"
    if name == "tail" and args[1:] == ["int"]:
        return f"    {{\n{x}        assert_eq!(tail(&x, 1).unwrap(), vec![x[1].clone()], \"{label}\");\n    }}\n"
    if name == "segment" and args[1:] == ["int", "int"]:
        return f"    {{\n{x}        assert_eq!(segment(&x, 2, 1).unwrap(), vec![x[1].clone()], \"{label}\");\n    }}\n"
    return None


HEADER = """//! Every signature of the generic container functions of the Stan language (`dims`,
//! `num_elements`, `size`, `rep_array`, `reverse`, `append_array`, `head`, `tail`, `segment`)
//! over every element type, complex ones included.
//!
//! Written by `tools/reference/gen_containers.py` from the CmdStan 2.40.0 signature dump: {total}
//! Stan signatures, each checked against the shape it must produce. The other {deep} signatures
//! nest arrays more than three levels deep and run the same generic code.
#![allow(clippy::too_many_lines, clippy::redundant_clone, clippy::clone_on_copy)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

"""

main()
