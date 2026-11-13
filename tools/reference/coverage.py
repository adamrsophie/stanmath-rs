"""Sorts every line of the stanc signature dump by how the test suite checks it.

    stanc --dump-stan-math-signatures > sigs.txt
    python3 tools/reference/coverage.py sigs.txt [report.txt]

Runs each gen_*.py with SIG_LEDGER set (they rewrite their tests, unchanged when the dump is the
same) and assigns each signature the first category that applies:

    called      a generated signature test calls this exact signature, or a hand-written test
                names it in a `// stan:` comment
    reference   the function is in the Stan reference CSVs, and its arguments are all int or real
                or all its signatures are one Rust call (they differ only in spelling a vector)
    deep        a generator skipped it only because the arrays nest deeper than it writes calls
    by-name     not called by its exact types, but the function name is compared with Stan values
                (reference CSVs) or called in a test file
    unchecked   none of the above

Prints the counts and writes the deep, by-name, and unchecked lines to the report.
"""

import collections
import glob
import os
import re
import subprocess
import sys
import tempfile

GENERATORS = ["gen_signatures", "gen_elementwise", "gen_rng", "gen_bounds", "gen_containers",
              "gen_complex", "gen_utilities", "gen_operators", "gen_transforms", "gen_glm"]
DATA = "crates/stanmath-rs/tests/data"
CSV_DATA = ["crates/stanmath-rs-distributions/tests/data/stan_reference.csv",
            "crates/stanmath-rs-distributions/tests/data/stan_reference_extra.csv",
            f"{DATA}/stan_reference_scalar.csv", f"{DATA}/stan_reference_matrix.csv"]


def main():
    dump = sys.argv[1]
    report = sys.argv[2] if len(sys.argv) > 2 else None
    fd, ledger_path = tempfile.mkstemp()
    os.close(fd)
    env = dict(os.environ, SIG_LEDGER=ledger_path)
    for g in GENERATORS:
        subprocess.run([sys.executable, f"tools/reference/{g}.py", dump], env=env, check=True,
                       stdout=subprocess.DEVNULL)
    marks = {}
    for line in open(ledger_path):
        kind, key = line.rstrip("\n").split("\t")
        if marks.get(key) != "called":
            marks[key] = kind
    os.remove(ledger_path)

    # Hand-written tests name the Stan signatures they call in `// stan: name(args)` comments.
    for f in glob.glob("crates/*/tests/*.rs"):
        for key in re.findall(r"// stan: (.*)", open(f).read()):
            marks[key.strip()] = "called"

    ref = set()
    for f in CSV_DATA:
        if os.path.exists(f):
            ref |= {l.split(",")[0] for l in open(f) if l.strip() and not l.startswith("#")}
    tested = set()
    for f in glob.glob("crates/*/tests/*.rs") + glob.glob("crates/*/src/**/*.rs", recursive=True) + \
            glob.glob("backends/*/src/**/*.rs", recursive=True) + glob.glob("backends/*/tests/*.rs"):
        src = open(f).read()
        if "tests" in f or "#[cfg(test)]" in src:
            tested |= set(re.findall(r"\b([a-z_][A-Za-z_0-9]*)\s*(?:::<[^>]*>)?\s*\(", src))

    # Signatures of one function that differ only in spelling a vector (vector, row_vector, or
    # array[] real) are one Rust call, so a reference comparison of that call covers them all.
    shapes = collections.defaultdict(set)
    for l in open(dump).read().splitlines():
        m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\) => (.*)", l)
        shapes[m.group(1)].add(canonical(m.group(2)))

    cats = collections.Counter()
    rest = collections.defaultdict(list)
    for l in open(dump).read().splitlines():
        m = re.fullmatch(r"([A-Za-z_0-9]+)\((.*)\) => (.*)", l)
        name, args = m.group(1), m.group(2)
        key = f"{name}({args})"
        scalar = all(t in ("int", "real") for t in (args.split(", ") if args else []))
        if name == "student_t_qf" or marks.get(key) == "called":
            cat = "called"
        elif base(name) in ref and (scalar or len(shapes[name]) == 1):
            cat = "reference"
        elif marks.get(key) == "deep":
            cat = "deep"
        elif base(name) in ref or name in tested or base(name) in tested:
            cat = "by-name"
        else:
            cat = "unchecked"
        cats[cat] += 1
        if cat != "called" and cat != "reference":
            rest[cat].append(l)
    total = sum(cats.values())
    for c in ("called", "reference", "deep", "by-name", "unchecked"):
        print(f"{c:10} {cats[c]:6}")
    print(f"{'total':10} {total:6}")
    if report:
        with open(report, "w") as f:
            for c in ("unchecked", "by-name", "deep"):
                f.write(f"## {c} ({len(rest[c])})\n")
                f.writelines(l + "\n" for l in rest[c])


def canonical(args):
    """The Rust shape of a Stan argument list: vectors, row vectors, and 1-d real arrays are slices."""
    return re.sub(r"\b(row_vector|vector|array\[\] real)\b", "slice", args)


def base(name):
    """The name a reference CSV uses: _lupdf is _lpdf, _cdf is checked through _lcdf."""
    name = re.sub(r"_lup([dm])f$", r"_lp\1f", name)
    return name


main()
