"""Records which lines of the stanc signature dump a generated test calls.

When SIG_LEDGER names a file, each generator appends one line per Stan signature it handles:
`called<TAB>name(args)` when the generated test calls that exact signature, or
`deep<TAB>name(args)` when it skips the signature because the arrays nest deeper than it writes
calls for. tools/reference/coverage.py runs every generator this way and reports the rest.
"""

import os


def record(kind, name, args):
    path = os.environ.get("SIG_LEDGER")
    if path:
        with open(path, "a") as f:
            f.write(f"{kind}\t{name}({', '.join(args)})\n")
