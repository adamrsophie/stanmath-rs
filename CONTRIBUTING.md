# Contributing

Thank you for helping. stanmath-rs values correctness over breadth: a new function is complete when
it is tested against an independent reference in every differentiation mode.

## Development checks

Run before every pull request:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo clippy --workspace --all-targets --no-default-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
```

Do not weaken a failing test to make it pass. If a tolerance must be looser than a few ulps, state
the reason next to it and in `docs/numerical-accuracy.md`.

## Adding a function

1. Add the `f64` value to `stanmath-rs-core::special` (or use an existing one).
2. Add a primitive in `stanmath-rs-autodiff` implementing `UnaryFn`, `BinaryFn`, or `NaryFn`. Write the
   derivative against `Real`, not `f64`, so that nested modes differentiate it. Check that the rule
   does not cancel when differentiated again.
3. Add reference points to `tools/reference/*.py`, regenerate with
   `uv run --with mpmath python tools/reference/<script>.py`, and add the function to the dispatch
   in `crates/stanmath-rs-autodiff/tests/reference.rs`.
4. Update `docs/compatibility.md`.

## Adding a distribution

1. Implement a `Kernel` in `stanmath-rs-distributions`: argument checks that mirror the Stan Math
   function, and a log density with analytic partials, generic over `Real`.
2. Add CDFs by composing differentiable special functions; choose the tail that does not cancel.
3. Add cases to `tools/stan-reference/stan_reference.cpp` and `tools/reference/distributions.py`,
   regenerate both fixtures, and add the dispatch in `tests/stan_reference.rs`.
4. Add the identities in `tests/identities.rs` (CDF derivative, normalization).

Regenerating the Stan fixture needs CmdStan or Stan Math:

```sh
cd tools/stan-reference
make STAN_MATH=$CMDSTAN/stan/lib/stan_math/ stan_reference
./stan_reference > ../../crates/stanmath-rs-distributions/tests/data/stan_reference.csv
```

(prepend the provenance comment lines as in the existing file).

## Style

* Safe Rust only; every crate forbids `unsafe`.
* American English spelling.
* No dashes as sentence punctuation in prose, comments, or documentation.
* Public items are documented; the build denies missing documentation warnings in CI.
