# API reference

The rustdoc reference for every crate is published with this site:

* [`stanmath-rs`](api/stanmath_rs/index.html): umbrella crate and prelude
* [`stanmath_rs_autodiff`](api/stanmath_rs_autodiff/index.html): `Real`, `Dual`, `Var`, `Tape`, drivers,
  and differentiable functions
* [`stanmath_rs_core`](api/stanmath_rs_core/index.html): errors, matrices, and `f64` special functions
* [`stanmath_rs_distributions`](api/stanmath_rs_distributions/index.html): probability distributions
* [`stanmath_rs_linalg`](api/stanmath_rs_linalg/index.html): linear algebra
* [`stanmath_rs_transforms`](api/stanmath_rs_transforms/index.html): constraining transforms
* [`stanmath_rs_solvers`](api/stanmath_rs_solvers/index.html): roots, systems, and quadrature

To build it locally, run `cargo doc --workspace --no-deps --all-features --open`.
