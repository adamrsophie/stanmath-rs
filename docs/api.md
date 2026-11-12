# API reference

The rustdoc reference for every crate is published with this site:

* [`probgrad`](api/probgrad/index.html): umbrella crate and prelude
* [`probgrad_autodiff`](api/probgrad_autodiff/index.html): `Real`, `Dual`, `Var`, `Tape`, drivers,
  and differentiable functions
* [`probgrad_core`](api/probgrad_core/index.html): errors, matrices, and `f64` special functions
* [`probgrad_distributions`](api/probgrad_distributions/index.html): probability distributions
* [`probgrad_linalg`](api/probgrad_linalg/index.html): linear algebra
* [`probgrad_transforms`](api/probgrad_transforms/index.html): constraining transforms
* [`probgrad_solvers`](api/probgrad_solvers/index.html): roots, systems, and quadrature

To build it locally, run `cargo doc --workspace --no-deps --all-features --open`.
