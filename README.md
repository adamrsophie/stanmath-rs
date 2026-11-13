# stanmath-rs

Differentiable mathematics for Bayesian computation in Rust: automatic differentiation, special
functions, and probability distributions, with Stan Math as the specification and correctness
baseline.

```rust
use stanmath_rs::prelude::*;

fn main() -> Result<()> {
    let observation = 1.3;
    let r = try_value_and_grad(|p| normal_lpdf(observation, p[0], p[1]), &[0.5, 1.2])?;
    println!("log density {} gradient {:?}", r.value, r.gradient);
    Ok(())
}
```

## Status

stanmath-rs is at version 0.1, under active development, and not yet published to crates.io.
Automatic differentiation, special functions, univariate and multivariate distributions, dense
linear algebra, parameter transforms, root finding, nonlinear systems, and quadrature are
implemented and tested. ODE solvers are next, and benchmark timings still have to be recorded on an
idle machine.
See [`docs/roadmap.md`](docs/roadmap.md).

Documentation, including the API reference, is at <https://adamrsophie.github.io/stanmath-rs/>.

## What it provides

* **Automatic differentiation.** Forward mode (`Dual<T>`), reverse mode (`Var` on a `Tape`), and any
  nesting of them. Drivers for gradients, Jacobians, Hessians, Hessian vector products, JVPs, and
  VJPs. Derivative rules are written generically, so second and higher derivatives are exact.
* **Memory safety without `unsafe`.** Every crate forbids `unsafe`. Variables borrow their tape, so
  the compiler rejects use after a tape is cleared or dropped, and closures passed to the drivers
  cannot capture variables from another tape.
* **Special functions** with derivatives in every argument: gamma, digamma, polygamma, beta, error
  functions, normal CDF and its inverse, regularized incomplete gamma and beta.
* **22 probability distributions** with Stan's names, parameterizations, and argument checks:
  vectorized log densities, log CDFs, log complementary CDFs, and samplers.
* **Multivariate distributions**: Dirichlet, multivariate normal (covariance and Cholesky), LKJ.
* **Linear algebra** with structured reverse-mode adjoints: products, Cholesky, triangular solves,
  log determinants, quadratic forms. No dense Jacobians, and an `n` by `n` product adds `n^2` tape
  nodes.
* **Constraining transforms** with log Jacobians, following Stan Math 5.4: bounds, ordered,
  simplex, sum to zero, unit vector, correlation and covariance Cholesky factors, covariance
  matrices.
* **Solvers** with exact derivatives of their solutions: scalar roots, nonlinear systems, and
  one-dimensional integrals (finite or infinite limits).
* **Fused likelihoods.** A vectorized log density over `n` observations records one reverse-mode
  node with analytic partials, like Stan's `operands_and_partials`.
* **Sampler ready.** `GradientWorkspace` matches the `logp(position, gradient)` interface of NUTS
  implementations; an integration test samples stanmath-rs models with `nuts-rs`.

## Installation

Until the first release, depend on the repository:

```toml
[dependencies]
stanmath-rs = { git = "https://github.com/adamrsophie/stanmath-rs" }
```

The minimum supported Rust version is 1.85 (edition 2024); the library crates are checked with
`cargo +1.85 check --workspace --lib --all-features`. Tests and benchmarks use newer development
dependencies and are run on stable. The `rand` feature (on by default)
enables the `*_rng` functions.

## Usage

Write a model once, generically over `T: Real`, and evaluate it in any mode:

```rust
use stanmath_rs::prelude::*;

fn log_posterior<T: Real>(p: &[T], y: &[f64]) -> Result<T> {
    let (mu, log_sigma) = (p[0], p[1]);
    let sigma = log_sigma.exp();
    Ok(normal_lpdf(y, mu, sigma)? + normal_lpdf(mu, 0.0, 10.0)? + log_sigma)
}

fn main() -> Result<()> {
    let y = [2.3, 1.9, 3.1];
    // Reverse mode.
    let g = try_grad(|p| log_posterior(p, &y), &[2.0, 0.0])?;
    // Forward over reverse.
    let h = hessian(|p| log_posterior(p, &y).unwrap(), &[2.0, 0.0]);
    println!("{g:?} {:?}", h.to_rows());
    Ok(())
}
```

Reverse-mode drivers take closures (`|p| model(p)`) because the closure must accept variables from
a tape created inside the driver. Forward-mode drivers accept generic functions directly.

Runnable examples are in [`crates/stanmath-rs/examples`](crates/stanmath-rs/examples):

| Example | Shows |
| --- | --- |
| `forward_mode` | derivatives, second derivatives, directional derivatives, JVP |
| `reverse_mode` | explicit tape, closure drivers, workspace reuse, VJP |
| `jacobian_hessian` | Jacobians in both modes, Hessians, Hessian vector products |
| `gradient_check` | comparing a gradient with finite differences |
| `normal_log_likelihood` | the example from the Stan Math paper, priors, Hessian |
| `logistic_regression` | simulated data, posterior mode by gradient ascent, Laplace approximation |
| `multivariate_normal` | correlation Cholesky factor, scales, LKJ prior, posterior mode |
| `transforms` | constraining transforms, log Jacobians, round trips |
| `solvers` | derivatives of a root, a nonlinear system, and an integral |

Run one with `cargo run --example logistic_regression`.

## Architecture

| Crate | Contents |
| --- | --- |
| `stanmath-rs-core` | `Error`, `Matrix<T>` container, `f64` special functions |
| `stanmath-rs-autodiff` | `Real`, `Dual`, `Var`, `Tape`, drivers, differentiable functions |
| `stanmath-rs-linalg` | matrix operations with structured adjoints |
| `stanmath-rs-distributions` | log densities, CDFs, samplers |
| `stanmath-rs-transforms` | constraining transforms with log Jacobians |
| `stanmath-rs-solvers` | root finding, nonlinear systems, quadrature |
| `stanmath-rs` | umbrella crate and prelude |

Design decisions and the alternatives considered are in
[`docs/architecture.md`](docs/architecture.md).

## Correctness and testing

* Every elementary and special function is checked in five modes (value, forward, reverse, forward
  over reverse, forward over forward) against mpmath references computed at 60 to 320 digits.
* Every distribution is checked against Stan Math 5.4.0 output. Where stanmath-rs and Stan disagree,
  a test requires stanmath-rs to be closer to the high-precision value; it is in all 45 such cases,
  including a 1.6% error in Stan's `gamma_lccdf` shape gradient and a missing term in its LKJ
  shape gradient at `eta = 1`.
* Identities tie the pieces together: CDF derivatives equal densities, densities integrate to one,
  Hessian modes agree, samplers have the right moments.

Details and tolerances: [`docs/numerical-accuracy.md`](docs/numerical-accuracy.md). Feature
coverage relative to Stan Math: [`docs/compatibility.md`](docs/compatibility.md).

```sh
cargo test --workspace
```

## Performance

Criterion benchmarks and a matching Stan Math timing harness exist, but timings have not yet been
recorded on an idle machine, so no speed claims are made. Measured tape sizes and the benchmark
method are in [`docs/performance.md`](docs/performance.md).

## Relationship to Stan Math

Stan Math is the specification: function names, parameterizations, argument checks, and boundary
behavior follow it, and its output is a test reference. No Stan Math code is translated; the
implementation is independent Rust. stanmath-rs is not affiliated with the Stan project.

## License

Copyright (C) 2026 Adam R Sophie.

stanmath-rs is free software: you can redistribute it and/or modify it under the terms of the
[GNU General Public License, version 3](LICENSE), as published by the Free Software Foundation.
It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the license for
details.

## Acknowledgments

Stan Math (BSD 3-clause) for the mathematical specification and reference values; mpmath for
high-precision references; the `libm` crate for `lgamma`, `erf`, and `erfc`; `rand` and
`rand_distr` for sampling.
