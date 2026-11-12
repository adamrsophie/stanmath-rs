# stanmath-rs

Differentiable mathematics for Bayesian computation in Rust: automatic differentiation, special
functions, and probability distributions, with Stan Math as the specification and correctness
baseline.

stanmath-rs is at version 0.1, under active development, and not yet published to crates.io. Source
code, issues, and the changelog are on [GitHub](https://github.com/adamrsophie/stanmath-rs).

```rust
use stanmath_rs::prelude::*;

fn main() -> Result<()> {
    let observation = 1.3;
    let r = try_value_and_grad(|p| normal_lpdf(observation, p[0], p[1]), &[0.5, 1.2])?;
    println!("log density {} gradient {:?}", r.value, r.gradient);
    Ok(())
}
```

## What is implemented

* Forward mode (`Dual<T>`), reverse mode (`Var` on a `Tape`), and any nesting of them, with drivers
  for gradients, Jacobians, Hessians, Hessian vector products, JVPs, and VJPs.
* Special functions with derivatives in every argument, including the regularized incomplete gamma
  and beta functions.
* 22 univariate distributions with Stan's names, parameterizations, and argument checks, plus
  Dirichlet, multivariate normal, and LKJ.
* Dense linear algebra with structured reverse-mode adjoints.
* Stan's constraining transforms with log Jacobians.
* Root finding, nonlinear systems, and one-dimensional quadrature, with derivatives of their
  solutions.

## Installation

Until the first release, depend on the repository:

```toml
[dependencies]
stanmath-rs = { git = "https://github.com/adamrsophie/stanmath-rs" }
```

The minimum supported Rust version is 1.85 (edition 2024).

## Guide

* [Architecture](architecture.md): the `Real` trait, the tape, broadcasting, fused kernels, and
  how derivatives of iterative algorithms are computed.
* [Compatibility with Stan Math](compatibility.md): what is implemented, what differs, and the
  measured differences from Stan Math 5.4.0.
* [Numerical accuracy](numerical-accuracy.md): reference values, tolerances, and fixes.
* [Performance](performance.md): tape sizes and the benchmark setup. No timing claims yet.
* [Roadmap](roadmap.md): milestones and what comes next.
* [API reference](api.md): rustdoc for every crate.

stanmath-rs is free software under the GNU General Public License, version 3.
