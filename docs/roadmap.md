# Roadmap

Each milestone ends at a gate: `cargo fmt --check`, `cargo clippy -D warnings` (all and no default
features), all tests, documentation tests, every example, and updated documentation.

## Milestone 1: working foundation (done)

- [x] Cargo workspace, CI, licenses
- [x] `Real` scalar trait; forward mode `Dual<T>`; reverse mode `Var` and `Tape`
- [x] Elementary functions with derivative rules exact to all orders
- [x] Drivers: gradient, Jacobian (both modes), Hessian, Hessian vector product, JVP, VJP
- [x] Gradient checking, reusable workspace for repeated evaluation
- [x] mpmath reference suite, property tests, compile-fail tests for tape safety
- [x] Examples: forward mode, reverse mode, Jacobians and Hessians, gradient checking

## Milestone 2: probability and numerical stability (done)

- [x] Special functions: gamma family, polygamma, beta, error functions, normal CDF and inverse,
      incomplete gamma and beta with derivatives in every argument
- [x] 22 univariate distributions with log densities, log CDFs, and samplers
- [x] Fused log density kernels (one tape node per vectorized likelihood)
- [x] Stan Math 5.4.0 reference harness; mpmath adjudication of disagreements
- [x] Examples: normal log likelihood, logistic regression

## Milestone 3: advanced differentiation and linear algebra (done, timings pending)

- [x] `stanmath-rs-linalg`: products, Cholesky, triangular solves, log determinants, quadratic forms,
      with structured reverse-mode adjoints through `Real::map_multi`
- [x] Criterion benchmarks and a Stan Math timing harness
- [ ] Record timings on an idle machine (`docs/performance.md`)
- [ ] Vector-valued forward mode (deferred until a benchmark shows Hessians matter)

## Milestone 4: Bayesian model infrastructure (done)

- [x] `stanmath-rs-transforms`: bounds, offset and multiplier, ordered, positive ordered, sum to zero,
      simplex, unit vector, correlation and covariance Cholesky factors, covariance matrices
- [x] Dirichlet, multivariate normal (covariance and Cholesky), LKJ correlation Cholesky
- [x] Integration test with NUTS from `nuts-rs`
- [ ] `corr_matrix` with Stan's partial correlation order

## Milestone 5: scientific computation (in progress)

- [x] Scalar root finding with derivatives exact to every order carried by the scalar type
- [x] Newton solver for nonlinear systems with exact derivatives
- [x] One-dimensional tanh-sinh quadrature with parameter derivatives
- [ ] ODE solutions with forward sensitivities on an existing Rust integrator
- [ ] Adjoint sensitivities
- [ ] Derivatives with respect to integration limits

## Later

- Remaining Stan Math scalar functions and distributions (see `docs/compatibility.md`)
- `propto` variants that drop constant terms
- Sparse matrix operations
- Bindings for R and Python
