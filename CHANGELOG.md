# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `stanmath-rs-core`: `Error` with distinct parameter and observation variants; column-major
  `Matrix<T>`; `f64` special functions (log domain helpers, gamma family, polygamma, beta, error
  functions, normal CDF, log CDF, and inverse).
- `stanmath-rs-autodiff`: `Real` scalar trait; forward mode `Dual<T>`; reverse mode `Var` and `Tape`
  with custom operations; gradient, Jacobian, Hessian, HVP, JVP, and VJP drivers; reusable
  `GradientWorkspace`; differentiable elementary and special functions, including incomplete gamma
  and beta with shape derivatives; broadcasting traits for vectorized functions.
- `stanmath-rs-distributions`: 22 univariate distributions with log densities, log CDFs, log
  complementary CDFs, and samplers.
- `stanmath-rs-linalg`: matrix products, Cholesky decomposition, triangular solves, SPD solves,
  inverses, and log determinants, quadratic forms, all with structured reverse-mode adjoints
  through the new `Real::map_multi` hook.
- `stanmath-rs-distributions`: `dirichlet_lpdf`, `multi_normal_lpdf`, `multi_normal_cholesky_lpdf`,
  `lkj_corr_cholesky_lpdf`.
- `stanmath-rs-transforms`: bounds, offset and multiplier, ordered, positive ordered, sum to zero,
  simplex, unit vector, correlation and covariance Cholesky factors, covariance matrices.
- `stanmath-rs-solvers`: `solve_root`, `solve_system`, `integrate`, with derivatives of solutions exact
  to the order of the scalar type.
- `Tape::stats`; criterion benchmarks; NUTS integration test with `nuts-rs`.
- Reference fixtures from mpmath and Stan Math 5.4.0, with the generators in `tools/`.
- Examples, CI workflow, and documentation.
- Licensed under the GNU General Public License, version 3 only.
