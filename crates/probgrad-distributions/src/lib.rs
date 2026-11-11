//! Differentiable probability distributions.
//!
//! Function names and parameterizations follow Stan Math: `normal_lpdf(y, mu, sigma)` is the log
//! density, `normal_lcdf` and `normal_lccdf` the log CDF and log complementary CDF, and
//! `normal_rng` draws a value (with the default `rand` feature).
//!
//! # Arguments
//!
//! Real arguments can be scalars, slices, `&Vec`s, or array references, and may mix `f64` data
//! with any differentiable scalar ([`Var`](probgrad_autodiff::Var),
//! [`Dual`](probgrad_autodiff::Dual)). Sequence arguments must have equal lengths, scalars
//! broadcast, and the result is the sum over elements. Integer arguments use `i64`.
//!
//! ```
//! use probgrad_autodiff::{grad, Real};
//! use probgrad_distributions::normal_lpdf;
//!
//! let y = [1.2, 0.4, 2.1];
//! let g = grad(|p| normal_lpdf(&y[..], p[0], p[1]).unwrap(), &[1.0, 0.8]);
//! assert_eq!(g.len(), 2);
//! ```
//!
//! # Errors and boundaries
//!
//! Invalid parameters return [`Error::InvalidParameter`](probgrad_core::Error) and observations
//! outside the support return [`Error::InvalidObservation`](probgrad_core::Error), following the
//! checks of the corresponding Stan Math function. Where Stan returns `-inf` instead of an error
//! (for example `lognormal_lpdf` at `y = 0`), so does probgrad; each function documents its cases.
//!
//! # Performance
//!
//! Log densities are fused: the sum over all elements is recorded as one reverse-mode node whose
//! partials are computed analytically, generically in the scalar type, so Hessians remain exact.

mod cdf;
mod check;
mod continuous;
mod discrete;
mod kernel;
mod multivariate;
#[cfg(feature = "rand")]
mod rng;

pub use cdf::*;
pub use continuous::{
    beta_lpdf, cauchy_lpdf, exponential_lpdf, gamma_lpdf, inv_gamma_lpdf, logistic_lpdf,
    lognormal_lpdf, normal_lpdf, std_normal_lpdf, student_t_lpdf, uniform_lpdf, weibull_lpdf,
};
pub use discrete::{
    bernoulli_logit_lpmf, bernoulli_lpmf, binomial_logit_lpmf, binomial_lpmf,
    categorical_logit_lpmf, categorical_lpmf, neg_binomial_2_log_lpmf, neg_binomial_2_lpmf,
    poisson_log_lpmf, poisson_lpmf,
};
pub use multivariate::{
    dirichlet_lpdf, lkj_corr_cholesky_lpdf, multi_normal_cholesky_lpdf, multi_normal_lpdf,
};
#[cfg(feature = "rand")]
pub use rng::*;
