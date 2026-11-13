//! Differentiable probability distributions.
//!
//! Function names and parameterizations follow Stan Math: `normal_lpdf(y, mu, sigma)` is the log
//! density, `normal_lcdf` and `normal_lccdf` the log CDF and log complementary CDF, and
//! `normal_rng` draws a value (with the default `rand` feature).
//!
//! # Arguments
//!
//! Real arguments can be scalars, slices, `&Vec`s, or array references, and may mix `f64` data
//! with any differentiable scalar ([`Var`](stanmath_rs_autodiff::Var),
//! [`Dual`](stanmath_rs_autodiff::Dual)). Sequence arguments must have equal lengths, scalars
//! broadcast, and the result is the sum over elements. Integer arguments use `i64`.
//!
//! ```
//! use stanmath_rs_autodiff::{grad, Real};
//! use stanmath_rs_distributions::normal_lpdf;
//!
//! let y = [1.2, 0.4, 2.1];
//! let g = grad(|p| normal_lpdf(&y[..], p[0], p[1]).unwrap(), &[1.0, 0.8]);
//! assert_eq!(g.len(), 2);
//! ```
//!
//! # Errors and boundaries
//!
//! Invalid parameters return [`Error::InvalidParameter`](stanmath_rs_core::Error) and observations
//! outside the support return [`Error::InvalidObservation`](stanmath_rs_core::Error), following the
//! checks of the corresponding Stan Math function. Where Stan returns `-inf` instead of an error
//! (for example `lognormal_lpdf` at `y = 0`), so does stanmath-rs; each function documents its cases.
//!
//! # Performance
//!
//! Log densities are fused: the sum over all elements is recorded as one reverse-mode node whose
//! partials are computed analytically, generically in the scalar type, so Hessians remain exact.

/// Defines a log density `$full` and its `propto` twin `$lup` (Stan's `_lupdf` or `_lupmf`) from
/// one body. In the body, `$p` is `false` for `$full` and `true` for `$lup`; with `base(f, g)`, the
/// alias names `f` in `$full` and `g` in `$lup`. Bounds go in the `where` clause.
macro_rules! propto_pair {
    (
        $(#[$doc:meta])*
        $full:ident, $lup:ident <$($g:ident),*> ($($a:ident: $t:ty),* $(,)?) -> $ret:ty
        where { $($w:tt)* }
        |$p:ident| $body:block
    ) => {
        $(#[$doc])*
        pub fn $full<$($g),*>($($a: $t),*) -> $ret
        where
            $($w)*
        {
            let $p = false;
            $body
        }

        #[doc = concat!(
            "[`", stringify!($full), "`] without the terms that depend only on data (Stan's `",
            stringify!($lup), "`)."
        )]
        pub fn $lup<$($g),*>($($a: $t),*) -> $ret
        where
            $($w)*
        {
            let $p = true;
            $body
        }
    };
    (
        $(#[$doc:meta])*
        $full:ident, $lup:ident <$($g:ident),*> ($($a:ident: $t:ty),* $(,)?) -> $ret:ty
        where { $($w:tt)* }
        base($bf:path, $bl:path) |$alias:ident| $body:block
    ) => {
        $(#[$doc])*
        pub fn $full<$($g),*>($($a: $t),*) -> $ret
        where
            $($w)*
        {
            let $alias = $bf;
            $body
        }

        #[doc = concat!(
            "[`", stringify!($full), "`] without the terms that depend only on data (Stan's `",
            stringify!($lup), "`)."
        )]
        pub fn $lup<$($g),*>($($a: $t),*) -> $ret
        where
            $($w)*
        {
            let $alias = $bl;
            $body
        }
    };
}

mod cdf;
mod cdf_plain;
mod check;
mod continuous;
mod continuous_extra;
mod discrete;
mod discrete_extra;
mod glm;
mod hmm;
mod kernel;
mod misc_dist;
mod multivariate;
mod multivariate_extra;
mod overloads;
#[cfg(feature = "rand")]
mod rng;
#[cfg(feature = "rand")]
mod rng_discrete_extra;
#[cfg(feature = "rand")]
mod rng_extra;
#[cfg(feature = "rand")]
mod rng_multivariate;
#[cfg(feature = "rand")]
mod rng_overloads;
#[cfg(feature = "rand")]
mod rng_wave2;
mod wiener_full;
mod yule_simon;

pub use cdf::*;
pub use cdf_plain::*;
pub use continuous::{
    beta_lpdf, beta_lupdf, cauchy_lpdf, cauchy_lupdf, exponential_lpdf, exponential_lupdf,
    gamma_lpdf, gamma_lupdf, inv_gamma_lpdf, inv_gamma_lupdf, logistic_lpdf, logistic_lupdf,
    lognormal_lpdf, lognormal_lupdf, normal_lpdf, normal_lupdf, std_normal_lpdf, std_normal_lupdf,
    student_t_lpdf, student_t_lupdf, uniform_lpdf, uniform_lupdf, weibull_lpdf, weibull_lupdf,
};
pub use continuous_extra::{
    beta_proportion_lccdf, beta_proportion_lcdf, beta_proportion_lpdf, beta_proportion_lupdf,
    chi_square_lccdf, chi_square_lcdf, chi_square_lpdf, chi_square_lupdf, double_exponential_lccdf,
    double_exponential_lcdf, double_exponential_lpdf, double_exponential_lupdf,
    exp_mod_normal_lccdf, exp_mod_normal_lcdf, exp_mod_normal_lpdf, exp_mod_normal_lupdf,
    frechet_lccdf, frechet_lcdf, frechet_lpdf, frechet_lupdf, gumbel_lccdf, gumbel_lcdf,
    gumbel_lpdf, gumbel_lupdf, inv_chi_square_lccdf, inv_chi_square_lcdf, inv_chi_square_lpdf,
    inv_chi_square_lupdf, loglogistic_lccdf, loglogistic_lcdf, loglogistic_lpdf, loglogistic_lupdf,
    pareto_lccdf, pareto_lcdf, pareto_lpdf, pareto_lupdf, pareto_type_2_lccdf, pareto_type_2_lcdf,
    pareto_type_2_lpdf, pareto_type_2_lupdf, rayleigh_lccdf, rayleigh_lcdf, rayleigh_lpdf,
    rayleigh_lupdf, scaled_inv_chi_square_lccdf, scaled_inv_chi_square_lcdf,
    scaled_inv_chi_square_lpdf, scaled_inv_chi_square_lupdf, skew_double_exponential_lccdf,
    skew_double_exponential_lcdf, skew_double_exponential_lpdf, skew_double_exponential_lupdf,
    skew_normal_lccdf, skew_normal_lcdf, skew_normal_lpdf, skew_normal_lupdf, von_mises_lccdf,
    von_mises_lcdf, von_mises_lpdf, von_mises_lupdf, wiener_lpdf, wiener_lupdf,
};
pub use discrete::{
    bernoulli_logit_lpmf, bernoulli_logit_lupmf, bernoulli_lpmf, bernoulli_lupmf,
    binomial_logit_lpmf, binomial_logit_lupmf, binomial_lpmf, binomial_lupmf,
    categorical_logit_lpmf, categorical_logit_lupmf, categorical_lpmf, categorical_lupmf,
    neg_binomial_2_log_lpmf, neg_binomial_2_log_lupmf, neg_binomial_2_lpmf, neg_binomial_2_lupmf,
    poisson_log_lpmf, poisson_log_lupmf, poisson_lpmf, poisson_lupmf,
};
pub use discrete_extra::*;
pub use glm::*;
pub use hmm::*;
pub use misc_dist::{normal_sufficient_lpdf, normal_sufficient_lupdf, student_t_qf};
pub use multivariate::{
    dirichlet_lpdf, dirichlet_lupdf, lkj_corr_cholesky_lpdf, lkj_corr_cholesky_lupdf,
    multi_normal_cholesky_lpdf, multi_normal_cholesky_lupdf, multi_normal_lpdf, multi_normal_lupdf,
};
pub use multivariate_extra::*;
pub use overloads::*;
#[cfg(feature = "rand")]
pub use rng::*;
#[cfg(feature = "rand")]
pub use rng_discrete_extra::*;
#[cfg(feature = "rand")]
pub use rng_extra::*;
#[cfg(feature = "rand")]
pub use rng_multivariate::*;
#[cfg(feature = "rand")]
pub use rng_overloads::*;
#[cfg(feature = "rand")]
pub use rng_wave2::*;
pub use wiener_full::{
    WienerArgs, WienerOut, wiener_full_lccdf_unnorm, wiener_full_lcdf_unnorm, wiener_full_lpdf,
    wiener_full_lupdf, wiener_full_prec_lccdf_unnorm, wiener_full_prec_lcdf_unnorm,
    wiener_full_prec_lpdf, wiener_full_prec_lupdf, wiener_lccdf_unnorm, wiener_lcdf_unnorm,
    wiener4_lccdf_unnorm, wiener4_lcdf_unnorm, wiener5_lpdf, wiener5_lupdf,
};
pub use yule_simon::{
    yule_simon_cdf, yule_simon_lccdf, yule_simon_lcdf, yule_simon_lpmf, yule_simon_lupmf,
};
