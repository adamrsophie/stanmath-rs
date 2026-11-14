//! Cumulative distribution functions on the probability scale (`*_cdf`).
//!
//! Stan's `*_cdf` returns the product of the per-element CDFs. Each function here is the
//! exponential of the matching `*_lcdf`, which sums the per-element log CDFs, so both take the
//! same arguments, apply the same checks, and differentiate through the same special functions.

use stanmath_rs_autodiff::{Broadcast, IntArg, Promoted, Real};
use stanmath_rs_core::Result;

use crate::cdf::{
    bernoulli_lcdf, beta_lcdf, binomial_lcdf, cauchy_lcdf, exponential_lcdf, gamma_lcdf,
    inv_gamma_lcdf, logistic_lcdf, lognormal_lcdf, neg_binomial_2_lcdf, normal_lcdf, poisson_lcdf,
    std_normal_lcdf, student_t_lcdf, uniform_lcdf, weibull_lcdf,
};
use crate::continuous_extra::{
    chi_square_lcdf, double_exponential_lcdf, exp_mod_normal_lcdf, frechet_lcdf, gumbel_lcdf,
    inv_chi_square_lcdf, loglogistic_lcdf, pareto_lcdf, pareto_type_2_lcdf, rayleigh_lcdf,
    scaled_inv_chi_square_lcdf, skew_double_exponential_lcdf, skew_normal_lcdf, von_mises_lcdf,
};
use crate::discrete_extra::{
    beta_binomial_lcdf, beta_neg_binomial_lcdf, discrete_range_lcdf, neg_binomial_lcdf,
    poisson_binomial_lcdf,
};

/// A CDF over real arguments as the exponential of the log CDF with the same arguments.
macro_rules! plain_cdf {
    ($cdf:ident, $lcdf:ident, $n:literal, ($($arg:ident: $ty:ident),+)) => {
        #[doc = concat!(
            "CDF on the probability scale: the product over elements of the per-element CDFs, ",
            "computed as the exponential of [`", stringify!($lcdf), "`], which takes the same ",
            "arguments and applies the same checks."
        )]
        pub fn $cdf<$($ty),+>($($arg: $ty),+) -> Result<Promoted<($($ty,)+), $n>>
        where
            ($($ty,)+): Broadcast<$n>,
        {
            $lcdf($($arg),+).map(|l| l.exp())
        }
    };
}

plain_cdf!(std_normal_cdf, std_normal_lcdf, 1, (y: Y));
plain_cdf!(normal_cdf, normal_lcdf, 3, (y: Y, mu: M, sigma: S));
plain_cdf!(cauchy_cdf, cauchy_lcdf, 3, (y: Y, mu: M, sigma: S));
plain_cdf!(logistic_cdf, logistic_lcdf, 3, (y: Y, mu: M, sigma: S));
plain_cdf!(student_t_cdf, student_t_lcdf, 4, (y: Y, nu: N, mu: M, sigma: S));
plain_cdf!(exponential_cdf, exponential_lcdf, 2, (y: Y, beta: B));
plain_cdf!(gamma_cdf, gamma_lcdf, 3, (y: Y, alpha: A, beta: B));
plain_cdf!(inv_gamma_cdf, inv_gamma_lcdf, 3, (y: Y, alpha: A, beta: B));
plain_cdf!(beta_cdf, beta_lcdf, 3, (y: Y, alpha: A, beta: B));
plain_cdf!(lognormal_cdf, lognormal_lcdf, 3, (y: Y, mu: M, sigma: S));
plain_cdf!(uniform_cdf, uniform_lcdf, 3, (y: Y, alpha: A, beta: B));
plain_cdf!(weibull_cdf, weibull_lcdf, 3, (y: Y, alpha: A, sigma: S));
plain_cdf!(
    skew_normal_cdf,
    skew_normal_lcdf,
    4,
    (y: Y, xi: X, omega: O, alpha: A)
);
plain_cdf!(
    double_exponential_cdf,
    double_exponential_lcdf,
    3,
    (y: Y, mu: M, sigma: S)
);
plain_cdf!(von_mises_cdf, von_mises_lcdf, 3, (y: Y, mu: M, kappa: K));
plain_cdf!(rayleigh_cdf, rayleigh_lcdf, 2, (y: Y, sigma: S));
plain_cdf!(pareto_cdf, pareto_lcdf, 3, (y: Y, y_min: M, alpha: A));
plain_cdf!(
    pareto_type_2_cdf,
    pareto_type_2_lcdf,
    4,
    (y: Y, mu: M, lambda: L, alpha: A)
);
plain_cdf!(frechet_cdf, frechet_lcdf, 3, (y: Y, alpha: A, sigma: S));
plain_cdf!(gumbel_cdf, gumbel_lcdf, 3, (y: Y, mu: M, beta: B));
plain_cdf!(chi_square_cdf, chi_square_lcdf, 2, (y: Y, nu: N));
plain_cdf!(inv_chi_square_cdf, inv_chi_square_lcdf, 2, (y: Y, nu: N));
plain_cdf!(
    scaled_inv_chi_square_cdf,
    scaled_inv_chi_square_lcdf,
    3,
    (y: Y, nu: N, s: S)
);
plain_cdf!(
    exp_mod_normal_cdf,
    exp_mod_normal_lcdf,
    4,
    (y: Y, mu: M, sigma: S, lambda: L)
);
plain_cdf!(
    skew_double_exponential_cdf,
    skew_double_exponential_lcdf,
    4,
    (y: Y, mu: M, sigma: S, tau: T)
);
plain_cdf!(loglogistic_cdf, loglogistic_lcdf, 3, (y: Y, alpha: A, beta: B));

// Discrete distributions: integer outcomes, so the signatures are written out.

/// Bernoulli CDF, the exponential of [`bernoulli_lcdf`].
pub fn bernoulli_cdf<N, T>(n: N, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    (T,): Broadcast<1>,
{
    bernoulli_lcdf(n, theta).map(|l| l.exp())
}

/// Binomial CDF, the exponential of [`binomial_lcdf`].
pub fn binomial_cdf<N, M, T>(n: N, trials: M, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    M: IntArg,
    (T,): Broadcast<1>,
{
    binomial_lcdf(n, trials, theta).map(|l| l.exp())
}

/// Poisson CDF, the exponential of [`poisson_lcdf`].
pub fn poisson_cdf<N, L>(n: N, lambda: L) -> Result<Promoted<(L,), 1>>
where
    N: IntArg,
    (L,): Broadcast<1>,
{
    poisson_lcdf(n, lambda).map(|l| l.exp())
}

/// Negative binomial (mean, dispersion) CDF, the exponential of [`neg_binomial_2_lcdf`].
pub fn neg_binomial_2_cdf<N, M, P>(n: N, mu: M, phi: P) -> Result<Promoted<(M, P), 2>>
where
    N: IntArg,
    (M, P): Broadcast<2>,
{
    neg_binomial_2_lcdf(n, mu, phi).map(|l| l.exp())
}

/// Negative binomial (shape, inverse scale) CDF, the exponential of [`neg_binomial_lcdf`].
pub fn neg_binomial_cdf<N, A, B>(n: N, alpha: A, beta: B) -> Result<Promoted<(A, B), 2>>
where
    N: IntArg,
    (A, B): Broadcast<2>,
{
    neg_binomial_lcdf(n, alpha, beta).map(|l| l.exp())
}

/// Beta binomial CDF, the exponential of [`beta_binomial_lcdf`].
pub fn beta_binomial_cdf<N, M, A, B>(
    n: N,
    trials: M,
    alpha: A,
    beta: B,
) -> Result<Promoted<(A, B), 2>>
where
    N: IntArg,
    M: IntArg,
    (A, B): Broadcast<2>,
{
    beta_binomial_lcdf(n, trials, alpha, beta).map(|l| l.exp())
}

/// Beta negative binomial CDF, the exponential of [`beta_neg_binomial_lcdf`].
pub fn beta_neg_binomial_cdf<N, R, A, B>(
    n: N,
    r: R,
    alpha: A,
    beta: B,
) -> Result<Promoted<(R, A, B), 3>>
where
    N: IntArg,
    (R, A, B): Broadcast<3>,
{
    beta_neg_binomial_lcdf(n, r, alpha, beta).map(|l| l.exp())
}

/// Discrete uniform CDF, the exponential of [`discrete_range_lcdf`].
pub fn discrete_range_cdf<Y, L, U>(y: Y, lower: L, upper: U) -> Result<f64>
where
    Y: IntArg,
    L: IntArg,
    U: IntArg,
{
    discrete_range_lcdf(y, lower, upper).map(f64::exp)
}

/// Poisson binomial CDF, the exponential of [`poisson_binomial_lcdf`].
pub fn poisson_binomial_cdf<N, T>(y: N, theta: &[T]) -> Result<T>
where
    N: IntArg,
    T: Real,
{
    poisson_binomial_lcdf(y, theta).map(|l| l.exp())
}
