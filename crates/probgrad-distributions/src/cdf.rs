//! Log cumulative distribution functions (`*_lcdf`) and log complementary CDFs (`*_lccdf`).
//!
//! These are composed from the differentiable special functions, which carry their own
//! derivative rules (including shape derivatives of the incomplete gamma and beta functions).
//! Each tail is computed directly rather than as `ln(1 - F)`, so that both stay accurate where the
//! other is close to zero.

use core::f64::consts::PI;

use probgrad_autodiff::functions::{
    atan2, beta_inc, gamma_p, gamma_q, log_Phi, log_inv_logit, log1m, log1m_exp, log1m_inv_logit,
};
use probgrad_autodiff::{Broadcast, IntArg, Promoted, Real, broadcast_len};
use probgrad_core::{Error, Result};

use crate::check::{obs, param};

/// `ln F` given closures for the lower tail `F` and the upper tail `1 - F`.
///
/// Uses `ln F` when `F <= 1/2` and `ln(1 - (1 - F))` otherwise, so the log of a probability near
/// one never loses the digits of its complement.
fn log_lower<T: Real>(lower: impl FnOnce() -> T, upper: impl FnOnce() -> T) -> T {
    let p = lower();
    if p.value() <= 0.5 {
        p.ln()
    } else {
        log1m(upper())
    }
}

/// `ln(1 - F)` given closures for the lower tail `F` and the upper tail `1 - F`.
fn log_upper<T: Real>(lower: impl FnOnce() -> T, upper: impl FnOnce() -> T) -> T {
    let q = upper();
    if q.value() <= 0.5 {
        q.ln()
    } else {
        log1m(lower())
    }
}

/// Sums `term` over broadcast elements after validating each with `check`.
fn sum_terms<A, const N: usize>(
    name: &'static str,
    args: A,
    ints: &[&dyn IntArg],
    check: impl Fn(&[i64], [f64; N]) -> Result<()>,
    term: impl Fn(&[i64], [Promoted<A, N>; N]) -> Promoted<A, N>,
) -> Result<Promoted<A, N>>
where
    A: Broadcast<N>,
{
    let mut lens = args.lens().to_vec();
    lens.extend(ints.iter().map(|a| a.int_len()));
    let n = broadcast_len(name, &lens)?;
    let mut total = Promoted::<A, N>::zero();
    let mut int_row = vec![0i64; ints.len()];
    for i in 0..n {
        for (slot, a) in int_row.iter_mut().zip(ints) {
            *slot = a.int_at(i);
        }
        let x = args.get(i);
        check(&int_row, x.map(|v| v.value()))?;
        total += term(&int_row, x);
    }
    Ok(total)
}

macro_rules! location_scale_cdfs {
    ($lcdf:ident, $lccdf:ident, $name_l:literal, $name_c:literal, $check:expr,
     |$z:ident| $lower:expr, $upper:expr) => {
        #[doc = concat!("Log CDF; see [`", stringify!($lcdf), "`] for the arguments.")]
        pub fn $lcdf<Y, M, S>(y: Y, mu: M, sigma: S) -> Result<Promoted<(Y, M, S), 3>>
        where
            (Y, M, S): Broadcast<3>,
        {
            sum_terms(
                $name_l,
                (y, mu, sigma),
                &[],
                |_, x| $check($name_l, x),
                |_, [y, mu, sigma]| {
                    let $z = (y - mu) / sigma;
                    $lower
                },
            )
        }

        #[doc = concat!("Log complementary CDF `ln(1 - F)`; see [`", stringify!($lcdf), "`].")]
        pub fn $lccdf<Y, M, S>(y: Y, mu: M, sigma: S) -> Result<Promoted<(Y, M, S), 3>>
        where
            (Y, M, S): Broadcast<3>,
        {
            sum_terms(
                $name_c,
                (y, mu, sigma),
                &[],
                |_, x| $check($name_c, x),
                |_, [y, mu, sigma]| {
                    let $z = (y - mu) / sigma;
                    $upper
                },
            )
        }
    };
}

fn check_location_scale(f: &'static str, [y, mu, sigma]: [f64; 3]) -> Result<()> {
    obs::not_nan(f, "y", y)?;
    param::finite(f, "mu", mu)?;
    param::positive_finite(f, "sigma", sigma)
}

location_scale_cdfs!(
    normal_lcdf,
    normal_lccdf,
    "normal_lcdf",
    "normal_lccdf",
    check_location_scale,
    |z| log_Phi(z),
    log_Phi(-z)
);
location_scale_cdfs!(
    cauchy_lcdf,
    cauchy_lccdf,
    "cauchy_lcdf",
    "cauchy_lccdf",
    check_location_scale,
    // F(z) = 1/2 + atan(z) / pi = atan2(1, -z) / pi, which has no cancellation as z -> -inf.
    |z| atan2(Promoted::<(Y, M, S), 3>::one(), -z).ln() - PI.ln(),
    atan2(Promoted::<(Y, M, S), 3>::one(), z).ln() - PI.ln()
);
location_scale_cdfs!(
    logistic_lcdf,
    logistic_lccdf,
    "logistic_lcdf",
    "logistic_lccdf",
    check_location_scale,
    |z| log_inv_logit(z),
    log1m_inv_logit(z)
);

/// Standard normal log CDF `ln Φ(y)`.
pub fn std_normal_lcdf<Y>(y: Y) -> Result<Promoted<(Y,), 1>>
where
    (Y,): Broadcast<1>,
{
    const F: &str = "std_normal_lcdf";
    sum_terms(
        F,
        (y,),
        &[],
        |_, [y]| obs::not_nan(F, "y", y),
        |_, [y]| log_Phi(y),
    )
}

/// Standard normal log complementary CDF `ln Φ(-y)`.
pub fn std_normal_lccdf<Y>(y: Y) -> Result<Promoted<(Y,), 1>>
where
    (Y,): Broadcast<1>,
{
    const F: &str = "std_normal_lccdf";
    sum_terms(
        F,
        (y,),
        &[],
        |_, [y]| obs::not_nan(F, "y", y),
        |_, [y]| log_Phi(-y),
    )
}

/// `ln F` of Student's t at standardized value `t`.
///
/// The tail beyond `|t|` is `I_x(nu/2, 1/2) / 2` with `x = nu / (nu + t^2)`. When `t^2 < nu`, `x` is
/// close to one; the complement `w = t^2 / (nu + t^2)` is then formed directly and
/// `I_x(nu/2, 1/2) = 1 - I_w(1/2, nu/2)` is used, so that `1 - x` is never computed by subtraction.
fn student_t_log_cdf<T: Real>(t: T, nu: T) -> T {
    let t2 = t.square();
    let half = T::from_f64(0.5);
    if t.value().abs() < 1e-4 * nu.value().sqrt() {
        // Near the center the beta-function form behaves like |t| and is not differentiable at
        // t = 0. Use F(t) = 1/2 + c1 t (1 - (nu + 1) t^2 / (6 nu)) with c1 = 1 / (sqrt(nu) B(nu/2, 1/2)),
        // whose truncation error is O(t^5), below 1e-20 relative in this range.
        let c1 = (-probgrad_autodiff::functions::lbeta(nu * 0.5, half)).exp() / nu.sqrt();
        let f = c1 * t * (-((nu + 1.0) * t2) / (nu * 6.0) + 1.0) + 0.5;
        return f.ln();
    }
    let lower_half = t.value() < 0.0;
    if t2.value() < nu.value() {
        // tail = (1 - J) / 2 with J = I_w(1/2, nu/2)
        let j = beta_inc(half, nu * 0.5, t2 / (nu + t2));
        if lower_half {
            log1m(j) - core::f64::consts::LN_2
        } else {
            j.ln_1p() - core::f64::consts::LN_2
        }
    } else {
        let tail = beta_inc(nu * 0.5, half, nu / (nu + t2)) * 0.5;
        if lower_half { tail.ln() } else { log1m(tail) }
    }
}

fn check_student_t(f: &'static str, [y, nu, mu, sigma]: [f64; 4]) -> Result<()> {
    obs::not_nan(f, "y", y)?;
    param::positive_finite(f, "nu", nu)?;
    param::finite(f, "mu", mu)?;
    param::positive_finite(f, "sigma", sigma)
}

/// Student's t log CDF.
pub fn student_t_lcdf<Y, N, M, S>(y: Y, nu: N, mu: M, sigma: S) -> Result<Promoted<(Y, N, M, S), 4>>
where
    (Y, N, M, S): Broadcast<4>,
{
    const F: &str = "student_t_lcdf";
    sum_terms(
        F,
        (y, nu, mu, sigma),
        &[],
        |_, x| check_student_t(F, x),
        |_, [y, nu, mu, s]| student_t_log_cdf((y - mu) / s, nu),
    )
}

/// Student's t log complementary CDF.
pub fn student_t_lccdf<Y, N, M, S>(
    y: Y,
    nu: N,
    mu: M,
    sigma: S,
) -> Result<Promoted<(Y, N, M, S), 4>>
where
    (Y, N, M, S): Broadcast<4>,
{
    const F: &str = "student_t_lccdf";
    sum_terms(
        F,
        (y, nu, mu, sigma),
        &[],
        |_, x| check_student_t(F, x),
        |_, [y, nu, mu, s]| student_t_log_cdf((mu - y) / s, nu),
    )
}

fn check_exponential(f: &'static str, [y, beta]: [f64; 2]) -> Result<()> {
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "beta", beta)
}

/// Exponential log CDF `ln(1 - exp(-beta y))`.
pub fn exponential_lcdf<Y, B>(y: Y, beta: B) -> Result<Promoted<(Y, B), 2>>
where
    (Y, B): Broadcast<2>,
{
    const F: &str = "exponential_lcdf";
    sum_terms(
        F,
        (y, beta),
        &[],
        |_, x| check_exponential(F, x),
        |_, [y, b]| log1m_exp(-(b * y)),
    )
}

/// Exponential log complementary CDF `-beta y`.
pub fn exponential_lccdf<Y, B>(y: Y, beta: B) -> Result<Promoted<(Y, B), 2>>
where
    (Y, B): Broadcast<2>,
{
    const F: &str = "exponential_lccdf";
    sum_terms(
        F,
        (y, beta),
        &[],
        |_, x| check_exponential(F, x),
        |_, [y, b]| -(b * y),
    )
}

fn check_gamma(f: &'static str, [y, alpha, beta]: [f64; 3]) -> Result<()> {
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "alpha", alpha)?;
    param::positive_finite(f, "beta", beta)
}

/// Gamma (shape, rate) log CDF `ln P(alpha, beta y)`.
pub fn gamma_lcdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    const F: &str = "gamma_lcdf";
    sum_terms(
        F,
        (y, alpha, beta),
        &[],
        |_, x| check_gamma(F, x),
        |_, [y, a, b]| log_lower(|| gamma_p(a, b * y), || gamma_q(a, b * y)),
    )
}

/// Gamma (shape, rate) log complementary CDF `ln Q(alpha, beta y)`.
pub fn gamma_lccdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    const F: &str = "gamma_lccdf";
    sum_terms(
        F,
        (y, alpha, beta),
        &[],
        |_, x| check_gamma(F, x),
        |_, [y, a, b]| log_upper(|| gamma_p(a, b * y), || gamma_q(a, b * y)),
    )
}

/// Inverse gamma (shape, scale) log CDF `ln Q(alpha, beta / y)`.
pub fn inv_gamma_lcdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    const F: &str = "inv_gamma_lcdf";
    sum_terms(
        F,
        (y, alpha, beta),
        &[],
        |_, x| check_gamma(F, x),
        |_, [y, a, b]| log_lower(|| gamma_q(a, b / y), || gamma_p(a, b / y)),
    )
}

/// Inverse gamma (shape, scale) log complementary CDF `ln P(alpha, beta / y)`.
pub fn inv_gamma_lccdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    const F: &str = "inv_gamma_lccdf";
    sum_terms(
        F,
        (y, alpha, beta),
        &[],
        |_, x| check_gamma(F, x),
        |_, [y, a, b]| log_upper(|| gamma_q(a, b / y), || gamma_p(a, b / y)),
    )
}

fn check_beta(f: &'static str, [y, a, b]: [f64; 3]) -> Result<()> {
    obs::unit_interval(f, "y", y)?;
    param::positive_finite(f, "alpha", a)?;
    param::positive_finite(f, "beta", b)
}

/// Beta log CDF `ln I_y(alpha, beta)`.
pub fn beta_lcdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    const F: &str = "beta_lcdf";
    sum_terms(
        F,
        (y, alpha, beta),
        &[],
        |_, x| check_beta(F, x),
        |_, [y, a, b]| log_lower(|| beta_inc(a, b, y), || beta_inc(b, a, -y + 1.0)),
    )
}

/// Beta log complementary CDF `ln I_{1-y}(beta, alpha)`.
pub fn beta_lccdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    const F: &str = "beta_lccdf";
    sum_terms(
        F,
        (y, alpha, beta),
        &[],
        |_, x| check_beta(F, x),
        |_, [y, a, b]| log_upper(|| beta_inc(a, b, y), || beta_inc(b, a, -y + 1.0)),
    )
}

fn check_lognormal(f: &'static str, [y, mu, sigma]: [f64; 3]) -> Result<()> {
    obs::nonnegative(f, "y", y)?;
    param::finite(f, "mu", mu)?;
    param::positive_finite(f, "sigma", sigma)
}

/// Lognormal log CDF; `y = 0` gives `-inf`.
pub fn lognormal_lcdf<Y, M, S>(y: Y, mu: M, sigma: S) -> Result<Promoted<(Y, M, S), 3>>
where
    (Y, M, S): Broadcast<3>,
{
    const F: &str = "lognormal_lcdf";
    sum_terms(
        F,
        (y, mu, sigma),
        &[],
        |_, x| check_lognormal(F, x),
        |_, [y, mu, s]| log_Phi((y.ln() - mu) / s),
    )
}

/// Lognormal log complementary CDF; `y = 0` gives `0`.
pub fn lognormal_lccdf<Y, M, S>(y: Y, mu: M, sigma: S) -> Result<Promoted<(Y, M, S), 3>>
where
    (Y, M, S): Broadcast<3>,
{
    const F: &str = "lognormal_lccdf";
    sum_terms(
        F,
        (y, mu, sigma),
        &[],
        |_, x| check_lognormal(F, x),
        |_, [y, mu, s]| log_Phi((mu - y.ln()) / s),
    )
}

fn check_uniform(f: &'static str, [y, alpha, beta]: [f64; 3]) -> Result<()> {
    obs::not_nan(f, "y", y)?;
    param::finite(f, "alpha", alpha)?;
    param::finite(f, "beta", beta)?;
    if beta > alpha {
        Ok(())
    } else {
        Err(Error::parameter(f, "beta", beta, "greater than alpha"))
    }
}

/// Uniform log CDF.
pub fn uniform_lcdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    const F: &str = "uniform_lcdf";
    sum_terms(
        F,
        (y, alpha, beta),
        &[],
        |_, x| check_uniform(F, x),
        |_, [y, a, b]| {
            if y.value() <= a.value() {
                Promoted::<(Y, A, B), 3>::from_f64(f64::NEG_INFINITY)
            } else if y.value() >= b.value() {
                Promoted::<(Y, A, B), 3>::zero()
            } else {
                ((y - a) / (b - a)).ln()
            }
        },
    )
}

/// Uniform log complementary CDF.
pub fn uniform_lccdf<Y, A, B>(y: Y, alpha: A, beta: B) -> Result<Promoted<(Y, A, B), 3>>
where
    (Y, A, B): Broadcast<3>,
{
    const F: &str = "uniform_lccdf";
    sum_terms(
        F,
        (y, alpha, beta),
        &[],
        |_, x| check_uniform(F, x),
        |_, [y, a, b]| {
            if y.value() <= a.value() {
                Promoted::<(Y, A, B), 3>::zero()
            } else if y.value() >= b.value() {
                Promoted::<(Y, A, B), 3>::from_f64(f64::NEG_INFINITY)
            } else {
                ((b - y) / (b - a)).ln()
            }
        },
    )
}

fn check_weibull(f: &'static str, [y, alpha, sigma]: [f64; 3]) -> Result<()> {
    obs::nonnegative(f, "y", y)?;
    param::positive_finite(f, "alpha", alpha)?;
    param::positive_finite(f, "sigma", sigma)
}

/// Weibull log CDF `ln(1 - exp(-(y / sigma)^alpha))`.
pub fn weibull_lcdf<Y, A, S>(y: Y, alpha: A, sigma: S) -> Result<Promoted<(Y, A, S), 3>>
where
    (Y, A, S): Broadcast<3>,
{
    const F: &str = "weibull_lcdf";
    sum_terms(
        F,
        (y, alpha, sigma),
        &[],
        |_, x| check_weibull(F, x),
        |_, [y, a, s]| log1m_exp(-(y / s).pow(a)),
    )
}

/// Weibull log complementary CDF `-(y / sigma)^alpha`.
pub fn weibull_lccdf<Y, A, S>(y: Y, alpha: A, sigma: S) -> Result<Promoted<(Y, A, S), 3>>
where
    (Y, A, S): Broadcast<3>,
{
    const F: &str = "weibull_lccdf";
    sum_terms(
        F,
        (y, alpha, sigma),
        &[],
        |_, x| check_weibull(F, x),
        |_, [y, a, s]| -(y / s).pow(a),
    )
}

/// Bernoulli log CDF of outcome `n`.
pub fn bernoulli_lcdf<N, T>(n: N, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    (T,): Broadcast<1>,
{
    const F: &str = "bernoulli_lcdf";
    sum_terms(
        F,
        (theta,),
        &[&n],
        |k, [t]| {
            obs::binary(F, "n", k[0])?;
            param::probability(F, "theta", t)
        },
        |k, [t]| {
            if k[0] == 0 {
                log1m(t)
            } else {
                Promoted::<(T,), 1>::zero()
            }
        },
    )
}

/// Bernoulli log complementary CDF `ln P(N > n)`.
pub fn bernoulli_lccdf<N, T>(n: N, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    (T,): Broadcast<1>,
{
    const F: &str = "bernoulli_lccdf";
    sum_terms(
        F,
        (theta,),
        &[&n],
        |k, [t]| {
            obs::binary(F, "n", k[0])?;
            param::probability(F, "theta", t)
        },
        |k, [t]| {
            if k[0] == 0 {
                t.ln()
            } else {
                Promoted::<(T,), 1>::from_f64(f64::NEG_INFINITY)
            }
        },
    )
}

fn check_binomial(f: &'static str, k: &[i64], theta: f64) -> Result<()> {
    let (n, big_n) = (k[0], k[1]);
    if big_n < 0 {
        return Err(Error::parameter(f, "N", big_n as f64, "nonnegative"));
    }
    if !(0..=big_n).contains(&n) {
        return Err(Error::observation(f, "n", n as f64, "in [0, N]"));
    }
    param::probability(f, "theta", theta)
}

/// Binomial log CDF `ln P(X <= n) = ln I_{1-theta}(N - n, n + 1)`.
pub fn binomial_lcdf<N, M, T>(n: N, trials: M, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    M: IntArg,
    (T,): Broadcast<1>,
{
    const F: &str = "binomial_lcdf";
    sum_terms(
        F,
        (theta,),
        &[&n, &trials],
        |k, [t]| check_binomial(F, k, t),
        |k, [t]| {
            let (n, big_n) = (k[0], k[1]);
            if n >= big_n {
                return Promoted::<(T,), 1>::zero();
            }
            let a = Promoted::<(T,), 1>::from_f64((big_n - n) as f64);
            let b = Promoted::<(T,), 1>::from_f64((n + 1) as f64);
            log_lower(|| beta_inc(a, b, -t + 1.0), || beta_inc(b, a, t))
        },
    )
}

/// Binomial log complementary CDF `ln P(X > n) = ln I_theta(n + 1, N - n)`.
pub fn binomial_lccdf<N, M, T>(n: N, trials: M, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    M: IntArg,
    (T,): Broadcast<1>,
{
    const F: &str = "binomial_lccdf";
    sum_terms(
        F,
        (theta,),
        &[&n, &trials],
        |k, [t]| check_binomial(F, k, t),
        |k, [t]| {
            let (n, big_n) = (k[0], k[1]);
            if n >= big_n {
                return Promoted::<(T,), 1>::from_f64(f64::NEG_INFINITY);
            }
            let a = Promoted::<(T,), 1>::from_f64((n + 1) as f64);
            let b = Promoted::<(T,), 1>::from_f64((big_n - n) as f64);
            log_upper(|| beta_inc(b, a, -t + 1.0), || beta_inc(a, b, t))
        },
    )
}

fn check_poisson(f: &'static str, k: &[i64], lambda: f64) -> Result<()> {
    obs::count(f, "n", k[0])?;
    param::nonnegative(f, "lambda", lambda)
}

/// Poisson log CDF `ln P(X <= n) = ln Q(n + 1, lambda)`.
pub fn poisson_lcdf<N, L>(n: N, lambda: L) -> Result<Promoted<(L,), 1>>
where
    N: IntArg,
    (L,): Broadcast<1>,
{
    const F: &str = "poisson_lcdf";
    sum_terms(
        F,
        (lambda,),
        &[&n],
        |k, [l]| check_poisson(F, k, l),
        |k, [l]| {
            let a = Promoted::<(L,), 1>::from_f64((k[0] + 1) as f64);
            log_lower(|| gamma_q(a, l), || gamma_p(a, l))
        },
    )
}

/// Poisson log complementary CDF `ln P(X > n) = ln P(n + 1, lambda)`.
pub fn poisson_lccdf<N, L>(n: N, lambda: L) -> Result<Promoted<(L,), 1>>
where
    N: IntArg,
    (L,): Broadcast<1>,
{
    const F: &str = "poisson_lccdf";
    sum_terms(
        F,
        (lambda,),
        &[&n],
        |k, [l]| check_poisson(F, k, l),
        |k, [l]| {
            let a = Promoted::<(L,), 1>::from_f64((k[0] + 1) as f64);
            log_upper(|| gamma_q(a, l), || gamma_p(a, l))
        },
    )
}

fn check_neg_binomial_2(f: &'static str, k: &[i64], [mu, phi]: [f64; 2]) -> Result<()> {
    obs::count(f, "n", k[0])?;
    param::positive_finite(f, "mu", mu)?;
    param::positive_finite(f, "phi", phi)
}

/// Negative binomial (mean, dispersion) log CDF `ln I_{phi/(mu+phi)}(phi, n + 1)`.
pub fn neg_binomial_2_lcdf<N, M, P>(n: N, mu: M, phi: P) -> Result<Promoted<(M, P), 2>>
where
    N: IntArg,
    (M, P): Broadcast<2>,
{
    const F: &str = "neg_binomial_2_lcdf";
    sum_terms(
        F,
        (mu, phi),
        &[&n],
        |k, x| check_neg_binomial_2(F, k, x),
        |k, [mu, phi]| {
            let b = Promoted::<(M, P), 2>::from_f64((k[0] + 1) as f64);
            let total = mu + phi;
            log_lower(
                || beta_inc(phi, b, phi / total),
                || beta_inc(b, phi, mu / total),
            )
        },
    )
}

/// Negative binomial (mean, dispersion) log complementary CDF
/// `ln I_{mu/(mu+phi)}(n + 1, phi)`.
pub fn neg_binomial_2_lccdf<N, M, P>(n: N, mu: M, phi: P) -> Result<Promoted<(M, P), 2>>
where
    N: IntArg,
    (M, P): Broadcast<2>,
{
    const F: &str = "neg_binomial_2_lccdf";
    sum_terms(
        F,
        (mu, phi),
        &[&n],
        |k, x| check_neg_binomial_2(F, k, x),
        |k, [mu, phi]| {
            let a = Promoted::<(M, P), 2>::from_f64((k[0] + 1) as f64);
            let total = mu + phi;
            log_upper(
                || beta_inc(phi, a, phi / total),
                || beta_inc(a, phi, mu / total),
            )
        },
    )
}
