//! Log probability mass functions of discrete distributions.
//!
//! Integer arguments (counts and outcomes) accept `i64` scalars or sequences (see
//! [`IntArg`]); real arguments broadcast as for the continuous distributions.

use stanmath_rs_autodiff::functions::{
    inv_logit, lbeta, lgamma, log_inv_logit, log_sum_exp, log_sum_exp_slice, log1m,
    log1m_inv_logit, log1p_exp,
};
use stanmath_rs_autodiff::special::digamma_diff;
use stanmath_rs_autodiff::{Broadcast, IntArg, Promoted, Real, broadcast_len};
use stanmath_rs_core::{Error, Result};

use crate::check::{obs, param};
use crate::kernel::{Keep, Kernel, sum_kernel, sum_kernel_propto};

/// `ln (N choose n)` for integers.
#[inline]
fn lchoose_int(big_n: i64, n: i64) -> f64 {
    let (nf, kf) = (big_n as f64, n as f64);
    -(nf + 1.0).ln() - stanmath_rs_core::special::lbeta(nf - kf + 1.0, kf + 1.0)
}

pub(crate) struct Bernoulli;

impl Kernel<1, 1> for Bernoulli {
    const NAME: &'static str = "bernoulli_lpmf";

    fn check([n]: [Option<i64>; 1], [theta]: [Option<f64>; 1]) -> Result<()> {
        obs::binary(Self::NAME, "n", n)?;
        param::probability(Self::NAME, "theta", theta)
    }

    fn eval<S: Real>([n]: [i64; 1], [theta]: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        if n == 1 {
            if let Some(g) = grad {
                g[0] = theta.recip();
            }
            theta.ln()
        } else {
            if let Some(g) = grad {
                g[0] = (theta - 1.0).recip();
            }
            log1m(theta)
        }
    }
}

/// Bernoulli log mass of outcome `n` in `{0, 1}` with success probability `theta`.
pub fn bernoulli_lpmf<N, T>(n: N, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    (T,): Broadcast<1>,
{
    sum_kernel::<Bernoulli, _, 1, 1>([&n], (theta,))
}

/// [`bernoulli_lpmf`] without the terms that depend only on data (Stan's `bernoulli_lupmf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn bernoulli_lupmf<N, T>(n: N, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    (T,): Broadcast<1>,
{
    sum_kernel_propto::<Bernoulli, _, 1, 1>([&n], (theta,))
}

pub(crate) struct BernoulliLogit;

impl Kernel<1, 1> for BernoulliLogit {
    const NAME: &'static str = "bernoulli_logit_lpmf";

    fn check([n]: [Option<i64>; 1], [alpha]: [Option<f64>; 1]) -> Result<()> {
        obs::binary(Self::NAME, "n", n)?;
        param::not_nan(Self::NAME, "alpha", alpha)
    }

    fn eval<S: Real>([n]: [i64; 1], [alpha]: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        if n == 1 {
            if let Some(g) = grad {
                g[0] = inv_logit(-alpha);
            }
            log_inv_logit(alpha)
        } else {
            if let Some(g) = grad {
                g[0] = -inv_logit(alpha);
            }
            log1m_inv_logit(alpha)
        }
    }
}

/// Bernoulli log mass with success probability `inv_logit(alpha)`, stable for any `alpha`.
///
/// This is the likelihood of logistic regression: pass the linear predictor as `alpha`.
pub fn bernoulli_logit_lpmf<N, A>(n: N, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel::<BernoulliLogit, _, 1, 1>([&n], (alpha,))
}

/// [`bernoulli_logit_lpmf`] without the terms that depend only on data (Stan's `bernoulli_logit_lupmf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn bernoulli_logit_lupmf<N, A>(n: N, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel_propto::<BernoulliLogit, _, 1, 1>([&n], (alpha,))
}

/// `N >= 0` and `n` in `[0, N]`, each checked when present.
fn check_trials(f: &'static str, n: Option<i64>, big_n: Option<i64>) -> Result<()> {
    if let Some(big_n) = big_n {
        if big_n < 0 {
            return Err(Error::parameter(f, "N", big_n as f64, "nonnegative"));
        }
        if let Some(n) = n.filter(|n| !(0..=big_n).contains(n)) {
            return Err(Error::observation(f, "n", n as f64, "in [0, N]"));
        }
    }
    Ok(())
}

pub(crate) struct Binomial;

impl Kernel<2, 1> for Binomial {
    const NAME: &'static str = "binomial_lpmf";

    fn check([n, big_n]: [Option<i64>; 2], [theta]: [Option<f64>; 1]) -> Result<()> {
        check_trials(Self::NAME, n, big_n)?;
        param::probability(Self::NAME, "theta", theta)
    }

    fn eval<S: Real>(ints: [i64; 2], x: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        Self::eval_propto(ints, x, grad, Keep::ALL)
    }

    fn eval_propto<S: Real>(
        [n, big_n]: [i64; 2],
        [theta]: [S; 1],
        grad: Option<&mut [S; 1]>,
        keep: Keep<1>,
    ) -> S {
        let (nf, mf) = (n as f64, (big_n - n) as f64);
        let mut lp = if keep.term(&[]) {
            S::from_f64(lchoose_int(big_n, n))
        } else {
            S::zero()
        };
        let mut d = S::zero();
        if n > 0 {
            lp += theta.ln() * nf;
            d += theta.recip() * nf;
        }
        if big_n > n {
            lp += log1m(theta) * mf;
            d += (theta - 1.0).recip() * mf;
        }
        if let Some(g) = grad {
            g[0] = d;
        }
        lp
    }
}

/// Binomial log mass of `n` successes in `N` trials with success probability `theta`.
pub fn binomial_lpmf<N, M, T>(n: N, trials: M, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    M: IntArg,
    (T,): Broadcast<1>,
{
    sum_kernel::<Binomial, _, 2, 1>([&n, &trials], (theta,))
}

/// [`binomial_lpmf`] without the terms that depend only on data (Stan's `binomial_lupmf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn binomial_lupmf<N, M, T>(n: N, trials: M, theta: T) -> Result<Promoted<(T,), 1>>
where
    N: IntArg,
    M: IntArg,
    (T,): Broadcast<1>,
{
    sum_kernel_propto::<Binomial, _, 2, 1>([&n, &trials], (theta,))
}

pub(crate) struct BinomialLogit;

impl Kernel<2, 1> for BinomialLogit {
    const NAME: &'static str = "binomial_logit_lpmf";

    fn check([n, big_n]: [Option<i64>; 2], [alpha]: [Option<f64>; 1]) -> Result<()> {
        check_trials(Self::NAME, n, big_n)?;
        param::finite(Self::NAME, "alpha", alpha)
    }

    fn eval<S: Real>(ints: [i64; 2], x: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        Self::eval_propto(ints, x, grad, Keep::ALL)
    }

    fn eval_propto<S: Real>(
        [n, big_n]: [i64; 2],
        [alpha]: [S; 1],
        grad: Option<&mut [S; 1]>,
        keep: Keep<1>,
    ) -> S {
        let (nf, mf) = (n as f64, (big_n - n) as f64);
        if let Some(g) = grad {
            g[0] = inv_logit(-alpha) * nf - inv_logit(alpha) * mf;
        }
        let lp = log_inv_logit(alpha) * nf + log1m_inv_logit(alpha) * mf;
        if keep.term(&[]) {
            lp + lchoose_int(big_n, n)
        } else {
            lp
        }
    }
}

/// Binomial log mass with success probability `inv_logit(alpha)`.
pub fn binomial_logit_lpmf<N, M, A>(n: N, trials: M, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    M: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel::<BinomialLogit, _, 2, 1>([&n, &trials], (alpha,))
}

/// [`binomial_logit_lpmf`] without the terms that depend only on data (Stan's `binomial_logit_lupmf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn binomial_logit_lupmf<N, M, A>(n: N, trials: M, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    M: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel_propto::<BinomialLogit, _, 2, 1>([&n, &trials], (alpha,))
}

pub(crate) struct Poisson;

impl Kernel<1, 1> for Poisson {
    const NAME: &'static str = "poisson_lpmf";

    fn check([n]: [Option<i64>; 1], [lambda]: [Option<f64>; 1]) -> Result<()> {
        obs::count(Self::NAME, "n", n)?;
        param::nonnegative(Self::NAME, "lambda", lambda)
    }

    fn eval<S: Real>(ints: [i64; 1], x: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        Self::eval_propto(ints, x, grad, Keep::ALL)
    }

    fn eval_propto<S: Real>(
        [n]: [i64; 1],
        [lambda]: [S; 1],
        grad: Option<&mut [S; 1]>,
        keep: Keep<1>,
    ) -> S {
        let nf = n as f64;
        let lv = lambda.value();
        if lv == f64::INFINITY || (lv == 0.0 && n > 0) {
            if let Some(g) = grad {
                g[0] = S::zero();
            }
            return S::from_f64(f64::NEG_INFINITY);
        }
        if let Some(g) = grad {
            g[0] = if n == 0 {
                S::from_f64(-1.0)
            } else {
                lambda.recip() * nf - 1.0
            };
        }
        let log_term = if n == 0 { S::zero() } else { lambda.ln() * nf };
        let lp = log_term - lambda;
        if keep.term(&[]) {
            lp - lgamma(nf + 1.0)
        } else {
            lp
        }
    }
}

/// Poisson log mass of count `n` with rate `lambda`.
///
/// Requires `n >= 0` and `lambda >= 0`; `lambda = 0` gives `0` for `n = 0` and `-inf` otherwise,
/// and `lambda = inf` gives `-inf`.
pub fn poisson_lpmf<N, L>(n: N, lambda: L) -> Result<Promoted<(L,), 1>>
where
    N: IntArg,
    (L,): Broadcast<1>,
{
    sum_kernel::<Poisson, _, 1, 1>([&n], (lambda,))
}

/// [`poisson_lpmf`] without the terms that depend only on data (Stan's `poisson_lupmf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn poisson_lupmf<N, L>(n: N, lambda: L) -> Result<Promoted<(L,), 1>>
where
    N: IntArg,
    (L,): Broadcast<1>,
{
    sum_kernel_propto::<Poisson, _, 1, 1>([&n], (lambda,))
}

pub(crate) struct PoissonLog;

impl Kernel<1, 1> for PoissonLog {
    const NAME: &'static str = "poisson_log_lpmf";

    fn check([n]: [Option<i64>; 1], [alpha]: [Option<f64>; 1]) -> Result<()> {
        obs::count(Self::NAME, "n", n)?;
        param::not_nan(Self::NAME, "alpha", alpha)
    }

    fn eval<S: Real>(ints: [i64; 1], x: [S; 1], grad: Option<&mut [S; 1]>) -> S {
        Self::eval_propto(ints, x, grad, Keep::ALL)
    }

    fn eval_propto<S: Real>(
        [n]: [i64; 1],
        [alpha]: [S; 1],
        grad: Option<&mut [S; 1]>,
        keep: Keep<1>,
    ) -> S {
        let nf = n as f64;
        let av = alpha.value();
        if av == f64::INFINITY || (av == f64::NEG_INFINITY && n > 0) {
            if let Some(g) = grad {
                g[0] = S::zero();
            }
            return S::from_f64(f64::NEG_INFINITY);
        }
        if av == f64::NEG_INFINITY {
            if let Some(g) = grad {
                g[0] = S::zero();
            }
            return S::zero();
        }
        let rate = alpha.exp();
        if let Some(g) = grad {
            g[0] = -rate + nf;
        }
        let lp = alpha * nf - rate;
        if keep.term(&[]) {
            lp - lgamma(nf + 1.0)
        } else {
            lp
        }
    }
}

/// Poisson log mass with log rate `alpha` (rate `exp(alpha)`).
pub fn poisson_log_lpmf<N, A>(n: N, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel::<PoissonLog, _, 1, 1>([&n], (alpha,))
}

/// [`poisson_log_lpmf`] without the terms that depend only on data (Stan's `poisson_log_lupmf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn poisson_log_lupmf<N, A>(n: N, alpha: A) -> Result<Promoted<(A,), 1>>
where
    N: IntArg,
    (A,): Broadcast<1>,
{
    sum_kernel_propto::<PoissonLog, _, 1, 1>([&n], (alpha,))
}

/// Shared part of the negative binomial kernels:
/// `ln Γ(n + φ) - ln Γ(n + 1) - ln Γ(φ) = -ln(n + φ) - ln B(φ, n + 1)`.
#[inline]
fn neg_binomial_coef<S: Real>(n: f64, phi: S) -> S {
    -(phi + n).ln() - lbeta(phi, S::from_f64(n + 1.0))
}

pub(crate) struct NegBinomial2;

impl Kernel<1, 2> for NegBinomial2 {
    const NAME: &'static str = "neg_binomial_2_lpmf";

    fn check([n]: [Option<i64>; 1], [mu, phi]: [Option<f64>; 2]) -> Result<()> {
        obs::count(Self::NAME, "n", n)?;
        param::positive_finite(Self::NAME, "mu", mu)?;
        param::positive_finite(Self::NAME, "phi", phi)
    }

    fn eval<S: Real>(ints: [i64; 1], x: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        Self::eval_propto(ints, x, grad, Keep::ALL)
    }

    fn eval_propto<S: Real>(
        [n]: [i64; 1],
        [mu, phi]: [S; 2],
        grad: Option<&mut [S; 2]>,
        keep: Keep<2>,
    ) -> S {
        let nf = n as f64;
        let mu_phi = mu + phi;
        let log1p_ratio = (mu / phi).ln_1p(); // ln((mu + phi) / phi)
        if let Some(g) = grad {
            g[0] = mu.recip() * nf - (phi + nf) / mu_phi;
            g[1] = -digamma_diff(phi, S::from_f64(nf)) - log1p_ratio + (mu - nf) / mu_phi;
        }
        let mut lp = -(phi * log1p_ratio);
        if n > 0 {
            if keep.term(&[0]) {
                lp += mu.ln() * nf;
            }
            lp -= mu_phi.ln() * nf;
        }
        if keep.term(&[1]) {
            lp += neg_binomial_coef(nf, phi);
        }
        lp
    }
}

/// Negative binomial log mass of count `n` with mean `mu` and dispersion `phi`.
///
/// The variance is `mu + mu^2 / phi`; `phi -> inf` recovers the Poisson distribution. Requires
/// `n >= 0` and `mu`, `phi` positive finite.
pub fn neg_binomial_2_lpmf<N, M, P>(n: N, mu: M, phi: P) -> Result<Promoted<(M, P), 2>>
where
    N: IntArg,
    (M, P): Broadcast<2>,
{
    sum_kernel::<NegBinomial2, _, 1, 2>([&n], (mu, phi))
}

/// [`neg_binomial_2_lpmf`] without the terms that depend only on data (Stan's `neg_binomial_2_lupmf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn neg_binomial_2_lupmf<N, M, P>(n: N, mu: M, phi: P) -> Result<Promoted<(M, P), 2>>
where
    N: IntArg,
    (M, P): Broadcast<2>,
{
    sum_kernel_propto::<NegBinomial2, _, 1, 2>([&n], (mu, phi))
}

pub(crate) struct NegBinomial2Log;

impl Kernel<1, 2> for NegBinomial2Log {
    const NAME: &'static str = "neg_binomial_2_log_lpmf";

    fn check([n]: [Option<i64>; 1], [eta, phi]: [Option<f64>; 2]) -> Result<()> {
        obs::count(Self::NAME, "n", n)?;
        param::finite(Self::NAME, "eta", eta)?;
        param::positive_finite(Self::NAME, "phi", phi)
    }

    fn eval<S: Real>(ints: [i64; 1], x: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        Self::eval_propto(ints, x, grad, Keep::ALL)
    }

    fn eval_propto<S: Real>(
        [n]: [i64; 1],
        [eta, phi]: [S; 2],
        grad: Option<&mut [S; 2]>,
        keep: Keep<2>,
    ) -> S {
        let nf = n as f64;
        let log_phi = phi.ln();
        // ln(1 + mu / phi) = log1p_exp(eta - ln phi); ln(mu + phi) = log_sum_exp(eta, ln phi)
        let log1p_ratio = log1p_exp(eta - log_phi);
        let log_mu_phi = log_sum_exp(eta, log_phi);
        if let Some(g) = grad {
            let mu = eta.exp();
            g[0] = -((phi + nf) * inv_logit(eta - log_phi)) + nf;
            g[1] = -digamma_diff(phi, S::from_f64(nf)) - log1p_ratio + (mu - nf) / (mu + phi);
        }
        let mut lp = -(log_mu_phi * nf) - phi * log1p_ratio;
        if keep.term(&[0]) {
            lp += eta * nf;
        }
        if keep.term(&[1]) {
            lp += neg_binomial_coef(nf, phi);
        }
        lp
    }
}

/// [`NegBinomial2Log`] with the terms grouped as in Stan's `neg_binomial_2_log_glm`, whose
/// `propto` form drops `phi ln phi` and `-ln Γ(n + 1)` on their own: `n eta`,
/// `-(n + phi) ln(exp(eta) + phi)`, `phi ln phi + ln Γ(n + phi) - ln Γ(phi)`, and `-ln Γ(n + 1)`.
pub(crate) struct NegBinomial2LogGlm;

impl Kernel<1, 2> for NegBinomial2LogGlm {
    const NAME: &'static str = "neg_binomial_2_log_glm_lpmf";

    fn check(ints: [Option<i64>; 1], x: [Option<f64>; 2]) -> Result<()> {
        NegBinomial2Log::check(ints, x)
    }

    fn eval<S: Real>(ints: [i64; 1], x: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        NegBinomial2Log::eval(ints, x, grad)
    }

    fn eval_propto<S: Real>(
        ints: [i64; 1],
        x: [S; 2],
        grad: Option<&mut [S; 2]>,
        keep: Keep<2>,
    ) -> S {
        if keep.term(&[]) {
            return NegBinomial2Log::eval(ints, x, grad);
        }
        if grad.is_some() {
            // Dropped terms depend only on data, so the gradient is that of the full mass.
            NegBinomial2Log::eval(ints, x, grad);
        }
        let ([n], [eta, phi]) = (ints, x);
        let nf = n as f64;
        let log_phi = phi.ln();
        let mut lp = -((phi + nf) * log_sum_exp(eta, log_phi));
        if keep.term(&[0]) {
            lp += eta * nf;
        }
        if keep.term(&[1]) {
            lp += phi * log_phi + neg_binomial_coef(nf, phi) + lgamma(nf + 1.0);
        }
        lp
    }
}

/// Negative binomial log mass with log mean `eta` and dispersion `phi`.
pub fn neg_binomial_2_log_lpmf<N, E, P>(n: N, eta: E, phi: P) -> Result<Promoted<(E, P), 2>>
where
    N: IntArg,
    (E, P): Broadcast<2>,
{
    sum_kernel::<NegBinomial2Log, _, 1, 2>([&n], (eta, phi))
}

/// [`neg_binomial_2_log_lpmf`] without the terms that depend only on data (Stan's `neg_binomial_2_log_lupmf`, the
/// `propto` form): zero when every argument is data, and with the same derivatives.
pub fn neg_binomial_2_log_lupmf<N, E, P>(n: N, eta: E, phi: P) -> Result<Promoted<(E, P), 2>>
where
    N: IntArg,
    (E, P): Broadcast<2>,
{
    sum_kernel_propto::<NegBinomial2Log, _, 1, 2>([&n], (eta, phi))
}

/// Tolerance for checking that probabilities sum to one, as in Stan's `check_simplex`.
const SIMPLEX_TOL: f64 = 1e-8;

fn check_categories(f: &'static str, n: &dyn IntArg, k: usize) -> Result<usize> {
    let len = broadcast_len(f, &[n.int_len()])?;
    for i in 0..len {
        let v = n.int_at(i);
        if v < 1 || v as usize > k {
            return Err(Error::observation(f, "n", v as f64, "in [1, K]"));
        }
    }
    Ok(len)
}

/// Categorical log mass of outcomes `n` in `1..=K` with probability vector `theta` (a simplex).
pub fn categorical_lpmf<N, T>(n: N, theta: &[T]) -> Result<T>
where
    N: IntArg,
    T: Real,
{
    const F: &str = "categorical_lpmf";
    let sum: f64 = theta.iter().map(Real::value).sum();
    if theta.iter().any(|t| t.is_nan() || t.value() < 0.0) || (sum - 1.0).abs() > SIMPLEX_TOL {
        return Err(Error::parameter(
            F,
            "theta",
            sum,
            "a simplex (nonnegative, summing to 1)",
        ));
    }
    let len = check_categories(F, &n, theta.len())?;
    let logs: Vec<T> = theta.iter().map(|t| t.ln()).collect();
    let mut lp = T::zero();
    for i in 0..len {
        lp += logs[n.int_at(i) as usize - 1];
    }
    Ok(lp)
}

/// Categorical log mass with probabilities `softmax(beta)`, stable for any `beta`.
pub fn categorical_logit_lpmf<N, T>(n: N, beta: &[T]) -> Result<T>
where
    N: IntArg,
    T: Real,
{
    const F: &str = "categorical_logit_lpmf";
    if let Some(b) = beta.iter().find(|b| !b.is_finite()) {
        return Err(Error::parameter(F, "beta", b.value(), "finite"));
    }
    let len = check_categories(F, &n, beta.len())?;
    if len == 0 {
        return Ok(T::zero());
    }
    let lse = log_sum_exp_slice(beta);
    let mut lp = T::zero();
    for i in 0..len {
        lp += beta[n.int_at(i) as usize - 1];
    }
    Ok(lp - lse * len as f64)
}
