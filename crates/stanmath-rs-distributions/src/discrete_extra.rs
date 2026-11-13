//! Further discrete distributions: beta binomial, hypergeometric, negative binomial (shape, rate),
//! beta negative binomial, discrete range, Poisson binomial, ordered logistic and probit,
//! multinomial, and Dirichlet multinomial.
//!
//! Densities are written from their definitions. Kernels obtain their partial derivatives by
//! forward mode on the generic density (one pass per real argument), so no partial is derived by
//! hand; the special functions they use (`lbeta`, `beta_inc`) carry accurate derivative rules.

use stanmath_rs_autodiff::functions::{
    beta_inc, lbeta, log_Phi, log_diff_exp, log_inv_logit, log_sum_exp, log_sum_exp_slice, log1m,
    log1m_exp, log1m_inv_logit, log1p,
};
use stanmath_rs_autodiff::{Arg, Broadcast, Dual, IntArg, Lift, Promoted, Real, broadcast_len};
use stanmath_rs_core::{Error, Result};

use crate::check::{obs, param};
use crate::kernel::{Kernel, sum_kernel, sum_terms};

/// Tolerance for simplex checks, as Stan's `CONSTRAINT_TOLERANCE`.
const SIMPLEX_TOL: f64 = 1e-8;

/// `ln (N choose n)` for integers `0 <= n <= N`.
#[inline]
fn lchoose_int(big_n: i64, n: i64) -> f64 {
    let (nf, kf) = (big_n as f64, n as f64);
    -(nf + 1.0).ln() - stanmath_rs_core::special::lbeta(nf - kf + 1.0, kf + 1.0)
}

/// Writes the partials of `f` at `x` into `grad`, by one forward-mode pass per argument.
fn dual_grad<S: Real, const N: usize>(
    x: [S; N],
    grad: &mut [S; N],
    f: impl Fn([Dual<S>; N]) -> Dual<S>,
) {
    for (j, g) in grad.iter_mut().enumerate() {
        let xd: [Dual<S>; N] = core::array::from_fn(|k| {
            if k == j {
                Dual::variable(x[k])
            } else {
                Dual::constant(x[k])
            }
        });
        *g = f(xd).eps;
    }
}

/// `ln F` given closures for the lower tail `F` and the upper tail `1 - F`.
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

/// Common length of integer-only arguments.
fn int_len(name: &'static str, ints: &[&dyn IntArg]) -> Result<usize> {
    let lens: Vec<Option<usize>> = ints.iter().map(|a| a.int_len()).collect();
    broadcast_len(name, &lens)
}

// Beta binomial

/// `ln P(n)` of the beta binomial distribution: `C(N, n) B(n + α, N - n + β) / B(α, β)`.
fn beta_binomial_density<S: Real>(n: i64, big_n: i64, [alpha, beta]: [S; 2]) -> S {
    if n < 0 || n > big_n {
        return S::from_f64(f64::NEG_INFINITY);
    }
    let (nf, mf) = (n as f64, (big_n - n) as f64);
    S::from_f64(lchoose_int(big_n, n)) + lbeta(alpha + nf, beta + mf) - lbeta(alpha, beta)
}

fn check_beta_binomial(
    f: &'static str,
    big_n: Option<i64>,
    [alpha, beta]: [Option<f64>; 2],
) -> Result<()> {
    if let Some(big_n) = big_n.filter(|&n| n < 0) {
        return Err(Error::parameter(f, "N", big_n as f64, "nonnegative"));
    }
    param::positive_finite(f, "alpha", alpha)?;
    param::positive_finite(f, "beta", beta)
}

pub(crate) struct BetaBinomial;

impl Kernel<2, 2> for BetaBinomial {
    const NAME: &'static str = "beta_binomial_lpmf";

    fn check([_, big_n]: [Option<i64>; 2], x: [Option<f64>; 2]) -> Result<()> {
        check_beta_binomial(Self::NAME, big_n, x)
    }

    fn eval<S: Real>([n, big_n]: [i64; 2], x: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        if let Some(g) = grad {
            dual_grad(x, g, |d| beta_binomial_density(n, big_n, d));
        }
        beta_binomial_density(n, big_n, x)
    }
}

/// Beta binomial log mass of `n` successes in `N` trials with prior shapes `alpha` and `beta`.
///
/// Requires `N >= 0` and `alpha`, `beta` positive finite. As in Stan, `n` outside `[0, N]` gives
/// `-inf` rather than an error.
pub fn beta_binomial_lpmf<N, M, A, B>(
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
    sum_kernel::<BetaBinomial, _, 2, 2>([&n, &trials], (alpha, beta))
}

/// `ln sum_{k in ks} P(k)` for the beta binomial distribution.
fn beta_binomial_log_sum<S: Real>(ks: core::ops::RangeInclusive<i64>, big_n: i64, x: [S; 2]) -> S {
    let terms: Vec<S> = ks.map(|k| beta_binomial_density(k, big_n, x)).collect();
    log_sum_exp_slice(&terms)
}

/// `(ln P(X <= n), ln P(X > n))` for `0 <= n < N`, each tail summed directly when it is the
/// shorter one and obtained by complement otherwise.
fn beta_binomial_tails<S: Real>(n: i64, big_n: i64, x: [S; 2]) -> (S, S) {
    if 2 * n < big_n {
        let lower = beta_binomial_log_sum(0..=n, big_n, x);
        (lower, log1m_exp(lower))
    } else {
        let upper = beta_binomial_log_sum(n + 1..=big_n, big_n, x);
        (log1m_exp(upper), upper)
    }
}

/// Beta binomial log CDF `ln P(X <= n)`, by summation of the mass function.
pub fn beta_binomial_lcdf<N, M, A, B>(
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
    const F: &str = "beta_binomial_lcdf";
    sum_terms(
        F,
        (alpha, beta),
        &[&n, &trials],
        |k, x| check_beta_binomial(F, k[1], x),
        |k, x| {
            let (n, big_n) = (k[0], k[1]);
            if n < 0 {
                Promoted::<(A, B), 2>::from_f64(f64::NEG_INFINITY)
            } else if n >= big_n {
                Promoted::<(A, B), 2>::zero()
            } else {
                beta_binomial_tails(n, big_n, x).0
            }
        },
    )
}

/// Beta binomial log complementary CDF `ln P(X > n)`, by summation of the mass function.
pub fn beta_binomial_lccdf<N, M, A, B>(
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
    const F: &str = "beta_binomial_lccdf";
    sum_terms(
        F,
        (alpha, beta),
        &[&n, &trials],
        |k, x| check_beta_binomial(F, k[1], x),
        |k, x| {
            let (n, big_n) = (k[0], k[1]);
            if n < 0 {
                Promoted::<(A, B), 2>::zero()
            } else if n >= big_n {
                Promoted::<(A, B), 2>::from_f64(f64::NEG_INFINITY)
            } else {
                beta_binomial_tails(n, big_n, x).1
            }
        },
    )
}

// Hypergeometric

/// Hypergeometric log mass of `n` successes in `N` draws without replacement from a population
/// of `a` successes and `b` failures: `C(a, n) C(b, N - n) / C(a + b, N)`.
///
/// Every argument is an integer, so the result is `f64`. Requires `a, b >= 0`, `0 <= N <= a + b`,
/// `0 <= n <= a`, and `0 <= N - n <= b`.
pub fn hypergeometric_lpmf<N, D, A, B>(n: N, draws: D, successes: A, failures: B) -> Result<f64>
where
    N: IntArg,
    D: IntArg,
    A: IntArg,
    B: IntArg,
{
    const F: &str = "hypergeometric_lpmf";
    let len = int_len(F, &[&n, &draws, &successes, &failures])?;
    let mut lp = 0.0;
    for i in 0..len {
        let (k, big_n, a, b) = (
            n.int_at(i),
            draws.int_at(i),
            successes.int_at(i),
            failures.int_at(i),
        );
        if a < 0 {
            return Err(Error::parameter(F, "a", a as f64, "nonnegative"));
        }
        if b < 0 {
            return Err(Error::parameter(F, "b", b as f64, "nonnegative"));
        }
        if !(0..=a + b).contains(&big_n) {
            return Err(Error::parameter(F, "N", big_n as f64, "in [0, a + b]"));
        }
        if !(0..=a).contains(&k) || !(0..=b).contains(&(big_n - k)) {
            return Err(Error::observation(
                F,
                "n",
                k as f64,
                "in [max(0, N - b), min(a, N)]",
            ));
        }
        lp += lchoose_int(a, k) + lchoose_int(b, big_n - k) - lchoose_int(a + b, big_n);
    }
    Ok(lp)
}

// Negative binomial (shape, rate)

/// `ln P(n)` of the negative binomial with shape `α` and inverse scale `β`:
/// `C(n + α - 1, n) (β / (1 + β))^α (1 / (1 + β))^n`.
fn neg_binomial_density<S: Real>(n: i64, [alpha, beta]: [S; 2]) -> S {
    let nf = n as f64;
    // ln Γ(n + α) - ln Γ(n + 1) - ln Γ(α) = -ln(n + α) - ln B(α, n + 1)
    let coef = -(alpha + nf).ln() - lbeta(alpha, S::from_f64(nf + 1.0));
    let tail = if n == 0 { S::zero() } else { log1p(beta) * nf };
    coef - alpha * log1p(beta.recip()) - tail
}

fn check_neg_binomial(
    f: &'static str,
    n: Option<i64>,
    [alpha, beta]: [Option<f64>; 2],
) -> Result<()> {
    obs::count(f, "n", n)?;
    param::positive_finite(f, "alpha", alpha)?;
    param::positive_finite(f, "beta", beta)
}

pub(crate) struct NegBinomial;

impl Kernel<1, 2> for NegBinomial {
    const NAME: &'static str = "neg_binomial_lpmf";

    fn check([n]: [Option<i64>; 1], x: [Option<f64>; 2]) -> Result<()> {
        check_neg_binomial(Self::NAME, n, x)
    }

    fn eval<S: Real>([n]: [i64; 1], x: [S; 2], grad: Option<&mut [S; 2]>) -> S {
        if let Some(g) = grad {
            dual_grad(x, g, |d| neg_binomial_density(n, d));
        }
        neg_binomial_density(n, x)
    }
}

/// Negative binomial log mass of count `n` with shape `alpha` and inverse scale `beta`.
///
/// The mean is `alpha / beta`. Requires `n >= 0` and `alpha`, `beta` positive finite.
pub fn neg_binomial_lpmf<N, A, B>(n: N, alpha: A, beta: B) -> Result<Promoted<(A, B), 2>>
where
    N: IntArg,
    (A, B): Broadcast<2>,
{
    sum_kernel::<NegBinomial, _, 1, 2>([&n], (alpha, beta))
}

/// Negative binomial (shape, inverse scale) log CDF `ln I_{β/(1+β)}(α, n + 1)`.
pub fn neg_binomial_lcdf<N, A, B>(n: N, alpha: A, beta: B) -> Result<Promoted<(A, B), 2>>
where
    N: IntArg,
    (A, B): Broadcast<2>,
{
    const F: &str = "neg_binomial_lcdf";
    sum_terms(
        F,
        (alpha, beta),
        &[&n],
        |k, x| check_neg_binomial(F, k[0], x),
        |k, [alpha, beta]| {
            let b = Promoted::<(A, B), 2>::from_f64((k[0] + 1) as f64);
            let p = beta / (beta + 1.0);
            let q = (beta + 1.0).recip();
            log_lower(|| beta_inc(alpha, b, p), || beta_inc(b, alpha, q))
        },
    )
}

/// Negative binomial (shape, inverse scale) log complementary CDF `ln I_{1/(1+β)}(n + 1, α)`.
pub fn neg_binomial_lccdf<N, A, B>(n: N, alpha: A, beta: B) -> Result<Promoted<(A, B), 2>>
where
    N: IntArg,
    (A, B): Broadcast<2>,
{
    const F: &str = "neg_binomial_lccdf";
    sum_terms(
        F,
        (alpha, beta),
        &[&n],
        |k, x| check_neg_binomial(F, k[0], x),
        |k, [alpha, beta]| {
            let b = Promoted::<(A, B), 2>::from_f64((k[0] + 1) as f64);
            let p = beta / (beta + 1.0);
            let q = (beta + 1.0).recip();
            log_upper(|| beta_inc(alpha, b, p), || beta_inc(b, alpha, q))
        },
    )
}

// Beta negative binomial

/// `ln P(n)` of the beta negative binomial: the number of failures before `r` successes when the
/// success probability is `Beta(α, β)`,
/// `Γ(n + r) / (n! Γ(r)) B(α + r, β + n) / B(α, β)`.
fn beta_neg_binomial_density<S: Real>(n: i64, [r, alpha, beta]: [S; 3]) -> S {
    let nf = n as f64;
    let coef = -(r + nf).ln() - lbeta(r, S::from_f64(nf + 1.0));
    coef + lbeta(r + alpha, beta + nf) - lbeta(alpha, beta)
}

fn check_beta_neg_binomial(
    f: &'static str,
    n: Option<i64>,
    [r, alpha, beta]: [Option<f64>; 3],
) -> Result<()> {
    obs::count(f, "n", n)?;
    param::positive_finite(f, "r", r)?;
    param::positive_finite(f, "alpha", alpha)?;
    param::positive_finite(f, "beta", beta)
}

pub(crate) struct BetaNegBinomial;

impl Kernel<1, 3> for BetaNegBinomial {
    const NAME: &'static str = "beta_neg_binomial_lpmf";

    fn check([n]: [Option<i64>; 1], x: [Option<f64>; 3]) -> Result<()> {
        check_beta_neg_binomial(Self::NAME, n, x)
    }

    fn eval<S: Real>([n]: [i64; 1], x: [S; 3], grad: Option<&mut [S; 3]>) -> S {
        if let Some(g) = grad {
            dual_grad(x, g, |d| beta_neg_binomial_density(n, d));
        }
        beta_neg_binomial_density(n, x)
    }
}

/// Beta negative binomial log mass of `n` failures before `r` successes, with success probability
/// distributed as `Beta(alpha, beta)`.
///
/// Requires `n >= 0` and `r`, `alpha`, `beta` positive finite.
pub fn beta_neg_binomial_lpmf<N, R, A, B>(
    n: N,
    r: R,
    alpha: A,
    beta: B,
) -> Result<Promoted<(R, A, B), 3>>
where
    N: IntArg,
    (R, A, B): Broadcast<3>,
{
    sum_kernel::<BetaNegBinomial, _, 1, 3>([&n], (r, alpha, beta))
}

/// `ln P(X <= n)` of the beta negative binomial by summation of the mass function.
fn beta_neg_binomial_log_cdf<S: Real>(n: i64, x: [S; 3]) -> S {
    let terms: Vec<S> = (0..=n).map(|k| beta_neg_binomial_density(k, x)).collect();
    log_sum_exp_slice(&terms)
}

/// Beta negative binomial log CDF `ln P(X <= n)`, by summation of the mass function.
pub fn beta_neg_binomial_lcdf<N, R, A, B>(
    n: N,
    r: R,
    alpha: A,
    beta: B,
) -> Result<Promoted<(R, A, B), 3>>
where
    N: IntArg,
    (R, A, B): Broadcast<3>,
{
    const F: &str = "beta_neg_binomial_lcdf";
    sum_terms(
        F,
        (r, alpha, beta),
        &[&n],
        |k, x| check_beta_neg_binomial(F, k[0], x),
        |k, x| beta_neg_binomial_log_cdf(k[0], x),
    )
}

/// Beta negative binomial log complementary CDF `ln P(X > n)`, as `ln(1 - P(X <= n))`.
///
/// The upper tail is obtained by complement, so it loses relative accuracy where `P(X > n)` is
/// far below machine epsilon.
pub fn beta_neg_binomial_lccdf<N, R, A, B>(
    n: N,
    r: R,
    alpha: A,
    beta: B,
) -> Result<Promoted<(R, A, B), 3>>
where
    N: IntArg,
    (R, A, B): Broadcast<3>,
{
    const F: &str = "beta_neg_binomial_lccdf";
    sum_terms(
        F,
        (r, alpha, beta),
        &[&n],
        |k, x| check_beta_neg_binomial(F, k[0], x),
        |k, x| log1m_exp(beta_neg_binomial_log_cdf(k[0], x)),
    )
}

// Discrete range

fn check_range(f: &'static str, lower: i64, upper: i64) -> Result<()> {
    if lower > upper {
        return Err(Error::parameter(
            f,
            "lower",
            lower as f64,
            "not greater than upper",
        ));
    }
    Ok(())
}

/// Common length of the discrete range arguments. With an empty sequence, scalar bounds are
/// still checked, as Stan does.
fn range_len(
    f: &'static str,
    y: &dyn IntArg,
    lower: &dyn IntArg,
    upper: &dyn IntArg,
) -> Result<usize> {
    let len = int_len(f, &[y, lower, upper])?;
    if let (0, Some(lo), Some(hi)) = (len, lower.int_scalar(), upper.int_scalar()) {
        check_range(f, lo, hi)?;
    }
    Ok(len)
}

/// Uniform log mass on the integers `lower..=upper`.
///
/// Requires `lower <= upper`; `y` outside the range gives `-inf`.
pub fn discrete_range_lpmf<Y, L, U>(y: Y, lower: L, upper: U) -> Result<f64>
where
    Y: IntArg,
    L: IntArg,
    U: IntArg,
{
    const F: &str = "discrete_range_lpmf";
    let len = range_len(F, &y, &lower, &upper)?;
    let mut lp = 0.0;
    for i in 0..len {
        let (v, lo, hi) = (y.int_at(i), lower.int_at(i), upper.int_at(i));
        check_range(F, lo, hi)?;
        lp += if (lo..=hi).contains(&v) {
            -((hi - lo + 1) as f64).ln()
        } else {
            f64::NEG_INFINITY
        };
    }
    Ok(lp)
}

/// Discrete range log CDF `ln P(Y <= y)`.
pub fn discrete_range_lcdf<Y, L, U>(y: Y, lower: L, upper: U) -> Result<f64>
where
    Y: IntArg,
    L: IntArg,
    U: IntArg,
{
    const F: &str = "discrete_range_lcdf";
    let len = range_len(F, &y, &lower, &upper)?;
    let mut lp = 0.0;
    for i in 0..len {
        let (v, lo, hi) = (y.int_at(i), lower.int_at(i), upper.int_at(i));
        check_range(F, lo, hi)?;
        lp += if v < lo {
            f64::NEG_INFINITY
        } else if v >= hi {
            0.0
        } else {
            ((v - lo + 1) as f64 / (hi - lo + 1) as f64).ln()
        };
    }
    Ok(lp)
}

/// Discrete range log complementary CDF `ln P(Y > y)`.
pub fn discrete_range_lccdf<Y, L, U>(y: Y, lower: L, upper: U) -> Result<f64>
where
    Y: IntArg,
    L: IntArg,
    U: IntArg,
{
    const F: &str = "discrete_range_lccdf";
    let len = range_len(F, &y, &lower, &upper)?;
    let mut lp = 0.0;
    for i in 0..len {
        let (v, lo, hi) = (y.int_at(i), lower.int_at(i), upper.int_at(i));
        check_range(F, lo, hi)?;
        lp += if v < lo {
            0.0
        } else if v >= hi {
            f64::NEG_INFINITY
        } else {
            ((hi - v) as f64 / (hi - lo + 1) as f64).ln()
        };
    }
    Ok(lp)
}

// Poisson binomial

fn check_probabilities<T: Real>(f: &'static str, theta: &[T]) -> Result<()> {
    if let Some(t) = theta.iter().find(|t| !(0.0..=1.0).contains(&t.value())) {
        return Err(Error::parameter(f, "theta", t.value(), "in [0, 1]"));
    }
    Ok(())
}

/// `ln P(Y = j)` for `j = 0..=K`, where `Y` is the number of successes among independent
/// Bernoulli trials with probabilities `theta`, by the recursion over trials.
fn poisson_binomial_log_probs<T: Real>(theta: &[T]) -> Vec<T> {
    let mut alpha = vec![T::zero()];
    for &t in theta {
        let (lp, lq) = (t.ln(), log1m(t));
        let mut next = Vec::with_capacity(alpha.len() + 1);
        next.push(alpha[0] + lq);
        for j in 1..alpha.len() {
            next.push(log_sum_exp(alpha[j] + lq, alpha[j - 1] + lp));
        }
        next.push(alpha[alpha.len() - 1] + lp);
        alpha = next;
    }
    alpha
}

fn poisson_binomial_sum<N, T>(
    f: &'static str,
    y: N,
    theta: &[T],
    term: impl Fn(i64, &[T]) -> T,
) -> Result<T>
where
    N: IntArg,
    T: Real,
{
    check_probabilities(f, theta)?;
    let len = int_len(f, &[&y])?;
    let k = theta.len() as i64;
    for i in 0..len {
        let v = y.int_at(i);
        if !(0..=k).contains(&v) {
            return Err(Error::observation(f, "y", v as f64, "in [0, size(theta)]"));
        }
    }
    if len == 0 {
        return Ok(T::zero());
    }
    let logp = poisson_binomial_log_probs(theta);
    let mut total = T::zero();
    for i in 0..len {
        total += term(y.int_at(i), &logp);
    }
    Ok(total)
}

/// Poisson binomial log mass: `y` successes among independent trials with success
/// probabilities `theta`.
///
/// Requires every `theta` in `[0, 1]` and `y` in `[0, size(theta)]`. Probabilities of exactly 0 or
/// 1 give infinite partials, as the logarithm of zero enters the recursion.
pub fn poisson_binomial_lpmf<N, T>(y: N, theta: &[T]) -> Result<T>
where
    N: IntArg,
    T: Real,
{
    poisson_binomial_sum("poisson_binomial_lpmf", y, theta, |v, lp| lp[v as usize])
}

/// Poisson binomial log CDF `ln P(Y <= y)`.
pub fn poisson_binomial_lcdf<N, T>(y: N, theta: &[T]) -> Result<T>
where
    N: IntArg,
    T: Real,
{
    poisson_binomial_sum("poisson_binomial_lcdf", y, theta, |v, lp| {
        log_sum_exp_slice(&lp[..=v as usize])
    })
}

/// Poisson binomial log complementary CDF `ln P(Y > y)`.
pub fn poisson_binomial_lccdf<N, T>(y: N, theta: &[T]) -> Result<T>
where
    N: IntArg,
    T: Real,
{
    poisson_binomial_sum("poisson_binomial_lccdf", y, theta, |v, lp| {
        log_sum_exp_slice(&lp[v as usize + 1..])
    })
}

// Ordered logistic and ordered probit

/// Checks that cutpoints are finite and strictly increasing.
pub(crate) fn check_cutpoints<T: Real>(f: &'static str, c: &[T]) -> Result<()> {
    if let Some(v) = c.iter().find(|v| !v.is_finite()) {
        return Err(Error::parameter(f, "c", v.value(), "finite"));
    }
    if let Some(w) = c.windows(2).find(|w| !(w[0].value() < w[1].value())) {
        return Err(Error::parameter(
            f,
            "c",
            w[1].value(),
            "strictly increasing",
        ));
    }
    Ok(())
}

/// `ln(inv_logit(x) - inv_logit(y))` for `x > y`, as
/// `ln inv_logit(x) + ln(1 - exp(y - x)) + ln(1 - inv_logit(y))`.
fn log_inv_logit_diff<T: Real>(x: T, y: T) -> T {
    log_inv_logit(x) + log1m_exp(y - x) + log1m_inv_logit(y)
}

fn ordered_logistic_term<T: Real>(y: usize, eta: T, c: &[T]) -> T {
    let k = c.len() + 1;
    if k == 1 {
        T::zero()
    } else if y == 1 {
        log1m_inv_logit(eta - c[0])
    } else if y == k {
        log_inv_logit(eta - c[k - 2])
    } else {
        log_inv_logit_diff(eta - c[y - 2], eta - c[y - 1])
    }
}

fn ordered_probit_term<T: Real>(y: usize, eta: T, c: &[T]) -> T {
    let k = c.len() + 1;
    if k == 1 {
        T::zero()
    } else if y == 1 {
        log_Phi(c[0] - eta)
    } else if y == k {
        log_Phi(eta - c[k - 2])
    } else {
        // Φ(a) - Φ(b) with a > b; in the upper region use Φ(-b) - Φ(-a) to keep the digits.
        let (a, b) = (eta - c[y - 2], eta - c[y - 1]);
        if (a + b).value() > 0.0 {
            log_diff_exp(log_Phi(-b), log_Phi(-a))
        } else {
            log_diff_exp(log_Phi(a), log_Phi(b))
        }
    }
}

fn ordered_sum<N, E, T>(
    f: &'static str,
    y: N,
    eta: E,
    c: &[T],
    term: fn(usize, T, &[T]) -> T,
) -> Result<T>
where
    N: IntArg,
    E: Arg,
    E::Elem: Lift<T>,
    T: Real,
{
    check_cutpoints(f, c)?;
    let k = c.len() + 1;
    let len = broadcast_len(f, &[y.int_len(), eta.arg_len()])?;
    let mut lp = T::zero();
    for i in 0..len {
        let v = y.int_at(i);
        if v < 1 || v as usize > k {
            return Err(Error::observation(f, "y", v as f64, "in [1, K]"));
        }
        let e: T = eta.arg_at(i).lift();
        if !e.is_finite() {
            return Err(Error::parameter(f, "eta", e.value(), "finite"));
        }
        lp += term(v as usize, e, c);
    }
    Ok(lp)
}

/// Ordered logistic log mass of categories `y` in `1..=K` with linear predictor `eta` and
/// cutpoints `c` (length `K - 1`).
///
/// `P(y = 1) = 1 - inv_logit(eta - c_1)`, `P(y = k) = inv_logit(eta - c_{k-1}) -
/// inv_logit(eta - c_k)`, `P(y = K) = inv_logit(eta - c_{K-1})`. Requires `eta` finite and `c`
/// finite and strictly increasing. `eta` is a scalar or a sequence matching `y`.
pub fn ordered_logistic_lpmf<N, E, T>(y: N, eta: E, c: &[T]) -> Result<T>
where
    N: IntArg,
    E: Arg,
    E::Elem: Lift<T>,
    T: Real,
{
    const F: &str = "ordered_logistic_lpmf";
    // Stan returns zero for an empty response before checking the cutpoints, unlike
    // ordered_probit (checked against Stan Math 5.4.0).
    if broadcast_len(F, &[y.int_len(), eta.arg_len()])? == 0 {
        return Ok(T::zero());
    }
    ordered_sum(F, y, eta, c, ordered_logistic_term)
}

/// Ordered probit log mass: as [`ordered_logistic_lpmf`] with the standard normal CDF in place of
/// `inv_logit`.
pub fn ordered_probit_lpmf<N, E, T>(y: N, eta: E, c: &[T]) -> Result<T>
where
    N: IntArg,
    E: Arg,
    E::Elem: Lift<T>,
    T: Real,
{
    ordered_sum("ordered_probit_lpmf", y, eta, c, ordered_probit_term)
}

// Multinomial family

/// Checks counts and returns their total and `ln(N! / prod n_k!)`.
fn multinomial_counts(
    f: &'static str,
    ns: &[i64],
    k: usize,
    name: &'static str,
) -> Result<(i64, f64)> {
    if ns.len() != k {
        return Err(Error::dimension(
            f,
            format!("{name} has length {k}, but there are {} counts", ns.len()),
        ));
    }
    for &n in ns {
        obs::count(f, "ns", n)?;
    }
    let total: i64 = ns.iter().sum();
    let coef = stanmath_rs_core::special::lgamma(total as f64 + 1.0)
        - ns.iter()
            .map(|&n| stanmath_rs_core::special::lgamma(n as f64 + 1.0))
            .sum::<f64>();
    Ok((total, coef))
}

/// Multinomial log mass of counts `ns` with category probabilities `theta` (a simplex).
pub fn multinomial_lpmf<T: Real>(ns: &[i64], theta: &[T]) -> Result<T> {
    const F: &str = "multinomial_lpmf";
    let sum: f64 = theta.iter().map(Real::value).sum();
    if theta.iter().any(|t| t.is_nan() || t.value() < 0.0) || (sum - 1.0).abs() > SIMPLEX_TOL {
        return Err(Error::parameter(
            F,
            "theta",
            sum,
            "a simplex (nonnegative, summing to 1)",
        ));
    }
    let (_, coef) = multinomial_counts(F, ns, theta.len(), "theta")?;
    let mut lp = T::from_f64(coef);
    for (&n, &t) in ns.iter().zip(theta) {
        if n > 0 {
            lp += t.ln() * n as f64;
        }
    }
    Ok(lp)
}

/// Multinomial log mass with probabilities `softmax(gamma)`, stable for any finite `gamma`.
pub fn multinomial_logit_lpmf<T: Real>(ns: &[i64], gamma: &[T]) -> Result<T> {
    const F: &str = "multinomial_logit_lpmf";
    if let Some(g) = gamma.iter().find(|g| !g.is_finite()) {
        return Err(Error::parameter(F, "gamma", g.value(), "finite"));
    }
    let (total, coef) = multinomial_counts(F, ns, gamma.len(), "gamma")?;
    if total == 0 {
        return Ok(T::zero());
    }
    let mut lp = T::from_f64(coef) - log_sum_exp_slice(gamma) * total as f64;
    for (&n, &g) in ns.iter().zip(gamma) {
        if n > 0 {
            lp += g * n as f64;
        }
    }
    Ok(lp)
}

/// Dirichlet multinomial log mass of counts `ns` with concentrations `alpha`:
/// `N! Γ(α₀) / Γ(N + α₀) prod_k Γ(n_k + α_k) / (n_k! Γ(α_k))`, `α₀ = sum α`.
///
/// Requires `alpha` positive finite and nonnegative counts. Zero total count gives 0.
pub fn dirichlet_multinomial_lpmf<T: Real>(ns: &[i64], alpha: &[T]) -> Result<T> {
    const F: &str = "dirichlet_multinomial_lpmf";
    if let Some(a) = alpha.iter().find(|a| !(a.value() > 0.0 && a.is_finite())) {
        return Err(Error::parameter(F, "alpha", a.value(), "positive finite"));
    }
    let (total, _) = multinomial_counts(F, ns, alpha.len(), "alpha")?;
    if total == 0 {
        return Ok(T::zero());
    }
    // N! Γ(α₀) / Γ(N + α₀) = N B(α₀, N) and Γ(n + α) / (n! Γ(α)) = 1 / (n B(α, n)).
    let alpha0: T = alpha.iter().copied().sum();
    let mut lp = lbeta(alpha0, T::from_f64(total as f64)) + (total as f64).ln();
    for (&n, &a) in ns.iter().zip(alpha) {
        if n > 0 {
            lp -= lbeta(a, T::from_f64(n as f64)) + (n as f64).ln();
        }
    }
    Ok(lp)
}
