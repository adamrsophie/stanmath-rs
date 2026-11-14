//! Further differentiable scalar functions with Stan Math names.
//!
//! Most are compositions of existing primitives, so their derivatives follow automatically and
//! stay exact under nesting. Three have their own rules: [`fmod`] (partials `1` and
//! `-trunc(a / b)`), [`std_normal_log_qf`] (derivative `1 / m(x)` with `m` the inverse Mills
//! ratio), and [`inv_inc_beta`] (implicit function theorem on `I_x(a, b) = p`).

use core::f64::consts::{FRAC_1_SQRT_2, LN_2};

use stanmath_rs_core::special as sp;
use stanmath_rs_core::{Error, Result};

use crate::functions::{
    inv_logit, log_inv_logit, log_sum_exp, log_sum_exp_slice, log1m, log1m_exp, log1p_exp,
};
use crate::real::{BinaryFn, NaryFn, Real, UnaryFn};
use crate::special::{BetaInc, InvMillsRatio, beta_inc, inv_Phi, lbeta, lchoose, lgamma};

/// Logistic approximation of the standard normal CDF, `inv_logit(0.07056 x^3 + 1.5976 x)`.
#[allow(non_snake_case)]
#[inline]
pub fn Phi_approx<T: Real>(x: T) -> T {
    inv_logit(x.powi(3) * 0.07056 + x * 1.5976)
}

/// Log loss of the prediction `y_hat` for the binary outcome `y`: `-ln(y_hat)` when `y = 1`,
/// `-ln(1 - y_hat)` otherwise.
#[inline]
pub fn binary_log_loss<T: Real>(y: i64, y_hat: T) -> T {
    if y == 0 { -log1m(y_hat) } else { -y_hat.ln() }
}

/// `ln Γ(x + 1) - ln Γ(x - n + 1)`, the log of the falling factorial for real arguments.
#[inline]
pub fn log_falling_factorial<T: Real>(x: T, n: T) -> T {
    // Stan Math rejects x <= 0 (checked against Stan Math 5.4.0).
    if !(x.value() > 0.0) {
        return T::from_f64(f64::NAN);
    }
    lgamma(x + 1.0) - lgamma(x - n + 1.0)
}

/// `ln Γ(x + n) - ln Γ(x)`, the log of the rising factorial for real arguments.
#[inline]
pub fn log_rising_factorial<T: Real>(x: T, n: T) -> T {
    // Stan Math rejects x <= 0 (checked against Stan Math 5.4.0).
    if !(x.value() > 0.0) {
        return T::from_f64(f64::NAN);
    }
    lgamma(x + n) - lgamma(x)
}

/// Falling factorial `x (x - 1) ... (x - n + 1)`; one for `n = 0` and NaN for negative `n`.
pub fn falling_factorial<T: Real>(x: T, n: i32) -> T {
    if n < 0 {
        return T::from_f64(f64::NAN);
    }
    let mut r = T::one();
    for k in 0..n {
        r *= x - f64::from(k);
    }
    r
}

/// Rising factorial `x (x + 1) ... (x + n - 1)`; one for `n = 0` and NaN for negative `n`.
pub fn rising_factorial<T: Real>(x: T, n: i32) -> T {
    if n < 0 {
        return T::from_f64(f64::NAN);
    }
    let mut r = T::one();
    for k in 0..n {
        r *= x + f64::from(k);
    }
    r
}

/// Log multivariate gamma function of dimension `k`:
/// `k (k - 1) / 4 ln(pi) + sum_{j = 1}^{k} ln Γ(x + (1 - j) / 2)`.
pub fn lmgamma<T: Real>(k: i32, x: T) -> T {
    let kf = f64::from(k);
    let mut r = T::from_f64(kf * (kf - 1.0) * 0.25 * sp::LN_PI);
    for j in 1..=k {
        r += lgamma(x + (1.0 - f64::from(j)) * 0.5);
    }
    r
}

/// `a ln(b)` with the convention `0 ln(0) = 0`.
#[inline]
pub fn lmultiply<T: Real>(a: T, b: T) -> T {
    if a.is_const_zero() || (a.value() == 0.0 && b.value() == 0.0) {
        T::zero()
    } else {
        a * b.ln()
    }
}

/// `a ln(b)` with the convention `0 ln(0) = 0` (Stan's older name for [`lmultiply`]).
#[inline]
pub fn multiply_log<T: Real>(a: T, b: T) -> T {
    lmultiply(a, b)
}

/// `ln(inv_logit(a) - inv_logit(b))` for `a > b`.
///
/// Computed as `ln(1 - exp(b - a)) + ln(inv_logit(a)) - ln(1 + exp(b))`, which follows from
/// `inv_logit(a) - inv_logit(b) = (e^a - e^b) / ((1 + e^a)(1 + e^b))` and has no cancellation in
/// either tail.
#[inline]
pub fn log_inv_logit_diff<T: Real>(a: T, b: T) -> T {
    log1m_exp(b - a) + log_inv_logit(a) - log1p_exp(b)
}

/// Inverse complementary log log link, `1 - exp(-exp(x))`.
#[inline]
pub fn inv_cloglog<T: Real>(x: T) -> T {
    -(-x.exp()).exp_m1()
}

/// Log of the two component mixture `theta exp(lp1) + (1 - theta) exp(lp2)`.
#[inline]
pub fn log_mix<T: Real>(theta: T, lp1: T, lp2: T) -> T {
    log_sum_exp(theta.ln() + lp1, log1m(theta) + lp2)
}

/// Log of the mixture `sum_i theta_i exp(lp_i)`, as `log_sum_exp(ln(theta) + lp)`.
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn log_mix_slice<T: Real>(theta: &[T], lp: &[T]) -> T {
    assert_eq!(theta.len(), lp.len(), "log_mix: slices differ in length");
    let terms: Vec<T> = theta.iter().zip(lp).map(|(&t, &l)| t.ln() + l).collect();
    log_sum_exp_slice(&terms)
}

/// Beta function `B(a, b) = exp(ln B(a, b))`.
#[inline]
pub fn beta<T: Real>(a: T, b: T) -> T {
    if a.value() > 0.0 && b.value() > 0.0 {
        lbeta(a, b).exp()
    } else {
        // Stan Math 5.4.0 returns exp(lgamma(a) + lgamma(b) - lgamma(a + b)) with lgamma the log
        // of |Γ|, so the result is |Γ(a) Γ(b) / Γ(a + b)| for nonpositive arguments.
        (lgamma(a) + lgamma(b) - lgamma(a + b)).exp()
    }
}

/// `ln (n choose k)` for real arguments (Stan's name for [`lchoose`]).
#[inline]
pub fn binomial_coefficient_log<T: Real>(n: T, k: T) -> T {
    lchoose(n, k)
}

/// Binomial coefficient `n choose k` for integers; zero when `k > n`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `n` or `k` is negative or the result does not fit in `i64`.
pub fn choose(n: i64, k: i64) -> Result<i64> {
    const F: &str = "choose";
    if n < 0 {
        return Err(Error::parameter(F, "n", n as f64, "nonnegative"));
    }
    if k < 0 {
        return Err(Error::parameter(F, "k", k as f64, "nonnegative"));
    }
    if k > n {
        return Ok(0);
    }
    let k = k.min(n - k);
    // After step i, r = C(n - k + i, i): every intermediate is an integer and the division is
    // exact. The product before dividing is below i64::MAX^2, which fits in i128.
    let mut r: i128 = 1;
    for i in 1..=k {
        r = r * i128::from(n - k + i) / i128::from(i);
        if r > i128::from(i64::MAX) {
            return Err(Error::parameter(
                F,
                "n",
                n as f64,
                "small enough for the result to fit in an i64",
            ));
        }
    }
    Ok(r as i64)
}

/// Regularized incomplete beta function `I_x(a, b)` (Stan's name for [`beta_inc`]).
#[inline]
pub fn inc_beta<T: Real>(a: T, b: T, x: T) -> T {
    beta_inc(a, b, x)
}

/// Inverse complementary error function, `erfc⁻¹(x)` for `x` in `[0, 2]`.
///
/// Uses `erfc(y) = 2 Φ(-y sqrt(2))`, so `erfc⁻¹(x) = -Φ⁻¹(x / 2) / sqrt(2)`.
#[inline]
pub fn inv_erfc<T: Real>(x: T) -> T {
    -(inv_Phi(x * 0.5) * FRAC_1_SQRT_2)
}

/// Standard normal quantile function `Φ⁻¹(p)` (Stan's name for [`inv_Phi`]).
#[inline]
pub fn std_normal_qf<T: Real>(p: T) -> T {
    inv_Phi(p)
}

/// `Φ⁻¹(exp(lp))`, the standard normal quantile of a log probability `lp <= 0`.
///
/// Stays accurate where `exp(lp)` underflows and where it rounds to one. The derivative is
/// `Φ(x) / φ(x)`, the reciprocal of the inverse Mills ratio at the result.
#[inline]
pub fn std_normal_log_qf<T: Real>(lp: T) -> T {
    lp.map_unary(&StdNormalLogQf)
}

/// `Φ⁻¹(exp(lp))`; derivative `1 / m(x)` where `m` is the inverse Mills ratio.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StdNormalLogQf;

impl UnaryFn for StdNormalLogQf {
    fn value(&self, lp: f64) -> f64 {
        std_normal_log_qf_value(lp)
    }

    fn derivative<S: Real>(&self, _lp: S, y: S) -> S {
        y.map_unary(&InvMillsRatio).recip()
    }
}

fn std_normal_log_qf_value(lp: f64) -> f64 {
    if lp.is_nan() || lp > 0.0 {
        return f64::NAN;
    }
    if lp == 0.0 {
        return f64::INFINITY;
    }
    if lp == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    if lp > -LN_2 {
        // Upper half: Φ(-x) = 1 - exp(lp) = -expm1(lp), with no cancellation.
        return -sp::inv_Phi(-lp.exp_m1());
    }
    // Lower half: Abramowitz and Stegun 26.2.23 as a start, then Newton on ln Φ(x) - lp, which is
    // concave in x, so the iteration converges monotonically (as in `inv_Phi`).
    let t = (-2.0 * lp).sqrt();
    let mut x = -(t
        - (2.515517 + t * (0.802853 + t * 0.010328))
            / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))));
    for _ in 0..16 {
        let step = (sp::log_Phi(x) - lp) / sp::inv_mills_ratio(x);
        x -= step;
        if step.abs() <= 1e-16 * x.abs().max(1e-300) {
            break;
        }
    }
    x
}

/// Inverse of the regularized incomplete beta function: the `x` in `[0, 1]` with
/// `I_x(a, b) = p`.
///
/// Requires `a > 0`, `b > 0`, and `p` in `[0, 1]`; returns NaN otherwise. Derivatives follow from
/// the implicit function theorem: with `f = x^(a - 1) (1 - x)^(b - 1) / B(a, b)` the density,
/// `dx/dp = 1 / f`, `dx/da = -(dI/da) / f`, and `dx/db = -(dI/db) / f`.
#[inline]
pub fn inv_inc_beta<T: Real>(a: T, b: T, p: T) -> T {
    T::map_nary(&[a, b, p], &InvIncBeta)
}

/// Inverse incomplete beta function with arguments `[a, b, p]`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct InvIncBeta;

impl NaryFn for InvIncBeta {
    fn value(&self, v: &[f64]) -> f64 {
        inv_inc_beta_value(v[0], v[1], v[2])
    }

    fn partials<S: Real>(&self, v: &[S], y: S, needed: &[bool], out: &mut [S]) {
        let (a, b, p) = (v[0], v[1], v[2]);
        let density = ((a - 1.0) * y.ln() + (b - 1.0) * (-y).ln_1p() - lbeta(a, b)).exp();
        if needed[2] {
            out[2] = density.recip();
        }
        if needed[0] || needed[1] {
            let mut d = [S::zero(); 3];
            BetaInc.partials(&[a, b, y], p, &[needed[0], needed[1], false], &mut d);
            if needed[0] {
                out[0] = -(d[0] / density);
            }
            if needed[1] {
                out[1] = -(d[1] / density);
            }
        }
    }
}

/// Safeguarded Newton iteration for `I_x(a, b) = p`: Newton steps that leave the bracket are
/// replaced by bisection (geometric near zero, where the bracket starts at zero).
fn inv_inc_beta_value(a: f64, b: f64, p: f64) -> f64 {
    if !(a > 0.0) || !(b > 0.0) || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    let lb = sp::lbeta(a, b);
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    let mut x = a / (a + b);
    for _ in 0..1000 {
        let f = beta_inc(a, b, x) - p;
        if f == 0.0 {
            return x;
        }
        if f < 0.0 {
            lo = x;
        } else {
            hi = x;
        }
        let density = ((a - 1.0) * x.ln() + (b - 1.0) * (-x).ln_1p() - lb).exp();
        let newton = x - f / density;
        let next = if newton > lo && newton < hi {
            newton
        } else if lo == 0.0 {
            hi * 0.01
        } else {
            0.5 * (lo + hi)
        };
        if (next - x).abs() <= 2.0 * f64::EPSILON * next || hi - lo <= f64::EPSILON * hi {
            return next;
        }
        x = next;
    }
    x
}

/// Floating point remainder `a - b trunc(a / b)`, with the sign of `a`.
///
/// Partials are `1` and `-trunc(a / b)`.
#[inline]
pub fn fmod<T: Real>(a: T, b: T) -> T {
    a.map_binary(b, &Fmod)
}

/// `a % b`; partials `1` and `-trunc(a / b)`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Fmod;

impl BinaryFn for Fmod {
    fn value(&self, a: f64, b: f64) -> f64 {
        a % b
    }

    fn partial_a<S: Real>(&self, _a: S, _b: S, _y: S) -> S {
        S::one()
    }

    fn partial_b<S: Real>(&self, a: S, b: S, _y: S) -> S {
        S::from_f64(-(a.value() / b.value()).trunc())
    }
}

/// Step function: zero for `x < 0`, one otherwise (including zero and NaN).
#[inline]
pub fn step<T: Real>(x: T) -> T {
    if x.value() < 0.0 { T::zero() } else { T::one() }
}

/// Integer step function: one for `x > 0`, zero otherwise (including zero and NaN).
#[inline]
pub fn int_step<T: Real>(x: T) -> i32 {
    i32::from(x.value() > 0.0)
}

/// `x 2^n`.
#[inline]
pub fn ldexp<T: Real>(x: T, n: i32) -> T {
    x * 2f64.powi(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choose_matches_small_cases() {
        assert_eq!(choose(5, 2), Ok(10));
        assert_eq!(choose(5, 7), Ok(0));
        assert_eq!(choose(0, 0), Ok(1));
        assert_eq!(choose(62, 31), Ok(465_428_353_255_261_088));
        assert!(choose(100, 50).is_err());
        assert!(choose(-1, 0).is_err());
    }

    #[test]
    fn inverses_round_trip() {
        for &(a, b, p) in &[(2.0, 3.0, 0.3), (0.5, 0.5, 0.9), (5.0, 1.0, 1e-8)] {
            let x = inv_inc_beta(a, b, p);
            assert!((beta_inc(a, b, x) - p).abs() <= 1e-12 * p, "{a} {b} {p}");
        }
        for &lp in &[-1e-10, -0.1, -LN_2, -5.0, -800.0] {
            let x = std_normal_log_qf(lp);
            assert!((sp::log_Phi(x) - lp).abs() <= 1e-12 * lp.abs(), "{lp}");
        }
    }
}
