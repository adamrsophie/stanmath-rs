//! Error functions and the standard normal distribution.
//!
//! `erf` and `erfc` come from the `libm` crate. `erfcx` multiplies `erfc` by an `exp(x^2)` that is
//! evaluated without the usual loss of accuracy, and switches to the asymptotic series (DLMF
//! 7.12.1) where `erfc` underflows. `log_Phi` uses `erfcx` in the lower tail so that it stays
//! accurate far beyond the point where `Phi` underflows. `inv_Phi` refines an Abramowitz and
//! Stegun 26.2.23 starting value with Newton steps on `log_Phi`, which converges to full double
//! precision in both tails.

use super::{SQRT_2, SQRT_PI, log1m};
use core::f64::consts::{FRAC_2_SQRT_PI, LN_2};

/// Error function.
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Complementary error function `1 - erf(x)`, accurate for large `x`.
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// `exp(x^2)` with a relative error of a few ulps, by splitting `x` so that the square of the
/// leading part is exact.
fn exp_x2(x: f64) -> f64 {
    let x = x.abs();
    let hi = (x * 4096.0).trunc() / 4096.0;
    let lo = x - hi;
    (hi * hi).exp() * (lo * (2.0 * hi + lo)).exp()
}

/// Scaled complementary error function `exp(x^2) erfc(x)`.
///
/// Overflows to `+inf` for `x < -26.6`, where the true value exceeds `f64::MAX`.
pub fn erfcx(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.0 {
        if x < -26.7 {
            return f64::INFINITY;
        }
        return 2.0 * exp_x2(x) - erfcx(-x);
    }
    if x < 26.0 {
        return erfc(x) * exp_x2(x);
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    // erfcx(x) ~ 1 / (x sqrt(pi)) * sum_k (-1)^k (2k - 1)!! / (2 x^2)^k
    let inv_2x2 = 1.0 / (2.0 * x * x);
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..20 {
        term *= -(2.0 * f64::from(k) - 1.0) * inv_2x2;
        sum += term;
        if term.abs() < 1e-17 {
            break;
        }
    }
    sum / (x * SQRT_PI)
}

/// Standard normal density `exp(-x^2 / 2) / sqrt(2 pi)`.
///
/// The square is split as in [`erfcx`] so that the density keeps full relative accuracy in the
/// tails, where the rounding error of `x * x` would otherwise be multiplied by `x^2 / 2`.
pub fn std_normal_pdf(x: f64) -> f64 {
    let x = x.abs();
    if !x.is_finite() {
        return if x.is_nan() { f64::NAN } else { 0.0 };
    }
    let hi = (x * 4096.0).trunc() / 4096.0;
    let lo = x - hi;
    super::INV_SQRT_2PI * (-0.5 * hi * hi).exp() * (-0.5 * lo * (2.0 * hi + lo)).exp()
}

/// `erfc(x / sqrt(2))` without the error from rounding `x / sqrt(2)`.
///
/// `erfc(z)` has relative condition number `2 z^2`, so the half ulp lost in forming `z` costs about
/// `log10(z^2)` digits in the tails. The rounding error `δ` of `z` is recovered with a fused
/// multiply-add and a two-term split of `1 / sqrt(2)`, then removed with the first-order
/// correction `erfc(z + δ) = erfc(z) (1 - δ 2 exp(-z^2) / (sqrt(pi) erfc(z)))`.
fn erfc_scaled(x: f64) -> f64 {
    const C_HI: f64 = core::f64::consts::FRAC_1_SQRT_2;
    const C_LO: f64 = -4.833_646_656_726_457e-17;
    let z = x * C_HI;
    let delta = x.mul_add(C_HI, -z) + x * C_LO;
    let e = erfc(z);
    if e == 0.0 || !delta.is_finite() {
        return e;
    }
    e - delta * FRAC_2_SQRT_PI * (-z * z).exp()
}

/// Standard normal cumulative distribution function `Φ(x)`.
#[allow(non_snake_case)]
pub fn Phi(x: f64) -> f64 {
    0.5 * erfc_scaled(-x)
}

/// `ln Φ(x)`, accurate in both tails.
///
/// Stays finite down to `x` near `-1.9e154`, far below where `Phi(x)` underflows (`x < -38.5`).
#[allow(non_snake_case)]
pub fn log_Phi(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        // Φ(x) = 1 - Φ(-x); ln(1 - q) is accurate for small q, unlike ln of a number near 1.
        return log1m(0.5 * erfc_scaled(x));
    }
    if x > -20.0 {
        return (0.5 * erfc_scaled(-x)).ln();
    }
    if x == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    // Φ(x) = erfcx(-x / sqrt 2) exp(-x^2 / 2) / 2
    erfcx(-x / SQRT_2).ln() - 0.5 * x * x - LN_2
}

/// Inverse Mills ratio `φ(x) / Φ(x)`, the derivative of [`log_Phi`].
pub fn inv_mills_ratio(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x > 0.0 {
        // Φ(x) is between 1/2 and 1 here; erfcx(-x / sqrt 2) would amplify the rounding of its
        // argument through exp(x^2 / 2).
        return std_normal_pdf(x) / Phi(x);
    }
    // φ(x) / Φ(x) = sqrt(2 / pi) / erfcx(-x / sqrt 2); erfcx is well conditioned for positive
    // arguments.
    FRAC_2_SQRT_PI / SQRT_2 / erfcx(-x / SQRT_2)
}

/// Inverse of the standard normal CDF, `Φ⁻¹(p)`.
///
/// Returns `-inf` at 0, `+inf` at 1, and `NaN` outside `[0, 1]`.
#[allow(non_snake_case)]
pub fn inv_Phi(p: f64) -> f64 {
    if p.is_nan() || !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    // Solve in the lower tail; 1 - p is exact for p >= 0.5.
    let (q, sign) = if p < 0.5 { (p, 1.0) } else { (1.0 - p, -1.0) };
    let log_q = q.ln();
    // Abramowitz and Stegun 26.2.23, absolute error below 4.5e-4.
    let t = (-2.0 * log_q).sqrt();
    let mut x = -(t
        - (2.515517 + t * (0.802853 + t * 0.010328))
            / (1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308))));
    // Newton on g(x) = ln Φ(x) - ln q, with g'(x) = φ(x) / Φ(x). g is concave, so the iteration
    // converges monotonically from this starting point.
    for _ in 0..8 {
        let step = (log_Phi(x) - log_q) / inv_mills_ratio(x);
        x -= step;
        if step.abs() <= 1e-16 * x.abs().max(1e-300) {
            break;
        }
    }
    sign * x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn erfcx_branches_are_continuous() {
        for &x in &[25.9, 26.0, 26.1] {
            let a = erfcx(x);
            let b = erfcx(x - 1e-9);
            assert!(rel(a, b) < 1e-9, "{x}");
        }
        assert_eq!(erfcx(f64::INFINITY), 0.0);
        assert_eq!(erfcx(-30.0), f64::INFINITY);
    }

    #[test]
    fn log_phi_branches_are_continuous() {
        for &x in &[-20.0, 5.0] {
            let a = log_Phi(x - 1e-12);
            let b = log_Phi(x + 1e-12);
            assert!((a - b).abs() < 1e-9 * a.abs().max(1e-20), "{x}: {a} {b}");
        }
        assert_eq!(log_Phi(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(log_Phi(f64::INFINITY), 0.0);
    }

    #[test]
    fn inv_phi_round_trips() {
        for &p in &[
            1e-300,
            1e-20,
            1e-5,
            0.025,
            0.3,
            0.5,
            0.7,
            0.975,
            1.0 - 1e-10,
        ] {
            let x = inv_Phi(p);
            assert!(rel(Phi(x), p) < 1e-13, "{p}: {x} -> {}", Phi(x));
        }
        assert_eq!(inv_Phi(0.0), f64::NEG_INFINITY);
        assert_eq!(inv_Phi(1.0), f64::INFINITY);
        assert!(inv_Phi(1.5).is_nan());
    }
}
