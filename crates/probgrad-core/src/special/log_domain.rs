//! Log-domain and logistic helpers.
//!
//! These avoid the overflow, underflow, and cancellation of the naive formulas. Each function
//! documents its behavior at the boundary of its domain.

/// `ln(1 + exp(x))`, also called softplus.
///
/// Uses `x + ln_1p(exp(-x))` for positive `x`, so large arguments do not overflow.
pub fn log1p_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// `ln(1 - exp(x))` for `x <= 0`.
///
/// Returns `-inf` at `x = 0` and `NaN` for `x > 0`. Switches between `ln(-expm1(x))` and
/// `ln_1p(-exp(x))` at `x = -ln 2`, following Maechler (2012), "Accurately computing
/// log(1 - exp(-|a|))".
pub fn log1m_exp(x: f64) -> f64 {
    if x > 0.0 {
        f64::NAN
    } else if x > -core::f64::consts::LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// `ln(1 - x)` for `x <= 1`.
pub fn log1m(x: f64) -> f64 {
    (-x).ln_1p()
}

/// `ln(1 + u) - u` for `u > -1`, accurate near zero where the naive form cancels.
///
/// Uses `ln(1 + u) = 2 atanh(z)` with `z = u / (2 + u)`, which gives
/// `ln(1 + u) - u = -z u + 2 (z^3 / 3 + z^5 / 5 + ...)` with no cancellation.
pub fn log1pmx(u: f64) -> f64 {
    if u.is_nan() || u < -1.0 {
        return f64::NAN;
    }
    if u.abs() > 0.5 {
        return u.ln_1p() - u;
    }
    let z = u / (2.0 + u);
    let z2 = z * z;
    let mut pow = z * z2;
    let mut sum = 0.0_f64;
    let mut k = 3.0_f64;
    // |z| <= 1/3, so 18 terms reach z^37 < 1e-17.
    while k < 40.0 {
        sum += pow / k;
        pow *= z2;
        k += 2.0;
    }
    -z * u + 2.0 * sum
}

/// `ln(exp(a) + exp(b))` without overflow.
///
/// `log_sum_exp(-inf, b) = b`. Two positive infinities give `+inf`.
pub fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a == f64::INFINITY || b == f64::INFINITY {
        return f64::INFINITY;
    }
    let m = a.max(b);
    m + (-(a - b).abs()).exp().ln_1p()
}

/// `ln(sum(exp(x)))` over a slice without overflow.
///
/// The empty sum is `-inf`.
pub fn log_sum_exp_slice(x: &[f64]) -> f64 {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if m.is_infinite() {
        // All entries -inf (or empty), or some entry +inf; NaN still propagates below.
        if x.iter().any(|v| v.is_nan()) {
            return f64::NAN;
        }
        return m;
    }
    m + x.iter().map(|&v| (v - m).exp()).sum::<f64>().ln()
}

/// `ln(exp(a) - exp(b))` for `a >= b`.
///
/// Returns `-inf` when `a == b`, `a` when `b = -inf`, and `NaN` when `a < b`.
pub fn log_diff_exp(a: f64, b: f64) -> f64 {
    if b > a {
        return f64::NAN;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a == f64::INFINITY {
        return if b == f64::INFINITY {
            f64::NAN
        } else {
            f64::INFINITY
        };
    }
    a + log1m_exp(b - a)
}

/// Logistic sigmoid `1 / (1 + exp(-x))`.
pub fn inv_logit(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Log odds `ln(p / (1 - p))` for `p` in `[0, 1]`.
pub fn logit(p: f64) -> f64 {
    p.ln() - (-p).ln_1p()
}

/// `ln(inv_logit(x))`, also called log-sigmoid.
pub fn log_inv_logit(x: f64) -> f64 {
    -log1p_exp(-x)
}

/// `ln(1 - inv_logit(x))`.
pub fn log1m_inv_logit(x: f64) -> f64 {
    -log1p_exp(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, rtol: f64) -> bool {
        a == b || (a - b).abs() <= rtol * b.abs().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn log1p_exp_is_stable() {
        assert_eq!(log1p_exp(1000.0), 1000.0);
        assert!(close(log1p_exp(-1000.0), 0.0, 0.0) || log1p_exp(-1000.0) < 1e-300);
        assert!(close(log1p_exp(-40.0), 4.248354255291589e-18, 1e-15));
        assert!(close(log1p_exp(0.0), core::f64::consts::LN_2, 1e-16));
        assert!(close(log1p_exp(30.0), 30.000000000000092, 1e-16));
    }

    #[test]
    fn log1m_exp_boundaries() {
        assert_eq!(log1m_exp(0.0), f64::NEG_INFINITY);
        assert!(log1m_exp(1.0).is_nan());
        // ln(1 - exp(-1e-20)) = ln(1e-20) to double precision.
        assert!(close(log1m_exp(-1e-20), -46.051701859880914, 1e-15));
        // ln(1 - exp(-50)) = -exp(-50) to double precision.
        assert!(close(log1m_exp(-50.0), -1.9287498479639178e-22, 1e-15));
        assert!(close(log1m_exp(-1.0), -0.4586751453870819, 1e-15));
    }

    #[test]
    fn log_sum_exp_handles_infinities() {
        assert_eq!(log_sum_exp(f64::NEG_INFINITY, 2.0), 2.0);
        assert_eq!(log_sum_exp(f64::INFINITY, f64::INFINITY), f64::INFINITY);
        assert!(close(
            log_sum_exp(1000.0, 1000.0),
            1000.0 + core::f64::consts::LN_2,
            1e-16
        ));
        assert!(log_sum_exp(f64::NAN, 1.0).is_nan());
        assert_eq!(log_sum_exp_slice(&[]), f64::NEG_INFINITY);
        assert_eq!(
            log_sum_exp_slice(&[f64::NEG_INFINITY; 3]),
            f64::NEG_INFINITY
        );
        assert!(close(
            log_sum_exp_slice(&[1.0, 2.0, 3.0]),
            3.40760596444438,
            1e-15
        ));
        assert!(log_sum_exp_slice(&[f64::INFINITY, f64::NAN]).is_nan());
    }

    #[test]
    fn log_diff_exp_boundaries() {
        assert_eq!(log_diff_exp(1.0, 1.0), f64::NEG_INFINITY);
        assert_eq!(log_diff_exp(1.0, f64::NEG_INFINITY), 1.0);
        assert!(log_diff_exp(1.0, 2.0).is_nan());
        assert!(close(log_diff_exp(2.0, 1.0), 1.5413248546129181, 1e-15));
    }

    #[test]
    fn logistic_functions() {
        assert_eq!(inv_logit(0.0), 0.5);
        assert!(close(inv_logit(-800.0), 0.0, 0.0) || inv_logit(-800.0) < 1e-300);
        assert_eq!(inv_logit(800.0), 1.0);
        assert!(close(inv_logit(-30.0), 9.357622968839299e-14, 1e-15));
        assert!(close(logit(0.25), -1.0986122886681098, 1e-15));
        assert!(close(log_inv_logit(-800.0), -800.0, 1e-16));
        assert!(close(log1m_inv_logit(800.0), -800.0, 1e-16));
        assert_eq!(logit(0.0), f64::NEG_INFINITY);
        assert_eq!(logit(1.0), f64::INFINITY);
    }
}
