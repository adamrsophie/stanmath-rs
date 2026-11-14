//! The real branches of the Lambert W function, the inverse of `w exp(w)` (DLMF 4.13).
//!
//! Starting values: the series about the branch point `x = -1/e` in `p = ±sqrt(2 (e x + 1))`,
//! `W = -1 + p - p^2 / 3 + 11 p^3 / 72` (DLMF §4.13), and the asymptotic form
//! `W ≈ L1 - L2 + L2 / L1` with `L1 = ln |x|`, `L2 = ln |L1|` (DLMF §4.13). Away from the
//! branch point the equation is solved in logarithmic form, `w + ln |w| = ln |x|`, by Newton's
//! method, which cannot overflow; near it by Halley's method on `w exp(w) - x = 0`.

use core::f64::consts::E;

/// `1 / e`.
const INV_E: f64 = 0.367_879_441_171_442_33;

/// Principal branch `W_0(x)` for `x >= -1/e`; `NaN` below.
pub fn lambert_w0(x: f64) -> f64 {
    if x.is_nan() || x < -INV_E {
        return f64::NAN;
    }
    if x == 0.0 || x == f64::INFINITY {
        return x;
    }
    if x < -0.25 {
        return halley(x, branch_series(x, 1.0));
    }
    if x < 3.0 {
        return halley(x, x.ln_1p());
    }
    let l1 = x.ln();
    let l2 = l1.ln();
    newton_log(l1, l1 - l2 + l2 / l1)
}

/// Lower branch `W_{-1}(x)` for `-1/e <= x < 0`; `-inf` at zero and `NaN` elsewhere.
pub fn lambert_wm1(x: f64) -> f64 {
    if x.is_nan() || x < -INV_E || x > 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < -0.25 {
        return halley(x, branch_series(x, -1.0));
    }
    let l1 = (-x).ln();
    let l2 = (-l1).ln();
    newton_log(l1, l1 - l2 + l2 / l1)
}

/// Series about the branch point; `sign` is `1` for `W_0` and `-1` for `W_{-1}`.
fn branch_series(x: f64, sign: f64) -> f64 {
    let p = sign * (2.0 * (E * x + 1.0)).max(0.0).sqrt();
    -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p * p * p
}

/// Halley's iteration for `w exp(w) = x`.
fn halley(x: f64, mut w: f64) -> f64 {
    for _ in 0..64 {
        let wp1 = w + 1.0;
        if wp1 == 0.0 {
            break;
        }
        let ew = w.exp();
        let f = w * ew - x;
        let denom = ew * wp1 - (w + 2.0) * f / (2.0 * wp1);
        let dw = f / denom;
        if !dw.is_finite() {
            break;
        }
        w -= dw;
        if !(dw.abs() > 4.0 * f64::EPSILON * w.abs()) {
            break;
        }
    }
    w
}

/// Newton's iteration for `w + ln |w| = log_x`, valid when `|w| > 1`.
fn newton_log(log_x: f64, mut w: f64) -> f64 {
    for _ in 0..64 {
        let g = w + w.abs().ln() - log_x;
        let dw = g / (1.0 + w.recip());
        w -= dw;
        if !(dw.abs() > 4.0 * f64::EPSILON * w.abs()) {
            break;
        }
    }
    w
}
