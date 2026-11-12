//! Bessel functions of integer order: `J_n`, `Y_n`, `I_n`, `K_n`.
//!
//! Algorithms:
//!
//! * `J_n`: the power series (DLMF 10.2.2) for `|x| <= 1`; Hankel's asymptotic expansion
//!   (DLMF 10.17.3) for `x > 25 + n^2 / 2`; otherwise Miller's backward recurrence (DLMF §10.74
//!   with recurrence 10.6.1), normalized by `J_0 + 2 (J_2 + J_4 + ...) = 1` (DLMF 10.12).
//! * `Y_n`: Hankel's expansion (DLMF 10.17.4) for large `x`; otherwise Neumann's expansion
//!   `Y_0 = (2/pi)(ln(x/2) + γ) J_0 - (4/pi) sum_k (-1)^k J_{2k} / k` (Abramowitz and Stegun
//!   9.1.88), its derivative for `Y_1 = -Y_0'` (DLMF 10.6.3), and forward recurrence
//!   (DLMF 10.6.1), which is stable for `Y`.
//! * `I_n`: the power series (DLMF 10.25.2), whose terms are all positive.
//! * `K_n`: the trapezoidal rule on `K_n(x) = ∫_0^∞ exp(-x cosh t) cosh(n t) dt` (DLMF 10.32.9).
//!   The integrand is analytic in a strip, so the rule converges geometrically; the step is
//!   scaled with the width of the integrand's peak.

use core::f64::consts::{FRAC_2_PI, PI};

use super::{cospi, lgamma, sinpi};

/// Euler's constant γ.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// Bessel function of the first kind `J_n(x)`.
pub fn bessel_j(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let m = n.unsigned_abs();
    let odd = m % 2 == 1;
    // J_{-n} = (-1)^n J_n and J_n(-x) = (-1)^n J_n(x) (DLMF §10.4, §10.11).
    let negate = odd && ((n < 0) != (x < 0.0));
    let v = j_nonnegative(m as usize, x.abs());
    if negate { -v } else { v }
}

/// Bessel function of the second kind `Y_n(x)` for `x > 0`; `NaN` otherwise.
pub fn bessel_y(n: i32, x: f64) -> f64 {
    if !(x > 0.0) {
        return f64::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let m = n.unsigned_abs() as usize;
    let v = y_positive(m, x);
    // Y_{-n} = (-1)^n Y_n (DLMF §10.4).
    if n < 0 && m % 2 == 1 { -v } else { v }
}

/// Modified Bessel function of the first kind `I_n(x)`.
pub fn bessel_i(n: i32, x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let m = n.unsigned_abs() as usize;
    let v = i_nonnegative(m, x.abs());
    // I_{-n} = I_n and I_n(-x) = (-1)^n I_n(x) (DLMF §10.27, §10.34).
    if x < 0.0 && m % 2 == 1 { -v } else { v }
}

/// Modified Bessel function of the second kind `K_n(x)` for `x >= 0` (infinite at zero); `NaN`
/// for negative `x`.
pub fn bessel_k(n: i32, x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x.is_infinite() {
        return 0.0;
    }
    // K_{-n} = K_n (DLMF §10.27).
    let nu = f64::from(n.unsigned_abs());
    // Peak of exp(nu t - x cosh t) and the width of the peak, (x^2 + nu^2)^(-1/4).
    let peak = (nu / x).asinh();
    let h = 0.125f64.min(0.5 / (x * x + nu * nu).sqrt().sqrt());
    // ln exp(-x cosh t) = -x - 2 x sinh(t/2)^2, without the cancellation in cosh t - 1.
    let log_f = |t: f64| {
        let s = (0.5 * t).sinh();
        -2.0 * x * s * s - x
    };
    let log_max = log_f(peak) + nu * peak;
    let mut sum = 0.5 * (log_f(0.0) - log_max).exp();
    let mut k = 1u32;
    loop {
        let t = f64::from(k) * h;
        let a = log_f(t);
        let term = 0.5 * ((a + nu * t - log_max).exp() + (a - nu * t - log_max).exp());
        sum += term;
        if (t > peak && term < 1e-18 * sum) || k >= 1_000_000 {
            break;
        }
        k += 1;
    }
    (log_max + (h * sum).ln()).exp()
}

/// `J_n(x)` for `x >= 0`.
fn j_nonnegative(n: usize, x: f64) -> f64 {
    if x.is_infinite() {
        return 0.0;
    }
    if x <= 1.0 {
        return j_series(n, x);
    }
    let nf = n as f64;
    if x > 25.0 + 0.5 * nf * nf {
        return hankel(nf, x).0;
    }
    j_values(x, n)[n]
}

/// `J_n(x)` by its power series; accurate for `0 <= x <= 1`.
fn j_series(n: usize, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    let nf = n as f64;
    let front = (nf * (0.5 * x).ln() - lgamma(nf + 1.0)).exp();
    let y = -0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    for k in 1..200 {
        let kf = f64::from(k);
        term *= y / (kf * (nf + kf));
        sum += term;
        if term.abs() < 1e-17 * sum.abs() {
            break;
        }
    }
    front * sum
}

/// `J_0(x), ..., J_nmax(x)` for `x >= 0`.
fn j_values(x: f64, nmax: usize) -> Vec<f64> {
    if x <= 1.0 {
        return (0..=nmax).map(|k| j_series(k, x)).collect();
    }
    // Miller's algorithm: start far enough beyond max(n, x) that J_start is negligible.
    let big = (nmax as f64).max(x);
    let mut start = (big + 30.0 + (40.0 * big).sqrt()).ceil() as usize;
    start += start % 2;
    let mut j = vec![0.0; start + 2];
    j[start] = 1.0;
    for k in (1..=start).rev() {
        j[k - 1] = (2.0 * k as f64 / x) * j[k] - j[k + 1];
        if j[k - 1].abs() > 1e250 {
            for v in &mut j[k - 1..] {
                *v *= 1e-250;
            }
        }
    }
    let mut norm = j[0];
    for k in (2..=start).step_by(2) {
        norm += 2.0 * j[k];
    }
    j.truncate(nmax + 1);
    for v in &mut j {
        *v /= norm;
    }
    j
}

/// `Y_n(x)` for `x > 0` and `n >= 0`.
fn y_positive(n: usize, x: f64) -> f64 {
    let nf = n as f64;
    if x > 25.0 + 0.5 * nf * nf {
        return hankel(nf, x).1;
    }
    let (mut y0, mut y1) = y_zero_one(x);
    if n == 0 {
        return y0;
    }
    for k in 1..n {
        let y2 = (2.0 * k as f64 / x) * y1 - y0;
        y0 = y1;
        y1 = y2;
        if y1.is_infinite() {
            break;
        }
    }
    y1
}

/// `(Y_0(x), Y_1(x))` by Neumann's expansion and its derivative.
fn y_zero_one(x: f64) -> (f64, f64) {
    let big = x.max(1.0);
    let kmax = ((big + 30.0 + (40.0 * big).sqrt()) / 2.0).ceil() as usize;
    let j = j_values(x, 2 * kmax + 1);
    let log_term = (0.5 * x).ln() + EULER_GAMMA;
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    for k in 1..=kmax {
        let sign = if k % 2 == 1 { -1.0 } else { 1.0 };
        let kf = k as f64;
        s0 += sign * j[2 * k] / kf;
        // J_{2k}' = (J_{2k-1} - J_{2k+1}) / 2 (DLMF 10.6.1).
        s1 += sign * (j[2 * k - 1] - j[2 * k + 1]) / kf;
    }
    let y0 = FRAC_2_PI * (log_term * j[0] - 2.0 * s0);
    let y1 = FRAC_2_PI * (log_term * j[1] - j[0] / x + s1);
    (y0, y1)
}

/// `(J_nu(x), Y_nu(x))` by Hankel's asymptotic expansion, for `x` large relative to `nu^2`.
fn hankel(nu: f64, x: f64) -> (f64, f64) {
    let mu = 4.0 * nu * nu;
    // P = a_0 - a_2 / x^2 + ..., Q = a_1 / x - a_3 / x^3 + ..., with
    // a_k(nu) = (mu - 1)(mu - 9)...(mu - (2k - 1)^2) / (k! 8^k).
    let mut p = 1.0;
    let mut q = 0.0;
    let mut term = 1.0f64;
    let mut prev = f64::INFINITY;
    for k in 1..200u32 {
        let odd = f64::from(2 * k - 1);
        let next = term * (mu - odd * odd) / (f64::from(k) * 8.0 * x);
        let size = next.abs();
        // Stop before the asymptotic series starts to diverge.
        if size > prev {
            break;
        }
        prev = size;
        term = next;
        match k % 4 {
            1 => q += term,
            2 => p -= term,
            3 => q -= term,
            _ => p += term,
        }
        if size < 1e-17 * (p.abs() + q.abs()) {
            break;
        }
    }
    // ω = x - (nu / 2 + 1 / 4) pi, formed with exact reduction of the phase shift.
    let phase = 0.5 * nu + 0.25;
    let (s, c) = x.sin_cos();
    let (sp, cp) = (sinpi(phase), cospi(phase));
    let cos_w = c * cp + s * sp;
    let sin_w = s * cp - c * sp;
    let amp = (2.0 / (PI * x)).sqrt();
    (amp * (p * cos_w - q * sin_w), amp * (p * sin_w + q * cos_w))
}

/// `I_n(x)` for `x >= 0`.
fn i_nonnegative(n: usize, x: f64) -> f64 {
    if x == 0.0 {
        return if n == 0 { 1.0 } else { 0.0 };
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let nf = n as f64;
    let y = 0.25 * x * x;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 0.0;
    loop {
        k += 1.0;
        term *= y / (k * (nf + k));
        sum += term;
        if sum.is_infinite() {
            return f64::INFINITY;
        }
        if term < 1e-17 * sum || k > 1e6 {
            break;
        }
    }
    (nf * (0.5 * x).ln() - lgamma(nf + 1.0) + sum.ln()).exp()
}
