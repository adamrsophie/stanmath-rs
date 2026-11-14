//! Gamma, beta, and polygamma functions.
//!
//! Algorithms: recurrence to a large argument followed by the Stirling or asymptotic series with
//! Bernoulli number coefficients (DLMF 5.11.1, 5.15.8), and the reflection formulas (DLMF 5.5.4,
//! 5.15.6) for negative arguments. `lgamma` and `tgamma` use the `libm` crate (a port of musl).

use super::{HALF_LN_2PI, log1m};
use core::f64::consts::PI;

/// `ln |Γ(x)|`. Infinite at the poles `x = 0, -1, -2, ...`.
pub fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

/// `Γ(x)`.
pub fn tgamma(x: f64) -> f64 {
    libm::tgamma(x)
}

/// Euler's constant γ.
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

/// `ζ(2), ζ(3), ..., ζ(40)`, computed with mpmath at 40 digits.
const ZETA: [f64; 39] = [
    1.644_934_066_848_226_4,
    1.202_056_903_159_594_2,
    1.082_323_233_711_138_1,
    1.036_927_755_143_37,
    1.017_343_061_984_449_2,
    1.008_349_277_381_923,
    1.004_077_356_197_944_4,
    1.002_008_392_826_082_1,
    1.000_994_575_127_818,
    1.000_494_188_604_119_4,
    1.000_246_086_553_308,
    1.000_122_713_347_578_5,
    1.000_061_248_135_058_8,
    1.000_030_588_236_307,
    1.000_015_282_259_408_6,
    1.000_007_637_197_637_9,
    1.000_003_817_293_265,
    1.000_001_908_212_716_5,
    1.000_000_953_962_033_8,
    1.000_000_476_932_986_9,
    1.000_000_238_450_502_7,
    1.000_000_119_219_926,
    1.000_000_059_608_189,
    1.000_000_029_803_503_4,
    1.000_000_014_901_554_9,
    1.000_000_007_450_711_8,
    1.000_000_003_725_334,
    1.000_000_001_862_659_8,
    1.000_000_000_931_327_5,
    1.000_000_000_465_662_8,
    1.000_000_000_232_831,
    1.000_000_000_116_415_5,
    1.000_000_000_058_207_7,
    1.000_000_000_029_103_8,
    1.000_000_000_014_552,
    1.000_000_000_007_276,
    1.000_000_000_003_638,
    1.000_000_000_001_819,
    1.000_000_000_000_909_5,
];

/// `ln Γ(1 + a)`, accurate for small `|a|`, where forming `1 + a` first would round away the low
/// digits of `a` (costing about 1e-13 relative accuracy at `a = 1e-3`).
///
/// Uses `ln Γ(1 + a) = -γ a + sum_{k >= 2} (-1)^k ζ(k) a^k / k` (DLMF 5.7.3) for `|a| < 0.2`.
pub fn lgamma1p(a: f64) -> f64 {
    if a.is_nan() || a.abs() >= 0.2 {
        return lgamma(1.0 + a);
    }
    let mut pow = -a;
    let mut terms = [0.0; 39];
    for (k, (t, z)) in terms.iter_mut().zip(ZETA).enumerate() {
        pow *= -a;
        *t = z * pow / (k + 2) as f64;
    }
    // Sum the smallest terms first.
    terms.iter().rev().sum::<f64>() - EULER_GAMMA * a
}

/// `sin(pi x)` with exact argument reduction, so that integer `x` gives exactly zero.
pub fn sinpi(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    let r = x - 2.0 * (x / 2.0).round(); // r in [-1, 1]
    let (r, sign) = if r > 0.5 {
        (1.0 - r, 1.0)
    } else if r < -0.5 {
        (-1.0 - r, 1.0)
    } else {
        (r, 1.0)
    };
    sign * (PI * r).sin()
}

/// `cos(pi x)` with exact argument reduction.
pub fn cospi(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    sinpi(x + 0.5)
}

/// Bernoulli numbers `B_2, B_4, ..., B_20`.
const BERNOULLI: [f64; 10] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
    43867.0 / 798.0,
    -174611.0 / 330.0,
];

/// Argument above which the asymptotic series are used.
const ASYMPTOTIC: f64 = 10.0;

/// Digamma `ψ(x) = d/dx ln Γ(x)`. `NaN` at the poles.
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // ψ(x) = ψ(1 - x) - pi cot(pi x)
        return digamma(1.0 - x) - PI * cospi(x) / sinpi(x);
    }
    let mut x = x;
    let mut acc = 0.0;
    while x < ASYMPTOTIC {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let inv2 = 1.0 / (x * x);
    // sum_{k>=1} B_{2k} / (2k x^{2k}), seven terms
    let mut series = 0.0;
    let mut pow = inv2;
    for (k, b) in BERNOULLI.iter().enumerate().take(7) {
        series += b / (2.0 * (k + 1) as f64) * pow;
        pow *= inv2;
    }
    acc + x.ln() - 0.5 / x - series
}

/// Trigamma `ψ'(x)`. `NaN` at the poles.
pub fn trigamma(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    if x <= 0.0 {
        if x == x.floor() {
            return f64::NAN;
        }
        // ψ'(1 - x) + ψ'(x) = pi^2 / sin^2(pi x)
        let s = sinpi(x);
        return PI * PI / (s * s) - trigamma(1.0 - x);
    }
    let mut x = x;
    let mut acc = 0.0;
    while x < ASYMPTOTIC {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // 1/x + 1/(2x^2) + sum_{k>=1} B_{2k} / x^{2k+1}
    let mut series = 0.0;
    let mut pow = inv2 * inv;
    for b in BERNOULLI.iter().take(8) {
        series += b * pow;
        pow *= inv2;
    }
    acc + inv + 0.5 * inv2 + series
}

/// Polygamma `ψ^(n)(x)`, the `n`th derivative of digamma.
///
/// `n = 0` is [`digamma`] and `n = 1` is [`trigamma`]. For `n >= 2`, negative arguments are
/// handled by the upward recurrence, which costs `O(|x|)`; arguments below `-1e5` and the poles
/// return `NaN`.
pub fn polygamma(n: u32, x: f64) -> f64 {
    match n {
        0 => return digamma(x),
        1 => return trigamma(x),
        _ => {}
    }
    if x.is_nan() || x < -1e5 || (x <= 0.0 && x == x.floor()) {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    let nf = f64::from(n);
    let sign = if n % 2 == 1 { 1.0 } else { -1.0 };
    let n_fact = tgamma(nf + 1.0);
    // ψ^(n)(x) = ψ^(n)(x + 1) + (-1)^(n+1) n! / x^(n+1)
    let threshold = ASYMPTOTIC + 5.0 + nf;
    let mut x = x;
    let mut acc = 0.0;
    while x < threshold {
        acc += x.powi(-(n as i32) - 1);
        x += 1.0;
    }
    // Asymptotic: (n-1)!/x^n + n!/(2 x^(n+1)) + sum_k B_{2k} (2k+n-1)! / ((2k)! x^(2k+n))
    let mut series = tgamma(nf) / x.powf(nf) + n_fact / (2.0 * x.powf(nf + 1.0));
    for (k, b) in BERNOULLI.iter().enumerate() {
        let two_k = 2.0 * (k + 1) as f64;
        let log_coef = lgamma(two_k + nf) - lgamma(two_k + 1.0) - (two_k + nf) * x.ln();
        let term = b * log_coef.exp();
        series += term;
        if term.abs() < 1e-17 * series.abs() {
            break;
        }
    }
    sign * (n_fact * acc + series)
}

/// The Stirling correction `δ(x) = ln Γ(x) - [(x - 1/2) ln x - x + ln(2 pi) / 2]` for `x > 0`.
///
/// Used to evaluate gamma ratios without cancellation.
pub fn lgamma_stirling_diff(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        return lgamma(x) - lgamma_stirling(x);
    }
    // δ(x) = δ(x + 1) + g(x) with g(x) = (x + 1/2) ln(1 + 1/x) - 1, and with t = 1 / (2x + 1),
    // g(x) = t^2 / 3 + t^4 / 5 + t^6 / 7 + ..., which has no cancellation.
    let mut x = x;
    let mut acc = 0.0;
    while x < ASYMPTOTIC {
        acc += stirling_step(x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    // sum_k B_{2k} / (2k (2k - 1) x^{2k - 1})
    let mut series = 0.0;
    let mut pow = inv;
    for (k, b) in BERNOULLI.iter().enumerate() {
        let two_k = 2.0 * (k + 1) as f64;
        series += b / (two_k * (two_k - 1.0)) * pow;
        pow *= inv2;
    }
    acc + series
}

/// `(x + 1/2) ln(1 + 1/x) - 1 = sum_{m >= 1} t^(2m) / (2m + 1)` with `t = 1 / (2x + 1)`, for
/// `x >= 1` (so `t <= 1/3`).
fn stirling_step(x: f64) -> f64 {
    let t = 1.0 / (2.0 * x + 1.0);
    let t2 = t * t;
    let mut pow = t2;
    let mut sum = 0.0;
    let mut k = 3.0;
    while k < 70.0 {
        sum += pow / k;
        pow *= t2;
        k += 2.0;
    }
    sum
}

/// The Stirling approximation `(x - 1/2) ln x - x + ln(2 pi) / 2` to `ln Γ(x)`.
pub fn lgamma_stirling(x: f64) -> f64 {
    (x - 0.5) * x.ln() - x + HALF_LN_2PI
}

/// `ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a + b)` for `a, b >= 0`, without cancellation for large
/// arguments.
///
/// Returns `+inf` when either argument is zero and `NaN` for negative arguments.
pub fn lbeta(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    let (x, y) = if a < b { (a, b) } else { (b, a) };
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if y == f64::INFINITY {
        return f64::NEG_INFINITY;
    }
    if y < ASYMPTOTIC {
        return lgamma(x) + lgamma(y) - lgamma(x + y);
    }
    let corr_y = lgamma_stirling_diff(y) - lgamma_stirling_diff(x + y);
    if x < ASYMPTOTIC {
        // ln Γ(y) - ln Γ(x + y) by Stirling, rearranged to avoid cancellation.
        return lgamma(x) - (y - 0.5) * (x / y).ln_1p() - x * (x + y).ln() + x + corr_y;
    }
    let ratio = x / (x + y);
    HALF_LN_2PI - 0.5 * y.ln()
        + (x - 0.5) * ratio.ln()
        + y * log1m(ratio)
        + lgamma_stirling_diff(x)
        + corr_y
}

/// `ln (n choose k)` for real `n >= k >= 0`.
pub fn lchoose(n: f64, k: f64) -> f64 {
    if n.is_nan() || k.is_nan() || k < 0.0 || k > n {
        return f64::NAN;
    }
    -(n + 1.0).ln() - lbeta(n - k + 1.0, k + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn lgamma1p_matches_lgamma_away_from_zero() {
        for &a in &[-0.19, -0.05, 0.0, 0.05, 0.19] {
            let direct = lgamma(1.0 + a);
            assert!(
                (lgamma1p(a) - direct).abs() < 1e-16 + 1e-15 * direct.abs(),
                "{a}"
            );
        }
        // ln Γ(1 + a) ~ -γ a for tiny a.
        assert!(
            rel(
                lgamma1p(1e-10),
                -EULER_GAMMA * 1e-10 + 0.822_467_033_424_113_2e-20
            ) < 1e-15
        );
    }

    #[test]
    fn digamma_known_values() {
        // ψ(1) = -γ, ψ(1/2) = -γ - 2 ln 2
        let euler = EULER_GAMMA;
        assert!(rel(digamma(1.0), -euler) < 1e-15);
        assert!(rel(digamma(0.5), -euler - 2.0 * core::f64::consts::LN_2) < 1e-15);
        assert!(digamma(0.0).is_nan());
        assert!(digamma(-2.0).is_nan());
    }

    #[test]
    fn trigamma_known_values() {
        // ψ'(1) = pi^2 / 6, ψ'(1/2) = pi^2 / 2
        assert!(rel(trigamma(1.0), PI * PI / 6.0) < 1e-15);
        assert!(rel(trigamma(0.5), PI * PI / 2.0) < 1e-15);
    }

    #[test]
    fn polygamma_matches_recurrence() {
        // ψ^(2)(1) = -2 ζ(3)
        let zeta3 = 1.202_056_903_159_594_3;
        assert!(rel(polygamma(2, 1.0), -2.0 * zeta3) < 1e-14);
        // ψ^(n)(x + 1) = ψ^(n)(x) + (-1)^n n! / x^(n+1)
        for n in 2..6u32 {
            let x = 2.7;
            let lhs = polygamma(n, x + 1.0);
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            let rhs = polygamma(n, x) + sign * tgamma(f64::from(n) + 1.0) / x.powi(n as i32 + 1);
            assert!(rel(lhs, rhs) < 1e-13, "n = {n}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn sinpi_is_exact_at_integers() {
        assert_eq!(sinpi(3.0), 0.0);
        assert_eq!(sinpi(-7.0), 0.0);
        assert_eq!(sinpi(0.5), 1.0);
        assert_eq!(sinpi(-0.5), -1.0);
        assert_eq!(cospi(1.0), -1.0);
    }

    #[test]
    fn lbeta_branches_agree() {
        // Compare the Stirling branches with the direct formula where both are accurate.
        for &(a, b) in &[(0.5, 12.0), (3.0, 40.0), (12.0, 15.0), (11.0, 11.0)] {
            let direct = lgamma(a) + lgamma(b) - lgamma(a + b);
            assert!(rel(lbeta(a, b), direct) < 1e-13, "{a} {b}");
        }
        assert_eq!(lbeta(0.0, 2.0), f64::INFINITY);
        assert!(lbeta(-1.0, 2.0).is_nan());
    }
}
