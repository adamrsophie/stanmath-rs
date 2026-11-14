//! Owen's T function.
//!
//! `T(h, a) = (1 / 2 pi) ∫_0^a exp(-h^2 (1 + x^2) / 2) / (1 + x^2) dx` (Owen 1956,
//! "Tables for computing bivariate normal probabilities"). `T` is even in `h` and odd in `a`. For
//! `0 < a <= 1` the integral is computed with composite 20-point Gauss-Legendre quadrature, with
//! panels no wider than the Gaussian factor `exp(-h^2 x^2 / 2)`. For `a > 1` it uses Owen's identity
//! `T(h, a) = (Φ(h) Q(a h) + Φ(a h) Q(h)) / 2 - T(a h, 1 / a)` for `h >= 0`, where `Q = 1 - Φ`,
//! written so that neither term cancels.

use core::f64::consts::PI;

use super::Phi;

/// Owen's T function `T(h, a)`.
pub fn owens_t(h: f64, a: f64) -> f64 {
    if h.is_nan() || a.is_nan() {
        return f64::NAN;
    }
    let t = owens_t_nonnegative(h.abs(), a.abs());
    if a < 0.0 { -t } else { t }
}

fn owens_t_nonnegative(h: f64, a: f64) -> f64 {
    if a == 0.0 || h.is_infinite() {
        return 0.0;
    }
    if a.is_infinite() {
        // T(h, ∞) = Q(|h|) / 2.
        return 0.5 * Phi(-h);
    }
    if a <= 1.0 {
        return owens_t_integral(h, a);
    }
    let ah = a * h;
    0.5 * (Phi(h) * Phi(-ah) + Phi(ah) * Phi(-h)) - owens_t_integral(ah, a.recip())
}

/// The defining integral for `h >= 0` and `0 < a <= 1`.
fn owens_t_integral(h: f64, a: f64) -> f64 {
    if h.is_infinite() {
        return 0.0;
    }
    let half_h2 = 0.5 * h * h;
    if half_h2 > 750.0 {
        return 0.0;
    }
    // Beyond x = 40 / h the factor exp(-h^2 x^2 / 2) is below exp(-800).
    let upper = if h > 0.0 { a.min(40.0 / h) } else { a };
    let width = if h > 1.0 { h.recip() } else { 1.0 };
    let panels = (upper / width).ceil().max(1.0) as usize;
    let step = upper / panels as f64;
    let (nodes, weights) = gauss_legendre(20);
    let half = 0.5 * step;
    let mut sum = 0.0;
    for p in 0..panels {
        let mid = (p as f64 + 0.5) * step;
        for (x, w) in nodes.iter().zip(&weights) {
            let u = mid + half * x;
            sum += w * (-half_h2 * u * u).exp() / (1.0 + u * u);
        }
    }
    (-half_h2).exp() * sum * half / (2.0 * PI)
}

/// `(P_n(z), P_n'(z))` by the three-term recurrence of the Legendre polynomials (DLMF 18.9.1).
fn legendre(n: usize, z: f64) -> (f64, f64) {
    let (mut p1, mut p2) = (1.0, 0.0);
    for j in 1..=n {
        let p3 = p2;
        p2 = p1;
        let jf = j as f64;
        p1 = ((2.0 * jf - 1.0) * z * p2 - (jf - 1.0) * p3) / jf;
    }
    // (z^2 - 1) P_n' = n (z P_n - P_{n-1}) (DLMF 18.9.18).
    let nf = n as f64;
    (p1, nf * (z * p1 - p2) / (z * z - 1.0))
}

#[allow(clippy::needless_range_loop)] // nodes are placed symmetrically by index
/// Nodes and weights of `n`-point Gauss-Legendre quadrature on `[-1, 1]` (DLMF 3.5.19), found by
/// Newton's method from the asymptotic estimates of the zeros of `P_n`.
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre(n, z);
            let dz = p / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        let (_, dp) = legendre(n, z);
        x[i] = -z;
        x[n - 1 - i] = z;
        let weight = 2.0 / ((1.0 - z * z) * dp * dp);
        w[i] = weight;
        w[n - 1 - i] = weight;
    }
    (x, w)
}
