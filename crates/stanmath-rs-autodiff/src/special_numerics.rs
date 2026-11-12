//! Differentiable Bessel functions, Owen's T, the Lambert W branches, and generalized
//! hypergeometric functions, with Stan Math's names.
//!
//! Values of the integer order Bessel functions, Owen's T, and Lambert W come from
//! `stanmath_rs_core::special`. The logarithm of the modified Bessel function of real order and the
//! hypergeometric functions are written once, generically over [`Real`]; derivatives with respect
//! to their parameters are obtained by running that algorithm on a dual number seeded in the
//! parameter, as for the incomplete gamma and beta functions. Derivative rules:
//!
//! * `J_n' = (J_{n-1} - J_{n+1}) / 2`, `Y_n' = (Y_{n-1} - Y_{n+1}) / 2`,
//!   `I_n' = (I_{n-1} + I_{n+1}) / 2`, `K_n' = -(K_{n-1} + K_{n+1}) / 2` (DLMF 10.6.1, 10.29.1);
//! * `d/dz ln I_v(z) = I_{v+1}(z) / I_v(z) + v / z` (DLMF 10.29.2);
//! * `dT/dh = -φ(h) erf(a h / sqrt 2) / 2` and `dT/da = exp(-h^2 (1 + a^2) / 2) / (2 pi (1 + a^2))`,
//!   by differentiating the defining integral;
//! * `W' = exp(-W) / (1 + W)` (DLMF 4.13.4);
//! * `d/dz pFq(a; b; z) = (prod a / prod b) pFq(a + 1; b + 1; z)` (DLMF 16.3.1).

use core::f64::consts::{LN_10, PI, SQRT_2};

use stanmath_rs_core::special as sp;

use crate::dual::Dual;
use crate::real::{BinaryFn, NaryFn, Real, UnaryFn};
use crate::special::{erf, lgamma};

/// Iteration cap for the series.
const MAX_ITER: usize = 1_000_000;
/// Relative size of the last term at which a series stops.
const TOL: f64 = 1e-17;
/// Extra iterations beyond the oracle's count, as a margin for third and higher derivatives.
const MARGIN: usize = 8;

/// How a series decides to stop.
#[derive(Clone, Copy, Debug)]
enum Stop {
    /// Stop when the last term is below tolerance in every forward-mode component.
    Converge,
    /// Run exactly this many terms.
    After(usize),
}

impl Stop {
    #[inline]
    fn done(self, iteration: usize, converged: impl FnOnce() -> bool) -> bool {
        match self {
            Stop::Converge => converged(),
            Stop::After(n) => iteration >= n,
        }
    }
}

/// A hyper-dual `f64` seeded along the all-ones direction, used to count the terms that second
/// derivatives need when the stopping test cannot see reverse-mode tangents.
fn seed(v: f64) -> Dual<Dual<f64>> {
    Dual::new(Dual::new(v, 1.0), Dual::new(1.0, 0.0))
}

macro_rules! bessel_integer_order {
    ($(#[$doc:meta])* $name:ident, $value:path, |$z:ident, $n:ident| $deriv:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name(pub i32);

        impl UnaryFn for $name {
            #[inline]
            fn value(&self, z: f64) -> f64 {
                $value(self.0, z)
            }

            #[inline]
            fn derivative<S: Real>(&self, $z: S, _y: S) -> S {
                let $n = self.0;
                $deriv
            }
        }
    };
}

bessel_integer_order!(
    /// `J_n(z)`; derivative `(J_{n-1}(z) - J_{n+1}(z)) / 2`.
    BesselJ,
    sp::bessel_j,
    |z, n| (z.map_unary(&BesselJ(n - 1)) - z.map_unary(&BesselJ(n + 1))) * 0.5
);
bessel_integer_order!(
    /// `Y_n(z)`; derivative `(Y_{n-1}(z) - Y_{n+1}(z)) / 2`.
    BesselY,
    sp::bessel_y,
    |z, n| (z.map_unary(&BesselY(n - 1)) - z.map_unary(&BesselY(n + 1))) * 0.5
);
bessel_integer_order!(
    /// `I_n(z)`; derivative `(I_{n-1}(z) + I_{n+1}(z)) / 2`.
    BesselI,
    sp::bessel_i,
    |z, n| (z.map_unary(&BesselI(n - 1)) + z.map_unary(&BesselI(n + 1))) * 0.5
);
bessel_integer_order!(
    /// `K_n(z)`; derivative `-(K_{n-1}(z) + K_{n+1}(z)) / 2`.
    BesselK,
    sp::bessel_k,
    |z, n| -((z.map_unary(&BesselK(n - 1)) + z.map_unary(&BesselK(n + 1))) * 0.5)
);

/// Bessel function of the first kind `J_v(z)` of integer order `v`.
#[inline]
pub fn bessel_first_kind<T: Real>(v: i32, z: T) -> T {
    z.map_unary(&BesselJ(v))
}

/// Bessel function of the second kind `Y_v(z)` of integer order `v`; requires `z > 0`.
#[inline]
pub fn bessel_second_kind<T: Real>(v: i32, z: T) -> T {
    z.map_unary(&BesselY(v))
}

/// Modified Bessel function of the first kind `I_v(z)` of integer order `v`.
#[inline]
pub fn modified_bessel_first_kind<T: Real>(v: i32, z: T) -> T {
    z.map_unary(&BesselI(v))
}

/// Modified Bessel function of the second kind `K_v(z)` of integer order `v`; requires `z > 0`.
#[inline]
pub fn modified_bessel_second_kind<T: Real>(v: i32, z: T) -> T {
    z.map_unary(&BesselK(v))
}

/// `ln I_v(z)` for real `v >= 0` and `z >= 0`, generically.
///
/// For `z > 30 + v^2` it uses the asymptotic expansion
/// `I_v(z) ~ exp(z) / sqrt(2 pi z) sum_k (-1)^k a_k(v) / z^k` (DLMF 10.40.1); otherwise the power
/// series `I_v(z) = (z/2)^v / Γ(v + 1) sum_k (z^2/4)^k Γ(v + 1) / (k! Γ(v + k + 1))`
/// (DLMF 10.25.2), whose terms are positive, rescaled to avoid overflow.
fn log_bessel_i_generic<S: Real>(v: S, z: S, stop: Stop) -> (S, usize) {
    let (vv, zv) = (v.value(), z.value());
    if vv.is_nan() || zv.is_nan() || vv < 0.0 || zv < 0.0 {
        return (S::from_f64(f64::NAN), 0);
    }
    if zv == 0.0 {
        return (
            S::from_f64(if vv == 0.0 { 0.0 } else { f64::NEG_INFINITY }),
            0,
        );
    }
    if zv == f64::INFINITY {
        return (S::from_f64(f64::INFINITY), 0);
    }
    if zv > 30.0 + vv * vv {
        let mu = v.square() * 4.0;
        let mut term = S::one();
        let mut sum = S::one();
        let mut prev = f64::INFINITY;
        let mut k = 0;
        while k < 500 {
            k += 1;
            let odd = (2 * k - 1) as f64;
            let next = -(term * (mu - odd * odd)) / (z * (8.0 * k as f64));
            // Stop before the asymptotic series starts to diverge.
            if next.value().abs() > prev {
                break;
            }
            prev = next.value().abs();
            term = next;
            sum += term;
            if stop.done(k, || term.magnitude() < TOL * sum.magnitude()) {
                break;
            }
        }
        return (z - (z * (2.0 * PI)).ln() * 0.5 + sum.ln(), k);
    }
    let y = z.square() * 0.25;
    let mut term = S::one();
    let mut sum = S::one();
    let mut offset = 0.0;
    let mut k = 0;
    while k < MAX_ITER {
        k += 1;
        let kf = k as f64;
        term = term * y / ((v + kf) * kf);
        sum += term;
        if sum.value() > 1e280 {
            sum *= 1e-280;
            term *= 1e-280;
            offset += 280.0 * LN_10;
        }
        if stop.done(k, || term.magnitude() < TOL * sum.magnitude()) {
            break;
        }
    }
    (v * (z * 0.5).ln() - lgamma(v + 1.0) + sum.ln() + offset, k)
}

/// `ln I_v(z)` with derivatives in both arguments. The order derivative differentiates the series;
/// the argument derivative is `I_{v+1}(z) / I_v(z) + v / z`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogBesselI;

impl BinaryFn for LogBesselI {
    fn value(&self, v: f64, z: f64) -> f64 {
        log_bessel_i_generic(v, z, Stop::Converge).0
    }

    fn partial_a<S: Real>(&self, v: S, z: S, _y: S) -> S {
        let stop = if S::ORDER == 0 {
            Stop::Converge
        } else {
            let (_, n) = log_bessel_i_generic(seed(v.value()), seed(z.value()), Stop::Converge);
            Stop::After(n + MARGIN)
        };
        log_bessel_i_generic(Dual::variable(v), Dual::constant(z), stop)
            .0
            .eps
    }

    fn partial_b<S: Real>(&self, v: S, z: S, y: S) -> S {
        ((v + 1.0).map_binary(z, self) - y).exp() + v / z
    }
}

/// `ln I_v(z)`, the logarithm of the modified Bessel function of the first kind, for real order
/// `v >= 0` and `z >= 0`, with derivatives in both arguments.
#[inline]
pub fn log_modified_bessel_first_kind<T: Real>(v: T, z: T) -> T {
    v.map_binary(z, &LogBesselI)
}

/// Owen's T function `T(h, a)` with partials in both arguments.
#[derive(Clone, Copy, Debug, Default)]
pub struct OwensT;

impl BinaryFn for OwensT {
    fn value(&self, h: f64, a: f64) -> f64 {
        sp::owens_t(h, a)
    }

    fn partial_a<S: Real>(&self, h: S, a: S, _y: S) -> S {
        -((h.square() * -0.5).exp() * sp::INV_SQRT_2PI * erf(a * h / SQRT_2) * 0.5)
    }

    fn partial_b<S: Real>(&self, h: S, a: S, _y: S) -> S {
        let one_a2 = a.square() + 1.0;
        (h.square() * one_a2 * -0.5).exp() / (one_a2 * (2.0 * PI))
    }
}

/// Owen's T function `T(h, a) = (1 / 2 pi) ∫_0^a exp(-h^2 (1 + x^2) / 2) / (1 + x^2) dx`.
#[inline]
pub fn owens_t<T: Real>(h: T, a: T) -> T {
    h.map_binary(a, &OwensT)
}

/// Principal branch of the Lambert W function; derivative `exp(-W) / (1 + W)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LambertW0;

impl UnaryFn for LambertW0 {
    #[inline]
    fn value(&self, x: f64) -> f64 {
        sp::lambert_w0(x)
    }

    #[inline]
    fn derivative<S: Real>(&self, _x: S, y: S) -> S {
        (-y).exp() / (y + 1.0)
    }
}

/// Lower branch of the Lambert W function; derivative `exp(-W) / (1 + W)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LambertWm1;

impl UnaryFn for LambertWm1 {
    #[inline]
    fn value(&self, x: f64) -> f64 {
        sp::lambert_wm1(x)
    }

    #[inline]
    fn derivative<S: Real>(&self, _x: S, y: S) -> S {
        (-y).exp() / (y + 1.0)
    }
}

/// Principal branch `W_0(x)` of the Lambert W function, for `x >= -1/e`.
#[inline]
pub fn lambert_w0<T: Real>(x: T) -> T {
    x.map_unary(&LambertW0)
}

/// Lower branch `W_{-1}(x)` of the Lambert W function, for `-1/e <= x < 0`.
#[inline]
pub fn lambert_wm1<T: Real>(x: T) -> T {
    x.map_unary(&LambertWm1)
}

fn is_nonpositive_integer(x: f64) -> bool {
    x <= 0.0 && x == x.floor()
}

/// The sign of `Γ(x)` for `x` not a pole.
fn gamma_sign(x: f64) -> f64 {
    if x > 0.0 || (-x).ceil() % 2.0 == 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// The series `sum_k prod (a)_k / prod (b)_k z^k / k!` (DLMF 16.2.1).
fn pfq_series<S: Real>(a: &[S], b: &[S], z: S, stop: Stop) -> (S, usize) {
    let mut term = S::one();
    let mut sum = S::one();
    let mut k = 0;
    while k < MAX_ITER {
        let kf = k as f64;
        let mut ratio = z / (kf + 1.0);
        for &ai in a {
            ratio *= ai + kf;
        }
        for &bi in b {
            ratio /= bi + kf;
        }
        term *= ratio;
        sum += term;
        k += 1;
        if stop.done(k, || term.magnitude() <= TOL * sum.magnitude()) {
            return (sum, k);
        }
    }
    (S::from_f64(f64::NAN), k)
}

/// `pFq(a; b; z)` where the series converges: always when `p <= q`, for `|z| < 1` when
/// `p = q + 1` (and at `|z| = 1` when `sum b - sum a > 0`), and for any `z` when some `a` is a
/// nonpositive integer (a polynomial). `NaN` otherwise.
fn pfq_generic<S: Real>(a: &[S], b: &[S], z: S, stop: Stop) -> (S, usize) {
    let nan = (S::from_f64(f64::NAN), 0);
    if a.iter().chain(b).any(Real::is_nan) || z.is_nan() {
        return nan;
    }
    if b.iter().any(|bi| is_nonpositive_integer(bi.value())) {
        return nan;
    }
    let polynomial = a.iter().any(|ai| is_nonpositive_integer(ai.value()));
    if !polynomial {
        let (p, q) = (a.len(), b.len());
        let size = z.value().abs();
        if p > q + 1 || (p == q + 1 && size > 1.0) {
            return nan;
        }
        if p == q + 1 && size == 1.0 {
            let excess =
                b.iter().map(Real::value).sum::<f64>() - a.iter().map(Real::value).sum::<f64>();
            if !(excess > 0.0) {
                return nan;
            }
        }
    }
    pfq_series(a, b, z, stop)
}

/// `2F1(a, b; c; z)` for `z <= 1`, generically.
///
/// Polynomial cases sum directly for any `z`. For `z < 0` Pfaff's transformation
/// `F(a, b; c; z) = (1 - z)^(-a) F(a, c - b; c; z / (z - 1))` (DLMF 15.8.1) maps the argument
/// into `(0, 1)`; `0 <= z < 1` uses the series; `z = 1` uses Gauss's sum
/// `Γ(c) Γ(c - a - b) / (Γ(c - a) Γ(c - b))` when `c - a - b > 0` (DLMF 15.4.20).
fn hyp2f1_generic<S: Real>(a: S, b: S, c: S, z: S, stop: Stop) -> (S, usize) {
    let (av, bv, cv, zv) = (a.value(), b.value(), c.value(), z.value());
    let nan = (S::from_f64(f64::NAN), 0);
    if av.is_nan() || bv.is_nan() || cv.is_nan() || zv.is_nan() || is_nonpositive_integer(cv) {
        return nan;
    }
    if is_nonpositive_integer(av) || is_nonpositive_integer(bv) {
        return pfq_series(&[a, b], &[c], z, stop);
    }
    if zv < 0.0 {
        let w = z / (z - 1.0);
        let (s, n) = pfq_series(&[a, c - b], &[c], w, stop);
        return ((-(a * (-z).ln_1p())).exp() * s, n);
    }
    if zv < 1.0 {
        return pfq_series(&[a, b], &[c], z, stop);
    }
    if zv == 1.0 && cv - av - bv > 0.0 {
        let sign =
            gamma_sign(cv) * gamma_sign(cv - av - bv) * gamma_sign(cv - av) * gamma_sign(cv - bv);
        let log = lgamma(c) + lgamma(c - a - b) - lgamma(c - a) - lgamma(c - b);
        return (log.exp() * sign, 0);
    }
    nan
}

/// Gauss's hypergeometric function `2F1(a, b; c; z)` with arguments `[a, b, c, z]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Hypergeometric2F1;

impl NaryFn for Hypergeometric2F1 {
    fn value(&self, x: &[f64]) -> f64 {
        hyp2f1_generic(x[0], x[1], x[2], x[3], Stop::Converge).0
    }

    #[allow(clippy::needless_range_loop)] // the index selects which argument to seed
    fn partials<S: Real>(&self, x: &[S], _y: S, needed: &[bool], out: &mut [S]) {
        if needed[..3].iter().any(|&n| n) {
            let stop = if S::ORDER == 0 {
                Stop::Converge
            } else {
                let s: Vec<Dual<Dual<f64>>> = x.iter().map(|v| seed(v.value())).collect();
                Stop::After(hyp2f1_generic(s[0], s[1], s[2], s[3], Stop::Converge).1 + MARGIN)
            };
            for i in 0..3 {
                if needed[i] {
                    let d = |j: usize| {
                        if j == i {
                            Dual::variable(x[j])
                        } else {
                            Dual::constant(x[j])
                        }
                    };
                    out[i] = hyp2f1_generic(d(0), d(1), d(2), d(3), stop).0.eps;
                }
            }
        }
        if needed[3] {
            let (a, b, c, z) = (x[0], x[1], x[2], x[3]);
            out[3] = a * b / c * S::map_nary(&[a + 1.0, b + 1.0, c + 1.0, z], self);
        }
    }
}

/// The generalized hypergeometric function `pFq` with arguments `[a_1, ..., a_p, b_1, ..., b_q, z]`.
#[derive(Clone, Copy, Debug)]
pub struct HypergeometricPFQ {
    /// Number of numerator parameters.
    pub p: usize,
    /// Number of denominator parameters.
    pub q: usize,
}

impl NaryFn for HypergeometricPFQ {
    fn value(&self, x: &[f64]) -> f64 {
        let (p, n) = (self.p, self.p + self.q);
        pfq_generic(&x[..p], &x[p..n], x[n], Stop::Converge).0
    }

    #[allow(clippy::needless_range_loop)] // the index selects which argument to seed
    fn partials<S: Real>(&self, x: &[S], _y: S, needed: &[bool], out: &mut [S]) {
        let (p, n) = (self.p, self.p + self.q);
        if needed[..n].iter().any(|&v| v) {
            let stop = if S::ORDER == 0 {
                Stop::Converge
            } else {
                let s: Vec<Dual<Dual<f64>>> = x.iter().map(|v| seed(v.value())).collect();
                Stop::After(pfq_generic(&s[..p], &s[p..n], s[n], Stop::Converge).1 + MARGIN)
            };
            for i in 0..n {
                if needed[i] {
                    let seeded: Vec<Dual<S>> = x
                        .iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            if j == i {
                                Dual::variable(v)
                            } else {
                                Dual::constant(v)
                            }
                        })
                        .collect();
                    out[i] = pfq_generic(&seeded[..p], &seeded[p..n], seeded[n], stop)
                        .0
                        .eps;
                }
            }
        }
        if needed[n] {
            let mut factor = S::one();
            for &a in &x[..p] {
                factor *= a;
            }
            for &b in &x[p..n] {
                factor /= b;
            }
            let shifted: Vec<S> = x[..n].iter().map(|&v| v + 1.0).chain([x[n]]).collect();
            out[n] = factor * S::map_nary(&shifted, self);
        }
    }
}

/// Gauss's hypergeometric function `2F1(a, b; c; z)` for `z <= 1` (any `z` when `a` or `b` is a
/// nonpositive integer), with derivatives in every argument.
#[allow(non_snake_case)]
#[inline]
pub fn hypergeometric_2F1<T: Real>(a: T, b: T, c: T, z: T) -> T {
    T::map_nary(&[a, b, c, z], &Hypergeometric2F1)
}

/// The generalized hypergeometric function `3F2(a; b; z)`, with derivatives in every argument.
#[allow(non_snake_case)]
#[inline]
pub fn hypergeometric_3F2<T: Real>(a: [T; 3], b: [T; 2], z: T) -> T {
    hypergeometric_pFq(&a, &b, z)
}

/// The generalized hypergeometric function `pFq(a; b; z)` by its series, with derivatives in
/// every argument. `NaN` where the series diverges.
#[allow(non_snake_case)]
pub fn hypergeometric_pFq<T: Real>(a: &[T], b: &[T], z: T) -> T {
    let args: Vec<T> = a.iter().chain(b).copied().chain([z]).collect();
    T::map_nary(
        &args,
        &HypergeometricPFQ {
            p: a.len(),
            q: b.len(),
        },
    )
}
