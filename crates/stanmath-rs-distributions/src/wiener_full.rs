//! Wiener diffusion model with inter-trial variability: first passage time log densities and
//! unnormalized log CDFs.
//!
//! A diffusion starts at `w a` between absorbing boundaries at `0` and `a` and drifts at rate `v`;
//! the first passage time is shifted by a nondecision time `t0`. As in Stan, every function refers
//! to absorption at the upper boundary. The full model adds inter-trial variability: the drift is
//! normal with mean `v` and standard deviation `sv`, the relative starting point is uniform on
//! `[w - sw / 2, w + sw / 2]`, and the nondecision time is uniform on `[t0, t0 + st0]`.
//!
//! Sources: Navarro and Fuss (2009) for the density series; Blurton, Kesselmeier and Gondan (2012)
//! for the CDF series; Hartmann and Klauer (2021) for the drift variability integrated in closed
//! form. The integrals over `sw` and `st0`, and over `sv` in the CDFs, where no closed form
//! exists, use adaptive Gauss-Kronrod quadrature on the unit interval.
//!
//! # Derivatives
//!
//! Series lengths and quadrature subdivisions are chosen from primal `f64` values only; the chosen
//! approximation is then evaluated in the caller's scalar type. Derivatives are therefore the
//! exact derivatives of that approximation, in every autodiff mode. Partials are not fused: in
//! reverse mode the whole computation is recorded on the tape.
//!
//! # Rust names
//!
//! Stan overloads `wiener_lpdf` for four, five, and seven parameters, and the unnormalized CDFs for
//! four and seven. Rust has no overloading, so the five and seven parameter densities use Stan
//! Math's header names [`wiener5_lpdf`] and [`wiener_full_lpdf`], and the CDFs use
//! [`wiener4_lcdf_unnorm`] and [`wiener_full_lcdf_unnorm`] (with their complements).
//! [`wiener_lcdf_unnorm`] and [`wiener_lccdf_unnorm`] are the four parameter versions, matching
//! the four parameter `wiener_lpdf`.

// The return types name one generic parameter per Wiener argument; aliases would only hide them.
#![allow(clippy::type_complexity)]

use core::f64::consts::PI;

use stanmath_rs_autodiff::functions::{log_Phi, log_diff_exp, log_sum_exp, log1m, log1m_exp};
use stanmath_rs_autodiff::{Broadcast, Lift, Promote, Promoted, Promoted2, Real, broadcast_len};
use stanmath_rs_core::special::HALF_LN_2PI;
use stanmath_rs_core::{Error, Result};

use crate::check::{obs, param};
use crate::continuous_extra::{param_nonnegative_finite, param_open_unit};

/// Default `precision`, as in Stan.
const DEFAULT_PRECISION: f64 = 1e-4;
/// Loosest relative tolerance of the quadratures, whatever the `precision` argument.
const MAX_TOLERANCE: f64 = 1e-6;
/// Relative truncation error of every series.
const SERIES_EPS: f64 = 1e-12;
/// Term cap for every series; not reached for valid arguments.
const MAX_TERMS: i64 = 1000;
/// Interval splits allowed in one adaptive quadrature.
const MAX_SPLITS: usize = 100;
/// Half width, in standard deviations, of the range integrated over the drift distribution.
const DRIFT_RANGE: f64 = 8.5;

// Vectorization over six and eight arguments.

/// Argument tuples of the Wiener functions: [`Broadcast`] extended to six and eight arguments.
///
/// Implemented for every tuple that implements `Broadcast<5>`, and for tuples of six and eight
/// arguments whose elements promote to a common scalar type.
pub trait WienerArgs<const N: usize> {
    /// The common scalar type of all arguments.
    type Out: Real;
    /// Lengths of the arguments (`None` for scalars).
    fn lens(&self) -> [Option<usize>; N];
    /// Elements at index `i`, each converted to the common scalar type.
    fn get(&self, i: usize) -> [Self::Out; N];
}

/// The common scalar type of a Wiener argument tuple.
pub type WienerOut<X, const N: usize> = <X as WienerArgs<N>>::Out;

impl<X: Broadcast<5>> WienerArgs<5> for X {
    type Out = Promoted<X, 5>;

    fn lens(&self) -> [Option<usize>; 5] {
        <X as Broadcast<5>>::lens(self)
    }

    fn get(&self, i: usize) -> [Self::Out; 5] {
        <X as Broadcast<5>>::get(self, i)
    }
}

type Out3<A, B, C> = Promoted<(A, B, C), 3>;
type Out4<A, B, C, D> = Promoted<(A, B, C, D), 4>;

impl<A, B, C, D, E, F> WienerArgs<6> for (A, B, C, D, E, F)
where
    A: Copy,
    B: Copy,
    C: Copy,
    D: Copy,
    E: Copy,
    F: Copy,
    (A, B, C): Broadcast<3>,
    (D, E, F): Broadcast<3>,
    Out3<A, B, C>: Promote<Out3<D, E, F>>,
    Out3<A, B, C>: Lift<Promoted2<Out3<A, B, C>, Out3<D, E, F>>>,
    Out3<D, E, F>: Lift<Promoted2<Out3<A, B, C>, Out3<D, E, F>>>,
{
    type Out = Promoted2<Out3<A, B, C>, Out3<D, E, F>>;

    fn lens(&self) -> [Option<usize>; 6] {
        let l = <(A, B, C) as Broadcast<3>>::lens(&(self.0, self.1, self.2));
        let r = <(D, E, F) as Broadcast<3>>::lens(&(self.3, self.4, self.5));
        [l[0], l[1], l[2], r[0], r[1], r[2]]
    }

    fn get(&self, i: usize) -> [Self::Out; 6] {
        let [a, b, c] = <(A, B, C) as Broadcast<3>>::get(&(self.0, self.1, self.2), i)
            .map(<Out3<A, B, C> as Lift<Self::Out>>::lift);
        let [d, e, f] = <(D, E, F) as Broadcast<3>>::get(&(self.3, self.4, self.5), i)
            .map(<Out3<D, E, F> as Lift<Self::Out>>::lift);
        [a, b, c, d, e, f]
    }
}

impl<A, B, C, D, E, F, G, H> WienerArgs<8> for (A, B, C, D, E, F, G, H)
where
    A: Copy,
    B: Copy,
    C: Copy,
    D: Copy,
    E: Copy,
    F: Copy,
    G: Copy,
    H: Copy,
    (A, B, C, D): Broadcast<4>,
    (E, F, G, H): Broadcast<4>,
    Out4<A, B, C, D>: Promote<Out4<E, F, G, H>>,
    Out4<A, B, C, D>: Lift<Promoted2<Out4<A, B, C, D>, Out4<E, F, G, H>>>,
    Out4<E, F, G, H>: Lift<Promoted2<Out4<A, B, C, D>, Out4<E, F, G, H>>>,
{
    type Out = Promoted2<Out4<A, B, C, D>, Out4<E, F, G, H>>;

    fn lens(&self) -> [Option<usize>; 8] {
        let l = <(A, B, C, D) as Broadcast<4>>::lens(&(self.0, self.1, self.2, self.3));
        let r = <(E, F, G, H) as Broadcast<4>>::lens(&(self.4, self.5, self.6, self.7));
        [l[0], l[1], l[2], l[3], r[0], r[1], r[2], r[3]]
    }

    fn get(&self, i: usize) -> [Self::Out; 8] {
        let [a, b, c, d] =
            <(A, B, C, D) as Broadcast<4>>::get(&(self.0, self.1, self.2, self.3), i)
                .map(<Out4<A, B, C, D> as Lift<Self::Out>>::lift);
        let [e, f, g, h] =
            <(E, F, G, H) as Broadcast<4>>::get(&(self.4, self.5, self.6, self.7), i)
                .map(<Out4<E, F, G, H> as Lift<Self::Out>>::lift);
        [a, b, c, d, e, f, g, h]
    }
}

/// Sums `term` over broadcast elements after validating each with `check`.
fn sum_elements<X: WienerArgs<N>, const N: usize>(
    function: &'static str,
    args: X,
    check: impl Fn([f64; N]) -> Result<()>,
    term: impl Fn([X::Out; N]) -> X::Out,
) -> Result<X::Out> {
    let n = broadcast_len(function, &args.lens())?;
    let mut total = <X::Out as Real>::zero();
    for i in 0..n {
        let x = args.get(i);
        check(x.map(|e| e.value()))?;
        total += term(x);
    }
    Ok(total)
}

/// Validates one element of the full model; `density` selects the observation checks of the log
/// densities (`y` positive finite and greater than `t0`) over those of the CDFs (`y` not NaN).
fn check_element(
    f: &'static str,
    [y, a, t0, w, v, sv, sw, st0]: [f64; 8],
    density: bool,
) -> Result<()> {
    if density {
        obs::positive_finite(f, "y", y)?;
    } else {
        obs::not_nan(f, "y", y)?;
    }
    param::positive_finite(f, "a", a)?;
    param_nonnegative_finite(f, "t0", t0)?;
    param_open_unit(f, "w", w)?;
    param::finite(f, "v", v)?;
    param_nonnegative_finite(f, "sv", sv)?;
    if !(0.0..1.0).contains(&sw) {
        return Err(Error::parameter(f, "sw", sw, "in [0, 1)"));
    }
    param_nonnegative_finite(f, "st0", st0)?;
    if w - 0.5 * sw <= 0.0 || w + 0.5 * sw >= 1.0 {
        return Err(Error::parameter(
            f,
            "sw",
            sw,
            "small enough that w +/- sw / 2 lies in (0, 1)",
        ));
    }
    if density && y <= t0 {
        return Err(Error::observation(
            f,
            "y",
            y,
            "greater than the nondecision time t0",
        ));
    }
    Ok(())
}

/// The quadrature tolerance for a `precision` argument.
fn tolerance(f: &'static str, precision: f64) -> Result<f64> {
    param::positive_finite(f, "precision", precision)?;
    Ok(precision.min(MAX_TOLERANCE))
}

#[inline]
fn neg_inf<T: Real>() -> T {
    T::from_f64(f64::NEG_INFINITY)
}

#[inline]
fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

// Series for the four parameter model.

/// True when the small time series needs fewer terms than the large time series, by the term
/// counts of Navarro and Fuss (2009) at absolute error `SERIES_EPS`.
fn prefer_small_time(u: f64) -> bool {
    let eps = SERIES_EPS;
    let large = {
        let floor = 1.0 / (PI * u.sqrt());
        if PI * u * eps < 1.0 {
            (-2.0 * (PI * u * eps).ln() / (PI * PI * u))
                .sqrt()
                .max(floor)
        } else {
            floor
        }
    };
    let small = {
        let arg = 2.0 * (2.0 * PI * u).sqrt() * eps;
        if arg < 1.0 {
            (2.0 + (-2.0 * u * arg.ln()).sqrt()).max(u.sqrt() + 1.0)
        } else {
            2.0
        }
    };
    small < large
}

/// `ln f(u | w)`: the log first passage density at the lower boundary of a driftless Wiener
/// process with unit boundary separation, started at `w`, at standardized time `u > 0`.
fn log_std_density<T: Real>(u: T, w: T) -> T {
    let uv = u.value();
    if prefer_small_time(uv) {
        // (2 pi u^3)^(-1/2) sum_k (w + 2k) exp(-(w + 2k)^2 / (2u)), with exp(-w^2 / (2u)) factored
        // out so that every remaining exponent, -2k(k + w) / u, is nonpositive.
        let mut sum = w;
        let j_min = uv.sqrt().ceil() as i64;
        for j in 1..=MAX_TERMS {
            let jf = j as f64;
            let up = (w + jf * 2.0) * (-((w + jf) * (jf * 2.0)) / u).exp();
            let down = (w - jf * 2.0) * (-((-w + jf) * (jf * 2.0)) / u).exp();
            sum += up + down;
            if j >= j_min && up.value().abs() + down.value().abs() <= SERIES_EPS * sum.value().abs()
            {
                break;
            }
        }
        sum.ln() - w.square() / (u * 2.0) - u.ln() * 1.5 - HALF_LN_2PI
    } else {
        // pi sum_{k >= 1} k exp(-k^2 pi^2 u / 2) sin(k pi w), with exp(-pi^2 u / 2) factored out.
        let mut sum = T::zero();
        let k_min = (1.0 / (PI * uv.sqrt())).ceil() as i64;
        for k in 1..=MAX_TERMS {
            let kf = k as f64;
            let decay = (-(u * ((kf * kf - 1.0) * PI * PI * 0.5))).exp();
            sum += (w * (kf * PI)).sin() * decay * kf;
            if k >= k_min && kf * decay.value() <= SERIES_EPS * sum.value().abs() {
                break;
            }
        }
        sum.ln() - u * (PI * PI * 0.5) + PI.ln()
    }
}

/// Log density of first passage at time `t` at the lower boundary, with boundary separation `a`,
/// drift `v`, relative start `w`, and a normal drift distribution with standard deviation `sv`
/// integrated out in closed form.
///
/// Averaging `exp(-v a w - v^2 t / 2)` over the drift gives
/// `exp((sv^2 a^2 w^2 - 2 a v w - v^2 t) / (2 (1 + sv^2 t))) / sqrt(1 + sv^2 t)`.
fn log_density_lower<T: Real>(t: T, a: T, v: T, w: T, sv: T) -> T {
    let one_plus = sv.square() * t + 1.0;
    let aw = a * w;
    let exponent = (sv.square() * aw.square() - v * aw * 2.0 - v.square() * t) / (one_plus * 2.0);
    exponent - one_plus.ln() * 0.5 - a.ln() * 2.0 + log_std_density(t / a.square(), w)
}

/// As [`log_density_lower`], at the upper boundary: the lower boundary with `w -> 1 - w` and
/// `v -> -v`.
fn log_density_upper<T: Real>(t: T, a: T, v: T, w: T, sv: T) -> T {
    log_density_lower(t, a, -v, -w + 1.0, sv)
}

/// Log probability of absorption at the lower boundary,
/// `P = (exp(-2 v a w) - exp(-2 v a)) / (1 - exp(-2 v a))`, or `1 - w` without drift.
fn log_prob_lower<T: Real>(a: T, v: T, w: T) -> T {
    let x = v * a * 2.0;
    let w1 = -w + 1.0;
    if x.value().abs() < 1e-3 {
        // With h(y) = ln(expm1(y) / y) = y / 2 + y^2 / 24 - y^4 / 2880 + O(y^6),
        // ln P = ln(1 - w) - x w + h(-x (1 - w)) - h(-x), accurate through fourth derivatives.
        let h = |y: T| y * 0.5 + y.square() / 24.0 - y.powi(4) / 2880.0;
        log1m(w) - x * w + h(-(x * w1)) - h(-x)
    } else if x.value() > 0.0 {
        -(x * w) + log1m_exp(-(x * w1)) - log1m_exp(-x)
    } else {
        log1m_exp(x * w1) - log1m_exp(x)
    }
}

/// Log probability of absorption at the upper boundary.
fn log_prob_upper<T: Real>(a: T, v: T, w: T) -> T {
    log_prob_lower(a, -v, -w + 1.0)
}

/// Small time series for `ln F(t)` at the lower boundary, by the method of images applied to the
/// first passage CDF of Brownian motion with drift (Blurton, Kesselmeier and Gondan 2012).
///
/// Image `k` sits at signed distance `c = a (w + 2k)` and contributes, with the sign `s` of `c`,
/// `s (exp(2 v a k) Phi(-s (c + v t) / sqrt(t)) + exp(-2 v a (w + k)) Phi(-s (c - v t) / sqrt(t)))`.
fn small_time_log_cdf<T: Real>(t: T, a: T, v: T, w: T) -> T {
    let sqrt_t = t.sqrt();
    let vt = v * t;
    let va = v * a;
    let image = |k: i64| -> T {
        let kf = k as f64;
        let c = a * (w + kf * 2.0);
        let s = if k >= 0 { -1.0 } else { 1.0 };
        let first = va * (kf * 2.0) + log_Phi((c + vt) * s / sqrt_t);
        let second = -(va * (w + kf) * 2.0) + log_Phi((c - vt) * s / sqrt_t);
        log_sum_exp(first, second)
    };
    let mut pos = image(0);
    let mut neg = image(-1);
    for j in 1..=MAX_TERMS {
        let p = image(j);
        let q = image(-j - 1);
        pos = log_sum_exp(pos, p);
        neg = log_sum_exp(neg, q);
        let (pv, nv) = (pos.value(), neg.value());
        let log_f = pv + (-(nv - pv).exp()).ln_1p();
        if p.value().max(q.value()) <= log_f + SERIES_EPS.ln() {
            break;
        }
    }
    log_diff_exp(pos, neg)
}

/// Large time series for `ln(P - F(t))` at the lower boundary (Feller):
/// `P - F(t) = 2 pi / a^2 exp(-v a w - v^2 t / 2)
/// sum_k k sin(k pi w) exp(-k^2 pi^2 t / (2 a^2)) / (v^2 + k^2 pi^2 / a^2)`.
fn large_time_log_survival<T: Real>(t: T, a: T, v: T, w: T) -> T {
    let a2 = a.square();
    let u = t / a2;
    let pi2_a2 = a2.recip() * (PI * PI);
    let v2 = v.square();
    let mut sum = T::zero();
    let k_min = (1.0 / (PI * u.value().sqrt())).ceil() as i64;
    for k in 1..=MAX_TERMS {
        let kf = k as f64;
        let decay = (-(u * ((kf * kf - 1.0) * PI * PI * 0.5))).exp() / (v2 + pi2_a2 * (kf * kf));
        sum += (w * (kf * PI)).sin() * decay * kf;
        if k >= k_min && kf * decay.value() <= SERIES_EPS * sum.value().abs() {
            break;
        }
    }
    sum.ln() - a.ln() * 2.0 - v * a * w - v2 * t * 0.5 - u * (PI * PI * 0.5) + (2.0 * PI).ln()
}

/// `ln F(t)` for the lower boundary, where `F(t)` is the probability of absorption there by time
/// `t`; with `complement`, `ln(P - F(t))`, the probability of absorption there after `t`.
fn log_cdf_lower<T: Real>(t: T, a: T, v: T, w: T, complement: bool) -> T {
    let tv = t.value();
    if tv <= 0.0 {
        return if complement {
            log_prob_lower(a, v, w)
        } else {
            neg_inf()
        };
    }
    if tv == f64::INFINITY {
        return if complement {
            neg_inf()
        } else {
            log_prob_lower(a, v, w)
        };
    }
    if prefer_small_time((t / a.square()).value()) {
        let log_f = small_time_log_cdf(t, a, v, w);
        if complement {
            log_diff_exp(log_prob_lower(a, v, w), log_f)
        } else {
            log_f
        }
    } else {
        let log_s = large_time_log_survival(t, a, v, w);
        if complement {
            log_s
        } else {
            log_diff_exp(log_prob_lower(a, v, w), log_s)
        }
    }
}

/// As [`log_cdf_lower`], at the upper boundary.
fn log_cdf_upper<T: Real>(t: T, a: T, v: T, w: T, complement: bool) -> T {
    log_cdf_lower(t, a, -v, -w + 1.0, complement)
}

// Adaptive quadrature.

/// Gauss-Kronrod (7, 15) abscissas on `[-1, 1]`, the nonnegative half in decreasing order.
const XGK: [f64; 8] = [
    0.9914553711208126,
    0.9491079123427585,
    0.8648644233597691,
    0.7415311855993945,
    0.5860872354676911,
    0.4058451513773972,
    0.20778495500789848,
    0.0,
];
/// Kronrod weights for [`XGK`].
const WGK: [f64; 8] = [
    0.022935322010529224,
    0.06309209262997856,
    0.10479001032225019,
    0.14065325971552592,
    0.1690047266392679,
    0.19035057806478542,
    0.20443294007529889,
    0.20948214108472782,
];
/// Gauss weights for the abscissas `XGK[1]`, `XGK[3]`, `XGK[5]`, `XGK[7]`.
const WG: [f64; 4] = [
    0.1294849661688697,
    0.27970539148927664,
    0.3818300505051189,
    0.4179591836734694,
];

struct Piece<T> {
    lo: f64,
    hi: f64,
    value: T,
    error: f64,
}

fn gauss_kronrod<T: Real>(f: &mut dyn FnMut(f64) -> T, lo: f64, hi: f64) -> Piece<T> {
    let center = 0.5 * (lo + hi);
    let half = 0.5 * (hi - lo);
    let f_center = f(center);
    let mut kronrod = f_center * WGK[7];
    let mut gauss = f_center * WG[3];
    for (j, (&x, &weight)) in XGK.iter().zip(&WGK).take(7).enumerate() {
        let dx = half * x;
        let pair = f(center - dx) + f(center + dx);
        kronrod += pair * weight;
        if j % 2 == 1 {
            gauss += pair * WG[j / 2];
        }
    }
    Piece {
        lo,
        hi,
        value: kronrod * half,
        error: (kronrod.value() - gauss.value()).abs() * half,
    }
}

/// Adaptive Gauss-Kronrod integral of `f` over `[0, 1]` to relative tolerance `tol`.
///
/// Subdivision uses primal values only, and the result is the sum of the accepted 15 point rules
/// evaluated in `T`.
// ponytail: rejected pieces stay on a reverse-mode tape unused; replay the final partition in a
// second pass if tape size matters.
fn integrate_unit<T: Real>(f: &mut dyn FnMut(f64) -> T, tol: f64) -> T {
    let mut pieces = vec![gauss_kronrod(f, 0.0, 1.0)];
    for _ in 0..MAX_SPLITS {
        let total: f64 = pieces.iter().map(|p| p.value.value()).sum();
        let error: f64 = pieces.iter().map(|p| p.error).sum();
        if error <= tol * total.abs() {
            break;
        }
        let worst = pieces
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.error.total_cmp(&y.1.error))
            .map_or(0, |(i, _)| i);
        let p = pieces.swap_remove(worst);
        let mid = 0.5 * (p.lo + p.hi);
        pieces.push(gauss_kronrod(f, p.lo, mid));
        pieces.push(gauss_kronrod(f, mid, p.hi));
    }
    pieces.into_iter().fold(T::zero(), |acc, p| acc + p.value)
}

/// The mean of `f(w')` over `w'` uniform on `[w - sw / 2, w + sw / 2]`, or `f(w)` when `sw = 0`.
fn avg_w<T: Real>(w: T, sw: T, tol: f64, f: &mut dyn FnMut(T) -> T) -> T {
    if sw.value() == 0.0 {
        return f(w);
    }
    let lo = w - sw * 0.5;
    integrate_unit(&mut |s| f(lo + sw * s), tol)
}

/// The mean of `f(v')` over `v'` normal with mean `v` and standard deviation `sv`, or `f(v)` when
/// `sv = 0`. The range `v +/- DRIFT_RANGE sv` omits a probability below `1e-16`.
fn avg_v<T: Real>(v: T, sv: T, tol: f64, f: &mut dyn FnMut(T) -> T) -> T {
    if sv.value() == 0.0 {
        return f(v);
    }
    integrate_unit(
        &mut |s| {
            let z = DRIFT_RANGE * (2.0 * s - 1.0);
            let weight = 2.0 * DRIFT_RANGE * (-0.5 * z * z - HALF_LN_2PI).exp();
            f(v + sv * z) * weight
        },
        tol,
    )
}

/// The mean of `g(w', v')` over the starting point and drift distributions.
fn avg_wv<T: Real>(w: T, sw: T, v: T, sv: T, tol: f64, g: &mut dyn FnMut(T, T) -> T) -> T {
    avg_w(w, sw, tol, &mut |ww| avg_v(v, sv, tol, &mut |vv| g(ww, vv)))
}

/// `(1 / st0)` times the integral of `f(y - tau)` over `tau` in `[t0, min(t0 + st0, y)]`, or
/// `f(y - t0)` when `st0 = 0`. Requires `y > t0` when `st0 > 0`.
fn avg_t<T: Real>(y: T, t0: T, st0: T, tol: f64, f: &mut dyn FnMut(T) -> T) -> T {
    let t = y - t0;
    if st0.value() == 0.0 {
        return f(t);
    }
    let width = if st0.value() < t.value() { st0 } else { t };
    integrate_unit(&mut |s| f(t - width * s), tol) * (width / st0)
}

// Full model elements.

/// Log density of one element of the full model at the upper boundary.
fn log_density_full<T: Real>([y, a, t0, w, v, sv, sw, st0]: [T; 8], tol: f64) -> T {
    let t = y - t0;
    if sw.value() == 0.0 && st0.value() == 0.0 {
        return log_density_upper(t, a, v, w, sv);
    }
    // Scale by the density at a central point so that the integrands stay near one.
    let mid = t.value() - 0.5 * st0.value().min(t.value());
    let shift = finite_or_zero(log_density_upper(
        mid,
        a.value(),
        v.value(),
        w.value(),
        sv.value(),
    ));
    let mut at_time = |time: T| {
        avg_w(w, sw, tol, &mut |ww| {
            (log_density_upper(time, a, v, ww, sv) - shift).exp()
        })
    };
    avg_t(y, t0, st0, tol, &mut at_time).ln() + shift
}

/// Unnormalized log CDF (or, with `complement`, log complementary CDF) of one element of the full
/// model at the upper boundary.
fn log_cdf_full<T: Real>([y, a, t0, w, v, sv, sw, st0]: [T; 8], tol: f64, complement: bool) -> T {
    let t = y - t0;
    if sv.value() == 0.0 && sw.value() == 0.0 && st0.value() == 0.0 {
        return log_cdf_upper(t, a, v, w, complement);
    }
    if !complement && t.value() <= 0.0 {
        return neg_inf();
    }
    let shift = finite_or_zero(log_cdf_upper(
        t.value(),
        a.value(),
        v.value(),
        w.value(),
        complement,
    ));
    let mut at_time = |time: T| {
        avg_wv(w, sw, v, sv, tol, &mut |ww, vv| {
            (log_cdf_upper(time, a, vv, ww, complement) - shift).exp()
        })
    };
    if complement && st0.value() > 0.0 && (t0 + st0).value() > y.value() {
        // Trials whose nondecision time exceeds y have not finished, so each contributes the whole
        // probability of the upper boundary.
        let p_bar = avg_wv(w, sw, v, sv, tol, &mut |ww, vv| {
            (log_prob_upper(a, vv, ww) - shift).exp()
        });
        let started = if t.value() > 0.0 {
            avg_t(y, t0, st0, tol, &mut at_time)
        } else {
            T::zero()
        };
        let unstarted = (t0 + st0 - y).min(st0) / st0;
        return (started + p_bar * unstarted).ln() + shift;
    }
    avg_t(y, t0, st0, tol, &mut at_time).ln() + shift
}

// Public functions.

/// Log density of the Wiener diffusion model with normal inter-trial drift variability, at the
/// upper boundary (Stan's five parameter `wiener_lpdf`): boundary separation `a`, nondecision time
/// `t0`, relative starting point `w`, mean drift `v`, drift standard deviation `sv`.
///
/// The drift distribution is integrated out in closed form, and the series is truncated at a
/// relative error of about `1e-12`. Requires `y > t0`, `y` and `a` positive finite, `t0` and `sv`
/// nonnegative finite, `w` in `(0, 1)`, `v` finite.
pub fn wiener5_lpdf<Y, A, T0, W, V, SV>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    sv: SV,
) -> Result<WienerOut<(Y, A, T0, W, V, SV), 6>>
where
    (Y, A, T0, W, V, SV): WienerArgs<6>,
{
    const F: &str = "wiener5_lpdf";
    sum_elements(
        F,
        (y, a, t0, w, v, sv),
        |[y, a, t0, w, v, sv]| check_element(F, [y, a, t0, w, v, sv, 0.0, 0.0], true),
        |[y, a, t0, w, v, sv]| log_density_upper(y - t0, a, v, w, sv),
    )
}

/// Log density of the seven parameter Wiener diffusion model at the upper boundary (Stan's seven
/// parameter `wiener_lpdf`).
///
/// As [`wiener5_lpdf`], with the relative starting point uniform on `[w - sw / 2, w + sw / 2]`
/// and the nondecision time uniform on `[t0, t0 + st0]`. Those two distributions are integrated
/// numerically to a relative tolerance of `1e-6`; see [`wiener_full_prec_lpdf`]. Further requires
/// `sw` in `[0, 1)` with `w +/- sw / 2` in `(0, 1)`, and `st0` nonnegative finite.
#[allow(clippy::too_many_arguments)]
pub fn wiener_full_lpdf<Y, A, T0, W, V, SV, SW, ST0>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    sv: SV,
    sw: SW,
    st0: ST0,
) -> Result<WienerOut<(Y, A, T0, W, V, SV, SW, ST0), 8>>
where
    (Y, A, T0, W, V, SV, SW, ST0): WienerArgs<8>,
{
    wiener_full_prec_lpdf(y, a, t0, w, v, sv, sw, st0, DEFAULT_PRECISION)
}

/// [`wiener_full_lpdf`] with an explicit `precision` (positive finite, Stan's default `1e-4`).
///
/// The quadratures use the relative tolerance `min(precision, 1e-6)`. In Stan the argument sets
/// the accuracy of the derivatives; here the derivatives are exact for the computed approximation,
/// so `precision` can only tighten it.
#[allow(clippy::too_many_arguments)]
pub fn wiener_full_prec_lpdf<Y, A, T0, W, V, SV, SW, ST0>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    sv: SV,
    sw: SW,
    st0: ST0,
    precision: f64,
) -> Result<WienerOut<(Y, A, T0, W, V, SV, SW, ST0), 8>>
where
    (Y, A, T0, W, V, SV, SW, ST0): WienerArgs<8>,
{
    const F: &str = "wiener_full_lpdf";
    let tol = tolerance(F, precision)?;
    sum_elements(
        F,
        (y, a, t0, w, v, sv, sw, st0),
        |x| check_element(F, x, true),
        |x| log_density_full(x, tol),
    )
}

/// Unnormalized log CDF of the four parameter Wiener model at the upper boundary:
/// `ln P(T <= y, absorbed at the upper boundary)`, which tends to the log probability of the upper
/// boundary rather than to zero as `y` grows.
///
/// Arguments as for the four parameter `wiener_lpdf`, with `a`, `t0`, `w`, `v` checked the same
/// way; `y` only must not be NaN, and `y <= t0` gives `-inf`. The series are truncated at a
/// relative error of about `1e-12`.
pub fn wiener4_lcdf_unnorm<Y, A, T0, W, V>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
) -> Result<WienerOut<(Y, A, T0, W, V), 5>>
where
    (Y, A, T0, W, V): WienerArgs<5>,
{
    wiener4_tail("wiener4_lcdf_unnorm", (y, a, t0, w, v), false)
}

/// Unnormalized log complementary CDF of the four parameter Wiener model at the upper boundary:
/// `ln P(T > y, absorbed at the upper boundary)`. Arguments as for [`wiener4_lcdf_unnorm`];
/// `y <= t0` gives the log probability of the upper boundary.
pub fn wiener4_lccdf_unnorm<Y, A, T0, W, V>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
) -> Result<WienerOut<(Y, A, T0, W, V), 5>>
where
    (Y, A, T0, W, V): WienerArgs<5>,
{
    wiener4_tail("wiener4_lccdf_unnorm", (y, a, t0, w, v), true)
}

fn wiener4_tail<X: WienerArgs<5>>(f: &'static str, args: X, complement: bool) -> Result<X::Out> {
    sum_elements(
        f,
        args,
        |[y, a, t0, w, v]| check_element(f, [y, a, t0, w, v, 0.0, 0.0, 0.0], false),
        |[y, a, t0, w, v]| log_cdf_upper(y - t0, a, v, w, complement),
    )
}

/// Stan language name of [`wiener4_lcdf_unnorm`] (four parameters).
pub fn wiener_lcdf_unnorm<Y, A, T0, W, V>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
) -> Result<WienerOut<(Y, A, T0, W, V), 5>>
where
    (Y, A, T0, W, V): WienerArgs<5>,
{
    wiener4_lcdf_unnorm(y, a, t0, w, v)
}

/// Stan language name of [`wiener4_lccdf_unnorm`] (four parameters).
pub fn wiener_lccdf_unnorm<Y, A, T0, W, V>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
) -> Result<WienerOut<(Y, A, T0, W, V), 5>>
where
    (Y, A, T0, W, V): WienerArgs<5>,
{
    wiener4_lccdf_unnorm(y, a, t0, w, v)
}

/// Unnormalized log CDF of the seven parameter Wiener model at the upper boundary,
/// `ln P(T <= y, absorbed at the upper boundary)`, with arguments as for [`wiener_full_lpdf`]
/// except that `y` only must not be NaN.
///
/// The drift distribution has no closed form here, so it is integrated numerically together with
/// the starting point and nondecision time distributions, to a relative tolerance of `1e-6`; see
/// [`wiener_full_prec_lcdf_unnorm`].
#[allow(clippy::too_many_arguments)]
pub fn wiener_full_lcdf_unnorm<Y, A, T0, W, V, SV, SW, ST0>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    sv: SV,
    sw: SW,
    st0: ST0,
) -> Result<WienerOut<(Y, A, T0, W, V, SV, SW, ST0), 8>>
where
    (Y, A, T0, W, V, SV, SW, ST0): WienerArgs<8>,
{
    wiener_full_prec_lcdf_unnorm(y, a, t0, w, v, sv, sw, st0, DEFAULT_PRECISION)
}

/// Unnormalized log complementary CDF of the seven parameter Wiener model at the upper boundary,
/// `ln P(T > y, absorbed at the upper boundary)`. Arguments as for [`wiener_full_lcdf_unnorm`].
#[allow(clippy::too_many_arguments)]
pub fn wiener_full_lccdf_unnorm<Y, A, T0, W, V, SV, SW, ST0>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    sv: SV,
    sw: SW,
    st0: ST0,
) -> Result<WienerOut<(Y, A, T0, W, V, SV, SW, ST0), 8>>
where
    (Y, A, T0, W, V, SV, SW, ST0): WienerArgs<8>,
{
    wiener_full_prec_lccdf_unnorm(y, a, t0, w, v, sv, sw, st0, DEFAULT_PRECISION)
}

/// [`wiener_full_lcdf_unnorm`] with an explicit `precision`, as for [`wiener_full_prec_lpdf`].
#[allow(clippy::too_many_arguments)]
pub fn wiener_full_prec_lcdf_unnorm<Y, A, T0, W, V, SV, SW, ST0>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    sv: SV,
    sw: SW,
    st0: ST0,
    precision: f64,
) -> Result<WienerOut<(Y, A, T0, W, V, SV, SW, ST0), 8>>
where
    (Y, A, T0, W, V, SV, SW, ST0): WienerArgs<8>,
{
    wiener_full_tail(
        "wiener_full_lcdf_unnorm",
        (y, a, t0, w, v, sv, sw, st0),
        precision,
        false,
    )
}

/// [`wiener_full_lccdf_unnorm`] with an explicit `precision`, as for [`wiener_full_prec_lpdf`].
#[allow(clippy::too_many_arguments)]
pub fn wiener_full_prec_lccdf_unnorm<Y, A, T0, W, V, SV, SW, ST0>(
    y: Y,
    a: A,
    t0: T0,
    w: W,
    v: V,
    sv: SV,
    sw: SW,
    st0: ST0,
    precision: f64,
) -> Result<WienerOut<(Y, A, T0, W, V, SV, SW, ST0), 8>>
where
    (Y, A, T0, W, V, SV, SW, ST0): WienerArgs<8>,
{
    wiener_full_tail(
        "wiener_full_lccdf_unnorm",
        (y, a, t0, w, v, sv, sw, st0),
        precision,
        true,
    )
}

fn wiener_full_tail<X: WienerArgs<8>>(
    f: &'static str,
    args: X,
    precision: f64,
    complement: bool,
) -> Result<X::Out> {
    let tol = tolerance(f, precision)?;
    sum_elements(
        f,
        args,
        |x| check_element(f, x, false),
        |x| log_cdf_full(x, tol, complement),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous_extra::wiener_lpdf;

    fn close(x: f64, y: f64, rel: f64) -> bool {
        (x - y).abs() <= rel * (1.0 + y.abs())
    }

    #[test]
    fn reduces_to_the_four_parameter_density() {
        for y in [0.35, 0.8, 2.5] {
            let four = wiener_lpdf(y, 1.4, 0.2, 0.45, 0.7).unwrap();
            let five = wiener5_lpdf(y, 1.4, 0.2, 0.45, 0.7, 0.0).unwrap();
            let full = wiener_full_lpdf(y, 1.4, 0.2, 0.45, 0.7, 0.0, 0.0, 0.0).unwrap();
            assert!(close(five, four, 1e-10), "{five} {four}");
            assert!(close(full, four, 1e-10), "{full} {four}");
        }
    }

    #[test]
    fn four_parameter_cdf_matches_density_and_probability() {
        let (a, t0, w, v) = (1.2, 0.15, 0.4, -0.6);
        let cdf = |y: f64| wiener4_lcdf_unnorm(y, a, t0, w, v).unwrap().exp();
        let h = 1e-5;
        // 0.3 and 0.9 use the small time series, 3.0 the large time series.
        for y in [0.3, 0.9, 3.0] {
            let numeric = (cdf(y + h) - cdf(y - h)) / (2.0 * h);
            let density = wiener5_lpdf(y, a, t0, w, v, 0.0).unwrap().exp();
            assert!(close(numeric, density, 1e-6), "{y}: {numeric} {density}");
            let total = cdf(y) + wiener4_lccdf_unnorm(y, a, t0, w, v).unwrap().exp();
            let p = log_prob_upper(a, v, w).exp();
            assert!(close(total, p, 1e-10), "{y}: {total} {p}");
        }
    }

    #[test]
    fn full_cdf_matches_full_density() {
        // The CDF integrates the drift numerically and the density in closed form.
        let (a, t0, w, v, sv, sw, st0) = (1.1, 0.2, 0.5, 0.8, 0.9, 0.2, 0.15);
        let cdf = |y: f64| {
            wiener_full_prec_lcdf_unnorm(y, a, t0, w, v, sv, sw, st0, 1e-10)
                .unwrap()
                .exp()
        };
        let h = 1e-4;
        // 0.3 lies inside the nondecision range [0.2, 0.35].
        for y in [0.3, 0.6, 1.5] {
            let numeric = (cdf(y + h) - cdf(y - h)) / (2.0 * h);
            let density = wiener_full_prec_lpdf(y, a, t0, w, v, sv, sw, st0, 1e-10)
                .unwrap()
                .exp();
            assert!(close(numeric, density, 1e-4), "{y}: {numeric} {density}");
            let survival = wiener_full_prec_lccdf_unnorm(y, a, t0, w, v, sv, sw, st0, 1e-10)
                .unwrap()
                .exp();
            let total = cdf(y) + survival;
            let large = cdf(1e3);
            assert!(close(total, large, 1e-8), "{y}: {total} {large}");
        }
    }
}
