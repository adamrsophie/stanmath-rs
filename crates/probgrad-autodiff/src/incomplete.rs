//! Regularized incomplete gamma and beta functions with derivatives in every argument.
//!
//! The values use the classical series and continued fractions (DLMF 8.7.1, 8.9.2, 8.17.22),
//! written once, generically over [`Real`]. Derivatives with respect to the integration limit have
//! closed forms. Derivatives with respect to the shape parameters do not; they are obtained by
//! running the same algorithm on a dual number seeded in that parameter, so the result is again
//! an ordinary [`Real`] expression that nested modes can differentiate.
//!
//! # Convergence of derivatives
//!
//! Stopping tests use [`Real::magnitude`], so forward-mode tangents must converge too, not only the
//! value. This matters because the continued fractions terminate exactly at integer shapes: the
//! value then converges after finitely many terms while its derivatives still depend on the tail.
//! When the partial derivative will itself be differentiated by reverse mode (`S::ORDER > 0`),
//! the tape components are not visible to the stopping test. In that case a hyper-dual `f64`
//! evaluation first finds how many iterations the second derivatives need, and the nested
//! evaluation runs exactly that many.

use core::f64::consts::PI;

use crate::dual::Dual;
use crate::functions::log1p;
use crate::real::{BinaryFn, NaryFn, Real};
use crate::special::{lbeta, lgamma, lgamma_stirling_diff, lgamma1p, log1pmx};

const MAX_ITER: usize = 10_000;
const TOL: f64 = 1e-17;
const TINY: f64 = 1e-300;
/// Extra iterations beyond the oracle's count, as a margin for third and higher derivatives.
const MARGIN: usize = 8;

/// How an iterative algorithm decides to stop.
#[derive(Clone, Copy, Debug)]
enum Stop {
    /// Stop when the update is below tolerance in every forward-mode component.
    Converge,
    /// Run exactly this many iterations.
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

/// `exp(-x) * exp(log_rest)`, splitting the exponent when that is safe so that the rounding error
/// of `-x` is not mixed into a large cancelling sum.
fn exp_split<S: Real>(x: S, log_rest: S) -> S {
    if x.value() < 700.0 {
        (-x).exp() * log_rest.exp()
    } else {
        (log_rest - x).exp()
    }
}

/// `x^a exp(-x) / Γ(a)`, evaluated through `log1pmx` and the Stirling correction for large `a`.
fn gamma_prefactor<S: Real>(a: S, x: S) -> S {
    if a.value() < 10.0 {
        exp_split(x, a * x.ln() - lgamma(a))
    } else {
        // ln of the prefactor is a (ln t - t + 1) + ln(a / 2 pi) / 2 - δ(a) with t = x / a. Near
        // t = 1 the bracket is log1pmx(t - 1); far from it, ln t is formed directly because
        // 1 + (t - 1) would lose the digits of a small t.
        let u = (x - a) / a;
        let core = if u.value().abs() < 0.5 {
            a * log1pmx(u)
        } else {
            a * (x / a).ln() - x + a
        };
        (a / (2.0 * PI)).sqrt() * (core - lgamma_stirling_diff(a)).exp()
    }
}

/// Series for `P(a, x)` (DLMF 8.7.1), used when `x < a + 1`.
fn gamma_p_series<S: Real>(a: S, x: S, stop: Stop) -> (S, usize) {
    let mut term = S::one();
    let mut sum = S::one();
    let mut ap = a;
    let mut n = 0;
    while n < MAX_ITER {
        n += 1;
        ap += 1.0;
        term = term * x / ap;
        sum += term;
        if stop.done(n, || term.magnitude() < TOL * sum.magnitude()) {
            break;
        }
    }
    // x^a exp(-x) / Γ(a + 1); for small a this avoids dividing an O(a) prefactor by a, which
    // cancels in the derivatives with respect to a.
    let front = if a.value() < 10.0 {
        exp_split(x, a * x.ln() - lgamma1p(a))
    } else {
        gamma_prefactor(a, x) / a
    };
    (front * sum, n)
}

/// `Q(a, x)` for small `a` and `x` by `Q = -expm1(t) - a e^t sum_{n>=1} (-x)^n / (n! (a + n))`
/// with `t = a ln x - ln Γ(1 + a)` (from DLMF 8.7.1 and 8.4.4), which avoids the cancellation in
/// `1 - P` when `P` is close to one.
fn gamma_q_small_a<S: Real>(a: S, x: S, stop: Stop) -> (S, usize) {
    let t = a * x.ln() - lgamma1p(a);
    let mut power = S::one();
    let mut sum = S::zero();
    let mut n = 0;
    while n < MAX_ITER {
        n += 1;
        let nf = n as f64;
        power = -(power * x) / nf;
        let term = power / (a + nf);
        sum += term;
        if stop.done(n, || term.magnitude() < TOL * sum.magnitude()) {
            break;
        }
    }
    (-t.exp_m1() - a * t.exp() * sum, n)
}

/// Value of the continued fraction `b0 + a1 / (b1 + a2 / (b2 + ...))`, where `term(n)` returns
/// `(a_n, b_n)` for `n >= 1`, by the modified Lentz method (Lentz 1976; Thompson and Barnett 1986).
///
/// `b0` must be nonzero. Returns the value and the number of terms used.
fn continued_fraction<S: Real>(
    b0: S,
    mut term: impl FnMut(f64) -> (S, S),
    stop: Stop,
) -> (S, usize) {
    let tiny = S::from_f64(TINY);
    let guard = |v: S| if v.value().abs() < TINY { tiny } else { v };
    let mut value = guard(b0);
    let mut c = value;
    let mut d = S::zero();
    let mut n = 0;
    while n < MAX_ITER {
        n += 1;
        let (a, b) = term(n as f64);
        d = guard(b + a * d).recip();
        c = guard(b + a / c);
        let delta = c * d;
        value *= delta;
        if stop.done(n, || (delta - 1.0).magnitude() < TOL) {
            break;
        }
    }
    (value, n)
}

/// `Q(a, x)` by the continued fraction of DLMF 8.9.2, used when `x >= a + 1`:
/// `Γ(a, x) = x^a exp(-x) / (x + 1 - a + 1 (a - 1) / (x + 3 - a + 2 (a - 2) / (x + 5 - a + ...)))`.
fn gamma_q_fraction<S: Real>(a: S, x: S, stop: Stop) -> (S, usize) {
    let (denominator, n) =
        continued_fraction(x + 1.0 - a, |k| ((a - k) * k, x + 1.0 - a + 2.0 * k), stop);
    (gamma_prefactor(a, x) / denominator, n)
}

fn gamma_inc<S: Real>(a: S, x: S, upper: bool, stop: Stop) -> (S, usize) {
    let (av, xv) = (a.value(), x.value());
    if av.is_nan() || xv.is_nan() || av <= 0.0 || xv < 0.0 {
        return (S::from_f64(f64::NAN), 0);
    }
    if xv == 0.0 {
        return (S::from_f64(if upper { 1.0 } else { 0.0 }), 0);
    }
    if xv == f64::INFINITY {
        return (S::from_f64(if upper { 0.0 } else { 1.0 }), 0);
    }
    if xv < av + 1.0 {
        if upper && av < 1.0 && xv <= 1.5 {
            return gamma_q_small_a(a, x, stop);
        }
        let (p, n) = gamma_p_series(a, x, stop);
        (if upper { -p + 1.0 } else { p }, n)
    } else {
        let (q, n) = gamma_q_fraction(a, x, stop);
        (if upper { q } else { -q + 1.0 }, n)
    }
}

/// Iteration count for derivatives of order two, from a hyper-dual `f64` evaluation seeded along
/// the all-ones direction in `(a, x)`.
fn gamma_oracle<S: Real>(a: f64, x: f64, upper: bool) -> Stop {
    if S::ORDER == 0 {
        return Stop::Converge;
    }
    let seed = |v: f64| Dual::new(Dual::new(v, 1.0), Dual::new(1.0, 0.0));
    let (_, n) = gamma_inc(seed(a), seed(x), upper, Stop::Converge);
    Stop::After(n + MARGIN)
}

/// Regularized incomplete gamma `P(a, x)` (`upper = false`) or `Q(a, x)` (`upper = true`).
#[derive(Clone, Copy, Debug)]
pub struct GammaInc {
    /// `true` for `Q(a, x)`, `false` for `P(a, x)`.
    pub upper: bool,
}

impl BinaryFn for GammaInc {
    fn value(&self, a: f64, x: f64) -> f64 {
        gamma_inc(a, x, self.upper, Stop::Converge).0
    }

    fn partial_a<S: Real>(&self, a: S, x: S, _y: S) -> S {
        let stop = gamma_oracle::<S>(a.value(), x.value(), self.upper);
        gamma_inc(Dual::variable(a), Dual::constant(x), self.upper, stop)
            .0
            .eps
    }

    fn partial_b<S: Real>(&self, a: S, x: S, _y: S) -> S {
        let d = if x.value() == 0.0 {
            // x^(a-1) exp(-x) / Γ(a) at x = 0.
            let av = a.value();
            S::from_f64(if av > 1.0 {
                0.0
            } else if av == 1.0 {
                1.0
            } else {
                f64::INFINITY
            })
        } else {
            gamma_prefactor(a, x) / x
        };
        if self.upper { -d } else { d }
    }
}

/// `I_x(a, b)` by the continued fraction of DLMF 8.17.22, valid when `x < (a + 1) / (a + b + 2)`:
/// `I_x(a, b) = x^a (1 - x)^b / (a B(a, b)) / (1 + d1 / (1 + d2 / (1 + ...)))` with
/// `d(2m + 1) = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1))` and
/// `d(2m) = m (b - m) x / ((a + 2m - 1)(a + 2m))`.
fn beta_inc_fraction<S: Real>(a: S, b: S, x: S, stop: Stop) -> (S, usize) {
    let front = (a * x.ln() + b * log1p(-x) - lbeta(a, b)).exp() / a;
    let one = S::one();
    let d = |k: f64| {
        let numerator = if k % 2.0 == 1.0 {
            let m = (k - 1.0) * 0.5;
            -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        } else {
            let m = k * 0.5;
            (b - m) * x * m / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        };
        (numerator, one)
    };
    let (denominator, n) = continued_fraction(one, d, stop);
    (front / denominator, n)
}

fn beta_inc_generic<S: Real>(a: S, b: S, x: S, stop: Stop) -> (S, usize) {
    let (av, bv, xv) = (a.value(), b.value(), x.value());
    if av.is_nan() || bv.is_nan() || xv.is_nan() || av <= 0.0 || bv <= 0.0 {
        return (S::from_f64(f64::NAN), 0);
    }
    if !(0.0..=1.0).contains(&xv) {
        return (S::from_f64(f64::NAN), 0);
    }
    if xv == 0.0 {
        return (S::zero(), 0);
    }
    if xv == 1.0 {
        return (S::one(), 0);
    }
    if xv > (av + 1.0) / (av + bv + 2.0) {
        let (v, n) = beta_inc_fraction(b, a, -x + 1.0, stop);
        (-v + 1.0, n)
    } else {
        beta_inc_fraction(a, b, x, stop)
    }
}

fn beta_oracle<S: Real>(a: f64, b: f64, x: f64) -> Stop {
    if S::ORDER == 0 {
        return Stop::Converge;
    }
    let seed = |v: f64| Dual::new(Dual::new(v, 1.0), Dual::new(1.0, 0.0));
    let (_, n) = beta_inc_generic(seed(a), seed(b), seed(x), Stop::Converge);
    Stop::After(n + MARGIN)
}

/// Regularized incomplete beta `I_x(a, b)` with arguments `[a, b, x]`.
#[derive(Clone, Copy, Debug, Default)]
pub struct BetaInc;

impl NaryFn for BetaInc {
    fn value(&self, v: &[f64]) -> f64 {
        beta_inc_generic(v[0], v[1], v[2], Stop::Converge).0
    }

    fn partials<S: Real>(&self, v: &[S], _y: S, needed: &[bool], out: &mut [S]) {
        let (a, b, x) = (v[0], v[1], v[2]);
        let stop = if needed[0] || needed[1] {
            beta_oracle::<S>(a.value(), b.value(), x.value())
        } else {
            Stop::Converge
        };
        if needed[0] {
            let r = beta_inc_generic(
                Dual::variable(a),
                Dual::constant(b),
                Dual::constant(x),
                stop,
            );
            out[0] = r.0.eps;
        }
        if needed[1] {
            let r = beta_inc_generic(
                Dual::constant(a),
                Dual::variable(b),
                Dual::constant(x),
                stop,
            );
            out[1] = r.0.eps;
        }
        if needed[2] {
            out[2] = ((a - 1.0) * x.ln() + (b - 1.0) * log1p(-x) - lbeta(a, b)).exp();
        }
    }
}
