//! Differentiable special functions.
//!
//! Values come from `stanmath_rs_core::special`; this module adds derivative rules. The regularized
//! incomplete gamma and beta functions ([`GammaInc`], [`BetaInc`]) differentiate their series and
//! continued fractions in forward mode to obtain shape derivatives.

use core::f64::consts::FRAC_2_SQRT_PI;

use stanmath_rs_core::special as sp;

pub use crate::incomplete::{BetaInc, GammaInc};
use crate::real::{BinaryFn, Real, UnaryFn};

macro_rules! unary {
    ($(#[$doc:meta])* $name:ident, |$x:ident| $value:expr, |$dx:ident, $dy:ident| $deriv:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl UnaryFn for $name {
            #[inline]
            fn value(&self, $x: f64) -> f64 {
                $value
            }

            #[inline]
            #[allow(unused_variables)]
            fn derivative<S: Real>(&self, $dx: S, $dy: S) -> S {
                $deriv
            }
        }
    };
}

unary!(
    /// `ln |Γ(x)|`; derivative `ψ(x)`.
    Lgamma,
    |x| sp::lgamma(x),
    |x, y| digamma(x)
);
unary!(
    /// `ln Γ(1 + a)`, accurate for small `a`; derivative `ψ(1 + a)`.
    Lgamma1p,
    |x| sp::lgamma1p(x),
    |x, y| digamma(x + 1.0)
);
unary!(
    /// `Γ(x)`; derivative `Γ(x) ψ(x)`.
    Tgamma,
    |x| sp::tgamma(x),
    |x, y| y * digamma(x)
);
unary!(
    /// Digamma `ψ(x)`; derivative `ψ'(x)`.
    Digamma,
    |x| sp::digamma(x),
    |x, y| trigamma(x)
);
unary!(
    /// Trigamma `ψ'(x)`; derivative `ψ''(x)`.
    Trigamma,
    |x| sp::trigamma(x),
    |x, y| polygamma(2, x)
);
unary!(
    /// Error function; derivative `2 exp(-x^2) / sqrt(pi)`.
    Erf,
    |x| sp::erf(x),
    |x, y| (-x.square()).exp() * FRAC_2_SQRT_PI
);
unary!(
    /// Complementary error function; derivative `-2 exp(-x^2) / sqrt(pi)`.
    Erfc,
    |x| sp::erfc(x),
    |x, y| (-x.square()).exp() * -FRAC_2_SQRT_PI
);
unary!(
    /// `exp(x^2) erfc(x)`; derivative `2 x erfcx(x) - 2 / sqrt(pi)`.
    ///
    /// The derivative loses about `log10(2 x^2)` digits to cancellation for large `x`.
    Erfcx,
    |x| sp::erfcx(x),
    |x, y| x * y * 2.0 - FRAC_2_SQRT_PI
);
unary!(
    /// Standard normal CDF `Φ(x)`; derivative `φ(x)`.
    StdNormalCdf,
    |x| sp::Phi(x),
    |x, y| x.map_unary(&StdNormalPdf)
);
unary!(
    /// Standard normal density `φ(x)`; derivative `-x φ(x)`.
    StdNormalPdf,
    |x| sp::std_normal_pdf(x),
    |x, y| -(x * y)
);
unary!(
    /// `ln Φ(x)`; derivative is the inverse Mills ratio `φ(x) / Φ(x)`.
    LogPhi,
    |x| sp::log_Phi(x),
    |x, y| x.map_unary(&InvMillsRatio)
);
unary!(
    /// Inverse Mills ratio `m(x) = φ(x) / Φ(x)`; derivative `-m(x) (x + m(x))`.
    InvMillsRatio,
    |x| sp::inv_mills_ratio(x),
    |x, y| -(y * x.map_unary(&InvMillsShift))
);
unary!(
    /// `x + φ(x) / Φ(x)`, computed without cancellation in the lower tail; derivative
    /// `1 - m(x) (x + m(x))`.
    InvMillsShift,
    |x| inv_mills_shift(x),
    |x, y| -(x.map_unary(&InvMillsRatio) * y) + 1.0
);
unary!(
    /// Inverse standard normal CDF; derivative `1 / φ(Φ⁻¹(p))`.
    InvPhi,
    |x| sp::inv_Phi(x),
    |x, y| y.map_unary(&StdNormalPdf).recip()
);
unary!(
    /// `ln(1 + u) - u`; derivative `-u / (1 + u)`.
    Log1pmx,
    |x| sp::log1pmx(x),
    |x, y| -x / (x + 1.0)
);
unary!(
    /// Stirling correction `δ(x)` of `ln Γ(x)`; derivative `ψ(x) - ln x + 1 / (2x)`, by its own
    /// asymptotic series for large `x` where that difference cancels.
    LgammaStirlingDiff,
    |x| sp::lgamma_stirling_diff(x),
    |x, y| stirling_diff_derivative(x)
);

/// Polygamma `ψ^(n)(x)`; derivative `ψ^(n+1)(x)`.
#[derive(Clone, Copy, Debug)]
pub struct Polygamma(pub u32);

impl UnaryFn for Polygamma {
    #[inline]
    fn value(&self, x: f64) -> f64 {
        sp::polygamma(self.0, x)
    }

    #[inline]
    fn derivative<S: Real>(&self, x: S, _y: S) -> S {
        polygamma(self.0 + 1, x)
    }
}

/// `x + φ(x) / Φ(x)`. For `x < -2` it uses the continued fraction
/// `x + m(x) = sqrt(2) r(z)`, `z = -x / sqrt(2)`, `r(z) = (1/2) / (z + 1 / (z + (3/2) / (z + ...)))`
/// (Laplace's continued fraction for `erfc`), which has no cancellation.
fn inv_mills_shift(x: f64) -> f64 {
    if x >= -2.0 || x.is_nan() {
        return x + sp::inv_mills_ratio(x);
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }
    let z = -x / core::f64::consts::SQRT_2;
    // Backward evaluation of K = (1/2) / (z + 1 / (z + (3/2) / (z + 2 / (z + ...)))).
    let mut tail = z;
    for k in (1..=120).rev() {
        tail = z + (f64::from(k) * 0.5 + 0.5) / tail;
    }
    core::f64::consts::SQRT_2 * 0.5 / tail
}

/// `B_{2k} / (2k)` for `k = 1..10`: the coefficients of the digamma asymptotic series.
const DIGAMMA_COEF: [f64; 10] = [
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
    1.0 / 12.0,
    -3617.0 / 8160.0,
    43867.0 / 14364.0,
    -174611.0 / 6600.0,
];

/// `δ'(x)`, the derivative of the Stirling correction, without the cancellation in
/// `ψ(x) - ln x + 1 / (2x)`.
fn stirling_diff_derivative<S: Real>(x: S) -> S {
    if x.value() < 1.0 {
        return digamma(x) - x.ln() + (x * 2.0).recip();
    }
    // δ(x) = δ(x + 1) + g(x), g(x) = sum_m t^(2m) / (2m + 1), t = 1 / (2x + 1), dt/dx = -2 t^2,
    // so g'(x) = -sum_m 4m t^(2m + 1) / (2m + 1).
    let mut x = x;
    let mut acc = S::zero();
    while x.value() < 10.0 {
        let t = (x * 2.0 + 1.0).recip();
        let t2 = t.square();
        let mut pow = t * t2;
        let mut sum = S::zero();
        for m in 1..34 {
            let m = f64::from(m);
            sum += pow * (4.0 * m / (2.0 * m + 1.0));
            pow *= t2;
        }
        acc -= sum;
        x += 1.0;
    }
    // d/dx sum_k B_{2k} / (2k (2k - 1) x^{2k - 1}) = -sum_k B_{2k} / (2k x^{2k})
    let inv2 = x.square().recip();
    let mut pow = inv2;
    let mut sum = S::zero();
    for c in DIGAMMA_COEF {
        sum += pow * c;
        pow *= inv2;
    }
    acc - sum
}

/// `ψ(x) - ψ(x + d)` for `x > 0`, `x + d > 0`, without cancellation when `d` is small relative to
/// `x` or both are large.
///
/// Shifts both arguments up with `ψ(y) = ψ(y + 1) - 1 / y` (the differences combine exactly into
/// `d / (y (y + d))`), then uses the asymptotic series term by term:
/// `ψ(x) - ψ(x + d) = -ln(1 + d / x) - d / (2 x (x + d)) - sum_k c_k (x^(-2k) - (x + d)^(-2k))`.
pub fn digamma_diff<S: Real>(x: S, d: S) -> S {
    let (xv, dv) = (x.value(), d.value());
    if xv.is_nan() || dv.is_nan() || xv <= 0.0 || xv + dv <= 0.0 {
        return digamma(x) - digamma(x + d);
    }
    let mut x = x;
    let mut acc = S::zero();
    while x.value() < 10.0 || (x + d).value() < 10.0 {
        // 1/x - 1/(x + d), written so that neither the value nor its derivatives cancel: the
        // product form when d is small relative to x, the difference form otherwise.
        acc += if d.value().abs() < x.value() {
            d / (x * (x + d))
        } else {
            x.recip() - (x + d).recip()
        };
        x += 1.0;
    }
    let r = d / x;
    let log_ratio = r.ln_1p();
    let mut sum = S::zero();
    let mut inv2k = S::one();
    let inv2 = x.square().recip();
    for (k, c) in DIGAMMA_COEF.iter().enumerate() {
        inv2k *= inv2;
        // x^(-2k) - (x + d)^(-2k) = x^(-2k) (1 - (1 + r)^(-2k)) = -x^(-2k) expm1(-2k ln(1 + r))
        let k2 = 2.0 * (k + 1) as f64;
        sum += -(inv2k * (log_ratio * -k2).exp_m1()) * *c;
    }
    -acc - log_ratio - d / (x * (x + d) * 2.0) - sum
}

/// `ln B(a, b)`; partials `ψ(a) - ψ(a + b)` and `ψ(b) - ψ(a + b)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lbeta;

impl BinaryFn for Lbeta {
    fn value(&self, a: f64, b: f64) -> f64 {
        sp::lbeta(a, b)
    }

    fn partial_a<S: Real>(&self, a: S, b: S, _y: S) -> S {
        digamma_diff(a, b)
    }

    fn partial_b<S: Real>(&self, a: S, b: S, _y: S) -> S {
        digamma_diff(b, a)
    }
}

/// `ln |Γ(x)|`.
#[inline]
pub fn lgamma<T: Real>(x: T) -> T {
    x.map_unary(&Lgamma)
}

/// `ln Γ(1 + a)`, accurate for small `a`.
#[inline]
pub fn lgamma1p<T: Real>(a: T) -> T {
    a.map_unary(&Lgamma1p)
}

/// `Γ(x)`.
#[inline]
pub fn tgamma<T: Real>(x: T) -> T {
    x.map_unary(&Tgamma)
}

/// Digamma `ψ(x)`.
#[inline]
pub fn digamma<T: Real>(x: T) -> T {
    x.map_unary(&Digamma)
}

/// Trigamma `ψ'(x)`.
#[inline]
pub fn trigamma<T: Real>(x: T) -> T {
    x.map_unary(&Trigamma)
}

/// Polygamma `ψ^(n)(x)`; `n >= 2` requires `x > 0`.
#[inline]
pub fn polygamma<T: Real>(n: u32, x: T) -> T {
    match n {
        0 => digamma(x),
        1 => trigamma(x),
        _ => x.map_unary(&Polygamma(n)),
    }
}

/// Error function.
#[inline]
pub fn erf<T: Real>(x: T) -> T {
    x.map_unary(&Erf)
}

/// Complementary error function.
#[inline]
pub fn erfc<T: Real>(x: T) -> T {
    x.map_unary(&Erfc)
}

/// Scaled complementary error function `exp(x^2) erfc(x)`.
#[inline]
pub fn erfcx<T: Real>(x: T) -> T {
    x.map_unary(&Erfcx)
}

/// Standard normal CDF `Φ(x)`.
#[allow(non_snake_case)]
#[inline]
pub fn Phi<T: Real>(x: T) -> T {
    x.map_unary(&StdNormalCdf)
}

/// `ln Φ(x)`, accurate in both tails.
#[allow(non_snake_case)]
#[inline]
pub fn log_Phi<T: Real>(x: T) -> T {
    x.map_unary(&LogPhi)
}

/// Inverse standard normal CDF `Φ⁻¹(p)`.
#[allow(non_snake_case)]
#[inline]
pub fn inv_Phi<T: Real>(p: T) -> T {
    p.map_unary(&InvPhi)
}

/// `ln(1 + u) - u`.
#[inline]
pub fn log1pmx<T: Real>(u: T) -> T {
    u.map_unary(&Log1pmx)
}

/// Stirling correction of `ln Γ(x)`.
#[inline]
pub fn lgamma_stirling_diff<T: Real>(x: T) -> T {
    x.map_unary(&LgammaStirlingDiff)
}

/// `ln B(a, b)`.
#[inline]
pub fn lbeta<T: Real>(a: T, b: T) -> T {
    a.map_binary(b, &Lbeta)
}

/// `ln (n choose k)` for real arguments.
#[inline]
pub fn lchoose<T: Real>(n: T, k: T) -> T {
    -(n + 1.0).ln() - lbeta(n - k + 1.0, k + 1.0)
}

/// Regularized lower incomplete gamma `P(a, x)`.
#[inline]
pub fn gamma_p<T: Real>(a: T, x: T) -> T {
    a.map_binary(x, &GammaInc { upper: false })
}

/// Regularized upper incomplete gamma `Q(a, x) = 1 - P(a, x)`.
#[inline]
pub fn gamma_q<T: Real>(a: T, x: T) -> T {
    a.map_binary(x, &GammaInc { upper: true })
}

/// Regularized incomplete beta `I_x(a, b)`.
#[inline]
pub fn beta_inc<T: Real>(a: T, b: T, x: T) -> T {
    T::map_nary(&[a, b, x], &BetaInc)
}
