//! Differentiable mathematical functions with Stan-style names.
//!
//! Every function is generic over [`Real`], so it works on `f64`, [`Dual`](crate::Dual), and
//! [`Var`](crate::Var). The derivative rule of each is listed on the corresponding primitive in
//! [`crate::primitives`].
//!
//! # Conventions at nondifferentiable points
//!
//! * `abs`, `fabs`: derivative zero at zero.
//! * `fmax`, `fmin`: ties return the second argument and take its derivative, as Stan Math.
//! * `floor`, `ceil`, `round`, `trunc`: derivative zero everywhere.
//! * `pow(a, b)` at `a = 0`: the partial with respect to `b` is zero.

use crate::primitives as p;
use crate::real::Real;

pub use crate::complex::{
    arg, complex_abs, complex_arg, complex_norm, conj, get_imag, get_real, norm, polar, proj,
    to_complex,
};
pub use crate::functions_extra::*;
pub use crate::library::*;
pub use crate::overloads::*;
pub use crate::scalar_wave2::*;
pub use crate::special_numerics::{
    bessel_first_kind, bessel_second_kind, hypergeometric_2F1, hypergeometric_3F2,
    hypergeometric_pFq, lambert_w0, lambert_wm1, log_modified_bessel_first_kind,
    modified_bessel_first_kind, modified_bessel_second_kind, owens_t,
};

pub use crate::special::{
    Phi, beta_inc, digamma, erf, erfc, erfcx, gamma_p, gamma_q, inv_Phi, lbeta, lchoose, lgamma,
    lgamma_stirling_diff, lgamma1p, log_Phi, log1pmx, polygamma, tgamma, trigamma,
};

macro_rules! unary_fns {
    ($($(#[$doc:meta])* $name:ident => $prim:expr;)*) => {
        $(
            $(#[$doc])*
            #[inline]
            pub fn $name<T: Real>(x: T) -> T {
                x.map_unary(&$prim)
            }
        )*
    };
}

unary_fns! {
    /// Exponential.
    exp => p::Exp;
    /// `2^x`.
    exp2 => p::Exp2;
    /// `exp(x) - 1`, accurate near zero.
    expm1 => p::ExpM1;
    /// Natural logarithm.
    log => p::Ln;
    /// `ln(1 + x)`, accurate near zero.
    log1p => p::Ln1p;
    /// `ln(1 - x)`, accurate near zero.
    log1m => p::Log1m;
    /// Base 2 logarithm.
    log2 => p::Log2;
    /// Base 10 logarithm.
    log10 => p::Log10;
    /// Square root.
    sqrt => p::Sqrt;
    /// Cube root.
    cbrt => p::Cbrt;
    /// `x^2`.
    square => p::Square;
    /// `1 / x`.
    inv => p::Recip;
    /// Absolute value; derivative zero at zero.
    abs => p::Abs;
    /// Absolute value (Stan's name for real arguments).
    fabs => p::Abs;
    /// Sine.
    sin => p::Sin;
    /// Cosine.
    cos => p::Cos;
    /// Tangent.
    tan => p::Tan;
    /// Arcsine.
    asin => p::Asin;
    /// Arccosine.
    acos => p::Acos;
    /// Arctangent.
    atan => p::Atan;
    /// Hyperbolic sine.
    sinh => p::Sinh;
    /// Hyperbolic cosine.
    cosh => p::Cosh;
    /// Hyperbolic tangent.
    tanh => p::Tanh;
    /// Inverse hyperbolic sine.
    asinh => p::Asinh;
    /// Inverse hyperbolic cosine.
    acosh => p::Acosh;
    /// Inverse hyperbolic tangent.
    atanh => p::Atanh;
    /// `ln(1 + exp(x))`.
    log1p_exp => p::Log1pExp;
    /// `ln(1 + exp(x))` (machine learning name for [`log1p_exp`]).
    softplus => p::Log1pExp;
    /// `ln(1 - exp(x))` for `x < 0`.
    log1m_exp => p::Log1mExp;
    /// Logistic sigmoid `1 / (1 + exp(-x))`.
    inv_logit => p::InvLogit;
    /// Log odds `ln(p / (1 - p))`.
    logit => p::Logit;
    /// `ln(inv_logit(x))`.
    log_inv_logit => p::LogInvLogit;
    /// `ln(inv_logit(x))` (machine learning name for [`log_inv_logit`]).
    log_sigmoid => p::LogInvLogit;
    /// `ln(1 - inv_logit(x))`.
    log1m_inv_logit => p::Log1mInvLogit;
}

/// `1 / sqrt(x)`.
#[inline]
pub fn inv_sqrt<T: Real>(x: T) -> T {
    x.powf(-0.5)
}

/// `1 / x^2`.
#[inline]
pub fn inv_square<T: Real>(x: T) -> T {
    x.powi(-2)
}

/// `a^b` with both arguments differentiable.
#[inline]
pub fn pow<T: Real>(a: T, b: T) -> T {
    a.map_binary(b, &p::Pow)
}

/// Four-quadrant arctangent of `y / x`.
#[inline]
pub fn atan2<T: Real>(y: T, x: T) -> T {
    y.map_binary(x, &p::Atan2)
}

/// `sqrt(a^2 + b^2)`.
#[inline]
pub fn hypot<T: Real>(a: T, b: T) -> T {
    a.map_binary(b, &p::Hypot)
}

/// `ln(exp(a) + exp(b))`.
#[inline]
pub fn log_sum_exp<T: Real>(a: T, b: T) -> T {
    a.map_binary(b, &p::LogSumExp)
}

/// `ln(sum(exp(x)))`; `-inf` for an empty slice. Records a single node in reverse mode.
#[inline]
pub fn log_sum_exp_slice<T: Real>(x: &[T]) -> T {
    T::map_nary(x, &p::LogSumExpN)
}

/// `ln(exp(a) - exp(b))` for `a >= b`.
#[inline]
pub fn log_diff_exp<T: Real>(a: T, b: T) -> T {
    a.map_binary(b, &p::LogDiffExp)
}

/// Sum of a slice. Records a single node in reverse mode.
#[inline]
pub fn sum<T: Real>(x: &[T]) -> T {
    T::map_nary(x, &p::SumN)
}

/// Dot product. Records a single node in reverse mode.
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn dot_product<T: Real>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len(), "dot_product: slices differ in length");
    let mut joined = Vec::with_capacity(2 * a.len());
    joined.extend_from_slice(a);
    joined.extend_from_slice(b);
    T::map_nary(&joined, &p::DotHalves)
}

/// `a * b + c`.
#[inline]
pub fn fma<T: Real>(a: T, b: T, c: T) -> T {
    a * b + c
}

/// The larger argument; ties return `b` and take its derivative, as Stan Math.
#[inline]
pub fn fmax<T: Real>(a: T, b: T) -> T {
    if b.value() >= a.value() || a.is_nan() {
        b
    } else {
        a
    }
}

/// The smaller argument; ties return `b` and take its derivative, as Stan Math.
#[inline]
pub fn fmin<T: Real>(a: T, b: T) -> T {
    if b.value() <= a.value() || a.is_nan() {
        b
    } else {
        a
    }
}

/// Positive difference `max(a - b, 0)`.
#[inline]
pub fn fdim<T: Real>(a: T, b: T) -> T {
    if a.value() > b.value() {
        a - b
    } else {
        T::zero()
    }
}

/// Softmax of a slice, computed stably.
pub fn softmax<T: Real>(x: &[T]) -> Vec<T> {
    let lse = log_sum_exp_slice(x);
    x.iter().map(|&xi| (xi - lse).exp()).collect()
}

/// Log softmax of a slice, computed stably.
pub fn log_softmax<T: Real>(x: &[T]) -> Vec<T> {
    let lse = log_sum_exp_slice(x);
    x.iter().map(|&xi| xi - lse).collect()
}
