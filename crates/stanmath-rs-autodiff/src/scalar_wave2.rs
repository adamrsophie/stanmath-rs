//! Scalar helpers with Stan Math names: two hypergeometric functions, signed log sums,
//! classification and logical predicates, and integer arithmetic with Stan's semantics.
//!
//! Predicates and logical functions look at primal values only and carry no derivatives, like
//! Stan's comparison operators. They accept `f64`, `i32`, `i64`, and every [`Real`] through the
//! [`Primal`] trait.

use stanmath_rs_core::{Error, Result};

use crate::functions::{log_diff_exp, log_sum_exp};
use crate::real::Real;
use crate::special_numerics::hypergeometric_pFq;

/// A scalar whose primal value can be read as an `f64`: integers and every [`Real`].
pub trait Primal: Copy {
    /// The primal value.
    fn primal(self) -> f64;
}

impl<T: Real> Primal for T {
    #[inline]
    fn primal(self) -> f64 {
        self.value()
    }
}

impl Primal for i32 {
    #[inline]
    fn primal(self) -> f64 {
        f64::from(self)
    }
}

impl Primal for i64 {
    #[inline]
    fn primal(self) -> f64 {
        self as f64
    }
}

/// `1F0(a; ; z) = (1 - z)^(-a)` (DLMF 15.4.6 with `b = c`), with derivatives in `a` and `z`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] unless `|z| < 1`, where the defining series converges.
#[allow(non_snake_case)]
pub fn hypergeometric_1F0<T: Real>(a: T, z: T) -> Result<T> {
    if !(z.value().abs() < 1.0) {
        return Err(Error::parameter(
            "hypergeometric_1F0",
            "z",
            z.value(),
            "less than 1 in absolute value",
        ));
    }
    Ok((-(a * (-z).ln_1p())).exp())
}

/// The confluent hypergeometric function `2F2(a; b; z)` by its series, which converges for every
/// `z`, with derivatives in every argument. `NaN` when some `b` is a nonpositive integer.
#[allow(non_snake_case)]
#[inline]
pub fn hypergeometric_2F2<T: Real>(a: [T; 2], b: [T; 2], z: T) -> T {
    hypergeometric_pFq(&a, &b, z)
}

/// `ln |s|` and the sign of `s = a_sign exp(a) + b_sign exp(b)`, computed without overflow.
///
/// Signs are `1` or `-1`. When the two terms cancel exactly the log magnitude is `-inf` and the
/// sign is `a_sign`.
pub fn log_sum_exp_signed<T: Real>(a: T, a_sign: i32, b: T, b_sign: i32) -> (T, i32) {
    if a_sign == b_sign {
        (log_sum_exp(a, b), a_sign)
    } else if a.value() >= b.value() {
        (log_diff_exp(a, b), a_sign)
    } else {
        (log_diff_exp(b, a), b_sign)
    }
}

/// A right inverse of softmax: `ln(x)` elementwise, so `softmax(inverse_softmax(x)) = x` for a
/// simplex `x`. The result is one representative; adding a constant gives the others.
pub fn inverse_softmax<T: Real>(simplex: &[T]) -> Vec<T> {
    simplex.iter().map(|&x| x.ln()).collect()
}

/// `|x|` with the sign of `y`. The derivative in `x` is `1` or `-1`; in `y` it is zero.
#[inline]
pub fn copysign<T: Real>(x: T, y: T) -> T {
    if x.value().is_sign_negative() == y.value().is_sign_negative() {
        x
    } else {
        -x
    }
}

/// True when the sign bit of the primal value is set (including `-0.0` and negative NaN).
#[inline]
pub fn signbit<X: Primal>(x: X) -> bool {
    x.primal().is_sign_negative()
}

/// True when the primal value is finite.
#[inline]
pub fn isfinite<X: Primal>(x: X) -> bool {
    x.primal().is_finite()
}

/// True when the primal value is positive or negative infinity.
#[inline]
pub fn isinf<X: Primal>(x: X) -> bool {
    x.primal().is_infinite()
}

/// True when the primal value is NaN.
#[inline]
pub fn isnan<X: Primal>(x: X) -> bool {
    x.primal().is_nan()
}

/// True when the primal value is normal: finite, nonzero, and not subnormal.
#[inline]
pub fn isnormal<X: Primal>(x: X) -> bool {
    x.primal().is_normal()
}

/// True when the primal value is infinite (Stan's name for [`isinf`]).
#[inline]
pub fn is_inf<X: Primal>(x: X) -> bool {
    isinf(x)
}

/// True when the primal value is NaN (Stan's name for [`isnan`]).
#[inline]
pub fn is_nan<X: Primal>(x: X) -> bool {
    isnan(x)
}

/// The unbiased binary exponent of `x`, `floor(log2 |x|)`, read from the bits so that values
/// just below a power of two are exact. `-inf` at zero, `+inf` at infinity, NaN at NaN. The
/// derivative is zero.
pub fn logb<T: Real>(x: T) -> T {
    T::from_f64(logb_f64(x.value()))
}

fn logb_f64(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let bits = x.abs().to_bits();
    let biased = (bits >> 52) as i64;
    if biased == 0 {
        // Subnormal: value = mantissa * 2^-1074, so the exponent is that of the top set bit.
        let top = 63 - i64::from((bits & ((1u64 << 52) - 1)).leading_zeros());
        (top - 1074) as f64
    } else {
        (biased - 1023) as f64
    }
}

/// `x 2^n` (the C name; same as [`ldexp`](crate::functions::ldexp)).
#[inline]
pub fn scalbn<T: Real>(x: T, n: i32) -> T {
    crate::functions::ldexp(x, n)
}

/// `x != 0 && y != 0` on primal values. Stan returns `1` or `0`; use `i32::from` for that.
#[inline]
pub fn logical_and<X: Primal, Y: Primal>(x: X, y: Y) -> bool {
    x.primal() != 0.0 && y.primal() != 0.0
}

/// `x != 0 || y != 0` on primal values.
#[inline]
pub fn logical_or<X: Primal, Y: Primal>(x: X, y: Y) -> bool {
    x.primal() != 0.0 || y.primal() != 0.0
}

/// `x == 0` on the primal value.
#[inline]
pub fn logical_negation<X: Primal>(x: X) -> bool {
    x.primal() == 0.0
}

/// `x == y` on primal values.
#[inline]
pub fn logical_eq<X: Primal, Y: Primal>(x: X, y: Y) -> bool {
    x.primal() == y.primal()
}

/// `x != y` on primal values.
#[inline]
pub fn logical_neq<X: Primal, Y: Primal>(x: X, y: Y) -> bool {
    x.primal() != y.primal()
}

/// `x < y` on primal values.
#[inline]
pub fn logical_lt<X: Primal, Y: Primal>(x: X, y: Y) -> bool {
    x.primal() < y.primal()
}

/// `x <= y` on primal values.
#[inline]
pub fn logical_lte<X: Primal, Y: Primal>(x: X, y: Y) -> bool {
    x.primal() <= y.primal()
}

/// `x > y` on primal values.
#[inline]
pub fn logical_gt<X: Primal, Y: Primal>(x: X, y: Y) -> bool {
    x.primal() > y.primal()
}

/// `x >= y` on primal values.
#[inline]
pub fn logical_gte<X: Primal, Y: Primal>(x: X, y: Y) -> bool {
    x.primal() >= y.primal()
}

/// `y_true` when `condition` holds, otherwise `y_false`. Only the selected value carries
/// derivatives.
#[inline]
pub fn if_else<T>(condition: bool, y_true: T, y_false: T) -> T {
    if condition { y_true } else { y_false }
}

/// Stan's name for [`if_else`] in its ternary form.
#[inline]
pub fn select<T>(condition: bool, y_true: T, y_false: T) -> T {
    if_else(condition, y_true, y_false)
}

/// Unary minus, `-x`.
#[inline]
pub fn minus<T: core::ops::Neg<Output = T>>(x: T) -> T {
    -x
}

/// Unary plus: returns `x` unchanged.
#[inline]
pub fn plus<T>(x: T) -> T {
    x
}

/// Real division `x / y`.
#[inline]
pub fn divide<T: Real>(x: T, y: T) -> T {
    x / y
}

/// Integer division truncated toward zero, as in Stan's `int / int`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `y` is zero or the quotient overflows (`i64::MIN / -1`).
pub fn divide_int(x: i64, y: i64) -> Result<i64> {
    if y == 0 {
        return Err(Error::parameter("divide", "y", 0.0, "nonzero"));
    }
    x.checked_div(y)
        .ok_or_else(|| Error::parameter("divide", "x", x as f64, "a quotient that fits in i64"))
}

/// Integer remainder `x % y` with the sign of `x`, as in Stan's `int % int`.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `y` is zero.
pub fn modulus(x: i64, y: i64) -> Result<i64> {
    if y == 0 {
        return Err(Error::parameter("modulus", "y", 0.0, "nonzero"));
    }
    // `i64::MIN % -1` is 0 mathematically; `wrapping_rem` returns that instead of panicking.
    Ok(x.wrapping_rem(y))
}

/// Largest integer not greater than `x`. The derivative is zero.
pub fn floor<T: Real>(x: T) -> T {
    x.floor()
}

/// Smallest integer not less than `x`. The derivative is zero.
pub fn ceil<T: Real>(x: T) -> T {
    x.ceil()
}

/// Nearest integer to `x`, with halves rounded away from zero. The derivative is zero.
pub fn round<T: Real>(x: T) -> T {
    x.round()
}

/// Integer part of `x`, rounded toward zero. The derivative is zero.
pub fn trunc<T: Real>(x: T) -> T {
    x.trunc()
}

/// Euler's number, Stan's nullary `e()`.
pub fn e() -> f64 {
    core::f64::consts::E
}

/// Stan's nullary `pi()`.
pub fn pi() -> f64 {
    core::f64::consts::PI
}

/// Stan's nullary `sqrt2()`.
pub fn sqrt2() -> f64 {
    core::f64::consts::SQRT_2
}

/// Stan's `machine_precision()`: the gap between 1 and the next larger `f64`.
pub fn machine_precision() -> f64 {
    f64::EPSILON
}

/// Stan's `positive_infinity()`.
pub fn positive_infinity() -> f64 {
    f64::INFINITY
}

/// Stan's `negative_infinity()`.
pub fn negative_infinity() -> f64 {
    f64::NEG_INFINITY
}

/// Stan's `not_a_number()`.
pub fn not_a_number() -> f64 {
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logb_reads_the_exponent() {
        assert_eq!(logb(1.0), 0.0);
        assert_eq!(logb(-8.0), 3.0);
        assert_eq!(logb(9_007_199_254_740_991.0), 52.0); // 2^53 - 1
        assert_eq!(logb(f64::MIN_POSITIVE), -1022.0);
        assert_eq!(logb(5e-324), -1074.0);
        assert_eq!(logb(0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn integer_ops_follow_c_semantics() {
        assert_eq!(divide_int(-7, 2), Ok(-3));
        assert_eq!(modulus(-7, 2), Ok(-1));
        assert_eq!(modulus(i64::MIN, -1), Ok(0));
        assert!(divide_int(1, 0).is_err());
        assert!(divide_int(i64::MIN, -1).is_err());
    }

    #[test]
    fn signed_log_sum_matches_direct_sum() {
        let (l, s) = log_sum_exp_signed(2.0, 1, 3.0, -1);
        assert_eq!(s, -1);
        assert!((l - (3f64.exp() - 2f64.exp()).ln()).abs() < 1e-14);
        assert!((hypergeometric_1F0(2.5, 0.3).unwrap() - 0.7f64.powf(-2.5)).abs() < 1e-14);
        assert!(hypergeometric_1F0(1.0, 1.0).is_err());
        assert!(logical_and(1, 2.0) && !logical_or(0, 0.0) && logical_lt(1i64, 1.5));
    }
}
