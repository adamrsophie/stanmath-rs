//! The [`Real`] scalar trait and the primitive function interfaces.

use core::fmt::{Debug, Display};
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::primitives as p;
use crate::vectorize::{Arg, Lift, Promote};

/// A scalar function of one argument with a derivative rule.
///
/// `value` evaluates the function in `f64`. `derivative` returns `f'(x)` in any scalar type `S`,
/// given the argument `x` and the already computed result `y = f(x)`. Because the derivative is
/// written against [`Real`] rather than `f64`, nesting forward mode over reverse mode (or over
/// forward mode) differentiates the derivative rule itself, which is what makes higher-order
/// derivatives exact.
///
/// Implement this trait to add a differentiable function, then call [`Real::map_unary`].
pub trait UnaryFn {
    /// `f(x)` in `f64`.
    fn value(&self, x: f64) -> f64;
    /// `f'(x)`, given `x` and `y = f(x)`.
    fn derivative<S: Real>(&self, x: S, y: S) -> S;
}

/// A scalar function of two arguments with partial derivative rules.
///
/// See [`UnaryFn`] for the conventions. The partials are only evaluated for arguments that carry
/// derivative information, so an expensive partial costs nothing when its argument is data.
pub trait BinaryFn {
    /// `f(a, b)` in `f64`.
    fn value(&self, a: f64, b: f64) -> f64;
    /// `df/da`, given `a`, `b`, and `y = f(a, b)`.
    fn partial_a<S: Real>(&self, a: S, b: S, y: S) -> S;
    /// `df/db`, given `a`, `b`, and `y = f(a, b)`.
    fn partial_b<S: Real>(&self, a: S, b: S, y: S) -> S;
}

/// A scalar function of a slice of arguments with a gradient rule.
///
/// This is the hook for fused operations such as `log_sum_exp` over a vector or a whole
/// log density with analytic partials. In reverse mode it records a single tape node with one edge
/// per argument instead of one node per elementary operation.
pub trait NaryFn {
    /// `f(x)` in `f64`.
    fn value(&self, x: &[f64]) -> f64;
    /// Writes `df/dx[i]` into `out[i]` for every `i` with `needed[i]`, given `x` and `y = f(x)`.
    ///
    /// `out` and `needed` have the same length as `x`, and `out` is zero on entry. Entries whose
    /// argument carries no derivative information are not needed and may be left at zero, which
    /// lets expensive partials (for example with respect to the shape of an incomplete beta
    /// function) be skipped when that argument is data.
    fn partials<S: Real>(&self, x: &[S], y: S, needed: &[bool], out: &mut [S]);

    /// Value and `f64` partials in one pass. Reverse mode calls this; override it when the value
    /// and the partials share intermediate results.
    fn value_and_partials(&self, x: &[f64], needed: &[bool], out: &mut [f64]) -> f64 {
        let y = self.value(x);
        self.partials(x, y, needed, out);
        y
    }
}

/// A function from a slice of scalars to a vector of scalars with a structured reverse rule.
///
/// This is the hook for matrix operations. `eval` is one generic algorithm that serves `f64` and
/// every forward-mode type (so nested modes stay exact). Reverse mode evaluates it in `f64` and
/// records a single [`Tape::custom_op`](crate::Tape::custom_op) whose adjoint is `reverse`, so a
/// product of two `n` by `n` matrices adds `n^2` nodes to the tape, not `n^3`.
pub trait MultiFn: Clone + Send + 'static {
    /// Evaluates the function. Errors (for example a matrix that is not positive definite)
    /// propagate to the caller.
    fn eval<S: Real>(&self, x: &[S]) -> stanmath_rs_core::Result<Vec<S>>;

    /// Adds `J^T y_adj` to `x_adj`, given the input values `x` and output values `y`. Entries of
    /// `x_adj` whose `needed` flag is false may be skipped.
    fn reverse(&self, x: &[f64], y: &[f64], y_adj: &[f64], needed: &[bool], x_adj: &mut [f64]);
}

/// A differentiable real scalar.
///
/// Implemented by `f64` (no derivatives), [`Dual`](crate::Dual) (forward mode), and
/// [`Var`](crate::Var) (reverse mode). Forward mode nests: `Dual<Var>` gives forward over reverse
/// second derivatives and `Dual<Dual<f64>>` gives forward over forward. Write model code generically
/// over `T: Real` and the same function serves every mode.
///
/// Comparisons (`==`, `<`, ...) compare primal values only.
///
/// Arithmetic with an `f64` on the right (`x * 2.0`) works in generic code. An `f64` on the left
/// (`2.0 * x`) works for the concrete types but cannot be expressed as a bound on a generic `T`.
pub trait Real:
    Copy
    + Debug
    + Display
    + PartialEq
    + PartialOrd
    + PartialEq<f64>
    + PartialOrd<f64>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + AddAssign<f64>
    + SubAssign<f64>
    + MulAssign<f64>
    + DivAssign<f64>
    + Sum
    + Arg<Elem = Self>
    + Promote<f64, Out = Self>
    + Promote<Self, Out = Self>
    + Lift<Self>
{
    /// Derivative order carried by this type: 0 for `f64`, 1 for `Var` and `Dual<f64>`, one more
    /// for each level of `Dual` nesting.
    ///
    /// Primitives whose derivative rule is itself an iterative algorithm use this to decide how
    /// carefully to converge: when `ORDER > 0` the rule will be differentiated again.
    const ORDER: usize;

    /// A constant: a value with no derivative information.
    fn from_f64(x: f64) -> Self;

    /// The primal `f64` value.
    fn value(&self) -> f64;

    /// Largest absolute value among the primal value and every forward-mode tangent component.
    ///
    /// Iterative algorithms use this in their stopping tests so that they keep iterating until the
    /// derivatives, not only the value, have converged. A continued fraction can terminate exactly
    /// in value (for example at an integer shape parameter) while its derivative still depends on
    /// the tail. Reverse-mode variables report the absolute value of their primal value.
    fn magnitude(&self) -> f64;

    /// True when `self` is exactly zero and carries no derivative information.
    ///
    /// Forward mode uses this to skip tangent terms that are structurally zero, which keeps
    /// non-finite partials (for example the `ln(0)` in `d/db 0^b`) out of the result when the
    /// corresponding argument is a constant.
    fn is_const_zero(&self) -> bool;

    /// Applies a unary primitive.
    fn map_unary<F: UnaryFn>(self, f: &F) -> Self;

    /// Applies a binary primitive.
    fn map_binary<F: BinaryFn>(self, other: Self, f: &F) -> Self;

    /// Applies an n-ary primitive.
    fn map_nary<F: NaryFn>(xs: &[Self], f: &F) -> Self;

    /// Applies a multi-output function such as a matrix operation.
    fn map_multi<F: MultiFn>(xs: &[Self], f: &F) -> stanmath_rs_core::Result<Vec<Self>> {
        f.eval(xs)
    }

    /// The constant zero.
    fn zero() -> Self {
        Self::from_f64(0.0)
    }

    /// The constant one.
    fn one() -> Self {
        Self::from_f64(1.0)
    }

    /// True when the primal value is NaN.
    fn is_nan(&self) -> bool {
        self.value().is_nan()
    }

    /// True when the primal value is finite.
    fn is_finite(&self) -> bool {
        self.value().is_finite()
    }

    /// Absolute value. The derivative at zero is taken to be zero.
    fn abs(self) -> Self {
        self.map_unary(&p::Abs)
    }

    /// Square root.
    fn sqrt(self) -> Self {
        self.map_unary(&p::Sqrt)
    }

    /// Cube root.
    fn cbrt(self) -> Self {
        self.map_unary(&p::Cbrt)
    }

    /// `x * x`.
    fn square(self) -> Self {
        self.map_unary(&p::Square)
    }

    /// `1 / x`.
    fn recip(self) -> Self {
        self.map_unary(&p::Recip)
    }

    /// Exponential.
    fn exp(self) -> Self {
        self.map_unary(&p::Exp)
    }

    /// `2^x`.
    fn exp2(self) -> Self {
        self.map_unary(&p::Exp2)
    }

    /// `exp(x) - 1`, accurate near zero.
    fn exp_m1(self) -> Self {
        self.map_unary(&p::ExpM1)
    }

    /// Natural logarithm.
    fn ln(self) -> Self {
        self.map_unary(&p::Ln)
    }

    /// `ln(1 + x)`, accurate near zero.
    fn ln_1p(self) -> Self {
        self.map_unary(&p::Ln1p)
    }

    /// Base 2 logarithm.
    fn log2(self) -> Self {
        self.map_unary(&p::Log2)
    }

    /// Base 10 logarithm.
    fn log10(self) -> Self {
        self.map_unary(&p::Log10)
    }

    /// Sine.
    fn sin(self) -> Self {
        self.map_unary(&p::Sin)
    }

    /// Cosine.
    fn cos(self) -> Self {
        self.map_unary(&p::Cos)
    }

    /// Tangent.
    fn tan(self) -> Self {
        self.map_unary(&p::Tan)
    }

    /// Arcsine.
    fn asin(self) -> Self {
        self.map_unary(&p::Asin)
    }

    /// Arccosine.
    fn acos(self) -> Self {
        self.map_unary(&p::Acos)
    }

    /// Arctangent.
    fn atan(self) -> Self {
        self.map_unary(&p::Atan)
    }

    /// Hyperbolic sine.
    fn sinh(self) -> Self {
        self.map_unary(&p::Sinh)
    }

    /// Hyperbolic cosine.
    fn cosh(self) -> Self {
        self.map_unary(&p::Cosh)
    }

    /// Hyperbolic tangent.
    fn tanh(self) -> Self {
        self.map_unary(&p::Tanh)
    }

    /// Inverse hyperbolic sine.
    fn asinh(self) -> Self {
        self.map_unary(&p::Asinh)
    }

    /// Inverse hyperbolic cosine.
    fn acosh(self) -> Self {
        self.map_unary(&p::Acosh)
    }

    /// Inverse hyperbolic tangent.
    fn atanh(self) -> Self {
        self.map_unary(&p::Atanh)
    }

    /// `x^n` for an integer exponent.
    fn powi(self, n: i32) -> Self {
        self.map_unary(&p::Powi(n))
    }

    /// `x^c` for a constant exponent.
    fn powf(self, c: f64) -> Self {
        self.map_unary(&p::Powf(c))
    }

    /// `x^e` where both base and exponent may carry derivatives.
    ///
    /// At a zero base the partial with respect to the exponent is taken to be zero.
    fn pow(self, e: Self) -> Self {
        self.map_binary(e, &p::Pow)
    }

    /// Four-quadrant arctangent of `self / x`.
    fn atan2(self, x: Self) -> Self {
        self.map_binary(x, &p::Atan2)
    }

    /// `sqrt(self^2 + other^2)` without undue overflow.
    fn hypot(self, other: Self) -> Self {
        self.map_binary(other, &p::Hypot)
    }

    /// Largest integer not greater than `x`. The derivative is zero.
    fn floor(self) -> Self {
        Self::from_f64(self.value().floor())
    }

    /// Smallest integer not less than `x`. The derivative is zero.
    fn ceil(self) -> Self {
        Self::from_f64(self.value().ceil())
    }

    /// Nearest integer, rounding half away from zero. The derivative is zero.
    fn round(self) -> Self {
        Self::from_f64(self.value().round())
    }

    /// Integer part. The derivative is zero.
    fn trunc(self) -> Self {
        Self::from_f64(self.value().trunc())
    }

    /// The larger of two values; ties return `self`, and a NaN argument returns the other one.
    fn max(self, other: Self) -> Self {
        if other.value() > self.value() || self.is_nan() {
            other
        } else {
            self
        }
    }

    /// The smaller of two values; ties return `self`, and a NaN argument returns the other one.
    fn min(self, other: Self) -> Self {
        if other.value() < self.value() || self.is_nan() {
            other
        } else {
            self
        }
    }
}

macro_rules! f64_methods {
    ($($name:ident),* $(,)?) => {
        $(
            #[inline]
            fn $name(self) -> Self {
                f64::$name(self)
            }
        )*
    };
}

impl Real for f64 {
    const ORDER: usize = 0;

    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }

    #[inline]
    fn value(&self) -> f64 {
        *self
    }

    #[inline]
    fn magnitude(&self) -> f64 {
        self.abs()
    }

    #[inline]
    fn is_const_zero(&self) -> bool {
        *self == 0.0
    }

    #[inline]
    fn map_unary<F: UnaryFn>(self, f: &F) -> Self {
        f.value(self)
    }

    #[inline]
    fn map_binary<F: BinaryFn>(self, other: Self, f: &F) -> Self {
        f.value(self, other)
    }

    #[inline]
    fn map_nary<F: NaryFn>(xs: &[Self], f: &F) -> Self {
        f.value(xs)
    }

    f64_methods!(
        sqrt, cbrt, exp, exp2, exp_m1, ln, ln_1p, log2, log10, sin, cos, tan, asin, acos, atan,
        sinh, cosh, tanh, asinh, acosh, atanh, recip,
    );

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn square(self) -> Self {
        self * self
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        f64::powi(self, n)
    }

    #[inline]
    fn powf(self, c: f64) -> Self {
        f64::powf(self, c)
    }

    #[inline]
    fn pow(self, e: Self) -> Self {
        f64::powf(self, e)
    }

    #[inline]
    fn atan2(self, x: Self) -> Self {
        f64::atan2(self, x)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        f64::hypot(self, other)
    }
}
