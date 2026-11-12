//! Primitive functions and their derivative rules.
//!
//! Each primitive is a small struct implementing [`UnaryFn`], [`BinaryFn`], or [`NaryFn`]. The
//! derivative rules are written against [`Real`], so they are themselves differentiable and nested
//! forward mode produces exact higher derivatives. The public entry points are the methods on
//! [`Real`] and the free functions in [`crate::functions`]; these structs are public so that
//! downstream crates can reuse them and so that users can see the rule behind every function.

use core::f64::consts::{LN_2, LN_10};

use stanmath_rs_core::special as sp;

use crate::real::{BinaryFn, NaryFn, Real, UnaryFn};

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
    /// `|x|`; derivative `sign(x)`, taken to be zero at zero.
    Abs,
    |x| x.abs(),
    |x, y| {
        let v = x.value();
        S::from_f64(if v > 0.0 {
            1.0
        } else if v < 0.0 {
            -1.0
        } else if v == 0.0 {
            0.0
        } else {
            f64::NAN
        })
    }
);
unary!(
    /// `sqrt(x)`; derivative `1 / (2 sqrt(x))`.
    Sqrt,
    |x| x.sqrt(),
    |x, y| (y * 2.0).recip()
);
unary!(
    /// `cbrt(x)`; derivative `1 / (3 cbrt(x)^2)`.
    Cbrt,
    |x| x.cbrt(),
    |x, y| (y.square() * 3.0).recip()
);
unary!(
    /// `x^2`; derivative `2x`.
    Square,
    |x| x * x,
    |x, y| x * 2.0
);
unary!(
    /// `1 / x`; derivative `-1 / x^2`.
    Recip,
    |x| x.recip(),
    |x, y| -y.square()
);
unary!(
    /// `exp(x)`; derivative `exp(x)`.
    Exp,
    |x| x.exp(),
    |x, y| y
);
unary!(
    /// `2^x`; derivative `2^x ln 2`.
    Exp2,
    |x| x.exp2(),
    |x, y| y * LN_2
);
unary!(
    /// `exp(x) - 1`; derivative `exp(x)` (not `y + 1`, which cancels for negative `x`).
    ExpM1,
    |x| x.exp_m1(),
    |x, y| x.exp()
);
unary!(
    /// `ln(x)`; derivative `1 / x`.
    Ln,
    |x| x.ln(),
    |x, y| x.recip()
);
unary!(
    /// `ln(1 + x)`; derivative `1 / (1 + x)`.
    Ln1p,
    |x| x.ln_1p(),
    |x, y| (x + 1.0).recip()
);
unary!(
    /// `log2(x)`; derivative `1 / (x ln 2)`.
    Log2,
    |x| x.log2(),
    |x, y| (x * LN_2).recip()
);
unary!(
    /// `log10(x)`; derivative `1 / (x ln 10)`.
    Log10,
    |x| x.log10(),
    |x, y| (x * LN_10).recip()
);
unary!(
    /// `sin(x)`; derivative `cos(x)`.
    Sin,
    |x| x.sin(),
    |x, y| x.cos()
);
unary!(
    /// `cos(x)`; derivative `-sin(x)`.
    Cos,
    |x| x.cos(),
    |x, y| -x.sin()
);
unary!(
    /// `tan(x)`; derivative `1 + tan(x)^2`.
    Tan,
    |x| x.tan(),
    |x, y| y.square() + 1.0
);
unary!(
    /// `asin(x)`; derivative `1 / sqrt((1 - x)(1 + x))`.
    Asin,
    |x| x.asin(),
    |x, y| x.map_unary(&InvSqrtOneMinusSquare)
);
unary!(
    /// `acos(x)`; derivative `-1 / sqrt((1 - x)(1 + x))`.
    Acos,
    |x| x.acos(),
    |x, y| -x.map_unary(&InvSqrtOneMinusSquare)
);
unary!(
    /// `atan(x)`; derivative `1 / (1 + x^2)`.
    Atan,
    |x| x.atan(),
    |x, y| (x.square() + 1.0).recip()
);
unary!(
    /// `sinh(x)`; derivative `cosh(x)`.
    Sinh,
    |x| x.sinh(),
    |x, y| x.cosh()
);
unary!(
    /// `cosh(x)`; derivative `sinh(x)`.
    Cosh,
    |x| x.cosh(),
    |x, y| x.sinh()
);
unary!(
    /// `tanh(x)`; derivative `1 / cosh(x)^2` (not `1 - tanh(x)^2`, which cancels for large `|x|`).
    Tanh,
    |x| x.tanh(),
    |x, y| x.cosh().square().recip()
);
unary!(
    /// `asinh(x)`; derivative `1 / hypot(x, 1)`.
    Asinh,
    |x| x.asinh(),
    |x, y| x.hypot(S::one()).recip()
);
unary!(
    /// `acosh(x)`; derivative `1 / sqrt((x - 1)(x + 1))`.
    Acosh,
    |x| x.acosh(),
    |x, y| ((x - 1.0) * (x + 1.0)).sqrt().recip()
);
unary!(
    /// `atanh(x)`; derivative `1 / ((1 - x)(1 + x))`.
    Atanh,
    |x| x.atanh(),
    |x, y| x.map_unary(&InvOneMinusSquare)
);
unary!(
    /// `1 / ((1 - x)(1 + x))`, the derivative of `atanh`; derivative `2 x y^2`.
    ///
    /// A primitive of its own so that its derivative is computed without the cancellation in
    /// `d/dx (1 - x)(1 + x) = (1 - x) - (1 + x)` near zero.
    InvOneMinusSquare,
    |x| ((1.0 - x) * (1.0 + x)).recip(),
    |x, y| x * y.square() * 2.0
);
unary!(
    /// `1 / sqrt((1 - x)(1 + x))`, the derivative of `asin`; derivative `x y^3`.
    InvSqrtOneMinusSquare,
    |x| ((1.0 - x) * (1.0 + x)).sqrt().recip(),
    |x, y| x * y.powi(3)
);
unary!(
    /// `ln(1 + exp(x))` (softplus); derivative `inv_logit(x)`.
    Log1pExp,
    |x| sp::log1p_exp(x),
    |x, y| x.map_unary(&InvLogit)
);
unary!(
    /// `ln(1 - exp(x))` for `x < 0`; derivative `-1 / expm1(-x)`.
    Log1mExp,
    |x| sp::log1m_exp(x),
    |x, y| -(-x).exp_m1().recip()
);
unary!(
    /// `ln(1 - x)`; derivative `1 / (x - 1)`.
    Log1m,
    |x| sp::log1m(x),
    |x, y| (x - 1.0).recip()
);
unary!(
    /// Logistic sigmoid; derivative `inv_logit(x) inv_logit(-x)`.
    InvLogit,
    |x| sp::inv_logit(x),
    |x, y| y * (-x).map_unary(&InvLogit)
);
unary!(
    /// `ln(p / (1 - p))`; derivative `1 / (p (1 - p))`.
    Logit,
    |x| sp::logit(x),
    |x, y| (x * (-x + 1.0)).recip()
);
unary!(
    /// `ln(inv_logit(x))`; derivative `inv_logit(-x)`.
    LogInvLogit,
    |x| sp::log_inv_logit(x),
    |x, y| (-x).map_unary(&InvLogit)
);
unary!(
    /// `ln(1 - inv_logit(x))`; derivative `-inv_logit(x)`.
    Log1mInvLogit,
    |x| sp::log1m_inv_logit(x),
    |x, y| -x.map_unary(&InvLogit)
);

/// `x^n` for a fixed integer `n`; derivative `n x^(n - 1)`.
#[derive(Clone, Copy, Debug)]
pub struct Powi(pub i32);

impl UnaryFn for Powi {
    #[inline]
    fn value(&self, x: f64) -> f64 {
        x.powi(self.0)
    }

    #[inline]
    fn derivative<S: Real>(&self, x: S, _y: S) -> S {
        match self.0 {
            0 => S::zero(),
            1 => S::one(),
            n => x.powi(n - 1) * f64::from(n),
        }
    }
}

/// `x^c` for a fixed real `c`; derivative `c x^(c - 1)`.
#[derive(Clone, Copy, Debug)]
pub struct Powf(pub f64);

impl UnaryFn for Powf {
    #[inline]
    fn value(&self, x: f64) -> f64 {
        x.powf(self.0)
    }

    #[inline]
    fn derivative<S: Real>(&self, x: S, _y: S) -> S {
        if self.0 == 0.0 {
            S::zero()
        } else if self.0 == 1.0 {
            S::one()
        } else {
            x.powf(self.0 - 1.0) * self.0
        }
    }
}

macro_rules! binary {
    ($(#[$doc:meta])* $name:ident, |$a:ident, $b:ident| $value:expr,
     |$pa:ident, $pb:ident, $py:ident| $da:expr, $db:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl BinaryFn for $name {
            #[inline]
            fn value(&self, $a: f64, $b: f64) -> f64 {
                $value
            }

            #[inline]
            #[allow(unused_variables)]
            fn partial_a<S: Real>(&self, $pa: S, $pb: S, $py: S) -> S {
                $da
            }

            #[inline]
            #[allow(unused_variables)]
            fn partial_b<S: Real>(&self, $pa: S, $pb: S, $py: S) -> S {
                $db
            }
        }
    };
}

binary!(
    /// `a^b`; partials `b a^(b - 1)` and `ln(a) a^b`, the latter taken to be zero at `a = 0`.
    Pow,
    |a, b| a.powf(b),
    |a, b, y| b * a.pow(b - 1.0),
    if a.value() == 0.0 { S::zero() } else { a.ln() * y }
);
binary!(
    /// `atan2(a, b)`; partials `b / (a^2 + b^2)` and `-a / (a^2 + b^2)`.
    Atan2,
    |a, b| a.atan2(b),
    |a, b, y| b / (a.square() + b.square()),
    -a / (a.square() + b.square())
);
binary!(
    /// `hypot(a, b)`; partials `a / hypot` and `b / hypot`, each a [`DivHypot`] primitive.
    Hypot,
    |a, b| a.hypot(b),
    |a, b, y| a.map_binary(b, &DivHypot),
    b.map_binary(a, &DivHypot)
);
binary!(
    /// `a / hypot(a, b)`, the partial of `hypot`; partials `b^2 / h^3` and `-a b / h^3`.
    ///
    /// Differentiating `a / hypot(a, b)` by the quotient rule would form `h^2 - a^2`, which
    /// cancels when `|a|` dominates; these closed forms do not.
    DivHypot,
    |a, b| a / a.hypot(b),
    |a, b, y| b.square() / a.hypot(b).powi(3),
    -(a * b) / a.hypot(b).powi(3)
);
binary!(
    /// `ln(exp(a) + exp(b))`; partials `inv_logit(a - b)` and `inv_logit(b - a)`.
    ///
    /// The logistic form is exact even when `y` has a large magnitude, and its own derivative
    /// avoids the `1 - exp(b - y)` cancellation.
    LogSumExp,
    |a, b| sp::log_sum_exp(a, b),
    |a, b, y| (a - b).map_unary(&InvLogit),
    (b - a).map_unary(&InvLogit)
);
binary!(
    /// `ln(exp(a) - exp(b))`; partials `-1 / expm1(b - a)` and `-1 / expm1(a - b)`.
    LogDiffExp,
    |a, b| sp::log_diff_exp(a, b),
    |a, b, y| -(b - a).exp_m1().recip(),
    -(a - b).exp_m1().recip()
);

/// Sum of a slice; every partial is one.
#[derive(Clone, Copy, Debug, Default)]
pub struct SumN;

impl NaryFn for SumN {
    fn value(&self, x: &[f64]) -> f64 {
        x.iter().sum()
    }

    fn partials<S: Real>(&self, _x: &[S], _y: S, _needed: &[bool], out: &mut [S]) {
        out.fill(S::one());
    }
}

/// `ln(sum(exp(x)))` over a slice; partials are the softmax `exp(x[i] - y)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LogSumExpN;

impl NaryFn for LogSumExpN {
    fn value(&self, x: &[f64]) -> f64 {
        sp::log_sum_exp_slice(x)
    }

    fn partials<S: Real>(&self, x: &[S], y: S, _needed: &[bool], out: &mut [S]) {
        for (o, &xi) in out.iter_mut().zip(x) {
            *o = (xi - y).exp();
        }
    }
}

/// Dot product of the two halves of a slice: `sum(x[i] * x[n + i])`.
#[derive(Clone, Copy, Debug, Default)]
pub struct DotHalves;

impl NaryFn for DotHalves {
    fn value(&self, x: &[f64]) -> f64 {
        let n = x.len() / 2;
        x[..n].iter().zip(&x[n..]).map(|(a, b)| a * b).sum()
    }

    fn partials<S: Real>(&self, x: &[S], _y: S, _needed: &[bool], out: &mut [S]) {
        let n = x.len() / 2;
        out[..n].copy_from_slice(&x[n..]);
        out[n..].copy_from_slice(&x[..n]);
    }
}
