//! Forward-mode automatic differentiation with dual numbers.

use core::cmp::Ordering;
use core::fmt;
use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::real::{BinaryFn, NaryFn, Real, UnaryFn};

/// A dual number `re + eps * ε` with `ε^2 = 0`.
///
/// Evaluating a function on `Dual { re: x, eps: v }` yields `Dual { re: f(x), eps: f'(x) v }`, the
/// value and the directional derivative along `v`. The components may themselves be any
/// [`Real`], so `Dual<Dual<f64>>` computes second derivatives and `Dual<Var>` computes Hessian
/// vector products by forward over reverse differentiation.
#[derive(Clone, Copy, Debug)]
pub struct Dual<T> {
    /// Primal part.
    pub re: T,
    /// Tangent part.
    pub eps: T,
}

/// First-order dual number over `f64`.
pub type Dual64 = Dual<f64>;

impl<T: Real> Dual<T> {
    /// Builds a dual number from its parts.
    #[inline]
    pub fn new(re: T, eps: T) -> Self {
        Dual { re, eps }
    }

    /// A variable with unit tangent: differentiate with respect to this value.
    #[inline]
    pub fn variable(re: T) -> Self {
        Dual { re, eps: T::one() }
    }

    /// A constant with zero tangent.
    #[inline]
    pub fn constant(re: T) -> Self {
        Dual { re, eps: T::zero() }
    }
}

impl<T: Real> fmt::Display for Dual<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}ε", self.re, self.eps)
    }
}

impl<T: Real> PartialEq for Dual<T> {
    fn eq(&self, other: &Self) -> bool {
        self.re == other.re
    }
}

impl<T: Real> PartialOrd for Dual<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.re.partial_cmp(&other.re)
    }
}

impl<T: Real> PartialEq<f64> for Dual<T> {
    fn eq(&self, other: &f64) -> bool {
        self.re == *other
    }
}

impl<T: Real> PartialOrd<f64> for Dual<T> {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.re.partial_cmp(other)
    }
}

/// `a * b` skipping structurally zero tangents so that constants never inject NaN or tape nodes.
#[inline]
fn tangent_product<T: Real>(eps: T, factor: T) -> Option<T> {
    if eps.is_const_zero() {
        None
    } else {
        Some(eps * factor)
    }
}

#[inline]
fn add_opt<T: Real>(a: Option<T>, b: Option<T>) -> T {
    match (a, b) {
        (Some(a), Some(b)) => a + b,
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => T::zero(),
    }
}

impl<T: Real> Add for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Dual {
            re: self.re + o.re,
            eps: self.eps + o.eps,
        }
    }
}

impl<T: Real> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Dual {
            re: self.re - o.re,
            eps: self.eps - o.eps,
        }
    }
}

impl<T: Real> Mul for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Dual {
            re: self.re * o.re,
            eps: add_opt(
                tangent_product(self.eps, o.re),
                tangent_product(o.eps, self.re),
            ),
        }
    }
}

impl<T: Real> Div for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        let q = self.re / o.re;
        let eps = if o.eps.is_const_zero() {
            self.eps / o.re
        } else {
            (self.eps - q * o.eps) / o.re
        };
        Dual { re: q, eps }
    }
}

impl<T: Real> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Dual {
            re: -self.re,
            eps: -self.eps,
        }
    }
}

impl<T: Real> Add<f64> for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, c: f64) -> Self {
        Dual {
            re: self.re + c,
            eps: self.eps,
        }
    }
}

impl<T: Real> Sub<f64> for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, c: f64) -> Self {
        Dual {
            re: self.re - c,
            eps: self.eps,
        }
    }
}

impl<T: Real> Mul<f64> for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, c: f64) -> Self {
        Dual {
            re: self.re * c,
            eps: self.eps * c,
        }
    }
}

impl<T: Real> Div<f64> for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, c: f64) -> Self {
        Dual {
            re: self.re / c,
            eps: self.eps / c,
        }
    }
}

impl<T: Real> Add<Dual<T>> for f64 {
    type Output = Dual<T>;
    #[inline]
    fn add(self, x: Dual<T>) -> Dual<T> {
        x + self
    }
}

impl<T: Real> Sub<Dual<T>> for f64 {
    type Output = Dual<T>;
    #[inline]
    fn sub(self, x: Dual<T>) -> Dual<T> {
        -x + self
    }
}

impl<T: Real> Mul<Dual<T>> for f64 {
    type Output = Dual<T>;
    #[inline]
    fn mul(self, x: Dual<T>) -> Dual<T> {
        x * self
    }
}

impl<T: Real> Div<Dual<T>> for f64 {
    type Output = Dual<T>;
    #[inline]
    fn div(self, x: Dual<T>) -> Dual<T> {
        Dual::constant(T::from_f64(self)) / x
    }
}

macro_rules! assign_ops {
    ($($trait:ident $method:ident $op:tt),*) => {
        $(
            impl<T: Real> $trait for Dual<T> {
                #[inline]
                fn $method(&mut self, o: Self) {
                    *self = *self $op o;
                }
            }

            impl<T: Real> $trait<f64> for Dual<T> {
                #[inline]
                fn $method(&mut self, o: f64) {
                    *self = *self $op o;
                }
            }
        )*
    };
}

assign_ops!(AddAssign add_assign +, SubAssign sub_assign -, MulAssign mul_assign *, DivAssign div_assign /);

impl<T: Real> Sum for Dual<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Dual::constant(T::zero()), |a, b| a + b)
    }
}

impl<T: Real> Real for Dual<T> {
    const ORDER: usize = T::ORDER + 1;

    #[inline]
    fn from_f64(x: f64) -> Self {
        Dual::constant(T::from_f64(x))
    }

    #[inline]
    fn value(&self) -> f64 {
        self.re.value()
    }

    #[inline]
    fn magnitude(&self) -> f64 {
        self.re.magnitude().max(self.eps.magnitude())
    }

    #[inline]
    fn is_const_zero(&self) -> bool {
        self.re.is_const_zero() && self.eps.is_const_zero()
    }

    #[inline]
    fn map_unary<F: UnaryFn>(self, f: &F) -> Self {
        let y = self.re.map_unary(f);
        let eps = if self.eps.is_const_zero() {
            T::zero()
        } else {
            self.eps * f.derivative(self.re, y)
        };
        Dual { re: y, eps }
    }

    #[inline]
    fn map_binary<F: BinaryFn>(self, other: Self, f: &F) -> Self {
        let y = self.re.map_binary(other.re, f);
        let da = if self.eps.is_const_zero() {
            None
        } else {
            Some(self.eps * f.partial_a(self.re, other.re, y))
        };
        let db = if other.eps.is_const_zero() {
            None
        } else {
            Some(other.eps * f.partial_b(self.re, other.re, y))
        };
        Dual {
            re: y,
            eps: add_opt(da, db),
        }
    }

    fn map_nary<F: NaryFn>(xs: &[Self], f: &F) -> Self {
        let re: Vec<T> = xs.iter().map(|x| x.re).collect();
        let y = T::map_nary(&re, f);
        if xs.iter().all(|x| x.eps.is_const_zero()) {
            return Dual::constant(y);
        }
        let needed: Vec<bool> = xs.iter().map(|x| !x.eps.is_const_zero()).collect();
        let mut partials = vec![T::zero(); xs.len()];
        f.partials(&re, y, &needed, &mut partials);
        let mut eps: Option<T> = None;
        for (x, &p) in xs.iter().zip(&partials) {
            if let Some(term) = tangent_product(x.eps, p) {
                eps = Some(match eps {
                    Some(e) => e + term,
                    None => term,
                });
            }
        }
        Dual {
            re: y,
            eps: eps.unwrap_or_else(T::zero),
        }
    }
}
