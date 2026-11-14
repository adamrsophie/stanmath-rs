//! Complex numbers over any [`Real`] scalar.
//!
//! [`Complex<T>`] stores a real and an imaginary part of the same differentiable type, so complex
//! arithmetic carries derivatives in every mode. Stan's complex functions are methods here
//! (`z.exp()`, `z.abs()`), plus free functions with Stan's names wherever those names do not
//! collide with the real functions (`get_real`, `conj`, `polar`, and `complex_abs` for Stan's
//! complex `abs`).
//!
//! Elementary functions use principal branches with the branch cuts of DLMF chapter 4 and C99
//! Annex G. Formulas are the textbook ones: they lose accuracy near branch points and overflow for
//! imaginary parts beyond about 355 in `tan` and `tanh`.

use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::real::Real;

/// A complex number `re + i im` with parts of type `T`.
///
/// Comparisons (`==`) compare both parts with the comparison of `T`, which looks at primal values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex<T> {
    /// Real part.
    pub re: T,
    /// Imaginary part.
    pub im: T,
}

impl<T: Real> Complex<T> {
    /// `re + i im`.
    #[inline]
    pub fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }

    /// A complex number with zero imaginary part.
    #[inline]
    pub fn from_real(re: T) -> Self {
        Complex { re, im: T::zero() }
    }

    /// A constant complex number.
    #[inline]
    pub fn from_f64(re: f64, im: f64) -> Self {
        Complex {
            re: T::from_f64(re),
            im: T::from_f64(im),
        }
    }

    /// The constant zero.
    #[inline]
    pub fn zero() -> Self {
        Self::from_f64(0.0, 0.0)
    }

    /// The constant one.
    #[inline]
    pub fn one() -> Self {
        Self::from_f64(1.0, 0.0)
    }

    /// The imaginary unit.
    #[inline]
    pub fn i() -> Self {
        Self::from_f64(0.0, 1.0)
    }

    /// Primal values of both parts.
    #[inline]
    pub fn value(&self) -> Complex<f64> {
        Complex {
            re: self.re.value(),
            im: self.im.value(),
        }
    }

    /// Multiplies both parts by a constant.
    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Complex {
            re: self.re * s,
            im: self.im * s,
        }
    }

    /// Complex conjugate `re - i im`.
    #[inline]
    pub fn conj(self) -> Self {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    /// Magnitude `|z|`, computed without undue overflow.
    #[inline]
    pub fn abs(self) -> T {
        self.re.hypot(self.im)
    }

    /// Phase angle in `(-pi, pi]`.
    #[inline]
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// Squared magnitude `re^2 + im^2` (Stan's `norm`).
    #[inline]
    pub fn norm(self) -> T {
        self.re.square() + self.im.square()
    }

    /// Projection onto the Riemann sphere: every infinity maps to `(inf, +/-0)`, other values are
    /// unchanged.
    pub fn proj(self) -> Self {
        if self.re.value().is_infinite() || self.im.value().is_infinite() {
            let im = if self.im.value().is_sign_negative() {
                -0.0
            } else {
                0.0
            };
            Self::from_f64(f64::INFINITY, im)
        } else {
            self
        }
    }

    /// `1 / z`.
    #[inline]
    pub fn recip(self) -> Self {
        Self::one() / self
    }

    /// Exponential.
    pub fn exp(self) -> Self {
        let e = self.re.exp();
        Complex {
            re: e * self.im.cos(),
            im: e * self.im.sin(),
        }
    }

    /// Natural logarithm, principal branch, cut along the negative real axis.
    pub fn ln(self) -> Self {
        Complex {
            re: self.abs().ln(),
            im: self.arg(),
        }
    }

    /// Natural logarithm (Stan's name).
    #[inline]
    pub fn log(self) -> Self {
        self.ln()
    }

    /// Base 10 logarithm.
    #[inline]
    pub fn log10(self) -> Self {
        self.ln().scale(core::f64::consts::LOG10_E)
    }

    /// Principal square root, with nonnegative real part.
    pub fn sqrt(self) -> Self {
        let r = self.abs();
        if r.value() == 0.0 {
            return Self::zero();
        }
        if self.re.value() >= 0.0 {
            let t = ((self.re + r) * 0.5).sqrt();
            Complex {
                re: t,
                im: self.im / (t * 2.0),
            }
        } else {
            let t = ((r - self.re) * 0.5).sqrt();
            let negative = self.im.value().is_sign_negative();
            let im_abs = if negative { -self.im } else { self.im };
            Complex {
                re: im_abs / (t * 2.0),
                im: if negative { -t } else { t },
            }
        }
    }

    /// `z^w` on the principal branch, `exp(w log z)`. `0^w` is one when `w` is zero and zero
    /// otherwise.
    pub fn pow(self, w: Self) -> Self {
        if self.re.value() == 0.0 && self.im.value() == 0.0 {
            return if w.re.value() == 0.0 && w.im.value() == 0.0 {
                Self::one()
            } else {
                Self::zero()
            };
        }
        (w * self.ln()).exp()
    }

    /// `z^e` for a real exponent.
    #[inline]
    pub fn powr(self, e: T) -> Self {
        self.pow(Self::from_real(e))
    }

    /// Sine.
    pub fn sin(self) -> Self {
        Complex {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    /// Cosine.
    pub fn cos(self) -> Self {
        Complex {
            re: self.re.cos() * self.im.cosh(),
            im: -(self.re.sin() * self.im.sinh()),
        }
    }

    /// Tangent.
    pub fn tan(self) -> Self {
        let (a, b) = (self.re * 2.0, self.im * 2.0);
        let den = a.cos() + b.cosh();
        Complex {
            re: a.sin() / den,
            im: b.sinh() / den,
        }
    }

    /// Hyperbolic sine.
    pub fn sinh(self) -> Self {
        Complex {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }

    /// Hyperbolic cosine.
    pub fn cosh(self) -> Self {
        Complex {
            re: self.re.cosh() * self.im.cos(),
            im: self.re.sinh() * self.im.sin(),
        }
    }

    /// Hyperbolic tangent.
    pub fn tanh(self) -> Self {
        let (a, b) = (self.re * 2.0, self.im * 2.0);
        let den = a.cosh() + b.cos();
        Complex {
            re: a.sinh() / den,
            im: b.sin() / den,
        }
    }

    /// Inverse hyperbolic sine, `log(z + sqrt(z^2 + 1))`, cuts on the imaginary axis beyond `+/-i`.
    pub fn asinh(self) -> Self {
        if self.re.value() < 0.0 {
            return -(-self).asinh();
        }
        (self + (self * self + T::one()).sqrt()).ln()
    }

    /// Inverse hyperbolic cosine, `log(z + sqrt(z + 1) sqrt(z - 1))`, cut along real `x < 1`.
    pub fn acosh(self) -> Self {
        (self + (self + T::one()).sqrt() * (self - T::one()).sqrt()).ln()
    }

    /// Inverse hyperbolic tangent, `(log(1 + z) - log(1 - z)) / 2`, cuts along real `|x| > 1`.
    /// Adding the real one keeps the sign of a zero imaginary part, which picks the side of the
    /// cut (`1 - z` built as `0 - z` would turn `-0` into `+0`).
    pub fn atanh(self) -> Self {
        ((self + T::one()).ln() - ((-self) + T::one()).ln()).scale(0.5)
    }

    /// Inverse sine, `-i asinh(i z)`.
    pub fn asin(self) -> Self {
        let w = Complex {
            re: -self.im,
            im: self.re,
        }
        .asinh();
        Complex {
            re: w.im,
            im: -w.re,
        }
    }

    /// Inverse cosine, `pi / 2 - asin(z)`.
    pub fn acos(self) -> Self {
        let w = self.asin();
        Complex {
            re: -w.re + core::f64::consts::FRAC_PI_2,
            im: -w.im,
        }
    }

    /// Inverse tangent, `-i atanh(i z)`.
    pub fn atan(self) -> Self {
        let w = Complex {
            re: -self.im,
            im: self.re,
        }
        .atanh();
        Complex {
            re: w.im,
            im: -w.re,
        }
    }
}

impl<T: Real> Add for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Complex {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

impl<T: Real> Sub for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Complex {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

impl<T: Real> Mul for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Complex {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

impl<T: Real> Div for Complex<T> {
    type Output = Self;
    /// Smith's algorithm, which avoids overflow in the intermediate squared magnitude.
    fn div(self, o: Self) -> Self {
        if o.re.value().abs() >= o.im.value().abs() {
            let r = o.im / o.re;
            let den = o.re + o.im * r;
            Complex {
                re: (self.re + self.im * r) / den,
                im: (self.im - self.re * r) / den,
            }
        } else {
            let r = o.re / o.im;
            let den = o.re * r + o.im;
            Complex {
                re: (self.re * r + self.im) / den,
                im: (self.im * r - self.re) / den,
            }
        }
    }
}

impl<T: Real> Neg for Complex<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Complex {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: Real> Add<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn add(self, o: T) -> Self {
        Complex {
            re: self.re + o,
            im: self.im,
        }
    }
}

impl<T: Real> Sub<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn sub(self, o: T) -> Self {
        Complex {
            re: self.re - o,
            im: self.im,
        }
    }
}

impl<T: Real> Mul<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn mul(self, o: T) -> Self {
        Complex {
            re: self.re * o,
            im: self.im * o,
        }
    }
}

impl<T: Real> Div<T> for Complex<T> {
    type Output = Self;
    #[inline]
    fn div(self, o: T) -> Self {
        Complex {
            re: self.re / o,
            im: self.im / o,
        }
    }
}

impl<T: Real> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl<T: Real> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl<T: Real> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}

impl<T: Real> DivAssign for Complex<T> {
    #[inline]
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}

impl<T: Real> Sum for Complex<T> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}

/// `re + i im` (Stan's `to_complex`).
#[inline]
pub fn to_complex<T: Real>(re: T, im: T) -> Complex<T> {
    Complex::new(re, im)
}

/// Real part.
#[inline]
pub fn get_real<T: Real>(z: Complex<T>) -> T {
    z.re
}

/// Imaginary part.
#[inline]
pub fn get_imag<T: Real>(z: Complex<T>) -> T {
    z.im
}

/// Complex conjugate.
#[inline]
pub fn conj<T: Real>(z: Complex<T>) -> Complex<T> {
    z.conj()
}

/// Projection onto the Riemann sphere; see [`Complex::proj`].
#[inline]
pub fn proj<T: Real>(z: Complex<T>) -> Complex<T> {
    z.proj()
}

/// The complex number with magnitude `r` and phase angle `theta`.
#[inline]
pub fn polar<T: Real>(r: T, theta: T) -> Complex<T> {
    Complex::new(r * theta.cos(), r * theta.sin())
}

/// Magnitude of a complex number (Stan's `abs` on complex arguments).
#[inline]
pub fn complex_abs<T: Real>(z: Complex<T>) -> T {
    z.abs()
}

/// Phase angle of a complex number (Stan's `arg`).
#[inline]
pub fn complex_arg<T: Real>(z: Complex<T>) -> T {
    z.arg()
}

/// Squared magnitude of a complex number (Stan's `norm`).
#[inline]
pub fn complex_norm<T: Real>(z: Complex<T>) -> T {
    z.norm()
}

/// Phase angle of a complex number, under Stan's name; same as [`complex_arg`].
#[inline]
pub fn arg<T: Real>(z: Complex<T>) -> T {
    z.arg()
}

/// Squared magnitude of a complex number, under Stan's name; same as [`complex_norm`].
#[inline]
pub fn norm<T: Real>(z: Complex<T>) -> T {
    z.norm()
}
