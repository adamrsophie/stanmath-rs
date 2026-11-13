//! Stan Math library functions that the Stan language does not expose: gradients of special
//! functions, complex helpers, and predicates. Gradients come from forward-mode passes over the
//! differentiable functions, so they work in any scalar type.

#![allow(non_snake_case)]

use crate::complex::Complex;
use crate::dual::Dual;
use crate::functions::{gamma_p, gamma_q, hypergeometric_2F1, hypergeometric_pFq, inc_beta, log};
use crate::real::Real;

/// The partial derivatives of `f` at `x`, one forward pass per input.
fn partials<T: Real, const N: usize>(x: [T; N], f: impl Fn([Dual<T>; N]) -> Dual<T>) -> [T; N] {
    core::array::from_fn(|k| {
        f(core::array::from_fn(|i| {
            Dual::new(x[i], if i == k { T::one() } else { T::zero() })
        }))
        .eps
    })
}

/// The partial derivatives of a function of a slice, one forward pass per input.
fn slice_partials<T: Real>(x: &[T], f: impl Fn(&[Dual<T>]) -> Dual<T>) -> Vec<T> {
    (0..x.len())
        .map(|k| {
            let d: Vec<Dual<T>> = x
                .iter()
                .enumerate()
                .map(|(i, &v)| Dual::new(v, if i == k { T::one() } else { T::zero() }))
                .collect();
            f(&d).eps
        })
        .collect()
}

/// `i z`.
pub fn i_times<T: Real>(z: Complex<T>) -> Complex<T> {
    Complex::new(-z.im, z.re)
}

/// `-i z`.
pub fn neg_i_times<T: Real>(z: Complex<T>) -> Complex<T> {
    Complex::new(z.im, -z.re)
}

/// Whether `x` equals its floor: true for whole numbers and infinities, false for NaN (checked
/// against Stan Math 5.4.0).
pub fn is_integer(x: f64) -> bool {
    x == x.floor()
}

/// Whether any of `xs` is NaN.
pub fn is_any_nan<T: Real>(xs: &[T]) -> bool {
    xs.iter().any(|x| x.value().is_nan())
}

/// Step size for a central finite difference at `u`: `eps^(1/3) max(1, |u|)`, which balances
/// truncation error (order `h^2`) against rounding error (order `eps / h`).
pub fn finite_diff_stepsize(u: f64) -> f64 {
    f64::EPSILON.cbrt() * u.abs().max(1.0)
}

/// `x += lambda y`, elementwise.
///
/// # Panics
///
/// Panics if `x` and `y` differ in length.
pub fn scaled_add(x: &mut [f64], y: &[f64], lambda: f64) {
    assert_eq!(x.len(), y.len(), "scaled_add: lengths differ");
    for (a, b) in x.iter_mut().zip(y) {
        *a += lambda * b;
    }
}

/// The gradient of `2F1(a1, a2; b1; z)` with respect to `(a1, a2, b1, z)`.
pub fn grad_2F1<T: Real>(a1: T, a2: T, b1: T, z: T) -> [T; 4] {
    partials([a1, a2, b1, z], |[a, b, c, x]| {
        hypergeometric_2F1(a, b, c, x)
    })
}

/// The gradient of `3F2(a1, a2, a3; b1, b2; z)` with respect to `(a1, a2, a3, b1, b2, z)`.
pub fn grad_F32<T: Real>(a: [T; 3], b: [T; 2], z: T) -> [T; 6] {
    partials([a[0], a[1], a[2], b[0], b[1], z], |v| {
        hypergeometric_pFq(&v[..3], &v[3..5], v[5])
    })
}

/// The gradient of `pFq(a; b; z)`: the partials with respect to `a`, to `b`, and to `z`.
pub fn grad_pFq<T: Real>(a: &[T], b: &[T], z: T) -> (Vec<T>, Vec<T>, T) {
    let (p, q) = (a.len(), b.len());
    let x: Vec<T> = a.iter().chain(b).copied().chain([z]).collect();
    let g = slice_partials(&x, |v| hypergeometric_pFq(&v[..p], &v[p..p + q], v[p + q]));
    (g[..p].to_vec(), g[p..p + q].to_vec(), g[p + q])
}

/// `∂I_z(a, b) / ∂a` of the regularized incomplete beta function.
pub fn inc_beta_dda<T: Real>(a: T, b: T, z: T) -> T {
    partials([a, b, z], |[a, b, z]| inc_beta(a, b, z))[0]
}

/// `∂I_z(a, b) / ∂b` of the regularized incomplete beta function.
pub fn inc_beta_ddb<T: Real>(a: T, b: T, z: T) -> T {
    partials([a, b, z], |[a, b, z]| inc_beta(a, b, z))[1]
}

/// `∂I_z(a, b) / ∂z` of the regularized incomplete beta function.
pub fn inc_beta_ddz<T: Real>(a: T, b: T, z: T) -> T {
    partials([a, b, z], |[a, b, z]| inc_beta(a, b, z))[2]
}

/// `(∂I_z(a, b) / ∂a, ∂I_z(a, b) / ∂b)`. Stan Math also takes precomputed digammas and beta;
/// they are not needed here.
pub fn grad_reg_inc_beta<T: Real>(a: T, b: T, z: T) -> (T, T) {
    let g = partials([a, b, z], |[a, b, z]| inc_beta(a, b, z));
    (g[0], g[1])
}

/// `∂Q(a, z) / ∂a` of the regularized upper incomplete gamma function.
pub fn grad_reg_inc_gamma<T: Real>(a: T, z: T) -> T {
    partials([a, z], |[a, z]| gamma_q(a, z))[0]
}

/// `∂P(a, z) / ∂a` of the regularized lower incomplete gamma function.
pub fn grad_reg_lower_inc_gamma<T: Real>(a: T, z: T) -> T {
    partials([a, z], |[a, z]| gamma_p(a, z))[0]
}

/// `(log Q(a, z), ∂ log Q(a, z) / ∂a)` of the regularized upper incomplete gamma function.
pub fn log_gamma_q_dgamma<T: Real>(a: T, z: T) -> (T, T) {
    let y = log(gamma_q(Dual::new(a, T::one()), Dual::new(z, T::zero())));
    (y.re, y.eps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-6 * (1.0 + b.abs())
    }

    #[test]
    fn gradients_match_finite_differences() {
        let h = 1e-6;
        let fd = |f: &dyn Fn(f64) -> f64, x: f64| (f(x + h) - f(x - h)) / (2.0 * h);
        let (a, b, z) = (1.7, 2.3, 0.4);
        assert!(close(inc_beta_dda(a, b, z), fd(&|t| inc_beta(t, b, z), a)));
        assert!(close(inc_beta_ddb(a, b, z), fd(&|t| inc_beta(a, t, z), b)));
        assert!(close(inc_beta_ddz(a, b, z), fd(&|t| inc_beta(a, b, t), z)));
        assert!(close(grad_reg_inc_gamma(a, b), fd(&|t| gamma_q(t, b), a)));
        assert!(close(
            grad_reg_lower_inc_gamma(a, b),
            fd(&|t| gamma_p(t, b), a)
        ));
        let (lq, d) = log_gamma_q_dgamma(a, b);
        assert!(close(lq, gamma_q(a, b).ln()));
        assert!(close(d, fd(&|t| gamma_q(t, b).ln(), a)));
        let g = grad_2F1(0.5, 1.5, 2.5, 0.3);
        assert!(close(
            g[3],
            fd(&|t| hypergeometric_2F1(0.5, 1.5, 2.5, t), 0.3)
        ));
        assert!(close(
            g[0],
            fd(&|t| hypergeometric_2F1(t, 1.5, 2.5, 0.3), 0.5)
        ));
        let (ga, gb, gz) = grad_pFq(&[0.5, 1.5], &[2.5], 0.3);
        assert!(close(ga[1], g[1]) && close(gb[0], g[2]) && close(gz, g[3]));
        let g3 = grad_F32([0.5, 1.5, 1.0], [2.5, 1.0], 0.3);
        assert!(close(g3[5], g[3]));
    }

    #[test]
    fn helpers() {
        let z = Complex::new(1.0, 2.0);
        let (p, n) = (i_times(z), neg_i_times(z));
        assert_eq!((p.re, p.im, n.re, n.im), (-2.0, 1.0, 2.0, -1.0));
        assert!(is_integer(3.0) && !is_integer(3.5) && is_integer(f64::INFINITY));
        assert!(!is_integer(f64::NAN));
        assert!(is_any_nan(&[1.0, f64::NAN]) && !is_any_nan(&[1.0]));
        let mut x = vec![1.0, 2.0];
        scaled_add(&mut x, &[1.0, 1.0], 0.5);
        assert_eq!(x, [1.5, 2.5]);
        assert!(finite_diff_stepsize(10.0) > finite_diff_stepsize(0.1));
    }
}
