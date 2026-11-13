//! Integration with differentiable limits (Stan's `integrate_1d`).

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::Result;

use crate::quadrature::{Integrand, QuadratureOptions, integrate};

/// `∫_{x0}^{x} f(s, θ) ds`, where `x` has primal value `x0`, through the derivative order of `T`.
///
/// With `δ = x - x0` (zero primal value), the integral is `δ ∫_0^1 f(x0 + u δ, θ) du`. Expanding
/// `f` in powers of `u δ`, the term of order `δ^(m + 1)` involves `∫_0^1 u^m du`, which Gauss
/// Legendre quadrature with `k` nodes on `[0, 1]` integrates exactly for `m <= 2k - 1`. Derivatives
/// up to order `T::ORDER` need `m <= T::ORDER - 1`, so `k = ceil(T::ORDER / 2)` nodes suffice.
pub(crate) fn limit_term<T: Real, F: Integrand>(f: &F, x: T, theta: &[T]) -> T {
    let x0 = x.value();
    if T::ORDER == 0 || !x0.is_finite() {
        return T::zero();
    }
    let delta = x - x0;
    let r = 0.5 * (3.0f64 / 5.0).sqrt();
    let s = 0.5 / 3f64.sqrt();
    let (nodes, weights): (&[f64], &[f64]) = match T::ORDER {
        1 | 2 => (&[0.5], &[1.0]),
        3 | 4 => (&[0.5 - s, 0.5 + s], &[0.5, 0.5]),
        _ => (
            &[0.5 - r, 0.5, 0.5 + r],
            &[5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0],
        ),
    };
    let mut total = T::zero();
    for (u, w) in nodes.iter().zip(weights) {
        total += f.eval(delta * *u + x0, theta) * *w;
    }
    total * delta
}

/// `∫_a^b f(x, θ) dx` with derivatives with respect to `θ` and to the limits `a` and `b`
/// (Stan's `integrate_1d`). Either limit may be infinite; an infinite limit has no derivative.
///
/// The derivative with respect to `b` is `f(b, θ)` and with respect to `a` is `-f(a, θ)`; higher
/// derivatives are exact up to the derivative order of `T` (up to six).
///
/// # Errors
///
/// As [`integrate`].
pub fn integrate_1d<T: Real, F: Integrand>(
    f: &F,
    a: T,
    b: T,
    theta: &[T],
    relative_tolerance: f64,
) -> Result<T> {
    let options = QuadratureOptions {
        relative_tolerance,
        ..QuadratureOptions::default()
    };
    let core = integrate(f, a.value(), b.value(), theta, &options)?;
    Ok(core + limit_term(f, b, theta) - limit_term(f, a, theta))
}
