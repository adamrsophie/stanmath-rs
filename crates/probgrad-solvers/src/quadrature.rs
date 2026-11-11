//! One-dimensional integration by tanh-sinh (double exponential) quadrature.
//!
//! On `[a, b]`, `x = c + h tanh(pi/2 sinh t)` maps the real line onto the interval with weights that
//! decay double exponentially, which handles integrable endpoint singularities. Infinite limits are
//! mapped to finite ones first. Nodes near an endpoint are formed from the complement
//! `1 - tanh(u) = 2 / (exp(2u) + 1)`, so they never round onto the endpoint itself. Because the nodes
//! and weights are constants, derivatives with respect to the parameters are the same rule applied
//! to the derivative of the integrand; the stopping test includes forward-mode tangents.

use core::f64::consts::FRAC_PI_2;

use probgrad_autodiff::Real;
use probgrad_core::{Error, Result};

/// An integrand `f(x, θ)`.
pub trait Integrand {
    /// Evaluates `f(x, θ)`.
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S;
}

/// Options for [`integrate`].
#[derive(Clone, Copy, Debug)]
pub struct QuadratureOptions {
    /// Stop when successive refinements differ by less than this, relative to the integral.
    pub relative_tolerance: f64,
    /// Maximum number of step halvings.
    pub max_levels: usize,
}

impl Default for QuadratureOptions {
    fn default() -> Self {
        QuadratureOptions {
            relative_tolerance: 1e-12,
            max_levels: 12,
        }
    }
}

/// `∫_a^b f(x, θ) dx`, with derivatives with respect to `θ`. Either limit may be infinite.
///
/// # Errors
///
/// [`Error::InvalidParameter`] for NaN or reversed limits, [`Error::NoConvergence`] when the
/// tolerance is not met, and an error when the integrand is not finite at a node.
pub fn integrate<T: Real, F: Integrand>(
    f: &F,
    a: f64,
    b: f64,
    theta: &[T],
    options: &QuadratureOptions,
) -> Result<T> {
    const NAME: &str = "integrate";
    if a.is_nan() || b.is_nan() || a > b {
        return Err(Error::parameter(
            NAME,
            "limits",
            b - a,
            "ordered and not NaN",
        ));
    }
    if a == b {
        return Ok(T::zero());
    }
    match (a.is_finite(), b.is_finite()) {
        (true, true) => tanh_sinh(|x| f.eval(x, theta), a, b, options),
        // x = a + s / (1 - s), s in [0, 1)
        (true, false) => tanh_sinh(
            |s: T| {
                let one_minus = -s + 1.0;
                f.eval(s / one_minus + a, theta) / one_minus.square()
            },
            0.0,
            1.0,
            options,
        ),
        // x = b - s / (1 - s), s in [0, 1)
        (false, true) => tanh_sinh(
            |s: T| {
                let one_minus = -s + 1.0;
                f.eval(-(s / one_minus) + b, theta) / one_minus.square()
            },
            0.0,
            1.0,
            options,
        ),
        // x = s / (1 - s^2), s in (-1, 1)
        (false, false) => tanh_sinh(
            |s: T| {
                let q = -s.square() + 1.0;
                f.eval(s / q, theta) * (s.square() + 1.0) / q.square()
            },
            -1.0,
            1.0,
            options,
        ),
    }
}

fn tanh_sinh<T: Real>(g: impl Fn(T) -> T, a: f64, b: f64, o: &QuadratureOptions) -> Result<T> {
    let half = 0.5 * (b - a);
    // Contribution of the nodes t = k h for the given k, symmetric about zero.
    let node_sum = |h: f64, odd_only: bool| -> Result<T> {
        let mut total = T::zero();
        if !odd_only {
            total += g(T::from_f64(0.5 * (a + b))) * (half * FRAC_PI_2);
        }
        // Each side stops on its own: once its node rounds onto its endpoint, or its weight
        // underflows. Near a = 0 nodes can approach the endpoint far more closely than near b.
        for side in [1.0, -1.0] {
            let mut k: i64 = 1;
            loop {
                let t = k as f64 * h;
                let u = FRAC_PI_2 * t.sinh();
                let cosh_u = u.cosh();
                let weight = half * FRAC_PI_2 * t.cosh() / (cosh_u * cosh_u);
                // Distance of the node from its endpoint: half (1 - tanh u).
                let gap = half * 2.0 / ((2.0 * u).exp() + 1.0);
                let x = if side > 0.0 { b - gap } else { a + gap };
                if !(weight > 1e-300) || !(gap > 0.0) || x == a || x == b {
                    break;
                }
                let value = g(T::from_f64(x));
                if !value.is_finite() {
                    return Err(Error::parameter(
                        "integrate",
                        "integrand",
                        value.value(),
                        "finite",
                    ));
                }
                total += value * weight;
                k += if odd_only { 2 } else { 1 };
            }
        }
        Ok(total)
    };
    let mut h = 1.0;
    let mut estimate = node_sum(h, false)? * h;
    for _ in 0..o.max_levels {
        h *= 0.5;
        let refined = estimate * 0.5 + node_sum(h, true)? * h;
        let change = (refined - estimate).magnitude();
        estimate = refined;
        if change <= o.relative_tolerance * estimate.magnitude().max(f64::MIN_POSITIVE) {
            return Ok(estimate);
        }
    }
    Err(Error::NoConvergence {
        function: "integrate",
        iterations: o.max_levels,
    })
}
