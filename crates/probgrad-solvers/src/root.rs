//! Scalar root finding with exact derivatives of the root.

use probgrad_autodiff::{Dual, Real};
use probgrad_core::{Error, Result};

use crate::newton_steps;

/// An equation `f(x, θ) = 0` in one unknown `x` with parameters `θ`.
pub trait Equation {
    /// Evaluates `f(x, θ)`.
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S;
}

/// Options for [`solve_root`].
#[derive(Clone, Copy, Debug)]
pub struct RootOptions {
    /// Lower end of a bracketing interval.
    pub lower: f64,
    /// Upper end of a bracketing interval; `f` must change sign between the ends.
    pub upper: f64,
    /// Relative tolerance on `x`.
    pub tolerance: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
}

impl RootOptions {
    /// Options with the bracket `[lower, upper]` and default tolerances.
    pub fn bracket(lower: f64, upper: f64) -> Self {
        RootOptions {
            lower,
            upper,
            tolerance: 1e-15,
            max_iterations: 200,
        }
    }
}

fn value_and_slope<F: Equation>(f: &F, x: f64, theta: &[f64]) -> (f64, f64) {
    let t: Vec<Dual<f64>> = theta.iter().map(|&v| Dual::constant(v)).collect();
    let y = f.eval(Dual::variable(x), &t);
    (y.re, y.eps)
}

/// Safeguarded Newton iteration in `f64`: Newton steps that leave the bracket are replaced by
/// bisection, and the bracket shrinks every iteration.
fn find_root<F: Equation>(f: &F, theta: &[f64], o: &RootOptions) -> Result<f64> {
    const NAME: &str = "solve_root";
    let (mut lo, mut hi) = (o.lower, o.upper);
    let (f_lo, f_hi) = (
        value_and_slope(f, lo, theta).0,
        value_and_slope(f, hi, theta).0,
    );
    if f_lo == 0.0 {
        return Ok(lo);
    }
    if f_hi == 0.0 {
        return Ok(hi);
    }
    if !(f_lo * f_hi < 0.0) {
        return Err(Error::parameter(
            NAME,
            "bracket",
            f_lo * f_hi,
            "a sign change of f",
        ));
    }
    let lo_negative = f_lo < 0.0;
    let mut x = 0.5 * (lo + hi);
    for _ in 0..o.max_iterations {
        let (fx, slope) = value_and_slope(f, x, theta);
        if fx == 0.0 {
            return Ok(x);
        }
        if (fx < 0.0) == lo_negative {
            lo = x;
        } else {
            hi = x;
        }
        let newton = x - fx / slope;
        let next = if newton > lo.min(hi) && newton < lo.max(hi) && newton.is_finite() {
            newton
        } else {
            0.5 * (lo + hi)
        };
        if (next - x).abs() <= o.tolerance * x.abs().max(1.0) {
            return Ok(next);
        }
        x = next;
    }
    Err(Error::NoConvergence {
        function: NAME,
        iterations: o.max_iterations,
    })
}

/// Solves `f(x, θ) = 0` for `x` within a bracket, with derivatives of the root with respect to `θ`
/// exact to the order carried by `T` (see the [crate documentation](crate)).
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `f` does not change sign over the bracket, and
/// [`Error::NoConvergence`] when the iteration limit is reached.
pub fn solve_root<T: Real, F: Equation>(f: &F, theta: &[T], options: &RootOptions) -> Result<T> {
    let values: Vec<f64> = theta.iter().map(Real::value).collect();
    let root = find_root(f, &values, options)?;
    let mut x = T::from_f64(root);
    let lifted: Vec<Dual<T>> = theta.iter().map(|&t| Dual::constant(t)).collect();
    for _ in 0..newton_steps(T::ORDER) {
        let y = f.eval(Dual::variable(x), &lifted);
        x -= y.re / y.eps;
    }
    Ok(x)
}
