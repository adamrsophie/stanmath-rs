//! Stan's deprecated solver interfaces, which pass real and integer data separately from the
//! parameters: `algebra_solver`, `algebra_solver_newton`, and `integrate_ode`.
//!
//! They forward to [`solve_powell_tol`], [`solve_newton_tol`], and
//! [`integrate_ode_rk45`](crate::integrate_ode_rk45). New
//! code should use those directly and capture data in the system type instead.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::Result;

use crate::powell::{solve_newton_tol, solve_powell_tol};
use crate::system::{EquationSystem, SystemOptions};

pub use crate::functors::integrate_ode_rk45 as integrate_ode;

/// An algebraic system in Stan's deprecated form `f(y, theta, x_r, x_i)`, which returns a vector
/// that is zero at the solution.
pub trait AlgebraSystemData {
    /// Evaluates the system at unknowns `y`, parameters `theta`, and data `x_r`, `x_i`.
    fn eval<S: Real>(&self, y: &[S], theta: &[S], x_r: &[f64], x_i: &[i64]) -> Vec<S>;
}

/// Binds the data of an [`AlgebraSystemData`] so it can be solved as an [`EquationSystem`].
struct WithData<'a, F> {
    f: &'a F,
    x_r: &'a [f64],
    x_i: &'a [i64],
}

impl<F: AlgebraSystemData> EquationSystem for WithData<'_, F> {
    fn eval<S: Real>(&self, x: &[S], theta: &[S]) -> Vec<S> {
        self.f.eval(x, theta, self.x_r, self.x_i)
    }
}

/// Solves `f(y, theta, x_r, x_i) = 0` by Powell's dogleg method. Stan's deprecated
/// `algebra_solver`; `options` holds Stan's optional `rel_tol`, `f_tol`, and `max_steps`
/// (`None` uses [`SystemOptions::default`]).
///
/// # Errors
///
/// As [`solve_powell_tol`].
pub fn algebra_solver<T: Real, F: AlgebraSystemData>(
    f: &F,
    y_guess: &[f64],
    theta: &[T],
    x_r: &[f64],
    x_i: &[i64],
    options: Option<&SystemOptions>,
) -> Result<Vec<T>> {
    let system = WithData { f, x_r, x_i };
    let default = SystemOptions::default();
    solve_powell_tol(&system, y_guess, theta, options.unwrap_or(&default))
}

/// Solves `f(y, theta, x_r, x_i) = 0` by Newton's method. Stan's deprecated
/// `algebra_solver_newton`; `options` as in [`algebra_solver`].
///
/// # Errors
///
/// As [`solve_newton_tol`].
pub fn algebra_solver_newton<T: Real, F: AlgebraSystemData>(
    f: &F,
    y_guess: &[f64],
    theta: &[T],
    x_r: &[f64],
    x_i: &[i64],
    options: Option<&SystemOptions>,
) -> Result<Vec<T>> {
    let system = WithData { f, x_r, x_i };
    let default = SystemOptions::default();
    solve_newton_tol(&system, y_guess, theta, options.unwrap_or(&default))
}
