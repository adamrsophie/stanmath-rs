//! Systems of nonlinear equations with exact derivatives of the solution.

use probgrad_autodiff::{Dual, Real};
use probgrad_core::{Error, Result};

use crate::newton_steps;

/// A system `F(x, θ) = 0` of `n` equations in `n` unknowns `x` with parameters `θ`.
pub trait EquationSystem {
    /// Evaluates `F(x, θ)`; the result has the same length as `x`.
    fn eval<S: Real>(&self, x: &[S], theta: &[S]) -> Vec<S>;
}

/// Options for [`solve_system`].
#[derive(Clone, Copy, Debug)]
pub struct SystemOptions {
    /// Stop when the infinity norm of the Newton step is below `tolerance * max(1, |x|)`.
    pub tolerance: f64,
    /// Stop when the infinity norm of `F` is below this value.
    pub function_tolerance: f64,
    /// Maximum number of Newton iterations.
    pub max_iterations: usize,
}

impl Default for SystemOptions {
    fn default() -> Self {
        SystemOptions {
            tolerance: 1e-14,
            function_tolerance: 1e-12,
            max_iterations: 100,
        }
    }
}

/// Value and Jacobian of `F` at `x`, by one forward-mode pass per unknown.
#[allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices
fn value_and_jacobian<T: Real, F: EquationSystem>(
    f: &F,
    x: &[T],
    theta: &[T],
) -> (Vec<T>, Vec<Vec<T>>) {
    let n = x.len();
    let lifted: Vec<Dual<T>> = theta.iter().map(|&t| Dual::constant(t)).collect();
    let mut value = Vec::new();
    let mut jac = vec![vec![T::zero(); n]; n];
    for j in 0..n {
        let xd: Vec<Dual<T>> = x
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                if k == j {
                    Dual::variable(v)
                } else {
                    Dual::constant(v)
                }
            })
            .collect();
        let y = f.eval(&xd, &lifted);
        if j == 0 {
            value = y.iter().map(|d| d.re).collect();
        }
        for (i, d) in y.iter().enumerate() {
            jac[i][j] = d.eps;
        }
    }
    if n == 0 {
        value = Vec::new();
    }
    (value, jac)
}

/// Solves `a z = b` by Gaussian elimination with partial pivoting chosen on the values.
#[allow(clippy::needless_range_loop)] // dense matrix loops read more clearly with indices
fn solve_linear<T: Real>(mut a: Vec<Vec<T>>, mut b: Vec<T>) -> Result<Vec<T>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].value().abs().total_cmp(&a[j][c].value().abs()))
            .unwrap_or(c);
        if a[p][c].value() == 0.0 || !a[p][c].is_finite() {
            return Err(Error::parameter(
                "solve_system",
                "Jacobian",
                0.0,
                "nonsingular",
            ));
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let factor = a[r][c] / a[c][c];
            for k in c..n {
                let v = a[c][k];
                a[r][k] -= factor * v;
            }
            let v = b[c];
            b[r] -= factor * v;
        }
    }
    let mut z = vec![T::zero(); n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for k in r + 1..n {
            s -= a[r][k] * z[k];
        }
        z[r] = s / a[r][r];
    }
    Ok(z)
}

fn newton<F: EquationSystem>(
    f: &F,
    guess: &[f64],
    theta: &[f64],
    o: &SystemOptions,
) -> Result<Vec<f64>> {
    let mut x = guess.to_vec();
    for _ in 0..o.max_iterations {
        let (fx, jac) = value_and_jacobian(f, &x, theta);
        let residual = fx.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if !residual.is_finite() {
            return Err(Error::parameter("solve_system", "F(x)", residual, "finite"));
        }
        let step = solve_linear(jac, fx)?;
        let size = step.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = x.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (xi, s) in x.iter_mut().zip(&step) {
            *xi -= s;
        }
        if size <= o.tolerance * scale && residual <= o.function_tolerance {
            return Ok(x);
        }
        if residual == 0.0 {
            return Ok(x);
        }
    }
    Err(Error::NoConvergence {
        function: "solve_system",
        iterations: o.max_iterations,
    })
}

/// Solves `F(x, θ) = 0` for `x` by Newton's method from `guess`, with derivatives of the solution
/// with respect to `θ` exact to the order carried by `T`.
///
/// The Newton iteration is undamped (no line search or trust region), so `guess` must be close
/// enough to the solution for Newton's method to converge and to keep the iterates inside the
/// domain of `F`.
///
/// # Errors
///
/// [`Error::NoConvergence`] when Newton's method does not converge, and
/// [`Error::InvalidParameter`] when the Jacobian is singular.
pub fn solve_system<T: Real, F: EquationSystem>(
    f: &F,
    guess: &[f64],
    theta: &[T],
    options: &SystemOptions,
) -> Result<Vec<T>> {
    let values: Vec<f64> = theta.iter().map(Real::value).collect();
    let root = newton(f, guess, &values, options)?;
    let mut x: Vec<T> = root.into_iter().map(T::from_f64).collect();
    for _ in 0..newton_steps(T::ORDER) {
        let (fx, jac) = value_and_jacobian(f, &x, theta);
        let step = solve_linear(jac, fx)?;
        for (xi, s) in x.iter_mut().zip(step) {
            *xi -= s;
        }
    }
    Ok(x)
}
