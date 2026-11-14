//! Fixed points `x = G(x, θ)`, Stan Math's `algebra_solver_fp`.
//!
//! The `f64` iteration is Anderson acceleration (Walker and Ni 2011, "Anderson acceleration for
//! fixed-point iterations", SIAM J. Numer. Anal. 49) with a memory of `memory` past steps, which
//! falls back to the plain iteration `x ← G(x, θ)` whenever the small least squares problem is
//! singular. Derivatives come from Newton steps on `x - G(x, θ) = 0` in the caller's scalar type,
//! as for [`solve_system`](crate::solve_system).

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Result};

use crate::newton_steps;
use crate::system::{EquationSystem, solve_linear, value_and_jacobian};

const NAME: &str = "solve_fixed_point";

/// Options for [`solve_fixed_point_tol`].
#[derive(Clone, Copy, Debug)]
pub struct FixedPointOptions {
    /// Stop when the infinity norm of `G(x, θ) - x` is below this value.
    pub function_tolerance: f64,
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Number of past steps used by Anderson acceleration; 0 gives the plain iteration.
    pub memory: usize,
}

impl Default for FixedPointOptions {
    /// Tolerance `1e-8`, 200 iterations, and a memory of 5. These are our choices, not checked
    /// against Stan Math's defaults.
    fn default() -> Self {
        FixedPointOptions {
            function_tolerance: 1e-8,
            max_iterations: 200,
            memory: 5,
        }
    }
}

/// `x - G(x, θ)`, whose root is the fixed point.
struct Residual<'a, F> {
    g: &'a F,
}

impl<F: EquationSystem> EquationSystem for Residual<'_, F> {
    fn eval<S: Real>(&self, x: &[S], theta: &[S]) -> Vec<S> {
        self.g
            .eval(x, theta)
            .into_iter()
            .zip(x)
            .map(|(g, &xi)| xi - g)
            .collect()
    }
}

fn inf_norm(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn anderson<F: EquationSystem>(
    g: &F,
    guess: &[f64],
    theta: &[f64],
    o: &FixedPointOptions,
) -> Result<Vec<f64>> {
    let n = guess.len();
    let map = |x: &[f64]| -> Result<Vec<f64>> {
        let gx = g.eval(x, theta);
        if gx.len() != n {
            return Err(Error::dimension(
                NAME,
                format!("G returned {} values for {n} unknowns", gx.len()),
            ));
        }
        if let Some(v) = gx.iter().find(|v| !v.is_finite()) {
            return Err(Error::parameter(NAME, "G(x)", *v, "finite"));
        }
        Ok(gx)
    };
    let mut x = guess.to_vec();
    let mut gx = map(&x)?;
    let mut f = sub(&gx, &x);
    // Differences of successive residuals and of successive map values, newest last.
    let mut df: Vec<Vec<f64>> = Vec::new();
    let mut dg: Vec<Vec<f64>> = Vec::new();
    for _ in 0..o.max_iterations {
        if inf_norm(&f) <= o.function_tolerance {
            return Ok(x);
        }
        let m = df.len();
        let mut next = gx.clone();
        if m > 0 {
            // Least squares min |f - dF gamma| through the normal equations.
            let a: Vec<Vec<f64>> = (0..m)
                .map(|i| {
                    (0..m)
                        .map(|j| df[i].iter().zip(&df[j]).map(|(p, q)| p * q).sum())
                        .collect()
                })
                .collect();
            let b: Vec<f64> = (0..m)
                .map(|i| df[i].iter().zip(&f).map(|(p, q)| p * q).sum())
                .collect();
            match solve_linear(NAME, a, b) {
                Ok(gamma) if gamma.iter().all(|v| v.is_finite()) => {
                    for (j, gj) in gamma.iter().enumerate() {
                        for (v, d) in next.iter_mut().zip(&dg[j]) {
                            *v -= gj * d;
                        }
                    }
                }
                _ => {
                    df.clear();
                    dg.clear();
                }
            }
        }
        let g_new = map(&next)?;
        let f_new = sub(&g_new, &next);
        if o.memory > 0 {
            df.push(sub(&f_new, &f));
            dg.push(sub(&g_new, &gx));
            if df.len() > o.memory {
                df.remove(0);
                dg.remove(0);
            }
        }
        x = next;
        gx = g_new;
        f = f_new;
    }
    if inf_norm(&f) <= o.function_tolerance {
        return Ok(x);
    }
    Err(Error::NoConvergence {
        function: NAME,
        iterations: o.max_iterations,
    })
}

/// Solves `x = G(x, θ)` from `guess`, where `g` evaluates `G`. Stan Math's `algebra_solver_fp`
/// without its scaling vectors.
///
/// The derivatives of the fixed point with respect to `θ` are exact to the order carried by `T`,
/// provided `I - ∂G/∂x` is nonsingular there.
///
/// # Errors
///
/// [`Error::NoConvergence`] when the iteration does not reach the tolerance,
/// [`Error::InvalidParameter`] when `G` is not finite or `I - ∂G/∂x` is singular at the solution,
/// and [`Error::DimensionMismatch`] when `G` does not return one value per unknown.
pub fn solve_fixed_point_tol<T: Real, F: EquationSystem>(
    g: &F,
    guess: &[f64],
    theta: &[T],
    options: &FixedPointOptions,
) -> Result<Vec<T>> {
    if !(options.function_tolerance > 0.0) {
        return Err(Error::parameter(
            NAME,
            "function_tolerance",
            options.function_tolerance,
            "positive",
        ));
    }
    let values: Vec<f64> = theta.iter().map(Real::value).collect();
    let point = anderson(g, guess, &values, options)?;
    let residual = Residual { g };
    let mut x: Vec<T> = point.into_iter().map(T::from_f64).collect();
    for _ in 0..newton_steps(T::ORDER) {
        let (fx, jac) = value_and_jacobian(&residual, &x, theta);
        let step = solve_linear(NAME, jac, fx)?;
        for (xi, s) in x.iter_mut().zip(step) {
            *xi -= s;
        }
    }
    Ok(x)
}

/// [`solve_fixed_point_tol`] with [`FixedPointOptions::default`].
///
/// # Errors
///
/// As [`solve_fixed_point_tol`].
pub fn solve_fixed_point<T: Real, F: EquationSystem>(
    g: &F,
    guess: &[f64],
    theta: &[T],
) -> Result<Vec<T>> {
    solve_fixed_point_tol(g, guess, theta, &FixedPointOptions::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::grad;

    /// G(x, theta) = theta cos x
    struct CosMap;
    impl EquationSystem for CosMap {
        fn eval<S: Real>(&self, x: &[S], theta: &[S]) -> Vec<S> {
            vec![theta[0] * x[0].cos()]
        }
    }

    #[test]
    fn derivative_matches_implicit_function_theorem() {
        let theta = 0.5;
        let x: f64 = solve_fixed_point(&CosMap, &[0.0], &[theta]).unwrap()[0];
        assert!((x - theta * x.cos()).abs() < 1e-8);
        let g = grad(
            |p| solve_fixed_point(&CosMap, &[0.0], p).unwrap()[0],
            &[theta],
        );
        let expected = x.cos() / (1.0 + theta * x.sin());
        assert!((g[0] - expected).abs() < 1e-6, "{} vs {expected}", g[0]);
    }
}
