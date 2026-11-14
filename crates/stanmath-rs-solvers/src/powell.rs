//! Powell's dogleg trust region method for nonlinear systems, and Stan's solver names.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Result};

use crate::newton_steps;
use crate::system::{
    EquationSystem, SystemOptions, solve_linear, solve_system, value_and_jacobian,
};

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `J v` for a dense row-major Jacobian.
fn mat_vec(j: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    j.iter()
        .map(|row| row.iter().zip(v).map(|(a, b)| a * b).sum())
        .collect()
}

/// `Jᵀ v` for a dense row-major Jacobian.
fn mat_t_vec(j: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    let n = j.first().map_or(0, Vec::len);
    let mut out = vec![0.0; n];
    for (row, vi) in j.iter().zip(v) {
        for (o, a) in out.iter_mut().zip(row) {
            *o += a * vi;
        }
    }
    out
}

/// The dogleg step inside a trust region of radius `delta` (Nocedal and Wright, *Numerical
/// Optimization*, section 4.1), from the Gauss Newton step `p_gn` (if the Jacobian is
/// nonsingular) and the Cauchy point along the steepest descent direction of `||F||^2 / 2`.
fn dogleg(jac: &[Vec<f64>], fx: &[f64], p_gn: Option<&[f64]>, delta: f64) -> Vec<f64> {
    if let Some(p) = p_gn {
        if norm2(p) <= delta {
            return p.to_vec();
        }
    }
    let g = mat_t_vec(jac, fx);
    let jg = mat_vec(jac, &g);
    let g2 = g.iter().map(|x| x * x).sum::<f64>();
    let jg2 = jg.iter().map(|x| x * x).sum::<f64>();
    let alpha = if jg2 > 0.0 { g2 / jg2 } else { 0.0 };
    let p_sd: Vec<f64> = g.iter().map(|x| -alpha * x).collect();
    let sd_norm = norm2(&p_sd);
    let Some(p_gn) = p_gn else {
        let scale = if sd_norm > delta {
            delta / sd_norm
        } else {
            1.0
        };
        return p_sd.iter().map(|x| x * scale).collect();
    };
    if sd_norm >= delta {
        let g_norm = g2.sqrt();
        if g_norm == 0.0 {
            return vec![0.0; g.len()];
        }
        return g.iter().map(|x| -delta * x / g_norm).collect();
    }
    // Solve ||p_sd + tau (p_gn - p_sd)|| = delta for tau in [0, 1].
    let d: Vec<f64> = p_gn.iter().zip(&p_sd).map(|(a, b)| a - b).collect();
    let a = d.iter().map(|x| x * x).sum::<f64>();
    let b = 2.0 * p_sd.iter().zip(&d).map(|(x, y)| x * y).sum::<f64>();
    let c = sd_norm * sd_norm - delta * delta;
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let tau = if a > 0.0 {
        ((-b + disc.sqrt()) / (2.0 * a)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    p_sd.iter().zip(&d).map(|(s, di)| s + tau * di).collect()
}

/// Finds a root of `F(·, θ)` in `f64` by the dogleg method.
fn powell<F: EquationSystem>(
    name: &'static str,
    f: &F,
    guess: &[f64],
    theta: &[f64],
    o: &SystemOptions,
) -> Result<Vec<f64>> {
    let mut x = guess.to_vec();
    let (mut fx, mut jac) = value_and_jacobian(f, &x, theta);
    if fx.len() != x.len() {
        return Err(Error::dimension(
            name,
            format!("{} equations in {} unknowns", fx.len(), x.len()),
        ));
    }
    let mut delta = 100.0 * norm2(&x).max(1.0);
    for _ in 0..o.max_iterations {
        let residual = fx.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if !residual.is_finite() {
            return Err(Error::parameter(name, "F(x)", residual, "finite"));
        }
        if residual <= o.function_tolerance {
            return Ok(x);
        }
        let p_gn = solve_linear(name, jac.clone(), fx.clone())
            .ok()
            .map(|s| s.into_iter().map(|v| -v).collect::<Vec<f64>>())
            .filter(|s| s.iter().all(|v| v.is_finite()));
        let p = dogleg(&jac, &fx, p_gn.as_deref(), delta);
        let p_norm = norm2(&p);
        let trial: Vec<f64> = x.iter().zip(&p).map(|(a, b)| a + b).collect();
        let (f_trial, j_trial) = value_and_jacobian(f, &trial, theta);
        let current = fx.iter().map(|v| v * v).sum::<f64>();
        let trial_sq = f_trial.iter().map(|v| v * v).sum::<f64>();
        let linear: Vec<f64> = fx
            .iter()
            .zip(mat_vec(&jac, &p))
            .map(|(a, b)| a + b)
            .collect();
        let predicted = current - linear.iter().map(|v| v * v).sum::<f64>();
        let actual = if trial_sq.is_finite() {
            current - trial_sq
        } else {
            f64::NEG_INFINITY
        };
        let rho = if predicted > 0.0 {
            actual / predicted
        } else {
            -1.0
        };
        if rho < 0.25 {
            delta = 0.25 * p_norm;
        } else if rho > 0.75 && p_norm >= 0.99 * delta {
            delta *= 2.0;
        }
        if rho > 1e-4 {
            x = trial;
            fx = f_trial;
            jac = j_trial;
            let scale = x.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            let size = p.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let residual = fx.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if size <= o.tolerance * scale && residual <= o.function_tolerance {
                return Ok(x);
            }
        }
        if !(delta > f64::EPSILON * norm2(&x).max(1.0)) {
            break;
        }
    }
    Err(Error::NoConvergence {
        function: name,
        iterations: o.max_iterations,
    })
}

/// Solves `F(x, θ) = 0` for `x` from `guess` by Powell's dogleg trust region method, with
/// derivatives of the solution exact to the order carried by `T` (Stan's `solve_powell_tol`).
///
/// The trust region makes this more robust than [`solve_system`] far from the solution. The
/// derivatives come from Newton steps in `T` at the converged root, as for [`solve_system`].
///
/// # Errors
///
/// [`Error::NoConvergence`] when the method stalls or runs out of iterations,
/// [`Error::InvalidParameter`] when `F` is not finite or the Jacobian at the solution is singular,
/// and [`Error::DimensionMismatch`] when `F` does not return one value per unknown.
pub fn solve_powell_tol<T: Real, F: EquationSystem>(
    f: &F,
    guess: &[f64],
    theta: &[T],
    options: &SystemOptions,
) -> Result<Vec<T>> {
    const NAME: &str = "solve_powell";
    let values: Vec<f64> = theta.iter().map(Real::value).collect();
    let root = powell(NAME, f, guess, &values, options)?;
    let mut x: Vec<T> = root.into_iter().map(T::from_f64).collect();
    for _ in 0..newton_steps(T::ORDER) {
        let (fx, jac) = value_and_jacobian(f, &x, theta);
        let step = solve_linear(NAME, jac, fx)?;
        for (xi, s) in x.iter_mut().zip(step) {
            *xi -= s;
        }
    }
    Ok(x)
}

/// [`solve_powell_tol`] with [`SystemOptions::default`] (Stan's `solve_powell`).
///
/// # Errors
///
/// As [`solve_powell_tol`].
pub fn solve_powell<T: Real, F: EquationSystem>(
    f: &F,
    guess: &[f64],
    theta: &[T],
) -> Result<Vec<T>> {
    solve_powell_tol(f, guess, theta, &SystemOptions::default())
}

/// [`solve_system`] under Stan's name `solve_newton_tol`.
///
/// # Errors
///
/// As [`solve_system`].
pub fn solve_newton_tol<T: Real, F: EquationSystem>(
    f: &F,
    guess: &[f64],
    theta: &[T],
    options: &SystemOptions,
) -> Result<Vec<T>> {
    solve_system(f, guess, theta, options)
}

/// [`solve_system`] with [`SystemOptions::default`] (Stan's `solve_newton`).
///
/// # Errors
///
/// As [`solve_system`].
pub fn solve_newton<T: Real, F: EquationSystem>(
    f: &F,
    guess: &[f64],
    theta: &[T],
) -> Result<Vec<T>> {
    solve_system(f, guess, theta, &SystemOptions::default())
}
