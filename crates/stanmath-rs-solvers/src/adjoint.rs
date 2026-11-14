//! Stan's `ode_adjoint_tol_ctl` under its own name and argument checks.
//!
//! Stan solves the sensitivities backward in time with CVODES. Our ODE solvers already return
//! exact derivatives of the computed solution through Newton refinement in the caller's scalar
//! type, so this function validates the adjoint controls as Stan does and then runs the forward
//! solver chosen by `solver_forward`. The backward, quadrature, checkpoint, and interpolation
//! settings have no effect on the result.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Result};

use crate::ode::{OdeOptions, OdeSystem, ode_adams_tol, ode_bdf_tol};

/// Control arguments of [`ode_adjoint_tol_ctl`], in Stan's order.
#[derive(Clone, Debug)]
pub struct AdjointOptions {
    /// Relative tolerance of the forward solve.
    pub rel_tol_forward: f64,
    /// Absolute tolerance of the forward solve, one entry per state.
    pub abs_tol_forward: Vec<f64>,
    /// Relative tolerance of the backward solve (validated only).
    pub rel_tol_backward: f64,
    /// Absolute tolerance of the backward solve, one entry per state (validated only).
    pub abs_tol_backward: Vec<f64>,
    /// Relative tolerance of the backward quadrature (validated only).
    pub rel_tol_quadrature: f64,
    /// Absolute tolerance of the backward quadrature (validated only).
    pub abs_tol_quadrature: f64,
    /// Maximum number of steps between two output times.
    pub max_num_steps: usize,
    /// Steps between checkpoints of the forward solution (validated only).
    pub num_steps_between_checkpoints: usize,
    /// 1 for Hermite, 2 for polynomial interpolation (validated only).
    pub interpolation_polynomial: i32,
    /// 1 for Adams (nonstiff), 2 for BDF (stiff).
    pub solver_forward: i32,
    /// 1 for Adams, 2 for BDF (validated only).
    pub solver_backward: i32,
}

const NAME: &str = "ode_adjoint_tol_ctl";

fn positive_finite(what: &'static str, v: f64) -> Result<()> {
    if v > 0.0 && v.is_finite() {
        Ok(())
    } else {
        Err(Error::parameter(NAME, what, v, "positive and finite"))
    }
}

fn tolerance_vector(what: &'static str, v: &[f64], states: usize) -> Result<()> {
    if v.len() != states {
        return Err(Error::dimension(
            NAME,
            format!("{what} has {} entries for {states} states", v.len()),
        ));
    }
    v.iter().try_for_each(|&x| positive_finite(what, x))
}

fn one_or_two(what: &'static str, v: i32) -> Result<()> {
    if v == 1 || v == 2 {
        Ok(())
    } else {
        Err(Error::parameter(NAME, what, f64::from(v), "1 or 2"))
    }
}

/// Solves `dy/dt = f(t, y, theta)` at the output times `ts` with Stan's adjoint interface.
///
/// The forward solve uses `rel_tol_forward`, the smallest entry of `abs_tol_forward`, and
/// `max_num_steps`, with [`ode_adams_tol`] for `solver_forward == 1` and [`ode_bdf_tol`] for
/// `solver_forward == 2`. Derivatives with respect to `y0` and `theta` are exact derivatives of
/// that solution, as for every solver in this crate.
///
/// # Errors
///
/// [`Error::InvalidParameter`] for a tolerance that is not positive and finite, a zero step count,
/// or a solver or interpolation code other than 1 or 2; [`Error::DimensionMismatch`] when a
/// tolerance vector does not have one entry per state; and every error of the chosen solver.
pub fn ode_adjoint_tol_ctl<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    options: &AdjointOptions,
    theta: &[T],
) -> Result<Vec<Vec<T>>> {
    let o = options;
    let n = y0.len();
    positive_finite("rel_tol_forward", o.rel_tol_forward)?;
    tolerance_vector("abs_tol_forward", &o.abs_tol_forward, n)?;
    positive_finite("rel_tol_backward", o.rel_tol_backward)?;
    tolerance_vector("abs_tol_backward", &o.abs_tol_backward, n)?;
    positive_finite("rel_tol_quadrature", o.rel_tol_quadrature)?;
    positive_finite("abs_tol_quadrature", o.abs_tol_quadrature)?;
    if o.max_num_steps == 0 {
        return Err(Error::parameter(NAME, "max_num_steps", 0.0, "positive"));
    }
    if o.num_steps_between_checkpoints == 0 {
        return Err(Error::parameter(
            NAME,
            "num_steps_between_checkpoints",
            0.0,
            "positive",
        ));
    }
    one_or_two("interpolation_polynomial", o.interpolation_polynomial)?;
    one_or_two("solver_forward", o.solver_forward)?;
    one_or_two("solver_backward", o.solver_backward)?;
    let forward = OdeOptions {
        relative_tolerance: o.rel_tol_forward,
        absolute_tolerance: o
            .abs_tol_forward
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        max_num_steps: o.max_num_steps,
    };
    if o.solver_forward == 1 {
        ode_adams_tol(f, y0, t0, ts, theta, &forward)
    } else {
        ode_bdf_tol(f, y0, t0, ts, theta, &forward)
    }
}
