//! ODE solutions with derivatives with respect to the initial time and the output times.
//!
//! Each interval `[t_(i-1), t_i]` is mapped onto `s` in `[0, 1]` by `t = a + s (b - a)`, which
//! turns `dy/dt = f(t, y, θ)` into `dy/ds = (b - a) f(a + s (b - a), y, θ)`. The endpoints `a`
//! and `b` then enter as parameters of a system integrated over fixed `s` limits, so the solvers'
//! usual derivatives with respect to parameters carry over to the times. As `b` moves, the
//! derivative of `y(b)` is `f(b, y(b), θ)`, and as `a` moves it is `-Φ f(a, y(a), θ)` with `Φ`
//! the sensitivity to the state at `a`, both up to the integration tolerance.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Result};

use crate::ode::{OdeOptions, OdeSystem, ode_adams_tol, ode_bdf_tol, ode_ckrk_tol, ode_rk45_tol};

/// The integrator used by [`ode_tol_times`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OdeMethod {
    /// [`ode_rk45_tol`].
    Rk45,
    /// [`ode_ckrk_tol`].
    Ckrk,
    /// [`ode_adams_tol`].
    Adams,
    /// [`ode_bdf_tol`].
    Bdf,
}

/// `f` on one interval, rescaled to `s` in `[0, 1]`. The last two parameters are the interval
/// endpoints `a` and `b`.
struct Rescaled<'a, F> {
    f: &'a F,
    n_theta: usize,
}

impl<F: OdeSystem> OdeSystem for Rescaled<'_, F> {
    fn eval<S: Real>(&self, s: S, y: &[S], theta: &[S]) -> Vec<S> {
        let (p, ends) = theta.split_at(self.n_theta);
        let (a, b) = (ends[0], ends[1]);
        let h = b - a;
        self.f
            .eval(a + s * h, y, p)
            .into_iter()
            .map(|v| v * h)
            .collect()
    }
}

/// Solves `dy/dt = f(t, y, θ)` from `y(t0) = y0` and returns `y` at each time in `ts`, with
/// derivatives with respect to `t0` and `ts` as well as `y0` and `theta`. The Functions Reference
/// types `initial_time` and `times` as `real` without the "data only" it puts on the tolerances,
/// so Stan programs may pass parameters there; this is that form.
///
/// Every interval between consecutive output times is solved separately with `method`, so the
/// step size control restarts at each output time. Equal consecutive times are allowed.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `t0` or a time is not finite, `ts[0]` is not greater than
/// `t0`, or `ts` is not sorted; [`Error::DimensionMismatch`] when `ts` is empty; and every error of
/// the chosen solver.
pub fn ode_tol_times<T: Real, F: OdeSystem>(
    method: OdeMethod,
    f: &F,
    y0: &[T],
    t0: T,
    ts: &[T],
    theta: &[T],
    options: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    const NAME: &str = "ode_tol_times";
    if !t0.value().is_finite() {
        return Err(Error::parameter(NAME, "initial time", t0.value(), "finite"));
    }
    if ts.is_empty() {
        return Err(Error::dimension(NAME, "no output times"));
    }
    if let Some(t) = ts.iter().find(|t| !t.value().is_finite()) {
        return Err(Error::parameter(NAME, "times", t.value(), "finite"));
    }
    if !(ts[0].value() > t0.value()) {
        return Err(Error::parameter(
            NAME,
            "times",
            ts[0].value(),
            "greater than the initial time",
        ));
    }
    if let Some(w) = ts.windows(2).find(|w| w[1].value() < w[0].value()) {
        return Err(Error::parameter(NAME, "times", w[1].value(), "sorted"));
    }
    let g = Rescaled {
        f,
        n_theta: theta.len(),
    };
    let mut params: Vec<T> = theta.to_vec();
    params.extend([t0, t0]);
    let k = params.len();
    let mut y = y0.to_vec();
    let mut start = t0;
    let mut out = Vec::with_capacity(ts.len());
    for &t in ts {
        params[k - 2] = start;
        params[k - 1] = t;
        let solved = match method {
            OdeMethod::Rk45 => ode_rk45_tol(&g, &y, 0.0, &[1.0], &params, options),
            OdeMethod::Ckrk => ode_ckrk_tol(&g, &y, 0.0, &[1.0], &params, options),
            OdeMethod::Adams => ode_adams_tol(&g, &y, 0.0, &[1.0], &params, options),
            OdeMethod::Bdf => ode_bdf_tol(&g, &y, 0.0, &[1.0], &params, options),
        }?;
        y = solved.into_iter().next().ok_or_else(|| {
            Error::dimension(NAME, "the solver returned no state for an interval")
        })?;
        out.push(y.clone());
        start = t;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use stanmath_rs_autodiff::grad;

    /// dy/dt = -k y
    struct Decay;
    impl OdeSystem for Decay {
        fn eval<S: Real>(&self, _t: S, y: &[S], theta: &[S]) -> Vec<S> {
            vec![-(theta[0] * y[0])]
        }
    }

    #[test]
    fn time_derivatives_match_closed_form() {
        let (y0, k, t0, t1) = (1.5, 0.7, 0.2, 1.3);
        let o = OdeOptions {
            relative_tolerance: 1e-10,
            absolute_tolerance: 1e-12,
            max_num_steps: 100_000,
        };
        for method in [OdeMethod::Rk45, OdeMethod::Ckrk, OdeMethod::Bdf] {
            let g = grad(
                |p| {
                    ode_tol_times(method, &Decay, &p[0..1], p[2], &p[3..4], &p[1..2], &o).unwrap()
                        [0][0]
                },
                &[y0, k, t0, t1],
            );
            let y = y0 * (-k * (t1 - t0)).exp();
            let expected = [y / y0, -(t1 - t0) * y, k * y, -k * y];
            for (a, b) in g.iter().zip(expected) {
                assert!((a - b).abs() < 1e-6, "{method:?}: {a} vs {b}");
            }
        }
    }
}
