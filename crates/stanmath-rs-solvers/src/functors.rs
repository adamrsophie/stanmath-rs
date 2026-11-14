//! Higher-order functions: Stan's `reduce_sum`, `reduce_sum_static`, `map_rect`, and the
//! deprecated `integrate_ode_*` interface.

use stanmath_rs_autodiff::Real;
use stanmath_rs_core::{Error, Result};

use crate::ode::{OdeOptions, OdeSystem, ode_adams_tol, ode_bdf_tol, ode_rk45_tol};

/// Sums `partial_sum` over consecutive slices of at most `grainsize` elements of `x`.
// Sequential, because a `Var` tape is not thread-safe and the closure cannot be re-run on another
// tape; `reduce_sum_parallel` in `parallel.rs` takes a generic trait instead and runs in parallel.
fn sliced_sum<X, T: Real>(
    name: &'static str,
    mut partial_sum: impl FnMut(&[X], usize, usize) -> Result<T>,
    x: &[X],
    grainsize: usize,
) -> Result<T> {
    if grainsize == 0 {
        return Err(Error::parameter(name, "grainsize", 0.0, "positive"));
    }
    let mut total = T::zero();
    let mut start = 0;
    while start < x.len() {
        let end = (start + grainsize).min(x.len());
        total += partial_sum(&x[start..end], start, end)?;
        start = end;
    }
    Ok(total)
}

/// Stan's `reduce_sum`: `Σ partial_sum(x[start..end], start, end)` over a partition of `x` into
/// slices of at most `grainsize` elements. Shared arguments are captured by the closure.
///
/// The partial sum receives the slice and its zero-based half-open range in `x`; Stan passes the
/// one-based inclusive range `start + 1` to `end`. The result does not depend on the partition
/// up to rounding. An empty `x` gives zero.
///
/// # Errors
///
/// [`Error::InvalidParameter`] when `grainsize` is zero, and any error from `partial_sum`.
///
/// ```
/// use stanmath_rs_solvers::reduce_sum;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let s: f64 = reduce_sum(|xs: &[f64], _, _| Ok(xs.iter().sum()), &x, 2).unwrap();
/// assert_eq!(s, 15.0);
/// ```
pub fn reduce_sum<X, T: Real>(
    partial_sum: impl FnMut(&[X], usize, usize) -> Result<T>,
    x: &[X],
    grainsize: usize,
) -> Result<T> {
    sliced_sum("reduce_sum", partial_sum, x, grainsize)
}

/// Stan's `reduce_sum_static`: as [`reduce_sum`], with the partition fixed by `grainsize` alone,
/// so the result is reproducible to the last bit.
///
/// # Errors
///
/// As [`reduce_sum`].
pub fn reduce_sum_static<X, T: Real>(
    partial_sum: impl FnMut(&[X], usize, usize) -> Result<T>,
    x: &[X],
    grainsize: usize,
) -> Result<T> {
    sliced_sum("reduce_sum_static", partial_sum, x, grainsize)
}

/// Stan's `map_rect`: evaluates `f(phi, thetas[j], x_r[j], x_i[j])` for every job `j` and
/// concatenates the results.
///
/// `phi` holds the shared parameters; `thetas`, `x_r`, and `x_i` hold the job-specific
/// parameters, real data, and integer data, one entry per job. Jobs run sequentially; see
/// [`map_rect_parallel`](crate::map_rect_parallel) for the parallel form.
///
/// # Errors
///
/// [`Error::DimensionMismatch`] when `thetas`, `x_r`, and `x_i` disagree on the number of jobs,
/// and any error from `f`.
pub fn map_rect<T: Real>(
    mut f: impl FnMut(&[T], &[T], &[f64], &[i64]) -> Result<Vec<T>>,
    phi: &[T],
    thetas: &[Vec<T>],
    x_r: &[Vec<f64>],
    x_i: &[Vec<i64>],
) -> Result<Vec<T>> {
    let jobs = thetas.len();
    if x_r.len() != jobs || x_i.len() != jobs {
        return Err(Error::dimension(
            "map_rect",
            format!(
                "{jobs} job parameter sets, {} real data sets, {} integer data sets",
                x_r.len(),
                x_i.len()
            ),
        ));
    }
    let mut out = Vec::new();
    for ((theta, xr), xi) in thetas.iter().zip(x_r).zip(x_i) {
        out.extend(f(phi, theta, xr, xi)?);
    }
    Ok(out)
}

/// A right-hand side in the deprecated form `f(t, y, θ, x_r, x_i)` used by `integrate_ode_*`.
pub trait OdeSystemData {
    /// Evaluates `dy/dt`.
    fn eval<S: Real>(&self, t: S, y: &[S], theta: &[S], x_r: &[f64], x_i: &[i64]) -> Vec<S>;
}

/// Binds the data of an [`OdeSystemData`] so it can be passed to the current ODE solvers.
struct BoundData<'a, F> {
    f: &'a F,
    x_r: &'a [f64],
    x_i: &'a [i64],
}

impl<F: OdeSystemData> OdeSystem for BoundData<'_, F> {
    fn eval<S: Real>(&self, t: S, y: &[S], theta: &[S]) -> Vec<S> {
        self.f.eval(t, y, theta, self.x_r, self.x_i)
    }
}

/// Stan's deprecated `integrate_ode_rk45`. `options` of `None` uses [`OdeOptions::NONSTIFF`],
/// the old interface's defaults.
///
/// # Errors
///
/// As [`ode_rk45_tol`](crate::ode_rk45_tol).
#[allow(clippy::too_many_arguments)] // mirrors Stan's argument list
pub fn integrate_ode_rk45<T: Real, F: OdeSystemData>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    x_r: &[f64],
    x_i: &[i64],
    options: Option<&OdeOptions>,
) -> Result<Vec<Vec<T>>> {
    let bound = BoundData { f, x_r, x_i };
    let o = options.copied().unwrap_or(OdeOptions::NONSTIFF);
    ode_rk45_tol(&bound, y0, t0, ts, theta, &o)
}

/// Stan's deprecated `integrate_ode_bdf`. `options` of `None` uses [`OdeOptions::STIFF`].
///
/// # Errors
///
/// As [`ode_bdf_tol`](crate::ode_bdf_tol).
#[allow(clippy::too_many_arguments)] // mirrors Stan's argument list
pub fn integrate_ode_bdf<T: Real, F: OdeSystemData>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    x_r: &[f64],
    x_i: &[i64],
    options: Option<&OdeOptions>,
) -> Result<Vec<Vec<T>>> {
    let bound = BoundData { f, x_r, x_i };
    let o = options.copied().unwrap_or(OdeOptions::STIFF);
    ode_bdf_tol(&bound, y0, t0, ts, theta, &o)
}

/// Stan's deprecated `integrate_ode_adams`. `options` of `None` uses [`OdeOptions::STIFF`].
///
/// # Errors
///
/// As [`ode_adams_tol`](crate::ode_adams_tol).
#[allow(clippy::too_many_arguments)] // mirrors Stan's argument list
pub fn integrate_ode_adams<T: Real, F: OdeSystemData>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    x_r: &[f64],
    x_i: &[i64],
    options: Option<&OdeOptions>,
) -> Result<Vec<Vec<T>>> {
    let bound = BoundData { f, x_r, x_i };
    let o = options.copied().unwrap_or(OdeOptions::STIFF);
    ode_adams_tol(&bound, y0, t0, ts, theta, &o)
}
