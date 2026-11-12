//! Ordinary differential equations with derivatives of the solution.
//!
//! # Methods
//!
//! * [`ode_rk45`]: the explicit Dormand and Prince 5(4) Runge Kutta pair (Dormand and Prince 1980,
//!   "A family of embedded Runge-Kutta formulae") with the step size control of Hairer, Norsett,
//!   and Wanner, *Solving Ordinary Differential Equations I*, section II.4. For nonstiff problems.
//! * [`ode_bdf`]: the three stage Radau IIA collocation method of order 5 (Hairer and Wanner,
//!   *Solving Ordinary Differential Equations II*, section IV.5), which is L-stable, with step size
//!   control by step doubling. For stiff problems. Stan Math uses CVODES BDF for `ode_bdf`; the two
//!   methods agree to the requested tolerance, not to rounding error.
//! * [`ode_ckrk`]: the Cash and Karp 5(4) pair (Cash and Karp 1990), as in Stan Math, under the
//!   same step size control as [`ode_rk45`].
//! * [`ode_adams`] is provided under Stan's name and uses the Dormand and Prince pair; Stan Math
//!   uses the variable order CVODES Adams method.
//!
//! # Derivatives
//!
//! Step sizes are chosen from primal values only, so every scalar type takes the same sequence of
//! steps. The explicit method is then differentiated step by step (in reverse mode the tape records
//! the steps). The implicit method solves its stage equations in `f64` and then takes Newton steps
//! in the caller's scalar type, as [`solve_system`](crate::solve_system) does, so its stage
//! derivatives are exact. In both cases the derivatives are the exact derivatives of the numerical
//! solution, which differ from the derivatives of the true solution by an amount of the order of
//! the integration tolerance. Stan Math instead integrates the forward sensitivity equations
//! under the same error control as the states; results agree to the tolerance.
//!
//! Output times and the initial time are data (`f64`) here;
//! [`ode_tol_times`](crate::ode_tol_times) takes them as parameters and gives derivatives with
//! respect to them.

use stanmath_rs_autodiff::{Dual, Real};
use stanmath_rs_core::{Error, Result};

use crate::newton_steps;
use crate::system::solve_linear;

/// A system of ordinary differential equations `dy/dt = f(t, y, θ)`.
///
/// Data that the right-hand side needs (Stan's variadic data arguments) can be stored in the
/// implementing type; `θ` carries the arguments that may have derivatives.
pub trait OdeSystem {
    /// Evaluates `f(t, y, θ)`; the result must have the same length as `y`.
    fn eval<S: Real>(&self, t: S, y: &[S], theta: &[S]) -> Vec<S>;
}

/// Tolerances and step limit for the ODE solvers.
#[derive(Clone, Copy, Debug)]
pub struct OdeOptions {
    /// Relative tolerance on each state.
    pub relative_tolerance: f64,
    /// Absolute tolerance on each state.
    pub absolute_tolerance: f64,
    /// Maximum number of steps between two consecutive output times.
    pub max_num_steps: usize,
}

impl OdeOptions {
    /// Stan's defaults for `ode_rk45` and `ode_ckrk`: tolerances `1e-6`, one million steps.
    pub const NONSTIFF: OdeOptions = OdeOptions {
        relative_tolerance: 1e-6,
        absolute_tolerance: 1e-6,
        max_num_steps: 1_000_000,
    };

    /// Stan's defaults for `ode_bdf` and `ode_adams`: tolerances `1e-10`, `10^8` steps.
    pub const STIFF: OdeOptions = OdeOptions {
        relative_tolerance: 1e-10,
        absolute_tolerance: 1e-10,
        max_num_steps: 100_000_000,
    };
}

impl Default for OdeOptions {
    fn default() -> Self {
        OdeOptions::NONSTIFF
    }
}

/// Checks the arguments as Stan's ODE functions do.
fn validate<T: Real>(
    name: &'static str,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    o: &OdeOptions,
) -> Result<()> {
    if y0.is_empty() {
        return Err(Error::dimension(name, "the initial state is empty"));
    }
    if let Some(v) = y0.iter().find(|v| !v.is_finite()) {
        return Err(Error::parameter(name, "initial state", v.value(), "finite"));
    }
    if !t0.is_finite() {
        return Err(Error::parameter(name, "initial time", t0, "finite"));
    }
    if ts.is_empty() {
        return Err(Error::dimension(name, "no output times"));
    }
    if let Some(&t) = ts.iter().find(|t| !t.is_finite()) {
        return Err(Error::parameter(name, "times", t, "finite"));
    }
    if !(ts[0] > t0) {
        return Err(Error::parameter(
            name,
            "times",
            ts[0],
            "greater than the initial time",
        ));
    }
    if let Some(w) = ts.windows(2).find(|w| w[1] < w[0]) {
        return Err(Error::parameter(name, "times", w[1], "sorted"));
    }
    if let Some(v) = theta.iter().find(|v| !v.is_finite()) {
        return Err(Error::parameter(name, "parameters", v.value(), "finite"));
    }
    let tol_ok = |v: f64| v > 0.0 && v.is_finite();
    if !tol_ok(o.relative_tolerance) {
        return Err(Error::parameter(
            name,
            "relative_tolerance",
            o.relative_tolerance,
            "positive finite",
        ));
    }
    if !tol_ok(o.absolute_tolerance) {
        return Err(Error::parameter(
            name,
            "absolute_tolerance",
            o.absolute_tolerance,
            "positive finite",
        ));
    }
    if o.max_num_steps == 0 {
        return Err(Error::parameter(name, "max_num_steps", 0.0, "positive"));
    }
    Ok(())
}

/// `f(t, y, θ)` with a check on the length of the result.
fn rhs<S: Real, F: OdeSystem>(
    f: &F,
    name: &'static str,
    t: f64,
    y: &[S],
    theta: &[S],
) -> Result<Vec<S>> {
    let out = f.eval(S::from_f64(t), y, theta);
    if out.len() != y.len() {
        return Err(Error::dimension(
            name,
            format!(
                "the right-hand side returned {} values for {} states",
                out.len(),
                y.len()
            ),
        ));
    }
    Ok(out)
}

/// `f(t, y, θ)` and its Jacobian `∂f/∂y`, by one forward-mode pass per state.
fn rhs_jacobian<T: Real, F: OdeSystem>(
    f: &F,
    name: &'static str,
    t: f64,
    y: &[T],
    theta: &[T],
) -> Result<(Vec<T>, Vec<Vec<T>>)> {
    let n = y.len();
    let th: Vec<Dual<T>> = theta.iter().map(|&v| Dual::constant(v)).collect();
    let mut value = Vec::with_capacity(n);
    let mut jac = vec![vec![T::zero(); n]; n];
    for col in 0..n {
        let yd: Vec<Dual<T>> = y
            .iter()
            .enumerate()
            .map(|(k, &v)| {
                if k == col {
                    Dual::variable(v)
                } else {
                    Dual::constant(v)
                }
            })
            .collect();
        let out = rhs(f, name, t, &yd, &th)?;
        if col == 0 {
            value = out.iter().map(|d| d.re).collect();
        }
        for (r, d) in out.iter().enumerate() {
            jac[r][col] = d.eps;
        }
    }
    Ok((value, jac))
}

/// Error weights `atol + rtol max(|y|, |y_new|)`.
fn weights(o: &OdeOptions, y: &[f64], y_new: &[f64]) -> Vec<f64> {
    y.iter()
        .zip(y_new)
        .map(|(a, b)| o.absolute_tolerance + o.relative_tolerance * a.abs().max(b.abs()))
        .collect()
}

/// Root mean square of `v / w`.
fn rms(v: &[f64], w: &[f64]) -> f64 {
    let n = v.len().max(1) as f64;
    (v.iter().zip(w).map(|(a, b)| (a / b) * (a / b)).sum::<f64>() / n).sqrt()
}

/// Starting step size (Hairer, Norsett, and Wanner, section II.4), from primal values.
#[allow(clippy::too_many_arguments)]
fn initial_step<F: OdeSystem>(
    f: &F,
    name: &'static str,
    t0: f64,
    y0: &[f64],
    f0: &[f64],
    theta: &[f64],
    span: f64,
    order: i32,
    o: &OdeOptions,
) -> Result<f64> {
    let w = weights(o, y0, y0);
    let d0 = rms(y0, &w);
    let d1 = rms(f0, &w);
    let mut h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    h0 = h0.min(span);
    let y1: Vec<f64> = y0.iter().zip(f0).map(|(y, d)| y + h0 * d).collect();
    let f1 = rhs(f, name, t0 + h0, &y1, theta)?;
    let diff: Vec<f64> = f1.iter().zip(f0).map(|(a, b)| a - b).collect();
    let d2 = rms(&diff, &w) / h0;
    let big = d1.max(d2);
    let h1 = if !(big > 1e-15) || !big.is_finite() {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / big).powf(1.0 / f64::from(order + 1))
    };
    Ok((100.0 * h0).min(h1).min(span))
}

/// Step size change factor after a step with scaled error `err`, for a method whose local error
/// is of order `p + 1`.
fn step_factor(err: f64, p: i32, max_growth: f64) -> f64 {
    if err == 0.0 {
        max_growth
    } else if !err.is_finite() {
        0.2
    } else {
        (0.9 * err.powf(-1.0 / f64::from(p + 1))).clamp(0.2, max_growth)
    }
}

/// Stops an integration whose step size has fallen below the resolution of `t`.
fn check_step(name: &'static str, t: f64, h: f64, steps: usize) -> Result<()> {
    if h <= 16.0 * f64::EPSILON * t.abs().max(1.0) || !h.is_finite() {
        Err(Error::NoConvergence {
            function: name,
            iterations: steps,
        })
    } else {
        Ok(())
    }
}

// Dormand and Prince 5(4). Row `s` holds the coefficients of stage `s` (row 0 is unused); row 6
// holds the fifth-order weights, which are also the coefficients of the seventh (FSAL) stage.
const DP_C: [f64; 7] = [0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0];
const DP_A: [[f64; 6]; 7] = [
    [0.0; 6],
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];
// Fifth-order weights minus the embedded fourth-order weights.
const DP_E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

// Cash and Karp 5(4) (Cash and Karp 1990, "A variable order Runge-Kutta method for initial value
// problems with rapidly varying right-hand sides", ACM TOMS 16). Row `s` holds the coefficients
// of stage `s` (row 0 is unused). The fifth-order solution is propagated.
const CK_C: [f64; 6] = [0.0, 0.2, 0.3, 0.6, 1.0, 7.0 / 8.0];
const CK_A: [[f64; 5]; 6] = [
    [0.0; 5],
    [0.2, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
    [0.3, -0.9, 1.2, 0.0, 0.0],
    [-11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0, 0.0],
    [
        1631.0 / 55296.0,
        175.0 / 512.0,
        575.0 / 13824.0,
        44275.0 / 110592.0,
        253.0 / 4096.0,
    ],
];
const CK_B: [f64; 6] = [
    37.0 / 378.0,
    0.0,
    250.0 / 621.0,
    125.0 / 594.0,
    0.0,
    512.0 / 1771.0,
];
// Fifth-order weights minus the embedded fourth-order weights.
const CK_E: [f64; 6] = [
    37.0 / 378.0 - 2825.0 / 27648.0,
    0.0,
    250.0 / 621.0 - 18575.0 / 48384.0,
    125.0 / 594.0 - 13525.0 / 55296.0,
    -277.0 / 14336.0,
    512.0 / 1771.0 - 0.25,
];

/// The explicit Runge Kutta pair used by [`rk45`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pair {
    DormandPrince,
    CashKarp,
}

/// `y + h Σ_j a_j k_j` for the nonzero coefficients `a`.
fn combine<T: Real>(y: &[T], h: f64, a: &[f64], k: &[Vec<T>]) -> Vec<T> {
    (0..y.len())
        .map(|i| {
            let mut acc = T::zero();
            for (aj, kj) in a.iter().zip(k) {
                if *aj != 0.0 {
                    acc += kj[i] * *aj;
                }
            }
            y[i] + acc * h
        })
        .collect()
}

/// One Dormand and Prince step from `(t, y)` with `k1 = f(t, y)`. Returns the new state,
/// `f(t + h, y_new)`, and the scaled error estimate computed from primal values.
#[allow(clippy::too_many_arguments)]
fn dopri_step<T: Real, F: OdeSystem>(
    f: &F,
    name: &'static str,
    t: f64,
    h: f64,
    y: &[T],
    k1: &[T],
    theta: &[T],
    o: &OdeOptions,
) -> Result<(Vec<T>, Vec<T>, f64)> {
    let mut k: Vec<Vec<T>> = Vec::with_capacity(7);
    k.push(k1.to_vec());
    for s in 1..6 {
        let ys = combine(y, h, &DP_A[s][..s], &k);
        k.push(rhs(f, name, t + DP_C[s] * h, &ys, theta)?);
    }
    let y_new = combine(y, h, &DP_A[6], &k);
    let k7 = rhs(f, name, t + h, &y_new, theta)?;
    k.push(k7);
    let n = y.len();
    let err: Vec<f64> = (0..n)
        .map(|i| h * (0..7).map(|j| DP_E[j] * k[j][i].value()).sum::<f64>())
        .collect();
    let yv: Vec<f64> = y.iter().map(Real::value).collect();
    let yn: Vec<f64> = y_new.iter().map(Real::value).collect();
    let e = rms(&err, &weights(o, &yv, &yn));
    let k7 = k.pop().unwrap_or_default();
    Ok((y_new, k7, e))
}

/// One Cash and Karp step from `(t, y)` with `k1 = f(t, y)`. Returns the new state and the scaled
/// error estimate computed from primal values.
#[allow(clippy::too_many_arguments)]
fn cash_karp_step<T: Real, F: OdeSystem>(
    f: &F,
    name: &'static str,
    t: f64,
    h: f64,
    y: &[T],
    k1: &[T],
    theta: &[T],
    o: &OdeOptions,
) -> Result<(Vec<T>, f64)> {
    let mut k: Vec<Vec<T>> = Vec::with_capacity(6);
    k.push(k1.to_vec());
    for s in 1..6 {
        let ys = combine(y, h, &CK_A[s][..s], &k);
        k.push(rhs(f, name, t + CK_C[s] * h, &ys, theta)?);
    }
    let y_new = combine(y, h, &CK_B, &k);
    let n = y.len();
    let err: Vec<f64> = (0..n)
        .map(|i| h * (0..6).map(|j| CK_E[j] * k[j][i].value()).sum::<f64>())
        .collect();
    let yv: Vec<f64> = y.iter().map(Real::value).collect();
    let yn: Vec<f64> = y_new.iter().map(Real::value).collect();
    let e = rms(&err, &weights(o, &yv, &yn));
    Ok((y_new, e))
}

/// Integrates with an explicit 5(4) pair and returns the states at `ts`.
#[allow(clippy::too_many_arguments)]
fn rk45<T: Real, F: OdeSystem>(
    pair: Pair,
    name: &'static str,
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    o: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    validate(name, y0, t0, ts, theta, o)?;
    let theta_v: Vec<f64> = theta.iter().map(Real::value).collect();
    let mut y = y0.to_vec();
    let mut t = t0;
    let mut k1 = rhs(f, name, t, &y, theta)?;
    let y0v: Vec<f64> = y.iter().map(Real::value).collect();
    let k1v: Vec<f64> = k1.iter().map(Real::value).collect();
    let mut h = initial_step(f, name, t0, &y0v, &k1v, &theta_v, ts[0] - t0, 4, o)?;
    let mut out = Vec::with_capacity(ts.len());
    for &t_out in ts {
        let mut steps = 0usize;
        while t < t_out {
            if steps >= o.max_num_steps {
                return Err(Error::NoConvergence {
                    function: name,
                    iterations: steps,
                });
            }
            steps += 1;
            let remaining = t_out - t;
            let last = h * (1.0 + 1e-10) >= remaining;
            let step = if last { remaining } else { h };
            let (y_new, k_next, err) = match pair {
                Pair::DormandPrince => dopri_step(f, name, t, step, &y, &k1, theta, o)?,
                Pair::CashKarp => {
                    let (y_new, err) = cash_karp_step(f, name, t, step, &y, &k1, theta, o)?;
                    (y_new, Vec::new(), err)
                }
            };
            if err <= 1.0 {
                t = if last { t_out } else { t + step };
                y = y_new;
                k1 = match pair {
                    Pair::DormandPrince => k_next,
                    Pair::CashKarp => rhs(f, name, t, &y, theta)?,
                };
                h = step * step_factor(err, 4, 5.0);
            } else {
                h = step * step_factor(err, 4, 1.0).min(0.9);
                check_step(name, t, h, steps)?;
            }
        }
        out.push(y.clone());
    }
    Ok(out)
}

/// Radau IIA (three stages, order 5): nodes `c`, coefficient matrix `a`, and `a⁻¹`.
pub(crate) struct Tableau {
    c: [f64; 3],
    a: [[f64; 3]; 3],
    a_inv: [[f64; 3]; 3],
}

impl Tableau {
    pub(crate) fn radau_iia() -> Self {
        let s6 = 6f64.sqrt();
        let c = [(4.0 - s6) / 10.0, (4.0 + s6) / 10.0, 1.0];
        let a = [
            [
                (88.0 - 7.0 * s6) / 360.0,
                (296.0 - 169.0 * s6) / 1800.0,
                (-2.0 + 3.0 * s6) / 225.0,
            ],
            [
                (296.0 + 169.0 * s6) / 1800.0,
                (88.0 + 7.0 * s6) / 360.0,
                (-2.0 - 3.0 * s6) / 225.0,
            ],
            [(16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0],
        ];
        Tableau {
            c,
            a,
            a_inv: invert3(&a),
        }
    }
}

/// Inverse of a 3 by 3 matrix by cofactors.
fn invert3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let cof =
        |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let det = m[0][0] * cof(1, 2, 1, 2) - m[0][1] * cof(1, 2, 0, 2) + m[0][2] * cof(1, 2, 0, 1);
    // adjugate: transpose of the cofactor matrix
    [
        [
            cof(1, 2, 1, 2) / det,
            -cof(0, 2, 1, 2) / det,
            cof(0, 1, 1, 2) / det,
        ],
        [
            -cof(1, 2, 0, 2) / det,
            cof(0, 2, 0, 2) / det,
            -cof(0, 1, 0, 2) / det,
        ],
        [
            cof(1, 2, 0, 1) / det,
            -cof(0, 2, 0, 1) / det,
            cof(0, 1, 0, 1) / det,
        ],
    ]
}

/// The implicit stage equations of a Radau IIA step for some problem class (ODE or DAE). The
/// unknowns are the stage increments `Z_i = Y_i - y`, stacked.
pub(crate) trait Stages {
    /// One full Newton step on the stage equations from `(t, y)` with step `h`, updating `z`.
    /// Returns the largest absolute change in the primal values, or `None` when the Newton matrix
    /// is singular or the residual is not finite.
    fn newton<T: Real>(
        &self,
        t: f64,
        h: f64,
        y: &[T],
        z: &mut [T],
        theta: &[T],
        tab: &Tableau,
    ) -> Result<Option<f64>>;
}

/// Applies a Newton correction `z -= J⁻¹ g` and reports the largest primal change.
fn apply_newton<T: Real>(
    name: &'static str,
    jac: Vec<Vec<T>>,
    g: Vec<T>,
    z: &mut [T],
) -> Option<f64> {
    if g.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let step = solve_linear(name, jac, g).ok()?;
    let mut size = 0.0f64;
    for (zi, s) in z.iter_mut().zip(&step) {
        size = size.max(s.value().abs());
        *zi -= *s;
    }
    size.is_finite().then_some(size)
}

/// Radau stages of `dy/dt = f(t, y, θ)`: `Z_i - h Σ_j a_ij f(t + c_j h, y + Z_j) = 0`.
struct OdeStages<'a, F> {
    f: &'a F,
    name: &'static str,
}

impl<F: OdeSystem> OdeStages<'_, F> {
    #[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
    fn step<T: Real>(
        &self,
        t: f64,
        h: f64,
        y: &[T],
        z: &mut [T],
        theta: &[T],
        tab: &Tableau,
    ) -> Result<Option<f64>> {
        let n = y.len();
        let m = 3 * n;
        let mut fs = Vec::with_capacity(3);
        let mut js = Vec::with_capacity(3);
        for j in 0..3 {
            let yj: Vec<T> = (0..n).map(|i| y[i] + z[j * n + i]).collect();
            let (fj, jj) = rhs_jacobian(self.f, self.name, t + tab.c[j] * h, &yj, theta)?;
            fs.push(fj);
            js.push(jj);
        }
        let mut g = vec![T::zero(); m];
        let mut jac = vec![vec![T::zero(); m]; m];
        for i in 0..3 {
            for r in 0..n {
                let mut acc = T::zero();
                for j in 0..3 {
                    acc += fs[j][r] * tab.a[i][j];
                }
                g[i * n + r] = z[i * n + r] - acc * h;
                for j in 0..3 {
                    for col in 0..n {
                        let mut v = -(js[j][r][col] * (tab.a[i][j] * h));
                        if i == j && r == col {
                            v += 1.0;
                        }
                        jac[i * n + r][j * n + col] = v;
                    }
                }
            }
        }
        Ok(apply_newton(self.name, jac, g, z))
    }
}

/// Radau stages of the implicit system `F(t, y, y', θ) = 0`: with `W = a⁻¹`, the stage derivatives
/// are `Y'_i = h⁻¹ Σ_j W_ij Z_j` and the equations are `F(t + c_i h, y + Z_i, Y'_i) = 0`.
struct DaeStages<'a, F> {
    f: &'a F,
    name: &'static str,
}

impl<F: DaeSystem> DaeStages<'_, F> {
    /// Residuals of all stages at stacked increments `z`.
    #[allow(clippy::too_many_arguments)]
    fn residual<S: Real>(
        &self,
        t: f64,
        h: f64,
        y: &[S],
        z: &[S],
        theta: &[S],
        tab: &Tableau,
    ) -> Result<Vec<S>> {
        let n = y.len();
        let mut out = Vec::with_capacity(3 * n);
        for i in 0..3 {
            let yi: Vec<S> = (0..n).map(|r| y[r] + z[i * n + r]).collect();
            let ypi: Vec<S> = (0..n)
                .map(|r| {
                    let mut acc = S::zero();
                    for j in 0..3 {
                        acc += z[j * n + r] * tab.a_inv[i][j];
                    }
                    acc / h
                })
                .collect();
            let fi = self.f.eval(S::from_f64(t + tab.c[i] * h), &yi, &ypi, theta);
            if fi.len() != n {
                return Err(Error::dimension(
                    self.name,
                    format!("the residual returned {} values for {} states", fi.len(), n),
                ));
            }
            out.extend(fi);
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn step<T: Real>(
        &self,
        t: f64,
        h: f64,
        y: &[T],
        z: &mut [T],
        theta: &[T],
        tab: &Tableau,
    ) -> Result<Option<f64>> {
        let m = z.len();
        let yd: Vec<Dual<T>> = y.iter().map(|&v| Dual::constant(v)).collect();
        let th: Vec<Dual<T>> = theta.iter().map(|&v| Dual::constant(v)).collect();
        let mut g = Vec::new();
        let mut jac = vec![vec![T::zero(); m]; m];
        for col in 0..m {
            let zd: Vec<Dual<T>> = z
                .iter()
                .enumerate()
                .map(|(k, &v)| {
                    if k == col {
                        Dual::variable(v)
                    } else {
                        Dual::constant(v)
                    }
                })
                .collect();
            let r = self.residual(t, h, &yd, &zd, &th, tab)?;
            if col == 0 {
                g = r.iter().map(|d| d.re).collect();
            }
            for (row, d) in r.iter().enumerate() {
                jac[row][col] = d.eps;
            }
        }
        Ok(apply_newton(self.name, jac, g, z))
    }
}

impl<F: OdeSystem> Stages for OdeStages<'_, F> {
    fn newton<T: Real>(
        &self,
        t: f64,
        h: f64,
        y: &[T],
        z: &mut [T],
        theta: &[T],
        tab: &Tableau,
    ) -> Result<Option<f64>> {
        self.step(t, h, y, z, theta, tab)
    }
}

impl<F: DaeSystem> Stages for DaeStages<'_, F> {
    fn newton<T: Real>(
        &self,
        t: f64,
        h: f64,
        y: &[T],
        z: &mut [T],
        theta: &[T],
        tab: &Tableau,
    ) -> Result<Option<f64>> {
        self.step(t, h, y, z, theta, tab)
    }
}

/// Solves the Radau stage equations in `f64` by Newton's method. `None` when Newton's method
/// does not converge, which makes the caller reject the step.
fn radau_stages<P: Stages>(
    p: &P,
    t: f64,
    h: f64,
    y: &[f64],
    theta: &[f64],
    tab: &Tableau,
) -> Result<Option<Vec<f64>>> {
    const MAX_ITERATIONS: usize = 12;
    let n = y.len();
    let mut z = vec![0.0; 3 * n];
    let mut previous = f64::INFINITY;
    for iteration in 0..MAX_ITERATIONS {
        let Some(size) = p.newton(t, h, y, &mut z, theta, tab)? else {
            return Ok(None);
        };
        let scale = y.iter().chain(&z).fold(1.0f64, |acc, v| acc.max(v.abs()));
        if size <= 64.0 * f64::EPSILON * scale {
            return Ok(Some(z));
        }
        // Rounding error floor reached: the change stopped shrinking but is already tiny.
        if iteration >= 2 && size > 0.5 * previous && size <= 1e-10 * scale {
            return Ok(Some(z));
        }
        previous = size;
    }
    Ok(None)
}

/// The end state `y + Z_3` of a Radau step (the method is stiffly accurate).
fn radau_end<T: Real>(y: &[T], z: &[T]) -> Vec<T> {
    let n = y.len();
    (0..n).map(|i| y[i] + z[2 * n + i]).collect()
}

/// One Radau step of size `h` and two of size `h / 2` from the same state, for error estimation
/// by step doubling.
struct DoubledStep {
    y_big: Vec<f64>,
    z_first: Vec<f64>,
    z_second: Vec<f64>,
    y_two: Vec<f64>,
}

/// Takes the steps of a [`DoubledStep`]; `None` when a stage solve fails.
fn radau_doubled<P: Stages>(
    p: &P,
    t: f64,
    h: f64,
    y: &[f64],
    theta: &[f64],
    tab: &Tableau,
) -> Result<Option<DoubledStep>> {
    let half = 0.5 * h;
    let Some(z_big) = radau_stages(p, t, h, y, theta, tab)? else {
        return Ok(None);
    };
    let y_big = radau_end(y, &z_big);
    let Some(z_first) = radau_stages(p, t, half, y, theta, tab)? else {
        return Ok(None);
    };
    let y_mid = radau_end(y, &z_first);
    let Some(z_second) = radau_stages(p, t + half, half, &y_mid, theta, tab)? else {
        return Ok(None);
    };
    let y_two = radau_end(&y_mid, &z_second);
    Ok(Some(DoubledStep {
        y_big,
        z_first,
        z_second,
        y_two,
    }))
}

/// An accepted Radau step: start time, step size, and converged `f64` stages.
struct RadauStep {
    t: f64,
    h: f64,
    z: Vec<f64>,
}

/// Integrates with Radau IIA from `y0` (already validated) with starting step `h0`, and returns
/// the states at `ts`.
#[allow(clippy::too_many_arguments)]
fn radau<T: Real, P: Stages>(
    name: &'static str,
    p: &P,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    h0: f64,
    o: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    let tab = Tableau::radau_iia();
    let theta_v: Vec<f64> = theta.iter().map(Real::value).collect();

    // Pass 1, in f64: choose the steps by step doubling and record the converged stages.
    let mut y: Vec<f64> = y0.iter().map(Real::value).collect();
    let mut t = t0;
    let mut h = h0;
    let mut steps: Vec<RadauStep> = Vec::new();
    let mut outputs_after: Vec<usize> = Vec::with_capacity(ts.len());
    let mut values: Vec<Vec<f64>> = Vec::with_capacity(ts.len());
    for &t_out in ts {
        let mut count = 0usize;
        while t < t_out {
            if count >= o.max_num_steps {
                return Err(Error::NoConvergence {
                    function: name,
                    iterations: count,
                });
            }
            count += 1;
            let remaining = t_out - t;
            let last = h * (1.0 + 1e-10) >= remaining;
            let big_h = if last { remaining } else { h };
            let half = 0.5 * big_h;
            let Some(DoubledStep {
                y_big,
                z_first,
                z_second,
                y_two,
            }) = radau_doubled(p, t, big_h, &y, &theta_v, &tab)?
            else {
                h = 0.25 * big_h;
                check_step(name, t, h, count)?;
                continue;
            };
            // Richardson: the error of the two half steps is about (y_two - y_big) / (2^5 - 1).
            let diff: Vec<f64> = y_two
                .iter()
                .zip(&y_big)
                .map(|(a, b)| (a - b) / 31.0)
                .collect();
            let err = rms(&diff, &weights(o, &y, &y_two));
            if err <= 1.0 {
                steps.push(RadauStep {
                    t,
                    h: half,
                    z: z_first,
                });
                steps.push(RadauStep {
                    t: t + half,
                    h: half,
                    z: z_second,
                });
                t = if last { t_out } else { t + big_h };
                y = y_two;
                h = big_h * step_factor(err, 5, 4.0);
            } else {
                h = big_h * step_factor(err, 5, 1.0).min(0.9);
                check_step(name, t, h, count)?;
            }
        }
        outputs_after.push(steps.len());
        values.push(y.clone());
    }
    if T::ORDER == 0 {
        return Ok(values
            .into_iter()
            .map(|v| v.into_iter().map(T::from_f64).collect())
            .collect());
    }

    // Pass 2, in T: replay the steps, refining each stage solution by Newton steps in T so that
    // its derivatives are exact.
    let mut y: Vec<T> = y0.to_vec();
    let mut out = Vec::with_capacity(ts.len());
    let mut next = 0;
    for (index, step) in steps.iter().enumerate() {
        let mut z: Vec<T> = step.z.iter().map(|&v| T::from_f64(v)).collect();
        for _ in 0..newton_steps(T::ORDER) {
            if p.newton(step.t, step.h, &y, &mut z, theta, &tab)?.is_none() {
                return Err(Error::parameter(
                    name,
                    "Newton matrix",
                    step.t,
                    "nonsingular",
                ));
            }
        }
        y = radau_end(&y, &z);
        while next < outputs_after.len() && outputs_after[next] == index + 1 {
            out.push(y.clone());
            next += 1;
        }
    }
    Ok(out)
}

/// Validates, picks a starting step, and runs [`radau`] on an ODE.
fn ode_radau<T: Real, F: OdeSystem>(
    name: &'static str,
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    o: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    validate(name, y0, t0, ts, theta, o)?;
    let theta_v: Vec<f64> = theta.iter().map(Real::value).collect();
    let y0v: Vec<f64> = y0.iter().map(Real::value).collect();
    let f0 = rhs(f, name, t0, &y0v, &theta_v)?;
    let h0 = initial_step(f, name, t0, &y0v, &f0, &theta_v, ts[0] - t0, 5, o)?;
    radau(name, &OdeStages { f, name }, y0, t0, ts, theta, h0, o)
}

/// A fully implicit differential algebraic system `F(t, y, y', θ) = 0` of index one.
pub trait DaeSystem {
    /// Evaluates the residual `F(t, y, y', θ)`; the result must have the same length as `y`.
    fn eval<S: Real>(&self, t: S, y: &[S], yp: &[S], theta: &[S]) -> Vec<S>;
}

/// Solves the index one system `F(t, y, y', θ) = 0` from consistent initial values `y(t0) = y0`,
/// `y'(t0) = yp0`, and returns `y` at each time in `ts`, using the Radau IIA method with
/// [`OdeOptions::STIFF`].
///
/// Stan Math uses IDAS for `dae`; results agree to the tolerance. `yp0` is used only to choose the
/// first step: the method does not need it, and its consistency is not checked. Derivatives with
/// respect to `yp0` are therefore zero.
///
/// # Errors
///
/// As [`ode_bdf`].
pub fn dae<T: Real, F: DaeSystem>(
    f: &F,
    y0: &[T],
    yp0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
) -> Result<Vec<Vec<T>>> {
    dae_radau("dae", f, y0, yp0, t0, ts, theta, &OdeOptions::STIFF)
}

/// [`dae`] with explicit tolerances and step limit (Stan's `dae_tol`).
///
/// # Errors
///
/// As [`dae`], plus invalid tolerances.
pub fn dae_tol<T: Real, F: DaeSystem>(
    f: &F,
    y0: &[T],
    yp0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    options: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    dae_radau("dae_tol", f, y0, yp0, t0, ts, theta, options)
}

#[allow(clippy::too_many_arguments)]
fn dae_radau<T: Real, F: DaeSystem>(
    name: &'static str,
    f: &F,
    y0: &[T],
    yp0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    o: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    validate(name, y0, t0, ts, theta, o)?;
    if yp0.len() != y0.len() {
        return Err(Error::dimension(
            name,
            format!("{} initial derivatives for {} states", yp0.len(), y0.len()),
        ));
    }
    if let Some(v) = yp0.iter().find(|v| !v.is_finite()) {
        return Err(Error::parameter(
            name,
            "initial derivative",
            v.value(),
            "finite",
        ));
    }
    let y0v: Vec<f64> = y0.iter().map(Real::value).collect();
    let yp0v: Vec<f64> = yp0.iter().map(Real::value).collect();
    let w = weights(o, &y0v, &y0v);
    let (d0, d1) = (rms(&y0v, &w), rms(&yp0v, &w));
    let span = ts[0] - t0;
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };
    radau(
        name,
        &DaeStages { f, name },
        y0,
        t0,
        ts,
        theta,
        h0.min(span),
        o,
    )
}

/// Solves `dy/dt = f(t, y, θ)` from `y(t0) = y0` and returns `y` at each time in `ts`, using the
/// Dormand and Prince 5(4) pair with [`OdeOptions::NONSTIFF`].
///
/// `ts` must be sorted and its first time greater than `t0`. Derivatives with respect to `y0` and
/// `θ` follow the module documentation.
///
/// # Errors
///
/// [`Error::InvalidParameter`] for invalid arguments, [`Error::DimensionMismatch`] when `f`
/// returns the wrong number of values, and [`Error::NoConvergence`] when `max_num_steps` is
/// exceeded or the step size underflows.
///
/// ```
/// use stanmath_rs_autodiff::Real;
/// use stanmath_rs_solvers::{OdeSystem, ode_rk45};
///
/// /// dy/dt = -k y
/// struct Decay;
/// impl OdeSystem for Decay {
///     fn eval<S: Real>(&self, _t: S, y: &[S], theta: &[S]) -> Vec<S> {
///         vec![-(theta[0] * y[0])]
///     }
/// }
///
/// let y = ode_rk45(&Decay, &[1.0], 0.0, &[1.0], &[0.5]).unwrap();
/// assert!((y[0][0] - (-0.5f64).exp()).abs() < 1e-5);
/// ```
pub fn ode_rk45<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
) -> Result<Vec<Vec<T>>> {
    rk45(
        Pair::DormandPrince,
        "ode_rk45",
        f,
        y0,
        t0,
        ts,
        theta,
        &OdeOptions::NONSTIFF,
    )
}

/// [`ode_rk45`] with explicit tolerances and step limit (Stan's `ode_rk45_tol`).
///
/// # Errors
///
/// As [`ode_rk45`], plus invalid tolerances.
pub fn ode_rk45_tol<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    options: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    rk45(
        Pair::DormandPrince,
        "ode_rk45_tol",
        f,
        y0,
        t0,
        ts,
        theta,
        options,
    )
}

/// Stan's `ode_ckrk`: the Cash and Karp 5(4) pair with [`OdeOptions::NONSTIFF`]. Step size
/// control is ours (that of [`ode_rk45`]), so results agree with Stan to the tolerance.
///
/// # Errors
///
/// As [`ode_rk45`].
pub fn ode_ckrk<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
) -> Result<Vec<Vec<T>>> {
    rk45(
        Pair::CashKarp,
        "ode_ckrk",
        f,
        y0,
        t0,
        ts,
        theta,
        &OdeOptions::NONSTIFF,
    )
}

/// [`ode_ckrk`] with explicit tolerances and step limit.
///
/// # Errors
///
/// As [`ode_rk45_tol`].
pub fn ode_ckrk_tol<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    options: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    rk45(
        Pair::CashKarp,
        "ode_ckrk_tol",
        f,
        y0,
        t0,
        ts,
        theta,
        options,
    )
}

/// Stan's `ode_adams` (CVODES Adams in Stan Math), computed with the Dormand and Prince pair and
/// [`OdeOptions::STIFF`] tolerances. Results agree with Stan to the tolerance.
///
/// # Errors
///
/// As [`ode_rk45`].
pub fn ode_adams<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
) -> Result<Vec<Vec<T>>> {
    rk45(
        Pair::DormandPrince,
        "ode_adams",
        f,
        y0,
        t0,
        ts,
        theta,
        &OdeOptions::STIFF,
    )
}

/// [`ode_adams`] with explicit tolerances and step limit.
///
/// # Errors
///
/// As [`ode_rk45_tol`].
pub fn ode_adams_tol<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    options: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    rk45(
        Pair::DormandPrince,
        "ode_adams_tol",
        f,
        y0,
        t0,
        ts,
        theta,
        options,
    )
}

/// Solves a stiff system `dy/dt = f(t, y, θ)` with the Radau IIA method and
/// [`OdeOptions::STIFF`], returning `y` at each time in `ts`.
///
/// The Jacobian of `f` is computed by forward-mode differentiation, so `f` needs no extra code.
/// The cost per step grows with the cube of the number of states.
///
/// # Errors
///
/// As [`ode_rk45`], and [`Error::InvalidParameter`] when the Newton matrix of an accepted step is
/// singular in the derivative pass.
pub fn ode_bdf<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
) -> Result<Vec<Vec<T>>> {
    ode_radau("ode_bdf", f, y0, t0, ts, theta, &OdeOptions::STIFF)
}

/// [`ode_bdf`] with explicit tolerances and step limit (Stan's `ode_bdf_tol`).
///
/// # Errors
///
/// As [`ode_bdf`], plus invalid tolerances.
pub fn ode_bdf_tol<T: Real, F: OdeSystem>(
    f: &F,
    y0: &[T],
    t0: f64,
    ts: &[f64],
    theta: &[T],
    options: &OdeOptions,
) -> Result<Vec<Vec<T>>> {
    ode_radau("ode_bdf_tol", f, y0, t0, ts, theta, options)
}
