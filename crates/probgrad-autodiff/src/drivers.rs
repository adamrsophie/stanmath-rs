//! High-level differentiation drivers.
//!
//! Reverse-mode drivers take a closure over [`Var`] slices. The closure must work for a tape of
//! any lifetime (`for<'t> Fn(&[Var<'t>]) -> Var<'t>`), which is what lets the driver create the
//! tape internally and makes it a compile error to smuggle in variables from another tape:
//!
//! ```compile_fail
//! use probgrad_autodiff::{grad, Tape};
//!
//! let outer = Tape::new();
//! let w = outer.var(2.0);
//! // `w` lives on `outer`, but the closure must accept variables from any tape.
//! let g = grad(|x| x[0] * w, &[1.0]);
//! ```
//!
//! A model written once as a generic function serves every driver; pass it through a closure:
//!
//! ```
//! use probgrad_autodiff::{grad, hessian, Real};
//!
//! fn rosenbrock<T: Real>(x: &[T]) -> T {
//!     (-x[0] + 1.0).square() + (x[1] - x[0].square()).square() * 100.0
//! }
//!
//! let x = [1.0, 1.0];
//! assert_eq!(grad(|v| rosenbrock(v), &x), vec![0.0, 0.0]);
//! let h = hessian(|v| rosenbrock(v), &x);
//! assert_eq!(h[(0, 0)], 802.0);
//! assert_eq!(h[(0, 1)], -400.0);
//! ```

use probgrad_core::Matrix;

use crate::dual::{Dual, Dual64};
use crate::real::Real;
use crate::tape::{Tape, Var};

/// A function value and its gradient.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueAndGrad {
    /// `f(x)`.
    pub value: f64,
    /// `∇f(x)`.
    pub gradient: Vec<f64>,
}

/// Gradient of a scalar function by reverse mode.
pub fn grad<F>(f: F, x: &[f64]) -> Vec<f64>
where
    F: for<'t> FnOnce(&[Var<'t>]) -> Var<'t>,
{
    value_and_grad(f, x).gradient
}

/// Value and gradient of a scalar function by reverse mode.
///
/// ```
/// use probgrad_autodiff::{value_and_grad, Real};
///
/// let r = value_and_grad(|p| p[0] * p[1].exp(), &[2.0, 0.0]);
/// assert_eq!(r.value, 2.0);
/// assert_eq!(r.gradient, vec![1.0, 2.0]);
/// ```
pub fn value_and_grad<F>(f: F, x: &[f64]) -> ValueAndGrad
where
    F: for<'t> FnOnce(&[Var<'t>]) -> Var<'t>,
{
    let tape = Tape::new();
    let xs = tape.vars(x);
    let y = f(&xs);
    let g = tape.gradient(y);
    ValueAndGrad {
        value: y.value(),
        gradient: g.wrt_slice(&xs),
    }
}

/// [`grad`] for closures that can fail, such as log densities that validate their arguments.
pub fn try_grad<F, E>(f: F, x: &[f64]) -> Result<Vec<f64>, E>
where
    F: for<'t> FnOnce(&[Var<'t>]) -> Result<Var<'t>, E>,
{
    try_value_and_grad(f, x).map(|r| r.gradient)
}

/// [`value_and_grad`] for closures that can fail.
///
/// ```
/// use probgrad_autodiff::{try_value_and_grad, Real};
///
/// let r = try_value_and_grad(
///     |p| if p[0] > 0.0 { Ok(p[0].ln()) } else { Err("x must be positive") },
///     &[2.0],
/// );
/// assert_eq!(r.unwrap().gradient, vec![0.5]);
/// ```
pub fn try_value_and_grad<F, E>(f: F, x: &[f64]) -> Result<ValueAndGrad, E>
where
    F: for<'t> FnOnce(&[Var<'t>]) -> Result<Var<'t>, E>,
{
    let tape = Tape::new();
    let xs = tape.vars(x);
    let y = f(&xs)?;
    let g = tape.gradient(y);
    Ok(ValueAndGrad {
        value: y.value(),
        gradient: g.wrt_slice(&xs),
    })
}

/// Reusable workspace for repeated gradient evaluations, for example inside a sampler.
///
/// Keeps the tape and adjoint buffers between calls so that steady-state evaluation does not
/// reallocate them, and writes the gradient into a caller-provided slice. The signature of
/// [`GradientWorkspace::try_value_and_grad_into`] matches the `logp(position, gradient)` shape used
/// by NUTS implementations such as `nuts-rs`.
#[derive(Debug, Default)]
pub struct GradientWorkspace {
    tape: Tape,
    adjoints: Vec<f64>,
}

impl GradientWorkspace {
    /// An empty workspace.
    pub fn new() -> Self {
        GradientWorkspace::default()
    }

    /// Evaluates `f(x)`, writes `∇f(x)` into `gradient`, and returns `f(x)`.
    ///
    /// # Panics
    ///
    /// Panics if `gradient.len() != x.len()`.
    pub fn value_and_grad_into<F>(&mut self, f: F, x: &[f64], gradient: &mut [f64]) -> f64
    where
        F: for<'t> FnOnce(&[Var<'t>]) -> Var<'t>,
    {
        match self.try_value_and_grad_into(
            |v| Ok::<_, core::convert::Infallible>(f(v)),
            x,
            gradient,
        ) {
            Ok(v) => v,
            Err(never) => match never {},
        }
    }

    /// Fallible version of [`GradientWorkspace::value_and_grad_into`].
    pub fn try_value_and_grad_into<F, E>(
        &mut self,
        f: F,
        x: &[f64],
        gradient: &mut [f64],
    ) -> Result<f64, E>
    where
        F: for<'t> FnOnce(&[Var<'t>]) -> Result<Var<'t>, E>,
    {
        assert_eq!(
            gradient.len(),
            x.len(),
            "GradientWorkspace: gradient buffer has the wrong length"
        );
        self.tape.clear();
        let tape = &self.tape;
        let xs = tape.vars(x);
        let y = f(&xs)?;
        tape.gradient_into(y, &mut self.adjoints);
        // Independent variables are the first nodes on the fresh tape.
        gradient.copy_from_slice(&self.adjoints[..x.len()]);
        Ok(y.value())
    }
}

/// Derivative of a scalar function of one variable by forward mode.
///
/// Forward-mode closures have no tape lifetime, so a generic function can be passed directly.
///
/// ```
/// use probgrad_autodiff::{derivative, Real};
///
/// fn f<T: Real>(x: T) -> T { x.sin() * x }
/// let d = derivative(f, 1.0);
/// assert!((d - (1.0f64.cos() + 1.0f64.sin())).abs() < 1e-15);
/// ```
pub fn derivative<F>(f: F, x: f64) -> f64
where
    F: FnOnce(Dual64) -> Dual64,
{
    f(Dual::variable(x)).eps
}

/// Value and derivative of a scalar function of one variable.
pub fn value_and_derivative<F>(f: F, x: f64) -> (f64, f64)
where
    F: FnOnce(Dual64) -> Dual64,
{
    let y = f(Dual::variable(x));
    (y.re, y.eps)
}

/// Second derivative of a scalar function of one variable by forward over forward mode.
pub fn second_derivative<F>(f: F, x: f64) -> f64
where
    F: FnOnce(Dual<Dual64>) -> Dual<Dual64>,
{
    let xd = Dual::new(Dual::variable(x), Dual::constant(1.0));
    f(xd).eps.eps
}

/// Directional derivative `∇f(x) · v` of a scalar function by forward mode.
///
/// # Panics
///
/// Panics if `x` and `v` differ in length.
pub fn directional_derivative<F>(f: F, x: &[f64], v: &[f64]) -> f64
where
    F: FnOnce(&[Dual64]) -> Dual64,
{
    assert_eq!(
        x.len(),
        v.len(),
        "directional_derivative: x and v differ in length"
    );
    let xs: Vec<Dual64> = x.iter().zip(v).map(|(&a, &b)| Dual::new(a, b)).collect();
    f(&xs).eps
}

/// Jacobian-vector product `J(x) v` of a vector function by forward mode.
///
/// Returns `(f(x), J(x) v)`.
///
/// # Panics
///
/// Panics if `x` and `v` differ in length.
pub fn jvp<F>(f: F, x: &[f64], v: &[f64]) -> (Vec<f64>, Vec<f64>)
where
    F: FnOnce(&[Dual64]) -> Vec<Dual64>,
{
    assert_eq!(x.len(), v.len(), "jvp: x and v differ in length");
    let xs: Vec<Dual64> = x.iter().zip(v).map(|(&a, &b)| Dual::new(a, b)).collect();
    f(&xs).into_iter().map(|y| (y.re, y.eps)).unzip()
}

/// Vector-Jacobian product `wᵀ J(x)` of a vector function by reverse mode.
///
/// Returns `(f(x), wᵀ J(x))`.
///
/// # Panics
///
/// Panics if the number of outputs differs from `w.len()`.
pub fn vjp<F>(f: F, x: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>)
where
    F: for<'t> FnOnce(&[Var<'t>]) -> Vec<Var<'t>>,
{
    let tape = Tape::new();
    let xs = tape.vars(x);
    let ys = f(&xs);
    let g = tape.vjp(&ys, w);
    (ys.iter().map(Real::value).collect(), g.wrt_slice(&xs))
}

/// Jacobian of a vector function by reverse mode: one evaluation and one sweep per output.
///
/// Returns `(f(x), J(x))` with `J[(i, j)] = d f_i / d x_j`. Prefer this when there are fewer
/// outputs than inputs; otherwise use [`jacobian_forward`].
pub fn jacobian<F>(f: F, x: &[f64]) -> (Vec<f64>, Matrix<f64>)
where
    F: for<'t> FnOnce(&[Var<'t>]) -> Vec<Var<'t>>,
{
    let tape = Tape::new();
    let xs = tape.vars(x);
    let ys = f(&xs);
    let mut jac = Matrix::zeros(ys.len(), x.len());
    let mut adjoints = Vec::new();
    for (i, &y) in ys.iter().enumerate() {
        tape.gradient_into(y, &mut adjoints);
        for j in 0..x.len() {
            jac[(i, j)] = adjoints[j];
        }
    }
    (ys.iter().map(Real::value).collect(), jac)
}

/// Jacobian of a vector function by forward mode: one evaluation per input.
///
/// Returns `(f(x), J(x))`.
pub fn jacobian_forward<F>(f: F, x: &[f64]) -> (Vec<f64>, Matrix<f64>)
where
    F: Fn(&[Dual64]) -> Vec<Dual64>,
{
    let n = x.len();
    let mut values = Vec::new();
    let mut columns = Vec::with_capacity(n);
    let mut xs: Vec<Dual64> = x.iter().map(|&a| Dual::constant(a)).collect();
    for j in 0..n {
        xs[j].eps = 1.0;
        let ys = f(&xs);
        xs[j].eps = 0.0;
        if j == 0 {
            values = ys.iter().map(|y| y.re).collect();
        }
        columns.push(ys.into_iter().map(|y| y.eps).collect::<Vec<_>>());
    }
    if n == 0 {
        values = f(&xs).iter().map(|y| y.re).collect();
    }
    let m = values.len();
    (values, Matrix::from_fn(m, n, |i, j| columns[j][i]))
}

/// Value, gradient, and Hessian of a scalar function.
#[derive(Clone, Debug, PartialEq)]
pub struct ValueGradHessian {
    /// `f(x)`.
    pub value: f64,
    /// `∇f(x)`.
    pub gradient: Vec<f64>,
    /// `∇²f(x)`.
    pub hessian: Matrix<f64>,
}

/// Hessian by forward over reverse mode: one evaluation and sweep per input.
///
/// The closure receives `Dual<Var>` arguments; pass a generic model through a closure,
/// `hessian(|x| model(x), &x0)`.
pub fn hessian<F>(f: F, x: &[f64]) -> Matrix<f64>
where
    F: for<'t> Fn(&[Dual<Var<'t>>]) -> Dual<Var<'t>>,
{
    value_grad_hessian(f, x).hessian
}

/// Value, gradient, and Hessian by forward over reverse mode.
pub fn value_grad_hessian<F>(f: F, x: &[f64]) -> ValueGradHessian
where
    F: for<'t> Fn(&[Dual<Var<'t>>]) -> Dual<Var<'t>>,
{
    let n = x.len();
    let mut hess = Matrix::zeros(n, n);
    let mut value = f64::NAN;
    let mut gradient = vec![0.0; n];
    let mut tape = Tape::new();
    let mut adjoints = Vec::new();
    for j in 0..n {
        tape.clear();
        let xs = tape.vars(x);
        let duals: Vec<Dual<Var<'_>>> = xs
            .iter()
            .enumerate()
            .map(|(i, &v)| Dual::new(v, Var::constant(if i == j { 1.0 } else { 0.0 })))
            .collect();
        let y = f(&duals);
        tape.gradient_into(y.eps, &mut adjoints);
        for i in 0..n {
            hess[(i, j)] = adjoints[i];
        }
        if j == 0 {
            value = y.re.value();
            tape.gradient_into(y.re, &mut adjoints);
            gradient.copy_from_slice(&adjoints[..n]);
        }
    }
    if n == 0 {
        value = f(&[]).re.value();
    }
    ValueGradHessian {
        value,
        gradient,
        hessian: hess,
    }
}

/// Hessian-vector product `∇²f(x) v` by forward over reverse mode in a single evaluation.
///
/// # Panics
///
/// Panics if `x` and `v` differ in length.
pub fn hessian_vector_product<F>(f: F, x: &[f64], v: &[f64]) -> Vec<f64>
where
    F: for<'t> FnOnce(&[Dual<Var<'t>>]) -> Dual<Var<'t>>,
{
    assert_eq!(
        x.len(),
        v.len(),
        "hessian_vector_product: x and v differ in length"
    );
    let tape = Tape::new();
    let xs = tape.vars(x);
    let duals: Vec<Dual<Var<'_>>> = xs
        .iter()
        .zip(v)
        .map(|(&xi, &vi)| Dual::new(xi, Var::constant(vi)))
        .collect();
    let y = f(&duals);
    tape.gradient(y.eps).wrt_slice(&xs)
}

/// Hessian by forward over forward mode: `n (n + 1) / 2` evaluations.
///
/// Slower than [`hessian`] for large `n` but independent of the tape; useful for cross-checking.
pub fn hessian_forward<F>(f: F, x: &[f64]) -> Matrix<f64>
where
    F: Fn(&[Dual<Dual64>]) -> Dual<Dual64>,
{
    let n = x.len();
    let mut hess = Matrix::zeros(n, n);
    let mut xs: Vec<Dual<Dual64>> = x
        .iter()
        .map(|&a| Dual::constant(Dual::constant(a)))
        .collect();
    for i in 0..n {
        for j in i..n {
            xs[i].re.eps = 1.0;
            xs[j].eps.re = 1.0;
            let h = f(&xs).eps.eps;
            xs[i].re.eps = 0.0;
            xs[j].eps.re = 0.0;
            hess[(i, j)] = h;
            hess[(j, i)] = h;
        }
    }
    hess
}
