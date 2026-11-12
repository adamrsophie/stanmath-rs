//! Further differentiation drivers with Stan Math names: third derivatives, directional products,
//! and finite-difference Hessians for checking.
//!
//! Closure conventions follow [`crate::drivers`]: reverse-mode closures must accept variables
//! from a tape of any lifetime, and a generic model is passed through a closure.

use stanmath_rs_core::Matrix;

use crate::check::finite_diff_gradient;
use crate::drivers::{ValueGradHessian, value_grad_hessian};
use crate::dual::{Dual, Dual64};
use crate::real::Real;
use crate::tape::{Tape, Var};

/// Value, Hessian, and the gradient of every Hessian entry.
#[derive(Clone, Debug, PartialEq)]
pub struct GradHessian {
    /// `f(x)`.
    pub value: f64,
    /// `∇²f(x)`.
    pub hessian: Matrix<f64>,
    /// `grad_hessian[i][(j, k)] = ∂³f / ∂x_i ∂x_j ∂x_k`.
    pub grad_hessian: Vec<Matrix<f64>>,
}

/// Value, Hessian, and third derivatives by forward over forward over reverse mode.
///
/// One evaluation and one reverse sweep per pair `j <= k`: the two tangents select `x_j` and
/// `x_k`, and the sweep of the second-order tangent gives `∂_i ∂_j ∂_k f` for every `i`.
pub fn grad_hessian<F>(f: F, x: &[f64]) -> GradHessian
where
    F: for<'t> Fn(&[Dual<Dual<Var<'t>>>]) -> Dual<Dual<Var<'t>>>,
{
    let n = x.len();
    let mut hessian = Matrix::zeros(n, n);
    let mut grad_hessian = vec![Matrix::zeros(n, n); n];
    let mut value = f64::NAN;
    let mut tape = Tape::new();
    let mut adjoints = Vec::new();
    for j in 0..n {
        for k in j..n {
            tape.clear();
            let xs = tape.vars(x);
            let args: Vec<Dual<Dual<Var<'_>>>> = xs
                .iter()
                .enumerate()
                .map(|(m, &v)| {
                    let inner = Dual::new(v, Var::constant(if m == k { 1.0 } else { 0.0 }));
                    let outer = Dual::constant(Var::constant(if m == j { 1.0 } else { 0.0 }));
                    Dual::new(inner, outer)
                })
                .collect();
            let y = f(&args);
            if j == 0 && k == 0 {
                value = y.re.re.value();
            }
            let h = y.eps.eps.value();
            hessian[(j, k)] = h;
            hessian[(k, j)] = h;
            tape.gradient_into(y.eps.eps, &mut adjoints);
            for (i, g) in grad_hessian.iter_mut().enumerate() {
                g[(j, k)] = adjoints[i];
                g[(k, j)] = adjoints[i];
            }
        }
    }
    if n == 0 {
        value = f(&[]).re.re.value();
    }
    GradHessian {
        value,
        hessian,
        grad_hessian,
    }
}

/// Value and directional derivative `∇f(x) · v` by forward mode, as `(f(x), ∇f(x) · v)`.
///
/// # Panics
///
/// Panics if `x` and `v` differ in length.
pub fn gradient_dot_vector<F>(f: F, x: &[f64], v: &[f64]) -> (f64, f64)
where
    F: FnOnce(&[Dual64]) -> Dual64,
{
    assert_eq!(
        x.len(),
        v.len(),
        "gradient_dot_vector: x and v differ in length"
    );
    let xs: Vec<Dual64> = x.iter().zip(v).map(|(&a, &b)| Dual::new(a, b)).collect();
    let y = f(&xs);
    (y.re, y.eps)
}

/// Value and Hessian vector product `∇²f(x) v` by forward over reverse mode in one evaluation,
/// as `(f(x), ∇²f(x) v)`. [`hessian_vector_product`](crate::hessian_vector_product) returns
/// only the product.
///
/// # Panics
///
/// Panics if `x` and `v` differ in length.
pub fn hessian_times_vector<F>(f: F, x: &[f64], v: &[f64]) -> (f64, Vec<f64>)
where
    F: for<'t> FnOnce(&[Dual<Var<'t>>]) -> Dual<Var<'t>>,
{
    assert_eq!(
        x.len(),
        v.len(),
        "hessian_times_vector: x and v differ in length"
    );
    let tape = Tape::new();
    let xs = tape.vars(x);
    let duals: Vec<Dual<Var<'_>>> = xs
        .iter()
        .zip(v)
        .map(|(&xi, &vi)| Dual::new(xi, Var::constant(vi)))
        .collect();
    let y = f(&duals);
    (y.re.value(), tape.gradient(y.eps).wrt_slice(&xs))
}

/// Gradient of `trace(M ∇²f(x))` with respect to `x`.
///
/// `trace(M H) = sum_j sum_k M[(j, k)] ∂_j ∂_k f`. Evaluation `j` seeds the outer tangent with
/// `e_j` and the inner tangent with row `j` of `M`, so its second-order tangent is the `j`th
/// term; the terms are summed on one tape and swept once.
///
/// # Panics
///
/// Panics unless `m` is `n` by `n` with `n = x.len()`.
pub fn grad_tr_mat_times_hessian<F>(f: F, x: &[f64], m: &Matrix<f64>) -> Vec<f64>
where
    F: for<'t> Fn(&[Dual<Dual<Var<'t>>>]) -> Dual<Dual<Var<'t>>>,
{
    let n = x.len();
    assert!(
        m.nrows() == n && m.ncols() == n,
        "grad_tr_mat_times_hessian: M must be {n} by {n}"
    );
    let tape = Tape::new();
    let xs = tape.vars(x);
    let mut total = Var::constant(0.0);
    for j in 0..n {
        let args: Vec<Dual<Dual<Var<'_>>>> = xs
            .iter()
            .enumerate()
            .map(|(c, &v)| {
                let inner = Dual::new(v, Var::constant(m[(j, c)]));
                let outer = Dual::constant(Var::constant(if c == j { 1.0 } else { 0.0 }));
                Dual::new(inner, outer)
            })
            .collect();
        total += f(&args).eps.eps;
    }
    tape.gradient(total).wrt_slice(&xs)
}

/// Value and partial derivative with respect to `x[n]` by forward mode, as `(f(x), ∂f/∂x_n)`.
///
/// # Panics
///
/// Panics if `n >= x.len()`.
pub fn partial_derivative<F>(f: F, x: &[f64], n: usize) -> (f64, f64)
where
    F: FnOnce(&[Dual64]) -> Dual64,
{
    assert!(n < x.len(), "partial_derivative: index {n} out of range");
    let xs: Vec<Dual64> = x
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            if i == n {
                Dual::variable(a)
            } else {
                Dual::constant(a)
            }
        })
        .collect();
    let y = f(&xs);
    (y.re, y.eps)
}

/// Fourth-order central difference weights for a first derivative, as `(offset, weight)`.
const D4: [(f64, f64); 4] = [
    (-2.0, 1.0 / 12.0),
    (-1.0, -8.0 / 12.0),
    (1.0, 8.0 / 12.0),
    (2.0, -1.0 / 12.0),
];

/// Sixth-order central difference weights for a first derivative, as `(offset, weight)`.
const D6: [(f64, f64); 6] = [
    (-3.0, -1.0 / 60.0),
    (-2.0, 9.0 / 60.0),
    (-1.0, -45.0 / 60.0),
    (1.0, 45.0 / 60.0),
    (2.0, -9.0 / 60.0),
    (3.0, 1.0 / 60.0),
];

/// Relative step for the finite-difference Hessians; the step for `x_i` is
/// `STEP * max(1, |x_i|)`.
const STEP: f64 = 1e-3;

fn step(xi: f64) -> f64 {
    STEP * xi.abs().max(1.0)
}

/// Mixed partial `∂²f/∂x_i∂x_j` by the fourth-order stencil along each coordinate, with relative
/// step `h`.
fn fd_hessian(f: &impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Matrix<f64> {
    let n = x.len();
    let mut hessian = Matrix::zeros(n, n);
    let mut xp = x.to_vec();
    for i in 0..n {
        for j in i..n {
            let (hi, hj) = (h * x[i].abs().max(1.0), h * x[j].abs().max(1.0));
            let mut acc = 0.0;
            for &(a, wa) in &D4 {
                for &(b, wb) in &D4 {
                    xp[i] += a * hi;
                    xp[j] += b * hj;
                    acc += wa * wb * f(&xp);
                    xp[i] = x[i];
                    xp[j] = x[j];
                }
            }
            let v = acc / (hi * hj);
            hessian[(i, j)] = v;
            hessian[(j, i)] = v;
        }
    }
    hessian
}

/// Gradient by sixth-order central differences with the step chosen from machine precision:
/// relative step `eps^(1/7)`, which balances truncation error (order `h^6`) against rounding
/// error (order `eps / h`). Stan Math's `finite_diff_gradient_auto`; the step rule is ours.
pub fn finite_diff_gradient_auto(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    finite_diff_gradient(f, x, f64::EPSILON.powf(1.0 / 7.0))
}

/// [`finite_diff_hessian`] with steps chosen from machine precision: relative step `eps^(1/6)`
/// for the Hessian, which balances truncation error (order `h^4`) against rounding error (order
/// `eps / h^2`), and the gradient of [`finite_diff_gradient_auto`]. Stan Math's
/// `finite_diff_hessian_auto`; the step rule is ours.
pub fn finite_diff_hessian_auto(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> ValueGradHessian {
    ValueGradHessian {
        value: f(x),
        gradient: finite_diff_gradient_auto(&f, x),
        hessian: fd_hessian(&f, x, f64::EPSILON.powf(1.0 / 6.0)),
    }
}

/// Value, gradient, and Hessian of an `f64` function by finite differences, for checking.
///
/// The gradient is the sixth-order [`finite_diff_gradient`]. Each Hessian entry applies the
/// fourth-order first derivative stencil along `x_i` and then along `x_j` (16 evaluations); on
/// the diagonal the two stencils compose into a fourth-order second derivative. Expect about
/// seven significant digits.
pub fn finite_diff_hessian(f: impl Fn(&[f64]) -> f64, x: &[f64]) -> ValueGradHessian {
    ValueGradHessian {
        value: f(x),
        gradient: finite_diff_gradient(&f, x, STEP),
        hessian: fd_hessian(&f, x, STEP),
    }
}

/// Value, Hessian, and third derivatives by sixth-order central differences of automatic
/// Hessians, for checking [`grad_hessian`].
///
/// `grad_hessian[i] = sum_k w_k H(x + k h_i e_i) / h_i`, where each `H` comes from the forward
/// over reverse [`hessian`](crate::hessian) driver.
pub fn finite_diff_grad_hessian<F>(f: F, x: &[f64]) -> GradHessian
where
    F: for<'t> Fn(&[Dual<Var<'t>>]) -> Dual<Var<'t>>,
{
    let n = x.len();
    let center = value_grad_hessian(&f, x);
    let mut xp = x.to_vec();
    let grad_hessian = (0..n)
        .map(|i| {
            let h = step(x[i]);
            let mut acc = Matrix::zeros(n, n);
            for &(k, w) in &D6 {
                xp[i] = x[i] + k * h;
                let hk = value_grad_hessian(&f, &xp).hessian;
                for r in 0..n {
                    for c in 0..n {
                        acc[(r, c)] += w * hk[(r, c)] / h;
                    }
                }
            }
            xp[i] = x[i];
            acc
        })
        .collect();
    GradHessian {
        value: center.value,
        hessian: center.hessian,
        grad_hessian,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // f = x0^3 x1 + exp(x1): third derivatives f_000 = 6 x1, f_001 = 6 x0, f_111 = exp(x1).
    fn cubic<T: Real>(x: &[T]) -> T {
        x[0].powi(3) * x[1] + x[1].exp()
    }

    #[test]
    fn third_derivatives_match_closed_form() {
        let x = [0.7, -0.4];
        let r = grad_hessian(|v| cubic(v), &x);
        assert!((r.hessian[(0, 1)] - 3.0 * 0.49).abs() < 1e-14);
        assert!((r.grad_hessian[0][(0, 0)] - 6.0 * -0.4).abs() < 1e-14);
        assert!((r.grad_hessian[1][(0, 0)] - 6.0 * 0.7).abs() < 1e-14);
        assert!((r.grad_hessian[0][(0, 1)] - 6.0 * 0.7).abs() < 1e-14);
        assert!((r.grad_hessian[1][(1, 1)] - (-0.4f64).exp()).abs() < 1e-14);
        let fd = finite_diff_grad_hessian(|v| cubic(v), &x);
        for i in 0..2 {
            assert!(fd.grad_hessian[i].max_abs_diff(&r.grad_hessian[i]) < 1e-8);
        }
        let fh = finite_diff_hessian(cubic::<f64>, &x);
        assert!(fh.hessian.max_abs_diff(&r.hessian) < 1e-6);
    }

    #[test]
    fn trace_gradient_matches_third_derivatives() {
        let x = [0.7, -0.4];
        let m = Matrix::from_rows(&[[1.0, 2.0], [-0.5, 3.0]]);
        let g = grad_tr_mat_times_hessian(|v| cubic(v), &x, &m);
        let r = grad_hessian(|v| cubic(v), &x);
        for (i, gi) in g.iter().enumerate() {
            let mut expect = 0.0;
            for j in 0..2 {
                for k in 0..2 {
                    expect += m[(j, k)] * r.grad_hessian[i][(k, j)];
                }
            }
            assert!((gi - expect).abs() < 1e-13);
        }
    }
}
