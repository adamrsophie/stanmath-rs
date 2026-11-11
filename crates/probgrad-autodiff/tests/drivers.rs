//! Consistency of the differentiation drivers with each other and with closed forms.

use probgrad_autodiff::functions::*;
use probgrad_autodiff::*;
use probgrad_core::Matrix;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// A scalar test function mixing most primitive kinds.
fn mixed<T: Real>(x: &[T]) -> T {
    let s = log_sum_exp_slice(&[x[0], x[1] * 2.0, x[2]]);
    s * x[0].sin() + (x[1] * x[2]).exp() / (x[0].square() + 1.0) + log1p_exp(x[2] - x[0])
}

/// A vector function `R^3 -> R^2`.
fn vector_fn<T: Real>(x: &[T]) -> Vec<T> {
    vec![
        x[0] * x[1] + x[2].cos(),
        log_sum_exp(x[0], x[2]) - x[1].powi(3),
    ]
}

fn rosenbrock<T: Real>(x: &[T]) -> T {
    (-x[0] + 1.0).square() + (x[1] - x[0].square()).square() * 100.0
}

const X: [f64; 3] = [0.3, -0.7, 1.1];

#[test]
fn forward_and_reverse_gradients_agree() {
    let g = grad(|v| mixed(v), &X);
    for (j, gj) in g.iter().enumerate() {
        let mut dir = [0.0; 3];
        dir[j] = 1.0;
        let fwd = directional_derivative(mixed, &X, &dir);
        assert!(close(fwd, *gj, 1e-14), "component {j}: {fwd} vs {gj}");
    }
    let c = check_gradient(|v| mixed(v), &X);
    assert!(c.passes(1e-9, 1e-11), "{c:?}");
}

#[test]
fn jacobians_agree_across_modes() {
    let (v1, j_rev) = jacobian(|x| vector_fn(x), &X);
    let (v2, j_fwd) = jacobian_forward(vector_fn, &X);
    assert_eq!(v1, v2);
    assert_eq!(j_rev.shape(), (2, 3));
    assert!(j_rev.max_abs_diff(&j_fwd) < 1e-14);
    // Analytic Jacobian.
    let (x0, x1, x2) = (X[0], X[1], X[2]);
    let s = (x0.exp() + x2.exp()).ln();
    let expected = Matrix::from_rows(&[
        [x1, x0, -x2.sin()],
        [(x0 - s).exp(), -3.0 * x1 * x1, (x2 - s).exp()],
    ]);
    assert!(j_rev.max_abs_diff(&expected) < 1e-15);
    // jvp and vjp are J v and w^T J.
    let v = [0.5, -1.0, 2.0];
    let (_, jv) = jvp(vector_fn, &X, &v);
    let w = [1.5, -0.25];
    let (_, wj) = vjp(|x| vector_fn(x), &X, &w);
    for i in 0..2 {
        let want: f64 = (0..3).map(|j| expected[(i, j)] * v[j]).sum();
        assert!(close(jv[i], want, 1e-15));
    }
    for j in 0..3 {
        let want: f64 = (0..2).map(|i| w[i] * expected[(i, j)]).sum();
        assert!(close(wj[j], want, 1e-15));
    }
}

#[test]
fn hessians_agree_across_modes() {
    let h = hessian(|v| mixed(v), &X);
    let hf = hessian_forward(mixed, &X);
    assert!(h.max_abs_diff(&hf) < 1e-13, "{h:?}\n{hf:?}");
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                close(h[(i, j)], h[(j, i)], 1e-13),
                "asymmetric at ({i}, {j})"
            );
        }
    }
    let v = [0.25, -2.0, 1.0];
    let hv = hessian_vector_product(|x| mixed(x), &X, &v);
    for i in 0..3 {
        let want: f64 = (0..3).map(|j| h[(i, j)] * v[j]).sum();
        assert!(close(hv[i], want, 1e-13));
    }
    let r = value_grad_hessian(|x| mixed(x), &X);
    assert_eq!(r.value, mixed(&X));
    assert_eq!(r.gradient, grad(|x| mixed(x), &X));
    assert_eq!(r.hessian, h);
}

#[test]
fn rosenbrock_closed_form() {
    let x = [-1.2, 1.0];
    let g = grad(|v| rosenbrock(v), &x);
    let (a, b) = (x[0], x[1]);
    assert!(close(
        g[0],
        -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
        1e-15
    ));
    assert!(close(g[1], 200.0 * (b - a * a), 1e-15));
    let h = hessian(|v| rosenbrock(v), &x);
    let expected = Matrix::from_rows(&[
        [2.0 - 400.0 * b + 1200.0 * a * a, -400.0 * a],
        [-400.0 * a, 200.0],
    ]);
    assert!(h.max_abs_diff(&expected) < 1e-12);
}

#[test]
fn third_derivative_by_nesting() {
    // d^3/dx^3 of x^2 sin(x) by forward over forward over reverse.
    fn f<T: Real>(x: T) -> T {
        x.square() * x.sin()
    }
    let x0 = 0.7_f64;
    let tape = Tape::new();
    let xv = tape.var(x0);
    let one = Var::constant(1.0);
    let zero = Var::constant(0.0);
    let x = Dual::new(Dual::new(xv, one), Dual::new(one, zero));
    let y = f(x);
    // y.eps.eps is f''(x) as a Var; its gradient is f'''(x).
    let d3 = y.eps.eps.gradient().wrt(xv);
    // f''' = 6 cos x - 6 x sin x - x^2 cos x.
    let exact = 6.0 * x0.cos() - 6.0 * x0 * x0.sin() - x0 * x0 * x0.cos();
    assert!(close(d3, exact, 1e-14), "{d3} vs {exact}");
    // f'' = 2 sin x + 4 x cos x - x^2 sin x.
    let d2 = 2.0 * x0.sin() + 4.0 * x0 * x0.cos() - x0 * x0 * x0.sin();
    assert!(close(y.eps.eps.value(), d2, 1e-15));
}

#[test]
fn nested_gradient_inside_closure() {
    // The outer function calls an inner gradient on values only; tapes stay separate.
    let outer = grad(
        |p| {
            let inner = grad(|q| q[0] * q[0] * 3.0, &[p[0].value()]);
            p[0] * inner[0]
        },
        &[2.0],
    );
    // d/dp (p * c) where c = 6 * p0 is treated as data.
    assert_eq!(outer, vec![12.0]);
}

#[test]
fn workspace_matches_one_shot_and_reuses_memory() {
    let mut ws = GradientWorkspace::new();
    let mut g = [0.0; 3];
    for k in 0..5 {
        let x = [X[0] + f64::from(k), X[1], X[2]];
        let v = ws.value_and_grad_into(|p| mixed(p), &x, &mut g);
        let r = value_and_grad(|p| mixed(p), &x);
        assert_eq!(v, r.value);
        assert_eq!(g.to_vec(), r.gradient);
    }
    let err = ws.try_value_and_grad_into(
        |p| {
            if p[0] > 0.0 {
                Ok(p[0])
            } else {
                Err("negative")
            }
        },
        &[-1.0],
        &mut [0.0],
    );
    assert_eq!(err, Err("negative"));
}

#[test]
fn gradients_are_independent_across_threads() {
    let expected: Vec<Vec<f64>> = (0..8)
        .map(|k| grad(|p| mixed(p), &[X[0] * f64::from(k), X[1], X[2]]))
        .collect();
    let handles: Vec<_> = (0..8)
        .map(|k| {
            std::thread::spawn(move || {
                let mut ws = GradientWorkspace::new();
                let mut g = vec![0.0; 3];
                for _ in 0..200 {
                    ws.value_and_grad_into(
                        |p| mixed(p),
                        &[X[0] * f64::from(k), X[1], X[2]],
                        &mut g,
                    );
                }
                g
            })
        })
        .collect();
    for (k, h) in handles.into_iter().enumerate() {
        assert_eq!(h.join().unwrap(), expected[k]);
    }
}

#[test]
fn forward_mode_scalar_drivers() {
    fn f<T: Real>(x: T) -> T {
        x.exp() * x.cos()
    }
    let x = 0.4;
    let (v, d) = value_and_derivative(f, x);
    assert_eq!(v, x.exp() * x.cos());
    assert!(close(d, x.exp() * (x.cos() - x.sin()), 1e-15));
    let d2 = second_derivative(f, x);
    assert!(close(d2, -2.0 * x.exp() * x.sin(), 1e-15));
}

#[test]
fn derivative_conventions_at_kinks() {
    assert_eq!(derivative(abs, 0.0), 0.0);
    assert_eq!(grad(|p| fmax(p[0], p[1]), &[1.0, 1.0]), vec![1.0, 0.0]);
    assert_eq!(grad(|p| p[0].floor() + p[0], &[2.5]), vec![1.0]);
    // d/db 0^b is taken to be zero; d/da 0^2 is zero.
    assert_eq!(grad(|p| pow(p[0], p[1]), &[0.0, 2.0]), vec![0.0, 0.0]);
    // Multiplying by a constant zero contributes nothing, even next to a NaN partial.
    assert_eq!(grad(|p| p[0].sqrt() * 0.0 + p[0], &[0.0]), vec![1.0]);
}
