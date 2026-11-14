//! The derivative drivers against Stan Math's C++ functionals.
//!
//! `tests/data/stan_reference_functionals.csv` is written by
//! `tools/stan-reference/stan_reference_functionals.cpp`, which documents the test functions. The
//! automatic differentiation drivers agree to `1e-12`; the finite difference drivers use different
//! stencils from Stan's and agree to `1e-6`.

use stanmath_rs::autodiff::{value_and_derivative, value_grad_hessian};
use stanmath_rs::prelude::*;

fn f<S: Real>(x: &[S]) -> S {
    x[0] * x[0] * x[1] + (x[0] * x[1]).exp() * x[2].sin() + x[2] * x[2] * x[2] / x[1]
}

fn g<S: Real>(x: &[S]) -> Vec<S> {
    vec![x[0] * x[1], x[0].sin() + x[2] * x[2], x[1].exp() * x[2]]
}

const X: [f64; 3] = [0.7, 1.3, -0.4];
const V: [f64; 3] = [0.5, -1.0, 2.0];

fn cat(head: f64, m: &Matrix<f64>) -> Vec<f64> {
    std::iter::once(head)
        .chain(m.as_slice().iter().copied())
        .collect()
}

fn ours(name: &str) -> Vec<f64> {
    let gh = |r: GradHessian| {
        let mut v = cat(r.value, &r.hessian);
        for s in &r.grad_hessian {
            v.extend_from_slice(s.as_slice());
        }
        v
    };
    match name {
        "gradient" => {
            let r = value_and_grad(|x| f(x), &X);
            std::iter::once(r.value).chain(r.gradient).collect()
        }
        "hessian" => {
            let r = value_grad_hessian(|x| f(x), &X);
            cat(r.value, &r.hessian)
        }
        "grad_hessian" => gh(grad_hessian(|x| f(x), &X)),
        "hessian_times_vector" => {
            let (v, hv) = hessian_times_vector(|x| f(x), &X, &V);
            std::iter::once(v).chain(hv).collect()
        }
        "gradient_dot_vector" => {
            let (v, d) = gradient_dot_vector(f, &X, &V);
            vec![v, d]
        }
        "grad_tr_mat_times_hessian" => {
            let m = Matrix::from_rows(&[[1.0, 0.2, -0.3], [0.4, 2.0, 0.1], [-0.5, 0.6, 1.5]]);
            std::iter::once(0.0)
                .chain(grad_tr_mat_times_hessian(|x| f(x), &X, &m))
                .collect()
        }
        "jacobian" => {
            let (v, j) = jacobian(|x| g(x), &X);
            std::iter::once(0.0)
                .chain(v)
                .chain(j.as_slice().iter().copied())
                .collect()
        }
        "derivative" => {
            let (v, d) = value_and_derivative(|t| t * t * t * t.sin(), 0.9);
            vec![v, d]
        }
        "partial_derivative" => {
            let (v, d) = partial_derivative(f, &X, 2);
            vec![v, d]
        }
        "finite_diff_gradient" => std::iter::once(f(&X))
            .chain(finite_diff_gradient_auto(f, &X))
            .collect(),
        "finite_diff_grad_hessian" => gh(finite_diff_grad_hessian(|x| f(x), &X)),
        other => panic!("no driver for {other}"),
    }
}

#[test]
fn drivers_match_stan() {
    let text = include_str!("data/stan_reference_functionals.csv");
    let mut failures = Vec::new();
    for line in text.lines() {
        let (name, values) = line.split_once(',').unwrap();
        let want: Vec<f64> = values.split(';').map(|v| v.parse().unwrap()).collect();
        let got = ours(name);
        let tol = if name.starts_with("finite_diff") {
            1e-6
        } else {
            1e-12
        };
        if got.len() != want.len()
            || got
                .iter()
                .zip(&want)
                .any(|(a, b)| (a - b).abs() > tol * b.abs().max(1.0))
        {
            failures.push(format!("{name}: ours {got:?}\n  Stan {want:?}"));
        }
    }
    assert_eq!(text.lines().count(), 11);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
