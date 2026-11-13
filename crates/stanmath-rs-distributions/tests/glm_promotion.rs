//! GLM result types promote across every argument, as in Stan: data `beta` with an autodiff
//! design matrix, and data `x`, `alpha`, and `beta` with an autodiff `phi`.

// Reference values keep the 17 significant digits that Stan Math printed.
#![allow(clippy::excessive_precision)]

use stanmath_rs_autodiff::{Var, value_and_grad};
use stanmath_rs_core::Matrix;
use stanmath_rs_distributions::*;

const X: [f64; 6] = [0.3, -0.2, 0.5, 0.1, 0.4, -0.6];
const BETA: [f64; 2] = [0.7, -0.4];

fn xm<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    Matrix::from_column_major(3, 2, v.to_vec())
}

fn same(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (u, v) in a.iter().zip(b) {
        assert!((u - v).abs() <= 1e-12 * (1.0 + v.abs()), "{u} vs {v}");
    }
}

#[test]
fn data_beta_with_an_autodiff_design_matrix() {
    let y = [1i64, 0, 3];
    let data_beta = value_and_grad(
        |v| poisson_log_glm_lpmf(&y[..], &xm(v), 0.2, &BETA).unwrap(),
        &X,
    );
    let lifted = value_and_grad(
        |v| {
            let b: Vec<Var<'_>> = BETA.iter().map(|&b| v[0] * 0.0 + b).collect();
            poisson_log_glm_lpmf(&y[..], &xm(v), 0.2, &b).unwrap()
        },
        &X,
    );
    same(&[data_beta.value], &[lifted.value]);
    same(&data_beta.gradient, &lifted.gradient);

    // A row vector design with data beta.
    let row = value_and_grad(
        |v| poisson_log_glm_lpmf_row(&y[..], &v[..2], 0.2, &BETA).unwrap(),
        &X[..2],
    );
    let full = value_and_grad(
        |v| {
            let x = Matrix::from_fn(3, 2, |_, j| v[j]);
            let b: Vec<Var<'_>> = BETA.iter().map(|&b| v[0] * 0.0 + b).collect();
            poisson_log_glm_lpmf(&y[..], &x, 0.2, &b).unwrap()
        },
        &X[..2],
    );
    same(&[row.value], &[full.value]);
    same(&row.gradient, &full.gradient);

    // A matrix beta, data, with an autodiff design matrix.
    let beta = Matrix::from_column_major(2, 3, vec![0.1, -0.3, 0.2, 0.4, -0.5, 0.6]);
    let yc = [1i64, 3, 2];
    let cat = value_and_grad(
        |v| categorical_logit_glm_lpmf(&yc[..], &xm(v), &[0.0, 0.1, -0.1], &beta).unwrap(),
        &X,
    );
    let cat_lifted = value_and_grad(
        |v| {
            let b = beta.map(|&e| v[0] * 0.0 + e);
            categorical_logit_glm_lpmf(&yc[..], &xm(v), &[0.0, 0.1, -0.1], &b).unwrap()
        },
        &X,
    );
    same(&[cat.value], &[cat_lifted.value]);
    same(&cat.gradient, &cat_lifted.gradient);
}

#[test]
fn propto_keeps_the_data_linear_predictor_out() {
    // Only phi is a parameter. Both propto forms drop the predictor term; the GLM form also drops
    // `log y!`, which neg_binomial_2_log keeps inside its binomial coefficient, so the two differ
    // by `sum log y! = log 6` and have the same gradient.
    let y = [1i64, 0, 3];
    let x = Matrix::from_column_major(3, 2, X.to_vec());
    let eta: Vec<f64> = (0..3)
        .map(|i| 0.2 + x[(i, 0)] * BETA[0] + x[(i, 1)] * BETA[1])
        .collect();
    let glm = value_and_grad(
        |v| neg_binomial_2_log_glm_lupmf(&y[..], &x, 0.2, &BETA, v[0]).unwrap(),
        &[1.5],
    );
    let plain = value_and_grad(
        |v| neg_binomial_2_log_lupmf(&y[..], &eta[..], v[0]).unwrap(),
        &[1.5],
    );
    same(&[glm.value - plain.value], &[6f64.ln()]);
    // Stan Math 5.4.0, neg_binomial_2_log_glm_lpmf<true> with only phi a var.
    same(&[glm.value, glm.gradient[0]], &[-5.1058654989833698, 0.28072083048372631]);
    same(&glm.gradient, &plain.gradient);
}
