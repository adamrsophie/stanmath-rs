//! Every `_glm_lpdf` and `_glm_lpmf` signature of the Stan language, compared with the sum of the
//! response distribution over the observations.
//!
//! Written by `tools/reference/gen_glm.py` from the CmdStan 2.40.0 signature dump: 58 Stan
//! signatures. `matrix x` is 3 by 2; `row_vector x` is 1 by 2 and shared by every observation;
//! `vector x` is 3 by 1. A real response is a one element slice shared by every observation. Each
//! GLM must equal the sum over the observations of its response distribution at
//! `x[i] beta + alpha[i]`, in value and gradient with respect to every real argument; the response
//! distributions are compared with Stan Math by `stan_reference.rs`.
//!
//! Stan Math 5.4.0 differs in one case: `poisson_log_glm_lpmf` with an int response and several
//! rows of `x` adds `-lgamma(y + 1)` once rather than once per observation. stanmath-rs returns the
//! sum.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::identity_op,
    clippy::erasing_op,
    clippy::needless_range_loop
)]

use stanmath_rs_autodiff::{ValueAndGrad, Var, try_value_and_grad};
use stanmath_rs_core::{Matrix, Result};
use stanmath_rs_distributions::*;

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

fn compare(
    label: &str,
    v: Result<ValueAndGrad>,
    s: Result<ValueAndGrad>,
    failures: &mut Vec<String>,
) {
    match (v, s) {
        (Ok(v), Ok(s))
            if close(v.value, s.value)
                && v.gradient.len() == s.gradient.len()
                && v.gradient
                    .iter()
                    .zip(&s.gradient)
                    .all(|(a, b)| close(*a, *b)) => {}
        (v, s) => failures.push(format!("{label}: {v:?} vs {s:?}")),
    }
}

#[test]
fn glm_signatures() {
    let mut failures = Vec::new();
    // stan: bernoulli_logit_glm_lpmf(int, matrix, real, vector)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                bernoulli_logit_glm_lpmf(1_i64, &xm, x[6], &x[7..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += bernoulli_logit_lpmf(1_i64, eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "bernoulli_logit_glm_lpmf(int, matrix, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: bernoulli_logit_glm_lpmf(int, matrix, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                bernoulli_logit_glm_lpmf(1_i64, &xm, &x[6..9], &x[9..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..2).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += bernoulli_logit_lpmf(1_i64, eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "bernoulli_logit_glm_lpmf(int, matrix, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: bernoulli_logit_glm_lpmf(array[] int, row_vector, real, vector)
    {
        let y0 = [1_i64, 0_i64, 1_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                bernoulli_logit_glm_lpmf(&y0[..], &xm, x[2], &x[3..5])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[2] + (0..2).map(|j| xm[(r, j)] * x[3 + j]).sum::<Var>();
                    acc += bernoulli_logit_lpmf(y0[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "bernoulli_logit_glm_lpmf(array[] int, row_vector, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: bernoulli_logit_glm_lpmf(array[] int, row_vector, vector, vector)
    {
        let y0 = [1_i64, 0_i64, 1_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                bernoulli_logit_glm_lpmf(&y0[..], &xm, &x[2..5], &x[5..7])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[2 + i] + (0..2).map(|j| xm[(r, j)] * x[5 + j]).sum::<Var>();
                    acc += bernoulli_logit_lpmf(y0[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "bernoulli_logit_glm_lpmf(array[] int, row_vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: bernoulli_logit_glm_lpmf(array[] int, matrix, real, vector)
    {
        let y0 = [1_i64, 0_i64, 1_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                bernoulli_logit_glm_lpmf(&y0[..], &xm, x[6], &x[7..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += bernoulli_logit_lpmf(y0[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "bernoulli_logit_glm_lpmf(array[] int, matrix, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: bernoulli_logit_glm_lpmf(array[] int, matrix, vector, vector)
    {
        let y0 = [1_i64, 0_i64, 1_i64];
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                bernoulli_logit_glm_lpmf(&y0[..], &xm, &x[6..9], &x[9..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..2).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += bernoulli_logit_lpmf(y0[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "bernoulli_logit_glm_lpmf(array[] int, matrix, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: binomial_logit_glm_lpmf(int, int, matrix, real, vector)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                binomial_logit_glm_lpmf(1_i64, 4_i64, &xm, x[6], &x[7..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += binomial_logit_lpmf(1_i64, 4_i64, eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_glm_lpmf(int, int, matrix, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: binomial_logit_glm_lpmf(int, int, matrix, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                binomial_logit_glm_lpmf(1_i64, 4_i64, &xm, &x[6..9], &x[9..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..2).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += binomial_logit_lpmf(1_i64, 4_i64, eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_glm_lpmf(int, int, matrix, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: binomial_logit_glm_lpmf(int, array[] int, matrix, real, vector)
    {
        let y1 = [4_i64, 5_i64, 6_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                binomial_logit_glm_lpmf(1_i64, &y1[..], &xm, x[6], &x[7..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += binomial_logit_lpmf(1_i64, y1[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_glm_lpmf(int, array[] int, matrix, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: binomial_logit_glm_lpmf(array[] int, int, matrix, real, vector)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                binomial_logit_glm_lpmf(&y0[..], 4_i64, &xm, x[6], &x[7..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += binomial_logit_lpmf(y0[i], 4_i64, eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_glm_lpmf(array[] int, int, matrix, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: binomial_logit_glm_lpmf(array[] int, array[] int, row_vector, real, vector)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let y1 = [4_i64, 5_i64, 6_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                binomial_logit_glm_lpmf(&y0[..], &y1[..], &xm, x[2], &x[3..5])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[2] + (0..2).map(|j| xm[(r, j)] * x[3 + j]).sum::<Var>();
                    acc += binomial_logit_lpmf(y0[i], y1[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_glm_lpmf(array[] int, array[] int, row_vector, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: binomial_logit_glm_lpmf(array[] int, array[] int, row_vector, vector, vector)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let y1 = [4_i64, 5_i64, 6_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                binomial_logit_glm_lpmf(&y0[..], &y1[..], &xm, &x[2..5], &x[5..7])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[2 + i] + (0..2).map(|j| xm[(r, j)] * x[5 + j]).sum::<Var>();
                    acc += binomial_logit_lpmf(y0[i], y1[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_glm_lpmf(array[] int, array[] int, row_vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: binomial_logit_glm_lpmf(array[] int, array[] int, matrix, real, vector)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let y1 = [4_i64, 5_i64, 6_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                binomial_logit_glm_lpmf(&y0[..], &y1[..], &xm, x[6], &x[7..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += binomial_logit_lpmf(y0[i], y1[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_glm_lpmf(array[] int, array[] int, matrix, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: binomial_logit_glm_lpmf(array[] int, array[] int, matrix, vector, vector)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let y1 = [4_i64, 5_i64, 6_i64];
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                binomial_logit_glm_lpmf(&y0[..], &y1[..], &xm, &x[6..9], &x[9..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..2).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += binomial_logit_lpmf(y0[i], y1[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_glm_lpmf(array[] int, array[] int, matrix, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: categorical_logit_glm_lpmf(int, row_vector, vector, matrix)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let bm = Matrix::from_column_major(2, 3, x[5..11].to_vec());
                categorical_logit_glm_lpmf(1_i64, &xm, &x[2..5], &bm)
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let bm = Matrix::from_column_major(2, 3, x[5..11].to_vec());
                let mut acc = x[0] * 0.0;
                for _i in 0..1 {
                    let r = 0;
                    let eta: Vec<_> = (0..3)
                        .map(|c| x[2 + c] + (0..2).map(|j| xm[(r, j)] * bm[(j, c)]).sum::<Var>())
                        .collect();
                    acc += categorical_logit_lpmf(1_i64, &eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "categorical_logit_glm_lpmf(int, row_vector, vector, matrix)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: categorical_logit_glm_lpmf(int, matrix, vector, matrix)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 0.32, 0.74, 1.16,
            1.58,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let bm = Matrix::from_column_major(2, 3, x[9..15].to_vec());
                categorical_logit_glm_lpmf(1_i64, &xm, &x[6..9], &bm)
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let bm = Matrix::from_column_major(2, 3, x[9..15].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta: Vec<_> = (0..3)
                        .map(|c| x[6 + c] + (0..2).map(|j| xm[(r, j)] * bm[(j, c)]).sum::<Var>())
                        .collect();
                    acc += categorical_logit_lpmf(1_i64, &eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "categorical_logit_glm_lpmf(int, matrix, vector, matrix)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: categorical_logit_glm_lpmf(array[] int, row_vector, vector, matrix)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let bm = Matrix::from_column_major(2, 3, x[5..11].to_vec());
                categorical_logit_glm_lpmf(&y0[..], &xm, &x[2..5], &bm)
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let bm = Matrix::from_column_major(2, 3, x[5..11].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta: Vec<_> = (0..3)
                        .map(|c| x[2 + c] + (0..2).map(|j| xm[(r, j)] * bm[(j, c)]).sum::<Var>())
                        .collect();
                    acc += categorical_logit_lpmf(y0[i], &eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "categorical_logit_glm_lpmf(array[] int, row_vector, vector, matrix)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: categorical_logit_glm_lpmf(array[] int, matrix, vector, matrix)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 0.32, 0.74, 1.16,
            1.58,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let bm = Matrix::from_column_major(2, 3, x[9..15].to_vec());
                categorical_logit_glm_lpmf(&y0[..], &xm, &x[6..9], &bm)
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let bm = Matrix::from_column_major(2, 3, x[9..15].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta: Vec<_> = (0..3)
                        .map(|c| x[6 + c] + (0..2).map(|j| xm[(r, j)] * bm[(j, c)]).sum::<Var>())
                        .collect();
                    acc += categorical_logit_lpmf(y0[i], &eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "categorical_logit_glm_lpmf(array[] int, matrix, vector, matrix)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: neg_binomial_2_log_glm_lpmf(int, matrix, real, vector, real)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 2.5];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                neg_binomial_2_log_glm_lpmf(2_i64, &xm, x[6], &x[7..9], x[9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += neg_binomial_2_log_lpmf(2_i64, eta, x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_glm_lpmf(int, matrix, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: neg_binomial_2_log_glm_lpmf(int, matrix, vector, vector, real)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 2.5,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                neg_binomial_2_log_glm_lpmf(2_i64, &xm, &x[6..9], &x[9..11], x[11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..2).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += neg_binomial_2_log_lpmf(2_i64, eta, x[11])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_glm_lpmf(int, matrix, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: neg_binomial_2_log_glm_lpmf(array[] int, row_vector, real, vector, real)
    {
        let y0 = [2_i64, 0_i64, 3_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, 2.5];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                neg_binomial_2_log_glm_lpmf(&y0[..], &xm, x[2], &x[3..5], x[5])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[2] + (0..2).map(|j| xm[(r, j)] * x[3 + j]).sum::<Var>();
                    acc += neg_binomial_2_log_lpmf(y0[i], eta, x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_glm_lpmf(array[] int, row_vector, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: neg_binomial_2_log_glm_lpmf(array[] int, row_vector, vector, vector, real)
    {
        let y0 = [2_i64, 0_i64, 3_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 2.5];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                neg_binomial_2_log_glm_lpmf(&y0[..], &xm, &x[2..5], &x[5..7], x[7])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[2 + i] + (0..2).map(|j| xm[(r, j)] * x[5 + j]).sum::<Var>();
                    acc += neg_binomial_2_log_lpmf(y0[i], eta, x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_glm_lpmf(array[] int, row_vector, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: neg_binomial_2_log_glm_lpmf(array[] int, matrix, real, vector, real)
    {
        let y0 = [2_i64, 0_i64, 3_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 2.5];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                neg_binomial_2_log_glm_lpmf(&y0[..], &xm, x[6], &x[7..9], x[9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += neg_binomial_2_log_lpmf(y0[i], eta, x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_glm_lpmf(array[] int, matrix, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: neg_binomial_2_log_glm_lpmf(array[] int, matrix, vector, vector, real)
    {
        let y0 = [2_i64, 0_i64, 3_i64];
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 2.5,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                neg_binomial_2_log_glm_lpmf(&y0[..], &xm, &x[6..9], &x[9..11], x[11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..2).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += neg_binomial_2_log_lpmf(y0[i], eta, x[11])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_glm_lpmf(array[] int, matrix, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, vector, real, vector, real)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 1.3];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[1..4].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, x[4], &x[5..6], x[6])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[1..4].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[4] + (0..1).map(|j| xm[(r, j)] * x[5 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, vector, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, vector, real, vector, vector)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.8, 1.1, 1.4];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[1..4].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, x[4], &x[5..6], &x[6..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[1..4].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[4] + (0..1).map(|j| xm[(r, j)] * x[5 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, vector, real, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, vector, vector, vector, real)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 1.3];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[1..4].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, &x[4..7], &x[7..8], x[8])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[1..4].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[4 + i] + (0..1).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, vector, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, vector, vector, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[1..4].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, &x[4..7], &x[7..8], &x[8..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[1..4].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[4 + i] + (0..1).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, vector, vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, row_vector, real, vector, real)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 1.3];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[1..3].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, x[3], &x[4..6], x[6])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[1..3].to_vec());
                let mut acc = x[0] * 0.0;
                for _i in 0..1 {
                    let r = 0;
                    let eta = x[3] + (0..2).map(|j| xm[(r, j)] * x[4 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, row_vector, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, row_vector, real, vector, vector)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.8];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[1..3].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, x[3], &x[4..6], &x[6..7])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[1..3].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    let r = 0;
                    let eta = x[3] + (0..2).map(|j| xm[(r, j)] * x[4 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, row_vector, real, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, row_vector, vector, vector, real)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 1.3];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[1..3].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, &x[3..4], &x[4..6], x[6])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[1..3].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    let r = 0;
                    let eta = x[3 + i] + (0..2).map(|j| xm[(r, j)] * x[4 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, row_vector, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, row_vector, vector, vector, vector)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.8];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[1..3].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, &x[3..4], &x[4..6], &x[6..7])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[1..3].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    let r = 0;
                    let eta = x[3 + i] + (0..2).map(|j| xm[(r, j)] * x[4 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, row_vector, vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, matrix, real, vector, real)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, 1.3,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[1..7].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, x[7], &x[8..10], x[10])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[1..7].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[7] + (0..2).map(|j| xm[(r, j)] * x[8 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, matrix, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, matrix, real, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, 0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[1..7].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, x[7], &x[8..10], &x[10..13])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[1..7].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[7] + (0..2).map(|j| xm[(r, j)] * x[8 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[10 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, matrix, real, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, matrix, vector, vector, real)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 0.32, 1.3,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[1..7].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, &x[7..10], &x[10..12], x[12])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[1..7].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[7 + i] + (0..2).map(|j| xm[(r, j)] * x[10 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[12])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, matrix, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(real, matrix, vector, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 0.32, 0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[1..7].to_vec());
                normal_id_glm_lpdf(&x[0..1], &xm, &x[7..10], &x[10..12], &x[12..15])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[1..7].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[7 + i] + (0..2).map(|j| xm[(r, j)] * x[10 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0], eta, x[12 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(real, matrix, vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, vector, real, vector, real)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 1.3];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[3..6].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, x[6], &x[7..8], x[8])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[3..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..1).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, vector, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, vector, real, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[3..6].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, x[6], &x[7..8], &x[8..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[3..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..1).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, vector, real, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, vector, vector, vector, real)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, 1.3,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[3..6].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, &x[6..9], &x[9..10], x[10])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[3..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..1).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, vector, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, vector, vector, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, 0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[3..6].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, &x[6..9], &x[9..10], &x[10..13])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 1, x[3..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..1).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[10 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, vector, vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, row_vector, real, vector, real)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 1.3];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[3..5].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, x[5], &x[6..8], x[8])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[3..5].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[5] + (0..2).map(|j| xm[(r, j)] * x[6 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, row_vector, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, row_vector, real, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[3..5].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, x[5], &x[6..8], &x[8..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[3..5].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[5] + (0..2).map(|j| xm[(r, j)] * x[6 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, row_vector, real, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, row_vector, vector, vector, real)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, 1.3,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[3..5].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, &x[5..8], &x[8..10], x[10])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[3..5].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[5 + i] + (0..2).map(|j| xm[(r, j)] * x[8 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, row_vector, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, row_vector, vector, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, 0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[3..5].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, &x[5..8], &x[8..10], &x[10..13])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[3..5].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[5 + i] + (0..2).map(|j| xm[(r, j)] * x[8 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[10 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, row_vector, vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, matrix, real, vector, real)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 0.32, 1.3,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[3..9].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, x[9], &x[10..12], x[12])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[3..9].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[9] + (0..2).map(|j| xm[(r, j)] * x[10 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[12])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, matrix, real, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, matrix, real, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 0.32, 0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[3..9].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, x[9], &x[10..12], &x[12..15])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[3..9].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[9] + (0..2).map(|j| xm[(r, j)] * x[10 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[12 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, matrix, real, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, matrix, vector, vector, real)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 0.32, 0.74, 1.16,
            1.3,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[3..9].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, &x[9..12], &x[12..14], x[14])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[3..9].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[9 + i] + (0..2).map(|j| xm[(r, j)] * x[12 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[14])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, matrix, vector, vector, real)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: normal_id_glm_lpdf(vector, matrix, vector, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1, 0.32, 0.74, 1.16,
            0.8, 1.1, 1.4,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[3..9].to_vec());
                normal_id_glm_lpdf(&x[0..3], &xm, &x[9..12], &x[12..14], &x[14..17])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[3..9].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[9 + i] + (0..2).map(|j| xm[(r, j)] * x[12 + j]).sum::<Var>();
                    acc += normal_lpdf(x[0 + i], eta, x[14 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "normal_id_glm_lpdf(vector, matrix, vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: ordered_logistic_glm_lpmf(int, row_vector, vector, vector)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, 2.18];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                ordered_logistic_glm_lpmf(1_i64, &xm, &x[2..4], &x[4..6])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for _i in 0..1 {
                    let r = 0;
                    let eta = (0..2).map(|j| xm[(r, j)] * x[2 + j]).sum::<Var>();
                    acc += ordered_logistic_lpmf(1_i64, eta, &x[4..6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "ordered_logistic_glm_lpmf(int, row_vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: ordered_logistic_glm_lpmf(int, matrix, vector, vector)
    {
        let p = [
            -0.6,
            -0.18,
            0.24,
            0.66,
            1.08,
            -0.35,
            0.07,
            0.49,
            0.91,
            2.0100000000000002,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                ordered_logistic_glm_lpmf(1_i64, &xm, &x[6..8], &x[8..10])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = (0..2).map(|j| xm[(r, j)] * x[6 + j]).sum::<Var>();
                    acc += ordered_logistic_lpmf(1_i64, eta, &x[8..10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "ordered_logistic_glm_lpmf(int, matrix, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: ordered_logistic_glm_lpmf(array[] int, row_vector, vector, vector)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, 2.18];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                ordered_logistic_glm_lpmf(&y0[..], &xm, &x[2..4], &x[4..6])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = (0..2).map(|j| xm[(r, j)] * x[2 + j]).sum::<Var>();
                    acc += ordered_logistic_lpmf(y0[i], eta, &x[4..6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "ordered_logistic_glm_lpmf(array[] int, row_vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: ordered_logistic_glm_lpmf(array[] int, matrix, vector, vector)
    {
        let y0 = [1_i64, 3_i64, 2_i64];
        let p = [
            -0.6,
            -0.18,
            0.24,
            0.66,
            1.08,
            -0.35,
            0.07,
            0.49,
            0.91,
            2.0100000000000002,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                ordered_logistic_glm_lpmf(&y0[..], &xm, &x[6..8], &x[8..10])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = (0..2).map(|j| xm[(r, j)] * x[6 + j]).sum::<Var>();
                    acc += ordered_logistic_lpmf(y0[i], eta, &x[8..10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "ordered_logistic_glm_lpmf(array[] int, matrix, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: poisson_log_glm_lpmf(int, matrix, real, vector)
    {
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                poisson_log_glm_lpmf(2_i64, &xm, x[6], &x[7..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += poisson_log_lpmf(2_i64, eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "poisson_log_glm_lpmf(int, matrix, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: poisson_log_glm_lpmf(int, matrix, vector, vector)
    {
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                poisson_log_glm_lpmf(2_i64, &xm, &x[6..9], &x[9..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..2).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += poisson_log_lpmf(2_i64, eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "poisson_log_glm_lpmf(int, matrix, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: poisson_log_glm_lpmf(array[] int, row_vector, real, vector)
    {
        let y0 = [2_i64, 0_i64, 3_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                poisson_log_glm_lpmf(&y0[..], &xm, x[2], &x[3..5])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[2] + (0..2).map(|j| xm[(r, j)] * x[3 + j]).sum::<Var>();
                    acc += poisson_log_lpmf(y0[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "poisson_log_glm_lpmf(array[] int, row_vector, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: poisson_log_glm_lpmf(array[] int, row_vector, vector, vector)
    {
        let y0 = [2_i64, 0_i64, 3_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                poisson_log_glm_lpmf(&y0[..], &xm, &x[2..5], &x[5..7])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(1, 2, x[0..2].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = 0;
                    let eta = x[2 + i] + (0..2).map(|j| xm[(r, j)] * x[5 + j]).sum::<Var>();
                    acc += poisson_log_lpmf(y0[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "poisson_log_glm_lpmf(array[] int, row_vector, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: poisson_log_glm_lpmf(array[] int, matrix, real, vector)
    {
        let y0 = [2_i64, 0_i64, 3_i64];
        let p = [-0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                poisson_log_glm_lpmf(&y0[..], &xm, x[6], &x[7..9])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6] + (0..2).map(|j| xm[(r, j)] * x[7 + j]).sum::<Var>();
                    acc += poisson_log_lpmf(y0[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "poisson_log_glm_lpmf(array[] int, matrix, real, vector)",
            v,
            s,
            &mut failures,
        );
    }
    // stan: poisson_log_glm_lpmf(array[] int, matrix, vector, vector)
    {
        let y0 = [2_i64, 0_i64, 3_i64];
        let p = [
            -0.6, -0.18, 0.24, 0.66, 1.08, -0.35, 0.07, 0.49, 0.91, 1.33, -0.1,
        ];
        let v = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                poisson_log_glm_lpmf(&y0[..], &xm, &x[6..9], &x[9..11])
            },
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let xm = Matrix::from_column_major(3, 2, x[0..6].to_vec());
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    let r = i;
                    let eta = x[6 + i] + (0..2).map(|j| xm[(r, j)] * x[9 + j]).sum::<Var>();
                    acc += poisson_log_lpmf(y0[i], eta)?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "poisson_log_glm_lpmf(array[] int, matrix, vector, vector)",
            v,
            s,
            &mut failures,
        );
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
