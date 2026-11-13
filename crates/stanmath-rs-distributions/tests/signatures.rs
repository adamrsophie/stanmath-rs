//! Every vectorized distribution signature of the Stan language, mapped to its Rust call.
//!
//! Written by `tools/reference/gen_signatures.py` from the CmdStan 2.40.0 signature dump: 11953
//! Stan signatures in 1273 Rust shapes (vector, row_vector, and array forms of one argument
//! are the same Rust slice). Each broadcast call must equal the sum (the product for `_cdf`) of
//! the scalar calls at each index, in value and gradient; the scalar calls are compared with Stan
//! Math by `stan_reference.rs` and `stan_reference_extra.rs`.
//!
//! Left to the Stan reference tests: bernoulli_logit_glm_lpmf, binomial_logit_glm_lpmf, categorical_logit_glm_lpmf, categorical_logit_lpmf, categorical_lpmf, dirichlet_lpdf, dirichlet_multinomial_lpmf, gaussian_dlm_obs_lpdf, inv_wishart_cholesky_lpdf, inv_wishart_lpdf, lkj_corr_cholesky_lpdf, lkj_corr_lpdf, lkj_cov_lpdf, multi_gp_cholesky_lpdf, multi_gp_lpdf, multi_normal_cholesky_lpdf, multi_normal_lpdf, multi_normal_prec_lpdf, multi_student_t_cholesky_lpdf, multi_student_t_lpdf, multinomial_logit_lpmf, multinomial_lpmf, neg_binomial_2_log_glm_lpmf, normal_id_glm_lpdf, ordered_logistic_glm_lpmf, ordered_logistic_lpmf, ordered_probit_lpmf, poisson_binomial_cdf, poisson_binomial_lccdf, poisson_binomial_lcdf, poisson_binomial_lpmf, poisson_log_glm_lpmf, wishart_cholesky_lpdf, wishart_lpdf.
#![allow(clippy::too_many_lines, clippy::unreadable_literal)]

use stanmath_rs_autodiff::{ValueAndGrad, try_value_and_grad};
use stanmath_rs_core::Result;
use stanmath_rs_distributions::*;

fn close(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

fn compare(
    label: &str,
    v: Result<ValueAndGrad>,
    s: Result<ValueAndGrad>,
    failures: &mut Vec<String>,
) {
    match (v, s) {
        (Err(_), Err(_)) => {}
        (Ok(v), Ok(s)) => {
            let same = close(v.value, s.value)
                && v.gradient
                    .iter()
                    .zip(&s.gradient)
                    .all(|(a, b)| close(*a, *b));
            if !same {
                failures.push(format!("{label}: {v:?} vs {s:?}"));
            }
        }
        (v, s) => failures.push(format!("{label}: {v:?} vs {s:?}")),
    }
}

fn compare_data(label: &str, v: Result<f64>, s: Result<f64>, failures: &mut Vec<String>) {
    match (v, s) {
        (Err(_), Err(_)) => {}
        (Ok(v), Ok(s)) if close(v, s) => {}
        (v, s) => failures.push(format!("{label}: {v:?} vs {s:?}")),
    }
}

#[test]
fn bernoulli_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3];
        let v = try_value_and_grad(|x| bernoulli_cdf(1i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= bernoulli_cdf(1i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_cdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312];
        let v = try_value_and_grad(|x| bernoulli_cdf(1i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= bernoulli_cdf(1i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_cdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3];
        let v = try_value_and_grad(|x| bernoulli_cdf(&[1i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= bernoulli_cdf(1i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_cdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312];
        let v = try_value_and_grad(|x| bernoulli_cdf(&[1i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= bernoulli_cdf(1i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_cdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn bernoulli_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3];
        let v = try_value_and_grad(|x| bernoulli_lccdf(0i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += bernoulli_lccdf(0i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lccdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312];
        let v = try_value_and_grad(|x| bernoulli_lccdf(0i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lccdf(0i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lccdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3];
        let v = try_value_and_grad(|x| bernoulli_lccdf(&[0i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lccdf(0i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lccdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312];
        let v = try_value_and_grad(|x| bernoulli_lccdf(&[0i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lccdf(0i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lccdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn bernoulli_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3];
        let v = try_value_and_grad(|x| bernoulli_lcdf(1i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += bernoulli_lcdf(1i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lcdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312];
        let v = try_value_and_grad(|x| bernoulli_lcdf(1i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lcdf(1i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lcdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3];
        let v = try_value_and_grad(|x| bernoulli_lcdf(&[1i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lcdf(1i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lcdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312];
        let v = try_value_and_grad(|x| bernoulli_lcdf(&[1i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lcdf(1i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lcdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn bernoulli_logit_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| bernoulli_logit_lpmf(1i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += bernoulli_logit_lpmf(1i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_logit_lpmf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| bernoulli_logit_lpmf(1i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_logit_lpmf(1i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_logit_lpmf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| bernoulli_logit_lpmf(&[1i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_logit_lpmf(1i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_logit_lpmf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| bernoulli_logit_lpmf(&[1i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_logit_lpmf(1i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_logit_lpmf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn bernoulli_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3];
        let v = try_value_and_grad(|x| bernoulli_lpmf(1i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += bernoulli_lpmf(1i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lpmf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312];
        let v = try_value_and_grad(|x| bernoulli_lpmf(1i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lpmf(1i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lpmf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3];
        let v = try_value_and_grad(|x| bernoulli_lpmf(&[1i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lpmf(1i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lpmf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312];
        let v = try_value_and_grad(|x| bernoulli_lpmf(&[1i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += bernoulli_lpmf(1i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("bernoulli_lpmf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_binomial_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_cdf(3i64, 10i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| beta_binomial_cdf(3i64, 10i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_cdf(3i64, 10i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| beta_binomial_cdf(3i64, 10i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_cdf(3i64, &[10i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(3i64, &[10i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(3i64, &[10i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(3i64, &[10i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_cdf(&[3i64; 3][..], 10i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(&[3i64; 3][..], 10i64, x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(&[3i64; 3][..], 10i64, &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(&[3i64; 3][..], 10i64, &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(&[3i64; 3][..], &[10i64; 3][..], x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(&[3i64; 3][..], &[10i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_cdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_cdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_binomial_cdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_binomial_cdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_binomial_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_lccdf(3i64, 10i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| beta_binomial_lccdf(3i64, 10i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_lccdf(3i64, 10i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| beta_binomial_lccdf(3i64, 10i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(3i64, &[10i64; 3][..], x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(3i64, &[10i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(3i64, &[10i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(3i64, &[10i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(&[3i64; 3][..], 10i64, x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(&[3i64; 3][..], 10i64, x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(&[3i64; 3][..], 10i64, &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(&[3i64; 3][..], 10i64, &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(&[3i64; 3][..], &[10i64; 3][..], x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(&[3i64; 3][..], &[10i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lccdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lccdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lccdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_binomial_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_lcdf(3i64, 10i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| beta_binomial_lcdf(3i64, 10i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_lcdf(3i64, 10i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| beta_binomial_lcdf(3i64, 10i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(3i64, &[10i64; 3][..], x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(3i64, &[10i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(3i64, &[10i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(3i64, &[10i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(&[3i64; 3][..], 10i64, x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(&[3i64; 3][..], 10i64, x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(&[3i64; 3][..], 10i64, &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(&[3i64; 3][..], 10i64, &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(&[3i64; 3][..], &[10i64; 3][..], x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(&[3i64; 3][..], &[10i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lcdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lcdf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lcdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_binomial_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_lpmf(3i64, 10i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| beta_binomial_lpmf(3i64, 10i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| beta_binomial_lpmf(3i64, 10i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| beta_binomial_lpmf(3i64, 10i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(3i64, &[10i64; 3][..], x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(3i64, &[10i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(3i64, &[10i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(3i64, &[10i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(&[3i64; 3][..], 10i64, x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(&[3i64; 3][..], 10i64, x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(&[3i64; 3][..], 10i64, &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(&[3i64; 3][..], 10i64, &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(&[3i64; 3][..], &[10i64; 3][..], x[0], x[1]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(&[3i64; 3][..], &[10i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| beta_binomial_lpmf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_binomial_lpmf(3i64, 10i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_binomial_lpmf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 2.0, 5.0];
        let v = try_value_and_grad(|x| beta_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= beta_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 2.04, 2.08, 5.0];
        let v = try_value_and_grad(|x| beta_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 2.04, 2.08, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 5.0];
        let v = try_value_and_grad(|x| beta_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 2.04, 2.08, 5.0];
        let v = try_value_and_grad(|x| beta_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 2.04, 2.08, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 2.0, 5.0];
        let v = try_value_and_grad(|x| beta_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 2.04, 2.08, 5.0];
        let v = try_value_and_grad(|x| beta_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 2.04, 2.08, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 5.0];
        let v = try_value_and_grad(|x| beta_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 2.04, 2.08, 5.0];
        let v = try_value_and_grad(|x| beta_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 2.04, 2.08, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 2.0, 5.0];
        let v = try_value_and_grad(|x| beta_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 2.04, 2.08, 5.0];
        let v = try_value_and_grad(|x| beta_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 2.04, 2.08, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 5.0];
        let v = try_value_and_grad(|x| beta_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 2.04, 2.08, 5.0];
        let v = try_value_and_grad(|x| beta_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 2.04, 2.08, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 2.0, 5.0];
        let v = try_value_and_grad(|x| beta_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 2.04, 2.08, 5.0];
        let v = try_value_and_grad(|x| beta_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 2.0, 2.04, 2.08, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 5.0];
        let v = try_value_and_grad(|x| beta_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 2.04, 2.08, 5.0];
        let v = try_value_and_grad(|x| beta_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 2.0, 2.04, 2.08, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_neg_binomial_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 3.7, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_cdf(3i64, x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(|x| beta_neg_binomial_cdf(3i64, x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 3.7740000000000005, 3.8480000000000003, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_cdf(3i64, x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(3i64, x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_cdf(3i64, &x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(3i64, &x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(3i64, &x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(3i64, &x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(&[3i64; 3][..], x[0], x[1], x[2]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(&[3i64; 3][..], x[0], x[1], &x[2..5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 3.7740000000000005, 3.8480000000000003, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(&[3i64; 3][..], x[0], &x[1..4], x[4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(&[3i64; 3][..], x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(&[3i64; 3][..], &x[0..3], x[3], x[4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(&[3i64; 3][..], &x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(&[3i64; 3][..], &x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_cdf(&[3i64; 3][..], &x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= beta_neg_binomial_cdf(3i64, x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_cdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_neg_binomial_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 3.7, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lccdf(3i64, x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(|x| beta_neg_binomial_lccdf(3i64, x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 3.7740000000000005, 3.8480000000000003, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lccdf(3i64, x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(3i64, x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lccdf(3i64, &x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(3i64, &x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(3i64, &x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(3i64, &x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(&[3i64; 3][..], x[0], x[1], x[2]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(&[3i64; 3][..], x[0], x[1], &x[2..5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 3.7740000000000005, 3.8480000000000003, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(&[3i64; 3][..], x[0], &x[1..4], x[4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(&[3i64; 3][..], x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(&[3i64; 3][..], &x[0..3], x[3], x[4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(&[3i64; 3][..], &x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(&[3i64; 3][..], &x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lccdf(&[3i64; 3][..], &x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lccdf(3i64, x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lccdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_neg_binomial_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 3.7, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lcdf(3i64, x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(|x| beta_neg_binomial_lcdf(3i64, x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 3.7740000000000005, 3.8480000000000003, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lcdf(3i64, x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(3i64, x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lcdf(3i64, &x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(3i64, &x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(3i64, &x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(3i64, &x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(&[3i64; 3][..], x[0], x[1], x[2]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(&[3i64; 3][..], x[0], x[1], &x[2..5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 3.7740000000000005, 3.8480000000000003, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(&[3i64; 3][..], x[0], &x[1..4], x[4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(&[3i64; 3][..], x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(&[3i64; 3][..], &x[0..3], x[3], x[4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(&[3i64; 3][..], &x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(&[3i64; 3][..], &x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lcdf(&[3i64; 3][..], &x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lcdf(3i64, x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lcdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_neg_binomial_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 3.7, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lpmf(3i64, x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(|x| beta_neg_binomial_lpmf(3i64, x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 3.7740000000000005, 3.8480000000000003, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lpmf(3i64, x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(3i64, x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5];
        let v = try_value_and_grad(|x| beta_neg_binomial_lpmf(3i64, &x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(3i64, &x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(3i64, &x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(3i64, &x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(&[3i64; 3][..], x[0], x[1], x[2]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(&[3i64; 3][..], x[0], x[1], &x[2..5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 3.7, 3.7740000000000005, 3.8480000000000003, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(&[3i64; 3][..], x[0], &x[1..4], x[4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(&[3i64; 3][..], x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(&[3i64; 3][..], &x[0..3], x[3], x[4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 3.7, 1.5, 1.53, 1.56];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(&[3i64; 3][..], &x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(&[3i64; 3][..], &x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            2.5,
            2.55,
            2.6,
            3.7,
            3.7740000000000005,
            3.8480000000000003,
            1.5,
            1.53,
            1.56,
        ];
        let v = try_value_and_grad(
            |x| beta_neg_binomial_lpmf(&[3i64; 3][..], &x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_neg_binomial_lpmf(3i64, x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_neg_binomial_lpmf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_proportion_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 0.4, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_proportion_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.4, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_proportion_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lccdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.4, 0.40800000000000003, 0.41600000000000004, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lccdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
            5.1,
            5.2,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lccdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.306, 0.312, 0.4, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lccdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.306, 0.312, 0.4, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_proportion_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lccdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.306,
            0.312,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lccdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.306,
            0.312,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
            5.1,
            5.2,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_proportion_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_proportion_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 0.4, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_proportion_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.4, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_proportion_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lcdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.4, 0.40800000000000003, 0.41600000000000004, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lcdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
            5.1,
            5.2,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lcdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.306, 0.312, 0.4, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lcdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.306, 0.312, 0.4, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_proportion_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lcdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.306,
            0.312,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lcdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.306,
            0.312,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
            5.1,
            5.2,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_proportion_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn beta_proportion_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 0.4, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += beta_proportion_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lpdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.4, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_proportion_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lpdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.4, 0.40800000000000003, 0.41600000000000004, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lpdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
            5.1,
            5.2,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lpdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.306, 0.312, 0.4, 5.0];
        let v = try_value_and_grad(|x| beta_proportion_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lpdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.3, 0.306, 0.312, 0.4, 5.0, 5.1, 5.2];
        let v = try_value_and_grad(|x| beta_proportion_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lpdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.306,
            0.312,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "beta_proportion_lpdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.3,
            0.306,
            0.312,
            0.4,
            0.40800000000000003,
            0.41600000000000004,
            5.0,
            5.1,
            5.2,
        ];
        let v = try_value_and_grad(|x| beta_proportion_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += beta_proportion_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("beta_proportion_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn binomial_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_cdf(3i64, 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= binomial_cdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_cdf(3i64, 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= binomial_cdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_cdf(3i64, &[10i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= binomial_cdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_cdf(3i64, &[10i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= binomial_cdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_cdf(&[3i64; 3][..], 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= binomial_cdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_cdf(&[3i64; 3][..], 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= binomial_cdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_cdf(&[3i64; 3][..], &[10i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= binomial_cdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(
            |x| binomial_cdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= binomial_cdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn binomial_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lccdf(3i64, 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += binomial_lccdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lccdf(3i64, 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lccdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lccdf(3i64, &[10i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lccdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lccdf(3i64, &[10i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lccdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lccdf(&[3i64; 3][..], 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lccdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lccdf(&[3i64; 3][..], 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lccdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(
            |x| binomial_lccdf(&[3i64; 3][..], &[10i64; 3][..], x[0]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lccdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(
            |x| binomial_lccdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lccdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn binomial_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lcdf(3i64, 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += binomial_lcdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lcdf(3i64, 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lcdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lcdf(3i64, &[10i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lcdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lcdf(3i64, &[10i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lcdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lcdf(&[3i64; 3][..], 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lcdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lcdf(&[3i64; 3][..], 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lcdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lcdf(&[3i64; 3][..], &[10i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lcdf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(
            |x| binomial_lcdf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lcdf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn binomial_logit_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [-0.4];
        let v = try_value_and_grad(|x| binomial_logit_lpmf(3i64, 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += binomial_logit_lpmf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_lpmf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [-0.4, -0.40800000000000003, -0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_logit_lpmf(3i64, 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_logit_lpmf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_lpmf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [-0.4];
        let v = try_value_and_grad(|x| binomial_logit_lpmf(3i64, &[10i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_logit_lpmf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_lpmf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [-0.4, -0.40800000000000003, -0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_logit_lpmf(3i64, &[10i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_logit_lpmf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_logit_lpmf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [-0.4];
        let v = try_value_and_grad(|x| binomial_logit_lpmf(&[3i64; 3][..], 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_logit_lpmf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "binomial_logit_lpmf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [-0.4, -0.40800000000000003, -0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_logit_lpmf(&[3i64; 3][..], 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_logit_lpmf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_logit_lpmf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [-0.4];
        let v = try_value_and_grad(
            |x| binomial_logit_lpmf(&[3i64; 3][..], &[10i64; 3][..], x[0]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_logit_lpmf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_logit_lpmf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [-0.4, -0.40800000000000003, -0.41600000000000004];
        let v = try_value_and_grad(
            |x| binomial_logit_lpmf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_logit_lpmf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_logit_lpmf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn binomial_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lpmf(3i64, 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += binomial_lpmf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lpmf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lpmf(3i64, 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lpmf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lpmf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lpmf(3i64, &[10i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lpmf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lpmf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lpmf(3i64, &[10i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lpmf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lpmf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lpmf(&[3i64; 3][..], 10i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lpmf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lpmf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(|x| binomial_lpmf(&[3i64; 3][..], 10i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lpmf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lpmf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.4];
        let v = try_value_and_grad(|x| binomial_lpmf(&[3i64; 3][..], &[10i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lpmf(3i64, 10i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lpmf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.4, 0.40800000000000003, 0.41600000000000004];
        let v = try_value_and_grad(
            |x| binomial_lpmf(&[3i64; 3][..], &[10i64; 3][..], &x[0..3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += binomial_lpmf(3i64, 10i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("binomial_lpmf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn cauchy_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.5, 2.0];
        let v = try_value_and_grad(|x| cauchy_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= cauchy_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= cauchy_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| cauchy_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= cauchy_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= cauchy_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0];
        let v = try_value_and_grad(|x| cauchy_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= cauchy_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= cauchy_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| cauchy_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= cauchy_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= cauchy_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn cauchy_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.5, 2.0];
        let v = try_value_and_grad(|x| cauchy_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += cauchy_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| cauchy_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0];
        let v = try_value_and_grad(|x| cauchy_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| cauchy_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn cauchy_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.5, 2.0];
        let v = try_value_and_grad(|x| cauchy_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += cauchy_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| cauchy_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0];
        let v = try_value_and_grad(|x| cauchy_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| cauchy_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn cauchy_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.5, 2.0];
        let v = try_value_and_grad(|x| cauchy_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += cauchy_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| cauchy_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0];
        let v = try_value_and_grad(|x| cauchy_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| cauchy_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| cauchy_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += cauchy_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("cauchy_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn chi_square_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 3.0];
        let v = try_value_and_grad(|x| chi_square_cdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= chi_square_cdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_cdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| chi_square_cdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= chi_square_cdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_cdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0];
        let v = try_value_and_grad(|x| chi_square_cdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= chi_square_cdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_cdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| chi_square_cdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= chi_square_cdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_cdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn chi_square_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 3.0];
        let v = try_value_and_grad(|x| chi_square_lccdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += chi_square_lccdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lccdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| chi_square_lccdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lccdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lccdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0];
        let v = try_value_and_grad(|x| chi_square_lccdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lccdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lccdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| chi_square_lccdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lccdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lccdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn chi_square_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 3.0];
        let v = try_value_and_grad(|x| chi_square_lcdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += chi_square_lcdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lcdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| chi_square_lcdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lcdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lcdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0];
        let v = try_value_and_grad(|x| chi_square_lcdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lcdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lcdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| chi_square_lcdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lcdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lcdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn chi_square_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 3.0];
        let v = try_value_and_grad(|x| chi_square_lpdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += chi_square_lpdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lpdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| chi_square_lpdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lpdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lpdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0];
        let v = try_value_and_grad(|x| chi_square_lpdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lpdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lpdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| chi_square_lpdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += chi_square_lpdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("chi_square_lpdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn discrete_range_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let v = discrete_range_cdf(3i64, 1i64, 5i64);
        let s = (0..1).try_fold(1.0, |acc, _i| {
            discrete_range_cdf(3i64, 1i64, 5i64).map(|r| acc * r)
        });
        compare_data(
            "discrete_range_cdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_cdf(3i64, 1i64, &[5i64; 3][..]);
        let s = (0..3).try_fold(1.0, |acc, _i| {
            discrete_range_cdf(3i64, 1i64, 5i64).map(|r| acc * r)
        });
        compare_data(
            "discrete_range_cdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_cdf(3i64, &[1i64; 3][..], 5i64);
        let s = (0..3).try_fold(1.0, |acc, _i| {
            discrete_range_cdf(3i64, 1i64, 5i64).map(|r| acc * r)
        });
        compare_data(
            "discrete_range_cdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_cdf(3i64, &[1i64; 3][..], &[5i64; 3][..]);
        let s = (0..3).try_fold(1.0, |acc, _i| {
            discrete_range_cdf(3i64, 1i64, 5i64).map(|r| acc * r)
        });
        compare_data("discrete_range_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let v = discrete_range_cdf(&[3i64; 3][..], 1i64, 5i64);
        let s = (0..3).try_fold(1.0, |acc, _i| {
            discrete_range_cdf(3i64, 1i64, 5i64).map(|r| acc * r)
        });
        compare_data(
            "discrete_range_cdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_cdf(&[3i64; 3][..], 1i64, &[5i64; 3][..]);
        let s = (0..3).try_fold(1.0, |acc, _i| {
            discrete_range_cdf(3i64, 1i64, 5i64).map(|r| acc * r)
        });
        compare_data("discrete_range_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let v = discrete_range_cdf(&[3i64; 3][..], &[1i64; 3][..], 5i64);
        let s = (0..3).try_fold(1.0, |acc, _i| {
            discrete_range_cdf(3i64, 1i64, 5i64).map(|r| acc * r)
        });
        compare_data("discrete_range_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let v = discrete_range_cdf(&[3i64; 3][..], &[1i64; 3][..], &[5i64; 3][..]);
        let s = (0..3).try_fold(1.0, |acc, _i| {
            discrete_range_cdf(3i64, 1i64, 5i64).map(|r| acc * r)
        });
        compare_data("discrete_range_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn discrete_range_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let v = discrete_range_lccdf(3i64, 1i64, 5i64);
        let s = (0..1).try_fold(0.0, |acc, _i| {
            discrete_range_lccdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lccdf(3i64, 1i64, &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lccdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lccdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lccdf(3i64, &[1i64; 3][..], 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lccdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lccdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lccdf(3i64, &[1i64; 3][..], &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lccdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lccdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lccdf(&[3i64; 3][..], 1i64, 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lccdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lccdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lccdf(&[3i64; 3][..], 1i64, &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lccdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lccdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lccdf(&[3i64; 3][..], &[1i64; 3][..], 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lccdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lccdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lccdf(&[3i64; 3][..], &[1i64; 3][..], &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lccdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn discrete_range_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let v = discrete_range_lcdf(3i64, 1i64, 5i64);
        let s = (0..1).try_fold(0.0, |acc, _i| {
            discrete_range_lcdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lcdf(3i64, 1i64, &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lcdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lcdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lcdf(3i64, &[1i64; 3][..], 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lcdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lcdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lcdf(3i64, &[1i64; 3][..], &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lcdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let v = discrete_range_lcdf(&[3i64; 3][..], 1i64, 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lcdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lcdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lcdf(&[3i64; 3][..], 1i64, &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lcdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let v = discrete_range_lcdf(&[3i64; 3][..], &[1i64; 3][..], 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lcdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let v = discrete_range_lcdf(&[3i64; 3][..], &[1i64; 3][..], &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lcdf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn discrete_range_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let v = discrete_range_lpmf(3i64, 1i64, 5i64);
        let s = (0..1).try_fold(0.0, |acc, _i| {
            discrete_range_lpmf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lpmf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lpmf(3i64, 1i64, &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lpmf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lpmf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lpmf(3i64, &[1i64; 3][..], 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lpmf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lpmf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lpmf(3i64, &[1i64; 3][..], &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lpmf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lpmf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let v = discrete_range_lpmf(&[3i64; 3][..], 1i64, 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lpmf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data(
            "discrete_range_lpmf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let v = discrete_range_lpmf(&[3i64; 3][..], 1i64, &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lpmf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lpmf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let v = discrete_range_lpmf(&[3i64; 3][..], &[1i64; 3][..], 5i64);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lpmf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lpmf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let v = discrete_range_lpmf(&[3i64; 3][..], &[1i64; 3][..], &[5i64; 3][..]);
        let s = (0..3).try_fold(0.0, |acc, _i| {
            discrete_range_lpmf(3i64, 1i64, 5i64).map(|r| acc + r)
        });
        compare_data("discrete_range_lpmf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn double_exponential_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| double_exponential_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= double_exponential_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_cdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= double_exponential_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_cdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| double_exponential_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= double_exponential_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_cdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= double_exponential_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_cdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| double_exponential_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= double_exponential_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_cdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= double_exponential_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_cdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| double_exponential_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= double_exponential_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_cdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= double_exponential_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("double_exponential_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn double_exponential_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += double_exponential_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lccdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lccdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lccdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lccdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lccdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lccdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(
            |x| double_exponential_lccdf(&x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lccdf(vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn double_exponential_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += double_exponential_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lcdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lcdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lcdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lcdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lcdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lcdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(
            |x| double_exponential_lcdf(&x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lcdf(vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn double_exponential_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += double_exponential_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lpdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lpdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lpdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lpdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lpdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| double_exponential_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lpdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| double_exponential_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lpdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(
            |x| double_exponential_lpdf(&x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += double_exponential_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "double_exponential_lpdf(vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn exp_mod_normal_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= exp_mod_normal_cdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 0.8, 0.8160000000000001, 0.8320000000000001];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            0.51,
            0.52,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            0.51,
            0.52,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_cdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_cdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_cdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            0.51,
            0.52,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_cdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(
            |x| exp_mod_normal_cdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            0.51,
            0.52,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_cdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exp_mod_normal_cdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_cdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn exp_mod_normal_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += exp_mod_normal_lccdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 0.8, 0.8160000000000001, 0.8320000000000001];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            0.51,
            0.52,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            0.51,
            0.52,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lccdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lccdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lccdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            0.51,
            0.52,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lccdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lccdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            0.51,
            0.52,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lccdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lccdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn exp_mod_normal_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += exp_mod_normal_lcdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 0.8, 0.8160000000000001, 0.8320000000000001];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            0.51,
            0.52,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            0.51,
            0.52,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lcdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lcdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lcdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            0.51,
            0.52,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lcdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lcdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            0.51,
            0.52,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lcdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lcdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn exp_mod_normal_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += exp_mod_normal_lpdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 0.8, 0.8160000000000001, 0.8320000000000001];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            0.51,
            0.52,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            0.5,
            0.51,
            0.52,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lpdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lpdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.8];
        let v = try_value_and_grad(|x| exp_mod_normal_lpdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            0.51,
            0.52,
            1.2,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lpdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.8];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lpdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3,
            1.326,
            1.352,
            0.5,
            0.51,
            0.52,
            1.2,
            1.224,
            1.248,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| exp_mod_normal_lpdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exp_mod_normal_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "exp_mod_normal_lpdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn exponential_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 2.0];
        let v = try_value_and_grad(|x| exponential_cdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= exponential_cdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_cdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| exponential_cdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exponential_cdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_cdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| exponential_cdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exponential_cdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_cdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| exponential_cdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= exponential_cdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_cdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn exponential_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 2.0];
        let v = try_value_and_grad(|x| exponential_lccdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += exponential_lccdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lccdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| exponential_lccdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lccdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lccdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| exponential_lccdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lccdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lccdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| exponential_lccdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lccdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lccdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn exponential_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 2.0];
        let v = try_value_and_grad(|x| exponential_lcdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += exponential_lcdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lcdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| exponential_lcdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lcdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lcdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| exponential_lcdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lcdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lcdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| exponential_lcdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lcdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lcdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn exponential_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 2.0];
        let v = try_value_and_grad(|x| exponential_lpdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += exponential_lpdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lpdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| exponential_lpdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lpdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lpdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| exponential_lpdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lpdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lpdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| exponential_lpdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += exponential_lpdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("exponential_lpdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn frechet_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 2.0, 1.2];
        let v = try_value_and_grad(|x| frechet_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= frechet_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= frechet_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 1.2];
        let v = try_value_and_grad(|x| frechet_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= frechet_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= frechet_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 1.2];
        let v = try_value_and_grad(|x| frechet_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= frechet_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= frechet_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 1.2];
        let v = try_value_and_grad(|x| frechet_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= frechet_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= frechet_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn frechet_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 2.0, 1.2];
        let v = try_value_and_grad(|x| frechet_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += frechet_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 1.2];
        let v = try_value_and_grad(|x| frechet_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 1.2];
        let v = try_value_and_grad(|x| frechet_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 1.2];
        let v = try_value_and_grad(|x| frechet_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn frechet_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 2.0, 1.2];
        let v = try_value_and_grad(|x| frechet_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += frechet_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 1.2];
        let v = try_value_and_grad(|x| frechet_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 1.2];
        let v = try_value_and_grad(|x| frechet_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 1.2];
        let v = try_value_and_grad(|x| frechet_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn frechet_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 2.0, 1.2];
        let v = try_value_and_grad(|x| frechet_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += frechet_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 1.2];
        let v = try_value_and_grad(|x| frechet_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 1.2];
        let v = try_value_and_grad(|x| frechet_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 1.2];
        let v = try_value_and_grad(|x| frechet_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| frechet_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += frechet_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("frechet_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn gamma_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 0.5];
        let v = try_value_and_grad(|x| gamma_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= gamma_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gamma_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| gamma_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gamma_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gamma_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5];
        let v = try_value_and_grad(|x| gamma_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gamma_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gamma_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| gamma_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gamma_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gamma_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn gamma_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 0.5];
        let v = try_value_and_grad(|x| gamma_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += gamma_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| gamma_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5];
        let v = try_value_and_grad(|x| gamma_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| gamma_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn gamma_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 0.5];
        let v = try_value_and_grad(|x| gamma_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += gamma_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| gamma_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5];
        let v = try_value_and_grad(|x| gamma_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| gamma_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn gamma_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 0.5];
        let v = try_value_and_grad(|x| gamma_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += gamma_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| gamma_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5];
        let v = try_value_and_grad(|x| gamma_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| gamma_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| gamma_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gamma_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gamma_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn gumbel_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| gumbel_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= gumbel_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gumbel_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| gumbel_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gumbel_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gumbel_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| gumbel_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gumbel_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gumbel_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| gumbel_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gumbel_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= gumbel_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn gumbel_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| gumbel_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += gumbel_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| gumbel_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| gumbel_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| gumbel_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn gumbel_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| gumbel_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += gumbel_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| gumbel_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| gumbel_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| gumbel_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn gumbel_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| gumbel_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += gumbel_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| gumbel_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| gumbel_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| gumbel_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| gumbel_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += gumbel_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("gumbel_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn hypergeometric_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let v = hypergeometric_lpmf(2i64, 5i64, 4i64, 6i64);
        let s = (0..1).try_fold(0.0, |acc, _i| {
            hypergeometric_lpmf(2i64, 5i64, 4i64, 6i64).map(|r| acc + r)
        });
        compare_data(
            "hypergeometric_lpmf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn inv_chi_square_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 3.0];
        let v = try_value_and_grad(|x| inv_chi_square_cdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= inv_chi_square_cdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_cdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| inv_chi_square_cdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_chi_square_cdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_cdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0];
        let v = try_value_and_grad(|x| inv_chi_square_cdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_chi_square_cdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_cdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| inv_chi_square_cdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_chi_square_cdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_cdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn inv_chi_square_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 3.0];
        let v = try_value_and_grad(|x| inv_chi_square_lccdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += inv_chi_square_lccdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lccdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| inv_chi_square_lccdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lccdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lccdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0];
        let v = try_value_and_grad(|x| inv_chi_square_lccdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lccdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lccdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| inv_chi_square_lccdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lccdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lccdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn inv_chi_square_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 3.0];
        let v = try_value_and_grad(|x| inv_chi_square_lcdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += inv_chi_square_lcdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lcdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| inv_chi_square_lcdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lcdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lcdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0];
        let v = try_value_and_grad(|x| inv_chi_square_lcdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lcdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lcdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| inv_chi_square_lcdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lcdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lcdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn inv_chi_square_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 3.0];
        let v = try_value_and_grad(|x| inv_chi_square_lpdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += inv_chi_square_lpdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lpdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| inv_chi_square_lpdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lpdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lpdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0];
        let v = try_value_and_grad(|x| inv_chi_square_lpdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lpdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lpdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| inv_chi_square_lpdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_chi_square_lpdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_chi_square_lpdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn inv_gamma_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= inv_gamma_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_gamma_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_gamma_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_gamma_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_gamma_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_gamma_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_gamma_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= inv_gamma_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn inv_gamma_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += inv_gamma_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "inv_gamma_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn inv_gamma_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += inv_gamma_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "inv_gamma_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn inv_gamma_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += inv_gamma_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "inv_gamma_lpdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5];
        let v = try_value_and_grad(|x| inv_gamma_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| inv_gamma_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += inv_gamma_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("inv_gamma_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn logistic_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.5, 2.0];
        let v = try_value_and_grad(|x| logistic_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= logistic_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= logistic_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| logistic_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= logistic_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= logistic_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0];
        let v = try_value_and_grad(|x| logistic_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= logistic_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= logistic_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| logistic_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= logistic_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= logistic_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn logistic_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.5, 2.0];
        let v = try_value_and_grad(|x| logistic_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += logistic_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "logistic_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| logistic_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0];
        let v = try_value_and_grad(|x| logistic_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| logistic_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn logistic_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.5, 2.0];
        let v = try_value_and_grad(|x| logistic_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += logistic_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| logistic_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0];
        let v = try_value_and_grad(|x| logistic_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| logistic_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn logistic_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.5, 2.0];
        let v = try_value_and_grad(|x| logistic_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += logistic_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| logistic_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0];
        let v = try_value_and_grad(|x| logistic_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| logistic_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| logistic_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += logistic_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("logistic_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn loglogistic_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 2.0, 3.0];
        let v = try_value_and_grad(|x| loglogistic_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= loglogistic_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "loglogistic_cdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 2.0, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| loglogistic_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= loglogistic_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 3.0];
        let v = try_value_and_grad(|x| loglogistic_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= loglogistic_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| loglogistic_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= loglogistic_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 3.0];
        let v = try_value_and_grad(|x| loglogistic_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= loglogistic_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| loglogistic_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= loglogistic_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 3.0];
        let v = try_value_and_grad(|x| loglogistic_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= loglogistic_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| loglogistic_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= loglogistic_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn loglogistic_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 2.0, 3.0];
        let v = try_value_and_grad(|x| loglogistic_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += loglogistic_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "loglogistic_lpdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 2.0, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| loglogistic_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += loglogistic_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 3.0];
        let v = try_value_and_grad(|x| loglogistic_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += loglogistic_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 2.0, 2.04, 2.08, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| loglogistic_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += loglogistic_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 3.0];
        let v = try_value_and_grad(|x| loglogistic_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += loglogistic_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| loglogistic_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += loglogistic_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 3.0];
        let v = try_value_and_grad(|x| loglogistic_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += loglogistic_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 2.0, 2.04, 2.08, 3.0, 3.06, 3.12];
        let v = try_value_and_grad(|x| loglogistic_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += loglogistic_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("loglogistic_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn lognormal_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.2, 0.7];
        let v = try_value_and_grad(|x| lognormal_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= lognormal_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.2, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| lognormal_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= lognormal_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.2, 0.20400000000000001, 0.20800000000000002, 0.7];
        let v = try_value_and_grad(|x| lognormal_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= lognormal_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
            0.714,
            0.728,
        ];
        let v = try_value_and_grad(|x| lognormal_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= lognormal_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.2, 0.7];
        let v = try_value_and_grad(|x| lognormal_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= lognormal_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.2, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| lognormal_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= lognormal_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            1.53,
            1.56,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
        ];
        let v = try_value_and_grad(|x| lognormal_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= lognormal_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            1.53,
            1.56,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
            0.714,
            0.728,
        ];
        let v = try_value_and_grad(|x| lognormal_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= lognormal_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn lognormal_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.2, 0.7];
        let v = try_value_and_grad(|x| lognormal_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += lognormal_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "lognormal_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 0.2, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| lognormal_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.2, 0.20400000000000001, 0.20800000000000002, 0.7];
        let v = try_value_and_grad(|x| lognormal_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
            0.714,
            0.728,
        ];
        let v = try_value_and_grad(|x| lognormal_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.2, 0.7];
        let v = try_value_and_grad(|x| lognormal_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.2, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| lognormal_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            1.53,
            1.56,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
        ];
        let v = try_value_and_grad(|x| lognormal_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            1.53,
            1.56,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
            0.714,
            0.728,
        ];
        let v = try_value_and_grad(|x| lognormal_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn lognormal_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.2, 0.7];
        let v = try_value_and_grad(|x| lognormal_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += lognormal_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "lognormal_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 0.2, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| lognormal_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.2, 0.20400000000000001, 0.20800000000000002, 0.7];
        let v = try_value_and_grad(|x| lognormal_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
            0.714,
            0.728,
        ];
        let v = try_value_and_grad(|x| lognormal_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.2, 0.7];
        let v = try_value_and_grad(|x| lognormal_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.2, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| lognormal_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            1.53,
            1.56,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
        ];
        let v = try_value_and_grad(|x| lognormal_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            1.53,
            1.56,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
            0.714,
            0.728,
        ];
        let v = try_value_and_grad(|x| lognormal_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn lognormal_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 0.2, 0.7];
        let v = try_value_and_grad(|x| lognormal_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += lognormal_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "lognormal_lpdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 0.2, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| lognormal_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 0.2, 0.20400000000000001, 0.20800000000000002, 0.7];
        let v = try_value_and_grad(|x| lognormal_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
            0.714,
            0.728,
        ];
        let v = try_value_and_grad(|x| lognormal_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.2, 0.7];
        let v = try_value_and_grad(|x| lognormal_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 0.2, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| lognormal_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            1.53,
            1.56,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
        ];
        let v = try_value_and_grad(|x| lognormal_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5,
            1.53,
            1.56,
            0.2,
            0.20400000000000001,
            0.20800000000000002,
            0.7,
            0.714,
            0.728,
        ];
        let v = try_value_and_grad(|x| lognormal_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += lognormal_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("lognormal_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_2_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_cdf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= neg_binomial_2_cdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_cdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_cdf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_2_cdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_cdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_cdf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_2_cdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_cdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_cdf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_2_cdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_cdf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_2_cdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_cdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_cdf(&[3i64; 3][..], x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_2_cdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_cdf(&[3i64; 3][..], &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_2_cdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(
            |x| neg_binomial_2_cdf(&[3i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_2_cdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_2_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lccdf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += neg_binomial_2_lccdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lccdf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lccdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lccdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lccdf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lccdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lccdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lccdf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lccdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lccdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lccdf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lccdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lccdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lccdf(&[3i64; 3][..], x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lccdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lccdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lccdf(&[3i64; 3][..], &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lccdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lccdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(
            |x| neg_binomial_2_lccdf(&[3i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lccdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_2_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lcdf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += neg_binomial_2_lcdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lcdf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lcdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lcdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lcdf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lcdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lcdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lcdf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lcdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lcdf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lcdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lcdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lcdf(&[3i64; 3][..], x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lcdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lcdf(&[3i64; 3][..], &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lcdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(
            |x| neg_binomial_2_lcdf(&[3i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lcdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_2_log_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.9, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_log_lpmf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += neg_binomial_2_log_lpmf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_lpmf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.9, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_log_lpmf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_log_lpmf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_lpmf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.9, 0.918, 0.936, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_log_lpmf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_log_lpmf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_lpmf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.9, 0.918, 0.936, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_log_lpmf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_log_lpmf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_lpmf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.9, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_log_lpmf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_log_lpmf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_lpmf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.9, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(
            |x| neg_binomial_2_log_lpmf(&[3i64; 3][..], x[0], &x[1..4]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_log_lpmf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_lpmf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.9, 0.918, 0.936, 4.0];
        let v = try_value_and_grad(
            |x| neg_binomial_2_log_lpmf(&[3i64; 3][..], &x[0..3], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_log_lpmf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_lpmf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.9, 0.918, 0.936, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(
            |x| neg_binomial_2_log_lpmf(&[3i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_log_lpmf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_log_lpmf(vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_2_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lpmf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += neg_binomial_2_lpmf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lpmf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lpmf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lpmf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lpmf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lpmf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lpmf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lpmf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lpmf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lpmf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lpmf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lpmf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lpmf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_2_lpmf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(|x| neg_binomial_2_lpmf(&[3i64; 3][..], x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lpmf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lpmf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0];
        let v = try_value_and_grad(|x| neg_binomial_2_lpmf(&[3i64; 3][..], &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lpmf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lpmf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 4.0, 4.08, 4.16];
        let v = try_value_and_grad(
            |x| neg_binomial_2_lpmf(&[3i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_2_lpmf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_2_lpmf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_cdf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= neg_binomial_cdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_cdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_cdf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_cdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_cdf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_cdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_cdf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_cdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_cdf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_cdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_cdf(&[3i64; 3][..], x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_cdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_cdf(&[3i64; 3][..], &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_cdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_cdf(&[3i64; 3][..], &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= neg_binomial_cdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lccdf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += neg_binomial_lccdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lccdf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lccdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lccdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lccdf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lccdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lccdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lccdf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lccdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lccdf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lccdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lccdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lccdf(&[3i64; 3][..], x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lccdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lccdf(&[3i64; 3][..], &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lccdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| neg_binomial_lccdf(&[3i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lccdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lcdf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += neg_binomial_lcdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lcdf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lcdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lcdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lcdf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lcdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lcdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lcdf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lcdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lcdf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lcdf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lcdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lcdf(&[3i64; 3][..], x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lcdf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lcdf(&[3i64; 3][..], &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lcdf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| neg_binomial_lcdf(&[3i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lcdf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn neg_binomial_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lpmf(3i64, x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += neg_binomial_lpmf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lpmf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lpmf(3i64, x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lpmf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lpmf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lpmf(3i64, &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lpmf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lpmf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lpmf(3i64, &x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lpmf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lpmf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lpmf(&[3i64; 3][..], x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lpmf(3i64, x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "neg_binomial_lpmf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [2.5, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(|x| neg_binomial_lpmf(&[3i64; 3][..], x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lpmf(3i64, x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lpmf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7];
        let v = try_value_and_grad(|x| neg_binomial_lpmf(&[3i64; 3][..], &x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lpmf(3i64, x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lpmf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6, 0.7, 0.714, 0.728];
        let v = try_value_and_grad(
            |x| neg_binomial_lpmf(&[3i64; 3][..], &x[0..3], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += neg_binomial_lpmf(3i64, x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("neg_binomial_lpmf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn normal_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| normal_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= normal_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= normal_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| normal_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= normal_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= normal_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| normal_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= normal_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= normal_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| normal_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= normal_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= normal_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn normal_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| normal_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += normal_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| normal_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| normal_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| normal_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn normal_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| normal_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += normal_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| normal_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| normal_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| normal_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn normal_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| normal_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += normal_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| normal_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| normal_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| normal_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| normal_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += normal_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("normal_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn pareto_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| pareto_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= pareto_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| pareto_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| pareto_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| pareto_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn pareto_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| pareto_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += pareto_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| pareto_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| pareto_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| pareto_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn pareto_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| pareto_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += pareto_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| pareto_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| pareto_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| pareto_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn pareto_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| pareto_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += pareto_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| pareto_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| pareto_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| pareto_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| pareto_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn pareto_type_2_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= pareto_type_2_cdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_cdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_cdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_cdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_cdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(
            |x| pareto_type_2_cdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_cdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| pareto_type_2_cdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= pareto_type_2_cdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("pareto_type_2_cdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn pareto_type_2_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += pareto_type_2_lccdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lccdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lccdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lccdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lccdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(
            |x| pareto_type_2_lccdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| pareto_type_2_lccdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lccdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn pareto_type_2_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += pareto_type_2_lcdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lcdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lcdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lcdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lcdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(
            |x| pareto_type_2_lcdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| pareto_type_2_lcdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lcdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn pareto_type_2_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += pareto_type_2_lpdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lpdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lpdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| pareto_type_2_lpdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| pareto_type_2_lpdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(
            |x| pareto_type_2_lpdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| pareto_type_2_lpdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += pareto_type_2_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "pareto_type_2_lpdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn poisson_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| poisson_cdf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= poisson_cdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_cdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| poisson_cdf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= poisson_cdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_cdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| poisson_cdf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= poisson_cdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_cdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| poisson_cdf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= poisson_cdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_cdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn poisson_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| poisson_lccdf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += poisson_lccdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lccdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| poisson_lccdf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lccdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lccdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| poisson_lccdf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lccdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lccdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| poisson_lccdf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lccdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lccdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn poisson_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| poisson_lcdf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += poisson_lcdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lcdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| poisson_lcdf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lcdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lcdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| poisson_lcdf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lcdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lcdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| poisson_lcdf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lcdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lcdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn poisson_log_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.9];
        let v = try_value_and_grad(|x| poisson_log_lpmf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += poisson_log_lpmf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_log_lpmf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.9, 0.918, 0.936];
        let v = try_value_and_grad(|x| poisson_log_lpmf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_log_lpmf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_log_lpmf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.9];
        let v = try_value_and_grad(|x| poisson_log_lpmf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_log_lpmf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_log_lpmf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.9, 0.918, 0.936];
        let v = try_value_and_grad(|x| poisson_log_lpmf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_log_lpmf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_log_lpmf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn poisson_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| poisson_lpmf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += poisson_lpmf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lpmf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| poisson_lpmf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lpmf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lpmf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| poisson_lpmf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lpmf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lpmf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| poisson_lpmf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += poisson_lpmf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("poisson_lpmf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn rayleigh_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5];
        let v = try_value_and_grad(|x| rayleigh_cdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= rayleigh_cdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_cdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| rayleigh_cdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= rayleigh_cdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_cdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5];
        let v = try_value_and_grad(|x| rayleigh_cdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= rayleigh_cdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_cdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| rayleigh_cdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= rayleigh_cdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_cdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn rayleigh_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5];
        let v = try_value_and_grad(|x| rayleigh_lccdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += rayleigh_lccdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lccdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| rayleigh_lccdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lccdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lccdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5];
        let v = try_value_and_grad(|x| rayleigh_lccdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lccdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lccdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| rayleigh_lccdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lccdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lccdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn rayleigh_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5];
        let v = try_value_and_grad(|x| rayleigh_lcdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += rayleigh_lcdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lcdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| rayleigh_lcdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lcdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lcdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5];
        let v = try_value_and_grad(|x| rayleigh_lcdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lcdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lcdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| rayleigh_lcdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lcdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lcdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn rayleigh_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5];
        let v = try_value_and_grad(|x| rayleigh_lpdf(x[0], x[1]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += rayleigh_lpdf(x[0], x[1])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lpdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| rayleigh_lpdf(x[0], &x[1..4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lpdf(x[0], x[1 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lpdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5];
        let v = try_value_and_grad(|x| rayleigh_lpdf(&x[0..3], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lpdf(x[0 + i], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lpdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52];
        let v = try_value_and_grad(|x| rayleigh_lpdf(&x[0..3], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += rayleigh_lpdf(x[0 + i], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("rayleigh_lpdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn scaled_inv_chi_square_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 3.0, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= scaled_inv_chi_square_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_cdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 3.0, 0.8, 0.8160000000000001, 0.8320000000000001];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= scaled_inv_chi_square_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_cdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 3.0, 3.06, 3.12, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= scaled_inv_chi_square_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_cdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            3.0,
            3.06,
            3.12,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= scaled_inv_chi_square_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_cdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= scaled_inv_chi_square_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_cdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            0.51,
            0.52,
            3.0,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= scaled_inv_chi_square_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_cdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 3.06, 3.12, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= scaled_inv_chi_square_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_cdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            0.51,
            0.52,
            3.0,
            3.06,
            3.12,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| scaled_inv_chi_square_cdf(&x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= scaled_inv_chi_square_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_cdf(vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn scaled_inv_chi_square_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 3.0, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += scaled_inv_chi_square_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 3.0, 0.8, 0.8160000000000001, 0.8320000000000001];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lccdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 3.0, 3.06, 3.12, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lccdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            3.0,
            3.06,
            3.12,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| scaled_inv_chi_square_lccdf(x[0], &x[1..4], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lccdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lccdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            0.51,
            0.52,
            3.0,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| scaled_inv_chi_square_lccdf(&x[0..3], x[3], &x[4..7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lccdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 3.06, 3.12, 0.8];
        let v = try_value_and_grad(
            |x| scaled_inv_chi_square_lccdf(&x[0..3], &x[3..6], x[6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lccdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            0.51,
            0.52,
            3.0,
            3.06,
            3.12,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| scaled_inv_chi_square_lccdf(&x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lccdf(vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn scaled_inv_chi_square_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 3.0, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += scaled_inv_chi_square_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 3.0, 0.8, 0.8160000000000001, 0.8320000000000001];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lcdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 3.0, 3.06, 3.12, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lcdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            3.0,
            3.06,
            3.12,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lcdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lcdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            0.51,
            0.52,
            3.0,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lcdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 3.06, 3.12, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lcdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            0.51,
            0.52,
            3.0,
            3.06,
            3.12,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| scaled_inv_chi_square_lcdf(&x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lcdf(vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn scaled_inv_chi_square_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.5, 3.0, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += scaled_inv_chi_square_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lpdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 3.0, 0.8, 0.8160000000000001, 0.8320000000000001];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lpdf(scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 3.0, 3.06, 3.12, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lpdf(scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            3.0,
            3.06,
            3.12,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lpdf(scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lpdf(vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            0.51,
            0.52,
            3.0,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lpdf(vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [0.5, 0.51, 0.52, 3.0, 3.06, 3.12, 0.8];
        let v = try_value_and_grad(|x| scaled_inv_chi_square_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lpdf(vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            0.5,
            0.51,
            0.52,
            3.0,
            3.06,
            3.12,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| scaled_inv_chi_square_lpdf(&x[0..3], &x[3..6], &x[6..9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += scaled_inv_chi_square_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "scaled_inv_chi_square_lpdf(vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn skew_double_exponential_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 0.3];
        let v = try_value_and_grad(|x| skew_double_exponential_cdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= skew_double_exponential_cdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(x[0], x[1], x[2], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(x[0], x[1], &x[2..5], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(x[0], x[1], &x[2..5], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(x[0], &x[1..4], x[4], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(x[0], &x[1..4], x[4], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(x[0], &x[1..4], &x[4..7], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(&x[0..3], x[3], x[4], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(&x[0..3], x[3], x[4], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(&x[0..3], x[3], &x[4..7], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(&x[0..3], &x[3..6], x[6], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312,
        ];
        let v = try_value_and_grad(
            |x| skew_double_exponential_cdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_double_exponential_cdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_cdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn skew_double_exponential_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(x[0], x[1], x[2], x[3]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += skew_double_exponential_lccdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(x[0], x[1], x[2], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(x[0], x[1], &x[2..5], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(x[0], x[1], &x[2..5], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(x[0], &x[1..4], x[4], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(x[0], &x[1..4], x[4], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(x[0], &x[1..4], &x[4..7], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(&x[0..3], x[3], x[4], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(&x[0..3], x[3], x[4], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(&x[0..3], x[3], &x[4..7], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(&x[0..3], &x[3..6], x[6], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312,
        ];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lccdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lccdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn skew_double_exponential_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 0.3];
        let v = try_value_and_grad(|x| skew_double_exponential_lcdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += skew_double_exponential_lcdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(x[0], x[1], x[2], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(x[0], x[1], &x[2..5], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(x[0], x[1], &x[2..5], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(x[0], &x[1..4], x[4], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(x[0], &x[1..4], x[4], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(x[0], &x[1..4], &x[4..7], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(&x[0..3], x[3], x[4], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(&x[0..3], x[3], x[4], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(&x[0..3], x[3], &x[4..7], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(&x[0..3], &x[3..6], x[6], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312,
        ];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lcdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lcdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn skew_double_exponential_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 0.3];
        let v = try_value_and_grad(|x| skew_double_exponential_lpdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += skew_double_exponential_lpdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(x[0], x[1], x[2], &x[3..6]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(x[0], x[1], &x[2..5], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(x[0], x[1], &x[2..5], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(x[0], &x[1..4], x[4], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(x[0], &x[1..4], x[4], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(x[0], &x[1..4], &x[4..7], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(&x[0..3], x[3], x[4], x[5]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(&x[0..3], x[3], x[4], &x[5..8]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(&x[0..3], x[3], &x[4..7], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(&x[0..3], &x[3..6], x[6], x[7]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 0.3, 0.306, 0.312];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 0.3, 0.306, 0.312,
        ];
        let v = try_value_and_grad(
            |x| skew_double_exponential_lpdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_double_exponential_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_double_exponential_lpdf(vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn skew_normal_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_cdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= skew_normal_cdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_cdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_cdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_cdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_cdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_cdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_cdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_cdf(x[0], &x[1..4], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_cdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_cdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_cdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_cdf(&x[0..3], x[3], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_cdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_cdf(&x[0..3], &x[3..6], x[6], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_cdf(&x[0..3], &x[3..6], &x[6..9], x[9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_cdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| skew_normal_cdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= skew_normal_cdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("skew_normal_cdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn skew_normal_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lccdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += skew_normal_lccdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lccdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lccdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lccdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lccdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lccdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lccdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lccdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lccdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lccdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lccdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lccdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lccdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lccdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(
            |x| skew_normal_lccdf(&x[0..3], &x[3..6], &x[6..9], x[9]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lccdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| skew_normal_lccdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("skew_normal_lccdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn skew_normal_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lcdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += skew_normal_lcdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lcdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lcdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lcdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lcdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lcdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lcdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lcdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lcdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lcdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lcdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lcdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lcdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lcdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lcdf(&x[0..3], &x[3..6], &x[6..9], x[9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lcdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| skew_normal_lcdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("skew_normal_lcdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn skew_normal_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lpdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += skew_normal_lpdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lpdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lpdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lpdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lpdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lpdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lpdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lpdf(x[0], &x[1..4], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lpdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| skew_normal_lpdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lpdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lpdf(&x[0..3], x[3], &x[4..7], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lpdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(
            |x| skew_normal_lpdf(&x[0..3], &x[3..6], x[6], &x[7..10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0];
        let v = try_value_and_grad(|x| skew_normal_lpdf(&x[0..3], &x[3..6], &x[6..9], x[9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "skew_normal_lpdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| skew_normal_lpdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += skew_normal_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("skew_normal_lpdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn std_normal_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [-3.0];
        let v = try_value_and_grad(|x| std_normal_cdf(x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= std_normal_cdf(x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("std_normal_cdf(scalar)", v, s, &mut failures);
    }

    {
        let p = [-3.0, -3.06, -3.12];
        let v = try_value_and_grad(|x| std_normal_cdf(&x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= std_normal_cdf(x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("std_normal_cdf(vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn std_normal_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [-3.0];
        let v = try_value_and_grad(|x| std_normal_lccdf(x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += std_normal_lccdf(x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("std_normal_lccdf(scalar)", v, s, &mut failures);
    }

    {
        let p = [-3.0, -3.06, -3.12];
        let v = try_value_and_grad(|x| std_normal_lccdf(&x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += std_normal_lccdf(x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("std_normal_lccdf(vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn std_normal_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [-3.0];
        let v = try_value_and_grad(|x| std_normal_lcdf(x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += std_normal_lcdf(x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("std_normal_lcdf(scalar)", v, s, &mut failures);
    }

    {
        let p = [-3.0, -3.06, -3.12];
        let v = try_value_and_grad(|x| std_normal_lcdf(&x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += std_normal_lcdf(x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("std_normal_lcdf(vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn std_normal_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [-3.0];
        let v = try_value_and_grad(|x| std_normal_lpdf(x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += std_normal_lpdf(x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("std_normal_lpdf(scalar)", v, s, &mut failures);
    }

    {
        let p = [-3.0, -3.06, -3.12];
        let v = try_value_and_grad(|x| std_normal_lpdf(&x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += std_normal_lpdf(x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("std_normal_lpdf(vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn student_t_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 3.0, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_cdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= student_t_cdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_cdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_cdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_cdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_cdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_cdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_cdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_cdf(x[0], &x[1..4], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_cdf(scalar, vec, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_cdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_cdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_cdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_cdf(&x[0..3], x[3], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_cdf(vec, scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_cdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_cdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_cdf(&x[0..3], &x[3..6], x[6], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_cdf(vec, vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_cdf(&x[0..3], &x[3..6], &x[6..9], x[9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_cdf(vec, vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| student_t_cdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= student_t_cdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_cdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn student_t_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 3.0, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lccdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += student_t_lccdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lccdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lccdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lccdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lccdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lccdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lccdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lccdf(x[0], &x[1..4], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lccdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lccdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lccdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lccdf(&x[0..3], x[3], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lccdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lccdf(&x[0..3], &x[3..6], x[6], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lccdf(&x[0..3], &x[3..6], &x[6..9], x[9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lccdf(vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| student_t_lccdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lccdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lccdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn student_t_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 3.0, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lcdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += student_t_lcdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lcdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lcdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lcdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lcdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lcdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lcdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lcdf(x[0], &x[1..4], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lcdf(scalar, vec, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lcdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lcdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lcdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lcdf(&x[0..3], x[3], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lcdf(vec, scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lcdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lcdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lcdf(&x[0..3], &x[3..6], x[6], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lcdf(vec, vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lcdf(&x[0..3], &x[3..6], &x[6..9], x[9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lcdf(vec, vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| student_t_lcdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lcdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lcdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn student_t_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 3.0, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lpdf(x[0], x[1], x[2], x[3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += student_t_lpdf(x[0], x[1], x[2], x[3])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lpdf(x[0], x[1], x[2], &x[3..6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0], x[1], x[2], x[3 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lpdf(x[0], x[1], &x[2..5], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0], x[1], x[2 + i], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lpdf(x[0], x[1], &x[2..5], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0], x[1], x[2 + i], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lpdf(x[0], &x[1..4], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0], x[1 + i], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lpdf(x[0], &x[1..4], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0], x[1 + i], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lpdf(x[0], &x[1..4], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0], x[1 + i], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lpdf(x[0], &x[1..4], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0], x[1 + i], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lpdf(scalar, vec, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lpdf(&x[0..3], x[3], x[4], x[5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0 + i], x[3], x[4], x[5])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lpdf(&x[0..3], x[3], x[4], &x[5..8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0 + i], x[3], x[4], x[5 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lpdf(&x[0..3], x[3], &x[4..7], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0 + i], x[3], x[4 + i], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lpdf(&x[0..3], x[3], &x[4..7], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0 + i], x[3], x[4 + i], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lpdf(vec, scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 2.0];
        let v = try_value_and_grad(|x| student_t_lpdf(&x[0..3], &x[3..6], x[6], x[7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0 + i], x[3 + i], x[6], x[7])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "student_t_lpdf(vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 2.0, 2.04, 2.08];
        let v = try_value_and_grad(|x| student_t_lpdf(&x[0..3], &x[3..6], x[6], &x[7..10]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0 + i], x[3 + i], x[6], x[7 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lpdf(vec, vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0];
        let v = try_value_and_grad(|x| student_t_lpdf(&x[0..3], &x[3..6], &x[6..9], x[9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lpdf(vec, vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [
            1.5, 1.53, 1.56, 3.0, 3.06, 3.12, 0.5, 0.51, 0.52, 2.0, 2.04, 2.08,
        ];
        let v = try_value_and_grad(
            |x| student_t_lpdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += student_t_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("student_t_lpdf(vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn uniform_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= uniform_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= uniform_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 0.0, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= uniform_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 0.0, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= uniform_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= uniform_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= uniform_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 0.0, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= uniform_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 0.0, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= uniform_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn uniform_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += uniform_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 0.0, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 0.0, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 0.0, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 0.0, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn uniform_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += uniform_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 0.0, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 0.0, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 0.0, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 0.0, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn uniform_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [0.3, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += uniform_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 0.0, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.0, 0.0, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 0.0, 0.0, 1.0];
        let v = try_value_and_grad(|x| uniform_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [0.3, 0.306, 0.312, 0.0, 0.0, 0.0, 1.0, 1.02, 1.04];
        let v = try_value_and_grad(|x| uniform_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += uniform_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("uniform_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn von_mises_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| von_mises_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= von_mises_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= von_mises_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| von_mises_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= von_mises_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= von_mises_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| von_mises_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= von_mises_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= von_mises_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| von_mises_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= von_mises_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= von_mises_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn von_mises_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| von_mises_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += von_mises_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "von_mises_lccdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| von_mises_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| von_mises_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| von_mises_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn von_mises_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| von_mises_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += von_mises_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "von_mises_lcdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| von_mises_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| von_mises_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| von_mises_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn von_mises_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.3, 0.5, 1.2];
        let v = try_value_and_grad(|x| von_mises_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += von_mises_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "von_mises_lpdf(scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.3, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| von_mises_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2];
        let v = try_value_and_grad(|x| von_mises_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2];
        let v = try_value_and_grad(|x| von_mises_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.3, 1.326, 1.352, 0.5, 0.51, 0.52, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| von_mises_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += von_mises_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("von_mises_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn weibull_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 1.2];
        let v = try_value_and_grad(|x| weibull_cdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= weibull_cdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_cdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_cdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= weibull_cdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_cdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 1.2];
        let v = try_value_and_grad(|x| weibull_cdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= weibull_cdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_cdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_cdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= weibull_cdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_cdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 1.2];
        let v = try_value_and_grad(|x| weibull_cdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= weibull_cdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_cdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_cdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= weibull_cdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_cdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 1.2];
        let v = try_value_and_grad(|x| weibull_cdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= weibull_cdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_cdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_cdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= weibull_cdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_cdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn weibull_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 1.2];
        let v = try_value_and_grad(|x| weibull_lccdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += weibull_lccdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lccdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lccdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lccdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lccdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 1.2];
        let v = try_value_and_grad(|x| weibull_lccdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lccdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lccdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lccdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lccdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lccdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 1.2];
        let v = try_value_and_grad(|x| weibull_lccdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lccdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lccdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lccdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lccdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lccdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 1.2];
        let v = try_value_and_grad(|x| weibull_lccdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lccdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lccdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lccdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lccdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lccdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn weibull_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 1.2];
        let v = try_value_and_grad(|x| weibull_lcdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += weibull_lcdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lcdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lcdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lcdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lcdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 1.2];
        let v = try_value_and_grad(|x| weibull_lcdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lcdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lcdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lcdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lcdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lcdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 1.2];
        let v = try_value_and_grad(|x| weibull_lcdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lcdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lcdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lcdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lcdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lcdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 1.2];
        let v = try_value_and_grad(|x| weibull_lcdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lcdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lcdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lcdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lcdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lcdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn weibull_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.5, 2.5, 1.2];
        let v = try_value_and_grad(|x| weibull_lpdf(x[0], x[1], x[2]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += weibull_lpdf(x[0], x[1], x[2])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lpdf(scalar, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lpdf(x[0], x[1], &x[2..5]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lpdf(x[0], x[1], x[2 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lpdf(scalar, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 1.2];
        let v = try_value_and_grad(|x| weibull_lpdf(x[0], &x[1..4], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lpdf(x[0], x[1 + i], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lpdf(scalar, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 2.5, 2.55, 2.6, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lpdf(x[0], &x[1..4], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lpdf(x[0], x[1 + i], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lpdf(scalar, vec, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 1.2];
        let v = try_value_and_grad(|x| weibull_lpdf(&x[0..3], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lpdf(x[0 + i], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lpdf(vec, scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lpdf(&x[0..3], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lpdf(x[0 + i], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lpdf(vec, scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 1.2];
        let v = try_value_and_grad(|x| weibull_lpdf(&x[0..3], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lpdf(x[0 + i], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lpdf(vec, vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [1.5, 1.53, 1.56, 2.5, 2.55, 2.6, 1.2, 1.224, 1.248];
        let v = try_value_and_grad(|x| weibull_lpdf(&x[0..3], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += weibull_lpdf(x[0 + i], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("weibull_lpdf(vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn wiener_lpdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [1.2, 2.0, 0.3, 0.5, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], x[1], x[2], x[3], x[4]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += wiener_lpdf(x[0], x[1], x[2], x[3], x[4])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            2.0,
            0.3,
            0.5,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], x[1], x[2], x[3], &x[4..7]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1], x[2], x[3], x[4 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 2.0, 0.3, 0.5, 0.51, 0.52, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], x[1], x[2], &x[3..6], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1], x[2], x[3 + i], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            2.0,
            0.3,
            0.5,
            0.51,
            0.52,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], x[1], x[2], &x[3..6], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1], x[2], x[3 + i], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 2.0, 0.3, 0.306, 0.312, 0.5, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], x[1], &x[2..5], x[5], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1], x[2 + i], x[5], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            2.0,
            0.3,
            0.306,
            0.312,
            0.5,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], x[1], &x[2..5], x[5], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1], x[2 + i], x[5], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 2.0, 0.3, 0.306, 0.312, 0.5, 0.51, 0.52, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], x[1], &x[2..5], &x[5..8], x[8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1], x[2 + i], x[5 + i], x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            2.0,
            0.3,
            0.306,
            0.312,
            0.5,
            0.51,
            0.52,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(x[0], x[1], &x[2..5], &x[5..8], &x[8..11]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1], x[2 + i], x[5 + i], x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 2.0, 2.04, 2.08, 0.3, 0.5, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], &x[1..4], x[4], x[5], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1 + i], x[4], x[5], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            2.0,
            2.04,
            2.08,
            0.3,
            0.5,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], &x[1..4], x[4], x[5], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1 + i], x[4], x[5], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 2.0, 2.04, 2.08, 0.3, 0.5, 0.51, 0.52, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], &x[1..4], x[4], &x[5..8], x[8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1 + i], x[4], x[5 + i], x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            2.0,
            2.04,
            2.08,
            0.3,
            0.5,
            0.51,
            0.52,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(x[0], &x[1..4], x[4], &x[5..8], &x[8..11]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1 + i], x[4], x[5 + i], x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 2.0, 2.04, 2.08, 0.3, 0.306, 0.312, 0.5, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(x[0], &x[1..4], &x[4..7], x[7], x[8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1 + i], x[4 + i], x[7], x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            2.0,
            2.04,
            2.08,
            0.3,
            0.306,
            0.312,
            0.5,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(x[0], &x[1..4], &x[4..7], x[7], &x[8..11]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1 + i], x[4 + i], x[7], x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2, 2.0, 2.04, 2.08, 0.3, 0.306, 0.312, 0.5, 0.51, 0.52, 0.8,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(x[0], &x[1..4], &x[4..7], &x[7..10], x[10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1 + i], x[4 + i], x[7 + i], x[10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            2.0,
            2.04,
            2.08,
            0.3,
            0.306,
            0.312,
            0.5,
            0.51,
            0.52,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(x[0], &x[1..4], &x[4..7], &x[7..10], &x[10..13]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0], x[1 + i], x[4 + i], x[7 + i], x[10 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(scalar, vec, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 1.224, 1.248, 2.0, 0.3, 0.5, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(&x[0..3], x[3], x[4], x[5], x[6]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3], x[4], x[5], x[6])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, scalar, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            1.224,
            1.248,
            2.0,
            0.3,
            0.5,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(|x| wiener_lpdf(&x[0..3], x[3], x[4], x[5], &x[6..9]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3], x[4], x[5], x[6 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, scalar, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 1.224, 1.248, 2.0, 0.3, 0.5, 0.51, 0.52, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(&x[0..3], x[3], x[4], &x[5..8], x[8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3], x[4], x[5 + i], x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, scalar, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            1.224,
            1.248,
            2.0,
            0.3,
            0.5,
            0.51,
            0.52,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], x[3], x[4], &x[5..8], &x[8..11]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3], x[4], x[5 + i], x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, scalar, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 1.224, 1.248, 2.0, 0.3, 0.306, 0.312, 0.5, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(&x[0..3], x[3], &x[4..7], x[7], x[8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3], x[4 + i], x[7], x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, scalar, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            1.224,
            1.248,
            2.0,
            0.3,
            0.306,
            0.312,
            0.5,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], x[3], &x[4..7], x[7], &x[8..11]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3], x[4 + i], x[7], x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, scalar, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2, 1.224, 1.248, 2.0, 0.3, 0.306, 0.312, 0.5, 0.51, 0.52, 0.8,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], x[3], &x[4..7], &x[7..10], x[10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3], x[4 + i], x[7 + i], x[10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, scalar, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            1.224,
            1.248,
            2.0,
            0.3,
            0.306,
            0.312,
            0.5,
            0.51,
            0.52,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], x[3], &x[4..7], &x[7..10], &x[10..13]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3], x[4 + i], x[7 + i], x[10 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, scalar, vec, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [1.2, 1.224, 1.248, 2.0, 2.04, 2.08, 0.3, 0.5, 0.8];
        let v = try_value_and_grad(|x| wiener_lpdf(&x[0..3], &x[3..6], x[6], x[7], x[8]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3 + i], x[6], x[7], x[8])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, vec, scalar, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            1.224,
            1.248,
            2.0,
            2.04,
            2.08,
            0.3,
            0.5,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], &x[3..6], x[6], x[7], &x[8..11]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3 + i], x[6], x[7], x[8 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, vec, scalar, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2, 1.224, 1.248, 2.0, 2.04, 2.08, 0.3, 0.5, 0.51, 0.52, 0.8,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], &x[3..6], x[6], &x[7..10], x[10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3 + i], x[6], x[7 + i], x[10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, vec, scalar, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            1.224,
            1.248,
            2.0,
            2.04,
            2.08,
            0.3,
            0.5,
            0.51,
            0.52,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], &x[3..6], x[6], &x[7..10], &x[10..13]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3 + i], x[6], x[7 + i], x[10 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, vec, scalar, vec, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2, 1.224, 1.248, 2.0, 2.04, 2.08, 0.3, 0.306, 0.312, 0.5, 0.8,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], &x[3..6], &x[6..9], x[9], x[10]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9], x[10])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, vec, vec, scalar, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            1.224,
            1.248,
            2.0,
            2.04,
            2.08,
            0.3,
            0.306,
            0.312,
            0.5,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], &x[3..6], &x[6..9], x[9], &x[10..13]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9], x[10 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, vec, vec, scalar, vec)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2, 1.224, 1.248, 2.0, 2.04, 2.08, 0.3, 0.306, 0.312, 0.5, 0.51, 0.52, 0.8,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12], x[12]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i], x[12])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare(
            "wiener_lpdf(vec, vec, vec, vec, scalar)",
            v,
            s,
            &mut failures,
        );
    }

    {
        let p = [
            1.2,
            1.224,
            1.248,
            2.0,
            2.04,
            2.08,
            0.3,
            0.306,
            0.312,
            0.5,
            0.51,
            0.52,
            0.8,
            0.8160000000000001,
            0.8320000000000001,
        ];
        let v = try_value_and_grad(
            |x| wiener_lpdf(&x[0..3], &x[3..6], &x[6..9], &x[9..12], &x[12..15]),
            &p,
        );
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += wiener_lpdf(x[0 + i], x[3 + i], x[6 + i], x[9 + i], x[12 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("wiener_lpdf(vec, vec, vec, vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn yule_simon_cdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| yule_simon_cdf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..1 {
                    acc *= yule_simon_cdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_cdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| yule_simon_cdf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= yule_simon_cdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_cdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| yule_simon_cdf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= yule_simon_cdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_cdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| yule_simon_cdf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0 + 1.0;
                for i in 0..3 {
                    acc *= yule_simon_cdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_cdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn yule_simon_lccdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| yule_simon_lccdf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += yule_simon_lccdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lccdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| yule_simon_lccdf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lccdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lccdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| yule_simon_lccdf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lccdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lccdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| yule_simon_lccdf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lccdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lccdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn yule_simon_lcdf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| yule_simon_lcdf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += yule_simon_lcdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lcdf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| yule_simon_lcdf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lcdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lcdf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| yule_simon_lcdf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lcdf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lcdf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| yule_simon_lcdf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lcdf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lcdf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn yule_simon_lpmf_signatures() {
    let mut failures = Vec::new();

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| yule_simon_lpmf(3i64, x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..1 {
                    acc += yule_simon_lpmf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lpmf(scalar, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| yule_simon_lpmf(3i64, &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lpmf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lpmf(scalar, vec)", v, s, &mut failures);
    }

    {
        let p = [2.5];
        let v = try_value_and_grad(|x| yule_simon_lpmf(&[3i64; 3][..], x[0]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lpmf(3i64, x[0])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lpmf(vec, scalar)", v, s, &mut failures);
    }

    {
        let p = [2.5, 2.55, 2.6];
        let v = try_value_and_grad(|x| yule_simon_lpmf(&[3i64; 3][..], &x[0..3]), &p);
        let s = try_value_and_grad(
            |x| {
                let mut acc = x[0] * 0.0;
                for i in 0..3 {
                    acc += yule_simon_lpmf(3i64, x[0 + i])?;
                }
                Ok(acc)
            },
            &p,
        );
        compare("yule_simon_lpmf(vec, vec)", v, s, &mut failures);
    }

    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
