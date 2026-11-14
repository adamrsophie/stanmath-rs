//! Values from `tools/stan-reference/stan_probe.cpp`, run against Stan Math 5.4.0 by the
//! "Stan Math probe" workflow on 2026-11-12. Each test pins one convention that the Functions
//! Reference leaves open.

// Reference values keep the 17 significant digits that Stan Math printed.
#![allow(clippy::excessive_precision)]

use stanmath_rs_core::Matrix;
use stanmath_rs_distributions::*;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

#[test]
fn normal_id_glm_reads_a_vector_x_as_one_column() {
    let y = [1.0, -0.5, 2.0];
    let x = [0.3, 1.2, -0.7];
    let v = normal_id_glm_lpdf_vector_x(&y[..], &x, 0.5, &[0.8], 1.2).unwrap();
    assert!(close(v, -6.1346136033292158), "{v}");
    let v = normal_id_glm_lpdf_vector_x(1.0, &x, 0.5, &[0.8], 1.2).unwrap();
    assert!(close(v, -3.7908636033292158), "{v}");
    assert!(normal_id_glm_lpdf_vector_x(&y[..], &x, 0.5, &[0.8, -0.2, 0.4], 1.2).is_err());
}

#[test]
fn gaussian_dlm_obs_reads_a_vector_v_as_the_diagonal() {
    let y = Matrix::from_rows(&[[0.5, 1.0, -0.3], [1.2, 0.1, 0.4]]);
    let f = Matrix::from_rows(&[[1.0, 0.3], [0.2, 1.0]]);
    let g = Matrix::from_rows(&[[0.9, 0.1], [0.0, 0.8]]);
    let w = Matrix::from_rows(&[[0.3, 0.05], [0.05, 0.2]]);
    let c0 = Matrix::from_rows(&[[1.0, 0.2], [0.2, 2.0]]);
    let m0 = [0.0, 0.5];
    let v = gaussian_dlm_obs_lpdf_vector_v(&y, &f, &g, &[0.5, 1.5], &w, &m0, &c0).unwrap();
    assert!((v - -7.9514266596732757).abs() < 1e-10, "{v}");
    let vm = Matrix::from_rows(&[[0.5, 0.0], [0.0, 1.5]]);
    let m = gaussian_dlm_obs_lpdf(&y, &f, &g, &vm, &w, &m0, &c0).unwrap();
    assert!((m - -7.9514266596732748).abs() < 1e-10, "{m}");
}

#[test]
fn normal_sufficient_takes_the_sum_of_squared_deviations() {
    // y = (1, 2.5, -0.5, 3): mean 1.5, sum of squared deviations 7.5.
    let v = normal_sufficient_lpdf(1.5, 7.5, 4i64, 0.7, 1.3).unwrap();
    assert!(close(v, -7.7015425516354012), "{v}");
    let direct = normal_lpdf(&[1.0, 2.5, -0.5, 3.0][..], 0.7, 1.3).unwrap();
    assert!(close(direct, -7.7015425516353995), "{direct}");
}

#[test]
fn lkj_cov_matches_stan() {
    let sigma = Matrix::from_rows(&[[2.0, 0.4, -0.3], [0.4, 1.5, 0.2], [-0.3, 0.2, 1.2]]);
    let v = lkj_cov_lpdf(&sigma, &[0.1, -0.2, 0.3], &[1.0, 0.8, 1.2], 1.5).unwrap();
    assert!(close(v, -4.6232239908126624), "{v}");
}

#[test]
fn edge_cases_of_the_newer_densities() {
    assert!(rayleigh_lpdf(0.0, 1.0).is_err());
    assert_eq!(
        gumbel_lpdf(0.5, 0.0, f64::INFINITY).unwrap(),
        f64::NEG_INFINITY
    );
    assert!(loglogistic_lpdf(0.0, 1.0, 2.0).is_err());
    assert!(loglogistic_lpdf(-0.5, 1.0, 2.0).is_err());
    let c = loglogistic_cdf(1.3, 0.9, 2.5).unwrap();
    assert!(close(c, 0.71490189009403016), "{c}");
}

#[test]
fn empty_sequences_still_validate_the_scalars() {
    let none: &[f64] = &[];
    let none_int: &[i64] = &[];
    assert!(normal_lpdf(none, 0.0, -1.0).is_err());
    assert!(normal_lpdf(none, f64::NAN, 1.0).is_err());
    assert!(normal_lpdf(none, &[0.0, 1.0][..], 1.0).is_err());
    assert!(normal_lpdf(none, none, -1.0).is_err());
    assert!(uniform_lpdf(none, 2.0, 1.0).is_err());
    assert!(binomial_lpmf(none_int, -1i64, 0.3).is_err());
    assert!(binomial_lpmf(none_int, 5i64, 1.5).is_err());
    assert!(binomial_lpmf(7i64, 5i64, none).is_err());
    assert!(normal_lcdf(none, 0.0, -1.0).is_err());
    assert!(poisson_lpmf(none_int, -1.0).is_err());
    assert!(student_t_lpdf(none, -1.0, 0.0, 1.0).is_err());
    assert!(gamma_lccdf(none, 2.0, -1.0).is_err());
    assert!(neg_binomial_2_lpmf(none_int, 1.0, -1.0).is_err());
    // Checks between an empty argument and a scalar do not apply.
    assert_eq!(pareto_lpdf(none, 2.0, 1.5).unwrap(), 0.0);
    assert_eq!(pareto_lpdf(1.0, 2.0, 1.5).unwrap(), f64::NEG_INFINITY);
    assert_eq!(wiener_lpdf(none, 1.5, 2.0, 0.5, 0.3).unwrap(), 0.0);
    // Valid scalars with an empty sequence still give zero.
    assert_eq!(normal_lpdf(none, 0.0, 1.0).unwrap(), 0.0);
}

#[test]
fn cdf_supports_at_zero() {
    assert_eq!(rayleigh_lcdf(0.0, 1.0).unwrap(), f64::NEG_INFINITY);
    assert_eq!(rayleigh_lccdf(0.0, 1.0).unwrap(), 0.0);
    assert_eq!(loglogistic_cdf(0.0, 1.0, 2.0).unwrap(), 0.0);
    assert!(loglogistic_cdf(-0.5, 1.0, 2.0).is_err());
}

#[test]
fn empty_responses_of_the_glm_ordered_and_categorical_families() {
    let none_int: &[i64] = &[];
    let none: &[f64] = &[];
    let x0 = Matrix::<f64>::zeros(0, 2);
    let beta = [0.4, -0.3];
    // Stan returns zero here before checking the other arguments.
    assert_eq!(
        poisson_log_glm_lpmf(none_int, &x0, f64::NAN, &beta).unwrap(),
        0.0
    );
    assert_eq!(
        bernoulli_logit_glm_lpmf(none_int, &x0, 0.0, &[f64::NAN, 0.1]).unwrap(),
        0.0
    );
    assert_eq!(
        ordered_logistic_lpmf(none_int, none, &[1.0, 0.0]).unwrap(),
        0.0
    );
    // These validate the other arguments first.
    assert!(normal_id_glm_lpdf(none, &x0, 0.0, &beta, -1.0).is_err());
    assert!(ordered_probit_lpmf(none_int, none, &[1.0, 0.0]).is_err());
    assert!(categorical_lpmf(none_int, &[0.5, 0.6]).is_err());
    assert!(discrete_range_lpmf(none_int, 5i64, 2i64).is_err());
    assert!(discrete_range_lpmf(3i64, 5i64, 2i64).is_err());
    // The other four GLMs.
    assert_eq!(
        binomial_logit_glm_lpmf(none_int, none_int, &x0, f64::NAN, &beta).unwrap(),
        0.0
    );
    assert_eq!(
        neg_binomial_2_log_glm_lpmf(none_int, &x0, 0.0, &beta, -1.0).unwrap(),
        0.0
    );
    assert!(neg_binomial_2_log_glm_lpmf(none_int, &x0, f64::NAN, &beta, 1.0).is_err());
    let beta23 = Matrix::from_rows(&[[0.1, -0.2, 0.3], [0.4, 0.0, -0.1]]);
    assert_eq!(
        categorical_logit_glm_lpmf(none_int, &x0, &[f64::NAN, 0.0, 0.1], &beta23).unwrap(),
        0.0
    );
    assert!(ordered_logistic_glm_lpmf(none_int, &x0, &beta, &[1.0, 0.0]).is_err());
}
