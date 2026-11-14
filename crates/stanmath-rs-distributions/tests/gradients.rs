//! Reverse-mode gradients of the distributions with hand-derived partials, checked against
//! sixth-order finite differences at interior points of the support.

use stanmath_rs_autodiff::{Var, check_gradient};
use stanmath_rs_distributions::*;

/// Asserts that the reverse-mode gradient of `f` at `x` matches finite differences.
fn check<F>(name: &str, f: F, x: &[f64])
where
    F: for<'t> Fn(&[Var<'t>]) -> Var<'t>,
{
    let c = check_gradient(f, x);
    assert!(c.value.is_finite(), "{name}: value {}", c.value);
    assert!(
        c.passes(1e-6, 1e-8),
        "{name} at {x:?}: autodiff {:?}, finite differences {:?}",
        c.autodiff,
        c.finite_diff
    );
}

/// Checks the log density, log CDF, and log complementary CDF of one family at one point.
macro_rules! family {
    ($lpdf:ident, $lcdf:ident, $lccdf:ident, $x:expr, |$p:ident| ($($a:expr),+ $(,)?)) => {
        check(stringify!($lpdf), |$p| $lpdf($($a),+).unwrap(), &$x);
        check(stringify!($lcdf), |$p| $lcdf($($a),+).unwrap(), &$x);
        check(stringify!($lccdf), |$p| $lccdf($($a),+).unwrap(), &$x);
    };
}

#[test]
fn continuous_families() {
    family!(
        skew_normal_lpdf,
        skew_normal_lcdf,
        skew_normal_lccdf,
        [0.4, -0.2, 1.3, 0.8],
        |p| (p[0], p[1], p[2], p[3])
    );
    family!(
        double_exponential_lpdf,
        double_exponential_lcdf,
        double_exponential_lccdf,
        [0.4, -0.2, 1.3],
        |p| (p[0], p[1], p[2])
    );
    family!(
        double_exponential_lpdf,
        double_exponential_lcdf,
        double_exponential_lccdf,
        [-0.9, -0.2, 1.3],
        |p| (p[0], p[1], p[2])
    );
    family!(
        von_mises_lpdf,
        von_mises_lcdf,
        von_mises_lccdf,
        [0.4, -0.2, 1.7],
        |p| (p[0], p[1], p[2])
    );
    family!(
        rayleigh_lpdf,
        rayleigh_lcdf,
        rayleigh_lccdf,
        [1.1, 0.8],
        |p| (p[0], p[1])
    );
    family!(
        pareto_lpdf,
        pareto_lcdf,
        pareto_lccdf,
        [2.5, 1.2, 2.3],
        |p| (p[0], p[1], p[2])
    );
    family!(
        pareto_type_2_lpdf,
        pareto_type_2_lcdf,
        pareto_type_2_lccdf,
        [1.5, 0.3, 1.2, 2.1],
        |p| (p[0], p[1], p[2], p[3])
    );
    family!(
        frechet_lpdf,
        frechet_lcdf,
        frechet_lccdf,
        [1.4, 2.2, 0.9],
        |p| (p[0], p[1], p[2])
    );
    family!(
        gumbel_lpdf,
        gumbel_lcdf,
        gumbel_lccdf,
        [0.7, 0.2, 1.3],
        |p| (p[0], p[1], p[2])
    );
    family!(
        chi_square_lpdf,
        chi_square_lcdf,
        chi_square_lccdf,
        [2.1, 3.4],
        |p| (p[0], p[1])
    );
    family!(
        inv_chi_square_lpdf,
        inv_chi_square_lcdf,
        inv_chi_square_lccdf,
        [0.6, 3.4],
        |p| (p[0], p[1])
    );
    family!(
        scaled_inv_chi_square_lpdf,
        scaled_inv_chi_square_lcdf,
        scaled_inv_chi_square_lccdf,
        [0.8, 3.4, 0.7],
        |p| (p[0], p[1], p[2])
    );
    family!(
        exp_mod_normal_lpdf,
        exp_mod_normal_lcdf,
        exp_mod_normal_lccdf,
        [1.1, 0.2, 0.8, 1.5],
        |p| (p[0], p[1], p[2], p[3])
    );
    family!(
        skew_double_exponential_lpdf,
        skew_double_exponential_lcdf,
        skew_double_exponential_lccdf,
        [0.7, 0.2, 1.1, 0.3],
        |p| (p[0], p[1], p[2], p[3])
    );
    family!(
        skew_double_exponential_lpdf,
        skew_double_exponential_lcdf,
        skew_double_exponential_lccdf,
        [-0.4, 0.2, 1.1, 0.3],
        |p| (p[0], p[1], p[2], p[3])
    );
    family!(
        loglogistic_lpdf,
        loglogistic_lcdf,
        loglogistic_lccdf,
        [1.3, 0.9, 2.5],
        |p| (p[0], p[1], p[2])
    );
    family!(
        beta_proportion_lpdf,
        beta_proportion_lcdf,
        beta_proportion_lccdf,
        [0.35, 0.4, 6.0],
        |p| (p[0], p[1], p[2])
    );
}

#[test]
fn wiener_densities() {
    check(
        "wiener_lpdf",
        |p| wiener_lpdf(p[0], p[1], p[2], p[3], p[4]).unwrap(),
        &[1.2, 1.5, 0.2, 0.45, 0.6],
    );
    check(
        "wiener5_lpdf",
        |p| wiener5_lpdf(p[0], p[1], p[2], p[3], p[4], p[5]).unwrap(),
        &[1.2, 1.5, 0.2, 0.45, 0.6, 0.7],
    );
}

#[test]
fn discrete_families() {
    let x = [2.3, 1.7];
    check(
        "beta_binomial_lpmf",
        |p| beta_binomial_lpmf(3i64, 10i64, p[0], p[1]).unwrap(),
        &x,
    );
    check(
        "beta_binomial_lcdf",
        |p| beta_binomial_lcdf(3i64, 10i64, p[0], p[1]).unwrap(),
        &x,
    );
    check(
        "beta_binomial_lccdf",
        |p| beta_binomial_lccdf(3i64, 10i64, p[0], p[1]).unwrap(),
        &x,
    );
    check(
        "neg_binomial_lpmf",
        |p| neg_binomial_lpmf(4i64, p[0], p[1]).unwrap(),
        &x,
    );
    check(
        "neg_binomial_lcdf",
        |p| neg_binomial_lcdf(4i64, p[0], p[1]).unwrap(),
        &x,
    );
    check(
        "neg_binomial_lccdf",
        |p| neg_binomial_lccdf(4i64, p[0], p[1]).unwrap(),
        &x,
    );
    let y = [2.5, 3.1, 1.8];
    check(
        "beta_neg_binomial_lpmf",
        |p| beta_neg_binomial_lpmf(4i64, p[0], p[1], p[2]).unwrap(),
        &y,
    );
    check(
        "beta_neg_binomial_lcdf",
        |p| beta_neg_binomial_lcdf(4i64, p[0], p[1], p[2]).unwrap(),
        &y,
    );
    check(
        "beta_neg_binomial_lccdf",
        |p| beta_neg_binomial_lccdf(4i64, p[0], p[1], p[2]).unwrap(),
        &y,
    );
    check(
        "yule_simon_lpmf",
        |p| yule_simon_lpmf(3i64, p[0]).unwrap(),
        &[1.6],
    );
    check(
        "yule_simon_lcdf",
        |p| yule_simon_lcdf(3i64, p[0]).unwrap(),
        &[1.6],
    );
    check(
        "yule_simon_lccdf",
        |p| yule_simon_lccdf(3i64, p[0]).unwrap(),
        &[1.6],
    );
}

#[test]
fn discrete_vector_parameters() {
    let theta = [0.2, 0.55, 0.7, 0.35];
    check(
        "poisson_binomial_lpmf",
        |p| poisson_binomial_lpmf(2i64, p).unwrap(),
        &theta,
    );
    check(
        "poisson_binomial_lcdf",
        |p| poisson_binomial_lcdf(2i64, p).unwrap(),
        &theta,
    );
    check(
        "poisson_binomial_lccdf",
        |p| poisson_binomial_lccdf(2i64, p).unwrap(),
        &theta,
    );
    // eta, then three ordered cutpoints.
    let eta_c = [0.3, -1.0, 0.4, 1.5];
    for y in 1..=4i64 {
        check(
            "ordered_logistic_lpmf",
            |p| ordered_logistic_lpmf(y, p[0], &p[1..]).unwrap(),
            &eta_c,
        );
        check(
            "ordered_probit_lpmf",
            |p| ordered_probit_lpmf(y, p[0], &p[1..]).unwrap(),
            &eta_c,
        );
    }
    let ns = [2i64, 0, 5];
    check(
        "multinomial_logit_lpmf",
        |p| multinomial_logit_lpmf(&ns, p).unwrap(),
        &[0.3, -0.8, 0.1],
    );
    check(
        "dirichlet_multinomial_lpmf",
        |p| dirichlet_multinomial_lpmf(&ns, p).unwrap(),
        &[1.3, 0.8, 2.1],
    );
}
