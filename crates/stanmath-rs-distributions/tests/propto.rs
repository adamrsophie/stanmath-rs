//! The `propto` forms (`_lupdf`, `_lupmf`): values against Stan Math 5.4.0, and a difference from
//! the full forms that does not depend on the variables.

// Reference values keep the 17 significant digits that Stan Math printed.
#![allow(clippy::excessive_precision)]

use stanmath_rs_autodiff::{Var, value_and_grad};
use stanmath_rs_core::Matrix;
use stanmath_rs_distributions::*;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + b.abs())
}

#[test]
fn propto_values_match_stan() {
    let ys = [1.0, 2.5, -0.5, 3.0];
    assert_eq!(normal_lupdf(&ys[..], 0.7, 1.3).unwrap(), 0.0);
    let cases: [(&str, f64, f64); 9] = [
        (
            "normal, mu a variable",
            value_and_grad(|v| normal_lupdf(&ys[..], v[0], 1.3).unwrap(), &[0.7]).value,
            -2.9763313609467446,
        ),
        (
            "normal, sigma a variable",
            value_and_grad(|v| normal_lupdf(&ys[..], 0.7, v[0]).unwrap(), &[1.3]).value,
            -4.0257884188167088,
        ),
        (
            "poisson",
            value_and_grad(|v| poisson_lupmf(&[2i64, 0, 5][..], v[0]).unwrap(), &[2.2]).value,
            -1.0807984774501085,
        ),
        (
            "gamma, alpha a variable",
            value_and_grad(|v| gamma_lupdf(1.7, v[0], 0.8).unwrap(), &[2.5]).value,
            -0.046599372165187891,
        ),
        (
            "gamma, beta a variable",
            value_and_grad(|v| gamma_lupdf(1.7, 2.5, v[0]).unwrap(), &[0.8]).value,
            -1.9178588782855244,
        ),
        (
            "gamma, y a variable",
            value_and_grad(|v| gamma_lupdf(v[0], 2.5, 0.8).unwrap(), &[1.7]).value,
            -0.56405762340674437,
        ),
        (
            "binomial",
            value_and_grad(|v| binomial_lupmf(4i64, 10i64, v[0]).unwrap(), &[0.3]).value,
            -6.9559408809361383,
        ),
        (
            "student_t, mu a variable",
            value_and_grad(|v| student_t_lupdf(1.1, 4.0, v[0], 1.5).unwrap(), &[0.2]).value,
            -0.2154442406026309,
        ),
        (
            "full normal",
            normal_lpdf(&ys[..], 0.7, 1.3).unwrap(),
            -7.7015425516353995,
        ),
    ];
    for (name, got, want) in cases {
        assert!(close(got, want), "{name}: {got} vs {want}");
    }
}

/// `f` with its one argument a variable at `x`.
fn var1(
    f: impl for<'t> FnOnce(&[stanmath_rs_autodiff::Var<'t>]) -> stanmath_rs_autodiff::Var<'t>,
    x: f64,
) -> f64 {
    value_and_grad(f, &[x]).value
}

#[test]
fn propto_values_of_the_other_kernels_match_stan() {
    let cases: [(&str, f64, f64); 22] = [
        (
            "std_normal y",
            var1(|v| std_normal_lupdf(v[0]).unwrap(), 0.4),
            -0.080000000000000016,
        ),
        (
            "cauchy sigma",
            var1(|v| cauchy_lupdf(0.5, -0.2, v[0]).unwrap(), 1.3),
            -0.51696061233350643,
        ),
        (
            "cauchy mu",
            var1(|v| cauchy_lupdf(0.5, v[0], 1.3).unwrap(), -0.2),
            -0.25459634786601543,
        ),
        (
            "logistic mu",
            var1(|v| logistic_lupdf(0.5, v[0], 0.8).unwrap(), 0.1),
            -1.4481539683602134,
        ),
        (
            "exponential y",
            var1(|v| exponential_lupdf(v[0], 1.9).unwrap(), 0.7),
            -1.3299999999999998,
        ),
        (
            "exponential beta",
            var1(|v| exponential_lupdf(0.7, v[0]).unwrap(), 1.9),
            -0.68814611382760515,
        ),
        (
            "inv_gamma beta",
            var1(|v| inv_gamma_lupdf(1.2, 2.5, v[0]).unwrap(), 0.8),
            -1.224525544952191,
        ),
        (
            "inv_gamma alpha",
            var1(|v| inv_gamma_lupdf(1.2, v[0], 0.8).unwrap(), 2.5),
            -1.4806671975372845,
        ),
        (
            "beta a",
            var1(|v| beta_lupdf(0.3, v[0], 3.5).unwrap(), 2.0),
            2.7538411632927806,
        ),
        (
            "beta y",
            var1(|v| beta_lupdf(v[0], 2.0, 3.5).unwrap(), 0.3),
            -2.095660164172767,
        ),
        (
            "lognormal y",
            var1(|v| lognormal_lupdf(v[0], 0.2, 0.6).unwrap(), 1.4),
            -1.2812783689486584,
        ),
        (
            "lognormal sigma",
            var1(|v| lognormal_lupdf(1.4, 0.2, v[0]).unwrap(), 0.6),
            -0.43398050856145476,
        ),
        (
            "uniform y",
            var1(|v| uniform_lupdf(v[0], 0.1, 1.5).unwrap(), 0.4),
            0.0,
        ),
        (
            "uniform alpha",
            var1(|v| uniform_lupdf(0.4, v[0], 1.5).unwrap(), 0.1),
            -0.33647223662121289,
        ),
        (
            "poisson_log alpha",
            var1(|v| poisson_log_lupmf(3i64, v[0]).unwrap(), 0.8),
            0.17445907150753226,
        ),
        (
            "binomial_logit alpha",
            var1(|v| binomial_logit_lupmf(4i64, 10i64, v[0]).unwrap(), -0.4),
            -6.7301525239995259,
        ),
        (
            "neg_binomial_2 mu",
            var1(|v| neg_binomial_2_lupmf(3i64, v[0], 1.5).unwrap(), 2.0),
            -2.949794154387074,
        ),
        (
            "neg_binomial_2 phi",
            var1(|v| neg_binomial_2_lupmf(3i64, 2.0, v[0]).unwrap(), 1.5),
            -4.2464763568172774,
        ),
        (
            "neg_binomial_2_log eta",
            var1(|v| neg_binomial_2_log_lupmf(3i64, v[0], 1.5).unwrap(), 0.6),
            -2.9945148130015662,
        ),
        (
            "weibull alpha",
            var1(|v| weibull_lupdf(1.2, v[0], 0.9).unwrap(), 1.5),
            -0.88493405784712087,
        ),
        (
            "gumbel mu",
            var1(|v| gumbel_lupdf(0.5, v[0], 2.0).unwrap(), 1.0),
            -1.0340254166877414,
        ),
        (
            "bernoulli_logit alpha",
            var1(|v| bernoulli_logit_lupmf(1i64, v[0]).unwrap(), 0.3),
            -0.55435524446852713,
        ),
    ];
    for (name, got, want) in cases {
        assert!(close(got, want), "{name}: {got} vs {want}");
    }
}

#[test]
fn glm_propto_values_match_stan() {
    let x = Matrix::from_rows(&[[0.5, -1.0], [1.2, 0.3], [-0.4, 0.8]]);
    let beta = [0.4, -0.3];
    let (gy, gb, gn, gc) = ([2i64, 0, 5], [1i64, 0, 1], [4i64, 3, 6], [1i64, 3, 2]);
    let gyr = [0.7, -1.1, 2.0];
    let cuts = [-0.5, 0.7];
    let alpha3 = [0.1, -0.2, 0.3];
    let b23 = [[0.1, -0.2, 0.3], [0.4, 0.0, -0.1]];
    let cases: [(&str, f64, f64); 9] = [
        (
            "poisson_log_glm",
            value_and_grad(
                |v| poisson_log_glm_lupmf(&gy[..], &x, 0.3, v).unwrap(),
                &beta,
            )
            .value,
            -4.0240938797715096,
        ),
        (
            "bernoulli_logit_glm",
            value_and_grad(
                |v| bernoulli_logit_glm_lupmf(&gb[..], &x, 0.3, v).unwrap(),
                &beta,
            )
            .value,
            -2.2120125952279732,
        ),
        (
            "binomial_logit_glm",
            value_and_grad(
                |v| binomial_logit_glm_lupmf(&gy[..], &gn[..], &x, 0.3, v).unwrap(),
                &beta,
            )
            .value,
            -10.740328431852411,
        ),
        (
            "neg_binomial_2_log_glm",
            value_and_grad(
                |v| neg_binomial_2_log_glm_lupmf(&gy[..], &x, 0.3, v, 1.5).unwrap(),
                &beta,
            )
            .value,
            -11.083325835945297,
        ),
        (
            "normal_id_glm, beta a variable",
            value_and_grad(
                |v| normal_id_glm_lupdf(&gyr[..], &x, 0.3, v, 0.8).unwrap(),
                &beta,
            )
            .value,
            -5.9563281250000006,
        ),
        (
            // beta fixes the scalar type, so it is a variable here too; sigma's terms are kept.
            "normal_id_glm, sigma a variable",
            value_and_grad(
                |v| normal_id_glm_lupdf(&gyr[..], &x, 0.3, &v[..2], v[2]).unwrap(),
                &[0.4, -0.3, 0.8],
            )
            .value,
            -5.2868974710573715,
        ),
        (
            "categorical_logit_glm",
            value_and_grad(
                |v| {
                    let b = Matrix::from_fn(2, 3, |i, j| v[3 + 3 * i + j]);
                    categorical_logit_glm_lupmf(&gc[..], &x, &v[..3], &b).unwrap()
                },
                &[
                    alpha3[0], alpha3[1], alpha3[2], b23[0][0], b23[0][1], b23[0][2], b23[1][0],
                    b23[1][1], b23[1][2],
                ],
            )
            .value,
            -3.5269526559487634,
        ),
        (
            "ordered_logistic_glm",
            value_and_grad(
                |v| ordered_logistic_glm_lupmf(&gc[..], &x, v, &cuts).unwrap(),
                &beta,
            )
            .value,
            -3.4634879770063729,
        ),
        (
            "neg_binomial_2_log_glm full",
            neg_binomial_2_log_glm_lpmf(&gy[..], &x, 0.3, &beta, 1.5).unwrap(),
            -7.6342716353258435,
        ),
    ];
    for (name, got, want) in cases {
        assert!(close(got, want), "{name}: {got} vs {want}");
    }
    // Every argument data gives zero.
    assert_eq!(poisson_log_glm_lupmf(&gy[..], &x, 0.3, &beta).unwrap(), 0.0);
    assert_eq!(
        normal_id_glm_lupdf(&gyr[..], &x, 0.3, &beta, 0.8).unwrap(),
        0.0
    );
    assert_eq!(
        ordered_logistic_glm_lupmf(&gc[..], &x, &beta, &cuts).unwrap(),
        0.0
    );
}

/// `full - propto` at `x` and at a second point, with the argument `j` the only variable: the
/// dropped terms must not depend on that argument. A term dropped although it depends on the
/// argument (a wrong dependency list) fails this.
fn check_gap(name: &str, j: usize, gap: impl Fn(f64) -> f64, x: f64) {
    let (a, b) = (gap(x), gap(x * 1.1 + 0.05));
    assert!(
        a.is_finite() && (a - b).abs() <= 1e-10 * (1.0 + a.abs()),
        "{name} arg {j}: {a} vs {b}"
    );
}

macro_rules! gap {
    ($n:expr, $j:expr, $x:expr, |$v:ident| $full:expr, $propto:expr) => {
        check_gap(
            $n,
            $j,
            |x| {
                value_and_grad(|$v| $full.unwrap(), &[x]).value
                    - value_and_grad(|$v| $propto.unwrap(), &[x]).value
            },
            $x,
        )
    };
}

/// With each argument in turn the only variable, `full - propto` does not depend on it. Also,
/// the `propto` form is zero when every argument is data.
macro_rules! only_data_terms_dropped {
    ($full:ident, $propto:ident, [$($i:expr),*], ($a:expr)) => {{
        let n = stringify!($propto);
        gap!(n, 0, $a, |v| $full($($i,)* v[0]), $propto($($i,)* v[0]));
        assert_eq!($propto($($i,)* $a).unwrap(), 0.0, "{n}");
    }};
    ($full:ident, $propto:ident, [$($i:expr),*], ($a:expr, $b:expr)) => {{
        let n = stringify!($propto);
        gap!(n, 0, $a, |v| $full($($i,)* v[0], $b), $propto($($i,)* v[0], $b));
        gap!(n, 1, $b, |v| $full($($i,)* $a, v[0]), $propto($($i,)* $a, v[0]));
        assert_eq!($propto($($i,)* $a, $b).unwrap(), 0.0, "{n}");
    }};
    ($full:ident, $propto:ident, [$($i:expr),*], ($a:expr, $b:expr, $c:expr)) => {{
        let n = stringify!($propto);
        gap!(n, 0, $a, |v| $full($($i,)* v[0], $b, $c), $propto($($i,)* v[0], $b, $c));
        gap!(n, 1, $b, |v| $full($($i,)* $a, v[0], $c), $propto($($i,)* $a, v[0], $c));
        gap!(n, 2, $c, |v| $full($($i,)* $a, $b, v[0]), $propto($($i,)* $a, $b, v[0]));
        assert_eq!($propto($($i,)* $a, $b, $c).unwrap(), 0.0, "{n}");
    }};
    ($full:ident, $propto:ident, [$($i:expr),*], ($a:expr, $b:expr, $c:expr, $d:expr)) => {{
        let n = stringify!($propto);
        gap!(n, 0, $a, |v| $full($($i,)* v[0], $b, $c, $d), $propto($($i,)* v[0], $b, $c, $d));
        gap!(n, 1, $b, |v| $full($($i,)* $a, v[0], $c, $d), $propto($($i,)* $a, v[0], $c, $d));
        gap!(n, 2, $c, |v| $full($($i,)* $a, $b, v[0], $d), $propto($($i,)* $a, $b, v[0], $d));
        gap!(n, 3, $d, |v| $full($($i,)* $a, $b, $c, v[0]), $propto($($i,)* $a, $b, $c, v[0]));
        assert_eq!($propto($($i,)* $a, $b, $c, $d).unwrap(), 0.0, "{n}");
    }};
}

#[test]
fn propto_drops_only_terms_free_of_the_variables() {
    only_data_terms_dropped!(normal_lpdf, normal_lupdf, [], (1.1, 0.3, 1.7));
    only_data_terms_dropped!(std_normal_lpdf, std_normal_lupdf, [], (0.4));
    only_data_terms_dropped!(student_t_lpdf, student_t_lupdf, [], (1.1, 4.0, 0.2, 1.5));
    only_data_terms_dropped!(cauchy_lpdf, cauchy_lupdf, [], (0.5, -0.2, 1.3));
    only_data_terms_dropped!(logistic_lpdf, logistic_lupdf, [], (0.5, 0.1, 0.8));
    only_data_terms_dropped!(exponential_lpdf, exponential_lupdf, [], (0.7, 1.9));
    only_data_terms_dropped!(gamma_lpdf, gamma_lupdf, [], (1.7, 2.5, 0.8));
    only_data_terms_dropped!(inv_gamma_lpdf, inv_gamma_lupdf, [], (1.2, 2.5, 0.8));
    only_data_terms_dropped!(beta_lpdf, beta_lupdf, [], (0.3, 2.0, 3.5));
    only_data_terms_dropped!(lognormal_lpdf, lognormal_lupdf, [], (1.4, 0.2, 0.6));
    only_data_terms_dropped!(uniform_lpdf, uniform_lupdf, [], (0.4, 0.1, 1.5));
    only_data_terms_dropped!(weibull_lpdf, weibull_lupdf, [], (1.2, 1.5, 0.9));
    only_data_terms_dropped!(binomial_lpmf, binomial_lupmf, [4i64, 10i64], (0.3));
    only_data_terms_dropped!(
        binomial_logit_lpmf,
        binomial_logit_lupmf,
        [4i64, 10i64],
        (-0.4)
    );
    only_data_terms_dropped!(poisson_lpmf, poisson_lupmf, [3i64], (2.2));
    only_data_terms_dropped!(poisson_log_lpmf, poisson_log_lupmf, [3i64], (0.8));
    only_data_terms_dropped!(
        neg_binomial_2_lpmf,
        neg_binomial_2_lupmf,
        [3i64],
        (2.0, 1.5)
    );
    only_data_terms_dropped!(
        neg_binomial_2_log_lpmf,
        neg_binomial_2_log_lupmf,
        [3i64],
        (0.6, 1.5)
    );
    only_data_terms_dropped!(bernoulli_logit_lpmf, bernoulli_logit_lupmf, [1i64], (0.3));
    only_data_terms_dropped!(gumbel_lpdf, gumbel_lupdf, [], (0.5, 1.0, 2.0));
}

#[test]
fn propto_checks_arguments_and_supports() {
    // Validation does not depend on propto.
    assert!(normal_lupdf(0.0, 0.0, -1.0).is_err());
    // Outside the support the value is -inf when an argument is a variable, as in Stan.
    let v = value_and_grad(|v| uniform_lupdf(2.0, v[0], 1.5).unwrap(), &[0.1]).value;
    assert_eq!(v, f64::NEG_INFINITY);
}

/// The value of `f` with its one argument an autodiff variable at `x`.
fn at(x: f64, f: impl for<'t> Fn(&[Var<'t>]) -> Var<'t>) -> f64 {
    value_and_grad(|v| f(v), &[x]).value
}

#[test]
fn remaining_kernel_propto_values_match_stan() {
    let cases: Vec<(&str, f64, f64)> = vec![
        (
            "skew_normal, y a variable",
            at(0.4, |v| skew_normal_lupdf(v[0], 0.1, 1.3, 0.8).unwrap()),
            0.11005981100662235,
        ),
        (
            "skew_normal, omega a variable",
            at(1.3, |v| skew_normal_lupdf(0.4, 0.1, v[0], 0.8).unwrap()),
            -0.15230445346086871,
        ),
        (
            "skew_normal, alpha a variable",
            at(0.8, |v| skew_normal_lupdf(0.4, 0.1, 1.3, v[0]).unwrap()),
            0.13668702994153359,
        ),
        (
            "skew_normal full",
            skew_normal_lpdf(0.4, 0.1, 1.3, 0.8).unwrap(),
            -1.0712429866655413,
        ),
        (
            "double_exponential, mu a variable",
            at(0.1, |v| double_exponential_lupdf(0.4, v[0], 1.3).unwrap()),
            -0.23076923076923078,
        ),
        (
            "double_exponential, sigma a variable",
            at(1.3, |v| double_exponential_lupdf(0.4, 0.1, v[0]).unwrap()),
            -0.49313349523672184,
        ),
        (
            "double_exponential full",
            double_exponential_lpdf(0.4, 0.1, 1.3).unwrap(),
            -1.1862806757966671,
        ),
        (
            "von_mises, mu a variable",
            at(0.1, |v| von_mises_lupdf(0.4, v[0], 2.0).unwrap()),
            1.910672978251212,
        ),
        (
            "von_mises, kappa a variable",
            at(2.0, |v| von_mises_lupdf(0.4, 0.1, v[0]).unwrap()),
            1.0866794367682557,
        ),
        (
            "von_mises full",
            von_mises_lpdf(0.4, 0.1, 2.0).unwrap(),
            -0.7511976296410896,
        ),
        (
            "rayleigh, y a variable",
            at(1.1, |v| rayleigh_lupdf(v[0], 0.8).unwrap()),
            -0.85000232019567501,
        ),
        (
            "rayleigh, sigma a variable",
            at(0.8, |v| rayleigh_lupdf(1.1, v[0]).unwrap()),
            -0.49902539737158058,
        ),
        (
            "rayleigh full",
            rayleigh_lpdf(1.1, 0.8).unwrap(),
            -0.40371521756725565,
        ),
        (
            "pareto, y a variable",
            at(2.0, |v| pareto_lupdf(v[0], 1.2, 3.0).unwrap()),
            -2.7725887222397811,
        ),
        (
            "pareto, y_min a variable",
            at(1.2, |v| pareto_lupdf(2.0, v[0], 3.0).unwrap()),
            0.5469646703818638,
        ),
        (
            "pareto, alpha a variable",
            at(3.0, |v| pareto_lupdf(2.0, 1.2, v[0]).unwrap()),
            -1.1270117631898076,
        ),
        (
            "pareto full",
            pareto_lpdf(2.0, 1.2, 3.0).unwrap(),
            -1.1270117631898076,
        ),
        (
            "pareto_type_2, mu a variable",
            at(0.5, |v| pareto_type_2_lupdf(2.0, v[0], 1.5, 3.0).unwrap()),
            -2.7725887222397811,
        ),
        (
            "pareto_type_2, lambda a variable",
            at(1.5, |v| pareto_type_2_lupdf(2.0, 0.5, v[0], 3.0).unwrap()),
            -3.1780538303479453,
        ),
        (
            "pareto_type_2, alpha a variable",
            at(3.0, |v| pareto_type_2_lupdf(2.0, 0.5, 1.5, v[0]).unwrap()),
            -1.6739764335716714,
        ),
        (
            "pareto_type_2 full",
            pareto_type_2_lpdf(2.0, 0.5, 1.5, 3.0).unwrap(),
            -2.0794415416798357,
        ),
        (
            "frechet, y a variable",
            at(1.3, |v| frechet_lupdf(v[0], 2.0, 0.9).unwrap()),
            -1.2663827342308756,
        ),
        (
            "frechet, alpha a variable",
            at(2.0, |v| frechet_lupdf(1.3, v[0], 0.9).unwrap()),
            -0.78395658498658272,
        ),
        (
            "frechet, sigma a variable",
            at(0.9, |v| frechet_lupdf(1.3, 2.0, v[0]).unwrap()),
            -0.69001097214405482,
        ),
        (
            "frechet full",
            frechet_lpdf(1.3, 2.0, 0.9).unwrap(),
            -0.78395658498658272,
        ),
        (
            "chi_square, y a variable",
            at(1.7, |v| chi_square_lupdf(v[0], 3.0).unwrap()),
            -0.58468587446891473,
        ),
        (
            "chi_square, nu a variable",
            at(3.0, |v| chi_square_lupdf(1.7, v[0]).unwrap()),
            -0.65362440767358754,
        ),
        (
            "chi_square full",
            chi_square_lpdf(1.7, 3.0).unwrap(),
            -1.5036244076735876,
        ),
        (
            "inv_chi_square, y a variable",
            at(0.7, |v| inv_chi_square_lupdf(v[0], 3.0).unwrap()),
            0.17740164556111682,
        ),
        (
            "inv_chi_square, nu a variable",
            at(3.0, |v| inv_chi_square_lupdf(0.7, v[0]).unwrap()),
            -0.027251173357841552,
        ),
        (
            "inv_chi_square full",
            inv_chi_square_lpdf(0.7, 3.0).unwrap(),
            -0.74153688764355585,
        ),
        (
            "scaled_inv_chi_square, y a variable",
            at(0.7, |v| {
                scaled_inv_chi_square_lupdf(v[0], 3.0, 0.8).unwrap()
            }),
            -0.47974121158174055,
        ),
        (
            "scaled_inv_chi_square, nu a variable",
            at(3.0, |v| {
                scaled_inv_chi_square_lupdf(0.7, v[0], 0.8).unwrap()
            }),
            -0.42019196572687795,
        ),
        (
            "scaled_inv_chi_square, s a variable",
            at(0.8, |v| {
                scaled_inv_chi_square_lupdf(0.7, 3.0, v[0]).unwrap()
            }),
            -2.040859225371201,
        ),
        (
            "scaled_inv_chi_square full",
            scaled_inv_chi_square_lpdf(0.7, 3.0, 0.8).unwrap(),
            -0.42019196572687789,
        ),
        (
            "exp_mod_normal, mu a variable",
            at(0.1, |v| exp_mod_normal_lupdf(0.4, v[0], 1.3, 0.8).unwrap()),
            -0.57055938562141995,
        ),
        (
            "exp_mod_normal, sigma a variable",
            at(1.3, |v| exp_mod_normal_lupdf(0.4, 0.1, v[0], 0.8).unwrap()),
            -0.57055938562141995,
        ),
        (
            "exp_mod_normal, lambda a variable",
            at(0.8, |v| exp_mod_normal_lupdf(0.4, 0.1, 1.3, v[0]).unwrap()),
            -0.79370293693562965,
        ),
        (
            "exp_mod_normal full",
            exp_mod_normal_lpdf(0.4, 0.1, 1.3, 0.8).unwrap(),
            -1.4868501174955751,
        ),
        (
            "skew_double_exponential, mu a variable",
            at(0.1, |v| {
                skew_double_exponential_lupdf(0.4, v[0], 1.3, 0.3).unwrap()
            }),
            -0.13846153846153847,
        ),
        (
            "skew_double_exponential, sigma a variable",
            at(1.3, |v| {
                skew_double_exponential_lupdf(0.4, 0.1, v[0], 0.3).unwrap()
            }),
            -0.40082580292902953,
        ),
        (
            "skew_double_exponential, tau a variable",
            at(0.3, |v| {
                skew_double_exponential_lupdf(0.4, 0.1, 1.3, v[0]).unwrap()
            }),
            -1.6991092867262068,
        ),
        (
            "skew_double_exponential full",
            skew_double_exponential_lpdf(0.4, 0.1, 1.3, 0.3).unwrap(),
            -1.2683263706337526,
        ),
        (
            "loglogistic, y a variable",
            at(1.3, |v| loglogistic_lupdf(v[0], 1.1, 2.5).unwrap()),
            -1.4536747090145306,
        ),
        (
            "loglogistic, alpha a variable",
            at(1.1, |v| loglogistic_lupdf(1.3, v[0], 2.5).unwrap()),
            -0.77565942665118781,
        ),
        (
            "loglogistic, beta a variable",
            at(2.5, |v| loglogistic_lupdf(1.3, 1.1, v[0]).unwrap()),
            -0.77565942665118781,
        ),
        (
            "loglogistic full",
            loglogistic_lpdf(1.3, 1.1, 2.5).unwrap(),
            -0.77565942665118781,
        ),
        (
            "beta_proportion, y a variable",
            at(0.3, |v| beta_proportion_lupdf(v[0], 0.4, 5.0).unwrap()),
            -1.9173226922034008,
        ),
        (
            "beta_proportion, mu a variable",
            at(0.4, |v| beta_proportion_lupdf(0.3, v[0], 5.0).unwrap()),
            -2.6104698727633462,
        ),
        (
            "beta_proportion, kappa a variable",
            at(5.0, |v| beta_proportion_lupdf(0.3, 0.4, v[0]).unwrap()),
            0.56758395758459956,
        ),
        (
            "beta_proportion full",
            beta_proportion_lpdf(0.3, 0.4, 5.0).unwrap(),
            0.56758395758459956,
        ),
        (
            "weibull, y a variable",
            at(1.2, |v| weibull_lupdf(v[0], 1.5, 0.9).unwrap()),
            -1.4484399394420246,
        ),
        (
            "weibull, sigma a variable",
            at(0.9, |v| weibull_lupdf(1.2, 1.5, v[0]).unwrap()),
            -1.3815599443522624,
        ),
        (
            "weibull full",
            weibull_lpdf(1.2, 1.5, 0.9).unwrap(),
            -0.88493405784712087,
        ),
        (
            "normal_sufficient, mu a variable",
            at(0.7, |v| {
                normal_sufficient_lupdf(1.5, 7.5, 4i64, v[0], 1.3).unwrap()
            }),
            -2.9763313609467454,
        ),
        (
            "normal_sufficient, sigma a variable",
            at(1.3, |v| {
                normal_sufficient_lupdf(1.5, 7.5, 4i64, 0.7, v[0]).unwrap()
            }),
            -4.0257884188167097,
        ),
        (
            "normal_sufficient, y_bar a variable",
            at(1.5, |v| {
                normal_sufficient_lupdf(v[0], 7.5, 4i64, 0.7, 1.3).unwrap()
            }),
            -2.9763313609467454,
        ),
        (
            "normal_sufficient full",
            normal_sufficient_lpdf(1.5, 7.5, 4i64, 0.7, 1.3).unwrap(),
            -7.7015425516354012,
        ),
        (
            "beta_binomial, alpha a variable",
            at(2.0, |v| {
                beta_binomial_lupmf(3i64, 10i64, v[0], 3.0).unwrap()
            }),
            -6.7264332225212691,
        ),
        (
            "beta_binomial full",
            beta_binomial_lpmf(3i64, 10i64, 2.0, 3.0).unwrap(),
            -1.9389414797392224,
        ),
        (
            "neg_binomial, alpha a variable",
            at(2.0, |v| neg_binomial_lupmf(3i64, v[0], 1.5).unwrap()),
            -2.3842290820345555,
        ),
        (
            "neg_binomial, beta a variable",
            at(1.5, |v| neg_binomial_lupmf(3i64, 2.0, v[0]).unwrap()),
            -3.7705234431544468,
        ),
        (
            "neg_binomial full",
            neg_binomial_lpmf(3i64, 2.0, 1.5).unwrap(),
            -2.3842290820345555,
        ),
        (
            "beta_neg_binomial, r a variable",
            at(2.5, |v| {
                beta_neg_binomial_lupmf(3i64, v[0], 3.0, 1.5).unwrap()
            }),
            -0.8357742165567712,
        ),
        (
            "beta_neg_binomial, alpha a variable",
            at(3.0, |v| {
                beta_neg_binomial_lupmf(3i64, 2.5, v[0], 1.5).unwrap()
            }),
            -0.8357742165567712,
        ),
        (
            "beta_neg_binomial, beta a variable",
            at(1.5, |v| {
                beta_neg_binomial_lupmf(3i64, 2.5, 3.0, v[0]).unwrap()
            }),
            -0.8357742165567712,
        ),
        (
            "beta_neg_binomial full",
            beta_neg_binomial_lpmf(3i64, 2.5, 3.0, 1.5).unwrap(),
            -2.6275336857848259,
        ),
        (
            "bernoulli, theta a variable",
            at(0.3, |v| bernoulli_lupmf(1i64, v[0]).unwrap()),
            -1.2039728043259361,
        ),
        (
            "bernoulli full",
            bernoulli_lpmf(1i64, 0.3).unwrap(),
            -1.2039728043259361,
        ),
        (
            "yule_simon, alpha a variable",
            at(2.0, |v| yule_simon_lupmf(3i64, v[0]).unwrap()),
            -3.401197381662155,
        ),
        (
            "yule_simon full",
            yule_simon_lpmf(3i64, 2.0).unwrap(),
            -2.7080502011022096,
        ),
        (
            "wiener4, alpha a variable",
            at(1.5, |v| wiener_lupdf(1.2, v[0], 0.3, 0.5, 0.4).unwrap()),
            -1.4121216265124346,
        ),
        (
            "wiener4 full",
            wiener_lpdf(1.2, 1.5, 0.3, 0.5, 0.4).unwrap(),
            -1.4121216265124346,
        ),
    ];
    let failures: Vec<String> = cases
        .iter()
        .filter(|&&(_, got, want)| !close(got, want))
        .map(|(name, got, want)| format!("{name}: {got:e} vs {want:e}"))
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
