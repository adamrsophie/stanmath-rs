//! The `propto` forms (`_lupdf`, `_lupmf`): values against Stan Math 5.4.0, and a difference from
//! the full forms that does not depend on the variables.

// Reference values keep the 17 significant digits that Stan Math printed.
#![allow(clippy::excessive_precision)]

use stanmath_rs_autodiff::value_and_grad;
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
