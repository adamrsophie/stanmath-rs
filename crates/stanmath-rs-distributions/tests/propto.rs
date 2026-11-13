//! The `propto` forms (`_lupdf`, `_lupmf`): values against Stan Math 5.4.0, and gradients equal to
//! those of the full forms.

// Reference values keep the 17 significant digits that Stan Math printed.
#![allow(clippy::excessive_precision)]

use stanmath_rs_autodiff::{grad, value_and_grad};
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

/// With each argument in turn the only variable, the `propto` form has the gradient of the full
/// form. A term dropped although it depends on that argument fails this. Also, the `propto` form
/// is zero when every argument is data.
macro_rules! same_gradients {
    ($full:ident, $propto:ident, [$($i:expr),*], ($a:expr)) => {{
        let f = grad(|v| $full($($i,)* v[0]).unwrap(), &[$a]);
        let p = grad(|v| $propto($($i,)* v[0]).unwrap(), &[$a]);
        assert!(close(p[0], f[0]), "{} arg 0: {} vs {}", stringify!($propto), p[0], f[0]);
        assert_eq!($propto($($i,)* $a).unwrap(), 0.0, "{}", stringify!($propto));
    }};
    ($full:ident, $propto:ident, [$($i:expr),*], ($a:expr, $b:expr)) => {{
        let pairs = [
            (
                grad(|v| $full($($i,)* v[0], $b).unwrap(), &[$a]),
                grad(|v| $propto($($i,)* v[0], $b).unwrap(), &[$a]),
            ),
            (
                grad(|v| $full($($i,)* $a, v[0]).unwrap(), &[$b]),
                grad(|v| $propto($($i,)* $a, v[0]).unwrap(), &[$b]),
            ),
        ];
        for (j, (f, p)) in pairs.iter().enumerate() {
            assert!(close(p[0], f[0]), "{} arg {j}: {} vs {}", stringify!($propto), p[0], f[0]);
        }
        assert_eq!($propto($($i,)* $a, $b).unwrap(), 0.0, "{}", stringify!($propto));
    }};
    ($full:ident, $propto:ident, [$($i:expr),*], ($a:expr, $b:expr, $c:expr)) => {{
        let pairs = [
            (
                grad(|v| $full($($i,)* v[0], $b, $c).unwrap(), &[$a]),
                grad(|v| $propto($($i,)* v[0], $b, $c).unwrap(), &[$a]),
            ),
            (
                grad(|v| $full($($i,)* $a, v[0], $c).unwrap(), &[$b]),
                grad(|v| $propto($($i,)* $a, v[0], $c).unwrap(), &[$b]),
            ),
            (
                grad(|v| $full($($i,)* $a, $b, v[0]).unwrap(), &[$c]),
                grad(|v| $propto($($i,)* $a, $b, v[0]).unwrap(), &[$c]),
            ),
        ];
        for (j, (f, p)) in pairs.iter().enumerate() {
            assert!(close(p[0], f[0]), "{} arg {j}: {} vs {}", stringify!($propto), p[0], f[0]);
        }
        assert_eq!($propto($($i,)* $a, $b, $c).unwrap(), 0.0, "{}", stringify!($propto));
    }};
    ($full:ident, $propto:ident, [$($i:expr),*], ($a:expr, $b:expr, $c:expr, $d:expr)) => {{
        let pairs = [
            (
                grad(|v| $full($($i,)* v[0], $b, $c, $d).unwrap(), &[$a]),
                grad(|v| $propto($($i,)* v[0], $b, $c, $d).unwrap(), &[$a]),
            ),
            (
                grad(|v| $full($($i,)* $a, v[0], $c, $d).unwrap(), &[$b]),
                grad(|v| $propto($($i,)* $a, v[0], $c, $d).unwrap(), &[$b]),
            ),
            (
                grad(|v| $full($($i,)* $a, $b, v[0], $d).unwrap(), &[$c]),
                grad(|v| $propto($($i,)* $a, $b, v[0], $d).unwrap(), &[$c]),
            ),
            (
                grad(|v| $full($($i,)* $a, $b, $c, v[0]).unwrap(), &[$d]),
                grad(|v| $propto($($i,)* $a, $b, $c, v[0]).unwrap(), &[$d]),
            ),
        ];
        for (j, (f, p)) in pairs.iter().enumerate() {
            assert!(close(p[0], f[0]), "{} arg {j}: {} vs {}", stringify!($propto), p[0], f[0]);
        }
        assert_eq!($propto($($i,)* $a, $b, $c, $d).unwrap(), 0.0, "{}", stringify!($propto));
    }};
}

#[test]
fn propto_keeps_the_gradient() {
    same_gradients!(normal_lpdf, normal_lupdf, [], (1.1, 0.3, 1.7));
    same_gradients!(std_normal_lpdf, std_normal_lupdf, [], (0.4));
    same_gradients!(student_t_lpdf, student_t_lupdf, [], (1.1, 4.0, 0.2, 1.5));
    same_gradients!(cauchy_lpdf, cauchy_lupdf, [], (0.5, -0.2, 1.3));
    same_gradients!(logistic_lpdf, logistic_lupdf, [], (0.5, 0.1, 0.8));
    same_gradients!(exponential_lpdf, exponential_lupdf, [], (0.7, 1.9));
    same_gradients!(gamma_lpdf, gamma_lupdf, [], (1.7, 2.5, 0.8));
    same_gradients!(inv_gamma_lpdf, inv_gamma_lupdf, [], (1.2, 2.5, 0.8));
    same_gradients!(beta_lpdf, beta_lupdf, [], (0.3, 2.0, 3.5));
    same_gradients!(lognormal_lpdf, lognormal_lupdf, [], (1.4, 0.2, 0.6));
    same_gradients!(uniform_lpdf, uniform_lupdf, [], (0.4, 0.1, 1.5));
    same_gradients!(weibull_lpdf, weibull_lupdf, [], (1.2, 1.5, 0.9));
    same_gradients!(binomial_lpmf, binomial_lupmf, [4i64, 10i64], (0.3));
    same_gradients!(
        binomial_logit_lpmf,
        binomial_logit_lupmf,
        [4i64, 10i64],
        (-0.4)
    );
    same_gradients!(poisson_lpmf, poisson_lupmf, [3i64], (2.2));
    same_gradients!(poisson_log_lpmf, poisson_log_lupmf, [3i64], (0.8));
    same_gradients!(
        neg_binomial_2_lpmf,
        neg_binomial_2_lupmf,
        [3i64],
        (2.0, 1.5)
    );
    same_gradients!(
        neg_binomial_2_log_lpmf,
        neg_binomial_2_log_lupmf,
        [3i64],
        (0.6, 1.5)
    );
    same_gradients!(bernoulli_logit_lpmf, bernoulli_logit_lupmf, [1i64], (0.3));
    same_gradients!(gumbel_lpdf, gumbel_lupdf, [], (0.5, 1.0, 2.0));
}

#[test]
fn propto_checks_arguments_and_supports() {
    // Validation does not depend on propto.
    assert!(normal_lupdf(0.0, 0.0, -1.0).is_err());
    // Outside the support the value is -inf when an argument is a variable, as in Stan.
    let v = value_and_grad(|v| uniform_lupdf(2.0, v[0], 1.5).unwrap(), &[0.1]).value;
    assert_eq!(v, f64::NEG_INFINITY);
}
