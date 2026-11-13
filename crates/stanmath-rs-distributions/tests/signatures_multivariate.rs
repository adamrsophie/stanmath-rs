//! The array signatures of the multivariate and ordered densities (`array[] vector` and
//! `array[] row_vector` arguments in the Stan language; the vector and row vector forms are the
//! same Rust slice). Each array call must equal the sum of the single observation calls, in value
//! and gradient, for the `_lpdf` and `_lupdf` forms; the single calls are compared with Stan Math
//! by `stan_reference.rs` and `stan_reference_extra.rs`.

use stanmath_rs_autodiff::{ValueAndGrad, Var, try_value_and_grad};
use stanmath_rs_core::{Matrix, Result};
use stanmath_rs_distributions::*;

/// Three observations, three locations, then the six values of a lower triangular or symmetric
/// 3 by 3 matrix, then nu.
const P: [f64; 25] = [
    0.3, -0.5, 1.2, 0.1, 0.4, 0.9, -0.7, 0.2, 0.6, // y
    0.1, 0.0, 0.8, -0.2, 0.3, 0.5, 0.4, -0.1, 0.2, // mu
    1.4, 0.2, 1.1, -0.3, 0.5, 0.9, // matrix
    4.5,
];

fn rows<'t>(x: &[Var<'t>]) -> Vec<Vec<Var<'t>>> {
    x.chunks(3).map(<[Var<'t>]>::to_vec).collect()
}

fn low<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    let z = Var::constant(0.0);
    Matrix::from_rows(&[[v[0], z, z], [v[1], v[2], z], [v[3], v[4], v[5]]])
}

/// `L Lᵀ` of `low`, symmetric positive definite.
fn spd<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    let l = low(v);
    let mut m = Matrix::from_rows(&[[v[0]; 3]; 3]);
    for i in 0..3 {
        for j in 0..3 {
            m[(i, j)] = (0..3).map(|k| l[(i, k)] * l[(j, k)]).sum();
        }
    }
    m
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

fn check(
    label: &str,
    v: Result<ValueAndGrad>,
    s: Result<ValueAndGrad>,
    failures: &mut Vec<String>,
) {
    match (v, s) {
        (Ok(v), Ok(s))
            if close(v.value, s.value)
                && v.gradient
                    .iter()
                    .zip(&s.gradient)
                    .all(|(a, b)| close(*a, *b)) => {}
        (v, s) => failures.push(format!("{label}: {v:?} vs {s:?}")),
    }
}

#[test]
fn multivariate_array_signatures() {
    let mut failures = Vec::new();
    // Shapes: (observations, locations) = (array, one), (array, array), (one, array).
    macro_rules! shapes {
        ($label:literal, $array:ident, $mu_array:ident, $single:ident, $mat:ident,
         nu: $($nu:tt)?) => {{
            for (name, ny, nm) in [("array, one", 3, 1), ("array, array", 3, 3)] {
                let v = try_value_and_grad(
                    |x| {
                        let y = rows(&x[..9])[..ny].to_vec();
                        let mu = rows(&x[9..18])[..nm].to_vec();
                        $array(&y, $(x[24] $nu)? &mu, &$mat(&x[18..24]))
                    },
                    &P,
                );
                let s = try_value_and_grad(
                    |x| {
                        let y = rows(&x[..9]);
                        let mu = rows(&x[9..18]);
                        let mut acc = x[0] * 0.0;
                        for i in 0..ny {
                            let m = if nm == 1 { &mu[0] } else { &mu[i] };
                            acc += $single(&y[i], $(x[24] $nu)? m, &$mat(&x[18..24]))?;
                        }
                        Ok(acc)
                    },
                    &P,
                );
                check(&format!("{}({name})", $label), v, s, &mut failures);
            }
            let v = try_value_and_grad(
                |x| $mu_array(&x[..3], $(x[24] $nu)? &rows(&x[9..18]), &$mat(&x[18..24])),
                &P,
            );
            let s = try_value_and_grad(
                |x| {
                    let mu = rows(&x[9..18]);
                    let mut acc = x[0] * 0.0;
                    for m in &mu {
                        acc += $single(&x[..3], $(x[24] $nu)? m, &$mat(&x[18..24]))?;
                    }
                    Ok(acc)
                },
                &P,
            );
            check(&format!("{}(one, array)", $label), v, s, &mut failures);
        }};
    }
    shapes!("multi_normal_lpdf", multi_normal_lpdf_array, multi_normal_lpdf_mu_array, multi_normal_lpdf, spd, nu:);
    shapes!("multi_normal_lupdf", multi_normal_lupdf_array, multi_normal_lupdf_mu_array, multi_normal_lupdf, spd, nu:);
    shapes!("multi_normal_cholesky_lpdf", multi_normal_cholesky_lpdf_array, multi_normal_cholesky_lpdf_mu_array, multi_normal_cholesky_lpdf, low, nu:);
    shapes!("multi_normal_cholesky_lupdf", multi_normal_cholesky_lupdf_array, multi_normal_cholesky_lupdf_mu_array, multi_normal_cholesky_lupdf, low, nu:);
    shapes!("multi_normal_prec_lpdf", multi_normal_prec_lpdf_array, multi_normal_prec_lpdf_mu_array, multi_normal_prec_lpdf, spd, nu:);
    shapes!("multi_normal_prec_lupdf", multi_normal_prec_lupdf_array, multi_normal_prec_lupdf_mu_array, multi_normal_prec_lupdf, spd, nu:);
    shapes!("multi_student_t_lpdf", multi_student_t_lpdf_array, multi_student_t_lpdf_mu_array, multi_student_t_lpdf, spd, nu: ,);
    shapes!("multi_student_t_lupdf", multi_student_t_lupdf_array, multi_student_t_lupdf_mu_array, multi_student_t_lupdf, spd, nu: ,);
    shapes!("multi_student_t_cholesky_lpdf", multi_student_t_cholesky_lpdf_array, multi_student_t_cholesky_lpdf_mu_array, multi_student_t_cholesky_lpdf, low, nu: ,);
    shapes!("multi_student_t_cholesky_lupdf", multi_student_t_cholesky_lupdf_array, multi_student_t_cholesky_lupdf_mu_array, multi_student_t_cholesky_lupdf, low, nu: ,);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Three simplexes, then three concentration vectors.
const D: [f64; 18] = [
    0.2, 0.3, 0.5, 0.1, 0.6, 0.3, 0.4, 0.4, 0.2, // theta
    1.5, 2.0, 0.7, 3.0, 1.1, 2.4, 0.9, 1.3, 1.8, // alpha
];

#[test]
fn dirichlet_array_signatures() {
    let mut failures = Vec::new();
    macro_rules! shapes {
        ($label:literal, $array:ident, $alpha_array:ident, $single:ident) => {{
            for (name, na) in [("array, one", 1), ("array, array", 3)] {
                let v = try_value_and_grad(
                    |x| $array(&rows(&x[..9]), &rows(&x[9..18])[..na].to_vec()),
                    &D,
                );
                let s = try_value_and_grad(
                    |x| {
                        let (t, a) = (rows(&x[..9]), rows(&x[9..18]));
                        let mut acc = x[0] * 0.0;
                        for i in 0..3 {
                            acc += $single(&t[i], &a[if na == 1 { 0 } else { i }])?;
                        }
                        Ok(acc)
                    },
                    &D,
                );
                check(&format!("{}({name})", $label), v, s, &mut failures);
            }
            let v = try_value_and_grad(|x| $alpha_array(&x[..3], &rows(&x[9..18])), &D);
            let s = try_value_and_grad(
                |x| {
                    let mut acc = x[0] * 0.0;
                    for a in rows(&x[9..18]) {
                        acc += $single(&x[..3], &a)?;
                    }
                    Ok(acc)
                },
                &D,
            );
            check(&format!("{}(one, array)", $label), v, s, &mut failures);
        }};
    }
    shapes!(
        "dirichlet_lpdf",
        dirichlet_lpdf_array,
        dirichlet_lpdf_alpha_array,
        dirichlet_lpdf
    );
    shapes!(
        "dirichlet_lupdf",
        dirichlet_lupdf_array,
        dirichlet_lupdf_alpha_array,
        dirichlet_lupdf
    );
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Three linear predictors, then three increasing cutpoint vectors.
const O: [f64; 12] = [
    0.3, -0.8, 1.6, -1.0, 0.2, 1.5, -0.5, 0.0, 0.7, -2.0, -0.4, 2.2,
];

#[test]
fn ordered_array_signatures() {
    let mut failures = Vec::new();
    let y = [1_i64, 3, 4];
    macro_rules! shapes {
        ($label:literal, $array:ident, $single:ident) => {{
            let v = try_value_and_grad(|x| $array(&y[..], &x[..3], &rows(&x[3..12])), &O);
            let s = try_value_and_grad(
                |x| {
                    let c = rows(&x[3..12]);
                    let mut acc = x[0] * 0.0;
                    for i in 0..3 {
                        acc += $single(y[i], x[i], &c[i])?;
                    }
                    Ok(acc)
                },
                &O,
            );
            check($label, v, s, &mut failures);
        }};
    }
    shapes!(
        "ordered_logistic_lpmf",
        ordered_logistic_lpmf_array,
        ordered_logistic_lpmf
    );
    shapes!(
        "ordered_logistic_lupmf",
        ordered_logistic_lupmf_array,
        ordered_logistic_lupmf
    );
    shapes!(
        "ordered_probit_lpmf",
        ordered_probit_lpmf_array,
        ordered_probit_lpmf
    );
    shapes!(
        "ordered_probit_lupmf",
        ordered_probit_lupmf_array,
        ordered_probit_lupmf
    );
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
