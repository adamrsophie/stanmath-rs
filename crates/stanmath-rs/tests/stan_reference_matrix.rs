//! Values and gradients against Stan Math 5.4.0 for the matrix functions and constraining
//! transforms of the Stan language. `tests/data/stan_reference_matrix.csv` is produced by
//! `tools/stan-reference/stan_reference_matrix.cpp`, which documents the inputs. A matrix or vector
//! result is reduced to `sum_k (1 + k / 10) r_k` over its column major entries; a `_jacobian` case
//! adds the log Jacobian to that sum.

use stanmath_rs::prelude::*;

/// The weighted sum of a result's column major entries.
trait WSum<'t> {
    fn wsum(self) -> Result<Var<'t>>;
}

fn weighted<'t>(e: &[Var<'t>]) -> Var<'t> {
    e.iter()
        .enumerate()
        .map(|(k, &v)| v * (1.0 + 0.1 * k as f64))
        .sum()
}

impl<'t> WSum<'t> for Var<'t> {
    fn wsum(self) -> Result<Var<'t>> {
        Ok(self)
    }
}

impl<'t> WSum<'t> for Vec<Var<'t>> {
    fn wsum(self) -> Result<Var<'t>> {
        Ok(weighted(&self))
    }
}

impl<'t> WSum<'t> for Matrix<Var<'t>> {
    fn wsum(self) -> Result<Var<'t>> {
        Ok(weighted(self.as_slice()))
    }
}

impl<'t, W: WSum<'t>> WSum<'t> for Result<W> {
    fn wsum(self) -> Result<Var<'t>> {
        self?.wsum()
    }
}

fn mat<'t>(v: &[Var<'t>], r: usize, c: usize) -> Matrix<Var<'t>> {
    Matrix::from_column_major(r, c, v[..r * c].to_vec())
}

/// A 3 by 3 lower triangular matrix from six values, row by row.
fn low<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    let z = Var::constant(0.0);
    Matrix::from_rows(&[[v[0], z, z], [v[1], v[2], z], [v[3], v[4], v[5]]])
}

/// A 3 by 3 symmetric matrix from its lower triangle, row by row.
fn sym<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    Matrix::from_rows(&[[v[0], v[1], v[3]], [v[1], v[2], v[4]], [v[3], v[4], v[5]]])
}

/// A `_constrain_lp` result plus its log Jacobian.
fn with_lp<'t, W: WSum<'t>>(x: &[Var<'t>], f: impl FnOnce(&mut Var<'t>) -> W) -> Result<Var<'t>> {
    let mut lp = x[0] * 0.0;
    let s = f(&mut lp).wsum()?;
    Ok(s + lp)
}

// The Stan signatures each arm of `eval` calls: a slice argument stands for a vector, row
// vector, or one dimensional real array, and the `matrix_` arms are the matrix forms.
// stan: determinant(matrix)
// stan: log_determinant(matrix)
// stan: trace(matrix)
// stan: crossprod(matrix)
// stan: tcrossprod(matrix)
// stan: inverse(matrix)
// stan: generalized_inverse(matrix)
// stan: matrix_exp(matrix)
// stan: matrix_exp_multiply(matrix, matrix)
// stan: transpose(matrix)
// stan: diagonal(matrix)
// stan: rows_dot_self(matrix)
// stan: symmetrize_from_lower_tri(matrix)
// stan: singular_values(matrix)
// stan: svd_U(matrix)
// stan: svd_V(matrix)
// stan: qr_Q(matrix)
// stan: qr_R(matrix)
// stan: qr_thin_Q(matrix)
// stan: qr_thin_R(matrix)
// stan: sum(matrix)
// stan: prod(matrix)
// stan: mean(matrix)
// stan: variance(matrix)
// stan: sd(matrix)
// stan: log_sum_exp(matrix)
// stan: max(matrix)
// stan: min(matrix)
// stan: log_determinant_spd(matrix)
// stan: cholesky_decompose(matrix)
// stan: eigenvalues_sym(matrix)
// stan: eigenvectors_sym(matrix)
// stan: inverse_spd(matrix)
// stan: chol2inv(matrix)
// stan: multiply_lower_tri_self_transpose(matrix)
// stan: dot_self(vector)
// stan: dot_self(row_vector)
// stan: log_sum_exp(vector)
// stan: log_sum_exp(row_vector)
// stan: log_sum_exp(array[] real)
// stan: max(vector)
// stan: max(row_vector)
// stan: max(array[] real)
// stan: min(vector)
// stan: min(row_vector)
// stan: min(array[] real)
// stan: mean(vector)
// stan: mean(row_vector)
// stan: mean(array[] real)
// stan: norm1(vector)
// stan: norm1(row_vector)
// stan: norm1(array[] real)
// stan: norm2(vector)
// stan: norm2(row_vector)
// stan: norm2(array[] real)
// stan: prod(vector)
// stan: prod(row_vector)
// stan: prod(array[] real)
// stan: sd(vector)
// stan: sd(row_vector)
// stan: sd(array[] real)
// stan: sum(vector)
// stan: sum(row_vector)
// stan: sum(array[] real)
// stan: variance(vector)
// stan: variance(row_vector)
// stan: variance(array[] real)
// stan: cumulative_sum(vector)
// stan: cumulative_sum(row_vector)
// stan: cumulative_sum(array[] real)
// stan: log_softmax(vector)
// stan: log_softmax(row_vector)
// stan: softmax(vector)
// stan: softmax(row_vector)
// stan: reverse(vector)
// stan: reverse(row_vector)
// stan: reverse(array[] real)
// stan: sort_asc(vector)
// stan: sort_asc(row_vector)
// stan: sort_asc(array[] real)
// stan: sort_desc(vector)
// stan: sort_desc(row_vector)
// stan: sort_desc(array[] real)
// stan: dot_product(vector, vector)
// stan: dot_product(vector, row_vector)
// stan: dot_product(row_vector, vector)
// stan: dot_product(row_vector, row_vector)
// stan: dot_product(array[] real, array[] real)
// stan: distance(vector, vector)
// stan: distance(vector, row_vector)
// stan: distance(row_vector, vector)
// stan: distance(row_vector, row_vector)
// stan: squared_distance(vector, vector)
// stan: squared_distance(vector, row_vector)
// stan: squared_distance(row_vector, vector)
// stan: squared_distance(row_vector, row_vector)
// stan: multiply(matrix, matrix)
// stan: mdivide_left(matrix, matrix)
// stan: mdivide_right(matrix, matrix)
// stan: quad_form(matrix, matrix)
// stan: mdivide_left_spd(matrix, matrix)
// stan: mdivide_right_spd(matrix, matrix)
// stan: quad_form_sym(matrix, matrix)
// stan: mdivide_left_tri_low(matrix, matrix)
// stan: mdivide_right_tri_low(matrix, matrix)
// stan: simplex_constrain(vector)
// stan: simplex_jacobian(vector)
// stan: simplex_unconstrain(vector)
// stan: sum_to_zero_constrain(vector)
// stan: sum_to_zero_jacobian(vector)
// stan: sum_to_zero_unconstrain(vector)
// stan: ordered_constrain(vector)
// stan: ordered_jacobian(vector)
// stan: ordered_unconstrain(vector)
// stan: positive_ordered_constrain(vector)
// stan: positive_ordered_jacobian(vector)
// stan: positive_ordered_unconstrain(vector)
// stan: unit_vector_constrain(vector)
// stan: unit_vector_jacobian(vector)
// stan: unit_vector_unconstrain(vector)
// stan: stochastic_column_constrain(matrix)
// stan: stochastic_column_jacobian(matrix)
// stan: stochastic_column_unconstrain(matrix)
// stan: stochastic_row_constrain(matrix)
// stan: stochastic_row_jacobian(matrix)
// stan: stochastic_row_unconstrain(matrix)
// stan: cholesky_factor_corr_constrain(vector, int)
// stan: cholesky_factor_corr_jacobian(vector, int)
// stan: cholesky_factor_corr_unconstrain(matrix)
// stan: cholesky_factor_cov_constrain(vector, int, int)
// stan: cholesky_factor_cov_jacobian(vector, int, int)
// stan: cholesky_factor_cov_unconstrain(matrix)
// stan: corr_matrix_constrain(vector, int)
// stan: corr_matrix_jacobian(vector, int)
// stan: corr_matrix_unconstrain(matrix)
// stan: cov_matrix_constrain(vector, int)
// stan: cov_matrix_jacobian(vector, int)
// stan: cov_matrix_unconstrain(matrix)
fn eval<'t>(name: &str, x: &[Var<'t>]) -> Result<Var<'t>> {
    let a = || mat(x, 3, 3);
    let b = || mat(&x[9..], 3, 2);
    match name {
        "determinant" => determinant(&a()).wsum(),
        "log_determinant" => log_determinant(&a()).wsum(),
        "trace" => trace(&a()).wsum(),
        "crossprod" => crossprod(&a()).wsum(),
        "tcrossprod" => tcrossprod(&a()).wsum(),
        "inverse" => inverse(&a()).wsum(),
        "generalized_inverse" => generalized_inverse(&a()).wsum(),
        "matrix_exp" => matrix_exp(&a()).wsum(),
        "transpose" => transpose(&a()).wsum(),
        "diagonal" => diagonal(&a()).wsum(),
        "rows_dot_self" => rows_dot_self(&a()).wsum(),
        "symmetrize_from_lower_tri" => symmetrize_from_lower_tri(&a()).wsum(),
        "singular_values" => singular_values(&a()).wsum(),
        "svd_U" => svd_U(&a()).wsum(),
        "svd_V" => svd_V(&a()).wsum(),
        "qr_Q" => qr_Q(&a()).wsum(),
        "qr_R" => qr_R(&a()).wsum(),
        "qr_thin_Q" => qr_thin_Q(&a()).wsum(),
        "qr_thin_R" => qr_thin_R(&a()).wsum(),
        "matrix_sum" => sum(x).wsum(),
        "matrix_prod" => prod(x).wsum(),
        "matrix_mean" => mean(x).wsum(),
        "matrix_variance" => variance(x).wsum(),
        "matrix_sd" => sd(x).wsum(),
        "matrix_log_sum_exp" => log_sum_exp_slice(x).wsum(),
        "matrix_max" => max(x).wsum(),
        "matrix_min" => min(x).wsum(),
        "log_determinant_spd" => log_determinant_spd(&sym(x)).wsum(),
        "cholesky_decompose" => cholesky_decompose(&sym(x)).wsum(),
        "eigenvalues_sym" => eigenvalues_sym(&sym(x)).wsum(),
        // Eigenvector signs are arbitrary; the reference reduces the sign invariant V diag(1, 2, 3) Vᵀ.
        "eigenvectors_sym" => {
            let v = eigenvectors_sym(&sym(x))?;
            let d = Matrix::from_rows(&[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
            let vd = multiply(&v, &d.map(|v| Var::constant(*v)))?;
            multiply(&vd, &transpose(&v)).wsum()
        }
        "inverse_spd" => inverse_spd(&sym(x)).wsum(),
        "chol2inv" => chol2inv(&low(x)).wsum(),
        "multiply_lower_tri_self_transpose" => multiply_lower_tri_self_transpose(&low(x)).wsum(),
        "dot_self" => dot_self(x).wsum(),
        "log_sum_exp" => log_sum_exp_slice(x).wsum(),
        "max" => max(x).wsum(),
        "min" => min(x).wsum(),
        "mean" => mean(x).wsum(),
        "norm1" => norm1(x).wsum(),
        "norm2" => norm2(x).wsum(),
        "prod" => prod(x).wsum(),
        "sd" => sd(x).wsum(),
        "sum" => sum(x).wsum(),
        "variance" => variance(x).wsum(),
        "cumulative_sum" => cumulative_sum(x).wsum(),
        "log_softmax" => log_softmax(x).wsum(),
        "softmax" => softmax(x).wsum(),
        "reverse" => reverse(x).wsum(),
        "sort_asc" => sort_asc(x).wsum(),
        "sort_desc" => sort_desc(x).wsum(),
        "dot_product" => dot_product(&x[..4], &x[4..]).wsum(),
        "distance" => distance(&x[..4], &x[4..]).wsum(),
        "squared_distance" => squared_distance(&x[..4], &x[4..]).wsum(),
        "multiply" => multiply(&a(), &b()).wsum(),
        "mdivide_left" => mdivide_left(&a(), &b()).wsum(),
        "mdivide_right" => mdivide_right(&mat(&x[9..], 2, 3), &a()).wsum(),
        "matrix_exp_multiply" => matrix_exp_multiply(&a(), &b()).wsum(),
        "quad_form" => quad_form_matrix(&a(), &b()).wsum(),
        "mdivide_left_spd" => mdivide_left_spd(&sym(x), &mat(&x[6..], 3, 2)).wsum(),
        "mdivide_right_spd" => mdivide_right_spd(&mat(&x[6..], 2, 3), &sym(x)).wsum(),
        "quad_form_sym" => quad_form_sym(&sym(x), &mat(&x[6..], 3, 2)).wsum(),
        "mdivide_left_tri_low" => mdivide_left_tri_low(&low(x), &mat(&x[6..], 3, 2)).wsum(),
        "mdivide_right_tri_low" => mdivide_right_tri_low(&mat(&x[6..], 2, 3), &low(x)).wsum(),
        "simplex_constrain" => simplex_constrain(x).wsum(),
        "simplex_jacobian" => with_lp(x, |lp| simplex_jacobian(x, lp)),
        "simplex_unconstrain" => simplex_unconstrain(x).wsum(),
        "sum_to_zero_constrain" => sum_to_zero_constrain(x).wsum(),
        "sum_to_zero_jacobian" => with_lp(x, |lp| sum_to_zero_jacobian(x, lp)),
        "sum_to_zero_unconstrain" => sum_to_zero_unconstrain(x).wsum(),
        "ordered_constrain" => ordered_constrain(x).wsum(),
        "ordered_jacobian" => with_lp(x, |lp| ordered_jacobian(x, lp)),
        "ordered_unconstrain" => ordered_unconstrain(x).wsum(),
        "positive_ordered_constrain" => positive_ordered_constrain(x).wsum(),
        "positive_ordered_jacobian" => with_lp(x, |lp| positive_ordered_jacobian(x, lp)),
        "positive_ordered_unconstrain" => positive_ordered_unconstrain(x).wsum(),
        "unit_vector_constrain" => unit_vector_constrain(x).wsum(),
        "unit_vector_jacobian" => with_lp(x, |lp| unit_vector_jacobian(x, lp)),
        "unit_vector_unconstrain" => unit_vector_unconstrain(x).wsum(),
        "stochastic_column_constrain" => stochastic_column_constrain(&mat(x, 3, 2)).wsum(),
        "stochastic_column_jacobian" => {
            with_lp(x, |lp| stochastic_column_jacobian(&mat(x, 3, 2), lp))
        }
        "stochastic_column_unconstrain" => stochastic_column_unconstrain(&mat(x, 3, 2)).wsum(),
        "stochastic_row_constrain" => stochastic_row_constrain(&mat(x, 3, 2)).wsum(),
        "stochastic_row_jacobian" => with_lp(x, |lp| stochastic_row_jacobian(&mat(x, 3, 2), lp)),
        "stochastic_row_unconstrain" => stochastic_row_unconstrain(&mat(x, 3, 2)).wsum(),
        "cholesky_factor_corr_constrain" => cholesky_factor_corr_constrain(x, 3).wsum(),
        "cholesky_factor_corr_jacobian" => with_lp(x, |lp| cholesky_factor_corr_jacobian(x, 3, lp)),
        "cholesky_factor_corr_unconstrain" => {
            cholesky_factor_corr_unconstrain(&cholesky_decompose(&sym(x))?).wsum()
        }
        "cholesky_factor_cov_constrain" => cholesky_factor_cov_constrain(x, 3, 3).wsum(),
        "cholesky_factor_cov_jacobian" => {
            with_lp(x, |lp| cholesky_factor_cov_jacobian(x, 3, 3, lp))
        }
        "cholesky_factor_cov_unconstrain" => cholesky_factor_cov_unconstrain(&low(x)).wsum(),
        "corr_matrix_constrain" => corr_matrix_constrain(x, 3).wsum(),
        "corr_matrix_jacobian" => with_lp(x, |lp| corr_matrix_jacobian(x, 3, lp)),
        "corr_matrix_unconstrain" => corr_matrix_unconstrain(&sym(x)).wsum(),
        "cov_matrix_constrain" => cov_matrix_constrain(x, 3).wsum(),
        "cov_matrix_jacobian" => with_lp(x, |lp| cov_matrix_jacobian(x, 3, lp)),
        "cov_matrix_unconstrain" => cov_matrix_unconstrain(&sym(x)).wsum(),
        other => panic!("no dispatch for {other}"),
    }
}

fn list(s: &str) -> Vec<f64> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(';').map(|v| v.parse().unwrap()).collect()
    }
}

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= 1e-10 * b.abs().max(1.0)
}

#[test]
fn matrix_functions_match_stan() {
    let path = format!(
        "{}/tests/data/stan_reference_matrix.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap();
    let mut failures = Vec::new();
    let mut count = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        count += 1;
        let f: Vec<&str> = line.split(',').collect();
        let reals = list(f[2]);
        match try_value_and_grad(|x| eval(f[0], x), &reals) {
            Err(e) => failures.push(format!("{}: we reject: {e}", f[0])),
            Ok(r) => {
                let want: f64 = f[3].parse().unwrap();
                let grad = list(f[4]);
                if !close(r.value, want) {
                    failures.push(format!("{}: value {} vs {want}", f[0], r.value));
                }
                for (k, (a, b)) in r.gradient.iter().zip(&grad).enumerate() {
                    if !close(*a, *b) {
                        failures.push(format!("{}: d/dx{k} {a} vs {b}", f[0]));
                    }
                }
            }
        }
    }
    assert!(count > 90);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
