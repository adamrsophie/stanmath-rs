//! Values and gradients against Stan Math 5.4.0 for the matrix functions and constraining
//! transforms of the Stan language. `tests/data/stan_reference_matrix.csv` is produced by
//! `tools/stan-reference/stan_reference_matrix.cpp`, which documents the inputs. A matrix or vector
//! result is reduced to `sum_k (1 + k / 10) r_k` over its column major entries; a `_jacobian` case
//! adds the log Jacobian to that sum.

use stanmath_rs::autodiff::Complex;
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
// stan: fma(real, real, real)
// stan: qr(matrix)
// stan: qr_thin(matrix)
// stan: svd(matrix)
// stan: eigendecompose_sym(matrix)
// stan: eigendecompose(matrix)
// stan: zeros_int_array(int)
// stan: zeros_vector(int)
// stan: eigenvectors(matrix)
// stan: complex_schur_decompose_t(matrix)
// stan: complex_schur_decompose(matrix)
// stan: complex_schur_decompose_u(matrix)
// stan: ones_int_array(int)
// stan: zeros_array(int)
// stan: ones_vector(int)
// stan: ones_row_vector(int)
// stan: zeros_row_vector(int)
// stan: ones_array(int)
// stan: linspaced_array(int, real, real)
// stan: linspaced_row_vector(int, real, real)
// stan: one_hot_array(int, int)
// stan: one_hot_row_vector(int, int)
// stan: rep_row_vector(real, int)
// stan: to_vector_array(matrix)
// stan: to_row_vector_array(matrix)
// stan: qr_thin(matrix)
// stan: csr_extract(matrix)
// stan: append_array(array[] real, array[] real)
// stan: append_col(matrix, matrix)
// stan: append_row(matrix, matrix)
// stan: block(matrix, int, int, int, int)
// stan: col(matrix, int)
// stan: row(matrix, int)
// stan: sub_col(matrix, int, int, int)
// stan: sub_row(matrix, int, int, int)
// stan: head(vector, int)
// stan: tail(vector, int)
// stan: segment(vector, int, int)
// stan: diag_matrix(vector)
// stan: rep_matrix(real, int, int)
// stan: rep_matrix(vector, int)
// stan: rep_vector(real, int)
// stan: rep_array(real, int)
// stan: to_array_1d(matrix)
// stan: to_array_2d(matrix)
// stan: to_matrix(array[] real, int, int)
// stan: to_matrix(array[] real, int, int, int)
// stan: to_vector(matrix)
// stan: to_row_vector(matrix)
// stan: to_complex(real, real)
// stan: get_real(complex)
// stan: get_imag(complex)
// stan: csr_extract_w(matrix)
// stan: csr_extract_v(matrix)
// stan: csr_extract_u(matrix)
// stan: identity_matrix(int)
// stan: linspaced_vector(int, real, real)
// stan: linspaced_int_array(int, int, int)
// stan: one_hot_vector(int, int)
// stan: one_hot_int_array(int, int)
// stan: uniform_simplex(int)
// stan: dims(matrix)
// stan: num_elements(matrix)
// stan: size(array[] real)
// stan: rows(matrix)
// stan: cols(matrix)
// stan: int_step(real)
// stan: is_inf(real)
// stan: logical_eq(real, real)
// stan: logical_neq(real, real)
// stan: logical_gt(real, real)
// stan: logical_gte(real, real)
// stan: logical_lte(real, real)
// stan: logical_negation(int)
// stan: polar(real, real)
// stan: trace_dot(matrix, matrix)
// stan: eigenvalues(matrix)
// stan: matrix_power(matrix, int)
// stan: scale_matrix_exp_multiply(real, matrix, matrix)
// stan: trace_quad_form(matrix, matrix)
// stan: trace_gen_quad_form(matrix, matrix, matrix)
// stan: quad_form_diag(matrix, vector)
// stan: columns_dot_product(matrix, matrix)
// stan: rows_dot_product(matrix, matrix)
// stan: columns_dot_self(matrix)
// stan: diag_pre_multiply(vector, matrix)
// stan: diag_post_multiply(matrix, vector)
// stan: add_diag(matrix, real)
// stan: add_diag(matrix, vector)
// stan: csr_matrix_times_vector(int, int, vector, array[] int, array[] int, vector)
// stan: csr_to_dense_matrix(int, int, vector, array[] int, array[] int)
// stan: fft(complex_vector)
// stan: inv_fft(complex_vector)
// stan: fft2(complex_matrix)
// stan: inv_fft2(complex_matrix)
// stan: gp_exp_quad_cov(array[] real, real, real)
// stan: gp_matern32_cov(array[] real, real, real)
// stan: gp_matern52_cov(array[] real, real, real)
// stan: gp_exponential_cov(array[] real, real, real)
// stan: gp_periodic_cov(array[] real, real, real, real)
// stan: gp_dot_prod_cov(array[] real, real)
// stan: hypergeometric_3F2(vector, vector, real)
// stan: hypergeometric_pFq(vector, vector, real)
// stan: quantile(data vector, data real)
// stan: rank(vector, int)
// stan: sort_indices_asc(vector)
// stan: sort_indices_desc(vector)
/// Four complex values from `x[0..8]`, (re, im) pairs.
fn cvec<'t>(x: &[Var<'t>]) -> Vec<Complex<Var<'t>>> {
    x[..8].chunks(2).map(|p| Complex::new(p[0], p[1])).collect()
}

/// `sum_k (1 + k / 10) re_k + (1 / 2 + k / 10) im_k` over column major entries.
fn cwsum<'t>(e: &[Complex<Var<'t>>]) -> Var<'t> {
    e.iter()
        .enumerate()
        .map(|(k, z)| z.re * (1.0 + 0.1 * k as f64) + z.im * (0.5 + 0.1 * k as f64))
        .sum()
}

/// The sparse matrix of the CSR cases: nonzeros `x[0..5]` at (1,1), (1,3), (2,2), (3,1), (3,3).
fn csr<'t>(x: &[Var<'t>]) -> CsrMatrix<Var<'t>> {
    let z = Var::constant(0.0);
    csr_extract(&Matrix::from_rows(&[
        [x[0], z, x[1]],
        [z, x[2], z],
        [x[3], z, x[4]],
    ]))
}

/// The 3 by 3 sparse matrix with nonzeros `x[0..5]` at (1,1), (1,3), (2,2), (3,1), (3,3).
fn sparse<'t>(x: &[Var<'t>]) -> Matrix<Var<'t>> {
    let z = Var::constant(0.0);
    Matrix::from_rows(&[[x[0], z, x[1]], [z, x[2], z], [x[3], z, x[4]]])
}

fn schur_diag<'t>(t: &Matrix<Complex<Var<'t>>>) -> Var<'t> {
    (0..t.nrows())
        .map(|k| t[(k, k)].abs() + t[(k, k)].re * t[(k, k)].re)
        .sum()
}

fn one_based(v: Vec<usize>) -> Vec<usize> {
    v.into_iter().map(|i| i + 1).collect()
}

/// An integer result reduced as `sum_k (1 + k / 10) r_k`, as a constant.
fn iws<'t, I: Copy + TryInto<i32>>(r: &[I]) -> Var<'t> {
    let v = |i: I| f64::from(i.try_into().unwrap_or(i32::MAX));
    Var::constant(
        r.iter()
            .enumerate()
            .map(|(k, &i)| v(i) * (1.0 + 0.1 * k as f64))
            .sum(),
    )
}

/// Four one based indices packed as decimal digits, first index lowest.
fn digits<'t>(r: Vec<usize>) -> Result<Var<'t>> {
    Ok(Var::constant(
        (r[0] + 10 * r[1] + 100 * r[2] + 1000 * r[3]) as f64,
    ))
}

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
        "fma" => Ok(fma(x[0], x[1], x[2])),
        "matrix_power" => matrix_power(&a(), 3).wsum(),
        "scale_matrix_exp_multiply" => scale_matrix_exp_multiply(x[15], &a(), &b()).wsum(),
        "trace_quad_form" => trace_quad_form(&sym(x), &mat(&x[6..], 3, 2)).wsum(),
        "trace_gen_quad_form" => {
            trace_gen_quad_form(&mat(x, 2, 2), &mat(&x[4..], 3, 3), &mat(&x[13..], 3, 2)).wsum()
        }
        "quad_form_diag" => quad_form_diag(&a(), &x[9..12]).wsum(),
        "columns_dot_product" => columns_dot_product(&mat(x, 3, 2), &mat(&x[6..], 3, 2)).wsum(),
        "rows_dot_product" => rows_dot_product(&mat(x, 3, 2), &mat(&x[6..], 3, 2)).wsum(),
        "columns_dot_self" => columns_dot_self(&mat(x, 3, 2)).wsum(),
        "diag_pre_multiply" => diag_pre_multiply(&x[0..3], &mat(&x[3..], 3, 3)).wsum(),
        "diag_post_multiply" => diag_post_multiply(&a(), &x[9..12]).wsum(),
        "add_diag" => add_diag(&a(), x[9]).wsum(),
        "add_diag_vector" => add_diag_vector(&a(), &x[9..12]).wsum(),
        "csr_matrix_times_vector" => csr_matrix_times_vector(&csr(x), &x[5..8]).wsum(),
        "csr_to_dense_matrix" => csr_to_dense_matrix(&csr(x)).wsum(),
        "fft" => Ok(cwsum(&fft(&cvec(x)))),
        "inv_fft" => Ok(cwsum(&inv_fft(&cvec(x)))),
        "fft2" => Ok(cwsum(
            fft2(&Matrix::from_column_major(2, 2, cvec(x))).as_slice(),
        )),
        "inv_fft2" => Ok(cwsum(
            inv_fft2(&Matrix::from_column_major(2, 2, cvec(x))).as_slice(),
        )),
        "gp_exp_quad_cov" => gp_exp_quad_cov(&gp_points_1d(&x[0..3]), x[3], x[4]).wsum(),
        "gp_matern32_cov" => gp_matern32_cov(&gp_points_1d(&x[0..3]), x[3], x[4]).wsum(),
        "gp_matern52_cov" => gp_matern52_cov(&gp_points_1d(&x[0..3]), x[3], x[4]).wsum(),
        "gp_exponential_cov" => gp_exponential_cov(&gp_points_1d(&x[0..3]), x[3], x[4]).wsum(),
        "gp_periodic_cov" => gp_periodic_cov(&gp_points_1d(&x[0..3]), x[3], x[4], x[5]).wsum(),
        "gp_dot_prod_cov" => gp_dot_prod_cov(&gp_points_1d(&x[0..3]), x[3]).wsum(),
        "hypergeometric_3F2" => Ok(hypergeometric_3F2([x[0], x[1], x[2]], [x[3], x[4]], x[5])),
        "hypergeometric_pFq" => Ok(hypergeometric_pFq(&x[0..2], &x[2..3], x[3])),
        "quantile" => {
            let d = [0.3, -0.5, 1.2, 0.7, 2.0, -1.1];
            Ok(Var::constant(
                quantile(&d, 0.3)? + 2.0 * quantile(&d, 0.85)?,
            ))
        }
        "rank" => {
            let d = [0.3, -0.5, 1.2, 0.7];
            Ok(Var::constant((rank(&d, 1)? + 10 * rank(&d, 3)?) as f64))
        }
        "sort_indices_asc" => digits(sort_indices_asc(&[0.3, -0.5, 1.2, 0.7])?),
        "sort_indices_desc" => digits(sort_indices_desc(&[0.3, -0.5, 1.2, 0.7])?),
        "polar" => Ok(cwsum(&[polar(x[0], x[1])])),
        "trace_dot" => trace_dot(&mat(x, 3, 2), &mat(&x[6..], 2, 3)).wsum(),
        "eigenvalues" => Ok(eigenvalues(&a())?
            .iter()
            .map(|l| l.abs() + l.re * l.re)
            .sum()),
        "append_array" => append_array(&x[0..2], &x[2..4]).wsum(),
        "append_col" => append_col(&a(), &b()).wsum(),
        "append_row" => append_row(&a(), &mat(&x[9..], 2, 3)).wsum(),
        "block" => block(&a(), 2, 1, 2, 2).wsum(),
        "col" => col(&a(), 2).wsum(),
        "row" => row(&a(), 3).wsum(),
        "sub_col" => sub_col(&a(), 1, 2, 2).wsum(),
        "sub_row" => sub_row(&a(), 2, 1, 3).wsum(),
        "head" => head(x, 2).wsum(),
        "tail" => tail(x, 3).wsum(),
        "segment" => segment(x, 2, 2).wsum(),
        "diag_matrix" => diag_matrix(x).wsum(),
        "rep_matrix" => rep_matrix(x[0], 2, 3).wsum(),
        "rep_matrix_vector" => rep_matrix_vector(x, 2).wsum(),
        "rep_vector" => rep_vector(x[0], 3).wsum(),
        "rep_array" => rep_array(x[0], 3).wsum(),
        "to_array_1d" => to_array_1d(&a()).wsum(),
        "to_array_2d" => to_array_2d(&a()).concat().wsum(),
        "to_matrix" => to_matrix(x, 2, 3).wsum(),
        "to_matrix_row_major" => to_matrix_row_major(x, 2, 3).wsum(),
        "to_vector" => to_vector(a().as_slice()).wsum(),
        "to_row_vector" => to_row_vector(a().as_slice()).wsum(),
        "to_complex" => Ok(cwsum(&[to_complex(x[0], x[1])])),
        "get_real" => Ok(get_real(Complex::new(x[0], x[1]))),
        "get_imag" => Ok(get_imag(Complex::new(x[0], x[1]))),
        "csr_extract_w" => csr_extract_w(&sparse(x)).wsum(),
        // stanmath-rs's CSR indices are zero based; Stan's are one based.
        "csr_extract_v" => Ok(iws(&one_based(csr_extract_v(&sparse(x))))),
        "csr_extract_u" => Ok(iws(&one_based(csr_extract_u(&sparse(x))))),
        // Tuple forms: each part reduced and summed, with weights so a swapped part is caught.
        "qr" => {
            let (q, r) = qr(&a());
            Ok(q.wsum()? + 2.0 * r.wsum()?)
        }
        "qr_thin" => {
            let (q, r) = qr_thin(&a());
            Ok(q.wsum()? + 2.0 * r.wsum()?)
        }
        "svd" => {
            let (u, d, v) = svd(&a())?;
            Ok(u.wsum()? + 2.0 * d.wsum()? + 3.0 * v.wsum()?)
        }
        "eigendecompose_sym" => {
            let (v, l) = eigendecompose_sym(&sym(x))?;
            let d = Matrix::from_rows(&[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
            let vd = multiply(&v, &d.map(|v| Var::constant(*v)))?;
            Ok(multiply(&vd, &transpose(&v)).wsum()? + 2.0 * l.wsum()?)
        }
        // Eigenvectors of a general matrix are not unique; the values part is reduced as eigenvalues.
        "eigendecompose" => Ok(eigendecompose(&a())?
            .1
            .iter()
            .map(|l| l.abs() + l.re * l.re)
            .sum()),
        // Order and phase free: sum_k |l_k| sum_i (1 + i / 10) |v_ik|^2 over unit length columns.
        "eigenvectors" => {
            let (l, v) = (eigenvalues(&a())?, eigenvectors(&a())?);
            Ok((0..l.len())
                .map(|k| {
                    let c: Var = (0..v.nrows())
                        .map(|i| (1.0 + 0.1 * i as f64) * v[(i, k)].norm())
                        .sum();
                    l[k].abs() * c
                })
                .sum())
        }
        // The diagonal of T holds the eigenvalues, reduced as eigenvalues.
        "complex_schur_decompose_t" => Ok(schur_diag(&complex_schur_decompose_t(&a())?)),
        // With the eigenvalues in the same order on T's diagonal, U is unique up to a phase per
        // column, so the reduction uses |u_ij|^2.
        "complex_schur_decompose_u" => {
            let u = complex_schur_decompose_u(&a())?;
            Ok((0..3)
                .flat_map(|j| (0..3).map(move |i| (i, j)))
                .map(|(i, j)| (1.0 + 0.1 * (3 * j + i) as f64) * u[(i, j)].norm())
                .sum())
        }
        "complex_schur_decompose" => Ok(schur_diag(&complex_schur_decompose(&a())?.1)),
        "zeros_ones" => Ok(iws(&zeros_int_array(3))
            + 2.0 * iws(&ones_int_array(3))
            + zeros_array(2).wsum()?
            + ones_vector(3).wsum()?
            + 3.0 * ones_row_vector(4).wsum()?
            + zeros_row_vector(2).wsum()?
            + 5.0 * ones_array(2).wsum()?
            + 7.0 * zeros_vector(2).wsum()?),
        "linspaced_array" => linspaced_array(5, -1.0, 2.0).wsum(),
        "linspaced_row_vector" => linspaced_row_vector(5, -1.0, 2.0).wsum(),
        "one_hot_array" => one_hot_array(4, 2).wsum(),
        "one_hot_row_vector" => one_hot_row_vector(4, 2).wsum(),
        "rep_row_vector" => rep_row_vector(x[0], 3).wsum(),
        "to_vector_array" => to_vector_array(&a()).concat().wsum(),
        "to_row_vector_array" => to_row_vector_array(&a()).concat().wsum(),
        "csr_extract" => {
            let c = csr_extract(&sparse(x));
            Ok(c.values.wsum()? + iws(&one_based(c.columns)) + 2.0 * iws(&one_based(c.row_starts)))
        }
        "identity_matrix" => identity_matrix(3).wsum(),
        "linspaced_vector" => linspaced_vector(5, -1.0, 2.0).wsum(),
        "linspaced_int_array" => Ok(iws(&linspaced_int_array(5, 1, 9)?)),
        "one_hot_vector" => one_hot_vector(4, 3).wsum(),
        "one_hot_int_array" => Ok(iws(&one_hot_int_array(4, 3)?)),
        "uniform_simplex" => uniform_simplex(4).wsum(),
        "dims" => Ok(iws(&dims(&mat(x, 3, 3))) + iws(&dims(&mat(x, 2, 3)))),
        "num_elements" => Ok(Var::constant(num_elements(&mat(x, 3, 2)) as f64)),
        "size" => Ok(Var::constant(size(&x[0..4]) as f64)),
        "rows" => Ok(Var::constant(rows(&mat(x, 3, 2)) as f64)),
        "cols" => Ok(Var::constant(cols(&mat(x, 3, 2)) as f64)),
        "int_step" => Ok(Var::constant(f64::from(
            int_step(-0.5) + 10 * int_step(0.0) + 100 * int_step(2.0),
        ))),
        "is_inf" => {
            let b = |v: f64| f64::from(u8::from(is_inf(v)));
            Ok(Var::constant(
                b(1.0) + 10.0 * b(f64::INFINITY) + 100.0 * b(-f64::INFINITY) + 1000.0 * b(f64::NAN),
            ))
        }
        "logical" => {
            let b = |v: bool| f64::from(u8::from(v));
            Ok(Var::constant(
                b(logical_eq(1.0, 1.0))
                    + 2.0 * b(logical_neq(1.0, 2.0))
                    + 4.0 * b(logical_gt(1.0, 2.0))
                    + 8.0 * b(logical_gte(2.0, 2.0))
                    + 16.0 * b(logical_lte(3.0, 2.0))
                    + 32.0 * b(logical_negation(0_i64))
                    + 64.0 * b(logical_eq(f64::NAN, f64::NAN))
                    + 128.0 * b(logical_neq(f64::NAN, f64::NAN)),
            ))
        }
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
    assert!(count > 185);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
