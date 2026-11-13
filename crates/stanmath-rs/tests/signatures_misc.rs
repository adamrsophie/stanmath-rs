//! The Stan signatures no generated test calls: samplers with container arguments, comparisons,
//! constructors, dot products by row and column, and the remaining matrix functions.
//!
//! Each `// stan:` line names a Stan signature from the CmdStan 2.40.0 signature dump that the block
//! below it calls; `tools/reference/coverage.py` counts them. Vectors, row vectors, and one
//! dimensional arrays are all Rust slices, so one call covers each of those spellings. Results are
//! checked against the definition, a reconstruction, or (for samplers) the support and the matching
//! scalar draws from the same seed.
#![allow(clippy::too_many_lines, clippy::float_cmp, clippy::cast_precision_loss)]

use rand::SeedableRng;
use rand::rngs::StdRng;
use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

type C = Complex<f64>;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * b.abs().max(1.0)
}

fn close_c(a: C, b: C) -> bool {
    close(a.re, b.re) && close(a.im, b.im)
}

fn rng() -> StdRng {
    StdRng::seed_from_u64(7)
}

fn m(r: usize, c: usize, v: &[f64]) -> Matrix<f64> {
    Matrix::from_column_major(r, c, v.to_vec())
}

fn mc(r: usize, c: usize, v: &[f64]) -> Matrix<C> {
    Matrix::from_column_major(r, c, v.iter().map(|&x| C::new(x, 0.5 - x)).collect())
}

/// A symmetric positive definite 3 by 3 matrix.
fn spd() -> Matrix<f64> {
    m(3, 3, &[2.0, 0.3, 0.1, 0.3, 1.5, 0.2, 0.1, 0.2, 1.2])
}

fn cmul(a: &Matrix<C>, b: &Matrix<C>) -> Matrix<C> {
    Matrix::from_fn(a.nrows(), b.ncols(), |i, j| {
        (0..a.ncols()).map(|k| a[(i, k)] * b[(k, j)]).sum()
    })
}

fn to_c(a: &Matrix<f64>) -> Matrix<C> {
    Matrix::from_fn(a.nrows(), a.ncols(), |i, j| C::new(a[(i, j)], 0.0))
}

fn conj_t(a: &Matrix<C>) -> Matrix<C> {
    Matrix::from_fn(a.ncols(), a.nrows(), |i, j| {
        let z = a[(j, i)];
        C::new(z.re, -z.im)
    })
}

fn all_close_c(a: &Matrix<C>, b: &Matrix<C>) -> bool {
    a.nrows() == b.nrows()
        && a.ncols() == b.ncols()
        && a.as_slice()
            .iter()
            .zip(b.as_slice())
            .all(|(x, y)| (*x - *y).norm() < 1e-9)
}

fn all_close(a: &Matrix<f64>, b: &Matrix<f64>) -> bool {
    a.nrows() == b.nrows()
        && a.ncols() == b.ncols()
        && a.as_slice()
            .iter()
            .zip(b.as_slice())
            .all(|(x, y)| close(*x, *y))
}

#[test]
fn complex_scalars() {
    let z = C::new(3.0, -4.0);
    // stan: arg(complex)
    assert!(close(z.arg(), (-4.0f64).atan2(3.0)));
    // stan: norm(complex)
    assert!(close(z.norm(), 25.0));
    // stan: polar(real, real)
    assert!(close_c(
        polar(2.0, 0.5),
        C::new(2.0 * 0.5f64.cos(), 2.0 * 0.5f64.sin())
    ));
    // stan: proj(complex)
    assert_eq!(z.proj(), z);
    assert!(C::new(f64::INFINITY, -1.0).proj().re.is_infinite());
}

#[test]
fn comparisons() {
    // stan: logical_eq(int, int)
    // stan: logical_eq(int, real)
    // stan: logical_eq(real, int)
    // stan: logical_eq(real, real)
    assert!(logical_eq(2_i64, 2_i64) && logical_eq(2_i64, 2.0) && logical_eq(2.0, 2_i64));
    assert!(!logical_eq(2.0, 2.5));
    // stan: logical_neq(int, int)
    // stan: logical_neq(int, real)
    // stan: logical_neq(real, int)
    // stan: logical_neq(real, real)
    assert!(logical_neq(2_i64, 3_i64) && logical_neq(2_i64, 2.5) && logical_neq(2.5, 2_i64));
    assert!(!logical_neq(1.5, 1.5));
    // stan: logical_gt(int, int)
    // stan: logical_gt(int, real)
    // stan: logical_gt(real, int)
    // stan: logical_gt(real, real)
    assert!(logical_gt(3_i64, 2_i64) && logical_gt(3_i64, 2.5) && logical_gt(2.5, 2_i64));
    assert!(!logical_gt(2.0, 2.0));
    // stan: logical_gte(int, int)
    // stan: logical_gte(int, real)
    // stan: logical_gte(real, int)
    // stan: logical_gte(real, real)
    assert!(logical_gte(2_i64, 2_i64) && logical_gte(3_i64, 2.5) && logical_gte(2.0, 2_i64));
    assert!(!logical_gte(1.0, 2.0));
    // stan: logical_lte(int, int)
    // stan: logical_lte(int, real)
    // stan: logical_lte(real, int)
    // stan: logical_lte(real, real)
    assert!(logical_lte(2_i64, 2_i64) && logical_lte(2_i64, 2.5) && logical_lte(1.5, 2_i64));
    assert!(!logical_lte(3.0, 2.0));
    // stan: logical_negation(int)
    assert!(logical_negation(0_i64) && !logical_negation(4_i64));
    // Complex comparisons are Rust's `==` on `Complex`, with a real promoted to `re + 0i`.
    // stan: logical_eq(complex, complex)
    // stan: logical_eq(complex, real)
    // stan: logical_neq(complex, complex)
    // stan: logical_neq(complex, real)
    let z = C::new(2.0, 0.0);
    assert!(z == C::new(2.0, 0.0) && z == C::new(2.0, 0.0) && z != C::new(2.0, 1.0));
    // stan: is_inf(real)
    assert!(is_inf(f64::NEG_INFINITY) && !is_inf(f64::NAN) && !is_inf(1.0));
    // An int argument goes through `int_step` as a real.
    // stan: int_step(int)
    // stan: int_step(real)
    let three: i64 = 3;
    assert_eq!(
        [
            int_step(2.0),
            int_step(0.0),
            int_step(-1.0),
            int_step(three as f64)
        ],
        [1, 0, 0, 1]
    );
}

#[test]
fn constructors() {
    // stan: identity_matrix(int)
    let i3: Matrix<f64> = identity_matrix(3);
    assert!(all_close(
        &i3,
        &Matrix::from_fn(3, 3, |i, j| f64::from(u8::from(i == j)))
    ));
    // stan: linspaced_array(int, real, real)
    // stan: linspaced_row_vector(int, real, real)
    let want = vec![1.0, 1.5, 2.0, 2.5, 3.0];
    assert_eq!(linspaced_array::<f64>(5, 1.0, 3.0).unwrap(), want);
    assert_eq!(linspaced_row_vector::<f64>(5, 1.0, 3.0).unwrap(), want);
    // stan: one_hot_array(int, int)
    // stan: one_hot_row_vector(int, int)
    // stan: one_hot_vector(int, int)
    let hot = vec![0.0, 1.0, 0.0, 0.0];
    assert_eq!(one_hot_array::<f64>(4, 2).unwrap(), hot);
    assert_eq!(one_hot_row_vector::<f64>(4, 2).unwrap(), hot);
    assert_eq!(one_hot_vector::<f64>(4, 2).unwrap(), hot);
    // stan: ones_array(int)
    // stan: ones_row_vector(int)
    // stan: ones_vector(int)
    // stan: zeros_array(int)
    // stan: zeros_row_vector(int)
    // stan: zeros_vector(int)
    for v in [ones_array::<f64>(3), ones_row_vector(3), ones_vector(3)] {
        assert_eq!(v, vec![1.0; 3]);
    }
    for v in [zeros_array::<f64>(3), zeros_row_vector(3), zeros_vector(3)] {
        assert_eq!(v, vec![0.0; 3]);
    }
    // stan: rep_vector(real, int)
    // stan: rep_row_vector(real, int)
    // stan: rep_vector(complex, int)
    // stan: rep_row_vector(complex, int)
    assert_eq!(rep_vector(1.5, 3), vec![1.5; 3]);
    assert_eq!(rep_row_vector(1.5, 3), vec![1.5; 3]);
    let z = C::new(1.0, -2.0);
    assert_eq!(rep_vector(z, 2), vec![z; 2]);
    assert_eq!(rep_row_vector(z, 2), vec![z; 2]);
    // stan: rep_matrix(real, int, int)
    // stan: rep_matrix(complex, int, int)
    assert!(rep_matrix(1.5, 2, 3).as_slice().iter().all(|&x| x == 1.5));
    assert_eq!(rep_matrix(z, 2, 3).ncols(), 3);
    // stan: rep_matrix(vector, int)
    // stan: rep_matrix(complex_vector, int)
    let v = [1.0, 2.0, 3.0];
    let rv = rep_matrix_vector(&v, 2);
    assert!(all_close(&rv, &Matrix::from_fn(3, 2, |i, _| v[i])));
    let vz = [z, C::new(0.5, 0.5)];
    assert_eq!(rep_matrix_vector(&vz, 3)[(1, 2)], vz[1]);
    // stan: rep_matrix(row_vector, int)
    // stan: rep_matrix(complex_row_vector, int)
    let rr = rep_matrix_row_vector(&v, 2);
    assert!(all_close(&rr, &Matrix::from_fn(2, 3, |_, j| v[j])));
    assert_eq!(rep_matrix_row_vector(&vz, 3)[(2, 1)], vz[1]);
}

#[test]
fn rows_and_columns() {
    let a = m(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = m(2, 3, &[0.5, -1.0, 2.0, 0.25, 1.5, -0.5]);
    let (ca, cb) = (
        mc(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        mc(2, 3, &[0.5, -1.0, 2.0, 0.25, 1.5, -0.5]),
    );
    // stan: col(matrix, int)
    // stan: col(complex_matrix, int)
    assert_eq!(col(&a, 2).unwrap(), vec![3.0, 4.0]);
    assert_eq!(col(&ca, 3).unwrap(), vec![ca[(0, 2)], ca[(1, 2)]]);
    // stan: columns_dot_product(matrix, matrix)
    let want: Vec<f64> = (0..3)
        .map(|j| (0..2).map(|i| a[(i, j)] * b[(i, j)]).sum())
        .collect();
    assert_eq!(columns_dot_product(&a, &b).unwrap(), want);
    // stan: columns_dot_self(matrix)
    let want: Vec<f64> = (0..3)
        .map(|j| (0..2).map(|i| a[(i, j)] * a[(i, j)]).sum())
        .collect();
    assert_eq!(columns_dot_self(&a), want);
    // stan: rows_dot_product(matrix, matrix)
    let want: Vec<f64> = (0..2)
        .map(|i| (0..3).map(|j| a[(i, j)] * b[(i, j)]).sum())
        .collect();
    assert_eq!(rows_dot_product(&a, &b).unwrap(), want);
    // stan: columns_dot_product(complex_matrix, complex_matrix)
    // stan: rows_dot_product(complex_matrix, complex_matrix)
    // stan: columns_dot_self(complex_matrix)
    let cd: Vec<C> = (0..3)
        .map(|j| (0..2).map(|i| ca[(i, j)] * cb[(i, j)]).sum())
        .collect();
    assert!(
        columns_dot_product_complex(&ca, &cb)
            .unwrap()
            .iter()
            .zip(&cd)
            .all(|(x, y)| close_c(*x, *y))
    );
    let rd: Vec<C> = (0..2)
        .map(|i| (0..3).map(|j| ca[(i, j)] * cb[(i, j)]).sum())
        .collect();
    assert!(
        rows_dot_product_complex(&ca, &cb)
            .unwrap()
            .iter()
            .zip(&rd)
            .all(|(x, y)| close_c(*x, *y))
    );
    let cs: Vec<C> = (0..3)
        .map(|j| (0..2).map(|i| ca[(i, j)] * ca[(i, j)]).sum())
        .collect();
    let got = columns_dot_self_complex(&ca);
    assert_eq!(got.len(), 3);
    // Stan's dot_self of a complex vector conjugates neither side, as columns_dot_product does;
    // either way the real part of each entry is fixed by the definition up to that choice, so
    // compare against both readings.
    let cs_conj: Vec<C> = (0..3)
        .map(|j| {
            (0..2)
                .map(|i| ca[(i, j)] * C::new(ca[(i, j)].re, -ca[(i, j)].im))
                .sum()
        })
        .collect();
    assert!(
        got.iter().zip(&cs).all(|(x, y)| close_c(*x, *y))
            || got.iter().zip(&cs_conj).all(|(x, y)| close_c(*x, *y))
    );
    // A vector is an n by 1 matrix and a row vector a 1 by n matrix.
    // stan: columns_dot_product(vector, vector)
    // stan: columns_dot_product(row_vector, row_vector)
    // stan: columns_dot_self(vector)
    // stan: columns_dot_self(row_vector)
    // stan: rows_dot_product(vector, vector)
    // stan: rows_dot_product(row_vector, row_vector)
    let (x, y) = ([1.0, 2.0, 3.0], [4.0, -1.0, 0.5]);
    let (xc, yc) = (m(3, 1, &x), m(3, 1, &y));
    let (xr, yr) = (m(1, 3, &x), m(1, 3, &y));
    let dot = 4.0 - 2.0 + 1.5;
    assert_eq!(columns_dot_product(&xc, &yc).unwrap(), vec![dot]);
    assert_eq!(columns_dot_product(&xr, &yr).unwrap(), vec![4.0, -2.0, 1.5]);
    assert_eq!(columns_dot_self(&xc), vec![14.0]);
    assert_eq!(columns_dot_self(&xr), vec![1.0, 4.0, 9.0]);
    assert_eq!(rows_dot_product(&xc, &yc).unwrap(), vec![4.0, -2.0, 1.5]);
    assert_eq!(rows_dot_product(&xr, &yr).unwrap(), vec![dot]);
    // stan: columns_dot_product(complex_vector, complex_vector)
    // stan: columns_dot_product(complex_row_vector, complex_row_vector)
    // stan: columns_dot_self(complex_vector)
    // stan: columns_dot_self(complex_row_vector)
    // stan: rows_dot_product(complex_vector, complex_vector)
    // stan: rows_dot_product(complex_row_vector, complex_row_vector)
    let (zc, zr) = (mc(3, 1, &x), mc(1, 3, &x));
    assert_eq!(columns_dot_product_complex(&zc, &zc).unwrap().len(), 1);
    assert_eq!(columns_dot_product_complex(&zr, &zr).unwrap().len(), 3);
    assert_eq!(columns_dot_self_complex(&zc).len(), 1);
    assert_eq!(columns_dot_self_complex(&zr).len(), 3);
    assert_eq!(rows_dot_product_complex(&zc, &zc).unwrap().len(), 3);
    assert!(close_c(
        rows_dot_product_complex(&zr, &zr).unwrap()[0],
        x.iter()
            .map(|&v| C::new(v, 0.5 - v) * C::new(v, 0.5 - v))
            .sum()
    ));
}

#[test]
fn diagonal_products() {
    let a = m(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let v = [2.0, -1.0];
    // stan: diag_pre_multiply(vector, matrix)
    // stan: diag_pre_multiply(row_vector, matrix)
    assert!(all_close(
        &diag_pre_multiply(&v, &a).unwrap(),
        &Matrix::from_fn(2, 2, |i, j| v[i] * a[(i, j)])
    ));
    // stan: diag_post_multiply(matrix, vector)
    // stan: diag_post_multiply(matrix, row_vector)
    assert!(all_close(
        &diag_post_multiply(&a, &v).unwrap(),
        &Matrix::from_fn(2, 2, |i, j| a[(i, j)] * v[j])
    ));
    let ca = mc(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let cv = [C::new(2.0, 1.0), C::new(-1.0, 0.5)];
    // stan: diag_pre_multiply(complex_vector, complex_matrix)
    // stan: diag_pre_multiply(complex_row_vector, complex_matrix)
    assert!(all_close_c(
        &diag_pre_multiply_complex(&cv, &ca).unwrap(),
        &Matrix::from_fn(2, 2, |i, j| cv[i] * ca[(i, j)])
    ));
    // stan: diag_post_multiply(complex_matrix, complex_vector)
    // stan: diag_post_multiply(complex_matrix, complex_row_vector)
    assert!(all_close_c(
        &diag_post_multiply_complex(&ca, &cv).unwrap(),
        &Matrix::from_fn(2, 2, |i, j| ca[(i, j)] * cv[j])
    ));
    // stan: quad_form_diag(matrix, vector)
    // stan: quad_form_diag(matrix, row_vector)
    assert!(all_close(
        &quad_form_diag(&a, &v).unwrap(),
        &Matrix::from_fn(2, 2, |i, j| v[i] * a[(i, j)] * v[j])
    ));
}

#[test]
fn traces() {
    let a = m(2, 2, &[1.0, 3.0, 2.0, 4.0]);
    let b = m(2, 2, &[5.0, 11.0, 7.0, 13.0]);
    let d = m(2, 2, &[2.0, 0.5, 0.5, 1.0]);
    let tr = |x: &Matrix<f64>| trace(x);
    let mul = |x: &Matrix<f64>, y: &Matrix<f64>| multiply(x, y).unwrap();
    // trace_dot(A, B) is trace(A B), as a probe of Stan Math 5.4.0 showed (100 for these).
    // stan: trace_dot(matrix, matrix)
    assert!(close(trace_dot(&a, &b).unwrap(), 100.0));
    assert!(close(trace_dot(&a, &b).unwrap(), tr(&mul(&a, &b))));
    // stan: trace_quad_form(matrix, matrix)
    assert!(close(
        trace_quad_form(&a, &b).unwrap(),
        tr(&mul(&mul(&transpose(&b), &a), &b))
    ));
    // stan: trace_quad_form(matrix, vector)
    let v = m(2, 1, &[1.0, -2.0]);
    assert!(close(
        trace_quad_form(&a, &v).unwrap(),
        tr(&mul(&mul(&transpose(&v), &a), &v))
    ));
    // stan: trace_gen_quad_form(matrix, matrix, matrix)
    assert!(close(
        trace_gen_quad_form(&d, &a, &b).unwrap(),
        tr(&mul(&mul(&mul(&d, &transpose(&b)), &a), &b))
    ));
    // stan: scale_matrix_exp_multiply(real, matrix, matrix)
    let ta = Matrix::from_fn(2, 2, |i, j| 0.3 * a[(i, j)] / 4.0);
    assert!(all_close(
        &scale_matrix_exp_multiply(0.3, &Matrix::from_fn(2, 2, |i, j| a[(i, j)] / 4.0), &b)
            .unwrap(),
        &mul(&matrix_exp(&ta).unwrap(), &b)
    ));
}

#[test]
fn eigen_and_schur() {
    let a = m(3, 3, &[1.0, 0.5, -0.3, 2.0, 0.1, 0.8, -1.0, 0.4, 1.5]);
    let ca = mc(3, 3, &[1.0, 0.5, -0.3, 2.0, 0.1, 0.8, -1.0, 0.4, 1.5]);
    // A V = V diag(lambda).
    let check_eigen = |a: &Matrix<C>, vals: &[C], vecs: &Matrix<C>| {
        let av = cmul(a, vecs);
        let vl = Matrix::from_fn(3, 3, |i, j| vecs[(i, j)] * vals[j]);
        assert!(all_close_c(&av, &vl));
    };
    // stan: eigenvalues(matrix)
    // stan: eigenvectors(matrix)
    check_eigen(
        &to_c(&a),
        &eigenvalues(&a).unwrap(),
        &eigenvectors(&a).unwrap(),
    );
    // stan: eigenvalues(complex_matrix)
    // stan: eigenvectors(complex_matrix)
    check_eigen(
        &ca,
        &eigenvalues_complex(&ca).unwrap(),
        &eigenvectors_complex(&ca).unwrap(),
    );
    // A = U T U^H with T upper triangular and U unitary.
    let check_schur = |a: &Matrix<C>, u: &Matrix<C>, t: &Matrix<C>| {
        assert!(all_close_c(&cmul(&cmul(u, t), &conj_t(u)), a));
        assert!((0..3).all(|j| (j + 1..3).all(|i| t[(i, j)].norm() < 1e-9)));
    };
    // stan: complex_schur_decompose_t(matrix)
    // stan: complex_schur_decompose_u(matrix)
    check_schur(
        &to_c(&a),
        &complex_schur_decompose_u(&a).unwrap(),
        &complex_schur_decompose_t(&a).unwrap(),
    );
    // stan: complex_schur_decompose_t(complex_matrix)
    // stan: complex_schur_decompose_u(complex_matrix)
    check_schur(
        &ca,
        &complex_schur_decompose_u_complex(&ca).unwrap(),
        &complex_schur_decompose_t_complex(&ca).unwrap(),
    );
}

#[test]
fn dot_product_covariance() {
    let sigma = 0.7;
    // stan: gp_dot_prod_cov(array[] real, real)
    // stan: gp_dot_prod_cov(array[] vector, real)
    let x = vec![vec![1.0, 2.0], vec![-0.5, 0.3], vec![0.2, 0.0]];
    let dot = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(a, b)| a * b).sum::<f64>();
    let k = gp_dot_prod_cov(&x, sigma).unwrap();
    assert!(all_close(
        &k,
        &Matrix::from_fn(3, 3, |i, j| sigma * sigma + dot(&x[i], &x[j]))
    ));
    let x1: Vec<Vec<f64>> = [1.0, -0.5, 0.2].iter().map(|&v| vec![v]).collect();
    let k1 = gp_dot_prod_cov(&x1, sigma).unwrap();
    assert!(close(k1[(0, 1)], sigma * sigma - 0.5));
    // stan: gp_dot_prod_cov(array[] real, array[] real, real)
    // stan: gp_dot_prod_cov(array[] vector, array[] vector, real)
    let y = vec![vec![0.5, 0.5], vec![1.0, -1.0]];
    let kc = gp_dot_prod_cov_cross(&x, &y, sigma).unwrap();
    assert!(all_close(
        &kc,
        &Matrix::from_fn(3, 2, |i, j| sigma * sigma + dot(&x[i], &y[j]))
    ));
    let y1: Vec<Vec<f64>> = [2.0, 3.0].iter().map(|&v| vec![v]).collect();
    assert!(close(
        gp_dot_prod_cov_cross(&x1, &y1, sigma).unwrap()[(2, 1)],
        sigma * sigma + 0.6
    ));
}

#[test]
fn wiener_cdfs() {
    let (y, a, t0, w, v, sv, sw, st0) = (1.2, 1.5, 0.3, 0.45, 0.7, 0.2, 0.1, 0.05);
    let ys = [1.2, 0.9, 2.0];
    let sum = |f: &dyn Fn(f64) -> f64| ys.iter().map(|&y| f(y)).sum::<f64>();
    // stan: wiener_lcdf_unnorm(real, real, real, real, real)
    // stan: wiener_lcdf_unnorm(vector, vector, vector, vector, vector)
    let s4 = wiener4_lcdf_unnorm(y, a, t0, w, v).unwrap();
    assert!(s4 < 0.0 && wiener_lcdf_unnorm(y, a, t0, w, v).unwrap() == s4);
    let v4 =
        wiener4_lcdf_unnorm(&ys[..], &[a; 3][..], &[t0; 3][..], &[w; 3][..], &[v; 3][..]).unwrap();
    assert!(close(
        v4,
        sum(&|y| wiener4_lcdf_unnorm(y, a, t0, w, v).unwrap())
    ));
    // stan: wiener_lccdf_unnorm(real, real, real, real, real)
    // stan: wiener_lccdf_unnorm(vector, vector, vector, vector, vector)
    let c4 = wiener4_lccdf_unnorm(y, a, t0, w, v).unwrap();
    assert!(c4 < 0.0 && wiener_lccdf_unnorm(y, a, t0, w, v).unwrap() == c4);
    let vc4 =
        wiener4_lccdf_unnorm(&ys[..], &[a; 3][..], &[t0; 3][..], &[w; 3][..], &[v; 3][..]).unwrap();
    assert!(close(
        vc4,
        sum(&|y| wiener4_lccdf_unnorm(y, a, t0, w, v).unwrap())
    ));
    // stan: wiener_lcdf_unnorm(real, real, real, real, real, real)
    // stan: wiener_lcdf_unnorm(vector, vector, vector, vector, vector, real)
    // stan: wiener_lccdf_unnorm(real, real, real, real, real, real)
    // stan: wiener_lccdf_unnorm(vector, vector, vector, vector, vector, real)
    let p4 = wiener4_prec_lcdf_unnorm(y, a, t0, w, v, 1e-6).unwrap();
    assert!((p4 - s4).abs() < 1e-4);
    let pv = wiener4_prec_lcdf_unnorm(
        &ys[..],
        &[a; 3][..],
        &[t0; 3][..],
        &[w; 3][..],
        &[v; 3][..],
        1e-6,
    )
    .unwrap();
    assert!(close(
        pv,
        sum(&|y| wiener4_prec_lcdf_unnorm(y, a, t0, w, v, 1e-6).unwrap())
    ));
    let pc = wiener4_prec_lccdf_unnorm(
        &ys[..],
        &[a; 3][..],
        &[t0; 3][..],
        &[w; 3][..],
        &[v; 3][..],
        1e-6,
    )
    .unwrap();
    assert!(close(
        pc,
        sum(&|y| wiener4_prec_lccdf_unnorm(y, a, t0, w, v, 1e-6).unwrap())
    ));
    // The unnormalized cdf and ccdf split the probability of the upper boundary, so their
    // probabilities sum to the same total at every y.
    let mass = |y: f64| {
        wiener4_prec_lcdf_unnorm(y, a, t0, w, v, 1e-6)
            .unwrap()
            .exp()
            + wiener4_prec_lccdf_unnorm(y, a, t0, w, v, 1e-6)
                .unwrap()
                .exp()
    };
    assert!((mass(1.2) - mass(2.0)).abs() < 1e-4 && mass(1.2) < 1.0);
    // stan: wiener_lcdf_unnorm(real, real, real, real, real, real, real, real)
    // stan: wiener_lcdf_unnorm(vector, vector, vector, vector, vector, vector, vector, vector)
    // stan: wiener_lccdf_unnorm(real, real, real, real, real, real, real, real)
    // stan: wiener_lccdf_unnorm(vector, vector, vector, vector, vector, vector, vector, vector)
    let f = wiener_full_lcdf_unnorm(y, a, t0, w, v, sv, sw, st0).unwrap();
    let fc = wiener_full_lccdf_unnorm(y, a, t0, w, v, sv, sw, st0).unwrap();
    let full_mass = |y: f64| {
        wiener_full_lcdf_unnorm(y, a, t0, w, v, sv, sw, st0)
            .unwrap()
            .exp()
            + wiener_full_lccdf_unnorm(y, a, t0, w, v, sv, sw, st0)
                .unwrap()
                .exp()
    };
    assert!((f.exp() + fc.exp() - full_mass(2.0)).abs() < 1e-3);
    let n = |x: f64| [x; 3];
    let fv = wiener_full_lcdf_unnorm(
        &ys[..],
        &n(a)[..],
        &n(t0)[..],
        &n(w)[..],
        &n(v)[..],
        &n(sv)[..],
        &n(sw)[..],
        &n(st0)[..],
    )
    .unwrap();
    assert!(close(
        fv,
        sum(&|y| wiener_full_lcdf_unnorm(y, a, t0, w, v, sv, sw, st0).unwrap())
    ));
    let fcv = wiener_full_lccdf_unnorm(
        &ys[..],
        &n(a)[..],
        &n(t0)[..],
        &n(w)[..],
        &n(v)[..],
        &n(sv)[..],
        &n(sw)[..],
        &n(st0)[..],
    )
    .unwrap();
    assert!(close(
        fcv,
        sum(&|y| wiener_full_lccdf_unnorm(y, a, t0, w, v, sv, sw, st0).unwrap())
    ));
    // stan: wiener_lcdf_unnorm(real, real, real, real, real, real, real, real, real)
    // stan: wiener_lcdf_unnorm(vector, vector, vector, vector, vector, vector, vector, vector, real)
    // stan: wiener_lccdf_unnorm(real, real, real, real, real, real, real, real, real)
    // stan: wiener_lccdf_unnorm(vector, vector, vector, vector, vector, vector, vector, vector, real)
    let fp = wiener_full_prec_lcdf_unnorm(y, a, t0, w, v, sv, sw, st0, 1e-6).unwrap();
    assert!((fp - f).abs() < 1e-3);
    let fpv = wiener_full_prec_lccdf_unnorm(
        &ys[..],
        &n(a)[..],
        &n(t0)[..],
        &n(w)[..],
        &n(v)[..],
        &n(sv)[..],
        &n(sw)[..],
        &n(st0)[..],
        1e-6,
    )
    .unwrap();
    assert!(close(
        fpv,
        sum(&|y| wiener_full_prec_lccdf_unnorm(y, a, t0, w, v, sv, sw, st0, 1e-6).unwrap())
    ));
}

#[test]
fn poisson_binomial() {
    let theta = [0.2, 0.5, 0.7];
    // P(N <= k) from the exact distribution of the sum of three Bernoulli draws.
    let mut p = [0.0; 4];
    for mask in 0..8_u32 {
        let pr: f64 = (0..3)
            .map(|i| {
                if mask >> i & 1 == 1 {
                    theta[i]
                } else {
                    1.0 - theta[i]
                }
            })
            .product();
        p[mask.count_ones() as usize] += pr;
    }
    // Stan's _cdf is the probability, and its array form the product (probed: 0.55 and 0.1116).
    let cdf = |k: usize| p[..=k].iter().sum::<f64>();
    // stan: poisson_binomial_cdf(int, vector)
    // stan: poisson_binomial_cdf(int, row_vector)
    // stan: poisson_binomial_cdf(int, array[] real)
    assert!(close(poisson_binomial_cdf(1_i64, &theta).unwrap(), cdf(1)));
    // stan: poisson_binomial_cdf(array[] int, vector)
    // stan: poisson_binomial_cdf(array[] int, row_vector)
    // stan: poisson_binomial_cdf(array[] int, array[] real)
    assert!(close(
        poisson_binomial_cdf(&[0_i64, 2][..], &theta).unwrap(),
        cdf(0) * cdf(2)
    ));
    // stan: poisson_binomial_rng(vector)
    // stan: poisson_binomial_rng(row_vector)
    // stan: poisson_binomial_rng(array[] real)
    let mut r = rng();
    assert!((0..200).all(|_| (0..=3).contains(&poisson_binomial_rng(&theta, &mut r).unwrap())));
}

#[test]
fn samplers() {
    let s = spd();
    let l = cholesky_decompose(&s).unwrap();
    let mu = [0.5, -1.0, 2.0];
    let mus = vec![mu.to_vec(), vec![0.0, 0.0, 1.0]];
    // An array form draws each vector in turn from the same stream.
    // stan: multi_normal_rng(vector, matrix)
    // stan: multi_normal_rng(row_vector, matrix)
    // stan: multi_normal_rng(array[] vector, matrix)
    // stan: multi_normal_rng(array[] row_vector, matrix)
    let (mut r1, mut r2) = (rng(), rng());
    let arr = multi_normal_rng_array(&mus, &s, &mut r1).unwrap();
    let seq: Vec<_> = mus
        .iter()
        .map(|m| multi_normal_rng(m, &s, &mut r2).unwrap())
        .collect();
    assert_eq!(arr, seq);
    // stan: multi_normal_cholesky_rng(vector, matrix)
    // stan: multi_normal_cholesky_rng(row_vector, matrix)
    // stan: multi_normal_cholesky_rng(array[] vector, matrix)
    // stan: multi_normal_cholesky_rng(array[] row_vector, matrix)
    let (mut r1, mut r2) = (rng(), rng());
    let arr = multi_normal_cholesky_rng_array(&mus, &l, &mut r1).unwrap();
    let seq: Vec<_> = mus
        .iter()
        .map(|m| multi_normal_cholesky_rng(m, &l, &mut r2).unwrap())
        .collect();
    assert_eq!(arr, seq);
    // stan: multi_student_t_rng(real, vector, matrix)
    // stan: multi_student_t_rng(real, row_vector, matrix)
    // stan: multi_student_t_rng(real, array[] vector, matrix)
    // stan: multi_student_t_rng(real, array[] row_vector, matrix)
    let (mut r1, mut r2) = (rng(), rng());
    let arr = multi_student_t_rng_array(4.0, &mus, &s, &mut r1).unwrap();
    let seq: Vec<_> = mus
        .iter()
        .map(|m| multi_student_t_rng(4.0, m, &s, &mut r2).unwrap())
        .collect();
    assert_eq!(arr, seq);
    // stan: multi_student_t_cholesky_rng(real, vector, matrix)
    // stan: multi_student_t_cholesky_rng(real, row_vector, matrix)
    // stan: multi_student_t_cholesky_rng(real, array[] vector, matrix)
    // stan: multi_student_t_cholesky_rng(real, array[] row_vector, matrix)
    let (mut r1, mut r2) = (rng(), rng());
    let arr = multi_student_t_cholesky_rng_array(4.0, &mus, &l, &mut r1).unwrap();
    let seq: Vec<_> = mus
        .iter()
        .map(|m| multi_student_t_cholesky_rng(4.0, m, &l, &mut r2).unwrap())
        .collect();
    assert_eq!(arr, seq);

    let mut r = rng();
    let symmetric_pd =
        |x: &Matrix<f64>| all_close(x, &transpose(x)) && cholesky_decompose(x).is_ok();
    let lower = |x: &Matrix<f64>| (0..x.ncols()).all(|j| (0..j).all(|i| x[(i, j)] == 0.0));
    // stan: wishart_rng(real, matrix)
    // stan: inv_wishart_rng(real, matrix)
    assert!(symmetric_pd(&wishart_rng(5.0, &s, &mut r).unwrap()));
    assert!(symmetric_pd(&inv_wishart_rng(5.0, &s, &mut r).unwrap()));
    // stan: wishart_cholesky_rng(real, matrix)
    // stan: inv_wishart_cholesky_rng(real, matrix)
    assert!(lower(&wishart_cholesky_rng(5.0, &l, &mut r).unwrap()));
    assert!(lower(&inv_wishart_cholesky_rng(5.0, &l, &mut r).unwrap()));
    // stan: lkj_corr_rng(int, real)
    let c = lkj_corr_rng(3, 2.0, &mut r).unwrap();
    assert!(symmetric_pd(&c) && (0..3).all(|i| close(c[(i, i)], 1.0)));
    // stan: lkj_corr_cholesky_rng(int, real)
    let lc = lkj_corr_cholesky_rng(3, 2.0, &mut r).unwrap();
    assert!(lower(&lc) && (0..3).all(|i| close((0..3).map(|j| lc[(i, j)].powi(2)).sum(), 1.0)));
    // stan: dirichlet_rng(vector)
    let d = dirichlet_rng(&[1.0, 2.0, 3.0], &mut r).unwrap();
    assert!(close(d.iter().sum(), 1.0) && d.iter().all(|&x| x > 0.0));
    // stan: multinomial_rng(vector, int)
    // stan: multinomial_logit_rng(vector, int)
    // stan: dirichlet_multinomial_rng(vector, int)
    assert_eq!(
        multinomial_rng(&[0.2, 0.3, 0.5], 20, &mut r)
            .unwrap()
            .iter()
            .sum::<i64>(),
        20
    );
    assert_eq!(
        multinomial_logit_rng(&[0.1, -0.4, 1.0], 20, &mut r)
            .unwrap()
            .iter()
            .sum::<i64>(),
        20
    );
    assert_eq!(
        dirichlet_multinomial_rng(&[1.0, 2.0, 3.0], 20, &mut r)
            .unwrap()
            .iter()
            .sum::<i64>(),
        20
    );
    // stan: categorical_logit_rng(vector)
    assert!((1..=3).contains(&categorical_logit_rng(&[0.1, -0.4, 1.0], &mut r).unwrap()));
    // stan: ordered_logistic_rng(real, vector)
    // stan: ordered_probit_rng(real, vector)
    let cuts = [-1.0, 0.0, 1.5];
    assert!((1..=4).contains(&ordered_logistic_rng(0.3, &cuts, &mut r).unwrap()));
    assert!((1..=4).contains(&ordered_probit_rng(0.3, &cuts, &mut r).unwrap()));
    // stan: std_normal_rng()
    assert!(std_normal_rng(&mut r).unwrap().is_finite());
    // stan: bernoulli_logit_glm_rng(matrix, vector, vector)
    let x = m(4, 2, &[1.0, 0.5, -0.3, 2.0, 0.1, -1.0, 0.7, 0.2]);
    let ys = bernoulli_logit_glm_rng(&x, &[0.1, -0.2, 0.3, 0.0][..], &[0.5, -1.0], &mut r).unwrap();
    assert!(ys.len() == 4 && ys.iter().all(|&y| y == 0 || y == 1));
    // stan: bernoulli_logit_glm_rng(row_vector, vector, vector)
    let y1 = bernoulli_logit_glm_rng_row(&[1.0, 0.5], &[0.1][..], &[0.5, -1.0], &mut r).unwrap();
    assert!(y1.len() == 1 && (y1[0] == 0 || y1[0] == 1));
}

#[test]
fn hidden_markov() {
    // Two states over three steps.
    let log_omegas = m(2, 3, &[-0.5, -1.2, -2.0, -0.3, -0.7, -0.9]);
    let gamma = m(2, 2, &[0.8, 0.3, 0.2, 0.7]);
    let rho = [0.6, 0.4];
    // The marginal by brute force over the four, eight state paths.
    let mut total = 0.0;
    for path in 0..8_usize {
        let s = [path & 1, path >> 1 & 1, path >> 2 & 1];
        let mut p = rho[s[0]] * log_omegas[(s[0], 0)].exp();
        for t in 1..3 {
            p *= gamma[(s[t - 1], s[t])] * log_omegas[(s[t], t)].exp();
        }
        total += p;
    }
    // stan: hmm_marginal(matrix, matrix, vector)
    assert!(close(
        hmm_marginal(&log_omegas, &gamma, &rho).unwrap(),
        total.ln()
    ));
    // stan: hmm_hidden_state_prob(matrix, matrix, vector)
    let probs = hmm_hidden_state_prob(&log_omegas, &gamma, &rho).unwrap();
    assert!((0..3).all(|t| close(probs[(0, t)] + probs[(1, t)], 1.0)));
    // stan: hmm_latent_rng(matrix, matrix, vector)
    let states = hmm_latent_rng(&log_omegas, &gamma, &rho, &mut rng()).unwrap();
    assert!(states.len() == 3 && states.iter().all(|&s| s == 1 || s == 2));
}

/// `y_i^2 - theta_i - x_r[0] = 0`, solved by `y_i = sqrt(theta_i + x_r[0])`.
struct Squares;

impl AlgebraSystemData for Squares {
    fn eval<S: Real>(&self, y: &[S], theta: &[S], x_r: &[f64], _x_i: &[i64]) -> Vec<S> {
        y.iter()
            .zip(theta)
            .map(|(&y, &t)| y * y - t - S::from_f64(x_r[0]))
            .collect()
    }
}

/// `dy/dt = -k y` with `k = theta[0] * x_i[0]`.
struct Decay;

impl OdeSystemData for Decay {
    fn eval<S: Real>(&self, _t: S, y: &[S], theta: &[S], _x_r: &[f64], x_i: &[i64]) -> Vec<S> {
        let k = theta[0] * S::from_f64(x_i[0] as f64);
        y.iter().map(|&v| -(k * v)).collect()
    }
}

/// `x^theta` on `[0, 1]`, whose integral is `1 / (theta + 1)`.
struct Power;

impl Integrand for Power {
    fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
        pow(x, theta[0])
    }
}

#[test]
fn functionals() {
    let theta = [2.0, 5.0];
    let opts = SystemOptions {
        tolerance: 1e-12,
        function_tolerance: 1e-10,
        max_iterations: 200,
    };
    let want: Vec<f64> = theta.iter().map(|t| (t + 0.25f64).sqrt()).collect();
    let near = |got: &[f64]| got.iter().zip(&want).all(|(g, w)| (g - w).abs() < 1e-8);
    // stan: algebra_solver((vector, vector, data array[] real, data array[] int) => vector, vector, vector, array[] real, array[] int)
    // stan: algebra_solver((vector, vector, data array[] real, data array[] int) => vector, vector, vector, array[] real, array[] int, real, real, real)
    for o in [None, Some(&opts)] {
        assert!(near(
            &algebra_solver(&Squares, &[1.0, 1.0], &theta, &[0.25], &[], o).unwrap()
        ));
    }
    // stan: algebra_solver_newton((vector, vector, data array[] real, data array[] int) => vector, vector, vector, array[] real, array[] int)
    // stan: algebra_solver_newton((vector, vector, data array[] real, data array[] int) => vector, vector, vector, array[] real, array[] int, real, real, real)
    for o in [None, Some(&opts)] {
        assert!(near(
            &algebra_solver_newton(&Squares, &[1.0, 1.0], &theta, &[0.25], &[], o).unwrap()
        ));
    }
    // The data arguments x_r and x_i of Stan's integrate_1d are captured by the Rust integrand.
    // stan: integrate_1d((real, real, array[] real, data array[] real, data array[] int) => real, real, real, array[] real, array[] real, array[] int)
    // stan: integrate_1d((real, real, array[] real, data array[] real, data array[] int) => real, real, real, array[] real, array[] real, array[] int, real)
    for tol in [1.5e-8, 1e-10] {
        assert!((integrate_1d(&Power, 0.0, 1.0, &[2.5], tol).unwrap() - 1.0 / 3.5).abs() < 1e-7);
    }
    let (y0, ts, k) = ([2.0, 0.5], [0.5, 1.0, 2.0], 0.3);
    let exact = |got: &[Vec<f64>]| {
        got.len() == ts.len()
            && got.iter().zip(&ts).all(|(row, t)| {
                row.iter()
                    .zip(&y0)
                    .all(|(v, y)| (v - y * (-k * 2.0 * t).exp()).abs() < 1e-5)
            })
    };
    let tight = OdeOptions {
        relative_tolerance: 1e-8,
        absolute_tolerance: 1e-8,
        max_num_steps: 100_000,
    };
    // stan: integrate_ode((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int)
    assert!(exact(
        &integrate_ode(&Decay, &y0, 0.0, &ts, &[k], &[], &[2], None).unwrap()
    ));
    // stan: integrate_ode_rk45((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int)
    // stan: integrate_ode_rk45((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
    // stan: integrate_ode_adams((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int)
    // stan: integrate_ode_adams((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
    // stan: integrate_ode_bdf((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int)
    // stan: integrate_ode_bdf((real, array[] real, array[] real, data array[] real, data array[] int) => array[] real, array[] real, real, array[] real, array[] real, array[] real, array[] int, real, real, real)
    for o in [None, Some(&tight)] {
        assert!(exact(
            &integrate_ode_rk45(&Decay, &y0, 0.0, &ts, &[k], &[], &[2], o).unwrap()
        ));
        assert!(exact(
            &integrate_ode_adams(&Decay, &y0, 0.0, &ts, &[k], &[], &[2], o).unwrap()
        ));
        assert!(exact(
            &integrate_ode_bdf(&Decay, &y0, 0.0, &ts, &[k], &[], &[2], o).unwrap()
        ));
    }
    // stan: map_rect((vector, vector, data array[] real, data array[] int) => vector, vector, array[] vector, array[,] real, array[,] int)
    let out = map_rect(
        |phi: &[f64], th: &[f64], xr: &[f64], xi: &[i64]| {
            Ok(vec![phi[0] * th[0] + xr[0] + xi[0] as f64])
        },
        &[2.0],
        &[vec![1.0], vec![3.0]],
        &[vec![0.5], vec![0.25]],
        &[vec![1], vec![2]],
    )
    .unwrap();
    assert_eq!(out, vec![3.5, 8.25]);
}
