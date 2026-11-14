//! The Stan signatures of the linear algebra, sorting, and indexing functions that no generator
//! reaches: products and quotients with a scalar, the vector forms of the solvers, the complex
//! forms of the matrix functions, the decompositions, the FFTs, the CSR functions, and the small
//! constructors. Each block names its Stan signatures in `// stan:` comments and checks the
//! Rust call against a defining property (a product, a reconstruction, a brute force sum), not
//! against itself.
#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::unreadable_literal,
    clippy::needless_range_loop
)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * b.abs().max(1.0)
}

fn cclose(a: C, b: C) -> bool {
    close(a.re, b.re) && close(a.im, b.im)
}

fn mclose(a: &Matrix<f64>, b: &Matrix<f64>) -> bool {
    a.shape() == b.shape()
        && a.as_slice()
            .iter()
            .zip(b.as_slice())
            .all(|(x, y)| close(*x, *y))
}

fn cmclose(a: &Matrix<C>, b: &Matrix<C>) -> bool {
    a.shape() == b.shape()
        && a.as_slice()
            .iter()
            .zip(b.as_slice())
            .all(|(x, y)| cclose(*x, *y))
}

fn vclose(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
}

fn cvclose(a: &[C], b: &[C]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| cclose(*x, *y))
}

/// A 3 by 3 real matrix.
fn a3() -> Matrix<f64> {
    Matrix::from_column_major(3, 3, vec![2.0, 0.5, -1.0, 0.3, 1.7, 0.4, -0.6, 0.2, 2.5])
}

/// A 3 by 3 symmetric positive definite matrix.
fn spd() -> Matrix<f64> {
    Matrix::from_column_major(3, 3, vec![4.0, 1.0, 0.5, 1.0, 3.0, -0.4, 0.5, -0.4, 2.0])
}

/// A 3 by 3 nonsingular complex matrix.
fn ca3() -> Matrix<C> {
    Matrix::from_fn(3, 3, |i, j| {
        c(
            1.0 + i as f64 - 0.5 * j as f64 + 0.7 * (i * j) as f64,
            0.3 * (i + 2 * j) as f64 - 0.4 + 0.2 * (i * i) as f64 + 0.5 * (j * j) as f64,
        )
    })
}

/// A 3 by 3 Hermitian complex matrix.
fn herm() -> Matrix<C> {
    let a = ca3();
    Matrix::from_fn(3, 3, |i, j| a[(i, j)] + a[(j, i)].conj())
}

fn cv() -> Vec<C> {
    vec![c(0.5, -1.0), c(1.5, 0.25), c(-0.75, 2.0)]
}

fn cmul(a: &Matrix<C>, b: &Matrix<C>) -> Matrix<C> {
    Matrix::from_fn(a.nrows(), b.ncols(), |i, j| {
        (0..a.ncols()).map(|k| a[(i, k)] * b[(k, j)]).sum()
    })
}

fn mul(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
    Matrix::from_fn(a.nrows(), b.ncols(), |i, j| {
        (0..a.ncols()).map(|k| a[(i, k)] * b[(k, j)]).sum()
    })
}

fn ctrans(a: &Matrix<C>) -> Matrix<C> {
    Matrix::from_fn(a.ncols(), a.nrows(), |i, j| a[(j, i)].conj())
}

#[test]
fn scalar_products_and_quotients() {
    let v = vec![0.5, -1.0, 2.0];
    let a = a3();
    // stan: multiply(real, vector)
    // stan: multiply(vector, real)
    // stan: multiply(real, row_vector)
    // stan: multiply(row_vector, real)
    let sv = scale(&Matrix::column_vector(v.clone()), 1.5);
    assert!(vclose(sv.as_slice(), &[0.75, -1.5, 3.0]));
    // stan: multiply(real, matrix)
    // stan: multiply(matrix, real)
    let sa = scale(&a, -2.0);
    assert!(mclose(&sa, &Matrix::from_fn(3, 3, |i, j| -2.0 * a[(i, j)])));
    // stan: multiply(vector, row_vector)
    let o = multiply_outer(&v, &[1.0, 3.0]);
    assert!(mclose(
        &o,
        &Matrix::from_fn(3, 2, |i, j| v[i] * [1.0, 3.0][j])
    ));
    // stan: multiply(row_vector, vector)
    assert!(close(dot_product(&v, &[1.0, 2.0, 3.0]), 4.5));
    // stan: multiply(row_vector, matrix)
    let rv = multiply_row_vector(&v, &a).unwrap();
    assert!(vclose(
        &rv,
        mul(&Matrix::from_column_major(1, 3, v.clone()), &a).as_slice()
    ));
    // stan: multiply(matrix, vector)
    let mv = multiply_vector(&a, &v).unwrap();
    assert!(vclose(
        &mv,
        mul(&a, &Matrix::column_vector(v.clone())).as_slice()
    ));

    // Complex scalars with complex containers: map the product (or quotient) over the entries,
    // which must equal the product with the diagonal matrix of the scalar.
    let z = c(0.4, -1.2);
    let ca = ca3();
    let zd = diag_matrix_complex(&[z; 3]);
    // stan: multiply(complex, complex_vector)
    // stan: multiply(complex_vector, complex)
    // stan: multiply(complex, complex_row_vector)
    // stan: multiply(complex_row_vector, complex)
    let zv: Vec<C> = cv().iter().map(|x| z * *x).collect();
    assert!(cvclose(&zv, &multiply_vector_complex(&zd, &cv()).unwrap()));
    // stan: multiply(complex, complex_matrix)
    // stan: multiply(complex_matrix, complex)
    assert!(cmclose(
        &ca.map(|x| *x * z),
        &multiply_complex(&zd, &ca).unwrap()
    ));
    // stan: divide(complex, complex_vector)
    // stan: divide(complex, complex_row_vector)
    // stan: divide(complex, complex_matrix)
    let q = ca.map(|x| z / *x);
    assert!(cmclose(
        &q.map(|x| *x * ca[(0, 0)]),
        &ca.map(|x| z / *x * ca[(0, 0)])
    ));
    assert!((0..9).all(|k| cclose(q.as_slice()[k] * ca.as_slice()[k], z)));
    // stan: divide(complex_vector, complex)
    // stan: divide(complex_row_vector, complex)
    // stan: divide(complex_matrix, complex)
    let d = ca.map(|x| *x / z);
    assert!(cmclose(&d.map(|x| *x * z), &ca));
    // stan: multiply(complex_vector, complex_row_vector)
    let co = multiply_outer(&cv(), &cv()[..2]);
    assert!(cmclose(
        &co,
        &Matrix::from_fn(3, 2, |i, j| cv()[i] * cv()[j])
    ));
    // stan: multiply(complex_row_vector, complex_vector)
    let dp = dot_product_complex(&cv(), &cv()).unwrap();
    assert!(cclose(dp, cv().iter().map(|x| x.conj() * *x).sum()));
    // stan: multiply(complex_row_vector, complex_matrix)
    let crv = multiply_row_vector_complex(&cv(), &ca).unwrap();
    assert!(cvclose(
        &crv,
        cmul(&Matrix::from_column_major(1, 3, cv()), &ca).as_slice()
    ));
    // stan: multiply(complex_matrix, complex_vector)
    let cmv = multiply_vector_complex(&ca, &cv()).unwrap();
    assert!(cvclose(
        &cmv,
        cmul(&ca, &Matrix::column_vector(cv())).as_slice()
    ));
    // stan: multiply(complex_matrix, complex_matrix)
    assert!(cmclose(
        &multiply_complex(&ca, &herm()).unwrap(),
        &cmul(&ca, &herm())
    ));
    // stan: add(complex, complex)
    // stan: subtract(complex, complex)
    // stan: plus(complex)
    assert!(cclose(z + c(1.0, 1.0), c(1.4, -0.2)));
    assert!(cclose(z - c(1.0, 1.0), c(-0.6, -2.2)));
    assert!(cclose(-(-z), z));
}

#[test]
fn shapes_and_slices() {
    let v = Matrix::column_vector(vec![1.0, 2.0, 3.0]);
    let r = Matrix::from_column_major(1, 3, vec![1.0, 2.0, 3.0]);
    let cvm = Matrix::column_vector(cv());
    let crm = Matrix::from_column_major(1, 3, cv());
    // stan: rows(vector)
    // stan: rows(row_vector)
    // stan: rows(matrix)
    // stan: rows(complex_vector)
    // stan: rows(complex_row_vector)
    // stan: rows(complex_matrix)
    // stan: cols(vector)
    // stan: cols(row_vector)
    // stan: cols(matrix)
    // stan: cols(complex_vector)
    // stan: cols(complex_row_vector)
    // stan: cols(complex_matrix)
    assert_eq!((rows(&v), cols(&v), rows(&r), cols(&r)), (3, 1, 1, 3));
    assert_eq!(
        (rows(&cvm), cols(&cvm), rows(&crm), cols(&crm)),
        (3, 1, 1, 3)
    );
    assert_eq!((rows(&a3()), cols(&ca3())), (3, 3));
    // stan: transpose(vector)
    // stan: transpose(row_vector)
    assert_eq!(transpose(&v).shape(), (1, 3));
    assert!(mclose(&transpose(&r), &v));
    // stan: transpose(complex_vector)
    // stan: transpose(complex_row_vector)
    // stan: transpose(complex_matrix)
    assert!(cmclose(&cvm.transpose(), &crm));
    let ct = ca3().transpose();
    assert!((0..3).all(|i| (0..3).all(|j| ct[(i, j)] == ca3()[(j, i)])));
    // stan: block(matrix, int, int, int, int)
    let b = block(&a3(), 2, 1, 2, 2).unwrap();
    assert!(mclose(&b, &Matrix::from_fn(2, 2, |i, j| a3()[(i + 1, j)])));
    // stan: block(complex_matrix, int, int, int, int)
    let cb = block(&ca3(), 1, 2, 3, 1).unwrap();
    assert!(cvclose(
        cb.as_slice(),
        &[ca3()[(0, 1)], ca3()[(1, 1)], ca3()[(2, 1)]]
    ));
    // stan: row(matrix, int)
    // stan: row(complex_matrix, int)
    assert!(vclose(&row(&a3(), 3).unwrap(), &[-1.0, 0.4, 2.5]));
    assert!(cvclose(
        &row(&ca3(), 1).unwrap(),
        &[ca3()[(0, 0)], ca3()[(0, 1)], ca3()[(0, 2)]]
    ));
    // stan: sub_row(matrix, int, int, int)
    // stan: sub_row(complex_matrix, int, int, int)
    assert!(vclose(&sub_row(&a3(), 2, 2, 2).unwrap(), &[1.7, 0.2]));
    assert!(cvclose(
        &sub_row(&ca3(), 3, 1, 2).unwrap(),
        &[ca3()[(2, 0)], ca3()[(2, 1)]]
    ));
    // stan: sub_col(matrix, int, int, int)
    // stan: sub_col(complex_matrix, int, int, int)
    assert!(vclose(&sub_col(&a3(), 2, 3, 2).unwrap(), &[0.2, 2.5]));
    assert!(cvclose(
        &sub_col(&ca3(), 1, 2, 2).unwrap(),
        &[ca3()[(0, 1)], ca3()[(1, 1)]]
    ));
    // stan: diag_matrix(vector)
    let dm = diag_matrix(&[1.0, 2.0]);
    assert!(mclose(
        &dm,
        &Matrix::from_column_major(2, 2, vec![1.0, 0.0, 0.0, 2.0])
    ));
    // stan: diag_matrix(complex_vector)
    let cdm = diag_matrix_complex(&cv());
    assert!(
        (0..3).all(|i| (0..3).all(|j| cdm[(i, j)] == if i == j { cv()[i] } else { c(0.0, 0.0) }))
    );
    // stan: diagonal(complex_matrix)
    assert!(cvclose(
        &diagonal_complex(&ca3()),
        &[ca3()[(0, 0)], ca3()[(1, 1)], ca3()[(2, 2)]]
    ));
    // stan: trace(complex_matrix)
    assert!(cclose(
        trace_complex(&ca3()),
        ca3()[(0, 0)] + ca3()[(1, 1)] + ca3()[(2, 2)]
    ));
    // stan: symmetrize_from_lower_tri(complex_matrix)
    let s = symmetrize_from_lower_tri_complex(&ca3()).unwrap();
    assert!((0..3).all(|i| (0..3).all(|j| s[(i, j)] == ca3()[(i.max(j), i.min(j))])));
    // stan: add_diag(matrix, real)
    // stan: add_diag(matrix, vector)
    // stan: add_diag(matrix, row_vector)
    let ad = add_diag(&a3(), 1.0);
    let adv = add_diag_vector(&a3(), &[1.0, 1.0, 1.0]).unwrap();
    assert!(mclose(&ad, &adv));
    assert!(close(ad[(1, 1)], 2.7) && close(ad[(0, 1)], 0.3));
    // stan: add_diag(complex_matrix, complex)
    // stan: add_diag(complex_matrix, complex_vector)
    // stan: add_diag(complex_matrix, complex_row_vector)
    let z = c(0.5, 0.5);
    assert!(cmclose(
        &add_diag_complex(&ca3(), z),
        &add_diag_vector_complex(&ca3(), &[z; 3]).unwrap()
    ));
    let acd = add_diag_vector_complex(&ca3(), &cv()).unwrap();
    assert!((0..3).all(|i| cclose(acd[(i, i)], ca3()[(i, i)] + cv()[i])));
}

#[test]
fn dot_products() {
    // stan: dot_product(complex_vector, complex_vector)
    // stan: dot_product(complex_vector, complex_row_vector)
    // stan: dot_product(complex_row_vector, complex_vector)
    // stan: dot_product(complex_row_vector, complex_row_vector)
    // stan: dot_product(array[] complex, array[] complex)
    let y = vec![c(1.0, 0.0), c(0.0, 1.0), c(2.0, -1.0)];
    let want: C = cv().iter().zip(&y).map(|(a, b)| a.conj() * *b).sum();
    assert!(cclose(dot_product_complex(&cv(), &y).unwrap(), want));
    // stan: dot_self(complex_vector)
    // stan: dot_self(complex_row_vector)
    assert!(cclose(
        dot_self_complex(&cv()),
        cv().iter().map(|a| a.conj() * *a).sum()
    ));
    // stan: rows_dot_self(vector)
    // stan: rows_dot_self(row_vector)
    assert!(vclose(
        rows_dot_self(&Matrix::column_vector(vec![2.0, -3.0])).as_slice(),
        &[4.0, 9.0]
    ));
    assert!(vclose(
        rows_dot_self(&Matrix::from_column_major(1, 2, vec![2.0, -3.0])).as_slice(),
        &[13.0]
    ));
    // stan: rows_dot_self(complex_vector)
    // stan: rows_dot_self(complex_row_vector)
    // stan: rows_dot_self(complex_matrix)
    let rd = rows_dot_self_complex(&ca3());
    for i in 0..3 {
        let want: C = (0..3).map(|j| ca3()[(i, j)].conj() * ca3()[(i, j)]).sum();
        assert!(cclose(rd.as_slice()[i], want), "{i}");
    }
    let rv = rows_dot_self_complex(&Matrix::column_vector(cv()));
    assert!((0..3).all(|i| cclose(rv.as_slice()[i], cv()[i].conj() * cv()[i])));
    let rr = rows_dot_self_complex(&Matrix::from_column_major(1, 3, cv()));
    assert!(cclose(
        rr.as_slice()[0],
        cv().iter().map(|a| a.conj() * *a).sum()
    ));
}

#[test]
fn sorting_ranking_and_quantiles() {
    let x = [0.3, -1.2, 2.5, 0.3, 1.1];
    let n = [4_i64, -2, 7, 0];
    // stan: sort_indices_asc(vector)
    // stan: sort_indices_asc(row_vector)
    // stan: sort_indices_asc(array[] real)
    // stan: sort_indices_asc(array[] int)
    assert_eq!(sort_indices_asc(&x).unwrap(), vec![2, 1, 4, 5, 3]);
    assert_eq!(sort_indices_asc(&n).unwrap(), vec![2, 4, 1, 3]);
    // stan: sort_indices_desc(vector)
    // stan: sort_indices_desc(row_vector)
    // stan: sort_indices_desc(array[] real)
    // stan: sort_indices_desc(array[] int)
    assert_eq!(sort_indices_desc(&x).unwrap(), vec![3, 5, 1, 4, 2]);
    assert_eq!(sort_indices_desc(&n).unwrap(), vec![3, 1, 4, 2]);
    // stan: rank(vector, int)
    // stan: rank(row_vector, int)
    // stan: rank(array[] real, int)
    // stan: rank(array[] int, int)
    assert_eq!(rank(&x, 5).unwrap(), 3);
    assert_eq!(rank(&n, 1).unwrap(), 2);
    // stan: sort_asc(array[] int)
    // stan: sort_desc(array[] int)
    assert_eq!(sort_asc(&n).unwrap(), vec![-2, 0, 4, 7]);
    assert_eq!(sort_desc(&n).unwrap(), vec![7, 4, 0, -2]);
    // stan: min(array[] int)
    // stan: max(array[] int)
    assert_eq!((min_int(&n).unwrap(), max_int(&n).unwrap()), (-2, 7));
    // stan: quantile(vector, real)
    // stan: quantile(row_vector, real)
    // stan: quantile(array[] real, real)
    // stan: quantile(vector, array[] real)
    // stan: quantile(row_vector, array[] real)
    // stan: quantile(array[] real, array[] real)
    // R's type 7 quantile, which Stan uses: sorted [-1.2, 0.3, 0.3, 1.1, 2.5].
    assert!(close(quantile(&x, 0.5).unwrap(), 0.3));
    assert!(close(quantile(&x, 0.9).unwrap(), 1.1 + 0.6 * 1.4));
    assert!(vclose(
        &quantile_array(&x, &[0.0, 0.25, 1.0]).unwrap(),
        &[-1.2, 0.3, 2.5]
    ));
}

#[test]
fn constructors() {
    // stan: linspaced_vector(int, real, real)
    assert!(vclose(
        &linspaced_vector::<f64>(5, 0.0, 1.0).unwrap(),
        &[0.0, 0.25, 0.5, 0.75, 1.0]
    ));
    // stan: linspaced_int_array(int, int, int)
    assert_eq!(linspaced_int_array(5, 1, 5).unwrap(), vec![1, 2, 3, 4, 5]);
    // stan: one_hot_int_array(int, int)
    assert_eq!(one_hot_int_array(4, 2).unwrap(), vec![0, 1, 0, 0]);
    // stan: ones_int_array(int)
    // stan: zeros_int_array(int)
    assert_eq!(ones_int_array(3), vec![1, 1, 1]);
    assert_eq!(zeros_int_array(2), vec![0, 0]);
    // stan: uniform_simplex(int)
    assert!(vclose(&uniform_simplex::<f64>(4).unwrap(), &[0.25; 4]));
    // stan: generate_laplace_options(int)
    // stan: generate_laplace_options(vector)
    let o = generate_laplace_options(3);
    assert_eq!(o.theta_init.as_deref(), Some(&[0.0, 0.0, 0.0][..]));
    let o = generate_laplace_options_from(&[0.5, -1.0]);
    assert_eq!(o.theta_init.as_deref(), Some(&[0.5, -1.0][..]));
}

#[test]
fn softmax_arrays() {
    let xs = [vec![0.5, -1.0, 2.0], vec![1.0, 1.0, 0.0]];
    // stan: softmax(array[] vector)
    // stan: softmax(array[] row_vector)
    // stan: log_softmax(array[] vector)
    // stan: log_softmax(array[] row_vector)
    let s: Vec<Vec<f64>> = xs.iter().map(|x| softmax(x)).collect();
    let ls: Vec<Vec<f64>> = xs.iter().map(|x| log_softmax(x)).collect();
    for (x, (s, ls)) in xs.iter().zip(s.iter().zip(&ls)) {
        let z: f64 = x.iter().map(|v| v.exp()).sum();
        assert!(vclose(
            s,
            &x.iter().map(|v| v.exp() / z).collect::<Vec<_>>()
        ));
        assert!(vclose(
            ls,
            &x.iter().map(|v| v - z.ln()).collect::<Vec<_>>()
        ));
    }
}

#[test]
fn solvers_and_quadratic_forms() {
    let b = vec![1.0, -2.0, 0.5];
    let col = |v: &[f64]| Matrix::column_vector(v.to_vec());
    // stan: mdivide_left(matrix, vector)
    let x = mdivide_left_vector(&a3(), &b).unwrap();
    assert!(vclose(mul(&a3(), &col(&x)).as_slice(), &b));
    // stan: mdivide_left_spd(matrix, vector)
    let x = mdivide_left_spd_vector(&spd(), &b).unwrap();
    assert!(vclose(mul(&spd(), &col(&x)).as_slice(), &b));
    // stan: mdivide_left_tri_low(matrix, vector)
    let l = cholesky_decompose(&spd()).unwrap();
    let x = mdivide_left_tri_low_vector(&l, &b).unwrap();
    assert!(vclose(mul(&l, &col(&x)).as_slice(), &b));
    let row = |v: &[f64]| Matrix::from_column_major(1, v.len(), v.to_vec());
    // stan: mdivide_right(row_vector, matrix)
    let x = mdivide_right_row_vector(&b, &a3()).unwrap();
    assert!(vclose(mul(&row(&x), &a3()).as_slice(), &b));
    // stan: mdivide_right_spd(row_vector, matrix)
    let x = mdivide_right_spd_row_vector(&b, &spd()).unwrap();
    assert!(vclose(mul(&row(&x), &spd()).as_slice(), &b));
    // stan: mdivide_right_tri_low(row_vector, matrix)
    let x = mdivide_right_tri_low_row_vector(&b, &l).unwrap();
    assert!(vclose(mul(&row(&x), &l).as_slice(), &b));
    // stan: mdivide_right(complex_row_vector, complex_matrix)
    // stan: mdivide_right(complex_matrix, complex_matrix)
    let cb = Matrix::from_column_major(1, 3, cv());
    let x = mdivide_right_complex(&cb, &ca3()).unwrap();
    assert!(cmclose(&cmul(&x, &ca3()), &cb));
    let x = mdivide_right_complex(&herm(), &ca3()).unwrap();
    assert!(cmclose(&cmul(&x, &ca3()), &herm()));
    // stan: quad_form(matrix, vector)
    let want: f64 = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| b[i] * a3()[(i, j)] * b[j])
        .sum();
    assert!(close(quad_form(&a3(), &b).unwrap(), want));
    // stan: quad_form_sym(matrix, vector)
    let want: f64 = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| b[i] * spd()[(i, j)] * b[j])
        .sum();
    assert!(close(quad_form_sym_vector(&spd(), &b).unwrap(), want));
    // stan: matrix_power(matrix, int)
    assert!(mclose(
        &matrix_power(&a3(), 3).unwrap(),
        &mul(&mul(&a3(), &a3()), &a3())
    ));
    assert!(mclose(
        &matrix_power(&a3(), 0).unwrap(),
        &Matrix::identity(3)
    ));
}

#[test]
fn sum_to_zero_matrix() {
    let y = Matrix::from_column_major(2, 3, vec![0.4, -1.1, 0.7, 0.2, -0.3, 1.5]);
    // stan: sum_to_zero_constrain(matrix)
    let x = sum_to_zero_matrix_constrain(&y);
    assert_eq!(x.shape(), (3, 4));
    assert!((0..3).all(|i| (0..4).map(|j| x[(i, j)]).sum::<f64>().abs() < 1e-12));
    assert!((0..4).all(|j| (0..3).map(|i| x[(i, j)]).sum::<f64>().abs() < 1e-12));
    // stan: sum_to_zero_jacobian(matrix)
    let mut lp = 0.0;
    assert!(mclose(&sum_to_zero_matrix_jacobian(&y, &mut lp), &x));
    assert_eq!(lp, 0.0);
    // stan: sum_to_zero_unconstrain(matrix)
    assert!(mclose(&sum_to_zero_matrix_unconstrain(&x).unwrap(), &y));
}

#[test]
fn decompositions() {
    let a = a3();
    let wide = Matrix::from_column_major(2, 3, vec![1.0, 0.5, -0.3, 2.0, 0.8, -1.1]);
    // stan: qr(matrix)
    // stan: qr_thin(matrix)
    for m in [&a, &wide, &wide.transpose()] {
        let (q, r) = qr(m);
        assert!(mclose(&mul(&q, &r), m));
        assert!(mclose(
            &mul(&q.transpose(), &q),
            &Matrix::identity(q.ncols())
        ));
        let (q, r) = qr_thin(m);
        assert!(mclose(&mul(&q, &r), m));
    }
    // stan: svd(matrix)
    let (u, s, v) = svd(&wide).unwrap();
    assert!(mclose(
        &mul(&mul(&u, &diag_matrix(&s)), &v.transpose()),
        &wide
    ));
    // stan: svd(complex_matrix)
    // stan: svd_U(complex_matrix)
    // stan: svd_V(complex_matrix)
    // stan: singular_values(complex_matrix)
    let ca = ca3();
    let (u, s, v) = (
        svd_U_complex(&ca).unwrap(),
        singular_values_complex(&ca).unwrap(),
        svd_V_complex(&ca).unwrap(),
    );
    let sd = diag_matrix_complex(&s.iter().map(|x| c(*x, 0.0)).collect::<Vec<_>>());
    assert!(cmclose(&cmul(&cmul(&u, &sd), &ctrans(&v)), &ca));
    assert!(s.windows(2).all(|w| w[0] >= w[1]));
    // stan: eigendecompose_sym(matrix)
    let (vec_, val) = eigendecompose_sym(&spd()).unwrap();
    assert!(mclose(&mul(&spd(), &vec_), &mul(&vec_, &diag_matrix(&val))));
    // stan: eigendecompose_sym(complex_matrix)
    // stan: eigenvalues_sym(complex_matrix)
    // stan: eigenvectors_sym(complex_matrix)
    let h = herm();
    let (hv, hl) = (
        eigenvectors_sym_complex(&h).unwrap(),
        eigenvalues_sym_complex(&h).unwrap(),
    );
    assert!(cmclose(
        &cmul(&h, &hv),
        &cmul(&hv, &diag_matrix_complex(&hl))
    ));
    assert!(hl.iter().all(|l| l.im.abs() < 1e-12) && hl.windows(2).all(|w| w[0].re <= w[1].re));
    // stan: eigendecompose(matrix)
    let (ev, el) = eigendecompose(&a).unwrap();
    let ac = a.map(|x| c(*x, 0.0));
    assert!(cmclose(
        &cmul(&ac, &ev),
        &cmul(&ev, &diag_matrix_complex(&el))
    ));
    // stan: eigendecompose(complex_matrix)
    let (ev, el) = (
        eigenvectors_complex(&ca).unwrap(),
        eigenvalues_complex(&ca).unwrap(),
    );
    assert!(cmclose(
        &cmul(&ca, &ev),
        &cmul(&ev, &diag_matrix_complex(&el))
    ));
    // stan: complex_schur_decompose(matrix)
    let (u, t) = complex_schur_decompose(&a).unwrap();
    assert!(cmclose(&cmul(&cmul(&u, &t), &ctrans(&u)), &ac));
    assert!((0..3).all(|i| (0..i).all(|j| t[(i, j)].abs() < 1e-12)));
    // stan: complex_schur_decompose(complex_matrix)
    let (u, t) = complex_schur_decompose_complex(&ca).unwrap();
    assert!(cmclose(&cmul(&cmul(&u, &t), &ctrans(&u)), &ca));
    assert!((0..3).all(|i| (0..i).all(|j| t[(i, j)].abs() < 1e-12)));
}

#[test]
fn fourier_transforms() {
    let x = cv();
    let dft = |x: &[C], sign: f64| -> Vec<C> {
        let n = x.len();
        (0..n)
            .map(|k| {
                (0..n)
                    .map(|j| {
                        let t = sign * 2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                        x[j] * c(t.cos(), t.sin())
                    })
                    .sum()
            })
            .collect()
    };
    // stan: fft(complex_vector)
    // stan: inv_fft(complex_vector)
    let f = fft(&x);
    assert!(cvclose(&f, &dft(&x, -1.0)));
    assert!(cvclose(&inv_fft(&f), &x));
    // stan: fft2(complex_matrix)
    // stan: inv_fft2(complex_matrix)
    let m = Matrix::from_fn(2, 3, |i, j| {
        c(i as f64 - 0.5 * j as f64, 0.25 * (i * j) as f64 + 0.1)
    });
    let f2 = fft2(&m);
    // A two dimensional DFT is the DFT of each row, then of each column.
    let rows_done: Vec<Vec<C>> = (0..2)
        .map(|i| dft(&(0..3).map(|j| m[(i, j)]).collect::<Vec<_>>(), -1.0))
        .collect();
    for j in 0..3 {
        let col = dft(&(0..2).map(|i| rows_done[i][j]).collect::<Vec<_>>(), -1.0);
        assert!((0..2).all(|i| cclose(f2[(i, j)], col[i])));
    }
    assert!(cmclose(&inv_fft2(&f2), &m));
}

#[test]
fn compressed_sparse_rows() {
    let a = Matrix::from_column_major(
        3,
        4,
        vec![1.0, 0.0, 4.0, 0.0, 2.0, 0.0, 0.0, 3.0, 5.0, 0.0, 0.0, 6.0],
    );
    // stan: csr_extract(matrix)
    // stan: csr_extract_w(matrix)
    // stan: csr_extract_v(matrix)
    // stan: csr_extract_u(matrix)
    let s = csr_extract(&a);
    assert_eq!(s.values, csr_extract_w(&a));
    assert_eq!(s.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(csr_extract_v(&a), vec![0, 1, 2, 0, 2, 3]);
    assert_eq!(csr_extract_u(&a), vec![0, 1, 3, 6]);
    // stan: csr_to_dense_matrix(int, int, vector, array[] int, array[] int)
    assert!(mclose(&csr_to_dense_matrix(&s).unwrap(), &a));
    // stan: csr_matrix_times_vector(int, int, vector, array[] int, array[] int, vector)
    let b = [1.0, -1.0, 0.5, 2.0];
    let got = csr_matrix_times_vector(&s, &b).unwrap();
    assert!(vclose(
        &got,
        mul(&a, &Matrix::column_vector(b.to_vec())).as_slice()
    ));
}
