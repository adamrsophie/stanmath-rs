//! Algebraic identities of the decompositions, solves, transforms, and matrix functions: every
//! factorization reconstructs its input, every solve inverts its product, and every transform has
//! its inverse. None of these needs a reference value.

use stanmath_rs_autodiff::Complex;
use stanmath_rs_core::Matrix;
use stanmath_rs_linalg::*;

type C = Complex<f64>;

const TOL: f64 = 1e-9;

/// A nonsymmetric 4 by 4 matrix with two real and two complex eigenvalues.
fn general() -> Matrix<f64> {
    Matrix::from_rows(&[
        [1.0, 2.0, -0.5, 0.3],
        [-1.5, 0.4, 1.1, 0.0],
        [0.2, -0.7, 2.2, 0.9],
        [0.6, 0.1, -0.4, -1.3],
    ])
}

/// A 3 by 3 symmetric positive definite matrix.
fn spd() -> Matrix<f64> {
    Matrix::from_rows(&[[4.0, 1.2, -0.7], [1.2, 3.0, 0.4], [-0.7, 0.4, 2.5]])
}

fn mul(a: &Matrix<f64>, b: &Matrix<f64>) -> Matrix<f64> {
    multiply(a, b).unwrap()
}

fn cmul(a: &Matrix<C>, b: &Matrix<C>) -> Matrix<C> {
    assert_eq!(a.ncols(), b.nrows());
    Matrix::from_fn(a.nrows(), b.ncols(), |i, j| {
        (0..a.ncols()).fold(C::new(0.0, 0.0), |s, k| s + a[(i, k)] * b[(k, j)])
    })
}

fn conj_transpose(a: &Matrix<C>) -> Matrix<C> {
    Matrix::from_fn(a.ncols(), a.nrows(), |i, j| {
        C::new(a[(j, i)].re, -a[(j, i)].im)
    })
}

fn to_complex(a: &Matrix<f64>) -> Matrix<C> {
    a.map(|&x| C::new(x, 0.0))
}

fn assert_close(name: &str, a: &Matrix<f64>, b: &Matrix<f64>) {
    let d = a.max_abs_diff(b);
    assert!(d < TOL, "{name}: max difference {d}");
}

fn assert_close_c(name: &str, a: &Matrix<C>, b: &Matrix<C>) {
    assert_eq!(a.shape(), b.shape(), "{name}: shapes");
    let d = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x.re - y.re).hypot(x.im - y.im))
        .fold(0.0, f64::max);
    assert!(d < TOL, "{name}: max difference {d}");
}

#[test]
fn qr_reconstructs_and_q_is_orthogonal() {
    let a = Matrix::from_rows(&[
        [1.0, 2.0, 0.5],
        [-1.0, 0.3, 2.0],
        [0.7, -0.4, 1.1],
        [2.0, 1.0, -1.0],
    ]);
    let (q, r) = qr(&a);
    assert_close("qr", &mul(&q, &r), &a);
    assert_close(
        "qr orthogonal",
        &mul(&q.transpose(), &q),
        &Matrix::identity(q.ncols()),
    );
    let (q, r) = qr_thin(&a);
    assert_eq!(q.shape(), (4, 3));
    assert_close("qr_thin", &mul(&q, &r), &a);
    for i in 0..r.nrows() {
        for j in 0..i.min(r.ncols()) {
            assert!(
                r[(i, j)].abs() < TOL,
                "R not upper triangular at ({i}, {j})"
            );
        }
    }
}

#[test]
fn svd_reconstructs() {
    let a = Matrix::from_rows(&[
        [1.0, 2.0, 0.5],
        [-1.0, 0.3, 2.0],
        [0.7, -0.4, 1.1],
        [2.0, 1.0, -1.0],
    ]);
    let (u, s, v) = svd(&a).unwrap();
    assert!(
        s.windows(2).all(|w| w[0] >= w[1]),
        "singular values not descending"
    );
    let us = Matrix::from_fn(u.nrows(), u.ncols(), |i, j| u[(i, j)] * s[j]);
    assert_close("svd", &mul(&us, &v.transpose()), &a);
}

#[test]
fn symmetric_eigendecomposition_reconstructs() {
    let a = spd();
    let (v, d) = eigendecompose_sym(&a).unwrap();
    let vd = Matrix::from_fn(3, 3, |i, j| v[(i, j)] * d[j]);
    assert_close("eigendecompose_sym", &mul(&vd, &v.transpose()), &a);
}

#[test]
fn complex_schur_reconstructs() {
    let a = general();
    let (u, t) = complex_schur_decompose(&a).unwrap();
    let back = cmul(&cmul(&u, &t), &conj_transpose(&u));
    assert_close_c("complex_schur", &back, &to_complex(&a));
    assert_close_c(
        "U unitary",
        &cmul(&conj_transpose(&u), &u),
        &to_complex(&Matrix::identity(4)),
    );
    for i in 0..4 {
        for j in 0..i {
            assert!(
                t[(i, j)].re.hypot(t[(i, j)].im) < TOL,
                "T not upper triangular"
            );
        }
    }
}

#[test]
fn eigenvectors_satisfy_the_eigen_equation() {
    let a = general();
    let (v, lambda) = eigendecompose(&a).unwrap();
    let av = cmul(&to_complex(&a), &v);
    let vl = Matrix::from_fn(4, 4, |i, j| v[(i, j)] * lambda[j]);
    assert_close_c("A V = V diag(lambda)", &av, &vl);
    let (pv, pd) = pseudo_eigendecompose(&a).unwrap();
    assert_close("A V = V D", &mul(&a, &pv), &mul(&pv, &pd));
}

#[test]
fn fft_round_trips() {
    let v: Vec<C> = (0..7)
        .map(|k| C::new((k as f64 * 0.7).sin(), (k as f64 * 0.3).cos()))
        .collect();
    let y = fft(&v);
    // y_0 is the sum.
    let s = v.iter().fold(C::new(0.0, 0.0), |a, &b| a + b);
    assert!((y[0].re - s.re).abs() < TOL && (y[0].im - s.im).abs() < TOL);
    let back = inv_fft(&y);
    for (a, b) in back.iter().zip(&v) {
        assert!((a.re - b.re).abs() < TOL && (a.im - b.im).abs() < TOL);
    }
    let m = Matrix::from_fn(3, 5, |i, j| {
        C::new(i as f64 - 0.5 * j as f64, (i * j) as f64 * 0.1)
    });
    assert_close_c("fft2", &inv_fft2(&fft2(&m)), &m);
}

#[test]
fn solves_invert_their_products() {
    let a = general();
    let b = Matrix::from_rows(&[[1.0, 0.5], [-2.0, 0.3], [0.7, 1.1], [0.0, -0.4]]);
    assert_close("mdivide_left", &mul(&a, &mdivide_left(&a, &b).unwrap()), &b);
    let bt = b.transpose();
    assert_close(
        "mdivide_right",
        &mul(&mdivide_right(&bt, &a).unwrap(), &a),
        &bt,
    );
    assert_close(
        "inverse",
        &mul(&a, &inverse(&a).unwrap()),
        &Matrix::identity(4),
    );

    let s = spd();
    let f = LdltFactor::new(&s).unwrap();
    let b3 = Matrix::from_rows(&[[1.0], [-0.5], [2.0]]);
    assert_close(
        "mdivide_left_ldlt",
        &mul(&s, &mdivide_left_ldlt(&f, &b3).unwrap()),
        &b3,
    );
    let ld = log_determinant_ldlt(&f);
    assert!((ld - log_determinant(&s).unwrap()).abs() < TOL);
    assert!((ld - determinant(&s).unwrap().ln()).abs() < TOL);

    let l = Matrix::from_rows(&[[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [-0.3, 0.8, 1.2]]);
    let u = l.transpose();
    for (tri, m) in [(Triangle::Lower, &l), (Triangle::Upper, &u)] {
        assert_close(
            "mdivide_left_tri",
            &mul(m, &mdivide_left_tri(m, &b3, tri).unwrap()),
            &b3,
        );
        let r = b3.transpose();
        assert_close(
            "mdivide_right_tri",
            &mul(&mdivide_right_tri(&r, m, tri).unwrap(), m),
            &r,
        );
    }
    // chol2inv(L) = (L L^T)^-1.
    let llt = mul(&l, &u);
    assert_close(
        "chol2inv",
        &mul(&llt, &chol2inv(&l).unwrap()),
        &Matrix::identity(3),
    );
}

#[test]
fn matrix_functions() {
    let a = general();
    assert_close(
        "matrix_power 0",
        &matrix_power(&a, 0).unwrap(),
        &Matrix::identity(4),
    );
    assert_close(
        "matrix_power 3",
        &matrix_power(&a, 3).unwrap(),
        &mul(&a, &mul(&a, &a)),
    );
    // exp(A) exp(-A) = I.
    let neg = a.map(|x| -x);
    let e = matrix_exp(&a).unwrap();
    assert_close(
        "matrix_exp",
        &mul(&e, &matrix_exp(&neg).unwrap()),
        &Matrix::identity(4),
    );
    // exp(diag(d)) = diag(exp(d)).
    let d = [0.3, -1.2, 2.0];
    let ed = matrix_exp(&diag_matrix(&d)).unwrap();
    assert_close("matrix_exp diagonal", &ed, &diag_matrix(&d.map(f64::exp)));
    let b = Matrix::from_rows(&[[1.0], [0.0], [2.0], [-1.0]]);
    assert_close(
        "matrix_exp_multiply",
        &matrix_exp_multiply(&a, &b).unwrap(),
        &mul(&e, &b),
    );
    // Moore-Penrose conditions for a tall matrix.
    let t = Matrix::from_rows(&[[1.0, 2.0], [-1.0, 0.3], [0.7, -0.4]]);
    let g = generalized_inverse(&t).unwrap();
    assert_close("A G A = A", &mul(&t, &mul(&g, &t)), &t);
    assert_close("G A G = G", &mul(&g, &mul(&t, &g)), &g);
}

#[test]
fn sparse_round_trip() {
    let a = Matrix::from_rows(&[
        [0.0, 1.5, 0.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, -1.0, 0.0],
    ]);
    let csr = csr_extract(&a);
    assert_close(
        "csr_to_dense_matrix",
        &csr_to_dense_matrix(&csr).unwrap(),
        &a,
    );
    assert_eq!(csr_extract_w(&a), vec![1.5, 2.0, 3.0, -1.0]);
    assert_eq!(csr_extract_v(&a), vec![1, 3, 0, 2]);
    assert_eq!(csr_extract_u(&a), vec![0, 2, 2, 4]);
    let x = [1.0, -1.0, 0.5, 2.0];
    let y = csr_matrix_times_vector(&csr, &x).unwrap();
    let dense = mul(&a, &Matrix::column_vector(x.to_vec()));
    assert_eq!(y.len(), 3);
    for (i, v) in y.iter().enumerate() {
        assert!((v - dense[(i, 0)]).abs() < TOL);
    }
}

#[test]
fn sorting_and_order_statistics() {
    let v = [3.0, -1.0, 2.5, 0.0, 2.5];
    assert_eq!(sort_asc(&v).unwrap(), vec![-1.0, 0.0, 2.5, 2.5, 3.0]);
    assert_eq!(sort_desc(&v).unwrap(), vec![3.0, 2.5, 2.5, 0.0, -1.0]);
    let idx = sort_indices_asc(&v).unwrap();
    // One based, as in Stan.
    let sorted: Vec<f64> = idx.iter().map(|&i| v[i - 1]).collect();
    assert_eq!(sorted, sort_asc(&v).unwrap());
    assert_eq!(rank(&v, 1).unwrap(), 4);
    assert_eq!(rank(&v, 2).unwrap(), 0);
    assert!(sort_asc(&[1.0, f64::NAN]).is_err());
    // The median of an odd sample.
    assert!((quantile(&[5.0, 1.0, 3.0], 0.5).unwrap() - 3.0).abs() < TOL);
    assert_eq!(prod(&[2.0, -1.5, 4.0]), -12.0);
    assert_eq!(prod_int(&[2, -3, 4]), -24);
}

#[test]
fn autocorrelation_starts_at_one() {
    let y = [0.3, 1.2, -0.4, 0.8, 2.1, -1.0, 0.5];
    let ac = autocorrelation(&y);
    assert!((ac[0] - 1.0).abs() < TOL);
    let acov = autocovariance(&y);
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    assert!((acov[0] - var).abs() < TOL, "{} vs {var}", acov[0]);
}

#[test]
fn gp_kernels_are_symmetric_with_the_marginal_variance_on_the_diagonal() {
    let x: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![0.5, -0.3], vec![2.0, 0.4]];
    let (sigma, l) = (1.3, 0.8);
    for (name, k) in [
        ("exp_quad", gp_exp_quad_cov(&x, sigma, l).unwrap()),
        ("matern32", gp_matern32_cov(&x, sigma, l).unwrap()),
        ("matern52", gp_matern52_cov(&x, sigma, l).unwrap()),
        ("exponential", gp_exponential_cov(&x, sigma, l).unwrap()),
        ("periodic", gp_periodic_cov(&x, sigma, l, 1.7).unwrap()),
    ] {
        assert_close(name, &k, &k.transpose());
        for i in 0..3 {
            assert!((k[(i, i)] - sigma * sigma).abs() < TOL, "{name} diagonal");
        }
    }
    let cross = gp_exp_quad_cov_cross(&x, &x, sigma, l).unwrap();
    assert_close("cross", &cross, &gp_exp_quad_cov(&x, sigma, l).unwrap());
}

/// Values from `tools/stan-reference/stan_probe.cpp` run against Stan Math 5.4.0 on 2026-11-12:
/// `dot_product` and the `_dot_self` forms conjugate the first argument, while
/// `columns_dot_product` and `rows_dot_product` do not.
#[test]
fn complex_dot_products_match_stan() {
    let a = [C::new(1.0, 2.0), C::new(3.0, -1.0)];
    let b = [C::new(2.0, -1.0), C::new(0.5, 1.0)];
    let eq = |z: C, re: f64, im: f64| (z.re - re).abs() < TOL && (z.im - im).abs() < TOL;
    assert!(eq(dot_product_complex(&a, &b).unwrap(), 0.5, -1.5));
    assert!(eq(dot_self_complex(&a), 15.0, 0.0));
    let m = Matrix::from_rows(&[
        [C::new(1.0, 2.0), C::new(0.0, -1.0)],
        [C::new(3.0, -1.0), C::new(2.0, 0.5)],
    ]);
    let n = Matrix::from_rows(&[
        [C::new(2.0, -1.0), C::new(1.0, 1.0)],
        [C::new(0.5, 1.0), C::new(-1.0, 2.0)],
    ]);
    let all = |got: Vec<C>, want: &[(f64, f64)]| {
        got.len() == want.len() && got.iter().zip(want).all(|(z, &(re, im))| eq(*z, re, im))
    };
    assert!(all(
        columns_dot_product_complex(&m, &n).unwrap(),
        &[(6.5, 5.5), (-2.0, 2.5)]
    ));
    assert!(all(
        rows_dot_product_complex(&m, &n).unwrap(),
        &[(5.0, 2.0), (-0.5, 6.0)]
    ));
    assert!(all(
        columns_dot_self_complex(&m),
        &[(15.0, 0.0), (5.25, 0.0)]
    ));
    assert!(all(rows_dot_self_complex(&m), &[(6.0, 0.0), (14.25, 0.0)]));
    // Matrix products do not conjugate, although multiply(row_vector, vector) does.
    assert!(all(
        multiply_vector_complex(&m, &b).unwrap(),
        &[(5.0, 2.5), (5.5, -2.75)]
    ));
    assert!(all(
        multiply_row_vector_complex(&a, &m).unwrap(),
        &[(5.0, -2.0), (8.5, -1.5)]
    ));
}
