//! Values, closed-form gradients, and agreement between reverse rules and the generic algorithm.

use stanmath_rs_autodiff::{
    Dual64, Real, Tape, Var, finite_diff_gradient, grad, hessian, hessian_forward,
    jacobian_forward, vjp,
};
use stanmath_rs_core::{Error, Matrix};
use stanmath_rs_linalg::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// A 3 by 3 symmetric positive definite matrix built from 6 parameters (its lower triangle).
fn spd<T: Real>(p: &[T]) -> Matrix<T> {
    Matrix::from_rows(&[[p[0], p[1], p[3]], [p[1], p[2], p[4]], [p[3], p[4], p[5]]])
}

const SPD: [f64; 6] = [4.0, 1.2, 3.0, -0.7, 0.4, 2.5];

/// Every output of every operation, flattened, as one vector function of the parameters.
fn all_outputs<T: Real>(p: &[T]) -> Vec<T> {
    let a = spd(&p[..6]);
    let b = Matrix::from_column_major(3, 2, p[6..12].to_vec());
    let l = cholesky_decompose(&a).unwrap();
    let mut out = Vec::new();
    out.extend(l.as_slice());
    out.extend(multiply(&a, &b).unwrap().as_slice());
    out.extend(mdivide_left_tri_low(&l, &b).unwrap().as_slice());
    out.extend(mdivide_left_tri_low_transpose(&l, &b).unwrap().as_slice());
    out.extend(mdivide_left_spd(&a, &b).unwrap().as_slice());
    out.push(log_determinant_spd(&a).unwrap());
    out.push(quad_form(&a, &p[6..9]).unwrap());
    out.extend(multiply_lower_tri_self_transpose(&l).unwrap().as_slice());
    out
}

const PARAMS: [f64; 12] = [
    4.0, 1.2, 3.0, -0.7, 0.4, 2.5, 0.3, -1.1, 0.8, 2.0, 0.5, -0.4,
];

/// Reverse mode (custom operations with `f64` adjoints) against forward mode (the generic
/// algorithm on dual numbers), for a weighted sum of every output.
#[test]
fn reverse_rules_match_generic_algorithm() {
    let (values, jac) = jacobian_forward(|p: &[Dual64]| all_outputs(p), &PARAMS);
    let weights: Vec<f64> = (0..values.len())
        .map(|i| ((i * 7 % 11) as f64) - 5.0)
        .collect();
    let (_, reverse) = vjp(|p| all_outputs(p), &PARAMS, &weights);
    for j in 0..PARAMS.len() {
        let forward: f64 = (0..values.len()).map(|i| weights[i] * jac[(i, j)]).sum();
        assert!(
            close(reverse[j], forward, 1e-12),
            "parameter {j}: {} vs {forward}",
            reverse[j]
        );
    }
    // And against finite differences of the f64 evaluation.
    let fd = finite_diff_gradient(
        |p| {
            all_outputs(p)
                .iter()
                .zip(&weights)
                .map(|(o, w)| o * w)
                .sum()
        },
        &PARAMS,
        1e-3,
    );
    for j in 0..PARAMS.len() {
        assert!(
            close(reverse[j], fd[j], 1e-8),
            "parameter {j}: {} vs fd {}",
            reverse[j],
            fd[j]
        );
    }
}

#[test]
fn cholesky_reconstructs_its_input() {
    let a = spd(&SPD);
    let l = cholesky_decompose(&a).unwrap();
    let back = multiply_lower_tri_self_transpose(&l).unwrap();
    assert!(back.max_abs_diff(&a) < 1e-14);
    for i in 0..3 {
        for j in i + 1..3 {
            assert_eq!(l[(i, j)], 0.0);
        }
    }
}

#[test]
fn log_determinant_gradient_is_the_inverse() {
    let g = grad(|p| log_determinant_spd(&spd(p)).unwrap(), &SPD);
    let inv = inverse_spd(&spd(&SPD)).unwrap();
    // Parameters are the lower triangle: diagonal entries once, off-diagonal entries twice.
    let expected = [
        inv[(0, 0)],
        2.0 * inv[(1, 0)],
        inv[(1, 1)],
        2.0 * inv[(2, 0)],
        2.0 * inv[(2, 1)],
        inv[(2, 2)],
    ];
    for (k, (a, b)) in g.iter().zip(&expected).enumerate() {
        assert!(close(*a, *b, 1e-14), "{k}: {a} vs {b}");
    }
    let id = multiply(&spd(&SPD), &inv).unwrap();
    assert!(id.max_abs_diff(&Matrix::identity(3)) < 1e-14);
}

#[test]
fn quadratic_form_gradients() {
    // d/dx x^T A x = (A + A^T) x and d/dA_ij = x_i x_j for a general (nonsymmetric) A.
    let a = [1.0, -2.0, 0.5, 3.0];
    let x = [0.7, -1.3];
    let g = grad(
        |p| {
            let m = Matrix::from_column_major(2, 2, p[..4].to_vec());
            quad_form(&m, &p[4..]).unwrap()
        },
        &[a[0], a[1], a[2], a[3], x[0], x[1]],
    );
    let m = Matrix::from_column_major(2, 2, a.to_vec());
    for i in 0..2 {
        let want: f64 = (0..2).map(|j| (m[(i, j)] + m[(j, i)]) * x[j]).sum();
        assert!(close(g[4 + i], want, 1e-15));
        for j in 0..2 {
            assert!(close(g[j * 2 + i], x[i] * x[j], 1e-15));
        }
    }
}

#[test]
fn hessians_through_cholesky_agree() {
    let h = hessian(|p| log_determinant_spd(&spd(p)).unwrap(), &SPD);
    let hf = hessian_forward(|p| log_determinant_spd(&spd(p)).unwrap(), &SPD);
    assert!(h.max_abs_diff(&hf) < 1e-13);
    // Rows of the Hessian against finite differences of the reverse-mode gradient.
    for j in 0..6 {
        let col = finite_diff_gradient(
            |p| grad(|q| log_determinant_spd(&spd(q)).unwrap(), p)[j],
            &SPD,
            1e-3,
        );
        for i in 0..6 {
            assert!(close(h[(i, j)], col[i], 1e-7), "({i}, {j})");
        }
    }
}

#[test]
fn tape_grows_with_output_size() {
    for n in [4, 8, 16] {
        let tape = Tape::new();
        let a: Vec<Var<'_>> = (0..n * n).map(|i| tape.var(i as f64 * 0.1)).collect();
        let b: Vec<Var<'_>> = (0..n * n)
            .map(|i| tape.var(1.0 - i as f64 * 0.05))
            .collect();
        let before = tape.len();
        let c = multiply(
            &Matrix::from_column_major(n, n, a.clone()),
            &Matrix::from_column_major(n, n, b.clone()),
        )
        .unwrap();
        assert_eq!(
            tape.len() - before,
            n * n,
            "one node per output, not per multiply-add"
        );
        // d/dA_il sum_ij (AB)_ij = sum_j B_lj
        let total: Var<'_> = c.as_slice().iter().copied().sum();
        let g = total.gradient();
        for l in 0..n {
            let row_sum: f64 = (0..n).map(|j| b[j * n + l].value()).sum();
            for i in 0..n {
                assert!(close(g.wrt(a[l * n + i]), row_sum, 1e-14));
            }
        }
    }
}

#[test]
fn invalid_inputs_return_errors() {
    let not_pd = Matrix::from_rows(&[[1.0, 2.0], [2.0, 1.0]]);
    assert!(matches!(
        cholesky_decompose(&not_pd),
        Err(Error::NotPositiveDefinite { .. })
    ));
    let asym = Matrix::from_rows(&[[2.0, 1.0], [0.0, 2.0]]);
    assert!(matches!(
        cholesky_decompose(&asym),
        Err(Error::NotSymmetric { .. })
    ));
    let nan = Matrix::from_rows(&[[f64::NAN, 0.0], [0.0, 1.0]]);
    assert!(cholesky_decompose(&nan).is_err());
    let rect = Matrix::<f64>::zeros(2, 3);
    assert!(matches!(
        cholesky_decompose(&rect),
        Err(Error::DimensionMismatch { .. })
    ));
    assert!(multiply(&rect, &rect).is_err());
    let singular = Matrix::from_rows(&[[0.0, 0.0], [1.0, 1.0]]);
    assert!(mdivide_left_tri_low(&singular, &Matrix::identity(2)).is_err());
    // Degenerate sizes.
    let empty = Matrix::<f64>::zeros(0, 0);
    assert_eq!(cholesky_decompose(&empty).unwrap().len(), 0);
    assert_eq!(log_determinant_spd(&empty).unwrap(), 0.0);
    let one = Matrix::from_rows(&[[9.0]]);
    assert_eq!(cholesky_decompose(&one).unwrap()[(0, 0)], 3.0);
    // Errors also surface through reverse mode instead of panicking.
    let r = stanmath_rs_autodiff::try_grad(
        |p| log_determinant_spd(&Matrix::from_rows(&[[p[0], p[1]], [p[1], p[0]]])),
        &[1.0, 2.0],
    );
    assert!(r.is_err());
}
