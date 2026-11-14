//! Every container signature of the elementwise Stan operators `add`, `subtract`,
//! `elt_multiply`, and `elt_divide`.
//!
//! Written by `tools/reference/gen_operators.py` from the CmdStan 2.40.0 signature dump: 72
//! Stan signatures. Two real matrices go through the library functions; the other forms are the
//! Rust operator through `apply_scalar_unary` or `apply_scalar_binary`.
#![allow(clippy::too_many_lines, clippy::unreadable_literal, clippy::float_cmp)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

fn flat<C: ScalarContainer>(c: &C) -> Vec<C::Scalar> {
    to_array_1d_nested(c)
}

#[test]
fn operator_signatures() {
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_unary(&y, |b| x + b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(real, vector)");
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_unary(&y, |b| x + b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(real, row_vector)");
    }
    {
        let x: f64 = 0.125;
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let got = flat(&apply_scalar_unary(&y, |b| x + b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(real, matrix)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a + y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(vector, real)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a + b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(vector, vector)");
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_unary(&y, |b| x + b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(complex, complex_vector)");
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_unary(&y, |b| x + b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(complex, complex_row_vector)");
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(1.375, -0.6875),
                Complex::new(1.5, -0.75),
                Complex::new(1.625, -0.8125),
                Complex::new(1.75, -0.875),
                Complex::new(1.875, -0.9375),
                Complex::new(2.0, -1.0),
            ],
        );
        let got = flat(&apply_scalar_unary(&y, |b| x + b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(complex, complex_matrix)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a + y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(row_vector, real)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a + b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(row_vector, row_vector)");
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a + y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(matrix, real)");
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let got = flat(&add(&x, &y).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(matrix, matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a + y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(complex_vector, complex)");
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a + b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "add(complex_vector, complex_vector)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a + y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(complex_row_vector, complex)");
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a + b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "add(complex_row_vector, complex_row_vector)"
        );
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        );
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a + y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(got, want[..got.len()], "add(complex_matrix, complex)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        );
        let y: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(1.375, -0.6875),
                Complex::new(1.5, -0.75),
                Complex::new(1.625, -0.8125),
                Complex::new(1.75, -0.875),
                Complex::new(1.875, -0.9375),
                Complex::new(2.0, -1.0),
            ],
        );
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a + b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a + b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "add(complex_matrix, complex_matrix)"
        );
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_unary(&y, |b| x / b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(real, vector)");
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_unary(&y, |b| x / b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(real, row_vector)");
    }
    {
        let x: f64 = 0.125;
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let got = flat(&apply_scalar_unary(&y, |b| x / b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(real, matrix)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a / y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(vector, real)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a / b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(vector, vector)");
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_unary(&y, |b| x / b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex, complex_vector)"
        );
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_unary(&y, |b| x / b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex, complex_row_vector)"
        );
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(1.375, -0.6875),
                Complex::new(1.5, -0.75),
                Complex::new(1.625, -0.8125),
                Complex::new(1.75, -0.875),
                Complex::new(1.875, -0.9375),
                Complex::new(2.0, -1.0),
            ],
        );
        let got = flat(&apply_scalar_unary(&y, |b| x / b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex, complex_matrix)"
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a / y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(row_vector, real)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a / b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(row_vector, row_vector)");
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a / y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(matrix, real)");
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let got = flat(&elt_divide(&x, &y).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_divide(matrix, matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a / y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex_vector, complex)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a / b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex_vector, complex_vector)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a / y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex_row_vector, complex)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a / b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex_row_vector, complex_row_vector)"
        );
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        );
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a / y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex_matrix, complex)"
        );
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        );
        let y: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(1.375, -0.6875),
                Complex::new(1.5, -0.75),
                Complex::new(1.625, -0.8125),
                Complex::new(1.75, -0.875),
                Complex::new(1.875, -0.9375),
                Complex::new(2.0, -1.0),
            ],
        );
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a / b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a / b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_divide(complex_matrix, complex_matrix)"
        );
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_unary(&y, |b| x * b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_multiply(real, vector)");
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_unary(&y, |b| x * b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_multiply(real, row_vector)");
    }
    {
        let x: f64 = 0.125;
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let got = flat(&apply_scalar_unary(&y, |b| x * b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_multiply(real, matrix)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a * y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_multiply(vector, real)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a * b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_multiply(vector, vector)");
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_unary(&y, |b| x * b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex, complex_vector)"
        );
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_unary(&y, |b| x * b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex, complex_row_vector)"
        );
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(1.375, -0.6875),
                Complex::new(1.5, -0.75),
                Complex::new(1.625, -0.8125),
                Complex::new(1.75, -0.875),
                Complex::new(1.875, -0.9375),
                Complex::new(2.0, -1.0),
            ],
        );
        let got = flat(&apply_scalar_unary(&y, |b| x * b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex, complex_matrix)"
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a * y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_multiply(row_vector, real)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a * b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(row_vector, row_vector)"
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a * y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_multiply(matrix, real)");
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let got = flat(&elt_multiply(&x, &y).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(got, want[..got.len()], "elt_multiply(matrix, matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a * y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex_vector, complex)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a * b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex_vector, complex_vector)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a * y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex_row_vector, complex)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a * b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex_row_vector, complex_row_vector)"
        );
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        );
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a * y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex_matrix, complex)"
        );
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        );
        let y: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(1.375, -0.6875),
                Complex::new(1.5, -0.75),
                Complex::new(1.625, -0.8125),
                Complex::new(1.75, -0.875),
                Complex::new(1.875, -0.9375),
                Complex::new(2.0, -1.0),
            ],
        );
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a * b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a * b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "elt_multiply(complex_matrix, complex_matrix)"
        );
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_unary(&y, |b| x - b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(real, vector)");
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_unary(&y, |b| x - b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(real, row_vector)");
    }
    {
        let x: f64 = 0.125;
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let got = flat(&apply_scalar_unary(&y, |b| x - b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(real, matrix)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a - y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(vector, real)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a - b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(vector, vector)");
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_unary(&y, |b| x - b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(complex, complex_vector)");
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_unary(&y, |b| x - b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "subtract(complex, complex_row_vector)"
        );
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(1.375, -0.6875),
                Complex::new(1.5, -0.75),
                Complex::new(1.625, -0.8125),
                Complex::new(1.75, -0.875),
                Complex::new(1.875, -0.9375),
                Complex::new(2.0, -1.0),
            ],
        );
        let got = flat(&apply_scalar_unary(&y, |b| x - b));
        let want: Vec<_> = vec![x; 6]
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(complex, complex_matrix)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a - y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(row_vector, real)");
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a - b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(row_vector, row_vector)");
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: f64 = 1.375;
        let got = flat(&apply_scalar_unary(&x, |a| a - y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(matrix, real)");
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let got = flat(&subtract(&x, &y).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(matrix, matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a - y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(complex_vector, complex)");
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a - b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "subtract(complex_vector, complex_vector)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a - y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "subtract(complex_row_vector, complex)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a - b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "subtract(complex_row_vector, complex_row_vector)"
        );
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        );
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = flat(&apply_scalar_unary(&x, |a| a - y));
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(vec![y; 6])
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(got, want[..got.len()], "subtract(complex_matrix, complex)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        );
        let y: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            3,
            vec![
                Complex::new(1.375, -0.6875),
                Complex::new(1.5, -0.75),
                Complex::new(1.625, -0.8125),
                Complex::new(1.75, -0.875),
                Complex::new(1.875, -0.9375),
                Complex::new(2.0, -1.0),
            ],
        );
        let got = flat(&apply_scalar_binary(&x, &y, |a, b| a - b).unwrap());
        let want: Vec<_> = flat(&x)
            .into_iter()
            .zip(flat(&y))
            .map(|(a, b)| a - b)
            .collect();
        assert_eq!(
            got,
            want[..got.len()],
            "subtract(complex_matrix, complex_matrix)"
        );
    }
}
