//! Every signature of the Stan conversion and utility functions `sum`, `prod`, `cumulative_sum`,
//! `to_vector`, `to_row_vector`, `to_array_1d`, `to_array_2d`, `to_vector_array`,
//! `to_row_vector_array`, `to_matrix`, `append_col`, `append_row`, `fma`, `log_mix`,
//! `hypergeometric_3F2`, and `hypergeometric_pFq`.
//!
//! Written by `tools/reference/gen_utilities.py` from the CmdStan 2.40.0 signature dump: 187
//! Stan signatures, each mapped to its Rust form and checked against the definition written out
//! from the input's elements.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::redundant_clone,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::redundant_closure
)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

trait Near {
    fn near(&self, other: &Self) -> bool;
}

impl Near for f64 {
    fn near(&self, other: &Self) -> bool {
        (self - other).abs() <= 1e-13 * other.abs().max(1.0)
    }
}

impl Near for i64 {
    fn near(&self, other: &Self) -> bool {
        self == other
    }
}

impl Near for Complex<f64> {
    fn near(&self, other: &Self) -> bool {
        self.re.near(&other.re) && self.im.near(&other.im)
    }
}

impl<T: Near> Near for Vec<T> {
    fn near(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a.near(b))
    }
}

impl<T: Near + Copy> Near for Matrix<T> {
    fn near(&self, other: &Self) -> bool {
        self.nrows() == other.nrows()
            && self.ncols() == other.ncols()
            && self
                .as_slice()
                .iter()
                .zip(other.as_slice())
                .all(|(a, b)| a.near(b))
    }
}

fn near<T: Near>(a: &T, b: &T) -> bool {
    a.near(b)
}

fn close(a: &f64, b: &f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

fn check(failures: &mut Vec<String>, label: &str, ok: bool) {
    if !ok {
        failures.push(label.to_string());
    }
}

fn flat<C: ScalarContainer>(c: &C) -> Vec<C::Scalar> {
    to_array_1d_nested(c)
}

fn running_sums<T: Copy + std::ops::Add<Output = T>>(v: Vec<T>) -> Vec<T> {
    let mut out: Vec<T> = Vec::new();
    for x in v {
        out.push(out.last().map_or(x, |&s| s + x));
    }
    out
}

/// `sum_n log(sum_k theta_k exp(lp_nk))`.
fn log_mix_definition(theta: &[f64], lps: &[Vec<f64>]) -> f64 {
    lps.iter()
        .map(|lp| {
            theta
                .iter()
                .zip(lp)
                .map(|(t, l)| t * l.exp())
                .sum::<f64>()
                .ln()
        })
        .sum()
}

/// The hypergeometric series `sum_k prod (a)_k / prod (b)_k z^k / k!`.
fn pfq_series(a: &[f64], b: &[f64], z: f64) -> f64 {
    let (mut term, mut s) = (1.0, 1.0);
    for k in 0..200 {
        let k = f64::from(k);
        term *= a.iter().map(|x| x + k).product::<f64>() / b.iter().map(|x| x + k).product::<f64>()
            * z
            / (k + 1.0);
        s += term;
    }
    s
}

#[test]
fn utility_signatures() {
    let mut failures: Vec<String> = Vec::new();
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = sum(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = sum(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(row_vector)", near(&got, &want));
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = sum(x.as_slice());
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(matrix)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = sum_complex(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(complex_vector)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = sum_complex(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(complex_row_vector)", near(&got, &want));
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
        let got = sum_complex(x.as_slice());
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(complex_matrix)", near(&got, &want));
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64];
        let got = x.iter().sum::<i64>();
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(array[] int)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = sum(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(array[] real)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = sum_complex(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a + b).unwrap();
        check(&mut failures, "sum(array[] complex)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = prod(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = prod(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(row_vector)", near(&got, &want));
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = prod(x.as_slice());
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(matrix)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = prod_complex(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(complex_vector)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = prod_complex(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(complex_row_vector)", near(&got, &want));
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
        let got = prod_complex(x.as_slice());
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(complex_matrix)", near(&got, &want));
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64];
        let got = prod_int(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(array[] int)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = prod(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(array[] real)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = prod_complex(&x);
        let want = flat(&x).into_iter().reduce(|a, b| a * b).unwrap();
        check(&mut failures, "prod(array[] complex)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = cumulative_sum(&x);
        let want = running_sums(flat(&x));
        check(&mut failures, "cumulative_sum(vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = cumulative_sum(&x);
        let want = running_sums(flat(&x));
        check(
            &mut failures,
            "cumulative_sum(row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = cumulative_sum_complex(&x);
        let want = running_sums(flat(&x));
        check(
            &mut failures,
            "cumulative_sum(complex_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = cumulative_sum_complex(&x);
        let want = running_sums(flat(&x));
        check(
            &mut failures,
            "cumulative_sum(complex_row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64];
        let got = cumulative_sum_int(&x);
        let want = running_sums(flat(&x));
        check(
            &mut failures,
            "cumulative_sum(array[] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = cumulative_sum(&x);
        let want = running_sums(flat(&x));
        check(
            &mut failures,
            "cumulative_sum(array[] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = cumulative_sum_complex(&x);
        let want = running_sums(flat(&x));
        check(
            &mut failures,
            "cumulative_sum(array[] complex)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_vector(&x);
        let want = flat(&x);
        check(&mut failures, "to_vector(vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_vector(&x);
        let want = flat(&x);
        check(&mut failures, "to_vector(row_vector)", near(&got, &want));
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = to_vector_matrix(&x);
        let want = flat(&x);
        check(&mut failures, "to_vector(matrix)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_vector(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_vector(complex_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_vector(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_vector(complex_row_vector)",
            near(&got, &want),
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
        let got = to_vector_matrix(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_vector(complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64];
        let got = to_vector_int::<f64>(&x);
        let want = flat(&x).into_iter().map(|v| v as f64).collect::<Vec<f64>>();
        check(&mut failures, "to_vector(array[] int)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_vector(&x);
        let want = flat(&x);
        check(&mut failures, "to_vector(array[] real)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_vector(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_vector(array[] complex)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_row_vector(&x);
        let want = flat(&x);
        check(&mut failures, "to_row_vector(vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_row_vector(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_row_vector(row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = to_row_vector_matrix(&x);
        let want = flat(&x);
        check(&mut failures, "to_row_vector(matrix)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_row_vector(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_row_vector(complex_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_row_vector(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_row_vector(complex_row_vector)",
            near(&got, &want),
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
        let got = to_row_vector_matrix(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_row_vector(complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64];
        let got = to_row_vector_int::<f64>(&x);
        let want = flat(&x).into_iter().map(|v| v as f64).collect::<Vec<f64>>();
        check(
            &mut failures,
            "to_row_vector(array[] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_row_vector(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_row_vector(array[] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_row_vector(&x);
        let want = flat(&x);
        check(
            &mut failures,
            "to_row_vector(array[] complex)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_array_1d_nested(&x);
        let want = vec![0.125, 0.25, 0.375];
        check(&mut failures, "to_array_1d(vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_array_1d_nested(&x);
        let want = vec![0.125, 0.25, 0.375];
        check(&mut failures, "to_array_1d(row_vector)", near(&got, &want));
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = to_array_1d(&x);
        let want = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        check(&mut failures, "to_array_1d(matrix)", near(&got, &want));
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        check(
            &mut failures,
            "to_array_1d(complex_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        check(
            &mut failures,
            "to_array_1d(complex_row_vector)",
            near(&got, &want),
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
        let got = to_array_1d(&x);
        let want = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        check(
            &mut failures,
            "to_array_1d(complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64];
        let got = to_array_1d_nested(&x);
        let want = vec![1_i64, 2_i64, 3_i64];
        check(&mut failures, "to_array_1d(array[] int)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_array_1d_nested(&x);
        let want = vec![0.125, 0.25, 0.375];
        check(
            &mut failures,
            "to_array_1d(array[] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]];
        let got = to_array_1d_nested(&x);
        let want = vec![1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64];
        check(
            &mut failures,
            "to_array_1d(array[,] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]];
        let got = to_array_1d_nested(&x);
        let want = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        check(
            &mut failures,
            "to_array_1d(array[,] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
            vec![vec![7_i64, 8_i64, 9_i64], vec![10_i64, 11_i64, 12_i64]],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64, 9_i64, 10_i64, 11_i64, 12_i64,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]],
            vec![vec![0.875, 1.0, 1.125], vec![1.25, 1.375, 1.5]],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<i64>>>> = vec![
            vec![
                vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                vec![vec![7_i64, 8_i64, 9_i64], vec![10_i64, 11_i64, 12_i64]],
            ],
            vec![
                vec![vec![13_i64, 14_i64, 15_i64], vec![16_i64, 17_i64, 18_i64]],
                vec![vec![19_i64, 20_i64, 21_i64], vec![22_i64, 23_i64, 24_i64]],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64, 9_i64, 10_i64, 11_i64, 12_i64,
            13_i64, 14_i64, 15_i64, 16_i64, 17_i64, 18_i64, 19_i64, 20_i64, 21_i64, 22_i64, 23_i64,
            24_i64,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<f64>>>> = vec![
            vec![
                vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]],
                vec![vec![0.875, 1.0, 1.125], vec![1.25, 1.375, 1.5]],
            ],
            vec![
                vec![vec![1.625, 1.75, 1.875], vec![2.0, 2.125, 2.25]],
                vec![vec![2.375, 2.5, 2.625], vec![2.75, 2.875, 3.0]],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75,
            1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<i64>>>>> = vec![
            vec![
                vec![
                    vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                    vec![vec![7_i64, 8_i64, 9_i64], vec![10_i64, 11_i64, 12_i64]],
                ],
                vec![
                    vec![vec![13_i64, 14_i64, 15_i64], vec![16_i64, 17_i64, 18_i64]],
                    vec![vec![19_i64, 20_i64, 21_i64], vec![22_i64, 23_i64, 24_i64]],
                ],
            ],
            vec![
                vec![
                    vec![vec![25_i64, 26_i64, 27_i64], vec![28_i64, 29_i64, 30_i64]],
                    vec![vec![31_i64, 32_i64, 33_i64], vec![34_i64, 35_i64, 36_i64]],
                ],
                vec![
                    vec![vec![37_i64, 38_i64, 39_i64], vec![40_i64, 41_i64, 42_i64]],
                    vec![vec![43_i64, 44_i64, 45_i64], vec![46_i64, 47_i64, 48_i64]],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64, 9_i64, 10_i64, 11_i64, 12_i64,
            13_i64, 14_i64, 15_i64, 16_i64, 17_i64, 18_i64, 19_i64, 20_i64, 21_i64, 22_i64, 23_i64,
            24_i64, 25_i64, 26_i64, 27_i64, 28_i64, 29_i64, 30_i64, 31_i64, 32_i64, 33_i64, 34_i64,
            35_i64, 36_i64, 37_i64, 38_i64, 39_i64, 40_i64, 41_i64, 42_i64, 43_i64, 44_i64, 45_i64,
            46_i64, 47_i64, 48_i64,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<f64>>>>> = vec![
            vec![
                vec![
                    vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]],
                    vec![vec![0.875, 1.0, 1.125], vec![1.25, 1.375, 1.5]],
                ],
                vec![
                    vec![vec![1.625, 1.75, 1.875], vec![2.0, 2.125, 2.25]],
                    vec![vec![2.375, 2.5, 2.625], vec![2.75, 2.875, 3.0]],
                ],
            ],
            vec![
                vec![
                    vec![vec![3.125, 3.25, 3.375], vec![3.5, 3.625, 3.75]],
                    vec![vec![3.875, 4.0, 4.125], vec![4.25, 4.375, 4.5]],
                ],
                vec![
                    vec![vec![4.625, 4.75, 4.875], vec![5.0, 5.125, 5.25]],
                    vec![vec![5.375, 5.5, 5.625], vec![5.75, 5.875, 6.0]],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75,
            1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5,
            3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25,
            5.375, 5.5, 5.625, 5.75, 5.875, 6.0,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<Vec<i64>>>>>> = vec![
            vec![
                vec![
                    vec![
                        vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                        vec![vec![7_i64, 8_i64, 9_i64], vec![10_i64, 11_i64, 12_i64]],
                    ],
                    vec![
                        vec![vec![13_i64, 14_i64, 15_i64], vec![16_i64, 17_i64, 18_i64]],
                        vec![vec![19_i64, 20_i64, 21_i64], vec![22_i64, 23_i64, 24_i64]],
                    ],
                ],
                vec![
                    vec![
                        vec![vec![25_i64, 26_i64, 27_i64], vec![28_i64, 29_i64, 30_i64]],
                        vec![vec![31_i64, 32_i64, 33_i64], vec![34_i64, 35_i64, 36_i64]],
                    ],
                    vec![
                        vec![vec![37_i64, 38_i64, 39_i64], vec![40_i64, 41_i64, 42_i64]],
                        vec![vec![43_i64, 44_i64, 45_i64], vec![46_i64, 47_i64, 48_i64]],
                    ],
                ],
            ],
            vec![
                vec![
                    vec![
                        vec![vec![49_i64, 50_i64, 51_i64], vec![52_i64, 53_i64, 54_i64]],
                        vec![vec![55_i64, 56_i64, 57_i64], vec![58_i64, 59_i64, 60_i64]],
                    ],
                    vec![
                        vec![vec![61_i64, 62_i64, 63_i64], vec![64_i64, 65_i64, 66_i64]],
                        vec![vec![67_i64, 68_i64, 69_i64], vec![70_i64, 71_i64, 72_i64]],
                    ],
                ],
                vec![
                    vec![
                        vec![vec![73_i64, 74_i64, 75_i64], vec![76_i64, 77_i64, 78_i64]],
                        vec![vec![79_i64, 80_i64, 81_i64], vec![82_i64, 83_i64, 84_i64]],
                    ],
                    vec![
                        vec![vec![85_i64, 86_i64, 87_i64], vec![88_i64, 89_i64, 90_i64]],
                        vec![vec![91_i64, 92_i64, 93_i64], vec![94_i64, 95_i64, 96_i64]],
                    ],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64, 9_i64, 10_i64, 11_i64, 12_i64,
            13_i64, 14_i64, 15_i64, 16_i64, 17_i64, 18_i64, 19_i64, 20_i64, 21_i64, 22_i64, 23_i64,
            24_i64, 25_i64, 26_i64, 27_i64, 28_i64, 29_i64, 30_i64, 31_i64, 32_i64, 33_i64, 34_i64,
            35_i64, 36_i64, 37_i64, 38_i64, 39_i64, 40_i64, 41_i64, 42_i64, 43_i64, 44_i64, 45_i64,
            46_i64, 47_i64, 48_i64, 49_i64, 50_i64, 51_i64, 52_i64, 53_i64, 54_i64, 55_i64, 56_i64,
            57_i64, 58_i64, 59_i64, 60_i64, 61_i64, 62_i64, 63_i64, 64_i64, 65_i64, 66_i64, 67_i64,
            68_i64, 69_i64, 70_i64, 71_i64, 72_i64, 73_i64, 74_i64, 75_i64, 76_i64, 77_i64, 78_i64,
            79_i64, 80_i64, 81_i64, 82_i64, 83_i64, 84_i64, 85_i64, 86_i64, 87_i64, 88_i64, 89_i64,
            90_i64, 91_i64, 92_i64, 93_i64, 94_i64, 95_i64, 96_i64,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,,] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>> = vec![
            vec![
                vec![
                    vec![
                        vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]],
                        vec![vec![0.875, 1.0, 1.125], vec![1.25, 1.375, 1.5]],
                    ],
                    vec![
                        vec![vec![1.625, 1.75, 1.875], vec![2.0, 2.125, 2.25]],
                        vec![vec![2.375, 2.5, 2.625], vec![2.75, 2.875, 3.0]],
                    ],
                ],
                vec![
                    vec![
                        vec![vec![3.125, 3.25, 3.375], vec![3.5, 3.625, 3.75]],
                        vec![vec![3.875, 4.0, 4.125], vec![4.25, 4.375, 4.5]],
                    ],
                    vec![
                        vec![vec![4.625, 4.75, 4.875], vec![5.0, 5.125, 5.25]],
                        vec![vec![5.375, 5.5, 5.625], vec![5.75, 5.875, 6.0]],
                    ],
                ],
            ],
            vec![
                vec![
                    vec![
                        vec![vec![6.125, 6.25, 6.375], vec![6.5, 6.625, 6.75]],
                        vec![vec![6.875, 7.0, 7.125], vec![7.25, 7.375, 7.5]],
                    ],
                    vec![
                        vec![vec![7.625, 7.75, 7.875], vec![8.0, 8.125, 8.25]],
                        vec![vec![8.375, 8.5, 8.625], vec![8.75, 8.875, 9.0]],
                    ],
                ],
                vec![
                    vec![
                        vec![vec![9.125, 9.25, 9.375], vec![9.5, 9.625, 9.75]],
                        vec![vec![9.875, 10.0, 10.125], vec![10.25, 10.375, 10.5]],
                    ],
                    vec![
                        vec![vec![10.625, 10.75, 10.875], vec![11.0, 11.125, 11.25]],
                        vec![vec![11.375, 11.5, 11.625], vec![11.75, 11.875, 12.0]],
                    ],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75,
            1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5,
            3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25,
            5.375, 5.5, 5.625, 5.75, 5.875, 6.0, 6.125, 6.25, 6.375, 6.5, 6.625, 6.75, 6.875, 7.0,
            7.125, 7.25, 7.375, 7.5, 7.625, 7.75, 7.875, 8.0, 8.125, 8.25, 8.375, 8.5, 8.625, 8.75,
            8.875, 9.0, 9.125, 9.25, 9.375, 9.5, 9.625, 9.75, 9.875, 10.0, 10.125, 10.25, 10.375,
            10.5, 10.625, 10.75, 10.875, 11.0, 11.125, 11.25, 11.375, 11.5, 11.625, 11.75, 11.875,
            12.0,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,,] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<Vec<Vec<i64>>>>>>> = vec![
            vec![
                vec![
                    vec![
                        vec![
                            vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                            vec![vec![7_i64, 8_i64, 9_i64], vec![10_i64, 11_i64, 12_i64]],
                        ],
                        vec![
                            vec![vec![13_i64, 14_i64, 15_i64], vec![16_i64, 17_i64, 18_i64]],
                            vec![vec![19_i64, 20_i64, 21_i64], vec![22_i64, 23_i64, 24_i64]],
                        ],
                    ],
                    vec![
                        vec![
                            vec![vec![25_i64, 26_i64, 27_i64], vec![28_i64, 29_i64, 30_i64]],
                            vec![vec![31_i64, 32_i64, 33_i64], vec![34_i64, 35_i64, 36_i64]],
                        ],
                        vec![
                            vec![vec![37_i64, 38_i64, 39_i64], vec![40_i64, 41_i64, 42_i64]],
                            vec![vec![43_i64, 44_i64, 45_i64], vec![46_i64, 47_i64, 48_i64]],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![vec![49_i64, 50_i64, 51_i64], vec![52_i64, 53_i64, 54_i64]],
                            vec![vec![55_i64, 56_i64, 57_i64], vec![58_i64, 59_i64, 60_i64]],
                        ],
                        vec![
                            vec![vec![61_i64, 62_i64, 63_i64], vec![64_i64, 65_i64, 66_i64]],
                            vec![vec![67_i64, 68_i64, 69_i64], vec![70_i64, 71_i64, 72_i64]],
                        ],
                    ],
                    vec![
                        vec![
                            vec![vec![73_i64, 74_i64, 75_i64], vec![76_i64, 77_i64, 78_i64]],
                            vec![vec![79_i64, 80_i64, 81_i64], vec![82_i64, 83_i64, 84_i64]],
                        ],
                        vec![
                            vec![vec![85_i64, 86_i64, 87_i64], vec![88_i64, 89_i64, 90_i64]],
                            vec![vec![91_i64, 92_i64, 93_i64], vec![94_i64, 95_i64, 96_i64]],
                        ],
                    ],
                ],
            ],
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![97_i64, 98_i64, 99_i64],
                                vec![100_i64, 101_i64, 102_i64],
                            ],
                            vec![
                                vec![103_i64, 104_i64, 105_i64],
                                vec![106_i64, 107_i64, 108_i64],
                            ],
                        ],
                        vec![
                            vec![
                                vec![109_i64, 110_i64, 111_i64],
                                vec![112_i64, 113_i64, 114_i64],
                            ],
                            vec![
                                vec![115_i64, 116_i64, 117_i64],
                                vec![118_i64, 119_i64, 120_i64],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![121_i64, 122_i64, 123_i64],
                                vec![124_i64, 125_i64, 126_i64],
                            ],
                            vec![
                                vec![127_i64, 128_i64, 129_i64],
                                vec![130_i64, 131_i64, 132_i64],
                            ],
                        ],
                        vec![
                            vec![
                                vec![133_i64, 134_i64, 135_i64],
                                vec![136_i64, 137_i64, 138_i64],
                            ],
                            vec![
                                vec![139_i64, 140_i64, 141_i64],
                                vec![142_i64, 143_i64, 144_i64],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![145_i64, 146_i64, 147_i64],
                                vec![148_i64, 149_i64, 150_i64],
                            ],
                            vec![
                                vec![151_i64, 152_i64, 153_i64],
                                vec![154_i64, 155_i64, 156_i64],
                            ],
                        ],
                        vec![
                            vec![
                                vec![157_i64, 158_i64, 159_i64],
                                vec![160_i64, 161_i64, 162_i64],
                            ],
                            vec![
                                vec![163_i64, 164_i64, 165_i64],
                                vec![166_i64, 167_i64, 168_i64],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![169_i64, 170_i64, 171_i64],
                                vec![172_i64, 173_i64, 174_i64],
                            ],
                            vec![
                                vec![175_i64, 176_i64, 177_i64],
                                vec![178_i64, 179_i64, 180_i64],
                            ],
                        ],
                        vec![
                            vec![
                                vec![181_i64, 182_i64, 183_i64],
                                vec![184_i64, 185_i64, 186_i64],
                            ],
                            vec![
                                vec![187_i64, 188_i64, 189_i64],
                                vec![190_i64, 191_i64, 192_i64],
                            ],
                        ],
                    ],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64, 9_i64, 10_i64, 11_i64, 12_i64,
            13_i64, 14_i64, 15_i64, 16_i64, 17_i64, 18_i64, 19_i64, 20_i64, 21_i64, 22_i64, 23_i64,
            24_i64, 25_i64, 26_i64, 27_i64, 28_i64, 29_i64, 30_i64, 31_i64, 32_i64, 33_i64, 34_i64,
            35_i64, 36_i64, 37_i64, 38_i64, 39_i64, 40_i64, 41_i64, 42_i64, 43_i64, 44_i64, 45_i64,
            46_i64, 47_i64, 48_i64, 49_i64, 50_i64, 51_i64, 52_i64, 53_i64, 54_i64, 55_i64, 56_i64,
            57_i64, 58_i64, 59_i64, 60_i64, 61_i64, 62_i64, 63_i64, 64_i64, 65_i64, 66_i64, 67_i64,
            68_i64, 69_i64, 70_i64, 71_i64, 72_i64, 73_i64, 74_i64, 75_i64, 76_i64, 77_i64, 78_i64,
            79_i64, 80_i64, 81_i64, 82_i64, 83_i64, 84_i64, 85_i64, 86_i64, 87_i64, 88_i64, 89_i64,
            90_i64, 91_i64, 92_i64, 93_i64, 94_i64, 95_i64, 96_i64, 97_i64, 98_i64, 99_i64,
            100_i64, 101_i64, 102_i64, 103_i64, 104_i64, 105_i64, 106_i64, 107_i64, 108_i64,
            109_i64, 110_i64, 111_i64, 112_i64, 113_i64, 114_i64, 115_i64, 116_i64, 117_i64,
            118_i64, 119_i64, 120_i64, 121_i64, 122_i64, 123_i64, 124_i64, 125_i64, 126_i64,
            127_i64, 128_i64, 129_i64, 130_i64, 131_i64, 132_i64, 133_i64, 134_i64, 135_i64,
            136_i64, 137_i64, 138_i64, 139_i64, 140_i64, 141_i64, 142_i64, 143_i64, 144_i64,
            145_i64, 146_i64, 147_i64, 148_i64, 149_i64, 150_i64, 151_i64, 152_i64, 153_i64,
            154_i64, 155_i64, 156_i64, 157_i64, 158_i64, 159_i64, 160_i64, 161_i64, 162_i64,
            163_i64, 164_i64, 165_i64, 166_i64, 167_i64, 168_i64, 169_i64, 170_i64, 171_i64,
            172_i64, 173_i64, 174_i64, 175_i64, 176_i64, 177_i64, 178_i64, 179_i64, 180_i64,
            181_i64, 182_i64, 183_i64, 184_i64, 185_i64, 186_i64, 187_i64, 188_i64, 189_i64,
            190_i64, 191_i64, 192_i64,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,,,] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>> = vec![
            vec![
                vec![
                    vec![
                        vec![
                            vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]],
                            vec![vec![0.875, 1.0, 1.125], vec![1.25, 1.375, 1.5]],
                        ],
                        vec![
                            vec![vec![1.625, 1.75, 1.875], vec![2.0, 2.125, 2.25]],
                            vec![vec![2.375, 2.5, 2.625], vec![2.75, 2.875, 3.0]],
                        ],
                    ],
                    vec![
                        vec![
                            vec![vec![3.125, 3.25, 3.375], vec![3.5, 3.625, 3.75]],
                            vec![vec![3.875, 4.0, 4.125], vec![4.25, 4.375, 4.5]],
                        ],
                        vec![
                            vec![vec![4.625, 4.75, 4.875], vec![5.0, 5.125, 5.25]],
                            vec![vec![5.375, 5.5, 5.625], vec![5.75, 5.875, 6.0]],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![vec![6.125, 6.25, 6.375], vec![6.5, 6.625, 6.75]],
                            vec![vec![6.875, 7.0, 7.125], vec![7.25, 7.375, 7.5]],
                        ],
                        vec![
                            vec![vec![7.625, 7.75, 7.875], vec![8.0, 8.125, 8.25]],
                            vec![vec![8.375, 8.5, 8.625], vec![8.75, 8.875, 9.0]],
                        ],
                    ],
                    vec![
                        vec![
                            vec![vec![9.125, 9.25, 9.375], vec![9.5, 9.625, 9.75]],
                            vec![vec![9.875, 10.0, 10.125], vec![10.25, 10.375, 10.5]],
                        ],
                        vec![
                            vec![vec![10.625, 10.75, 10.875], vec![11.0, 11.125, 11.25]],
                            vec![vec![11.375, 11.5, 11.625], vec![11.75, 11.875, 12.0]],
                        ],
                    ],
                ],
            ],
            vec![
                vec![
                    vec![
                        vec![
                            vec![vec![12.125, 12.25, 12.375], vec![12.5, 12.625, 12.75]],
                            vec![vec![12.875, 13.0, 13.125], vec![13.25, 13.375, 13.5]],
                        ],
                        vec![
                            vec![vec![13.625, 13.75, 13.875], vec![14.0, 14.125, 14.25]],
                            vec![vec![14.375, 14.5, 14.625], vec![14.75, 14.875, 15.0]],
                        ],
                    ],
                    vec![
                        vec![
                            vec![vec![15.125, 15.25, 15.375], vec![15.5, 15.625, 15.75]],
                            vec![vec![15.875, 16.0, 16.125], vec![16.25, 16.375, 16.5]],
                        ],
                        vec![
                            vec![vec![16.625, 16.75, 16.875], vec![17.0, 17.125, 17.25]],
                            vec![vec![17.375, 17.5, 17.625], vec![17.75, 17.875, 18.0]],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![vec![18.125, 18.25, 18.375], vec![18.5, 18.625, 18.75]],
                            vec![vec![18.875, 19.0, 19.125], vec![19.25, 19.375, 19.5]],
                        ],
                        vec![
                            vec![vec![19.625, 19.75, 19.875], vec![20.0, 20.125, 20.25]],
                            vec![vec![20.375, 20.5, 20.625], vec![20.75, 20.875, 21.0]],
                        ],
                    ],
                    vec![
                        vec![
                            vec![vec![21.125, 21.25, 21.375], vec![21.5, 21.625, 21.75]],
                            vec![vec![21.875, 22.0, 22.125], vec![22.25, 22.375, 22.5]],
                        ],
                        vec![
                            vec![vec![22.625, 22.75, 22.875], vec![23.0, 23.125, 23.25]],
                            vec![vec![23.375, 23.5, 23.625], vec![23.75, 23.875, 24.0]],
                        ],
                    ],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75,
            1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5,
            3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25,
            5.375, 5.5, 5.625, 5.75, 5.875, 6.0, 6.125, 6.25, 6.375, 6.5, 6.625, 6.75, 6.875, 7.0,
            7.125, 7.25, 7.375, 7.5, 7.625, 7.75, 7.875, 8.0, 8.125, 8.25, 8.375, 8.5, 8.625, 8.75,
            8.875, 9.0, 9.125, 9.25, 9.375, 9.5, 9.625, 9.75, 9.875, 10.0, 10.125, 10.25, 10.375,
            10.5, 10.625, 10.75, 10.875, 11.0, 11.125, 11.25, 11.375, 11.5, 11.625, 11.75, 11.875,
            12.0, 12.125, 12.25, 12.375, 12.5, 12.625, 12.75, 12.875, 13.0, 13.125, 13.25, 13.375,
            13.5, 13.625, 13.75, 13.875, 14.0, 14.125, 14.25, 14.375, 14.5, 14.625, 14.75, 14.875,
            15.0, 15.125, 15.25, 15.375, 15.5, 15.625, 15.75, 15.875, 16.0, 16.125, 16.25, 16.375,
            16.5, 16.625, 16.75, 16.875, 17.0, 17.125, 17.25, 17.375, 17.5, 17.625, 17.75, 17.875,
            18.0, 18.125, 18.25, 18.375, 18.5, 18.625, 18.75, 18.875, 19.0, 19.125, 19.25, 19.375,
            19.5, 19.625, 19.75, 19.875, 20.0, 20.125, 20.25, 20.375, 20.5, 20.625, 20.75, 20.875,
            21.0, 21.125, 21.25, 21.375, 21.5, 21.625, 21.75, 21.875, 22.0, 22.125, 22.25, 22.375,
            22.5, 22.625, 22.75, 22.875, 23.0, 23.125, 23.25, 23.375, 23.5, 23.625, 23.75, 23.875,
            24.0,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,,,] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<i64>>>>>>>> = vec![
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                                vec![vec![7_i64, 8_i64, 9_i64], vec![10_i64, 11_i64, 12_i64]],
                            ],
                            vec![
                                vec![vec![13_i64, 14_i64, 15_i64], vec![16_i64, 17_i64, 18_i64]],
                                vec![vec![19_i64, 20_i64, 21_i64], vec![22_i64, 23_i64, 24_i64]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![25_i64, 26_i64, 27_i64], vec![28_i64, 29_i64, 30_i64]],
                                vec![vec![31_i64, 32_i64, 33_i64], vec![34_i64, 35_i64, 36_i64]],
                            ],
                            vec![
                                vec![vec![37_i64, 38_i64, 39_i64], vec![40_i64, 41_i64, 42_i64]],
                                vec![vec![43_i64, 44_i64, 45_i64], vec![46_i64, 47_i64, 48_i64]],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![vec![49_i64, 50_i64, 51_i64], vec![52_i64, 53_i64, 54_i64]],
                                vec![vec![55_i64, 56_i64, 57_i64], vec![58_i64, 59_i64, 60_i64]],
                            ],
                            vec![
                                vec![vec![61_i64, 62_i64, 63_i64], vec![64_i64, 65_i64, 66_i64]],
                                vec![vec![67_i64, 68_i64, 69_i64], vec![70_i64, 71_i64, 72_i64]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![73_i64, 74_i64, 75_i64], vec![76_i64, 77_i64, 78_i64]],
                                vec![vec![79_i64, 80_i64, 81_i64], vec![82_i64, 83_i64, 84_i64]],
                            ],
                            vec![
                                vec![vec![85_i64, 86_i64, 87_i64], vec![88_i64, 89_i64, 90_i64]],
                                vec![vec![91_i64, 92_i64, 93_i64], vec![94_i64, 95_i64, 96_i64]],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![97_i64, 98_i64, 99_i64],
                                    vec![100_i64, 101_i64, 102_i64],
                                ],
                                vec![
                                    vec![103_i64, 104_i64, 105_i64],
                                    vec![106_i64, 107_i64, 108_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![109_i64, 110_i64, 111_i64],
                                    vec![112_i64, 113_i64, 114_i64],
                                ],
                                vec![
                                    vec![115_i64, 116_i64, 117_i64],
                                    vec![118_i64, 119_i64, 120_i64],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![121_i64, 122_i64, 123_i64],
                                    vec![124_i64, 125_i64, 126_i64],
                                ],
                                vec![
                                    vec![127_i64, 128_i64, 129_i64],
                                    vec![130_i64, 131_i64, 132_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![133_i64, 134_i64, 135_i64],
                                    vec![136_i64, 137_i64, 138_i64],
                                ],
                                vec![
                                    vec![139_i64, 140_i64, 141_i64],
                                    vec![142_i64, 143_i64, 144_i64],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![145_i64, 146_i64, 147_i64],
                                    vec![148_i64, 149_i64, 150_i64],
                                ],
                                vec![
                                    vec![151_i64, 152_i64, 153_i64],
                                    vec![154_i64, 155_i64, 156_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![157_i64, 158_i64, 159_i64],
                                    vec![160_i64, 161_i64, 162_i64],
                                ],
                                vec![
                                    vec![163_i64, 164_i64, 165_i64],
                                    vec![166_i64, 167_i64, 168_i64],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![169_i64, 170_i64, 171_i64],
                                    vec![172_i64, 173_i64, 174_i64],
                                ],
                                vec![
                                    vec![175_i64, 176_i64, 177_i64],
                                    vec![178_i64, 179_i64, 180_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![181_i64, 182_i64, 183_i64],
                                    vec![184_i64, 185_i64, 186_i64],
                                ],
                                vec![
                                    vec![187_i64, 188_i64, 189_i64],
                                    vec![190_i64, 191_i64, 192_i64],
                                ],
                            ],
                        ],
                    ],
                ],
            ],
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![193_i64, 194_i64, 195_i64],
                                    vec![196_i64, 197_i64, 198_i64],
                                ],
                                vec![
                                    vec![199_i64, 200_i64, 201_i64],
                                    vec![202_i64, 203_i64, 204_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![205_i64, 206_i64, 207_i64],
                                    vec![208_i64, 209_i64, 210_i64],
                                ],
                                vec![
                                    vec![211_i64, 212_i64, 213_i64],
                                    vec![214_i64, 215_i64, 216_i64],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![217_i64, 218_i64, 219_i64],
                                    vec![220_i64, 221_i64, 222_i64],
                                ],
                                vec![
                                    vec![223_i64, 224_i64, 225_i64],
                                    vec![226_i64, 227_i64, 228_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![229_i64, 230_i64, 231_i64],
                                    vec![232_i64, 233_i64, 234_i64],
                                ],
                                vec![
                                    vec![235_i64, 236_i64, 237_i64],
                                    vec![238_i64, 239_i64, 240_i64],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![241_i64, 242_i64, 243_i64],
                                    vec![244_i64, 245_i64, 246_i64],
                                ],
                                vec![
                                    vec![247_i64, 248_i64, 249_i64],
                                    vec![250_i64, 251_i64, 252_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![253_i64, 254_i64, 255_i64],
                                    vec![256_i64, 257_i64, 258_i64],
                                ],
                                vec![
                                    vec![259_i64, 260_i64, 261_i64],
                                    vec![262_i64, 263_i64, 264_i64],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![265_i64, 266_i64, 267_i64],
                                    vec![268_i64, 269_i64, 270_i64],
                                ],
                                vec![
                                    vec![271_i64, 272_i64, 273_i64],
                                    vec![274_i64, 275_i64, 276_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![277_i64, 278_i64, 279_i64],
                                    vec![280_i64, 281_i64, 282_i64],
                                ],
                                vec![
                                    vec![283_i64, 284_i64, 285_i64],
                                    vec![286_i64, 287_i64, 288_i64],
                                ],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![289_i64, 290_i64, 291_i64],
                                    vec![292_i64, 293_i64, 294_i64],
                                ],
                                vec![
                                    vec![295_i64, 296_i64, 297_i64],
                                    vec![298_i64, 299_i64, 300_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![301_i64, 302_i64, 303_i64],
                                    vec![304_i64, 305_i64, 306_i64],
                                ],
                                vec![
                                    vec![307_i64, 308_i64, 309_i64],
                                    vec![310_i64, 311_i64, 312_i64],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![313_i64, 314_i64, 315_i64],
                                    vec![316_i64, 317_i64, 318_i64],
                                ],
                                vec![
                                    vec![319_i64, 320_i64, 321_i64],
                                    vec![322_i64, 323_i64, 324_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![325_i64, 326_i64, 327_i64],
                                    vec![328_i64, 329_i64, 330_i64],
                                ],
                                vec![
                                    vec![331_i64, 332_i64, 333_i64],
                                    vec![334_i64, 335_i64, 336_i64],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![337_i64, 338_i64, 339_i64],
                                    vec![340_i64, 341_i64, 342_i64],
                                ],
                                vec![
                                    vec![343_i64, 344_i64, 345_i64],
                                    vec![346_i64, 347_i64, 348_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![349_i64, 350_i64, 351_i64],
                                    vec![352_i64, 353_i64, 354_i64],
                                ],
                                vec![
                                    vec![355_i64, 356_i64, 357_i64],
                                    vec![358_i64, 359_i64, 360_i64],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![361_i64, 362_i64, 363_i64],
                                    vec![364_i64, 365_i64, 366_i64],
                                ],
                                vec![
                                    vec![367_i64, 368_i64, 369_i64],
                                    vec![370_i64, 371_i64, 372_i64],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![373_i64, 374_i64, 375_i64],
                                    vec![376_i64, 377_i64, 378_i64],
                                ],
                                vec![
                                    vec![379_i64, 380_i64, 381_i64],
                                    vec![382_i64, 383_i64, 384_i64],
                                ],
                            ],
                        ],
                    ],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64, 9_i64, 10_i64, 11_i64, 12_i64,
            13_i64, 14_i64, 15_i64, 16_i64, 17_i64, 18_i64, 19_i64, 20_i64, 21_i64, 22_i64, 23_i64,
            24_i64, 25_i64, 26_i64, 27_i64, 28_i64, 29_i64, 30_i64, 31_i64, 32_i64, 33_i64, 34_i64,
            35_i64, 36_i64, 37_i64, 38_i64, 39_i64, 40_i64, 41_i64, 42_i64, 43_i64, 44_i64, 45_i64,
            46_i64, 47_i64, 48_i64, 49_i64, 50_i64, 51_i64, 52_i64, 53_i64, 54_i64, 55_i64, 56_i64,
            57_i64, 58_i64, 59_i64, 60_i64, 61_i64, 62_i64, 63_i64, 64_i64, 65_i64, 66_i64, 67_i64,
            68_i64, 69_i64, 70_i64, 71_i64, 72_i64, 73_i64, 74_i64, 75_i64, 76_i64, 77_i64, 78_i64,
            79_i64, 80_i64, 81_i64, 82_i64, 83_i64, 84_i64, 85_i64, 86_i64, 87_i64, 88_i64, 89_i64,
            90_i64, 91_i64, 92_i64, 93_i64, 94_i64, 95_i64, 96_i64, 97_i64, 98_i64, 99_i64,
            100_i64, 101_i64, 102_i64, 103_i64, 104_i64, 105_i64, 106_i64, 107_i64, 108_i64,
            109_i64, 110_i64, 111_i64, 112_i64, 113_i64, 114_i64, 115_i64, 116_i64, 117_i64,
            118_i64, 119_i64, 120_i64, 121_i64, 122_i64, 123_i64, 124_i64, 125_i64, 126_i64,
            127_i64, 128_i64, 129_i64, 130_i64, 131_i64, 132_i64, 133_i64, 134_i64, 135_i64,
            136_i64, 137_i64, 138_i64, 139_i64, 140_i64, 141_i64, 142_i64, 143_i64, 144_i64,
            145_i64, 146_i64, 147_i64, 148_i64, 149_i64, 150_i64, 151_i64, 152_i64, 153_i64,
            154_i64, 155_i64, 156_i64, 157_i64, 158_i64, 159_i64, 160_i64, 161_i64, 162_i64,
            163_i64, 164_i64, 165_i64, 166_i64, 167_i64, 168_i64, 169_i64, 170_i64, 171_i64,
            172_i64, 173_i64, 174_i64, 175_i64, 176_i64, 177_i64, 178_i64, 179_i64, 180_i64,
            181_i64, 182_i64, 183_i64, 184_i64, 185_i64, 186_i64, 187_i64, 188_i64, 189_i64,
            190_i64, 191_i64, 192_i64, 193_i64, 194_i64, 195_i64, 196_i64, 197_i64, 198_i64,
            199_i64, 200_i64, 201_i64, 202_i64, 203_i64, 204_i64, 205_i64, 206_i64, 207_i64,
            208_i64, 209_i64, 210_i64, 211_i64, 212_i64, 213_i64, 214_i64, 215_i64, 216_i64,
            217_i64, 218_i64, 219_i64, 220_i64, 221_i64, 222_i64, 223_i64, 224_i64, 225_i64,
            226_i64, 227_i64, 228_i64, 229_i64, 230_i64, 231_i64, 232_i64, 233_i64, 234_i64,
            235_i64, 236_i64, 237_i64, 238_i64, 239_i64, 240_i64, 241_i64, 242_i64, 243_i64,
            244_i64, 245_i64, 246_i64, 247_i64, 248_i64, 249_i64, 250_i64, 251_i64, 252_i64,
            253_i64, 254_i64, 255_i64, 256_i64, 257_i64, 258_i64, 259_i64, 260_i64, 261_i64,
            262_i64, 263_i64, 264_i64, 265_i64, 266_i64, 267_i64, 268_i64, 269_i64, 270_i64,
            271_i64, 272_i64, 273_i64, 274_i64, 275_i64, 276_i64, 277_i64, 278_i64, 279_i64,
            280_i64, 281_i64, 282_i64, 283_i64, 284_i64, 285_i64, 286_i64, 287_i64, 288_i64,
            289_i64, 290_i64, 291_i64, 292_i64, 293_i64, 294_i64, 295_i64, 296_i64, 297_i64,
            298_i64, 299_i64, 300_i64, 301_i64, 302_i64, 303_i64, 304_i64, 305_i64, 306_i64,
            307_i64, 308_i64, 309_i64, 310_i64, 311_i64, 312_i64, 313_i64, 314_i64, 315_i64,
            316_i64, 317_i64, 318_i64, 319_i64, 320_i64, 321_i64, 322_i64, 323_i64, 324_i64,
            325_i64, 326_i64, 327_i64, 328_i64, 329_i64, 330_i64, 331_i64, 332_i64, 333_i64,
            334_i64, 335_i64, 336_i64, 337_i64, 338_i64, 339_i64, 340_i64, 341_i64, 342_i64,
            343_i64, 344_i64, 345_i64, 346_i64, 347_i64, 348_i64, 349_i64, 350_i64, 351_i64,
            352_i64, 353_i64, 354_i64, 355_i64, 356_i64, 357_i64, 358_i64, 359_i64, 360_i64,
            361_i64, 362_i64, 363_i64, 364_i64, 365_i64, 366_i64, 367_i64, 368_i64, 369_i64,
            370_i64, 371_i64, 372_i64, 373_i64, 374_i64, 375_i64, 376_i64, 377_i64, 378_i64,
            379_i64, 380_i64, 381_i64, 382_i64, 383_i64, 384_i64,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,,,,] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>> = vec![
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]],
                                vec![vec![0.875, 1.0, 1.125], vec![1.25, 1.375, 1.5]],
                            ],
                            vec![
                                vec![vec![1.625, 1.75, 1.875], vec![2.0, 2.125, 2.25]],
                                vec![vec![2.375, 2.5, 2.625], vec![2.75, 2.875, 3.0]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![3.125, 3.25, 3.375], vec![3.5, 3.625, 3.75]],
                                vec![vec![3.875, 4.0, 4.125], vec![4.25, 4.375, 4.5]],
                            ],
                            vec![
                                vec![vec![4.625, 4.75, 4.875], vec![5.0, 5.125, 5.25]],
                                vec![vec![5.375, 5.5, 5.625], vec![5.75, 5.875, 6.0]],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![vec![6.125, 6.25, 6.375], vec![6.5, 6.625, 6.75]],
                                vec![vec![6.875, 7.0, 7.125], vec![7.25, 7.375, 7.5]],
                            ],
                            vec![
                                vec![vec![7.625, 7.75, 7.875], vec![8.0, 8.125, 8.25]],
                                vec![vec![8.375, 8.5, 8.625], vec![8.75, 8.875, 9.0]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![9.125, 9.25, 9.375], vec![9.5, 9.625, 9.75]],
                                vec![vec![9.875, 10.0, 10.125], vec![10.25, 10.375, 10.5]],
                            ],
                            vec![
                                vec![vec![10.625, 10.75, 10.875], vec![11.0, 11.125, 11.25]],
                                vec![vec![11.375, 11.5, 11.625], vec![11.75, 11.875, 12.0]],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![vec![12.125, 12.25, 12.375], vec![12.5, 12.625, 12.75]],
                                vec![vec![12.875, 13.0, 13.125], vec![13.25, 13.375, 13.5]],
                            ],
                            vec![
                                vec![vec![13.625, 13.75, 13.875], vec![14.0, 14.125, 14.25]],
                                vec![vec![14.375, 14.5, 14.625], vec![14.75, 14.875, 15.0]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![15.125, 15.25, 15.375], vec![15.5, 15.625, 15.75]],
                                vec![vec![15.875, 16.0, 16.125], vec![16.25, 16.375, 16.5]],
                            ],
                            vec![
                                vec![vec![16.625, 16.75, 16.875], vec![17.0, 17.125, 17.25]],
                                vec![vec![17.375, 17.5, 17.625], vec![17.75, 17.875, 18.0]],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![vec![18.125, 18.25, 18.375], vec![18.5, 18.625, 18.75]],
                                vec![vec![18.875, 19.0, 19.125], vec![19.25, 19.375, 19.5]],
                            ],
                            vec![
                                vec![vec![19.625, 19.75, 19.875], vec![20.0, 20.125, 20.25]],
                                vec![vec![20.375, 20.5, 20.625], vec![20.75, 20.875, 21.0]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![21.125, 21.25, 21.375], vec![21.5, 21.625, 21.75]],
                                vec![vec![21.875, 22.0, 22.125], vec![22.25, 22.375, 22.5]],
                            ],
                            vec![
                                vec![vec![22.625, 22.75, 22.875], vec![23.0, 23.125, 23.25]],
                                vec![vec![23.375, 23.5, 23.625], vec![23.75, 23.875, 24.0]],
                            ],
                        ],
                    ],
                ],
            ],
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![vec![24.125, 24.25, 24.375], vec![24.5, 24.625, 24.75]],
                                vec![vec![24.875, 25.0, 25.125], vec![25.25, 25.375, 25.5]],
                            ],
                            vec![
                                vec![vec![25.625, 25.75, 25.875], vec![26.0, 26.125, 26.25]],
                                vec![vec![26.375, 26.5, 26.625], vec![26.75, 26.875, 27.0]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![27.125, 27.25, 27.375], vec![27.5, 27.625, 27.75]],
                                vec![vec![27.875, 28.0, 28.125], vec![28.25, 28.375, 28.5]],
                            ],
                            vec![
                                vec![vec![28.625, 28.75, 28.875], vec![29.0, 29.125, 29.25]],
                                vec![vec![29.375, 29.5, 29.625], vec![29.75, 29.875, 30.0]],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![vec![30.125, 30.25, 30.375], vec![30.5, 30.625, 30.75]],
                                vec![vec![30.875, 31.0, 31.125], vec![31.25, 31.375, 31.5]],
                            ],
                            vec![
                                vec![vec![31.625, 31.75, 31.875], vec![32.0, 32.125, 32.25]],
                                vec![vec![32.375, 32.5, 32.625], vec![32.75, 32.875, 33.0]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![33.125, 33.25, 33.375], vec![33.5, 33.625, 33.75]],
                                vec![vec![33.875, 34.0, 34.125], vec![34.25, 34.375, 34.5]],
                            ],
                            vec![
                                vec![vec![34.625, 34.75, 34.875], vec![35.0, 35.125, 35.25]],
                                vec![vec![35.375, 35.5, 35.625], vec![35.75, 35.875, 36.0]],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![vec![36.125, 36.25, 36.375], vec![36.5, 36.625, 36.75]],
                                vec![vec![36.875, 37.0, 37.125], vec![37.25, 37.375, 37.5]],
                            ],
                            vec![
                                vec![vec![37.625, 37.75, 37.875], vec![38.0, 38.125, 38.25]],
                                vec![vec![38.375, 38.5, 38.625], vec![38.75, 38.875, 39.0]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![39.125, 39.25, 39.375], vec![39.5, 39.625, 39.75]],
                                vec![vec![39.875, 40.0, 40.125], vec![40.25, 40.375, 40.5]],
                            ],
                            vec![
                                vec![vec![40.625, 40.75, 40.875], vec![41.0, 41.125, 41.25]],
                                vec![vec![41.375, 41.5, 41.625], vec![41.75, 41.875, 42.0]],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![vec![42.125, 42.25, 42.375], vec![42.5, 42.625, 42.75]],
                                vec![vec![42.875, 43.0, 43.125], vec![43.25, 43.375, 43.5]],
                            ],
                            vec![
                                vec![vec![43.625, 43.75, 43.875], vec![44.0, 44.125, 44.25]],
                                vec![vec![44.375, 44.5, 44.625], vec![44.75, 44.875, 45.0]],
                            ],
                        ],
                        vec![
                            vec![
                                vec![vec![45.125, 45.25, 45.375], vec![45.5, 45.625, 45.75]],
                                vec![vec![45.875, 46.0, 46.125], vec![46.25, 46.375, 46.5]],
                            ],
                            vec![
                                vec![vec![46.625, 46.75, 46.875], vec![47.0, 47.125, 47.25]],
                                vec![vec![47.375, 47.5, 47.625], vec![47.75, 47.875, 48.0]],
                            ],
                        ],
                    ],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75,
            1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5,
            3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25,
            5.375, 5.5, 5.625, 5.75, 5.875, 6.0, 6.125, 6.25, 6.375, 6.5, 6.625, 6.75, 6.875, 7.0,
            7.125, 7.25, 7.375, 7.5, 7.625, 7.75, 7.875, 8.0, 8.125, 8.25, 8.375, 8.5, 8.625, 8.75,
            8.875, 9.0, 9.125, 9.25, 9.375, 9.5, 9.625, 9.75, 9.875, 10.0, 10.125, 10.25, 10.375,
            10.5, 10.625, 10.75, 10.875, 11.0, 11.125, 11.25, 11.375, 11.5, 11.625, 11.75, 11.875,
            12.0, 12.125, 12.25, 12.375, 12.5, 12.625, 12.75, 12.875, 13.0, 13.125, 13.25, 13.375,
            13.5, 13.625, 13.75, 13.875, 14.0, 14.125, 14.25, 14.375, 14.5, 14.625, 14.75, 14.875,
            15.0, 15.125, 15.25, 15.375, 15.5, 15.625, 15.75, 15.875, 16.0, 16.125, 16.25, 16.375,
            16.5, 16.625, 16.75, 16.875, 17.0, 17.125, 17.25, 17.375, 17.5, 17.625, 17.75, 17.875,
            18.0, 18.125, 18.25, 18.375, 18.5, 18.625, 18.75, 18.875, 19.0, 19.125, 19.25, 19.375,
            19.5, 19.625, 19.75, 19.875, 20.0, 20.125, 20.25, 20.375, 20.5, 20.625, 20.75, 20.875,
            21.0, 21.125, 21.25, 21.375, 21.5, 21.625, 21.75, 21.875, 22.0, 22.125, 22.25, 22.375,
            22.5, 22.625, 22.75, 22.875, 23.0, 23.125, 23.25, 23.375, 23.5, 23.625, 23.75, 23.875,
            24.0, 24.125, 24.25, 24.375, 24.5, 24.625, 24.75, 24.875, 25.0, 25.125, 25.25, 25.375,
            25.5, 25.625, 25.75, 25.875, 26.0, 26.125, 26.25, 26.375, 26.5, 26.625, 26.75, 26.875,
            27.0, 27.125, 27.25, 27.375, 27.5, 27.625, 27.75, 27.875, 28.0, 28.125, 28.25, 28.375,
            28.5, 28.625, 28.75, 28.875, 29.0, 29.125, 29.25, 29.375, 29.5, 29.625, 29.75, 29.875,
            30.0, 30.125, 30.25, 30.375, 30.5, 30.625, 30.75, 30.875, 31.0, 31.125, 31.25, 31.375,
            31.5, 31.625, 31.75, 31.875, 32.0, 32.125, 32.25, 32.375, 32.5, 32.625, 32.75, 32.875,
            33.0, 33.125, 33.25, 33.375, 33.5, 33.625, 33.75, 33.875, 34.0, 34.125, 34.25, 34.375,
            34.5, 34.625, 34.75, 34.875, 35.0, 35.125, 35.25, 35.375, 35.5, 35.625, 35.75, 35.875,
            36.0, 36.125, 36.25, 36.375, 36.5, 36.625, 36.75, 36.875, 37.0, 37.125, 37.25, 37.375,
            37.5, 37.625, 37.75, 37.875, 38.0, 38.125, 38.25, 38.375, 38.5, 38.625, 38.75, 38.875,
            39.0, 39.125, 39.25, 39.375, 39.5, 39.625, 39.75, 39.875, 40.0, 40.125, 40.25, 40.375,
            40.5, 40.625, 40.75, 40.875, 41.0, 41.125, 41.25, 41.375, 41.5, 41.625, 41.75, 41.875,
            42.0, 42.125, 42.25, 42.375, 42.5, 42.625, 42.75, 42.875, 43.0, 43.125, 43.25, 43.375,
            43.5, 43.625, 43.75, 43.875, 44.0, 44.125, 44.25, 44.375, 44.5, 44.625, 44.75, 44.875,
            45.0, 45.125, 45.25, 45.375, 45.5, 45.625, 45.75, 45.875, 46.0, 46.125, 46.25, 46.375,
            46.5, 46.625, 46.75, 46.875, 47.0, 47.125, 47.25, 47.375, 47.5, 47.625, 47.75, 47.875,
            48.0,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,,,,] real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<i64>>>>>>>>> = vec![
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]],
                                    vec![vec![7_i64, 8_i64, 9_i64], vec![10_i64, 11_i64, 12_i64]],
                                ],
                                vec![
                                    vec![
                                        vec![13_i64, 14_i64, 15_i64],
                                        vec![16_i64, 17_i64, 18_i64],
                                    ],
                                    vec![
                                        vec![19_i64, 20_i64, 21_i64],
                                        vec![22_i64, 23_i64, 24_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![25_i64, 26_i64, 27_i64],
                                        vec![28_i64, 29_i64, 30_i64],
                                    ],
                                    vec![
                                        vec![31_i64, 32_i64, 33_i64],
                                        vec![34_i64, 35_i64, 36_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![37_i64, 38_i64, 39_i64],
                                        vec![40_i64, 41_i64, 42_i64],
                                    ],
                                    vec![
                                        vec![43_i64, 44_i64, 45_i64],
                                        vec![46_i64, 47_i64, 48_i64],
                                    ],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![49_i64, 50_i64, 51_i64],
                                        vec![52_i64, 53_i64, 54_i64],
                                    ],
                                    vec![
                                        vec![55_i64, 56_i64, 57_i64],
                                        vec![58_i64, 59_i64, 60_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![61_i64, 62_i64, 63_i64],
                                        vec![64_i64, 65_i64, 66_i64],
                                    ],
                                    vec![
                                        vec![67_i64, 68_i64, 69_i64],
                                        vec![70_i64, 71_i64, 72_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![73_i64, 74_i64, 75_i64],
                                        vec![76_i64, 77_i64, 78_i64],
                                    ],
                                    vec![
                                        vec![79_i64, 80_i64, 81_i64],
                                        vec![82_i64, 83_i64, 84_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![85_i64, 86_i64, 87_i64],
                                        vec![88_i64, 89_i64, 90_i64],
                                    ],
                                    vec![
                                        vec![91_i64, 92_i64, 93_i64],
                                        vec![94_i64, 95_i64, 96_i64],
                                    ],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![97_i64, 98_i64, 99_i64],
                                        vec![100_i64, 101_i64, 102_i64],
                                    ],
                                    vec![
                                        vec![103_i64, 104_i64, 105_i64],
                                        vec![106_i64, 107_i64, 108_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![109_i64, 110_i64, 111_i64],
                                        vec![112_i64, 113_i64, 114_i64],
                                    ],
                                    vec![
                                        vec![115_i64, 116_i64, 117_i64],
                                        vec![118_i64, 119_i64, 120_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![121_i64, 122_i64, 123_i64],
                                        vec![124_i64, 125_i64, 126_i64],
                                    ],
                                    vec![
                                        vec![127_i64, 128_i64, 129_i64],
                                        vec![130_i64, 131_i64, 132_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![133_i64, 134_i64, 135_i64],
                                        vec![136_i64, 137_i64, 138_i64],
                                    ],
                                    vec![
                                        vec![139_i64, 140_i64, 141_i64],
                                        vec![142_i64, 143_i64, 144_i64],
                                    ],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![145_i64, 146_i64, 147_i64],
                                        vec![148_i64, 149_i64, 150_i64],
                                    ],
                                    vec![
                                        vec![151_i64, 152_i64, 153_i64],
                                        vec![154_i64, 155_i64, 156_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![157_i64, 158_i64, 159_i64],
                                        vec![160_i64, 161_i64, 162_i64],
                                    ],
                                    vec![
                                        vec![163_i64, 164_i64, 165_i64],
                                        vec![166_i64, 167_i64, 168_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![169_i64, 170_i64, 171_i64],
                                        vec![172_i64, 173_i64, 174_i64],
                                    ],
                                    vec![
                                        vec![175_i64, 176_i64, 177_i64],
                                        vec![178_i64, 179_i64, 180_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![181_i64, 182_i64, 183_i64],
                                        vec![184_i64, 185_i64, 186_i64],
                                    ],
                                    vec![
                                        vec![187_i64, 188_i64, 189_i64],
                                        vec![190_i64, 191_i64, 192_i64],
                                    ],
                                ],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![193_i64, 194_i64, 195_i64],
                                        vec![196_i64, 197_i64, 198_i64],
                                    ],
                                    vec![
                                        vec![199_i64, 200_i64, 201_i64],
                                        vec![202_i64, 203_i64, 204_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![205_i64, 206_i64, 207_i64],
                                        vec![208_i64, 209_i64, 210_i64],
                                    ],
                                    vec![
                                        vec![211_i64, 212_i64, 213_i64],
                                        vec![214_i64, 215_i64, 216_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![217_i64, 218_i64, 219_i64],
                                        vec![220_i64, 221_i64, 222_i64],
                                    ],
                                    vec![
                                        vec![223_i64, 224_i64, 225_i64],
                                        vec![226_i64, 227_i64, 228_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![229_i64, 230_i64, 231_i64],
                                        vec![232_i64, 233_i64, 234_i64],
                                    ],
                                    vec![
                                        vec![235_i64, 236_i64, 237_i64],
                                        vec![238_i64, 239_i64, 240_i64],
                                    ],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![241_i64, 242_i64, 243_i64],
                                        vec![244_i64, 245_i64, 246_i64],
                                    ],
                                    vec![
                                        vec![247_i64, 248_i64, 249_i64],
                                        vec![250_i64, 251_i64, 252_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![253_i64, 254_i64, 255_i64],
                                        vec![256_i64, 257_i64, 258_i64],
                                    ],
                                    vec![
                                        vec![259_i64, 260_i64, 261_i64],
                                        vec![262_i64, 263_i64, 264_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![265_i64, 266_i64, 267_i64],
                                        vec![268_i64, 269_i64, 270_i64],
                                    ],
                                    vec![
                                        vec![271_i64, 272_i64, 273_i64],
                                        vec![274_i64, 275_i64, 276_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![277_i64, 278_i64, 279_i64],
                                        vec![280_i64, 281_i64, 282_i64],
                                    ],
                                    vec![
                                        vec![283_i64, 284_i64, 285_i64],
                                        vec![286_i64, 287_i64, 288_i64],
                                    ],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![289_i64, 290_i64, 291_i64],
                                        vec![292_i64, 293_i64, 294_i64],
                                    ],
                                    vec![
                                        vec![295_i64, 296_i64, 297_i64],
                                        vec![298_i64, 299_i64, 300_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![301_i64, 302_i64, 303_i64],
                                        vec![304_i64, 305_i64, 306_i64],
                                    ],
                                    vec![
                                        vec![307_i64, 308_i64, 309_i64],
                                        vec![310_i64, 311_i64, 312_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![313_i64, 314_i64, 315_i64],
                                        vec![316_i64, 317_i64, 318_i64],
                                    ],
                                    vec![
                                        vec![319_i64, 320_i64, 321_i64],
                                        vec![322_i64, 323_i64, 324_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![325_i64, 326_i64, 327_i64],
                                        vec![328_i64, 329_i64, 330_i64],
                                    ],
                                    vec![
                                        vec![331_i64, 332_i64, 333_i64],
                                        vec![334_i64, 335_i64, 336_i64],
                                    ],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![337_i64, 338_i64, 339_i64],
                                        vec![340_i64, 341_i64, 342_i64],
                                    ],
                                    vec![
                                        vec![343_i64, 344_i64, 345_i64],
                                        vec![346_i64, 347_i64, 348_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![349_i64, 350_i64, 351_i64],
                                        vec![352_i64, 353_i64, 354_i64],
                                    ],
                                    vec![
                                        vec![355_i64, 356_i64, 357_i64],
                                        vec![358_i64, 359_i64, 360_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![361_i64, 362_i64, 363_i64],
                                        vec![364_i64, 365_i64, 366_i64],
                                    ],
                                    vec![
                                        vec![367_i64, 368_i64, 369_i64],
                                        vec![370_i64, 371_i64, 372_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![373_i64, 374_i64, 375_i64],
                                        vec![376_i64, 377_i64, 378_i64],
                                    ],
                                    vec![
                                        vec![379_i64, 380_i64, 381_i64],
                                        vec![382_i64, 383_i64, 384_i64],
                                    ],
                                ],
                            ],
                        ],
                    ],
                ],
            ],
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![385_i64, 386_i64, 387_i64],
                                        vec![388_i64, 389_i64, 390_i64],
                                    ],
                                    vec![
                                        vec![391_i64, 392_i64, 393_i64],
                                        vec![394_i64, 395_i64, 396_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![397_i64, 398_i64, 399_i64],
                                        vec![400_i64, 401_i64, 402_i64],
                                    ],
                                    vec![
                                        vec![403_i64, 404_i64, 405_i64],
                                        vec![406_i64, 407_i64, 408_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![409_i64, 410_i64, 411_i64],
                                        vec![412_i64, 413_i64, 414_i64],
                                    ],
                                    vec![
                                        vec![415_i64, 416_i64, 417_i64],
                                        vec![418_i64, 419_i64, 420_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![421_i64, 422_i64, 423_i64],
                                        vec![424_i64, 425_i64, 426_i64],
                                    ],
                                    vec![
                                        vec![427_i64, 428_i64, 429_i64],
                                        vec![430_i64, 431_i64, 432_i64],
                                    ],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![433_i64, 434_i64, 435_i64],
                                        vec![436_i64, 437_i64, 438_i64],
                                    ],
                                    vec![
                                        vec![439_i64, 440_i64, 441_i64],
                                        vec![442_i64, 443_i64, 444_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![445_i64, 446_i64, 447_i64],
                                        vec![448_i64, 449_i64, 450_i64],
                                    ],
                                    vec![
                                        vec![451_i64, 452_i64, 453_i64],
                                        vec![454_i64, 455_i64, 456_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![457_i64, 458_i64, 459_i64],
                                        vec![460_i64, 461_i64, 462_i64],
                                    ],
                                    vec![
                                        vec![463_i64, 464_i64, 465_i64],
                                        vec![466_i64, 467_i64, 468_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![469_i64, 470_i64, 471_i64],
                                        vec![472_i64, 473_i64, 474_i64],
                                    ],
                                    vec![
                                        vec![475_i64, 476_i64, 477_i64],
                                        vec![478_i64, 479_i64, 480_i64],
                                    ],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![481_i64, 482_i64, 483_i64],
                                        vec![484_i64, 485_i64, 486_i64],
                                    ],
                                    vec![
                                        vec![487_i64, 488_i64, 489_i64],
                                        vec![490_i64, 491_i64, 492_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![493_i64, 494_i64, 495_i64],
                                        vec![496_i64, 497_i64, 498_i64],
                                    ],
                                    vec![
                                        vec![499_i64, 500_i64, 501_i64],
                                        vec![502_i64, 503_i64, 504_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![505_i64, 506_i64, 507_i64],
                                        vec![508_i64, 509_i64, 510_i64],
                                    ],
                                    vec![
                                        vec![511_i64, 512_i64, 513_i64],
                                        vec![514_i64, 515_i64, 516_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![517_i64, 518_i64, 519_i64],
                                        vec![520_i64, 521_i64, 522_i64],
                                    ],
                                    vec![
                                        vec![523_i64, 524_i64, 525_i64],
                                        vec![526_i64, 527_i64, 528_i64],
                                    ],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![529_i64, 530_i64, 531_i64],
                                        vec![532_i64, 533_i64, 534_i64],
                                    ],
                                    vec![
                                        vec![535_i64, 536_i64, 537_i64],
                                        vec![538_i64, 539_i64, 540_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![541_i64, 542_i64, 543_i64],
                                        vec![544_i64, 545_i64, 546_i64],
                                    ],
                                    vec![
                                        vec![547_i64, 548_i64, 549_i64],
                                        vec![550_i64, 551_i64, 552_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![553_i64, 554_i64, 555_i64],
                                        vec![556_i64, 557_i64, 558_i64],
                                    ],
                                    vec![
                                        vec![559_i64, 560_i64, 561_i64],
                                        vec![562_i64, 563_i64, 564_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![565_i64, 566_i64, 567_i64],
                                        vec![568_i64, 569_i64, 570_i64],
                                    ],
                                    vec![
                                        vec![571_i64, 572_i64, 573_i64],
                                        vec![574_i64, 575_i64, 576_i64],
                                    ],
                                ],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![577_i64, 578_i64, 579_i64],
                                        vec![580_i64, 581_i64, 582_i64],
                                    ],
                                    vec![
                                        vec![583_i64, 584_i64, 585_i64],
                                        vec![586_i64, 587_i64, 588_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![589_i64, 590_i64, 591_i64],
                                        vec![592_i64, 593_i64, 594_i64],
                                    ],
                                    vec![
                                        vec![595_i64, 596_i64, 597_i64],
                                        vec![598_i64, 599_i64, 600_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![601_i64, 602_i64, 603_i64],
                                        vec![604_i64, 605_i64, 606_i64],
                                    ],
                                    vec![
                                        vec![607_i64, 608_i64, 609_i64],
                                        vec![610_i64, 611_i64, 612_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![613_i64, 614_i64, 615_i64],
                                        vec![616_i64, 617_i64, 618_i64],
                                    ],
                                    vec![
                                        vec![619_i64, 620_i64, 621_i64],
                                        vec![622_i64, 623_i64, 624_i64],
                                    ],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![625_i64, 626_i64, 627_i64],
                                        vec![628_i64, 629_i64, 630_i64],
                                    ],
                                    vec![
                                        vec![631_i64, 632_i64, 633_i64],
                                        vec![634_i64, 635_i64, 636_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![637_i64, 638_i64, 639_i64],
                                        vec![640_i64, 641_i64, 642_i64],
                                    ],
                                    vec![
                                        vec![643_i64, 644_i64, 645_i64],
                                        vec![646_i64, 647_i64, 648_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![649_i64, 650_i64, 651_i64],
                                        vec![652_i64, 653_i64, 654_i64],
                                    ],
                                    vec![
                                        vec![655_i64, 656_i64, 657_i64],
                                        vec![658_i64, 659_i64, 660_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![661_i64, 662_i64, 663_i64],
                                        vec![664_i64, 665_i64, 666_i64],
                                    ],
                                    vec![
                                        vec![667_i64, 668_i64, 669_i64],
                                        vec![670_i64, 671_i64, 672_i64],
                                    ],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![673_i64, 674_i64, 675_i64],
                                        vec![676_i64, 677_i64, 678_i64],
                                    ],
                                    vec![
                                        vec![679_i64, 680_i64, 681_i64],
                                        vec![682_i64, 683_i64, 684_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![685_i64, 686_i64, 687_i64],
                                        vec![688_i64, 689_i64, 690_i64],
                                    ],
                                    vec![
                                        vec![691_i64, 692_i64, 693_i64],
                                        vec![694_i64, 695_i64, 696_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![697_i64, 698_i64, 699_i64],
                                        vec![700_i64, 701_i64, 702_i64],
                                    ],
                                    vec![
                                        vec![703_i64, 704_i64, 705_i64],
                                        vec![706_i64, 707_i64, 708_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![709_i64, 710_i64, 711_i64],
                                        vec![712_i64, 713_i64, 714_i64],
                                    ],
                                    vec![
                                        vec![715_i64, 716_i64, 717_i64],
                                        vec![718_i64, 719_i64, 720_i64],
                                    ],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![
                                        vec![721_i64, 722_i64, 723_i64],
                                        vec![724_i64, 725_i64, 726_i64],
                                    ],
                                    vec![
                                        vec![727_i64, 728_i64, 729_i64],
                                        vec![730_i64, 731_i64, 732_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![733_i64, 734_i64, 735_i64],
                                        vec![736_i64, 737_i64, 738_i64],
                                    ],
                                    vec![
                                        vec![739_i64, 740_i64, 741_i64],
                                        vec![742_i64, 743_i64, 744_i64],
                                    ],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![
                                        vec![745_i64, 746_i64, 747_i64],
                                        vec![748_i64, 749_i64, 750_i64],
                                    ],
                                    vec![
                                        vec![751_i64, 752_i64, 753_i64],
                                        vec![754_i64, 755_i64, 756_i64],
                                    ],
                                ],
                                vec![
                                    vec![
                                        vec![757_i64, 758_i64, 759_i64],
                                        vec![760_i64, 761_i64, 762_i64],
                                    ],
                                    vec![
                                        vec![763_i64, 764_i64, 765_i64],
                                        vec![766_i64, 767_i64, 768_i64],
                                    ],
                                ],
                            ],
                        ],
                    ],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64, 7_i64, 8_i64, 9_i64, 10_i64, 11_i64, 12_i64,
            13_i64, 14_i64, 15_i64, 16_i64, 17_i64, 18_i64, 19_i64, 20_i64, 21_i64, 22_i64, 23_i64,
            24_i64, 25_i64, 26_i64, 27_i64, 28_i64, 29_i64, 30_i64, 31_i64, 32_i64, 33_i64, 34_i64,
            35_i64, 36_i64, 37_i64, 38_i64, 39_i64, 40_i64, 41_i64, 42_i64, 43_i64, 44_i64, 45_i64,
            46_i64, 47_i64, 48_i64, 49_i64, 50_i64, 51_i64, 52_i64, 53_i64, 54_i64, 55_i64, 56_i64,
            57_i64, 58_i64, 59_i64, 60_i64, 61_i64, 62_i64, 63_i64, 64_i64, 65_i64, 66_i64, 67_i64,
            68_i64, 69_i64, 70_i64, 71_i64, 72_i64, 73_i64, 74_i64, 75_i64, 76_i64, 77_i64, 78_i64,
            79_i64, 80_i64, 81_i64, 82_i64, 83_i64, 84_i64, 85_i64, 86_i64, 87_i64, 88_i64, 89_i64,
            90_i64, 91_i64, 92_i64, 93_i64, 94_i64, 95_i64, 96_i64, 97_i64, 98_i64, 99_i64,
            100_i64, 101_i64, 102_i64, 103_i64, 104_i64, 105_i64, 106_i64, 107_i64, 108_i64,
            109_i64, 110_i64, 111_i64, 112_i64, 113_i64, 114_i64, 115_i64, 116_i64, 117_i64,
            118_i64, 119_i64, 120_i64, 121_i64, 122_i64, 123_i64, 124_i64, 125_i64, 126_i64,
            127_i64, 128_i64, 129_i64, 130_i64, 131_i64, 132_i64, 133_i64, 134_i64, 135_i64,
            136_i64, 137_i64, 138_i64, 139_i64, 140_i64, 141_i64, 142_i64, 143_i64, 144_i64,
            145_i64, 146_i64, 147_i64, 148_i64, 149_i64, 150_i64, 151_i64, 152_i64, 153_i64,
            154_i64, 155_i64, 156_i64, 157_i64, 158_i64, 159_i64, 160_i64, 161_i64, 162_i64,
            163_i64, 164_i64, 165_i64, 166_i64, 167_i64, 168_i64, 169_i64, 170_i64, 171_i64,
            172_i64, 173_i64, 174_i64, 175_i64, 176_i64, 177_i64, 178_i64, 179_i64, 180_i64,
            181_i64, 182_i64, 183_i64, 184_i64, 185_i64, 186_i64, 187_i64, 188_i64, 189_i64,
            190_i64, 191_i64, 192_i64, 193_i64, 194_i64, 195_i64, 196_i64, 197_i64, 198_i64,
            199_i64, 200_i64, 201_i64, 202_i64, 203_i64, 204_i64, 205_i64, 206_i64, 207_i64,
            208_i64, 209_i64, 210_i64, 211_i64, 212_i64, 213_i64, 214_i64, 215_i64, 216_i64,
            217_i64, 218_i64, 219_i64, 220_i64, 221_i64, 222_i64, 223_i64, 224_i64, 225_i64,
            226_i64, 227_i64, 228_i64, 229_i64, 230_i64, 231_i64, 232_i64, 233_i64, 234_i64,
            235_i64, 236_i64, 237_i64, 238_i64, 239_i64, 240_i64, 241_i64, 242_i64, 243_i64,
            244_i64, 245_i64, 246_i64, 247_i64, 248_i64, 249_i64, 250_i64, 251_i64, 252_i64,
            253_i64, 254_i64, 255_i64, 256_i64, 257_i64, 258_i64, 259_i64, 260_i64, 261_i64,
            262_i64, 263_i64, 264_i64, 265_i64, 266_i64, 267_i64, 268_i64, 269_i64, 270_i64,
            271_i64, 272_i64, 273_i64, 274_i64, 275_i64, 276_i64, 277_i64, 278_i64, 279_i64,
            280_i64, 281_i64, 282_i64, 283_i64, 284_i64, 285_i64, 286_i64, 287_i64, 288_i64,
            289_i64, 290_i64, 291_i64, 292_i64, 293_i64, 294_i64, 295_i64, 296_i64, 297_i64,
            298_i64, 299_i64, 300_i64, 301_i64, 302_i64, 303_i64, 304_i64, 305_i64, 306_i64,
            307_i64, 308_i64, 309_i64, 310_i64, 311_i64, 312_i64, 313_i64, 314_i64, 315_i64,
            316_i64, 317_i64, 318_i64, 319_i64, 320_i64, 321_i64, 322_i64, 323_i64, 324_i64,
            325_i64, 326_i64, 327_i64, 328_i64, 329_i64, 330_i64, 331_i64, 332_i64, 333_i64,
            334_i64, 335_i64, 336_i64, 337_i64, 338_i64, 339_i64, 340_i64, 341_i64, 342_i64,
            343_i64, 344_i64, 345_i64, 346_i64, 347_i64, 348_i64, 349_i64, 350_i64, 351_i64,
            352_i64, 353_i64, 354_i64, 355_i64, 356_i64, 357_i64, 358_i64, 359_i64, 360_i64,
            361_i64, 362_i64, 363_i64, 364_i64, 365_i64, 366_i64, 367_i64, 368_i64, 369_i64,
            370_i64, 371_i64, 372_i64, 373_i64, 374_i64, 375_i64, 376_i64, 377_i64, 378_i64,
            379_i64, 380_i64, 381_i64, 382_i64, 383_i64, 384_i64, 385_i64, 386_i64, 387_i64,
            388_i64, 389_i64, 390_i64, 391_i64, 392_i64, 393_i64, 394_i64, 395_i64, 396_i64,
            397_i64, 398_i64, 399_i64, 400_i64, 401_i64, 402_i64, 403_i64, 404_i64, 405_i64,
            406_i64, 407_i64, 408_i64, 409_i64, 410_i64, 411_i64, 412_i64, 413_i64, 414_i64,
            415_i64, 416_i64, 417_i64, 418_i64, 419_i64, 420_i64, 421_i64, 422_i64, 423_i64,
            424_i64, 425_i64, 426_i64, 427_i64, 428_i64, 429_i64, 430_i64, 431_i64, 432_i64,
            433_i64, 434_i64, 435_i64, 436_i64, 437_i64, 438_i64, 439_i64, 440_i64, 441_i64,
            442_i64, 443_i64, 444_i64, 445_i64, 446_i64, 447_i64, 448_i64, 449_i64, 450_i64,
            451_i64, 452_i64, 453_i64, 454_i64, 455_i64, 456_i64, 457_i64, 458_i64, 459_i64,
            460_i64, 461_i64, 462_i64, 463_i64, 464_i64, 465_i64, 466_i64, 467_i64, 468_i64,
            469_i64, 470_i64, 471_i64, 472_i64, 473_i64, 474_i64, 475_i64, 476_i64, 477_i64,
            478_i64, 479_i64, 480_i64, 481_i64, 482_i64, 483_i64, 484_i64, 485_i64, 486_i64,
            487_i64, 488_i64, 489_i64, 490_i64, 491_i64, 492_i64, 493_i64, 494_i64, 495_i64,
            496_i64, 497_i64, 498_i64, 499_i64, 500_i64, 501_i64, 502_i64, 503_i64, 504_i64,
            505_i64, 506_i64, 507_i64, 508_i64, 509_i64, 510_i64, 511_i64, 512_i64, 513_i64,
            514_i64, 515_i64, 516_i64, 517_i64, 518_i64, 519_i64, 520_i64, 521_i64, 522_i64,
            523_i64, 524_i64, 525_i64, 526_i64, 527_i64, 528_i64, 529_i64, 530_i64, 531_i64,
            532_i64, 533_i64, 534_i64, 535_i64, 536_i64, 537_i64, 538_i64, 539_i64, 540_i64,
            541_i64, 542_i64, 543_i64, 544_i64, 545_i64, 546_i64, 547_i64, 548_i64, 549_i64,
            550_i64, 551_i64, 552_i64, 553_i64, 554_i64, 555_i64, 556_i64, 557_i64, 558_i64,
            559_i64, 560_i64, 561_i64, 562_i64, 563_i64, 564_i64, 565_i64, 566_i64, 567_i64,
            568_i64, 569_i64, 570_i64, 571_i64, 572_i64, 573_i64, 574_i64, 575_i64, 576_i64,
            577_i64, 578_i64, 579_i64, 580_i64, 581_i64, 582_i64, 583_i64, 584_i64, 585_i64,
            586_i64, 587_i64, 588_i64, 589_i64, 590_i64, 591_i64, 592_i64, 593_i64, 594_i64,
            595_i64, 596_i64, 597_i64, 598_i64, 599_i64, 600_i64, 601_i64, 602_i64, 603_i64,
            604_i64, 605_i64, 606_i64, 607_i64, 608_i64, 609_i64, 610_i64, 611_i64, 612_i64,
            613_i64, 614_i64, 615_i64, 616_i64, 617_i64, 618_i64, 619_i64, 620_i64, 621_i64,
            622_i64, 623_i64, 624_i64, 625_i64, 626_i64, 627_i64, 628_i64, 629_i64, 630_i64,
            631_i64, 632_i64, 633_i64, 634_i64, 635_i64, 636_i64, 637_i64, 638_i64, 639_i64,
            640_i64, 641_i64, 642_i64, 643_i64, 644_i64, 645_i64, 646_i64, 647_i64, 648_i64,
            649_i64, 650_i64, 651_i64, 652_i64, 653_i64, 654_i64, 655_i64, 656_i64, 657_i64,
            658_i64, 659_i64, 660_i64, 661_i64, 662_i64, 663_i64, 664_i64, 665_i64, 666_i64,
            667_i64, 668_i64, 669_i64, 670_i64, 671_i64, 672_i64, 673_i64, 674_i64, 675_i64,
            676_i64, 677_i64, 678_i64, 679_i64, 680_i64, 681_i64, 682_i64, 683_i64, 684_i64,
            685_i64, 686_i64, 687_i64, 688_i64, 689_i64, 690_i64, 691_i64, 692_i64, 693_i64,
            694_i64, 695_i64, 696_i64, 697_i64, 698_i64, 699_i64, 700_i64, 701_i64, 702_i64,
            703_i64, 704_i64, 705_i64, 706_i64, 707_i64, 708_i64, 709_i64, 710_i64, 711_i64,
            712_i64, 713_i64, 714_i64, 715_i64, 716_i64, 717_i64, 718_i64, 719_i64, 720_i64,
            721_i64, 722_i64, 723_i64, 724_i64, 725_i64, 726_i64, 727_i64, 728_i64, 729_i64,
            730_i64, 731_i64, 732_i64, 733_i64, 734_i64, 735_i64, 736_i64, 737_i64, 738_i64,
            739_i64, 740_i64, 741_i64, 742_i64, 743_i64, 744_i64, 745_i64, 746_i64, 747_i64,
            748_i64, 749_i64, 750_i64, 751_i64, 752_i64, 753_i64, 754_i64, 755_i64, 756_i64,
            757_i64, 758_i64, 759_i64, 760_i64, 761_i64, 762_i64, 763_i64, 764_i64, 765_i64,
            766_i64, 767_i64, 768_i64,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,,,,,] int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>>> = vec![
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]],
                                    vec![vec![0.875, 1.0, 1.125], vec![1.25, 1.375, 1.5]],
                                ],
                                vec![
                                    vec![vec![1.625, 1.75, 1.875], vec![2.0, 2.125, 2.25]],
                                    vec![vec![2.375, 2.5, 2.625], vec![2.75, 2.875, 3.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![3.125, 3.25, 3.375], vec![3.5, 3.625, 3.75]],
                                    vec![vec![3.875, 4.0, 4.125], vec![4.25, 4.375, 4.5]],
                                ],
                                vec![
                                    vec![vec![4.625, 4.75, 4.875], vec![5.0, 5.125, 5.25]],
                                    vec![vec![5.375, 5.5, 5.625], vec![5.75, 5.875, 6.0]],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![vec![6.125, 6.25, 6.375], vec![6.5, 6.625, 6.75]],
                                    vec![vec![6.875, 7.0, 7.125], vec![7.25, 7.375, 7.5]],
                                ],
                                vec![
                                    vec![vec![7.625, 7.75, 7.875], vec![8.0, 8.125, 8.25]],
                                    vec![vec![8.375, 8.5, 8.625], vec![8.75, 8.875, 9.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![9.125, 9.25, 9.375], vec![9.5, 9.625, 9.75]],
                                    vec![vec![9.875, 10.0, 10.125], vec![10.25, 10.375, 10.5]],
                                ],
                                vec![
                                    vec![vec![10.625, 10.75, 10.875], vec![11.0, 11.125, 11.25]],
                                    vec![vec![11.375, 11.5, 11.625], vec![11.75, 11.875, 12.0]],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![12.125, 12.25, 12.375], vec![12.5, 12.625, 12.75]],
                                    vec![vec![12.875, 13.0, 13.125], vec![13.25, 13.375, 13.5]],
                                ],
                                vec![
                                    vec![vec![13.625, 13.75, 13.875], vec![14.0, 14.125, 14.25]],
                                    vec![vec![14.375, 14.5, 14.625], vec![14.75, 14.875, 15.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![15.125, 15.25, 15.375], vec![15.5, 15.625, 15.75]],
                                    vec![vec![15.875, 16.0, 16.125], vec![16.25, 16.375, 16.5]],
                                ],
                                vec![
                                    vec![vec![16.625, 16.75, 16.875], vec![17.0, 17.125, 17.25]],
                                    vec![vec![17.375, 17.5, 17.625], vec![17.75, 17.875, 18.0]],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![vec![18.125, 18.25, 18.375], vec![18.5, 18.625, 18.75]],
                                    vec![vec![18.875, 19.0, 19.125], vec![19.25, 19.375, 19.5]],
                                ],
                                vec![
                                    vec![vec![19.625, 19.75, 19.875], vec![20.0, 20.125, 20.25]],
                                    vec![vec![20.375, 20.5, 20.625], vec![20.75, 20.875, 21.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![21.125, 21.25, 21.375], vec![21.5, 21.625, 21.75]],
                                    vec![vec![21.875, 22.0, 22.125], vec![22.25, 22.375, 22.5]],
                                ],
                                vec![
                                    vec![vec![22.625, 22.75, 22.875], vec![23.0, 23.125, 23.25]],
                                    vec![vec![23.375, 23.5, 23.625], vec![23.75, 23.875, 24.0]],
                                ],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![24.125, 24.25, 24.375], vec![24.5, 24.625, 24.75]],
                                    vec![vec![24.875, 25.0, 25.125], vec![25.25, 25.375, 25.5]],
                                ],
                                vec![
                                    vec![vec![25.625, 25.75, 25.875], vec![26.0, 26.125, 26.25]],
                                    vec![vec![26.375, 26.5, 26.625], vec![26.75, 26.875, 27.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![27.125, 27.25, 27.375], vec![27.5, 27.625, 27.75]],
                                    vec![vec![27.875, 28.0, 28.125], vec![28.25, 28.375, 28.5]],
                                ],
                                vec![
                                    vec![vec![28.625, 28.75, 28.875], vec![29.0, 29.125, 29.25]],
                                    vec![vec![29.375, 29.5, 29.625], vec![29.75, 29.875, 30.0]],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![vec![30.125, 30.25, 30.375], vec![30.5, 30.625, 30.75]],
                                    vec![vec![30.875, 31.0, 31.125], vec![31.25, 31.375, 31.5]],
                                ],
                                vec![
                                    vec![vec![31.625, 31.75, 31.875], vec![32.0, 32.125, 32.25]],
                                    vec![vec![32.375, 32.5, 32.625], vec![32.75, 32.875, 33.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![33.125, 33.25, 33.375], vec![33.5, 33.625, 33.75]],
                                    vec![vec![33.875, 34.0, 34.125], vec![34.25, 34.375, 34.5]],
                                ],
                                vec![
                                    vec![vec![34.625, 34.75, 34.875], vec![35.0, 35.125, 35.25]],
                                    vec![vec![35.375, 35.5, 35.625], vec![35.75, 35.875, 36.0]],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![36.125, 36.25, 36.375], vec![36.5, 36.625, 36.75]],
                                    vec![vec![36.875, 37.0, 37.125], vec![37.25, 37.375, 37.5]],
                                ],
                                vec![
                                    vec![vec![37.625, 37.75, 37.875], vec![38.0, 38.125, 38.25]],
                                    vec![vec![38.375, 38.5, 38.625], vec![38.75, 38.875, 39.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![39.125, 39.25, 39.375], vec![39.5, 39.625, 39.75]],
                                    vec![vec![39.875, 40.0, 40.125], vec![40.25, 40.375, 40.5]],
                                ],
                                vec![
                                    vec![vec![40.625, 40.75, 40.875], vec![41.0, 41.125, 41.25]],
                                    vec![vec![41.375, 41.5, 41.625], vec![41.75, 41.875, 42.0]],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![vec![42.125, 42.25, 42.375], vec![42.5, 42.625, 42.75]],
                                    vec![vec![42.875, 43.0, 43.125], vec![43.25, 43.375, 43.5]],
                                ],
                                vec![
                                    vec![vec![43.625, 43.75, 43.875], vec![44.0, 44.125, 44.25]],
                                    vec![vec![44.375, 44.5, 44.625], vec![44.75, 44.875, 45.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![45.125, 45.25, 45.375], vec![45.5, 45.625, 45.75]],
                                    vec![vec![45.875, 46.0, 46.125], vec![46.25, 46.375, 46.5]],
                                ],
                                vec![
                                    vec![vec![46.625, 46.75, 46.875], vec![47.0, 47.125, 47.25]],
                                    vec![vec![47.375, 47.5, 47.625], vec![47.75, 47.875, 48.0]],
                                ],
                            ],
                        ],
                    ],
                ],
            ],
            vec![
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![48.125, 48.25, 48.375], vec![48.5, 48.625, 48.75]],
                                    vec![vec![48.875, 49.0, 49.125], vec![49.25, 49.375, 49.5]],
                                ],
                                vec![
                                    vec![vec![49.625, 49.75, 49.875], vec![50.0, 50.125, 50.25]],
                                    vec![vec![50.375, 50.5, 50.625], vec![50.75, 50.875, 51.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![51.125, 51.25, 51.375], vec![51.5, 51.625, 51.75]],
                                    vec![vec![51.875, 52.0, 52.125], vec![52.25, 52.375, 52.5]],
                                ],
                                vec![
                                    vec![vec![52.625, 52.75, 52.875], vec![53.0, 53.125, 53.25]],
                                    vec![vec![53.375, 53.5, 53.625], vec![53.75, 53.875, 54.0]],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![vec![54.125, 54.25, 54.375], vec![54.5, 54.625, 54.75]],
                                    vec![vec![54.875, 55.0, 55.125], vec![55.25, 55.375, 55.5]],
                                ],
                                vec![
                                    vec![vec![55.625, 55.75, 55.875], vec![56.0, 56.125, 56.25]],
                                    vec![vec![56.375, 56.5, 56.625], vec![56.75, 56.875, 57.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![57.125, 57.25, 57.375], vec![57.5, 57.625, 57.75]],
                                    vec![vec![57.875, 58.0, 58.125], vec![58.25, 58.375, 58.5]],
                                ],
                                vec![
                                    vec![vec![58.625, 58.75, 58.875], vec![59.0, 59.125, 59.25]],
                                    vec![vec![59.375, 59.5, 59.625], vec![59.75, 59.875, 60.0]],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![60.125, 60.25, 60.375], vec![60.5, 60.625, 60.75]],
                                    vec![vec![60.875, 61.0, 61.125], vec![61.25, 61.375, 61.5]],
                                ],
                                vec![
                                    vec![vec![61.625, 61.75, 61.875], vec![62.0, 62.125, 62.25]],
                                    vec![vec![62.375, 62.5, 62.625], vec![62.75, 62.875, 63.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![63.125, 63.25, 63.375], vec![63.5, 63.625, 63.75]],
                                    vec![vec![63.875, 64.0, 64.125], vec![64.25, 64.375, 64.5]],
                                ],
                                vec![
                                    vec![vec![64.625, 64.75, 64.875], vec![65.0, 65.125, 65.25]],
                                    vec![vec![65.375, 65.5, 65.625], vec![65.75, 65.875, 66.0]],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![vec![66.125, 66.25, 66.375], vec![66.5, 66.625, 66.75]],
                                    vec![vec![66.875, 67.0, 67.125], vec![67.25, 67.375, 67.5]],
                                ],
                                vec![
                                    vec![vec![67.625, 67.75, 67.875], vec![68.0, 68.125, 68.25]],
                                    vec![vec![68.375, 68.5, 68.625], vec![68.75, 68.875, 69.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![69.125, 69.25, 69.375], vec![69.5, 69.625, 69.75]],
                                    vec![vec![69.875, 70.0, 70.125], vec![70.25, 70.375, 70.5]],
                                ],
                                vec![
                                    vec![vec![70.625, 70.75, 70.875], vec![71.0, 71.125, 71.25]],
                                    vec![vec![71.375, 71.5, 71.625], vec![71.75, 71.875, 72.0]],
                                ],
                            ],
                        ],
                    ],
                ],
                vec![
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![72.125, 72.25, 72.375], vec![72.5, 72.625, 72.75]],
                                    vec![vec![72.875, 73.0, 73.125], vec![73.25, 73.375, 73.5]],
                                ],
                                vec![
                                    vec![vec![73.625, 73.75, 73.875], vec![74.0, 74.125, 74.25]],
                                    vec![vec![74.375, 74.5, 74.625], vec![74.75, 74.875, 75.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![75.125, 75.25, 75.375], vec![75.5, 75.625, 75.75]],
                                    vec![vec![75.875, 76.0, 76.125], vec![76.25, 76.375, 76.5]],
                                ],
                                vec![
                                    vec![vec![76.625, 76.75, 76.875], vec![77.0, 77.125, 77.25]],
                                    vec![vec![77.375, 77.5, 77.625], vec![77.75, 77.875, 78.0]],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![vec![78.125, 78.25, 78.375], vec![78.5, 78.625, 78.75]],
                                    vec![vec![78.875, 79.0, 79.125], vec![79.25, 79.375, 79.5]],
                                ],
                                vec![
                                    vec![vec![79.625, 79.75, 79.875], vec![80.0, 80.125, 80.25]],
                                    vec![vec![80.375, 80.5, 80.625], vec![80.75, 80.875, 81.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![81.125, 81.25, 81.375], vec![81.5, 81.625, 81.75]],
                                    vec![vec![81.875, 82.0, 82.125], vec![82.25, 82.375, 82.5]],
                                ],
                                vec![
                                    vec![vec![82.625, 82.75, 82.875], vec![83.0, 83.125, 83.25]],
                                    vec![vec![83.375, 83.5, 83.625], vec![83.75, 83.875, 84.0]],
                                ],
                            ],
                        ],
                    ],
                    vec![
                        vec![
                            vec![
                                vec![
                                    vec![vec![84.125, 84.25, 84.375], vec![84.5, 84.625, 84.75]],
                                    vec![vec![84.875, 85.0, 85.125], vec![85.25, 85.375, 85.5]],
                                ],
                                vec![
                                    vec![vec![85.625, 85.75, 85.875], vec![86.0, 86.125, 86.25]],
                                    vec![vec![86.375, 86.5, 86.625], vec![86.75, 86.875, 87.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![87.125, 87.25, 87.375], vec![87.5, 87.625, 87.75]],
                                    vec![vec![87.875, 88.0, 88.125], vec![88.25, 88.375, 88.5]],
                                ],
                                vec![
                                    vec![vec![88.625, 88.75, 88.875], vec![89.0, 89.125, 89.25]],
                                    vec![vec![89.375, 89.5, 89.625], vec![89.75, 89.875, 90.0]],
                                ],
                            ],
                        ],
                        vec![
                            vec![
                                vec![
                                    vec![vec![90.125, 90.25, 90.375], vec![90.5, 90.625, 90.75]],
                                    vec![vec![90.875, 91.0, 91.125], vec![91.25, 91.375, 91.5]],
                                ],
                                vec![
                                    vec![vec![91.625, 91.75, 91.875], vec![92.0, 92.125, 92.25]],
                                    vec![vec![92.375, 92.5, 92.625], vec![92.75, 92.875, 93.0]],
                                ],
                            ],
                            vec![
                                vec![
                                    vec![vec![93.125, 93.25, 93.375], vec![93.5, 93.625, 93.75]],
                                    vec![vec![93.875, 94.0, 94.125], vec![94.25, 94.375, 94.5]],
                                ],
                                vec![
                                    vec![vec![94.625, 94.75, 94.875], vec![95.0, 95.125, 95.25]],
                                    vec![vec![95.375, 95.5, 95.625], vec![95.75, 95.875, 96.0]],
                                ],
                            ],
                        ],
                    ],
                ],
            ],
        ];
        let got = to_array_1d_nested(&x);
        let want = vec![
            0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75,
            1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5,
            3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25,
            5.375, 5.5, 5.625, 5.75, 5.875, 6.0, 6.125, 6.25, 6.375, 6.5, 6.625, 6.75, 6.875, 7.0,
            7.125, 7.25, 7.375, 7.5, 7.625, 7.75, 7.875, 8.0, 8.125, 8.25, 8.375, 8.5, 8.625, 8.75,
            8.875, 9.0, 9.125, 9.25, 9.375, 9.5, 9.625, 9.75, 9.875, 10.0, 10.125, 10.25, 10.375,
            10.5, 10.625, 10.75, 10.875, 11.0, 11.125, 11.25, 11.375, 11.5, 11.625, 11.75, 11.875,
            12.0, 12.125, 12.25, 12.375, 12.5, 12.625, 12.75, 12.875, 13.0, 13.125, 13.25, 13.375,
            13.5, 13.625, 13.75, 13.875, 14.0, 14.125, 14.25, 14.375, 14.5, 14.625, 14.75, 14.875,
            15.0, 15.125, 15.25, 15.375, 15.5, 15.625, 15.75, 15.875, 16.0, 16.125, 16.25, 16.375,
            16.5, 16.625, 16.75, 16.875, 17.0, 17.125, 17.25, 17.375, 17.5, 17.625, 17.75, 17.875,
            18.0, 18.125, 18.25, 18.375, 18.5, 18.625, 18.75, 18.875, 19.0, 19.125, 19.25, 19.375,
            19.5, 19.625, 19.75, 19.875, 20.0, 20.125, 20.25, 20.375, 20.5, 20.625, 20.75, 20.875,
            21.0, 21.125, 21.25, 21.375, 21.5, 21.625, 21.75, 21.875, 22.0, 22.125, 22.25, 22.375,
            22.5, 22.625, 22.75, 22.875, 23.0, 23.125, 23.25, 23.375, 23.5, 23.625, 23.75, 23.875,
            24.0, 24.125, 24.25, 24.375, 24.5, 24.625, 24.75, 24.875, 25.0, 25.125, 25.25, 25.375,
            25.5, 25.625, 25.75, 25.875, 26.0, 26.125, 26.25, 26.375, 26.5, 26.625, 26.75, 26.875,
            27.0, 27.125, 27.25, 27.375, 27.5, 27.625, 27.75, 27.875, 28.0, 28.125, 28.25, 28.375,
            28.5, 28.625, 28.75, 28.875, 29.0, 29.125, 29.25, 29.375, 29.5, 29.625, 29.75, 29.875,
            30.0, 30.125, 30.25, 30.375, 30.5, 30.625, 30.75, 30.875, 31.0, 31.125, 31.25, 31.375,
            31.5, 31.625, 31.75, 31.875, 32.0, 32.125, 32.25, 32.375, 32.5, 32.625, 32.75, 32.875,
            33.0, 33.125, 33.25, 33.375, 33.5, 33.625, 33.75, 33.875, 34.0, 34.125, 34.25, 34.375,
            34.5, 34.625, 34.75, 34.875, 35.0, 35.125, 35.25, 35.375, 35.5, 35.625, 35.75, 35.875,
            36.0, 36.125, 36.25, 36.375, 36.5, 36.625, 36.75, 36.875, 37.0, 37.125, 37.25, 37.375,
            37.5, 37.625, 37.75, 37.875, 38.0, 38.125, 38.25, 38.375, 38.5, 38.625, 38.75, 38.875,
            39.0, 39.125, 39.25, 39.375, 39.5, 39.625, 39.75, 39.875, 40.0, 40.125, 40.25, 40.375,
            40.5, 40.625, 40.75, 40.875, 41.0, 41.125, 41.25, 41.375, 41.5, 41.625, 41.75, 41.875,
            42.0, 42.125, 42.25, 42.375, 42.5, 42.625, 42.75, 42.875, 43.0, 43.125, 43.25, 43.375,
            43.5, 43.625, 43.75, 43.875, 44.0, 44.125, 44.25, 44.375, 44.5, 44.625, 44.75, 44.875,
            45.0, 45.125, 45.25, 45.375, 45.5, 45.625, 45.75, 45.875, 46.0, 46.125, 46.25, 46.375,
            46.5, 46.625, 46.75, 46.875, 47.0, 47.125, 47.25, 47.375, 47.5, 47.625, 47.75, 47.875,
            48.0, 48.125, 48.25, 48.375, 48.5, 48.625, 48.75, 48.875, 49.0, 49.125, 49.25, 49.375,
            49.5, 49.625, 49.75, 49.875, 50.0, 50.125, 50.25, 50.375, 50.5, 50.625, 50.75, 50.875,
            51.0, 51.125, 51.25, 51.375, 51.5, 51.625, 51.75, 51.875, 52.0, 52.125, 52.25, 52.375,
            52.5, 52.625, 52.75, 52.875, 53.0, 53.125, 53.25, 53.375, 53.5, 53.625, 53.75, 53.875,
            54.0, 54.125, 54.25, 54.375, 54.5, 54.625, 54.75, 54.875, 55.0, 55.125, 55.25, 55.375,
            55.5, 55.625, 55.75, 55.875, 56.0, 56.125, 56.25, 56.375, 56.5, 56.625, 56.75, 56.875,
            57.0, 57.125, 57.25, 57.375, 57.5, 57.625, 57.75, 57.875, 58.0, 58.125, 58.25, 58.375,
            58.5, 58.625, 58.75, 58.875, 59.0, 59.125, 59.25, 59.375, 59.5, 59.625, 59.75, 59.875,
            60.0, 60.125, 60.25, 60.375, 60.5, 60.625, 60.75, 60.875, 61.0, 61.125, 61.25, 61.375,
            61.5, 61.625, 61.75, 61.875, 62.0, 62.125, 62.25, 62.375, 62.5, 62.625, 62.75, 62.875,
            63.0, 63.125, 63.25, 63.375, 63.5, 63.625, 63.75, 63.875, 64.0, 64.125, 64.25, 64.375,
            64.5, 64.625, 64.75, 64.875, 65.0, 65.125, 65.25, 65.375, 65.5, 65.625, 65.75, 65.875,
            66.0, 66.125, 66.25, 66.375, 66.5, 66.625, 66.75, 66.875, 67.0, 67.125, 67.25, 67.375,
            67.5, 67.625, 67.75, 67.875, 68.0, 68.125, 68.25, 68.375, 68.5, 68.625, 68.75, 68.875,
            69.0, 69.125, 69.25, 69.375, 69.5, 69.625, 69.75, 69.875, 70.0, 70.125, 70.25, 70.375,
            70.5, 70.625, 70.75, 70.875, 71.0, 71.125, 71.25, 71.375, 71.5, 71.625, 71.75, 71.875,
            72.0, 72.125, 72.25, 72.375, 72.5, 72.625, 72.75, 72.875, 73.0, 73.125, 73.25, 73.375,
            73.5, 73.625, 73.75, 73.875, 74.0, 74.125, 74.25, 74.375, 74.5, 74.625, 74.75, 74.875,
            75.0, 75.125, 75.25, 75.375, 75.5, 75.625, 75.75, 75.875, 76.0, 76.125, 76.25, 76.375,
            76.5, 76.625, 76.75, 76.875, 77.0, 77.125, 77.25, 77.375, 77.5, 77.625, 77.75, 77.875,
            78.0, 78.125, 78.25, 78.375, 78.5, 78.625, 78.75, 78.875, 79.0, 79.125, 79.25, 79.375,
            79.5, 79.625, 79.75, 79.875, 80.0, 80.125, 80.25, 80.375, 80.5, 80.625, 80.75, 80.875,
            81.0, 81.125, 81.25, 81.375, 81.5, 81.625, 81.75, 81.875, 82.0, 82.125, 82.25, 82.375,
            82.5, 82.625, 82.75, 82.875, 83.0, 83.125, 83.25, 83.375, 83.5, 83.625, 83.75, 83.875,
            84.0, 84.125, 84.25, 84.375, 84.5, 84.625, 84.75, 84.875, 85.0, 85.125, 85.25, 85.375,
            85.5, 85.625, 85.75, 85.875, 86.0, 86.125, 86.25, 86.375, 86.5, 86.625, 86.75, 86.875,
            87.0, 87.125, 87.25, 87.375, 87.5, 87.625, 87.75, 87.875, 88.0, 88.125, 88.25, 88.375,
            88.5, 88.625, 88.75, 88.875, 89.0, 89.125, 89.25, 89.375, 89.5, 89.625, 89.75, 89.875,
            90.0, 90.125, 90.25, 90.375, 90.5, 90.625, 90.75, 90.875, 91.0, 91.125, 91.25, 91.375,
            91.5, 91.625, 91.75, 91.875, 92.0, 92.125, 92.25, 92.375, 92.5, 92.625, 92.75, 92.875,
            93.0, 93.125, 93.25, 93.375, 93.5, 93.625, 93.75, 93.875, 94.0, 94.125, 94.25, 94.375,
            94.5, 94.625, 94.75, 94.875, 95.0, 95.125, 95.25, 95.375, 95.5, 95.625, 95.75, 95.875,
            96.0,
        ];
        check(
            &mut failures,
            "to_array_1d(array[,,,,,,,,] real)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = to_array_2d(&x);
        let want = (0..2)
            .map(|i| (0..3).map(|j| x[(i, j)]).collect())
            .collect::<Vec<Vec<_>>>();
        check(&mut failures, "to_array_2d(matrix)", near(&got, &want));
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
        let got = to_array_2d(&x);
        let want = (0..2)
            .map(|i| (0..3).map(|j| x[(i, j)]).collect())
            .collect::<Vec<Vec<_>>>();
        check(
            &mut failures,
            "to_array_2d(complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = to_vector_array(&x);
        let want = (0..3)
            .map(|j| (0..2).map(|i| x[(i, j)]).collect())
            .collect::<Vec<Vec<_>>>();
        check(&mut failures, "to_vector_array(matrix)", near(&got, &want));
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
        let got = to_vector_array(&x);
        let want = (0..3)
            .map(|j| (0..2).map(|i| x[(i, j)]).collect())
            .collect::<Vec<Vec<_>>>();
        check(
            &mut failures,
            "to_vector_array(complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = to_row_vector_array(&x);
        let want = (0..2)
            .map(|i| (0..3).map(|j| x[(i, j)]).collect())
            .collect::<Vec<Vec<_>>>();
        check(
            &mut failures,
            "to_row_vector_array(matrix)",
            near(&got, &want),
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
        let got = to_row_vector_array(&x);
        let want = (0..2)
            .map(|i| (0..3).map(|j| x[(i, j)]).collect())
            .collect::<Vec<Vec<_>>>();
        check(
            &mut failures,
            "to_row_vector_array(complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_matrix_from_vector(&x);
        let want = Matrix::from_fn(3, 1, |i, j| x[i + j]);
        check(&mut failures, "to_matrix(vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix(&x, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(vector, int, int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(vector, int, int, int) col_major=1",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(vector, int, int, int) col_major=0",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let got = to_matrix_from_row_vector(&x);
        let want = Matrix::from_fn(1, 3, |i, j| x[i + j]);
        check(&mut failures, "to_matrix(row_vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix(&x, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(row_vector, int, int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(row_vector, int, int, int) col_major=1",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(row_vector, int, int, int) col_major=0",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let got = x.clone();
        let want = x.clone();
        check(&mut failures, "to_matrix(matrix)", near(&got, &want));
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let f = to_array_1d(&x);
        let got = to_matrix_reshape(&x, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(matrix, int, int)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let f = to_array_1d(&x);
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(matrix, int, int, int) col_major=1",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let f = to_array_1d(&x);
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(matrix, int, int, int) col_major=0",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_matrix_from_vector(&x);
        let want = Matrix::from_fn(3, 1, |i, j| x[i + j]);
        check(
            &mut failures,
            "to_matrix(complex_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix(&x, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(complex_vector, int, int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(complex_vector, int, int, int) col_major=1",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(complex_vector, int, int, int) col_major=0",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let got = to_matrix_from_row_vector(&x);
        let want = Matrix::from_fn(1, 3, |i, j| x[i + j]);
        check(
            &mut failures,
            "to_matrix(complex_row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix(&x, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(complex_row_vector, int, int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(complex_row_vector, int, int, int) col_major=1",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(complex_row_vector, int, int, int) col_major=0",
            near(&got, &want),
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
        let got = x.clone();
        let want = x.clone();
        check(
            &mut failures,
            "to_matrix(complex_matrix)",
            near(&got, &want),
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
        let f = to_array_1d(&x);
        let got = to_matrix_reshape(&x, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(complex_matrix, int, int)",
            near(&got, &want),
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
        let f = to_array_1d(&x);
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(complex_matrix, int, int, int) col_major=1",
            near(&got, &want),
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
        let f = to_array_1d(&x);
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(complex_matrix, int, int, int) col_major=0",
            near(&got, &want),
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64];
        let xs = to_vector_int::<f64>(&x);
        let f = xs.clone();
        let got = to_matrix(&xs, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(array[] int, int, int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64];
        let xs = to_vector_int::<f64>(&x);
        let f = xs.clone();
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(array[] int, int, int, int) col_major=1",
            near(&got, &want),
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64, 3_i64, 4_i64, 5_i64, 6_i64];
        let xs = to_vector_int::<f64>(&x);
        let f = xs.clone();
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(array[] int, int, int, int) col_major=0",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix(&x, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(array[] real, int, int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(array[] real, int, int, int) col_major=1",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75];
        let f = x.clone();
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(array[] real, int, int, int) col_major=0",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]];
        let got = to_matrix_from_vector_array(&x).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| x[j][i]);
        check(
            &mut failures,
            "to_matrix(array[] vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix(&x, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(array[] complex, int, int)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[i + 3 * j]);
        check(
            &mut failures,
            "to_matrix(array[] complex, int, int, int) col_major=1",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
            Complex::new(0.5, -0.25),
            Complex::new(0.625, -0.3125),
            Complex::new(0.75, -0.375),
        ];
        let f = x.clone();
        let got = to_matrix_row_major(&f, 3, 2).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| f[2 * i + j]);
        check(
            &mut failures,
            "to_matrix(array[] complex, int, int, int) col_major=0",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]];
        let got = to_matrix_from_array_2d(&x).unwrap();
        let want = Matrix::from_fn(2, 3, |i, j| x[i][j]);
        check(
            &mut failures,
            "to_matrix(array[] row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
            ],
            vec![
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        ];
        let got = to_matrix_from_vector_array(&x).unwrap();
        let want = Matrix::from_fn(3, 2, |i, j| x[j][i]);
        check(
            &mut failures,
            "to_matrix(array[] complex_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
            ],
            vec![
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        ];
        let got = to_matrix_from_array_2d(&x).unwrap();
        let want = Matrix::from_fn(2, 3, |i, j| x[i][j]);
        check(
            &mut failures,
            "to_matrix(array[] complex_row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64, 3_i64], vec![4_i64, 5_i64, 6_i64]];
        let got = to_matrix_from_int_array_2d::<f64>(&x).unwrap();
        let want = Matrix::from_fn(2, 3, |i, j| x[i][j] as f64);
        check(&mut failures, "to_matrix(array[,] int)", near(&got, &want));
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.125, 0.25, 0.375], vec![0.5, 0.625, 0.75]];
        let got = to_matrix_from_array_2d(&x).unwrap();
        let want = Matrix::from_fn(2, 3, |i, j| x[i][j]);
        check(&mut failures, "to_matrix(array[,] real)", near(&got, &want));
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![
                Complex::new(0.125, -0.0625),
                Complex::new(0.25, -0.125),
                Complex::new(0.375, -0.1875),
            ],
            vec![
                Complex::new(0.5, -0.25),
                Complex::new(0.625, -0.3125),
                Complex::new(0.75, -0.375),
            ],
        ];
        let got = to_matrix_from_array_2d(&x).unwrap();
        let want = Matrix::from_fn(2, 3, |i, j| x[i][j]);
        check(
            &mut failures,
            "to_matrix(array[,] complex)",
            near(&got, &want),
        );
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = append_array(&[x], &y);
        let want = [vec![x], y.clone()].concat();
        check(
            &mut failures,
            "append_col(real, row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let (ma, mb) = (
            Matrix::from_column_major(x.len(), 1, x.clone()),
            Matrix::from_column_major(y.len(), 1, y.clone()),
        );
        let got = append_col_vectors(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| {
            if j < ma.ncols() {
                ma[(i, j)]
            } else {
                mb[(i, j - ma.ncols())]
            }
        });
        check(
            &mut failures,
            "append_col(vector, vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let x = x[..2].to_vec();
        let (ma, mb) = (Matrix::from_column_major(x.len(), 1, x.clone()), y.clone());
        let got = append_col_vector_matrix(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| {
            if j < ma.ncols() {
                ma[(i, j)]
            } else {
                mb[(i, j - ma.ncols())]
            }
        });
        check(
            &mut failures,
            "append_col(vector, matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = append_array(&[x], &y);
        let want = [vec![x], y.clone()].concat();
        check(
            &mut failures,
            "append_col(complex, complex_row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = append_array(&x, &[y]);
        let want = [x.clone(), vec![y]].concat();
        check(
            &mut failures,
            "append_col(row_vector, real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = append_array(&x, &y);
        let want = [x.clone(), y.clone()].concat();
        check(
            &mut failures,
            "append_col(row_vector, row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let y = y[..2].to_vec();
        let (ma, mb) = (x.clone(), Matrix::from_column_major(y.len(), 1, y.clone()));
        let got = append_col_matrix_vector(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| {
            if j < ma.ncols() {
                ma[(i, j)]
            } else {
                mb[(i, j - ma.ncols())]
            }
        });
        check(
            &mut failures,
            "append_col(matrix, vector)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let (ma, mb) = (x.clone(), y.clone());
        let got = append_col(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| {
            if j < ma.ncols() {
                ma[(i, j)]
            } else {
                mb[(i, j - ma.ncols())]
            }
        });
        check(
            &mut failures,
            "append_col(matrix, matrix)",
            near(&got, &want),
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
        let (ma, mb) = (
            Matrix::from_column_major(x.len(), 1, x.clone()),
            Matrix::from_column_major(y.len(), 1, y.clone()),
        );
        let got = append_col_vectors(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| {
            if j < ma.ncols() {
                ma[(i, j)]
            } else {
                mb[(i, j - ma.ncols())]
            }
        });
        check(
            &mut failures,
            "append_col(complex_vector, complex_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
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
        let x = x[..2].to_vec();
        let (ma, mb) = (Matrix::from_column_major(x.len(), 1, x.clone()), y.clone());
        let got = append_col_vector_matrix(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| {
            if j < ma.ncols() {
                ma[(i, j)]
            } else {
                mb[(i, j - ma.ncols())]
            }
        });
        check(
            &mut failures,
            "append_col(complex_vector, complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = append_array(&x, &[y]);
        let want = [x.clone(), vec![y]].concat();
        check(
            &mut failures,
            "append_col(complex_row_vector, complex)",
            near(&got, &want),
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
        let got = append_array(&x, &y);
        let want = [x.clone(), y.clone()].concat();
        check(
            &mut failures,
            "append_col(complex_row_vector, complex_row_vector)",
            near(&got, &want),
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
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let y = y[..2].to_vec();
        let (ma, mb) = (x.clone(), Matrix::from_column_major(y.len(), 1, y.clone()));
        let got = append_col_matrix_vector(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| {
            if j < ma.ncols() {
                ma[(i, j)]
            } else {
                mb[(i, j - ma.ncols())]
            }
        });
        check(
            &mut failures,
            "append_col(complex_matrix, complex_vector)",
            near(&got, &want),
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
        let (ma, mb) = (x.clone(), y.clone());
        let got = append_col(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows(), ma.ncols() + mb.ncols(), |i, j| {
            if j < ma.ncols() {
                ma[(i, j)]
            } else {
                mb[(i, j - ma.ncols())]
            }
        });
        check(
            &mut failures,
            "append_col(complex_matrix, complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = append_array(&[x], &y);
        let want = [vec![x], y.clone()].concat();
        check(&mut failures, "append_row(real, vector)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let got = append_array(&x, &[y]);
        let want = [x.clone(), vec![y]].concat();
        check(&mut failures, "append_row(vector, real)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = append_array(&x, &y);
        let want = [x.clone(), y.clone()].concat();
        check(
            &mut failures,
            "append_row(vector, vector)",
            near(&got, &want),
        );
    }
    {
        let x: Complex<f64> = Complex::new(0.125, -0.0625);
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let got = append_array(&[x], &y);
        let want = [vec![x], y.clone()].concat();
        check(
            &mut failures,
            "append_row(complex, complex_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let (ma, mb) = (
            Matrix::from_column_major(1, x.len(), x.clone()),
            Matrix::from_column_major(1, y.len(), y.clone()),
        );
        let got = append_row_row_vectors(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| {
            if i < ma.nrows() {
                ma[(i, j)]
            } else {
                mb[(i - ma.nrows(), j)]
            }
        });
        check(
            &mut failures,
            "append_row(row_vector, row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let (ma, mb) = (Matrix::from_column_major(1, x.len(), x.clone()), y.clone());
        let got = append_row_row_vector_matrix(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| {
            if i < ma.nrows() {
                ma[(i, j)]
            } else {
                mb[(i - ma.nrows(), j)]
            }
        });
        check(
            &mut failures,
            "append_row(row_vector, matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let (ma, mb) = (x.clone(), Matrix::from_column_major(1, y.len(), y.clone()));
        let got = append_row_matrix_row_vector(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| {
            if i < ma.nrows() {
                ma[(i, j)]
            } else {
                mb[(i - ma.nrows(), j)]
            }
        });
        check(
            &mut failures,
            "append_row(matrix, row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let (ma, mb) = (x.clone(), y.clone());
        let got = append_row(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| {
            if i < ma.nrows() {
                ma[(i, j)]
            } else {
                mb[(i - ma.nrows(), j)]
            }
        });
        check(
            &mut failures,
            "append_row(matrix, matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
        let y: Complex<f64> = Complex::new(1.375, -0.6875);
        let got = append_array(&x, &[y]);
        let want = [x.clone(), vec![y]].concat();
        check(
            &mut failures,
            "append_row(complex_vector, complex)",
            near(&got, &want),
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
        let got = append_array(&x, &y);
        let want = [x.clone(), y.clone()].concat();
        check(
            &mut failures,
            "append_row(complex_vector, complex_vector)",
            near(&got, &want),
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
        let (ma, mb) = (
            Matrix::from_column_major(1, x.len(), x.clone()),
            Matrix::from_column_major(1, y.len(), y.clone()),
        );
        let got = append_row_row_vectors(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| {
            if i < ma.nrows() {
                ma[(i, j)]
            } else {
                mb[(i - ma.nrows(), j)]
            }
        });
        check(
            &mut failures,
            "append_row(complex_row_vector, complex_row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![
            Complex::new(0.125, -0.0625),
            Complex::new(0.25, -0.125),
            Complex::new(0.375, -0.1875),
        ];
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
        let (ma, mb) = (Matrix::from_column_major(1, x.len(), x.clone()), y.clone());
        let got = append_row_row_vector_matrix(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| {
            if i < ma.nrows() {
                ma[(i, j)]
            } else {
                mb[(i - ma.nrows(), j)]
            }
        });
        check(
            &mut failures,
            "append_row(complex_row_vector, complex_matrix)",
            near(&got, &want),
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
        let y: Vec<Complex<f64>> = vec![
            Complex::new(1.375, -0.6875),
            Complex::new(1.5, -0.75),
            Complex::new(1.625, -0.8125),
        ];
        let (ma, mb) = (x.clone(), Matrix::from_column_major(1, y.len(), y.clone()));
        let got = append_row_matrix_row_vector(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| {
            if i < ma.nrows() {
                ma[(i, j)]
            } else {
                mb[(i - ma.nrows(), j)]
            }
        });
        check(
            &mut failures,
            "append_row(complex_matrix, complex_row_vector)",
            near(&got, &want),
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
        let (ma, mb) = (x.clone(), y.clone());
        let got = append_row(&x, &y).unwrap();
        let want = Matrix::from_fn(ma.nrows() + mb.nrows(), ma.ncols(), |i, j| {
            if i < ma.nrows() {
                ma[(i, j)]
            } else {
                mb[(i - ma.nrows(), j)]
            }
        });
        check(
            &mut failures,
            "append_row(complex_matrix, complex_matrix)",
            near(&got, &want),
        );
    }
    {
        let x: f64 = 0.125;
        let y: f64 = 1.375;
        let z: f64 = 2.625;
        let got = fma(x, y, z);
        let want = x * y + z;
        check(&mut failures, "fma(real, real, real)", close(&got, &want));
    }
    {
        let x: f64 = 0.125;
        let y: f64 = 1.375;
        let z: Vec<f64> = vec![2.625, 2.75, 2.875];
        let zs = flat(&z);
        let got = flat(&apply_scalar_unary(&z, |z_| fma(x, y, z_)));
        let want = (0..zs.len())
            .map(|i| x.mul_add(y, zs[i]))
            .collect::<Vec<f64>>();
        check(&mut failures, "fma(real, real, vector)", near(&got, &want));
    }
    {
        let x: f64 = 0.125;
        let y: f64 = 1.375;
        let z: Vec<f64> = vec![2.625, 2.75, 2.875];
        let zs = flat(&z);
        let got = flat(&apply_scalar_unary(&z, |z_| fma(x, y, z_)));
        let want = (0..zs.len())
            .map(|i| x.mul_add(y, zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(real, real, row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: f64 = 0.125;
        let y: f64 = 1.375;
        let z: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![2.625, 2.75, 2.875, 3.0, 3.125, 3.25]);
        let zs = flat(&z);
        let got = flat(&apply_scalar_unary(&z, |z_| fma(x, y, z_)));
        let want = (0..zs.len())
            .map(|i| x.mul_add(y, zs[i]))
            .collect::<Vec<f64>>();
        check(&mut failures, "fma(real, real, matrix)", near(&got, &want));
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z: f64 = 2.625;
        let ys = flat(&y);
        let got = flat(&apply_scalar_unary(&y, |y_| fma(x, y_, z)));
        let want = (0..ys.len())
            .map(|i| x.mul_add(ys[i], z))
            .collect::<Vec<f64>>();
        check(&mut failures, "fma(real, vector, real)", near(&got, &want));
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z: Vec<f64> = vec![2.625, 2.75, 2.875];
        let ys = flat(&y);
        let zs = flat(&z);
        let got = flat(&apply_scalar_binary(&y, &z, |y_, z_| fma(x, y_, z_)).unwrap());
        let want = (0..ys.len())
            .map(|i| x.mul_add(ys[i], zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(real, vector, vector)",
            near(&got, &want),
        );
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z: f64 = 2.625;
        let ys = flat(&y);
        let got = flat(&apply_scalar_unary(&y, |y_| fma(x, y_, z)));
        let want = (0..ys.len())
            .map(|i| x.mul_add(ys[i], z))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(real, row_vector, real)",
            near(&got, &want),
        );
    }
    {
        let x: f64 = 0.125;
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z: Vec<f64> = vec![2.625, 2.75, 2.875];
        let ys = flat(&y);
        let zs = flat(&z);
        let got = flat(&apply_scalar_binary(&y, &z, |y_, z_| fma(x, y_, z_)).unwrap());
        let want = (0..ys.len())
            .map(|i| x.mul_add(ys[i], zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(real, row_vector, row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: f64 = 0.125;
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let z: f64 = 2.625;
        let ys = flat(&y);
        let got = flat(&apply_scalar_unary(&y, |y_| fma(x, y_, z)));
        let want = (0..ys.len())
            .map(|i| x.mul_add(ys[i], z))
            .collect::<Vec<f64>>();
        check(&mut failures, "fma(real, matrix, real)", near(&got, &want));
    }
    {
        let x: f64 = 0.125;
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let z: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![2.625, 2.75, 2.875, 3.0, 3.125, 3.25]);
        let ys = flat(&y);
        let zs = flat(&z);
        let got = flat(&apply_scalar_binary(&y, &z, |y_, z_| fma(x, y_, z_)).unwrap());
        let want = (0..ys.len())
            .map(|i| x.mul_add(ys[i], zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(real, matrix, matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let z: f64 = 2.625;
        let xs = flat(&x);
        let got = flat(&apply_scalar_unary(&x, |x_| fma(x_, y, z)));
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(y, z))
            .collect::<Vec<f64>>();
        check(&mut failures, "fma(vector, real, real)", near(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let z: Vec<f64> = vec![2.625, 2.75, 2.875];
        let xs = flat(&x);
        let zs = flat(&z);
        let got = flat(&apply_scalar_binary(&x, &z, |x_, z_| fma(x_, y, z_)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(y, zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(vector, real, vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z: f64 = 2.625;
        let xs = flat(&x);
        let ys = flat(&y);
        let got = flat(&apply_scalar_binary(&x, &y, |x_, y_| fma(x_, y_, z)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(ys[i], z))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(vector, vector, real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z: Vec<f64> = vec![2.625, 2.75, 2.875];
        let xs = flat(&x);
        let ys = flat(&y);
        let zs = flat(&z);
        let got = flat(&apply_scalar_ternary(&x, &y, &z, |x_, y_, z_| fma(x_, y_, z_)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(ys[i], zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(vector, vector, vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let z: f64 = 2.625;
        let xs = flat(&x);
        let got = flat(&apply_scalar_unary(&x, |x_| fma(x_, y, z)));
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(y, z))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(row_vector, real, real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: f64 = 1.375;
        let z: Vec<f64> = vec![2.625, 2.75, 2.875];
        let xs = flat(&x);
        let zs = flat(&z);
        let got = flat(&apply_scalar_binary(&x, &z, |x_, z_| fma(x_, y, z_)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(y, zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(row_vector, real, row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z: f64 = 2.625;
        let xs = flat(&x);
        let ys = flat(&y);
        let got = flat(&apply_scalar_binary(&x, &y, |x_, y_| fma(x_, y_, z)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(ys[i], z))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(row_vector, row_vector, real)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z: Vec<f64> = vec![2.625, 2.75, 2.875];
        let xs = flat(&x);
        let ys = flat(&y);
        let zs = flat(&z);
        let got = flat(&apply_scalar_ternary(&x, &y, &z, |x_, y_, z_| fma(x_, y_, z_)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(ys[i], zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(row_vector, row_vector, row_vector)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: f64 = 1.375;
        let z: f64 = 2.625;
        let xs = flat(&x);
        let got = flat(&apply_scalar_unary(&x, |x_| fma(x_, y, z)));
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(y, z))
            .collect::<Vec<f64>>();
        check(&mut failures, "fma(matrix, real, real)", near(&got, &want));
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: f64 = 1.375;
        let z: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![2.625, 2.75, 2.875, 3.0, 3.125, 3.25]);
        let xs = flat(&x);
        let zs = flat(&z);
        let got = flat(&apply_scalar_binary(&x, &z, |x_, z_| fma(x_, y, z_)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(y, zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(matrix, real, matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let z: f64 = 2.625;
        let xs = flat(&x);
        let ys = flat(&y);
        let got = flat(&apply_scalar_binary(&x, &y, |x_, y_| fma(x_, y_, z)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(ys[i], z))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(matrix, matrix, real)",
            near(&got, &want),
        );
    }
    {
        let x: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![0.125, 0.25, 0.375, 0.5, 0.625, 0.75]);
        let y: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![1.375, 1.5, 1.625, 1.75, 1.875, 2.0]);
        let z: Matrix<f64> =
            Matrix::from_column_major(2, 3, vec![2.625, 2.75, 2.875, 3.0, 3.125, 3.25]);
        let xs = flat(&x);
        let ys = flat(&y);
        let zs = flat(&z);
        let got = flat(&apply_scalar_ternary(&x, &y, &z, |x_, y_, z_| fma(x_, y_, z_)).unwrap());
        let want = (0..xs.len())
            .map(|i| xs[i].mul_add(ys[i], zs[i]))
            .collect::<Vec<f64>>();
        check(
            &mut failures,
            "fma(matrix, matrix, matrix)",
            near(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(&mut failures, "log_mix(vector, vector)", close(&got, &want));
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(
            &mut failures,
            "log_mix(vector, row_vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(
            &mut failures,
            "log_mix(vector, array[] real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<Vec<f64>> = vec![vec![1.375, 1.5, 1.625], vec![1.75, 1.875, 2.0]];
        let got = log_mix_array(&x, &y);
        let want = log_mix_definition(&x, &y);
        check(
            &mut failures,
            "log_mix(vector, array[] vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<Vec<f64>> = vec![vec![1.375, 1.5, 1.625], vec![1.75, 1.875, 2.0]];
        let got = log_mix_array(&x, &y);
        let want = log_mix_definition(&x, &y);
        check(
            &mut failures,
            "log_mix(vector, array[] row_vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(
            &mut failures,
            "log_mix(row_vector, vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(
            &mut failures,
            "log_mix(row_vector, row_vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(
            &mut failures,
            "log_mix(row_vector, array[] real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<Vec<f64>> = vec![vec![1.375, 1.5, 1.625], vec![1.75, 1.875, 2.0]];
        let got = log_mix_array(&x, &y);
        let want = log_mix_definition(&x, &y);
        check(
            &mut failures,
            "log_mix(row_vector, array[] vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<Vec<f64>> = vec![vec![1.375, 1.5, 1.625], vec![1.75, 1.875, 2.0]];
        let got = log_mix_array(&x, &y);
        let want = log_mix_definition(&x, &y);
        check(
            &mut failures,
            "log_mix(row_vector, array[] row_vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(
            &mut failures,
            "log_mix(array[] real, vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(
            &mut failures,
            "log_mix(array[] real, row_vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let got = log_mix_slice(&x, &y);
        let want = log_mix_definition(&x, std::slice::from_ref(&y));
        check(
            &mut failures,
            "log_mix(array[] real, array[] real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<Vec<f64>> = vec![vec![1.375, 1.5, 1.625], vec![1.75, 1.875, 2.0]];
        let got = log_mix_array(&x, &y);
        let want = log_mix_definition(&x, &y);
        check(
            &mut failures,
            "log_mix(array[] real, array[] vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<Vec<f64>> = vec![vec![1.375, 1.5, 1.625], vec![1.75, 1.875, 2.0]];
        let got = log_mix_array(&x, &y);
        let want = log_mix_definition(&x, &y);
        check(
            &mut failures,
            "log_mix(array[] real, array[] row_vector)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(vector, vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(vector, row_vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(vector, array[] real, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(row_vector, vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(row_vector, row_vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(row_vector, array[] real, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(array[] real, vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(array[] real, row_vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_3F2([x[0], x[1], x[2]], [y[0], y[1]], z);
        let want = pfq_series(&x, &y[..2], z);
        check(
            &mut failures,
            "hypergeometric_3F2(array[] real, array[] real, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(vector, vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(vector, row_vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(vector, array[] real, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(row_vector, vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(row_vector, row_vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(row_vector, array[] real, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(array[] real, vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(array[] real, row_vector, real)",
            close(&got, &want),
        );
    }
    {
        let x: Vec<f64> = vec![0.125, 0.25, 0.375];
        let y: Vec<f64> = vec![1.375, 1.5, 1.625];
        let z = 0.3;
        let got = hypergeometric_pFq(&x, &y, z);
        let want = pfq_series(&x, &y, z);
        check(
            &mut failures,
            "hypergeometric_pFq(array[] real, array[] real, real)",
            close(&got, &want),
        );
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
