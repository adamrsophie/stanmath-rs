//! Every vectorized form of a scalar function of one or two arguments in the Stan language,
//! called through `apply_scalar_unary` and `apply_scalar_binary`.
//!
//! Written by `tools/reference/gen_elementwise.py` from the CmdStan 2.40.0 signature dump: 2563
//! Stan signatures of 94 functions in 1490 Rust shapes. Each call must keep the container's
//! shape and return the scalar function of every element; the scalar functions are compared with
//! Stan Math by `stan_reference_scalar.rs`. The other 3345 signatures nest arrays more than 3
//! `Vec` levels deep; they go through the same recursive impl, which `deep_arrays` checks to depth 8.
//! `add`, `subtract`, `multiply`, `elt_multiply`, and `elt_divide` are Rust's operators.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::unnecessary_cast,
    clippy::approx_constant,
    clippy::redundant_closure,
    clippy::type_complexity
)]

use stanmath_rs::prelude::*;

/// The scalar functions return a value or a `Result`; a rejected argument becomes NaN.
trait Eval {
    fn eval(self) -> f64;
}

impl Eval for f64 {
    fn eval(self) -> f64 {
        self
    }
}

impl Eval for i64 {
    fn eval(self) -> f64 {
        self as f64
    }
}

impl Eval for i32 {
    fn eval(self) -> f64 {
        f64::from(self)
    }
}

impl<E: Eval> Eval for Result<E> {
    fn eval(self) -> f64 {
        self.map_or(f64::NAN, Eval::eval)
    }
}

fn eval(v: impl Eval) -> f64 {
    v.eval()
}

/// The scalars of a container in order.
fn leaves<C: ScalarContainer<Scalar = f64>>(c: &C) -> Vec<f64> {
    let mut out = Vec::new();
    apply_scalar_unary(c, |v| out.push(v));
    out
}

fn compare(label: &str, got: Vec<f64>, want: Vec<f64>, failures: &mut Vec<String>) {
    let same = got.len() == want.len()
        && got
            .iter()
            .zip(&want)
            .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()));
    if !same {
        failures.push(format!("{label}: {got:?} vs {want:?}"));
    }
}

#[test]
fn elementwise_signatures() {
    let mut failures = Vec::new();
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare(
            "Phi(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare(
            "Phi(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare(
            "Phi(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare(
            "Phi(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare(
            "Phi(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare(
            "Phi(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare(
            "Phi(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare(
            "Phi(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare("Phi(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi(v as f64)));
        compare("Phi(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(Phi_approx(v as f64)));
        compare(
            "Phi_approx(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare(
            "abs(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare(
            "abs(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare(
            "abs(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare(
            "abs(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare(
            "abs(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare(
            "abs(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare(
            "abs(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare(
            "abs(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare("abs(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(abs(v as f64)));
        compare("abs(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare(
            "acos(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare(
            "acos(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare(
            "acos(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare(
            "acos(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare(
            "acos(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare(
            "acos(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare(
            "acos(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare(
            "acos(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare("acos(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acos(v as f64)));
        compare("acos(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.751, 1.802, 1.853, 1.904]);
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.751, 1.802, 1.853, 1.904]),
            Matrix::from_column_major(2, 2, vec![1.955, 2.006, 2.057, 2.108]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.751, 1.802, 1.853, 1.904]),
                Matrix::from_column_major(2, 2, vec![1.955, 2.006, 2.057, 2.108]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.159, 2.21, 2.261, 2.312]),
                Matrix::from_column_major(2, 2, vec![2.363, 2.414, 2.465, 2.516]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.751, 1.802, 1.853, 1.904]),
                    Matrix::from_column_major(2, 2, vec![1.955, 2.006, 2.057, 2.108]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.159, 2.21, 2.261, 2.312]),
                    Matrix::from_column_major(2, 2, vec![2.363, 2.414, 2.465, 2.516]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.567, 2.618, 2.669, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.771, 2.822, 2.873, 2.924]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.975, 3.026, 3.077, 3.128]),
                    Matrix::from_column_major(2, 2, vec![3.179, 3.23, 3.281, 3.332]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.751, 1.802], vec![1.853, 1.904]],
            vec![vec![1.955, 2.006], vec![2.057, 2.108]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.751, 1.802], vec![1.853, 1.904]];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![1.751, 1.802];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![2_i64, 2_i64];
        let got = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(acosh(v as f64)));
        compare(
            "acosh(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare(
            "asin(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare(
            "asin(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare(
            "asin(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare(
            "asin(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare(
            "asin(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare(
            "asin(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare(
            "asin(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare(
            "asin(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare("asin(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asin(v as f64)));
        compare("asin(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(asinh(v as f64)));
        compare(
            "asinh(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare(
            "atan(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare(
            "atan(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare(
            "atan(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare(
            "atan(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare(
            "atan(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare(
            "atan(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare(
            "atan(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare(
            "atan(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare("atan(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atan(v as f64)));
        compare("atan(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(atan2(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(atan2(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "atan2(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(atanh(v as f64)));
        compare(
            "atanh(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "beta(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(beta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(beta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "beta(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare(
            "cbrt(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare(
            "cbrt(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare(
            "cbrt(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare(
            "cbrt(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare(
            "cbrt(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare(
            "cbrt(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare(
            "cbrt(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare(
            "cbrt(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare("cbrt(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cbrt(v as f64)));
        compare("cbrt(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare(
            "ceil(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare(
            "ceil(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare(
            "ceil(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare(
            "ceil(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare(
            "ceil(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare(
            "ceil(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare(
            "ceil(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare(
            "ceil(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare("ceil(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(ceil(v as f64)));
        compare("ceil(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(chi_square_cdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "chi_square_cdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(chi_square_cdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(chi_square_cdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "chi_square_cdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(chi_square_cdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "chi_square_cdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(chi_square_lccdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "chi_square_lccdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(chi_square_lccdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(chi_square_lccdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "chi_square_lccdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(chi_square_lccdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "chi_square_lccdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(chi_square_lcdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "chi_square_lcdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(chi_square_lcdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(chi_square_lcdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "chi_square_lcdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(chi_square_lcdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "chi_square_lcdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(chi_square_lpdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "chi_square_lpdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(chi_square_lpdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(chi_square_lpdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "chi_square_lpdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(chi_square_lpdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "chi_square_lpdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(choose(a as _, b as _))).unwrap();
        let n = leaves(&got).len();
        compare(
            "choose(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(choose(a as _, b as _))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
        ];
        let b: i64 = 2_i64;
        let got = apply_scalar_unary(&a, |a| eval(choose(a as _, b as _)));
        let n = leaves(&got).len();
        compare(
            "choose(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]];
        let b: Vec<Vec<i64>> = vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(choose(a as _, b as _))).unwrap();
        let n = leaves(&got).len();
        compare(
            "choose(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(choose(a as _, b as _))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]];
        let b: i64 = 2_i64;
        let got = apply_scalar_unary(&a, |a| eval(choose(a as _, b as _)));
        let n = leaves(&got).len();
        compare(
            "choose(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![2_i64, 2_i64];
        let b: Vec<i64> = vec![2_i64, 2_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(choose(a as _, b as _))).unwrap();
        let n = leaves(&got).len();
        compare(
            "choose(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(choose(a as _, b as _))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![2_i64, 2_i64];
        let b: i64 = 2_i64;
        let got = apply_scalar_unary(&a, |a| eval(choose(a as _, b as _)));
        let n = leaves(&got).len();
        compare(
            "choose(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 2_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
            vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(choose(a as _, b as _)));
        let n = leaves(&got).len();
        compare(
            "choose(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 2_i64;
        let b: Vec<Vec<i64>> = vec![vec![2_i64, 2_i64], vec![2_i64, 2_i64]];
        let got = apply_scalar_unary(&b, |b| eval(choose(a as _, b as _)));
        let n = leaves(&got).len();
        compare(
            "choose(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 2_i64;
        let b: Vec<i64> = vec![2_i64, 2_i64];
        let got = apply_scalar_unary(&b, |b| eval(choose(a as _, b as _)));
        let n = leaves(&got).len();
        compare(
            "choose(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare(
            "cos(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare(
            "cos(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare(
            "cos(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare(
            "cos(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare(
            "cos(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare(
            "cos(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare(
            "cos(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare(
            "cos(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare("cos(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cos(v as f64)));
        compare("cos(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare(
            "cosh(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare(
            "cosh(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare(
            "cosh(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare(
            "cosh(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare(
            "cosh(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare(
            "cosh(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare(
            "cosh(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare(
            "cosh(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare("cosh(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(cosh(v as f64)));
        compare("cosh(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(digamma(v as f64)));
        compare(
            "digamma(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(divide(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "divide(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(divide(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "divide(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(divide(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "divide(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(divide(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "divide(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare(
            "erf(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare(
            "erf(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare(
            "erf(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare(
            "erf(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare(
            "erf(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare(
            "erf(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare(
            "erf(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare(
            "erf(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare("erf(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erf(v as f64)));
        compare("erf(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare(
            "erfc(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare(
            "erfc(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare(
            "erfc(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare(
            "erfc(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare(
            "erfc(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare(
            "erfc(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare(
            "erfc(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare(
            "erfc(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare("erfc(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(erfc(v as f64)));
        compare("erfc(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare(
            "exp(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare(
            "exp(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare(
            "exp(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare(
            "exp(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare(
            "exp(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare(
            "exp(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare(
            "exp(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare(
            "exp(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare("exp(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp(v as f64)));
        compare("exp(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare(
            "exp2(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare(
            "exp2(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare(
            "exp2(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare(
            "exp2(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare(
            "exp2(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare(
            "exp2(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare(
            "exp2(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare(
            "exp2(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare("exp2(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(exp2(v as f64)));
        compare("exp2(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(expm1(v as f64)));
        compare(
            "expm1(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(exponential_cdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "exponential_cdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(exponential_cdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(exponential_cdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "exponential_cdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(exponential_cdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "exponential_cdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(exponential_lccdf(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "exponential_lccdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(exponential_lccdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(exponential_lccdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "exponential_lccdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(exponential_lccdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "exponential_lccdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(exponential_lcdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "exponential_lcdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(exponential_lcdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(exponential_lcdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "exponential_lcdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(exponential_lcdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "exponential_lcdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(exponential_lpdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "exponential_lpdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(exponential_lpdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(exponential_lpdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "exponential_lpdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(exponential_lpdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "exponential_lpdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fdim(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(fdim(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fdim(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(floor(v as f64)));
        compare(
            "floor(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmax(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(fmax(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmax(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmin(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(fmin(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmin(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(fmod(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(fmod(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "fmod(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_p(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_p(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(gamma_p(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_p(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_q(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(gamma_q(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(gamma_q(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "gamma_q(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(hypot(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(hypot(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "hypot(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare(
            "inv(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare(
            "inv(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare(
            "inv(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare(
            "inv(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare(
            "inv(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare(
            "inv(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare(
            "inv(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare(
            "inv(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare("inv(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv(v as f64)));
        compare("inv(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_Phi(v as f64)));
        compare(
            "inv_Phi(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(inv_chi_square_cdf(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_cdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(inv_chi_square_cdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(inv_chi_square_cdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_cdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(inv_chi_square_cdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_cdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(inv_chi_square_lccdf(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lccdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(inv_chi_square_lccdf(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(inv_chi_square_lccdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lccdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(inv_chi_square_lccdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lccdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(inv_chi_square_lcdf(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lcdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(inv_chi_square_lcdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(inv_chi_square_lcdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lcdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(inv_chi_square_lcdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lcdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(inv_chi_square_lpdf(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lpdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(inv_chi_square_lpdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(inv_chi_square_lpdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lpdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(inv_chi_square_lpdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "inv_chi_square_lpdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_cloglog(v as f64)));
        compare(
            "inv_cloglog(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_erfc(v as f64)));
        compare(
            "inv_erfc(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_logit(v as f64)));
        compare(
            "inv_logit(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_sqrt(v as f64)));
        compare(
            "inv_sqrt(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(inv_square(v as f64)));
        compare(
            "inv_square(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_w0(v as f64)));
        compare(
            "lambert_w0(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-0.206, -0.212, -0.218, -0.224]);
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-0.206, -0.212, -0.218, -0.224]),
            Matrix::from_column_major(2, 2, vec![-0.23, -0.236, -0.242, -0.248]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-0.206, -0.212, -0.218, -0.224]),
                Matrix::from_column_major(2, 2, vec![-0.23, -0.236, -0.242, -0.248]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-0.254, -0.26, -0.266, -0.272]),
                Matrix::from_column_major(2, 2, vec![-0.278, -0.284, -0.29, -0.296]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-0.206, -0.212, -0.218, -0.224]),
                    Matrix::from_column_major(2, 2, vec![-0.23, -0.236, -0.242, -0.248]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-0.254, -0.26, -0.266, -0.272]),
                    Matrix::from_column_major(2, 2, vec![-0.278, -0.284, -0.29, -0.296]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-0.302, -0.308, -0.314, -0.32]),
                    Matrix::from_column_major(2, 2, vec![-0.326, -0.332, -0.338, -0.344]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-0.35, -0.356, -0.362, -0.368]),
                    Matrix::from_column_major(2, 2, vec![-0.374, -0.38, -0.386, -0.392]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-0.206, -0.212], vec![-0.218, -0.224]],
            vec![vec![-0.23, -0.236], vec![-0.242, -0.248]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![-0.206, -0.212], vec![-0.218, -0.224]];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![-0.206, -0.212];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lambert_wm1(v as f64)));
        compare(
            "lambert_wm1(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lbeta(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(lbeta(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lbeta(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(lchoose(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(lchoose(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lchoose(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(lgamma(v as f64)));
        compare(
            "lgamma(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(lmultiply(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(lmultiply(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "lmultiply(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare(
            "log(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare(
            "log(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare(
            "log(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare(
            "log(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare(
            "log(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare(
            "log(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare(
            "log(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare(
            "log(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare("log(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log(v as f64)));
        compare("log(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m(v as f64)));
        compare(
            "log1m(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_exp(v as f64)));
        compare(
            "log1m_exp(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1m_inv_logit(v as f64)));
        compare(
            "log1m_inv_logit(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p(v as f64)));
        compare(
            "log1p(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log1p_exp(v as f64)));
        compare(
            "log1p_exp(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_diff_exp(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_diff_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_diff_exp(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_falling_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_falling_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_falling_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_falling_factorial(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(log_inv_logit(v as f64)));
        compare(
            "log_inv_logit(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
            .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_inv_logit_diff(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_inv_logit_diff(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_inv_logit_diff(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_modified_bessel_first_kind(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| {
            eval(log_modified_bessel_first_kind(a as f64, b as f64))
        });
        let n = leaves(&got).len();
        compare(
            "log_modified_bessel_first_kind(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| {
            eval(log_rising_factorial(a as f64, b as f64))
        })
        .unwrap();
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| {
                    eval(log_rising_factorial(a as f64, b as f64))
                })
                .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_rising_factorial(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_rising_factorial(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(log_sum_exp(a as f64, b as f64))).unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(log_sum_exp(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "log_sum_exp(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(logit(v as f64)));
        compare(
            "logit(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(minus(v as f64)));
        compare(
            "minus(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "owens_t(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(owens_t(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(owens_t(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "owens_t(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(plus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(plus(v as f64)));
        compare(
            "plus(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(plus(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(plus(v as f64)));
        compare("plus(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Matrix<f64>, Matrix<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Matrix<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Matrix<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Matrix<f64>>, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Matrix<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Matrix<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Matrix<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Matrix<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<Matrix<f64>>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<Matrix<f64>>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<f64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<f64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<i64>>>, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<i64>>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<Vec<i64>>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<f64>>, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<f64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<f64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<i64>>, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<i64>>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<Vec<i64>>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<f64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "pow(Vec<i64>, Vec<i64>)",
            leaves(&got),
            leaves(&apply_scalar_binary(&a, &b, |a, b| eval(pow(a as f64, b as f64))).unwrap()),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<i64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<i64> = vec![1_i64, 1_i64];
        let b: i64 = 1_i64;
        let got = apply_scalar_unary(&a, |a| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(Vec<i64>, i64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(f64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]);
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Matrix<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
            Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.721, 0.742, 0.763, 0.784]),
                    Matrix::from_column_major(2, 2, vec![0.805, 0.826, 0.847, 0.868]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.889, 0.91, 0.931, 0.952]),
                    Matrix::from_column_major(2, 2, vec![0.973, 0.994, 1.015, 1.036]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.057, 1.078, 1.099, 1.12]),
                    Matrix::from_column_major(2, 2, vec![1.141, 1.162, 1.183, 1.204]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.225, 1.246, 1.267, 1.288]),
                    Matrix::from_column_major(2, 2, vec![1.309, 1.33, 1.351, 1.372]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.721, 0.742], vec![0.763, 0.784]],
            vec![vec![0.805, 0.826], vec![0.847, 0.868]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<f64>> = vec![vec![0.721, 0.742], vec![0.763, 0.784]];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: i64 = 1_i64;
        let b: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&b, |b| eval(pow(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "pow(i64, Vec<i64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(rayleigh_cdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "rayleigh_cdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(rayleigh_cdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(rayleigh_cdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "rayleigh_cdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(rayleigh_cdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "rayleigh_cdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(rayleigh_lccdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "rayleigh_lccdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(rayleigh_lccdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(rayleigh_lccdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "rayleigh_lccdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(rayleigh_lccdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "rayleigh_lccdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(rayleigh_lcdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "rayleigh_lcdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(rayleigh_lcdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(rayleigh_lcdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "rayleigh_lcdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(rayleigh_lcdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "rayleigh_lcdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.721, 0.742];
        let got =
            apply_scalar_binary(&a, &b, |a, b| eval(rayleigh_lpdf(a as f64, b as f64))).unwrap();
        let n = leaves(&got).len();
        compare(
            "rayleigh_lpdf(Vec<f64>, Vec<f64>)",
            leaves(&got),
            leaves(
                &apply_scalar_binary(&a, &b, |a, b| eval(rayleigh_lpdf(a as f64, b as f64)))
                    .unwrap(),
            ),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.721;
        let got = apply_scalar_unary(&a, |a| eval(rayleigh_lpdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "rayleigh_lpdf(Vec<f64>, f64)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let a: f64 = 0.309;
        let b: Vec<f64> = vec![0.721, 0.742];
        let got = apply_scalar_unary(&b, |b| eval(rayleigh_lpdf(a as f64, b as f64)));
        let n = leaves(&got).len();
        compare(
            "rayleigh_lpdf(f64, Vec<f64>)",
            leaves(&got),
            leaves(&got),
            &mut failures,
        );
        assert!(n > 0);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(round(v as f64)));
        compare(
            "round(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare(
            "sin(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare(
            "sin(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare(
            "sin(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare(
            "sin(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare(
            "sin(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare(
            "sin(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare(
            "sin(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare(
            "sin(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare("sin(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sin(v as f64)));
        compare("sin(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare(
            "sinh(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare(
            "sinh(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare(
            "sinh(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare(
            "sinh(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare(
            "sinh(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare(
            "sinh(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare(
            "sinh(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare(
            "sinh(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare("sinh(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sinh(v as f64)));
        compare("sinh(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare(
            "sqrt(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare(
            "sqrt(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare(
            "sqrt(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare(
            "sqrt(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare(
            "sqrt(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare(
            "sqrt(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare(
            "sqrt(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare(
            "sqrt(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare("sqrt(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(sqrt(v as f64)));
        compare("sqrt(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(square(v as f64)));
        compare(
            "square(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_cdf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_cdf(v as f64)));
        compare(
            "std_normal_cdf(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_lccdf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_lccdf(v as f64)));
        compare(
            "std_normal_lccdf(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_lcdf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_lcdf(v as f64)));
        compare(
            "std_normal_lcdf(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-0.618, -0.636, -0.654, -0.672]);
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-0.618, -0.636, -0.654, -0.672]),
            Matrix::from_column_major(2, 2, vec![-0.69, -0.708, -0.726, -0.744]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-0.618, -0.636, -0.654, -0.672]),
                Matrix::from_column_major(2, 2, vec![-0.69, -0.708, -0.726, -0.744]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-0.762, -0.78, -0.798, -0.816]),
                Matrix::from_column_major(2, 2, vec![-0.834, -0.852, -0.87, -0.888]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-0.618, -0.636, -0.654, -0.672]),
                    Matrix::from_column_major(2, 2, vec![-0.69, -0.708, -0.726, -0.744]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-0.762, -0.78, -0.798, -0.816]),
                    Matrix::from_column_major(2, 2, vec![-0.834, -0.852, -0.87, -0.888]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-0.906, -0.924, -0.942, -0.96]),
                    Matrix::from_column_major(2, 2, vec![-0.978, -0.996, -1.014, -1.032]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.05, -1.068, -1.086, -1.104]),
                    Matrix::from_column_major(2, 2, vec![-1.122, -1.14, -1.158, -1.176]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-0.618, -0.636], vec![-0.654, -0.672]],
            vec![vec![-0.69, -0.708], vec![-0.726, -0.744]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![-0.618, -0.636], vec![-0.654, -0.672]];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![-0.618, -0.636];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_log_qf(v as f64)));
        compare(
            "std_normal_log_qf(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_lpdf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_lpdf(v as f64)));
        compare(
            "std_normal_lpdf(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(std_normal_qf(v as f64)));
        compare(
            "std_normal_qf(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare(
            "tan(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare(
            "tan(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare(
            "tan(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare(
            "tan(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare(
            "tan(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare(
            "tan(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare(
            "tan(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare(
            "tan(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare("tan(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tan(v as f64)));
        compare("tan(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare(
            "tanh(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare(
            "tanh(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare(
            "tanh(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare(
            "tanh(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare(
            "tanh(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare(
            "tanh(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare(
            "tanh(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare(
            "tanh(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare("tanh(Vec<f64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tanh(v as f64)));
        compare("tanh(Vec<i64>)", leaves(&got), leaves(&want), &mut failures);
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(tgamma(v as f64)));
        compare(
            "tgamma(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trigamma(v as f64)));
        compare(
            "trigamma(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Matrix<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<Matrix<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<Vec<Matrix<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
                    Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.381, 0.39, 0.399, 0.408]),
                    Matrix::from_column_major(2, 2, vec![0.417, 0.426, 0.435, 0.444]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.453, 0.462, 0.471, 0.48]),
                    Matrix::from_column_major(2, 2, vec![0.489, 0.498, 0.507, 0.516]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.525, 0.534, 0.543, 0.552]),
                    Matrix::from_column_major(2, 2, vec![0.561, 0.57, 0.579, 0.588]),
                ],
            ],
        ];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<Vec<Vec<Matrix<f64>>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<Vec<Vec<f64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
            vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]],
        ];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<Vec<Vec<i64>>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<Vec<f64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 1_i64], vec![1_i64, 1_i64]];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<Vec<i64>>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<f64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 1_i64];
        let got = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        let want = apply_scalar_unary(&x, |v| eval(trunc(v as f64)));
        compare(
            "trunc(Vec<i64>)",
            leaves(&got),
            leaves(&want),
            &mut failures,
        );
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn deep_arrays() {
    // array[,,,,,,,] real, the deepest array in the signature dump.
    let mut x = vec![0.5, -1.0];
    let mut depth8: Vec<Vec<Vec<Vec<Vec<Vec<Vec<Vec<f64>>>>>>>> = Vec::new();
    for _ in 0..2 {
        depth8 = vec![vec![vec![vec![vec![vec![vec![x.clone()]]]]]]];
        x.push(2.0);
    }
    let got = apply_scalar_unary(&depth8, exp);
    let want: Vec<f64> = leaves(&depth8).into_iter().map(f64::exp).collect();
    assert_eq!(leaves(&got), want);
    let pairs = apply_scalar_binary(&depth8, &got, |a, b| pow(b, a)).unwrap();
    assert_eq!(leaves(&pairs).len(), want.len());
}
