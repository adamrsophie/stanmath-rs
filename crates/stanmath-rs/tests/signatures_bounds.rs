//! Every container signature of the bound transforms in the Stan language, through
//! `apply_scalar_unary`, `apply_scalar_binary`, and `apply_scalar_ternary`.
//!
//! Written by `tools/reference/gen_bounds.py` from the CmdStan 2.40.0 signature dump: 468 Stan
//! signatures in 252 Rust shapes. Each call must equal the scalar transform of every element,
//! and a `_jacobian` call's lp the sum of theirs. The other 684 signatures nest arrays more than
//! 3 `Vec` levels deep and share the same recursive impl.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::approx_constant,
    clippy::type_complexity,
    clippy::float_cmp
)]

use stanmath_rs::prelude::*;

/// The transforms return a value or a `Result`; a rejected argument becomes NaN.
trait Eval {
    fn eval(self) -> f64;
}

impl Eval for f64 {
    fn eval(self) -> f64 {
        self
    }
}

impl Eval for Result<f64> {
    fn eval(self) -> f64 {
        self.unwrap_or(f64::NAN)
    }
}

fn eval(v: impl Eval) -> f64 {
    v.eval()
}

fn leaves<C: ScalarContainer<Scalar = f64>>(c: &C) -> Vec<f64> {
    let mut out = Vec::new();
    apply_scalar_unary(c, |v| out.push(v));
    out
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

#[test]
fn bound_transform_signatures() {
    let mut failures = Vec::new();
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_bound_constrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_bound_constrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<Vec<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_constrain(Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_bound_jacobian(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_bound_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<Vec<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_bound_jacobian(Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_bound_unconstrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_bound_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_bound_unconstrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<Vec<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(lower_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_bound_unconstrain(Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Matrix<f64>, Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Matrix<f64>, Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Matrix<f64>, f64, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Matrix<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Matrix<f64>>, f64, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Vec<Matrix<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Matrix<f64>>>, f64, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Vec<Vec<Matrix<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Vec<Matrix<f64>>>>, f64, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Vec<Matrix<f64>>>>, f64, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<Vec<f64>>>, f64, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Vec<Vec<Vec<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let c: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<f64>>, Vec<Vec<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let c: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_constrain(Vec<Vec<f64>>, f64, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Vec<Vec<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let c: Vec<f64> = vec![2.06, 2.12];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Vec<f64>, Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Vec<f64>, Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let c: Vec<f64> = vec![2.06, 2.12];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Vec<f64>, f64, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_constrain(Vec<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Matrix<f64>, Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Matrix<f64>, Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Matrix<f64>, f64, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Matrix<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Matrix<f64>>, Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Matrix<f64>>, Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Matrix<f64>>, f64, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<Matrix<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Matrix<f64>>>, f64, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<Vec<Matrix<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, f64, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, f64, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<Vec<f64>>>, f64, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<Vec<Vec<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let c: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("lower_upper_bound_jacobian(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<Vec<f64>>, Vec<Vec<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let c: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<Vec<f64>>, f64, Vec<Vec<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<Vec<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let c: Vec<f64> = vec![2.06, 2.12];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<f64>, Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<f64>, Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let c: Vec<f64> = vec![2.06, 2.12];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<f64>, f64, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "lower_upper_bound_jacobian(Vec<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Matrix<f64>, Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]);
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Matrix<f64>, Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Matrix<f64>, f64, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Matrix<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
            Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
        ];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Matrix<f64>>, f64, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Vec<Matrix<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
            ],
        ];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Matrix<f64>>>, f64, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Matrix<f64>>>, f64, f64): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.03, -1.06, -1.09, -1.12]),
                    Matrix::from_column_major(2, 2, vec![-1.15, -1.18, -1.21, -1.24]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.27, -1.3, -1.33, -1.36]),
                    Matrix::from_column_major(2, 2, vec![-1.39, -1.42, -1.45, -1.48]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.51, -1.54, -1.57, -1.6]),
                    Matrix::from_column_major(2, 2, vec![-1.63, -1.66, -1.69, -1.72]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![-1.75, -1.78, -1.81, -1.84]),
                    Matrix::from_column_major(2, 2, vec![-1.87, -1.9, -1.93, -1.96]),
                ],
            ],
        ];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, f64, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, f64, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-1.03, -1.06], vec![-1.09, -1.12]],
            vec![vec![-1.15, -1.18], vec![-1.21, -1.24]],
        ];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<Vec<f64>>>, f64, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Vec<Vec<Vec<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let c: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![-1.03, -1.06], vec![-1.09, -1.12]];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<f64>>, Vec<Vec<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let c: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("lower_upper_bound_unconstrain(Vec<Vec<f64>>, f64, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Vec<Vec<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let c: Vec<f64> = vec![2.06, 2.12];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Vec<f64>, Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![-1.03, -1.06];
        let c: f64 = 2.06;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(lower_upper_bound_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Vec<f64>, Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let c: Vec<f64> = vec![2.06, 2.12];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(lower_upper_bound_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Vec<f64>, f64, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = -1.03;
        let c: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(lower_upper_bound_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(lower_upper_bound_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "lower_upper_bound_unconstrain(Vec<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]);
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]);
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Matrix<f64>, Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]);
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Matrix<f64>, Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.515;
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]);
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Matrix<f64>, f64, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Matrix<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
            Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
        ];
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
            Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
            Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
        ];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.515;
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
            Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Matrix<f64>>, f64, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Vec<Matrix<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
            ],
        ];
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
            ],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
            ],
        ];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Matrix<f64>>>, f64, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Vec<Vec<Matrix<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                    Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                    Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.755, 0.77, 0.785, 0.8]),
                    Matrix::from_column_major(2, 2, vec![0.815, 0.83, 0.845, 0.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.875, 0.89, 0.905, 0.92]),
                    Matrix::from_column_major(2, 2, vec![0.935, 0.95, 0.965, 0.98]),
                ],
            ],
        ];
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                    Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                    Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.265, 2.31, 2.355, 2.4]),
                    Matrix::from_column_major(2, 2, vec![2.445, 2.49, 2.535, 2.58]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.625, 2.67, 2.715, 2.76]),
                    Matrix::from_column_major(2, 2, vec![2.805, 2.85, 2.895, 2.94]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                    Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                    Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.755, 0.77, 0.785, 0.8]),
                    Matrix::from_column_major(2, 2, vec![0.815, 0.83, 0.845, 0.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.875, 0.89, 0.905, 0.92]),
                    Matrix::from_column_major(2, 2, vec![0.935, 0.95, 0.965, 0.98]),
                ],
            ],
        ];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                    Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                    Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.265, 2.31, 2.355, 2.4]),
                    Matrix::from_column_major(2, 2, vec![2.445, 2.49, 2.535, 2.58]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.625, 2.67, 2.715, 2.76]),
                    Matrix::from_column_major(2, 2, vec![2.805, 2.85, 2.895, 2.94]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Vec<Matrix<f64>>>>, f64, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Vec<Matrix<f64>>>>, f64, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.515, 0.53], vec![0.545, 0.56]],
            vec![vec![0.575, 0.59], vec![0.605, 0.62]],
        ];
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.545, 1.59], vec![1.635, 1.68]],
            vec![vec![1.725, 1.77], vec![1.815, 1.86]],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.515, 0.53], vec![0.545, 0.56]],
            vec![vec![0.575, 0.59], vec![0.605, 0.62]],
        ];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.515;
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.545, 1.59], vec![1.635, 1.68]],
            vec![vec![1.725, 1.77], vec![1.815, 1.86]],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<Vec<f64>>>, f64, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Vec<Vec<Vec<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.515, 0.53], vec![0.545, 0.56]];
        let c: Vec<Vec<f64>> = vec![vec![1.545, 1.59], vec![1.635, 1.68]];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.515, 0.53], vec![0.545, 0.56]];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<f64>>, Vec<Vec<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.515;
        let c: Vec<Vec<f64>> = vec![vec![1.545, 1.59], vec![1.635, 1.68]];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_constrain(Vec<Vec<f64>>, f64, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Vec<Vec<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.515, 0.53];
        let c: Vec<f64> = vec![1.545, 1.59];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_constrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Vec<f64>, Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.515, 0.53];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_constrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Vec<f64>, Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.515;
        let c: Vec<f64> = vec![1.545, 1.59];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_constrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Vec<f64>, f64, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_constrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_constrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_constrain(Vec<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]);
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]);
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Matrix<f64>, Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]);
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Matrix<f64>, Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.515;
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]);
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Matrix<f64>, f64, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Matrix<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
            Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
        ];
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
            Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Matrix<f64>>, Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
            Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
        ];
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Matrix<f64>>, Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.515;
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
            Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Matrix<f64>>, f64, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<Matrix<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
            ],
        ];
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
            ],
        ];
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Matrix<f64>>>, f64, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<Vec<Matrix<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                    Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                    Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.755, 0.77, 0.785, 0.8]),
                    Matrix::from_column_major(2, 2, vec![0.815, 0.83, 0.845, 0.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.875, 0.89, 0.905, 0.92]),
                    Matrix::from_column_major(2, 2, vec![0.935, 0.95, 0.965, 0.98]),
                ],
            ],
        ];
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                    Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                    Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.265, 2.31, 2.355, 2.4]),
                    Matrix::from_column_major(2, 2, vec![2.445, 2.49, 2.535, 2.58]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.625, 2.67, 2.715, 2.76]),
                    Matrix::from_column_major(2, 2, vec![2.805, 2.85, 2.895, 2.94]),
                ],
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                    Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                    Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.755, 0.77, 0.785, 0.8]),
                    Matrix::from_column_major(2, 2, vec![0.815, 0.83, 0.845, 0.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.875, 0.89, 0.905, 0.92]),
                    Matrix::from_column_major(2, 2, vec![0.935, 0.95, 0.965, 0.98]),
                ],
            ],
        ];
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                    Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                    Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.265, 2.31, 2.355, 2.4]),
                    Matrix::from_column_major(2, 2, vec![2.445, 2.49, 2.535, 2.58]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.625, 2.67, 2.715, 2.76]),
                    Matrix::from_column_major(2, 2, vec![2.805, 2.85, 2.895, 2.94]),
                ],
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, f64, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, f64, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.515, 0.53], vec![0.545, 0.56]],
            vec![vec![0.575, 0.59], vec![0.605, 0.62]],
        ];
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.545, 1.59], vec![1.635, 1.68]],
            vec![vec![1.725, 1.77], vec![1.815, 1.86]],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.515, 0.53], vec![0.545, 0.56]],
            vec![vec![0.575, 0.59], vec![0.605, 0.62]],
        ];
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.515;
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.545, 1.59], vec![1.635, 1.68]],
            vec![vec![1.725, 1.77], vec![1.815, 1.86]],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<Vec<f64>>>, f64, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<Vec<Vec<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.515, 0.53], vec![0.545, 0.56]];
        let c: Vec<Vec<f64>> = vec![vec![1.545, 1.59], vec![1.635, 1.68]];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("offset_multiplier_jacobian(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.515, 0.53], vec![0.545, 0.56]];
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<Vec<f64>>, Vec<Vec<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.515;
        let c: Vec<Vec<f64>> = vec![vec![1.545, 1.59], vec![1.635, 1.68]];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<Vec<f64>>, f64, Vec<Vec<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<Vec<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.515, 0.53];
        let c: Vec<f64> = vec![1.545, 1.59];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_jacobian(x_, b_, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<f64>, Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.515, 0.53];
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_jacobian(x_, b_, c, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], bs[i], c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<f64>, Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.515;
        let c: Vec<f64> = vec![1.545, 1.59];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_jacobian(x_, b, c_, &mut lp))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, cs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<f64>, f64, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_jacobian(x_, b, c, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_jacobian(xs[i], b, c, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "offset_multiplier_jacobian(Vec<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]);
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]);
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Matrix<f64>, Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]);
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Matrix<f64>, Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.515;
        let c: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]);
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Matrix<f64>, f64, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Matrix<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
            Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
        ];
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
            Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
            Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
        ];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.515;
        let c: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
            Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Matrix<f64>>, f64, Vec<Matrix<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Vec<Matrix<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
            ],
        ];
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
            ],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
            ],
        ];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Matrix<f64>>>, f64, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Matrix<f64>>>, f64, f64): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                    Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                    Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.755, 0.77, 0.785, 0.8]),
                    Matrix::from_column_major(2, 2, vec![0.815, 0.83, 0.845, 0.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.875, 0.89, 0.905, 0.92]),
                    Matrix::from_column_major(2, 2, vec![0.935, 0.95, 0.965, 0.98]),
                ],
            ],
        ];
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                    Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                    Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.265, 2.31, 2.355, 2.4]),
                    Matrix::from_column_major(2, 2, vec![2.445, 2.49, 2.535, 2.58]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.625, 2.67, 2.715, 2.76]),
                    Matrix::from_column_major(2, 2, vec![2.805, 2.85, 2.895, 2.94]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.515, 0.53, 0.545, 0.56]),
                    Matrix::from_column_major(2, 2, vec![0.575, 0.59, 0.605, 0.62]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.635, 0.65, 0.665, 0.68]),
                    Matrix::from_column_major(2, 2, vec![0.695, 0.71, 0.725, 0.74]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.755, 0.77, 0.785, 0.8]),
                    Matrix::from_column_major(2, 2, vec![0.815, 0.83, 0.845, 0.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.875, 0.89, 0.905, 0.92]),
                    Matrix::from_column_major(2, 2, vec![0.935, 0.95, 0.965, 0.98]),
                ],
            ],
        ];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.545, 1.59, 1.635, 1.68]),
                    Matrix::from_column_major(2, 2, vec![1.725, 1.77, 1.815, 1.86]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![1.905, 1.95, 1.995, 2.04]),
                    Matrix::from_column_major(2, 2, vec![2.085, 2.13, 2.175, 2.22]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.265, 2.31, 2.355, 2.4]),
                    Matrix::from_column_major(2, 2, vec![2.445, 2.49, 2.535, 2.58]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.625, 2.67, 2.715, 2.76]),
                    Matrix::from_column_major(2, 2, vec![2.805, 2.85, 2.895, 2.94]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, f64, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, f64, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.515, 0.53], vec![0.545, 0.56]],
            vec![vec![0.575, 0.59], vec![0.605, 0.62]],
        ];
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.545, 1.59], vec![1.635, 1.68]],
            vec![vec![1.725, 1.77], vec![1.815, 1.86]],
        ];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.515, 0.53], vec![0.545, 0.56]],
            vec![vec![0.575, 0.59], vec![0.605, 0.62]],
        ];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.515;
        let c: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.545, 1.59], vec![1.635, 1.68]],
            vec![vec![1.725, 1.77], vec![1.815, 1.86]],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<Vec<f64>>>, f64, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Vec<Vec<Vec<f64>>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.515, 0.53], vec![0.545, 0.56]];
        let c: Vec<Vec<f64>> = vec![vec![1.545, 1.59], vec![1.635, 1.68]];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![0.515, 0.53], vec![0.545, 0.56]];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<f64>>, Vec<Vec<f64>>, f64): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.515;
        let c: Vec<Vec<f64>> = vec![vec![1.545, 1.59], vec![1.635, 1.68]];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("offset_multiplier_unconstrain(Vec<Vec<f64>>, f64, Vec<Vec<f64>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Vec<Vec<f64>>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.515, 0.53];
        let c: Vec<f64> = vec![1.545, 1.59];
        let got = leaves(
            &apply_scalar_ternary(&x, &b, &c, |x_, b_, c_| {
                eval(offset_multiplier_unconstrain(x_, b_, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Vec<f64>, Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![0.515, 0.53];
        let c: f64 = 1.545;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| {
                eval(offset_multiplier_unconstrain(x_, b_, c))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], bs[i], c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Vec<f64>, Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.515;
        let c: Vec<f64> = vec![1.545, 1.59];
        let got = leaves(
            &apply_scalar_binary(&x, &c, |x_, c_| {
                eval(offset_multiplier_unconstrain(x_, b, c_))
            })
            .unwrap(),
        );
        let xs = leaves(&x);
        let cs = leaves(&c);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, cs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Vec<f64>, f64, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 0.515;
        let c: f64 = 1.545;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(offset_multiplier_unconstrain(x_, b, c))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(offset_multiplier_unconstrain(xs[i], b, c)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "offset_multiplier_unconstrain(Vec<f64>, f64, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("upper_bound_constrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("upper_bound_constrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<Vec<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![2.06, 2.12];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_constrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_constrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_constrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_constrain(Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("upper_bound_jacobian(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!("upper_bound_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<Vec<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![2.06, 2.12];
        let mut lp = 0.0;
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_jacobian(x_, b_, &mut lp)))
                .unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], bs[i], &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 2.06;
        let mut lp = 0.0;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_jacobian(x_, b, &mut lp))
        }));
        let xs = leaves(&x);
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_jacobian(xs[i], b, &mut lp)))
            .collect();
        if !(got == want && close(got_lp, lp)) {
            failures.push(format!(
                "upper_bound_jacobian(Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]);
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Matrix<f64>, Matrix<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]);
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Matrix<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
            Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<Matrix<f64>>, Vec<Matrix<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![0.309, 0.318, 0.327, 0.336]),
            Matrix::from_column_major(2, 2, vec![0.345, 0.354, 0.363, 0.372]),
        ];
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<Matrix<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("upper_bound_unconstrain(Vec<Vec<Matrix<f64>>>, Vec<Vec<Matrix<f64>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<Vec<Matrix<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
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
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![2.06, 2.12, 2.18, 2.24]),
                    Matrix::from_column_major(2, 2, vec![2.3, 2.36, 2.42, 2.48]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![2.54, 2.6, 2.66, 2.72]),
                    Matrix::from_column_major(2, 2, vec![2.78, 2.84, 2.9, 2.96]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![3.02, 3.08, 3.14, 3.2]),
                    Matrix::from_column_major(2, 2, vec![3.26, 3.32, 3.38, 3.44]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![3.5, 3.56, 3.62, 3.68]),
                    Matrix::from_column_major(2, 2, vec![3.74, 3.8, 3.86, 3.92]),
                ],
            ],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("upper_bound_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, Vec<Vec<Vec<Matrix<f64>>>>): {got:?} vs {want:?}"));
        }
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
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<Vec<Vec<Matrix<f64>>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![2.06, 2.12], vec![2.18, 2.24]],
            vec![vec![2.3, 2.36], vec![2.42, 2.48]],
        ];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!("upper_bound_unconstrain(Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>): {got:?} vs {want:?}"));
        }
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![0.309, 0.318], vec![0.327, 0.336]],
            vec![vec![0.345, 0.354], vec![0.363, 0.372]],
        ];
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<Vec<Vec<f64>>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: Vec<Vec<f64>> = vec![vec![2.06, 2.12], vec![2.18, 2.24]];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<Vec<f64>>, Vec<Vec<f64>>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![0.309, 0.318], vec![0.327, 0.336]];
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<Vec<f64>>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: Vec<f64> = vec![2.06, 2.12];
        let got = leaves(
            &apply_scalar_binary(&x, &b, |x_, b_| eval(upper_bound_unconstrain(x_, b_))).unwrap(),
        );
        let xs = leaves(&x);
        let bs = leaves(&b);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], bs[i])))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<f64>, Vec<f64>): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x: Vec<f64> = vec![0.309, 0.318];
        let b: f64 = 2.06;
        let got = leaves(&apply_scalar_unary(&x, |x_| {
            eval(upper_bound_unconstrain(x_, b))
        }));
        let xs = leaves(&x);
        let want: Vec<f64> = (0..xs.len())
            .map(|i| eval(upper_bound_unconstrain(xs[i], b)))
            .collect();
        if !(got == want) {
            failures.push(format!(
                "upper_bound_unconstrain(Vec<f64>, f64): {got:?} vs {want:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
