//! Every signature of `to_complex`, `get_real`, `get_imag`, and `conj` in the Stan language, and
//! the complex signatures of `abs`, `minus`, and `pow`, through `apply_scalar_unary` and
//! `apply_scalar_binary`. Over complex containers, `abs`, `minus`, and `pow` are the `Complex`
//! methods (`abs`, negation, `pow`) at each element; `stan_reference_complex.rs` compares those
//! with Stan Math.
//!
//! Written by `tools/reference/gen_complex.py` from the CmdStan 2.40.0 signature dump: 152 Stan
//! signatures, each checked against the definition or the scalar method at every element. The other 204 signatures nest arrays more
//! than three levels deep and share the same recursive impl.
#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::redundant_closure
)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

fn leaves<C: ScalarContainer>(c: &C) -> Vec<C::Scalar> {
    let mut out = Vec::new();
    apply_scalar_unary(c, |v| out.push(v));
    out
}

#[test]
fn complex_signatures() {
    {
        let x: Complex<f64> = Complex::new(1.25, -1.5);
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(complex)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(complex_vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(complex_row_vector)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.25, -1.5),
                Complex::new(2.25, -2.5),
                Complex::new(3.25, -3.5),
                Complex::new(4.25, -4.5),
            ],
        );
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(complex_matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[] complex)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[] complex_vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[] complex_row_vector)");
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.25, -1.5),
                    Complex::new(2.25, -2.5),
                    Complex::new(3.25, -3.5),
                    Complex::new(4.25, -4.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.25, -5.5),
                    Complex::new(6.25, -6.5),
                    Complex::new(7.25, -7.5),
                    Complex::new(8.25, -8.5),
                ],
            ),
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[] complex_matrix)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[,] complex_vector)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[,] complex_row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.25, -1.5),
                        Complex::new(2.25, -2.5),
                        Complex::new(3.25, -3.5),
                        Complex::new(4.25, -4.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.25, -5.5),
                        Complex::new(6.25, -6.5),
                        Complex::new(7.25, -7.5),
                        Complex::new(8.25, -8.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.25, -9.5),
                        Complex::new(10.25, -10.5),
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                    ],
                ),
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[,] complex_matrix)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[,,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.25, -1.5),
                            Complex::new(2.25, -2.5),
                            Complex::new(3.25, -3.5),
                            Complex::new(4.25, -4.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.25, -5.5),
                            Complex::new(6.25, -6.5),
                            Complex::new(7.25, -7.5),
                            Complex::new(8.25, -8.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.25, -9.5),
                            Complex::new(10.25, -10.5),
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                        ],
                    ),
                ],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| z.abs()));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.abs()).collect();
        assert_eq!(got, want, "abs(array[,,] complex_matrix)");
    }
    {
        let x: Complex<f64> = Complex::new(1.25, -1.5);
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(complex)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(complex_vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(complex_row_vector)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.25, -1.5),
                Complex::new(2.25, -2.5),
                Complex::new(3.25, -3.5),
                Complex::new(4.25, -4.5),
            ],
        );
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(complex_matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[] complex)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[] complex_vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[] complex_row_vector)");
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.25, -1.5),
                    Complex::new(2.25, -2.5),
                    Complex::new(3.25, -3.5),
                    Complex::new(4.25, -4.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.25, -5.5),
                    Complex::new(6.25, -6.5),
                    Complex::new(7.25, -7.5),
                    Complex::new(8.25, -8.5),
                ],
            ),
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[] complex_matrix)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[,] complex_vector)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[,] complex_row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.25, -1.5),
                        Complex::new(2.25, -2.5),
                        Complex::new(3.25, -3.5),
                        Complex::new(4.25, -4.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.25, -5.5),
                        Complex::new(6.25, -6.5),
                        Complex::new(7.25, -7.5),
                        Complex::new(8.25, -8.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.25, -9.5),
                        Complex::new(10.25, -10.5),
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                    ],
                ),
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[,] complex_matrix)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[,,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.25, -1.5),
                            Complex::new(2.25, -2.5),
                            Complex::new(3.25, -3.5),
                            Complex::new(4.25, -4.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.25, -5.5),
                            Complex::new(6.25, -6.5),
                            Complex::new(7.25, -7.5),
                            Complex::new(8.25, -8.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.25, -9.5),
                            Complex::new(10.25, -10.5),
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                        ],
                    ),
                ],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, conj));
        let want: Vec<_> = leaves(&x)
            .into_iter()
            .map(|z| Complex::new(z.re, -z.im))
            .collect();
        assert_eq!(got, want, "conj(array[,,] complex_matrix)");
    }
    {
        let x: Complex<f64> = Complex::new(1.25, -1.5);
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(complex)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(complex_vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(complex_row_vector)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.25, -1.5),
                Complex::new(2.25, -2.5),
                Complex::new(3.25, -3.5),
                Complex::new(4.25, -4.5),
            ],
        );
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(complex_matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[] complex)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[] complex_vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[] complex_row_vector)");
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.25, -1.5),
                    Complex::new(2.25, -2.5),
                    Complex::new(3.25, -3.5),
                    Complex::new(4.25, -4.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.25, -5.5),
                    Complex::new(6.25, -6.5),
                    Complex::new(7.25, -7.5),
                    Complex::new(8.25, -8.5),
                ],
            ),
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[] complex_matrix)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[,] complex_vector)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[,] complex_row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.25, -1.5),
                        Complex::new(2.25, -2.5),
                        Complex::new(3.25, -3.5),
                        Complex::new(4.25, -4.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.25, -5.5),
                        Complex::new(6.25, -6.5),
                        Complex::new(7.25, -7.5),
                        Complex::new(8.25, -8.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.25, -9.5),
                        Complex::new(10.25, -10.5),
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                    ],
                ),
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[,] complex_matrix)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[,,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.25, -1.5),
                            Complex::new(2.25, -2.5),
                            Complex::new(3.25, -3.5),
                            Complex::new(4.25, -4.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.25, -5.5),
                            Complex::new(6.25, -6.5),
                            Complex::new(7.25, -7.5),
                            Complex::new(8.25, -8.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.25, -9.5),
                            Complex::new(10.25, -10.5),
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                        ],
                    ),
                ],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_imag));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.im).collect();
        assert_eq!(got, want, "get_imag(array[,,] complex_matrix)");
    }
    {
        let x: Complex<f64> = Complex::new(1.25, -1.5);
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(complex)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(complex_vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(complex_row_vector)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.25, -1.5),
                Complex::new(2.25, -2.5),
                Complex::new(3.25, -3.5),
                Complex::new(4.25, -4.5),
            ],
        );
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(complex_matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[] complex)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[] complex_vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[] complex_row_vector)");
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.25, -1.5),
                    Complex::new(2.25, -2.5),
                    Complex::new(3.25, -3.5),
                    Complex::new(4.25, -4.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.25, -5.5),
                    Complex::new(6.25, -6.5),
                    Complex::new(7.25, -7.5),
                    Complex::new(8.25, -8.5),
                ],
            ),
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[] complex_matrix)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[,] complex_vector)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[,] complex_row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.25, -1.5),
                        Complex::new(2.25, -2.5),
                        Complex::new(3.25, -3.5),
                        Complex::new(4.25, -4.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.25, -5.5),
                        Complex::new(6.25, -6.5),
                        Complex::new(7.25, -7.5),
                        Complex::new(8.25, -8.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.25, -9.5),
                        Complex::new(10.25, -10.5),
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                    ],
                ),
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[,] complex_matrix)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[,,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.25, -1.5),
                            Complex::new(2.25, -2.5),
                            Complex::new(3.25, -3.5),
                            Complex::new(4.25, -4.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.25, -5.5),
                            Complex::new(6.25, -6.5),
                            Complex::new(7.25, -7.5),
                            Complex::new(8.25, -8.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.25, -9.5),
                            Complex::new(10.25, -10.5),
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                        ],
                    ),
                ],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, get_real));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| z.re).collect();
        assert_eq!(got, want, "get_real(array[,,] complex_matrix)");
    }
    {
        let x: Complex<f64> = Complex::new(1.25, -1.5);
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(complex)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(complex_vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(complex_row_vector)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.25, -1.5),
                Complex::new(2.25, -2.5),
                Complex::new(3.25, -3.5),
                Complex::new(4.25, -4.5),
            ],
        );
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(complex_matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[] complex)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[] complex_vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[] complex_row_vector)");
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.25, -1.5),
                    Complex::new(2.25, -2.5),
                    Complex::new(3.25, -3.5),
                    Complex::new(4.25, -4.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.25, -5.5),
                    Complex::new(6.25, -6.5),
                    Complex::new(7.25, -7.5),
                    Complex::new(8.25, -8.5),
                ],
            ),
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[] complex_matrix)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[,] complex_vector)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[,] complex_row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.25, -1.5),
                        Complex::new(2.25, -2.5),
                        Complex::new(3.25, -3.5),
                        Complex::new(4.25, -4.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.25, -5.5),
                        Complex::new(6.25, -6.5),
                        Complex::new(7.25, -7.5),
                        Complex::new(8.25, -8.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.25, -9.5),
                        Complex::new(10.25, -10.5),
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                    ],
                ),
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[,] complex_matrix)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[,,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.25, -1.5),
                            Complex::new(2.25, -2.5),
                            Complex::new(3.25, -3.5),
                            Complex::new(4.25, -4.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.25, -5.5),
                            Complex::new(6.25, -6.5),
                            Complex::new(7.25, -7.5),
                            Complex::new(8.25, -8.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.25, -9.5),
                            Complex::new(10.25, -10.5),
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                        ],
                    ),
                ],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&x, |z: Complex<f64>| -z));
        let want: Vec<_> = leaves(&x).into_iter().map(|z| -z).collect();
        assert_eq!(got, want, "minus(array[,,] complex_matrix)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = vec![a.pow(b)];
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, complex)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Complex<f64>> = vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, complex_vector)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Complex<f64>> = vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, complex_row_vector)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(11.25, -11.5),
                Complex::new(12.25, -12.5),
                Complex::new(13.25, -13.5),
                Complex::new(14.25, -14.5),
            ],
        );
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, complex_matrix)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Complex<f64>> = vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[] complex)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
            vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[] complex_vector)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
            vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[] complex_row_vector)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(11.25, -11.5),
                    Complex::new(12.25, -12.5),
                    Complex::new(13.25, -13.5),
                    Complex::new(14.25, -14.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(15.25, -15.5),
                    Complex::new(16.25, -16.5),
                    Complex::new(17.25, -17.5),
                    Complex::new(18.25, -18.5),
                ],
            ),
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[] complex_matrix)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
            vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[,] complex)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
                vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
            ],
            vec![
                vec![Complex::new(15.25, -15.5), Complex::new(16.25, -16.5)],
                vec![Complex::new(17.25, -17.5), Complex::new(18.25, -18.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[,] complex_vector)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
                vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
            ],
            vec![
                vec![Complex::new(15.25, -15.5), Complex::new(16.25, -16.5)],
                vec![Complex::new(17.25, -17.5), Complex::new(18.25, -18.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[,] complex_row_vector)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                        Complex::new(17.25, -17.5),
                        Complex::new(18.25, -18.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(19.25, -19.5),
                        Complex::new(20.25, -20.5),
                        Complex::new(21.25, -21.5),
                        Complex::new(22.25, -22.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(23.25, -23.5),
                        Complex::new(24.25, -24.5),
                        Complex::new(25.25, -25.5),
                        Complex::new(26.25, -26.5),
                    ],
                ),
            ],
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[,] complex_matrix)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
                vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
            ],
            vec![
                vec![Complex::new(15.25, -15.5), Complex::new(16.25, -16.5)],
                vec![Complex::new(17.25, -17.5), Complex::new(18.25, -18.5)],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[,,] complex)");
    }
    {
        let a: Complex<f64> = Complex::new(1.25, -1.5);
        let b: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                            Complex::new(33.25, -33.5),
                            Complex::new(34.25, -34.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(35.25, -35.5),
                            Complex::new(36.25, -36.5),
                            Complex::new(37.25, -37.5),
                            Complex::new(38.25, -38.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(39.25, -39.5),
                            Complex::new(40.25, -40.5),
                            Complex::new(41.25, -41.5),
                            Complex::new(42.25, -42.5),
                        ],
                    ),
                ],
            ],
        ];
        let got = leaves(&apply_scalar_unary(&b, |w| a.pow(w)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex, array[,,] complex_matrix)");
    }
    {
        let a: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex_vector, complex)");
    }
    {
        let a: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let b: Vec<Complex<f64>> = vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex_vector, complex_vector)");
    }
    {
        let a: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex_row_vector, complex)");
    }
    {
        let a: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let b: Vec<Complex<f64>> = vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex_row_vector, complex_row_vector)");
    }
    {
        let a: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.25, -1.5),
                Complex::new(2.25, -2.5),
                Complex::new(3.25, -3.5),
                Complex::new(4.25, -4.5),
            ],
        );
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex_matrix, complex)");
    }
    {
        let a: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.25, -1.5),
                Complex::new(2.25, -2.5),
                Complex::new(3.25, -3.5),
                Complex::new(4.25, -4.5),
            ],
        );
        let b: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(11.25, -11.5),
                Complex::new(12.25, -12.5),
                Complex::new(13.25, -13.5),
                Complex::new(14.25, -14.5),
            ],
        );
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(complex_matrix, complex_matrix)");
    }
    {
        let a: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[] complex, complex)");
    }
    {
        let a: Vec<Complex<f64>> = vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)];
        let b: Vec<Complex<f64>> = vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[] complex, array[] complex)");
    }
    {
        let a: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[] complex_vector, complex)");
    }
    {
        let a: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let b: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
            vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(
            got, want,
            "pow(array[] complex_vector, array[] complex_vector)"
        );
    }
    {
        let a: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[] complex_row_vector, complex)");
    }
    {
        let a: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let b: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
            vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(
            got, want,
            "pow(array[] complex_row_vector, array[] complex_row_vector)"
        );
    }
    {
        let a: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.25, -1.5),
                    Complex::new(2.25, -2.5),
                    Complex::new(3.25, -3.5),
                    Complex::new(4.25, -4.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.25, -5.5),
                    Complex::new(6.25, -6.5),
                    Complex::new(7.25, -7.5),
                    Complex::new(8.25, -8.5),
                ],
            ),
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[] complex_matrix, complex)");
    }
    {
        let a: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.25, -1.5),
                    Complex::new(2.25, -2.5),
                    Complex::new(3.25, -3.5),
                    Complex::new(4.25, -4.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.25, -5.5),
                    Complex::new(6.25, -6.5),
                    Complex::new(7.25, -7.5),
                    Complex::new(8.25, -8.5),
                ],
            ),
        ];
        let b: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(11.25, -11.5),
                    Complex::new(12.25, -12.5),
                    Complex::new(13.25, -13.5),
                    Complex::new(14.25, -14.5),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(15.25, -15.5),
                    Complex::new(16.25, -16.5),
                    Complex::new(17.25, -17.5),
                    Complex::new(18.25, -18.5),
                ],
            ),
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(
            got, want,
            "pow(array[] complex_matrix, array[] complex_matrix)"
        );
    }
    {
        let a: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[,] complex, complex)");
    }
    {
        let a: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
            vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
        ];
        let b: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
            vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[,] complex, array[,] complex)");
    }
    {
        let a: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[,] complex_vector, complex)");
    }
    {
        let a: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let b: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
                vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
            ],
            vec![
                vec![Complex::new(15.25, -15.5), Complex::new(16.25, -16.5)],
                vec![Complex::new(17.25, -17.5), Complex::new(18.25, -18.5)],
            ],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(
            got, want,
            "pow(array[,] complex_vector, array[,] complex_vector)"
        );
    }
    {
        let a: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[,] complex_row_vector, complex)");
    }
    {
        let a: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let b: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
                vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
            ],
            vec![
                vec![Complex::new(15.25, -15.5), Complex::new(16.25, -16.5)],
                vec![Complex::new(17.25, -17.5), Complex::new(18.25, -18.5)],
            ],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(
            got, want,
            "pow(array[,] complex_row_vector, array[,] complex_row_vector)"
        );
    }
    {
        let a: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.25, -1.5),
                        Complex::new(2.25, -2.5),
                        Complex::new(3.25, -3.5),
                        Complex::new(4.25, -4.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.25, -5.5),
                        Complex::new(6.25, -6.5),
                        Complex::new(7.25, -7.5),
                        Complex::new(8.25, -8.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.25, -9.5),
                        Complex::new(10.25, -10.5),
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                    ],
                ),
            ],
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[,] complex_matrix, complex)");
    }
    {
        let a: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.25, -1.5),
                        Complex::new(2.25, -2.5),
                        Complex::new(3.25, -3.5),
                        Complex::new(4.25, -4.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.25, -5.5),
                        Complex::new(6.25, -6.5),
                        Complex::new(7.25, -7.5),
                        Complex::new(8.25, -8.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.25, -9.5),
                        Complex::new(10.25, -10.5),
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                    ],
                ),
            ],
        ];
        let b: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(11.25, -11.5),
                        Complex::new(12.25, -12.5),
                        Complex::new(13.25, -13.5),
                        Complex::new(14.25, -14.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(15.25, -15.5),
                        Complex::new(16.25, -16.5),
                        Complex::new(17.25, -17.5),
                        Complex::new(18.25, -18.5),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(19.25, -19.5),
                        Complex::new(20.25, -20.5),
                        Complex::new(21.25, -21.5),
                        Complex::new(22.25, -22.5),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(23.25, -23.5),
                        Complex::new(24.25, -24.5),
                        Complex::new(25.25, -25.5),
                        Complex::new(26.25, -26.5),
                    ],
                ),
            ],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(
            got, want,
            "pow(array[,] complex_matrix, array[,] complex_matrix)"
        );
    }
    {
        let a: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[,,] complex, complex)");
    }
    {
        let a: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.25, -1.5), Complex::new(2.25, -2.5)],
                vec![Complex::new(3.25, -3.5), Complex::new(4.25, -4.5)],
            ],
            vec![
                vec![Complex::new(5.25, -5.5), Complex::new(6.25, -6.5)],
                vec![Complex::new(7.25, -7.5), Complex::new(8.25, -8.5)],
            ],
        ];
        let b: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(11.25, -11.5), Complex::new(12.25, -12.5)],
                vec![Complex::new(13.25, -13.5), Complex::new(14.25, -14.5)],
            ],
            vec![
                vec![Complex::new(15.25, -15.5), Complex::new(16.25, -16.5)],
                vec![Complex::new(17.25, -17.5), Complex::new(18.25, -18.5)],
            ],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[,,] complex, array[,,] complex)");
    }
    {
        let a: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.25, -1.5),
                            Complex::new(2.25, -2.5),
                            Complex::new(3.25, -3.5),
                            Complex::new(4.25, -4.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.25, -5.5),
                            Complex::new(6.25, -6.5),
                            Complex::new(7.25, -7.5),
                            Complex::new(8.25, -8.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.25, -9.5),
                            Complex::new(10.25, -10.5),
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                        ],
                    ),
                ],
            ],
        ];
        let b: Complex<f64> = Complex::new(11.25, -11.5);
        let got = leaves(&apply_scalar_unary(&a, |z| z.pow(b)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(got, want, "pow(array[,,] complex_matrix, complex)");
    }
    {
        let a: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.25, -1.5),
                            Complex::new(2.25, -2.5),
                            Complex::new(3.25, -3.5),
                            Complex::new(4.25, -4.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.25, -5.5),
                            Complex::new(6.25, -6.5),
                            Complex::new(7.25, -7.5),
                            Complex::new(8.25, -8.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.25, -9.5),
                            Complex::new(10.25, -10.5),
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                        ],
                    ),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(11.25, -11.5),
                            Complex::new(12.25, -12.5),
                            Complex::new(13.25, -13.5),
                            Complex::new(14.25, -14.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(15.25, -15.5),
                            Complex::new(16.25, -16.5),
                            Complex::new(17.25, -17.5),
                            Complex::new(18.25, -18.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(19.25, -19.5),
                            Complex::new(20.25, -20.5),
                            Complex::new(21.25, -21.5),
                            Complex::new(22.25, -22.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(23.25, -23.5),
                            Complex::new(24.25, -24.5),
                            Complex::new(25.25, -25.5),
                            Complex::new(26.25, -26.5),
                        ],
                    ),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(27.25, -27.5),
                            Complex::new(28.25, -28.5),
                            Complex::new(29.25, -29.5),
                            Complex::new(30.25, -30.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(31.25, -31.5),
                            Complex::new(32.25, -32.5),
                            Complex::new(33.25, -33.5),
                            Complex::new(34.25, -34.5),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(35.25, -35.5),
                            Complex::new(36.25, -36.5),
                            Complex::new(37.25, -37.5),
                            Complex::new(38.25, -38.5),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(39.25, -39.5),
                            Complex::new(40.25, -40.5),
                            Complex::new(41.25, -41.5),
                            Complex::new(42.25, -42.5),
                        ],
                    ),
                ],
            ],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |z, w| z.pow(w)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(z, w)| z.pow(w))
            .collect();
        assert_eq!(
            got, want,
            "pow(array[,,] complex_matrix, array[,,] complex_matrix)"
        );
    }
    assert_eq!(to_complex(0.0, 0.0), Complex::new(0.0, 0.0), "to_complex()");
    {
        let a: f64 = 1.25;
        assert_eq!(to_complex(a, 0.0), Complex::new(a, 0.0), "to_complex(real)");
    }
    {
        let a: f64 = 1.25;
        let b: f64 = 11.25;
        let (x, y) = (a, b);
        let got = vec![to_complex(x, y)];
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, real)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<f64> = vec![11.25, 12.25];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, vector)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<f64> = vec![11.25, 12.25];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, row_vector)");
    }
    {
        let a: f64 = 1.25;
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![11.25, 12.25, 13.25, 14.25]);
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, matrix)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<f64> = vec![11.25, 12.25];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[] real)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Vec<f64>> = vec![vec![11.25, 12.25], vec![13.25, 14.25]];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[] vector)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Vec<f64>> = vec![vec![11.25, 12.25], vec![13.25, 14.25]];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[] row_vector)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![11.25, 12.25, 13.25, 14.25]),
            Matrix::from_column_major(2, 2, vec![15.25, 16.25, 17.25, 18.25]),
        ];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[] matrix)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Vec<f64>> = vec![vec![11.25, 12.25], vec![13.25, 14.25]];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[,] real)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![11.25, 12.25], vec![13.25, 14.25]],
            vec![vec![15.25, 16.25], vec![17.25, 18.25]],
        ];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[,] vector)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![11.25, 12.25], vec![13.25, 14.25]],
            vec![vec![15.25, 16.25], vec![17.25, 18.25]],
        ];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[,] row_vector)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![11.25, 12.25, 13.25, 14.25]),
                Matrix::from_column_major(2, 2, vec![15.25, 16.25, 17.25, 18.25]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![19.25, 20.25, 21.25, 22.25]),
                Matrix::from_column_major(2, 2, vec![23.25, 24.25, 25.25, 26.25]),
            ],
        ];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[,] matrix)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![11.25, 12.25], vec![13.25, 14.25]],
            vec![vec![15.25, 16.25], vec![17.25, 18.25]],
        ];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[,,] real)");
    }
    {
        let a: f64 = 1.25;
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![11.25, 12.25, 13.25, 14.25]),
                    Matrix::from_column_major(2, 2, vec![15.25, 16.25, 17.25, 18.25]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![19.25, 20.25, 21.25, 22.25]),
                    Matrix::from_column_major(2, 2, vec![23.25, 24.25, 25.25, 26.25]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![27.25, 28.25, 29.25, 30.25]),
                    Matrix::from_column_major(2, 2, vec![31.25, 32.25, 33.25, 34.25]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![35.25, 36.25, 37.25, 38.25]),
                    Matrix::from_column_major(2, 2, vec![39.25, 40.25, 41.25, 42.25]),
                ],
            ],
        ];
        let x = a;
        let got = leaves(&apply_scalar_unary(&b, |y| to_complex(x, y)));
        let want: Vec<Complex<f64>> = vec![a; got.len()]
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(real, array[,,] matrix)");
    }
    {
        let a: Vec<f64> = vec![1.25, 2.25];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(vector, real)");
    }
    {
        let a: Vec<f64> = vec![1.25, 2.25];
        let b: Vec<f64> = vec![11.25, 12.25];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(vector, vector)");
    }
    {
        let a: Vec<f64> = vec![1.25, 2.25];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(row_vector, real)");
    }
    {
        let a: Vec<f64> = vec![1.25, 2.25];
        let b: Vec<f64> = vec![11.25, 12.25];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(row_vector, row_vector)");
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.25, 2.25, 3.25, 4.25]);
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(matrix, real)");
    }
    {
        let a: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.25, 2.25, 3.25, 4.25]);
        let b: Matrix<f64> = Matrix::from_column_major(2, 2, vec![11.25, 12.25, 13.25, 14.25]);
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(matrix, matrix)");
    }
    {
        let a: Vec<f64> = vec![1.25, 2.25];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[] real, real)");
    }
    {
        let a: Vec<f64> = vec![1.25, 2.25];
        let b: Vec<f64> = vec![11.25, 12.25];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[] real, array[] real)");
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![1.25, 2.25], vec![3.25, 4.25]];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[] vector, real)");
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![1.25, 2.25], vec![3.25, 4.25]];
        let b: Vec<Vec<f64>> = vec![vec![11.25, 12.25], vec![13.25, 14.25]];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[] vector, array[] vector)");
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![1.25, 2.25], vec![3.25, 4.25]];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[] row_vector, real)");
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![1.25, 2.25], vec![3.25, 4.25]];
        let b: Vec<Vec<f64>> = vec![vec![11.25, 12.25], vec![13.25, 14.25]];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(
            got, want,
            "to_complex(array[] row_vector, array[] row_vector)"
        );
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.25, 2.25, 3.25, 4.25]),
            Matrix::from_column_major(2, 2, vec![5.25, 6.25, 7.25, 8.25]),
        ];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[] matrix, real)");
    }
    {
        let a: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.25, 2.25, 3.25, 4.25]),
            Matrix::from_column_major(2, 2, vec![5.25, 6.25, 7.25, 8.25]),
        ];
        let b: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![11.25, 12.25, 13.25, 14.25]),
            Matrix::from_column_major(2, 2, vec![15.25, 16.25, 17.25, 18.25]),
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[] matrix, array[] matrix)");
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![1.25, 2.25], vec![3.25, 4.25]];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,] real, real)");
    }
    {
        let a: Vec<Vec<f64>> = vec![vec![1.25, 2.25], vec![3.25, 4.25]];
        let b: Vec<Vec<f64>> = vec![vec![11.25, 12.25], vec![13.25, 14.25]];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,] real, array[,] real)");
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.25, 2.25], vec![3.25, 4.25]],
            vec![vec![5.25, 6.25], vec![7.25, 8.25]],
        ];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,] vector, real)");
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.25, 2.25], vec![3.25, 4.25]],
            vec![vec![5.25, 6.25], vec![7.25, 8.25]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![11.25, 12.25], vec![13.25, 14.25]],
            vec![vec![15.25, 16.25], vec![17.25, 18.25]],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,] vector, array[,] vector)");
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.25, 2.25], vec![3.25, 4.25]],
            vec![vec![5.25, 6.25], vec![7.25, 8.25]],
        ];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,] row_vector, real)");
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.25, 2.25], vec![3.25, 4.25]],
            vec![vec![5.25, 6.25], vec![7.25, 8.25]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![11.25, 12.25], vec![13.25, 14.25]],
            vec![vec![15.25, 16.25], vec![17.25, 18.25]],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(
            got, want,
            "to_complex(array[,] row_vector, array[,] row_vector)"
        );
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.25, 2.25, 3.25, 4.25]),
                Matrix::from_column_major(2, 2, vec![5.25, 6.25, 7.25, 8.25]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.25, 10.25, 11.25, 12.25]),
                Matrix::from_column_major(2, 2, vec![13.25, 14.25, 15.25, 16.25]),
            ],
        ];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,] matrix, real)");
    }
    {
        let a: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.25, 2.25, 3.25, 4.25]),
                Matrix::from_column_major(2, 2, vec![5.25, 6.25, 7.25, 8.25]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.25, 10.25, 11.25, 12.25]),
                Matrix::from_column_major(2, 2, vec![13.25, 14.25, 15.25, 16.25]),
            ],
        ];
        let b: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![11.25, 12.25, 13.25, 14.25]),
                Matrix::from_column_major(2, 2, vec![15.25, 16.25, 17.25, 18.25]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![19.25, 20.25, 21.25, 22.25]),
                Matrix::from_column_major(2, 2, vec![23.25, 24.25, 25.25, 26.25]),
            ],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,] matrix, array[,] matrix)");
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.25, 2.25], vec![3.25, 4.25]],
            vec![vec![5.25, 6.25], vec![7.25, 8.25]],
        ];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,,] real, real)");
    }
    {
        let a: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.25, 2.25], vec![3.25, 4.25]],
            vec![vec![5.25, 6.25], vec![7.25, 8.25]],
        ];
        let b: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![11.25, 12.25], vec![13.25, 14.25]],
            vec![vec![15.25, 16.25], vec![17.25, 18.25]],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,,] real, array[,,] real)");
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.25, 2.25, 3.25, 4.25]),
                    Matrix::from_column_major(2, 2, vec![5.25, 6.25, 7.25, 8.25]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.25, 10.25, 11.25, 12.25]),
                    Matrix::from_column_major(2, 2, vec![13.25, 14.25, 15.25, 16.25]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.25, 18.25, 19.25, 20.25]),
                    Matrix::from_column_major(2, 2, vec![21.25, 22.25, 23.25, 24.25]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.25, 26.25, 27.25, 28.25]),
                    Matrix::from_column_major(2, 2, vec![29.25, 30.25, 31.25, 32.25]),
                ],
            ],
        ];
        let b: f64 = 11.25;
        let y = b;
        let got = leaves(&apply_scalar_unary(&a, |x| to_complex(x, y)));
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(vec![b; got.len()])
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,,] matrix, real)");
    }
    {
        let a: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.25, 2.25, 3.25, 4.25]),
                    Matrix::from_column_major(2, 2, vec![5.25, 6.25, 7.25, 8.25]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.25, 10.25, 11.25, 12.25]),
                    Matrix::from_column_major(2, 2, vec![13.25, 14.25, 15.25, 16.25]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.25, 18.25, 19.25, 20.25]),
                    Matrix::from_column_major(2, 2, vec![21.25, 22.25, 23.25, 24.25]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.25, 26.25, 27.25, 28.25]),
                    Matrix::from_column_major(2, 2, vec![29.25, 30.25, 31.25, 32.25]),
                ],
            ],
        ];
        let b: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![11.25, 12.25, 13.25, 14.25]),
                    Matrix::from_column_major(2, 2, vec![15.25, 16.25, 17.25, 18.25]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![19.25, 20.25, 21.25, 22.25]),
                    Matrix::from_column_major(2, 2, vec![23.25, 24.25, 25.25, 26.25]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![27.25, 28.25, 29.25, 30.25]),
                    Matrix::from_column_major(2, 2, vec![31.25, 32.25, 33.25, 34.25]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![35.25, 36.25, 37.25, 38.25]),
                    Matrix::from_column_major(2, 2, vec![39.25, 40.25, 41.25, 42.25]),
                ],
            ],
        ];
        let got = leaves(&apply_scalar_binary(&a, &b, |x, y| to_complex(x, y)).unwrap());
        let want: Vec<Complex<f64>> = leaves(&a)
            .into_iter()
            .zip(leaves(&b))
            .map(|(x, y)| Complex::new(x, y))
            .collect();
        assert_eq!(got, want, "to_complex(array[,,] matrix, array[,,] matrix)");
    }
}
