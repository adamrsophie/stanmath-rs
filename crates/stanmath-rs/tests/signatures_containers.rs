//! Every signature of the generic container functions of the Stan language (`dims`,
//! `num_elements`, `size`, `rep_array`, `reverse`, `append_array`, `head`, `tail`, `segment`)
//! over every element type, complex ones included.
//!
//! Written by `tools/reference/gen_containers.py` from the CmdStan 2.40.0 signature dump: 302
//! Stan signatures, each checked against the shape it must produce. The other 239 signatures
//! nest arrays more than three levels deep and run the same generic code.
#![allow(clippy::too_many_lines, clippy::redundant_clone, clippy::clone_on_copy)]

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

#[test]
fn container_signatures() {
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[] int, array[] int)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[] real, array[] real)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[] vector, array[] vector)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[] complex, array[] complex)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[] row_vector, array[] row_vector)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[] matrix, array[] matrix)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[] complex_vector, array[] complex_vector)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[] complex_row_vector, array[] complex_row_vector)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[] complex_matrix, array[] complex_matrix)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[,] int, array[,] int)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[,] real, array[,] real)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[,] vector, array[,] vector)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[,] complex, array[,] complex)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[,] row_vector, array[,] row_vector)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[,] matrix, array[,] matrix)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[,] complex_vector, array[,] complex_vector)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[,] complex_row_vector, array[,] complex_row_vector)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[,] complex_matrix, array[,] complex_matrix)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]],
            vec![vec![5_i64, 6_i64], vec![7_i64, 8_i64]],
        ];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[,,] int, array[,,] int)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = append_array(&x, &x);
        assert_eq!(r.len(), 4, "append_array(array[,,] real, array[,,] real)");
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[,,] complex, array[,,] complex)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                    Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                    Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.5, 18.5, 19.5, 20.5]),
                    Matrix::from_column_major(2, 2, vec![21.5, 22.5, 23.5, 24.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.5, 26.5, 27.5, 28.5]),
                    Matrix::from_column_major(2, 2, vec![29.5, 30.5, 31.5, 32.5]),
                ],
            ],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[,,] matrix, array[,,] matrix)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.5, -1.0),
                            Complex::new(2.5, -2.0),
                            Complex::new(3.5, -3.0),
                            Complex::new(4.5, -4.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.5, -5.0),
                            Complex::new(6.5, -6.0),
                            Complex::new(7.5, -7.0),
                            Complex::new(8.5, -8.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.5, -9.0),
                            Complex::new(10.5, -10.0),
                            Complex::new(11.5, -11.0),
                            Complex::new(12.5, -12.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.5, -13.0),
                            Complex::new(14.5, -14.0),
                            Complex::new(15.5, -15.0),
                            Complex::new(16.5, -16.0),
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
                            Complex::new(17.5, -17.0),
                            Complex::new(18.5, -18.0),
                            Complex::new(19.5, -19.0),
                            Complex::new(20.5, -20.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.5, -21.0),
                            Complex::new(22.5, -22.0),
                            Complex::new(23.5, -23.0),
                            Complex::new(24.5, -24.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.5, -25.0),
                            Complex::new(26.5, -26.0),
                            Complex::new(27.5, -27.0),
                            Complex::new(28.5, -28.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.5, -29.0),
                            Complex::new(30.5, -30.0),
                            Complex::new(31.5, -31.0),
                            Complex::new(32.5, -32.0),
                        ],
                    ),
                ],
            ],
        ];
        let r = append_array(&x, &x);
        assert_eq!(
            r.len(),
            4,
            "append_array(array[,,] complex_matrix, array[,,] complex_matrix)"
        );
        assert_eq!(r[2], x[0]);
    }
    {
        let x: i64 = 1_i64;
        assert_eq!(dims(&x), vec![], "dims(int)");
    }
    {
        let x: f64 = 1.5;
        assert_eq!(dims(&x), vec![], "dims(real)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(dims(&x), vec![2], "dims(vector)");
    }
    {
        let x: Complex<f64> = Complex::new(1.5, -1.0);
        assert_eq!(dims(&x), vec![], "dims(complex)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(dims(&x), vec![2], "dims(row_vector)");
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]);
        assert_eq!(dims(&x), vec![2, 2], "dims(matrix)");
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        assert_eq!(dims(&x), vec![2], "dims(array[] int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(dims(&x), vec![2], "dims(array[] real)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(dims(&x), vec![2, 2], "dims(array[] vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(dims(&x), vec![2], "dims(array[] complex)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(dims(&x), vec![2, 2], "dims(array[] row_vector)");
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[] matrix)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(dims(&x), vec![2, 2], "dims(array[] complex_vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(dims(&x), vec![2, 2], "dims(array[] complex_row_vector)");
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[] complex_matrix)");
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        assert_eq!(dims(&x), vec![2, 2], "dims(array[,] int)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(dims(&x), vec![2, 2], "dims(array[,] real)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[,] vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(dims(&x), vec![2, 2], "dims(array[,] complex)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[,] row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2, 2], "dims(array[,] matrix)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[,] complex_vector)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[,] complex_row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2, 2], "dims(array[,] complex_matrix)");
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]],
            vec![vec![5_i64, 6_i64], vec![7_i64, 8_i64]],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[,,] int)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[,,] real)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2], "dims(array[,,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                    Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                    Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.5, 18.5, 19.5, 20.5]),
                    Matrix::from_column_major(2, 2, vec![21.5, 22.5, 23.5, 24.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.5, 26.5, 27.5, 28.5]),
                    Matrix::from_column_major(2, 2, vec![29.5, 30.5, 31.5, 32.5]),
                ],
            ],
        ];
        assert_eq!(dims(&x), vec![2, 2, 2, 2, 2], "dims(array[,,] matrix)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.5, -1.0),
                            Complex::new(2.5, -2.0),
                            Complex::new(3.5, -3.0),
                            Complex::new(4.5, -4.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.5, -5.0),
                            Complex::new(6.5, -6.0),
                            Complex::new(7.5, -7.0),
                            Complex::new(8.5, -8.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.5, -9.0),
                            Complex::new(10.5, -10.0),
                            Complex::new(11.5, -11.0),
                            Complex::new(12.5, -12.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.5, -13.0),
                            Complex::new(14.5, -14.0),
                            Complex::new(15.5, -15.0),
                            Complex::new(16.5, -16.0),
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
                            Complex::new(17.5, -17.0),
                            Complex::new(18.5, -18.0),
                            Complex::new(19.5, -19.0),
                            Complex::new(20.5, -20.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.5, -21.0),
                            Complex::new(22.5, -22.0),
                            Complex::new(23.5, -23.0),
                            Complex::new(24.5, -24.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.5, -25.0),
                            Complex::new(26.5, -26.0),
                            Complex::new(27.5, -27.0),
                            Complex::new(28.5, -28.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.5, -29.0),
                            Complex::new(30.5, -30.0),
                            Complex::new(31.5, -31.0),
                            Complex::new(32.5, -32.0),
                        ],
                    ),
                ],
            ],
        ];
        assert_eq!(
            dims(&x),
            vec![2, 2, 2, 2, 2],
            "dims(array[,,] complex_matrix)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(vector, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(row_vector, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(complex_vector, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(complex_row_vector, int)"
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] int, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] real, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] vector, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] complex, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] row_vector, int)"
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] complex_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] complex_row_vector, int)"
        );
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[] complex_matrix, int)"
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] real, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] vector, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] complex, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] row_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] complex_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] complex_row_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,] complex_matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]],
            vec![vec![5_i64, 6_i64], vec![7_i64, 8_i64]],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,,] int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,,] real, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,,] complex, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                    Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                    Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.5, 18.5, 19.5, 20.5]),
                    Matrix::from_column_major(2, 2, vec![21.5, 22.5, 23.5, 24.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.5, 26.5, 27.5, 28.5]),
                    Matrix::from_column_major(2, 2, vec![29.5, 30.5, 31.5, 32.5]),
                ],
            ],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,,] matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.5, -1.0),
                            Complex::new(2.5, -2.0),
                            Complex::new(3.5, -3.0),
                            Complex::new(4.5, -4.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.5, -5.0),
                            Complex::new(6.5, -6.0),
                            Complex::new(7.5, -7.0),
                            Complex::new(8.5, -8.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.5, -9.0),
                            Complex::new(10.5, -10.0),
                            Complex::new(11.5, -11.0),
                            Complex::new(12.5, -12.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.5, -13.0),
                            Complex::new(14.5, -14.0),
                            Complex::new(15.5, -15.0),
                            Complex::new(16.5, -16.0),
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
                            Complex::new(17.5, -17.0),
                            Complex::new(18.5, -18.0),
                            Complex::new(19.5, -19.0),
                            Complex::new(20.5, -20.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.5, -21.0),
                            Complex::new(22.5, -22.0),
                            Complex::new(23.5, -23.0),
                            Complex::new(24.5, -24.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.5, -25.0),
                            Complex::new(26.5, -26.0),
                            Complex::new(27.5, -27.0),
                            Complex::new(28.5, -28.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.5, -29.0),
                            Complex::new(30.5, -30.0),
                            Complex::new(31.5, -31.0),
                            Complex::new(32.5, -32.0),
                        ],
                    ),
                ],
            ],
        ];
        assert_eq!(
            head(&x, 1).unwrap(),
            vec![x[0].clone()],
            "head(array[,,] complex_matrix, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(num_elements(&x), 2, "num_elements(vector)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(num_elements(&x), 2, "num_elements(row_vector)");
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]);
        assert_eq!(num_elements(&x), 4, "num_elements(matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(num_elements(&x), 2, "num_elements(complex_vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(num_elements(&x), 2, "num_elements(complex_row_vector)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.5, -1.0),
                Complex::new(2.5, -2.0),
                Complex::new(3.5, -3.0),
                Complex::new(4.5, -4.0),
            ],
        );
        assert_eq!(num_elements(&x), 4, "num_elements(complex_matrix)");
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        assert_eq!(num_elements(&x), 2, "num_elements(array[] int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(num_elements(&x), 2, "num_elements(array[] real)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(num_elements(&x), 4, "num_elements(array[] vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(num_elements(&x), 2, "num_elements(array[] complex)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(num_elements(&x), 4, "num_elements(array[] row_vector)");
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        assert_eq!(num_elements(&x), 8, "num_elements(array[] matrix)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(num_elements(&x), 4, "num_elements(array[] complex_vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            num_elements(&x),
            4,
            "num_elements(array[] complex_row_vector)"
        );
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        assert_eq!(num_elements(&x), 8, "num_elements(array[] complex_matrix)");
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        assert_eq!(num_elements(&x), 4, "num_elements(array[,] int)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(num_elements(&x), 4, "num_elements(array[,] real)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(num_elements(&x), 8, "num_elements(array[,] vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(num_elements(&x), 4, "num_elements(array[,] complex)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(num_elements(&x), 8, "num_elements(array[,] row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        assert_eq!(num_elements(&x), 16, "num_elements(array[,] matrix)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(num_elements(&x), 8, "num_elements(array[,] complex_vector)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            num_elements(&x),
            8,
            "num_elements(array[,] complex_row_vector)"
        );
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        assert_eq!(
            num_elements(&x),
            16,
            "num_elements(array[,] complex_matrix)"
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]],
            vec![vec![5_i64, 6_i64], vec![7_i64, 8_i64]],
        ];
        assert_eq!(num_elements(&x), 8, "num_elements(array[,,] int)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(num_elements(&x), 8, "num_elements(array[,,] real)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(num_elements(&x), 8, "num_elements(array[,,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                    Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                    Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.5, 18.5, 19.5, 20.5]),
                    Matrix::from_column_major(2, 2, vec![21.5, 22.5, 23.5, 24.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.5, 26.5, 27.5, 28.5]),
                    Matrix::from_column_major(2, 2, vec![29.5, 30.5, 31.5, 32.5]),
                ],
            ],
        ];
        assert_eq!(num_elements(&x), 32, "num_elements(array[,,] matrix)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.5, -1.0),
                            Complex::new(2.5, -2.0),
                            Complex::new(3.5, -3.0),
                            Complex::new(4.5, -4.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.5, -5.0),
                            Complex::new(6.5, -6.0),
                            Complex::new(7.5, -7.0),
                            Complex::new(8.5, -8.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.5, -9.0),
                            Complex::new(10.5, -10.0),
                            Complex::new(11.5, -11.0),
                            Complex::new(12.5, -12.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.5, -13.0),
                            Complex::new(14.5, -14.0),
                            Complex::new(15.5, -15.0),
                            Complex::new(16.5, -16.0),
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
                            Complex::new(17.5, -17.0),
                            Complex::new(18.5, -18.0),
                            Complex::new(19.5, -19.0),
                            Complex::new(20.5, -20.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.5, -21.0),
                            Complex::new(22.5, -22.0),
                            Complex::new(23.5, -23.0),
                            Complex::new(24.5, -24.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.5, -25.0),
                            Complex::new(26.5, -26.0),
                            Complex::new(27.5, -27.0),
                            Complex::new(28.5, -28.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.5, -29.0),
                            Complex::new(30.5, -30.0),
                            Complex::new(31.5, -31.0),
                            Complex::new(32.5, -32.0),
                        ],
                    ),
                ],
            ],
        ];
        assert_eq!(
            num_elements(&x),
            32,
            "num_elements(array[,,] complex_matrix)"
        );
    }
    {
        let x: i64 = 1_i64;
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3], "rep_array(int, int)");
    }
    {
        let x: i64 = 1_i64;
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(int, int, int)");
    }
    {
        let x: i64 = 1_i64;
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(dims(&r), vec![3, 2, 4], "rep_array(int, int, int, int)");
    }
    {
        let x: f64 = 1.5;
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3], "rep_array(real, int)");
    }
    {
        let x: f64 = 1.5;
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(real, int, int)");
    }
    {
        let x: f64 = 1.5;
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(dims(&r), vec![3, 2, 4], "rep_array(real, int, int, int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(vector, int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(vector, int, int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2],
            "rep_array(vector, int, int, int)"
        );
    }
    {
        let x: Complex<f64> = Complex::new(1.5, -1.0);
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3], "rep_array(complex, int)");
    }
    {
        let x: Complex<f64> = Complex::new(1.5, -1.0);
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(complex, int, int)");
    }
    {
        let x: Complex<f64> = Complex::new(1.5, -1.0);
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(dims(&r), vec![3, 2, 4], "rep_array(complex, int, int, int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(row_vector, int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(row_vector, int, int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2],
            "rep_array(row_vector, int, int, int)"
        );
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]);
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(matrix, int)");
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]);
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(dims(&r), vec![3, 2, 2, 2], "rep_array(matrix, int, int)");
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]);
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(matrix, int, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(complex_vector, int)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2],
            "rep_array(complex_vector, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2],
            "rep_array(complex_vector, int, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(complex_row_vector, int)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2],
            "rep_array(complex_row_vector, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2],
            "rep_array(complex_row_vector, int, int, int)"
        );
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.5, -1.0),
                Complex::new(2.5, -2.0),
                Complex::new(3.5, -3.0),
                Complex::new(4.5, -4.0),
            ],
        );
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(complex_matrix, int)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.5, -1.0),
                Complex::new(2.5, -2.0),
                Complex::new(3.5, -3.0),
                Complex::new(4.5, -4.0),
            ],
        );
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(complex_matrix, int, int)"
        );
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.5, -1.0),
                Complex::new(2.5, -2.0),
                Complex::new(3.5, -3.0),
                Complex::new(4.5, -4.0),
            ],
        );
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(complex_matrix, int, int, int)"
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(array[] int, int)");
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(array[] int, int, int)");
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2],
            "rep_array(array[] int, int, int, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(array[] real, int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(array[] real, int, int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2],
            "rep_array(array[] real, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(array[] vector, int)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[] vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(array[] vector, int, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2], "rep_array(array[] complex, int)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2],
            "rep_array(array[] complex, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2],
            "rep_array(array[] complex, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2],
            "rep_array(array[] row_vector, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[] row_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(array[] row_vector, int, int, int)"
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2, 2, 2], "rep_array(array[] matrix, int)");
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2],
            "rep_array(array[] matrix, int, int)"
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2, 2],
            "rep_array(array[] matrix, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2],
            "rep_array(array[] complex_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[] complex_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(array[] complex_vector, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2],
            "rep_array(array[] complex_row_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[] complex_row_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(array[] complex_row_vector, int, int, int)"
        );
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[] complex_matrix, int)"
        );
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2],
            "rep_array(array[] complex_matrix, int, int)"
        );
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2, 2],
            "rep_array(array[] complex_matrix, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(array[,] int, int)");
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[,] int, int, int)"
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(array[,] int, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(array[,] real, int)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[,] real, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(array[,] real, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[,] vector, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2],
            "rep_array(array[,] vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2, 2],
            "rep_array(array[,] vector, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(dims(&r), vec![3, 2, 2], "rep_array(array[,] complex, int)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[,] complex, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2],
            "rep_array(array[,] complex, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[,] row_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2],
            "rep_array(array[,] row_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2, 2],
            "rep_array(array[,] row_vector, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2],
            "rep_array(array[,] matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2, 2],
            "rep_array(array[,] matrix, int, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2, 2, 2],
            "rep_array(array[,] matrix, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[,] complex_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2],
            "rep_array(array[,] complex_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2, 2],
            "rep_array(array[,] complex_vector, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2],
            "rep_array(array[,] complex_row_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2],
            "rep_array(array[,] complex_row_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2, 2],
            "rep_array(array[,] complex_row_vector, int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        let r = rep_array(x.clone(), 3);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2],
            "rep_array(array[,] complex_matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        let r = rep_array_2d(x.clone(), 3, 2);
        assert_eq!(
            dims(&r),
            vec![3, 2, 2, 2, 2, 2],
            "rep_array(array[,] complex_matrix, int, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        let r = rep_array_3d(x.clone(), 3, 2, 4);
        assert_eq!(
            dims(&r),
            vec![3, 2, 4, 2, 2, 2, 2],
            "rep_array(array[,] complex_matrix, int, int, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(row_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(complex_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(complex_row_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] int)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] real)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] complex)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] row_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] matrix)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] complex_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] complex_row_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[] complex_matrix)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] int)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] real)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] complex)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] row_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] matrix)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] complex_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] complex_row_vector)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,] complex_matrix)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]],
            vec![vec![5_i64, 6_i64], vec![7_i64, 8_i64]],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,,] int)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,,] real)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,,] complex)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                    Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                    Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.5, 18.5, 19.5, 20.5]),
                    Matrix::from_column_major(2, 2, vec![21.5, 22.5, 23.5, 24.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.5, 26.5, 27.5, 28.5]),
                    Matrix::from_column_major(2, 2, vec![29.5, 30.5, 31.5, 32.5]),
                ],
            ],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,,] matrix)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.5, -1.0),
                            Complex::new(2.5, -2.0),
                            Complex::new(3.5, -3.0),
                            Complex::new(4.5, -4.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.5, -5.0),
                            Complex::new(6.5, -6.0),
                            Complex::new(7.5, -7.0),
                            Complex::new(8.5, -8.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.5, -9.0),
                            Complex::new(10.5, -10.0),
                            Complex::new(11.5, -11.0),
                            Complex::new(12.5, -12.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.5, -13.0),
                            Complex::new(14.5, -14.0),
                            Complex::new(15.5, -15.0),
                            Complex::new(16.5, -16.0),
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
                            Complex::new(17.5, -17.0),
                            Complex::new(18.5, -18.0),
                            Complex::new(19.5, -19.0),
                            Complex::new(20.5, -20.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.5, -21.0),
                            Complex::new(22.5, -22.0),
                            Complex::new(23.5, -23.0),
                            Complex::new(24.5, -24.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.5, -25.0),
                            Complex::new(26.5, -26.0),
                            Complex::new(27.5, -27.0),
                            Complex::new(28.5, -28.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.5, -29.0),
                            Complex::new(30.5, -30.0),
                            Complex::new(31.5, -31.0),
                            Complex::new(32.5, -32.0),
                        ],
                    ),
                ],
            ],
        ];
        let r = reverse(&x);
        assert_eq!(r[0], x[1], "reverse(array[,,] complex_matrix)");
        assert_eq!(reverse(&r), x);
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(vector, int, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(row_vector, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(complex_vector, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(complex_row_vector, int, int)"
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] int, int, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] real, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] vector, int, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] complex, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] row_vector, int, int)"
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] matrix, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] complex_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] complex_row_vector, int, int)"
        );
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[] complex_matrix, int, int)"
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] int, int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] real, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] complex, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] row_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] matrix, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] complex_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] complex_row_vector, int, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,] complex_matrix, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]],
            vec![vec![5_i64, 6_i64], vec![7_i64, 8_i64]],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,,] int, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,,] real, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,,] complex, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                    Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                    Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.5, 18.5, 19.5, 20.5]),
                    Matrix::from_column_major(2, 2, vec![21.5, 22.5, 23.5, 24.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.5, 26.5, 27.5, 28.5]),
                    Matrix::from_column_major(2, 2, vec![29.5, 30.5, 31.5, 32.5]),
                ],
            ],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,,] matrix, int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.5, -1.0),
                            Complex::new(2.5, -2.0),
                            Complex::new(3.5, -3.0),
                            Complex::new(4.5, -4.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.5, -5.0),
                            Complex::new(6.5, -6.0),
                            Complex::new(7.5, -7.0),
                            Complex::new(8.5, -8.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.5, -9.0),
                            Complex::new(10.5, -10.0),
                            Complex::new(11.5, -11.0),
                            Complex::new(12.5, -12.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.5, -13.0),
                            Complex::new(14.5, -14.0),
                            Complex::new(15.5, -15.0),
                            Complex::new(16.5, -16.0),
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
                            Complex::new(17.5, -17.0),
                            Complex::new(18.5, -18.0),
                            Complex::new(19.5, -19.0),
                            Complex::new(20.5, -20.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.5, -21.0),
                            Complex::new(22.5, -22.0),
                            Complex::new(23.5, -23.0),
                            Complex::new(24.5, -24.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.5, -25.0),
                            Complex::new(26.5, -26.0),
                            Complex::new(27.5, -27.0),
                            Complex::new(28.5, -28.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.5, -29.0),
                            Complex::new(30.5, -30.0),
                            Complex::new(31.5, -31.0),
                            Complex::new(32.5, -32.0),
                        ],
                    ),
                ],
            ],
        ];
        assert_eq!(
            segment(&x, 2, 1).unwrap(),
            vec![x[1].clone()],
            "segment(array[,,] complex_matrix, int, int)"
        );
    }
    {
        let x: i64 = 1_i64;
        assert_eq!(num_elements(&x), 1, "size(int)");
    }
    {
        let x: f64 = 1.5;
        assert_eq!(num_elements(&x), 1, "size(real)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(num_elements(&x), 2, "size(vector)");
    }
    {
        let x: Complex<f64> = Complex::new(1.5, -1.0);
        assert_eq!(num_elements(&x), 1, "size(complex)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(num_elements(&x), 2, "size(row_vector)");
    }
    {
        let x: Matrix<f64> = Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]);
        assert_eq!(num_elements(&x), 4, "size(matrix)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(num_elements(&x), 2, "size(complex_vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(num_elements(&x), 2, "size(complex_row_vector)");
    }
    {
        let x: Matrix<Complex<f64>> = Matrix::from_column_major(
            2,
            2,
            vec![
                Complex::new(1.5, -1.0),
                Complex::new(2.5, -2.0),
                Complex::new(3.5, -3.0),
                Complex::new(4.5, -4.0),
            ],
        );
        assert_eq!(num_elements(&x), 4, "size(complex_matrix)");
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        assert_eq!(size(&x), 2, "size(array[] int)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(size(&x), 2, "size(array[] real)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(size(&x), 2, "size(array[] vector)");
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(size(&x), 2, "size(array[] complex)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(size(&x), 2, "size(array[] row_vector)");
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        assert_eq!(size(&x), 2, "size(array[] matrix)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(size(&x), 2, "size(array[] complex_vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(size(&x), 2, "size(array[] complex_row_vector)");
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        assert_eq!(size(&x), 2, "size(array[] complex_matrix)");
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        assert_eq!(size(&x), 2, "size(array[,] int)");
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(size(&x), 2, "size(array[,] real)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(size(&x), 2, "size(array[,] vector)");
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(size(&x), 2, "size(array[,] complex)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(size(&x), 2, "size(array[,] row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        assert_eq!(size(&x), 2, "size(array[,] matrix)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(size(&x), 2, "size(array[,] complex_vector)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(size(&x), 2, "size(array[,] complex_row_vector)");
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        assert_eq!(size(&x), 2, "size(array[,] complex_matrix)");
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]],
            vec![vec![5_i64, 6_i64], vec![7_i64, 8_i64]],
        ];
        assert_eq!(size(&x), 2, "size(array[,,] int)");
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(size(&x), 2, "size(array[,,] real)");
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(size(&x), 2, "size(array[,,] complex)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                    Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                    Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.5, 18.5, 19.5, 20.5]),
                    Matrix::from_column_major(2, 2, vec![21.5, 22.5, 23.5, 24.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.5, 26.5, 27.5, 28.5]),
                    Matrix::from_column_major(2, 2, vec![29.5, 30.5, 31.5, 32.5]),
                ],
            ],
        ];
        assert_eq!(size(&x), 2, "size(array[,,] matrix)");
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.5, -1.0),
                            Complex::new(2.5, -2.0),
                            Complex::new(3.5, -3.0),
                            Complex::new(4.5, -4.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.5, -5.0),
                            Complex::new(6.5, -6.0),
                            Complex::new(7.5, -7.0),
                            Complex::new(8.5, -8.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.5, -9.0),
                            Complex::new(10.5, -10.0),
                            Complex::new(11.5, -11.0),
                            Complex::new(12.5, -12.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.5, -13.0),
                            Complex::new(14.5, -14.0),
                            Complex::new(15.5, -15.0),
                            Complex::new(16.5, -16.0),
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
                            Complex::new(17.5, -17.0),
                            Complex::new(18.5, -18.0),
                            Complex::new(19.5, -19.0),
                            Complex::new(20.5, -20.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.5, -21.0),
                            Complex::new(22.5, -22.0),
                            Complex::new(23.5, -23.0),
                            Complex::new(24.5, -24.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.5, -25.0),
                            Complex::new(26.5, -26.0),
                            Complex::new(27.5, -27.0),
                            Complex::new(28.5, -28.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.5, -29.0),
                            Complex::new(30.5, -30.0),
                            Complex::new(31.5, -31.0),
                            Complex::new(32.5, -32.0),
                        ],
                    ),
                ],
            ],
        ];
        assert_eq!(size(&x), 2, "size(array[,,] complex_matrix)");
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(vector, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(row_vector, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(complex_vector, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(complex_row_vector, int)"
        );
    }
    {
        let x: Vec<i64> = vec![1_i64, 2_i64];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] int, int)"
        );
    }
    {
        let x: Vec<f64> = vec![1.5, 2.5];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] real, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] vector, int)"
        );
    }
    {
        let x: Vec<Complex<f64>> = vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] complex, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] row_vector, int)"
        );
    }
    {
        let x: Vec<Matrix<f64>> = vec![
            Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
            Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] complex_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] complex_row_vector, int)"
        );
    }
    {
        let x: Vec<Matrix<Complex<f64>>> = vec![
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(1.5, -1.0),
                    Complex::new(2.5, -2.0),
                    Complex::new(3.5, -3.0),
                    Complex::new(4.5, -4.0),
                ],
            ),
            Matrix::from_column_major(
                2,
                2,
                vec![
                    Complex::new(5.5, -5.0),
                    Complex::new(6.5, -6.0),
                    Complex::new(7.5, -7.0),
                    Complex::new(8.5, -8.0),
                ],
            ),
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[] complex_matrix, int)"
        );
    }
    {
        let x: Vec<Vec<i64>> = vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] int, int)"
        );
    }
    {
        let x: Vec<Vec<f64>> = vec![vec![1.5, 2.5], vec![3.5, 4.5]];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] real, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] vector, int)"
        );
    }
    {
        let x: Vec<Vec<Complex<f64>>> = vec![
            vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
            vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] complex, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] row_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<f64>>> = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
            ],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] complex_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] complex_row_vector, int)"
        );
    }
    {
        let x: Vec<Vec<Matrix<Complex<f64>>>> = vec![
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(1.5, -1.0),
                        Complex::new(2.5, -2.0),
                        Complex::new(3.5, -3.0),
                        Complex::new(4.5, -4.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(5.5, -5.0),
                        Complex::new(6.5, -6.0),
                        Complex::new(7.5, -7.0),
                        Complex::new(8.5, -8.0),
                    ],
                ),
            ],
            vec![
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(9.5, -9.0),
                        Complex::new(10.5, -10.0),
                        Complex::new(11.5, -11.0),
                        Complex::new(12.5, -12.0),
                    ],
                ),
                Matrix::from_column_major(
                    2,
                    2,
                    vec![
                        Complex::new(13.5, -13.0),
                        Complex::new(14.5, -14.0),
                        Complex::new(15.5, -15.0),
                        Complex::new(16.5, -16.0),
                    ],
                ),
            ],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,] complex_matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<i64>>> = vec![
            vec![vec![1_i64, 2_i64], vec![3_i64, 4_i64]],
            vec![vec![5_i64, 6_i64], vec![7_i64, 8_i64]],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,,] int, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.5, 2.5], vec![3.5, 4.5]],
            vec![vec![5.5, 6.5], vec![7.5, 8.5]],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,,] real, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Complex<f64>>>> = vec![
            vec![
                vec![Complex::new(1.5, -1.0), Complex::new(2.5, -2.0)],
                vec![Complex::new(3.5, -3.0), Complex::new(4.5, -4.0)],
            ],
            vec![
                vec![Complex::new(5.5, -5.0), Complex::new(6.5, -6.0)],
                vec![Complex::new(7.5, -7.0), Complex::new(8.5, -8.0)],
            ],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,,] complex, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<f64>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![1.5, 2.5, 3.5, 4.5]),
                    Matrix::from_column_major(2, 2, vec![5.5, 6.5, 7.5, 8.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![9.5, 10.5, 11.5, 12.5]),
                    Matrix::from_column_major(2, 2, vec![13.5, 14.5, 15.5, 16.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![17.5, 18.5, 19.5, 20.5]),
                    Matrix::from_column_major(2, 2, vec![21.5, 22.5, 23.5, 24.5]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![25.5, 26.5, 27.5, 28.5]),
                    Matrix::from_column_major(2, 2, vec![29.5, 30.5, 31.5, 32.5]),
                ],
            ],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,,] matrix, int)"
        );
    }
    {
        let x: Vec<Vec<Vec<Matrix<Complex<f64>>>>> = vec![
            vec![
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(1.5, -1.0),
                            Complex::new(2.5, -2.0),
                            Complex::new(3.5, -3.0),
                            Complex::new(4.5, -4.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(5.5, -5.0),
                            Complex::new(6.5, -6.0),
                            Complex::new(7.5, -7.0),
                            Complex::new(8.5, -8.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(9.5, -9.0),
                            Complex::new(10.5, -10.0),
                            Complex::new(11.5, -11.0),
                            Complex::new(12.5, -12.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(13.5, -13.0),
                            Complex::new(14.5, -14.0),
                            Complex::new(15.5, -15.0),
                            Complex::new(16.5, -16.0),
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
                            Complex::new(17.5, -17.0),
                            Complex::new(18.5, -18.0),
                            Complex::new(19.5, -19.0),
                            Complex::new(20.5, -20.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(21.5, -21.0),
                            Complex::new(22.5, -22.0),
                            Complex::new(23.5, -23.0),
                            Complex::new(24.5, -24.0),
                        ],
                    ),
                ],
                vec![
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(25.5, -25.0),
                            Complex::new(26.5, -26.0),
                            Complex::new(27.5, -27.0),
                            Complex::new(28.5, -28.0),
                        ],
                    ),
                    Matrix::from_column_major(
                        2,
                        2,
                        vec![
                            Complex::new(29.5, -29.0),
                            Complex::new(30.5, -30.0),
                            Complex::new(31.5, -31.0),
                            Complex::new(32.5, -32.0),
                        ],
                    ),
                ],
            ],
        ];
        assert_eq!(
            tail(&x, 1).unwrap(),
            vec![x[1].clone()],
            "tail(array[,,] complex_matrix, int)"
        );
    }
}
