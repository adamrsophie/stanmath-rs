//! Every array signature of the structural transforms in the Stan language (`_constrain`,
//! `_jacobian`, and `_unconstrain` of simplex, sum_to_zero, ordered, positive_ordered,
//! unit_vector, stochastic_row, stochastic_column, cov_matrix, corr_matrix, cholesky_factor_corr,
//! and cholesky_factor_cov).
//!
//! Written by `tools/reference/gen_transforms.py` from the CmdStan 2.40.0 signature dump: 108
//! Stan signatures in 108 Rust shapes. Stan's array forms apply the transform to each element;
//! in Rust they are an iterator map over the array, and a `_jacobian` map adds every element's log
//! Jacobian to one `lp`. Each call must keep the array's shape and equal the transform of every
//! element in order, and its `lp` the sum of theirs; the single transforms are compared with Stan
//! Math by `stan_reference_matrix.rs`. The other 144 signatures nest arrays more than 3
//! deep and are the same nested map.
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::redundant_closure,
    clippy::useless_vec
)]

use stanmath_rs::prelude::*;

#[test]
fn transform_array_signatures() {
    let mut failures = Vec::new();
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let got = x
            .iter()
            .map(|v| cholesky_factor_corr_constrain(v, 3).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_constrain(v, 3).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_constrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cholesky_factor_corr_constrain(v, 3).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_constrain(v, 3).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cholesky_factor_corr_constrain(v, 3).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_constrain(v, 3).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| cholesky_factor_corr_jacobian(v, 3, &mut lp).unwrap())
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cholesky_factor_corr_jacobian(array vec) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_jacobian(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cholesky_factor_corr_jacobian(v, 3, &mut lp).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cholesky_factor_corr_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cholesky_factor_corr_jacobian(v, 3, &mut lp).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cholesky_factor_corr_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            cholesky_factor_corr_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
        ];
        let flat_x: Vec<_> = vec![
            cholesky_factor_corr_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|v| cholesky_factor_corr_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                cholesky_factor_corr_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
                cholesky_factor_corr_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
            ],
            vec![
                cholesky_factor_corr_constrain(&vec![0.5, -0.4, 1.1], 3).unwrap(),
                cholesky_factor_corr_constrain(&vec![0.0, 0.9, -0.6], 3).unwrap(),
            ],
        ];
        let flat_x: Vec<_> = vec![
            cholesky_factor_corr_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![0.5, -0.4, 1.1], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![0.0, 0.9, -0.6], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cholesky_factor_corr_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    cholesky_factor_corr_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
                    cholesky_factor_corr_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
                ],
                vec![
                    cholesky_factor_corr_constrain(&vec![0.5, -0.4, 1.1], 3).unwrap(),
                    cholesky_factor_corr_constrain(&vec![0.0, 0.9, -0.6], 3).unwrap(),
                ],
            ],
            vec![
                vec![
                    cholesky_factor_corr_constrain(&vec![0.7, -0.2, 1.3], 3).unwrap(),
                    cholesky_factor_corr_constrain(&vec![0.2, 1.1, -0.4], 3).unwrap(),
                ],
                vec![
                    cholesky_factor_corr_constrain(&vec![0.9, 0.0, 1.5], 3).unwrap(),
                    cholesky_factor_corr_constrain(&vec![0.4, 1.3, -0.2], 3).unwrap(),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            cholesky_factor_corr_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![0.5, -0.4, 1.1], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![0.0, 0.9, -0.6], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![0.7, -0.2, 1.3], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![0.2, 1.1, -0.4], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![0.9, 0.0, 1.5], 3).unwrap(),
            cholesky_factor_corr_constrain(&vec![0.4, 1.3, -0.2], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cholesky_factor_corr_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_corr_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_corr_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4],
        ];
        let got = x
            .iter()
            .map(|v| cholesky_factor_cov_constrain(v, 3, 2).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_constrain(v, 3, 2).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_constrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![0.3, -0.6, 0.9, -1.2, 1.5],
                vec![-0.2, 0.7, -0.8, 1.3, -1.4],
            ],
            vec![
                vec![0.5, -0.4, 1.1, -1.0, 1.7],
                vec![0.0, 0.9, -0.6, 1.5, -1.2],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4],
            vec![0.5, -0.4, 1.1, -1.0, 1.7],
            vec![0.0, 0.9, -0.6, 1.5, -1.2],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cholesky_factor_cov_constrain(v, 3, 2).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_constrain(v, 3, 2).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    vec![0.3, -0.6, 0.9, -1.2, 1.5],
                    vec![-0.2, 0.7, -0.8, 1.3, -1.4],
                ],
                vec![
                    vec![0.5, -0.4, 1.1, -1.0, 1.7],
                    vec![0.0, 0.9, -0.6, 1.5, -1.2],
                ],
            ],
            vec![
                vec![
                    vec![0.7, -0.2, 1.3, -0.8, 1.9],
                    vec![0.2, 1.1, -0.4, 1.7, -1.0],
                ],
                vec![
                    vec![0.9, 0.0, 1.5, -0.6, 2.1],
                    vec![0.4, 1.3, -0.2, 1.9, -0.8],
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4],
            vec![0.5, -0.4, 1.1, -1.0, 1.7],
            vec![0.0, 0.9, -0.6, 1.5, -1.2],
            vec![0.7, -0.2, 1.3, -0.8, 1.9],
            vec![0.2, 1.1, -0.4, 1.7, -1.0],
            vec![0.9, 0.0, 1.5, -0.6, 2.1],
            vec![0.4, 1.3, -0.2, 1.9, -0.8],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cholesky_factor_cov_constrain(v, 3, 2).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_constrain(v, 3, 2).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| cholesky_factor_cov_jacobian(v, 3, 2, &mut lp).unwrap())
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_jacobian(v, 3, 2, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cholesky_factor_cov_jacobian(array vec) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_jacobian(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![0.3, -0.6, 0.9, -1.2, 1.5],
                vec![-0.2, 0.7, -0.8, 1.3, -1.4],
            ],
            vec![
                vec![0.5, -0.4, 1.1, -1.0, 1.7],
                vec![0.0, 0.9, -0.6, 1.5, -1.2],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4],
            vec![0.5, -0.4, 1.1, -1.0, 1.7],
            vec![0.0, 0.9, -0.6, 1.5, -1.2],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cholesky_factor_cov_jacobian(v, 3, 2, &mut lp).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_jacobian(v, 3, 2, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cholesky_factor_cov_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    vec![0.3, -0.6, 0.9, -1.2, 1.5],
                    vec![-0.2, 0.7, -0.8, 1.3, -1.4],
                ],
                vec![
                    vec![0.5, -0.4, 1.1, -1.0, 1.7],
                    vec![0.0, 0.9, -0.6, 1.5, -1.2],
                ],
            ],
            vec![
                vec![
                    vec![0.7, -0.2, 1.3, -0.8, 1.9],
                    vec![0.2, 1.1, -0.4, 1.7, -1.0],
                ],
                vec![
                    vec![0.9, 0.0, 1.5, -0.6, 2.1],
                    vec![0.4, 1.3, -0.2, 1.9, -0.8],
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4],
            vec![0.5, -0.4, 1.1, -1.0, 1.7],
            vec![0.0, 0.9, -0.6, 1.5, -1.2],
            vec![0.7, -0.2, 1.3, -0.8, 1.9],
            vec![0.2, 1.1, -0.4, 1.7, -1.0],
            vec![0.9, 0.0, 1.5, -0.6, 2.1],
            vec![0.4, 1.3, -0.2, 1.9, -0.8],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cholesky_factor_cov_jacobian(v, 3, 2, &mut lp).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_jacobian(v, 3, 2, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cholesky_factor_cov_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            cholesky_factor_cov_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4], 3, 2).unwrap(),
        ];
        let flat_x: Vec<_> = vec![
            cholesky_factor_cov_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4], 3, 2).unwrap(),
        ];
        let got = x
            .iter()
            .map(|v| cholesky_factor_cov_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                cholesky_factor_cov_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5], 3, 2).unwrap(),
                cholesky_factor_cov_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4], 3, 2).unwrap(),
            ],
            vec![
                cholesky_factor_cov_constrain(&vec![0.5, -0.4, 1.1, -1.0, 1.7], 3, 2).unwrap(),
                cholesky_factor_cov_constrain(&vec![0.0, 0.9, -0.6, 1.5, -1.2], 3, 2).unwrap(),
            ],
        ];
        let flat_x: Vec<_> = vec![
            cholesky_factor_cov_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![0.5, -0.4, 1.1, -1.0, 1.7], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![0.0, 0.9, -0.6, 1.5, -1.2], 3, 2).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cholesky_factor_cov_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    cholesky_factor_cov_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5], 3, 2).unwrap(),
                    cholesky_factor_cov_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4], 3, 2).unwrap(),
                ],
                vec![
                    cholesky_factor_cov_constrain(&vec![0.5, -0.4, 1.1, -1.0, 1.7], 3, 2).unwrap(),
                    cholesky_factor_cov_constrain(&vec![0.0, 0.9, -0.6, 1.5, -1.2], 3, 2).unwrap(),
                ],
            ],
            vec![
                vec![
                    cholesky_factor_cov_constrain(&vec![0.7, -0.2, 1.3, -0.8, 1.9], 3, 2).unwrap(),
                    cholesky_factor_cov_constrain(&vec![0.2, 1.1, -0.4, 1.7, -1.0], 3, 2).unwrap(),
                ],
                vec![
                    cholesky_factor_cov_constrain(&vec![0.9, 0.0, 1.5, -0.6, 2.1], 3, 2).unwrap(),
                    cholesky_factor_cov_constrain(&vec![0.4, 1.3, -0.2, 1.9, -0.8], 3, 2).unwrap(),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            cholesky_factor_cov_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![0.5, -0.4, 1.1, -1.0, 1.7], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![0.0, 0.9, -0.6, 1.5, -1.2], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![0.7, -0.2, 1.3, -0.8, 1.9], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![0.2, 1.1, -0.4, 1.7, -1.0], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![0.9, 0.0, 1.5, -0.6, 2.1], 3, 2).unwrap(),
            cholesky_factor_cov_constrain(&vec![0.4, 1.3, -0.2, 1.9, -0.8], 3, 2).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cholesky_factor_cov_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cholesky_factor_cov_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cholesky_factor_cov_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let got = x
            .iter()
            .map(|v| corr_matrix_constrain(v, 3).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_constrain(v, 3).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_constrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| corr_matrix_constrain(v, 3).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_constrain(v, 3).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| corr_matrix_constrain(v, 3).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_constrain(v, 3).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| corr_matrix_jacobian(v, 3, &mut lp).unwrap())
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "corr_matrix_jacobian(array vec) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_jacobian(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| corr_matrix_jacobian(v, 3, &mut lp).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "corr_matrix_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| corr_matrix_jacobian(v, 3, &mut lp).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "corr_matrix_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            corr_matrix_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
            corr_matrix_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
        ];
        let flat_x: Vec<_> = vec![
            corr_matrix_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
            corr_matrix_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|v| corr_matrix_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                corr_matrix_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
                corr_matrix_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
            ],
            vec![
                corr_matrix_constrain(&vec![0.5, -0.4, 1.1], 3).unwrap(),
                corr_matrix_constrain(&vec![0.0, 0.9, -0.6], 3).unwrap(),
            ],
        ];
        let flat_x: Vec<_> = vec![
            corr_matrix_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
            corr_matrix_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
            corr_matrix_constrain(&vec![0.5, -0.4, 1.1], 3).unwrap(),
            corr_matrix_constrain(&vec![0.0, 0.9, -0.6], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| corr_matrix_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    corr_matrix_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
                    corr_matrix_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
                ],
                vec![
                    corr_matrix_constrain(&vec![0.5, -0.4, 1.1], 3).unwrap(),
                    corr_matrix_constrain(&vec![0.0, 0.9, -0.6], 3).unwrap(),
                ],
            ],
            vec![
                vec![
                    corr_matrix_constrain(&vec![0.7, -0.2, 1.3], 3).unwrap(),
                    corr_matrix_constrain(&vec![0.2, 1.1, -0.4], 3).unwrap(),
                ],
                vec![
                    corr_matrix_constrain(&vec![0.9, 0.0, 1.5], 3).unwrap(),
                    corr_matrix_constrain(&vec![0.4, 1.3, -0.2], 3).unwrap(),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            corr_matrix_constrain(&vec![0.3, -0.6, 0.9], 3).unwrap(),
            corr_matrix_constrain(&vec![-0.2, 0.7, -0.8], 3).unwrap(),
            corr_matrix_constrain(&vec![0.5, -0.4, 1.1], 3).unwrap(),
            corr_matrix_constrain(&vec![0.0, 0.9, -0.6], 3).unwrap(),
            corr_matrix_constrain(&vec![0.7, -0.2, 1.3], 3).unwrap(),
            corr_matrix_constrain(&vec![0.2, 1.1, -0.4], 3).unwrap(),
            corr_matrix_constrain(&vec![0.9, 0.0, 1.5], 3).unwrap(),
            corr_matrix_constrain(&vec![0.4, 1.3, -0.2], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| corr_matrix_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| corr_matrix_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "corr_matrix_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
        ];
        let got = x
            .iter()
            .map(|v| cov_matrix_constrain(v, 3).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_constrain(v, 3).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_constrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
                vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
            ],
            vec![
                vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6],
                vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
            vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6],
            vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cov_matrix_constrain(v, 3).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_constrain(v, 3).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
                    vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
                ],
                vec![
                    vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6],
                    vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1],
                ],
            ],
            vec![
                vec![
                    vec![0.7, -0.2, 1.3, -0.8, 1.9, -1.4],
                    vec![0.2, 1.1, -0.4, 1.7, -1.0, 2.3],
                ],
                vec![
                    vec![0.9, 0.0, 1.5, -0.6, 2.1, -1.2],
                    vec![0.4, 1.3, -0.2, 1.9, -0.8, 2.5],
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
            vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6],
            vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1],
            vec![0.7, -0.2, 1.3, -0.8, 1.9, -1.4],
            vec![0.2, 1.1, -0.4, 1.7, -1.0, 2.3],
            vec![0.9, 0.0, 1.5, -0.6, 2.1, -1.2],
            vec![0.4, 1.3, -0.2, 1.9, -0.8, 2.5],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cov_matrix_constrain(v, 3).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_constrain(v, 3).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| cov_matrix_jacobian(v, 3, &mut lp).unwrap())
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cov_matrix_jacobian(array vec) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_jacobian(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
                vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
            ],
            vec![
                vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6],
                vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
            vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6],
            vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cov_matrix_jacobian(v, 3, &mut lp).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cov_matrix_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
                    vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
                ],
                vec![
                    vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6],
                    vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1],
                ],
            ],
            vec![
                vec![
                    vec![0.7, -0.2, 1.3, -0.8, 1.9, -1.4],
                    vec![0.2, 1.1, -0.4, 1.7, -1.0, 2.3],
                ],
                vec![
                    vec![0.9, 0.0, 1.5, -0.6, 2.1, -1.2],
                    vec![0.4, 1.3, -0.2, 1.9, -0.8, 2.5],
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8],
            vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9],
            vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6],
            vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1],
            vec![0.7, -0.2, 1.3, -0.8, 1.9, -1.4],
            vec![0.2, 1.1, -0.4, 1.7, -1.0, 2.3],
            vec![0.9, 0.0, 1.5, -0.6, 2.1, -1.2],
            vec![0.4, 1.3, -0.2, 1.9, -0.8, 2.5],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cov_matrix_jacobian(v, 3, &mut lp).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_jacobian(v, 3, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "cov_matrix_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            cov_matrix_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8], 3).unwrap(),
            cov_matrix_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9], 3).unwrap(),
        ];
        let flat_x: Vec<_> = vec![
            cov_matrix_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8], 3).unwrap(),
            cov_matrix_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|v| cov_matrix_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                cov_matrix_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8], 3).unwrap(),
                cov_matrix_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9], 3).unwrap(),
            ],
            vec![
                cov_matrix_constrain(&vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6], 3).unwrap(),
                cov_matrix_constrain(&vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1], 3).unwrap(),
            ],
        ];
        let flat_x: Vec<_> = vec![
            cov_matrix_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8], 3).unwrap(),
            cov_matrix_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9], 3).unwrap(),
            cov_matrix_constrain(&vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6], 3).unwrap(),
            cov_matrix_constrain(&vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| cov_matrix_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    cov_matrix_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8], 3).unwrap(),
                    cov_matrix_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9], 3).unwrap(),
                ],
                vec![
                    cov_matrix_constrain(&vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6], 3).unwrap(),
                    cov_matrix_constrain(&vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1], 3).unwrap(),
                ],
            ],
            vec![
                vec![
                    cov_matrix_constrain(&vec![0.7, -0.2, 1.3, -0.8, 1.9, -1.4], 3).unwrap(),
                    cov_matrix_constrain(&vec![0.2, 1.1, -0.4, 1.7, -1.0, 2.3], 3).unwrap(),
                ],
                vec![
                    cov_matrix_constrain(&vec![0.9, 0.0, 1.5, -0.6, 2.1, -1.2], 3).unwrap(),
                    cov_matrix_constrain(&vec![0.4, 1.3, -0.2, 1.9, -0.8, 2.5], 3).unwrap(),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            cov_matrix_constrain(&vec![0.3, -0.6, 0.9, -1.2, 1.5, -1.8], 3).unwrap(),
            cov_matrix_constrain(&vec![-0.2, 0.7, -0.8, 1.3, -1.4, 1.9], 3).unwrap(),
            cov_matrix_constrain(&vec![0.5, -0.4, 1.1, -1.0, 1.7, -1.6], 3).unwrap(),
            cov_matrix_constrain(&vec![0.0, 0.9, -0.6, 1.5, -1.2, 2.1], 3).unwrap(),
            cov_matrix_constrain(&vec![0.7, -0.2, 1.3, -0.8, 1.9, -1.4], 3).unwrap(),
            cov_matrix_constrain(&vec![0.2, 1.1, -0.4, 1.7, -1.0, 2.3], 3).unwrap(),
            cov_matrix_constrain(&vec![0.9, 0.0, 1.5, -0.6, 2.1, -1.2], 3).unwrap(),
            cov_matrix_constrain(&vec![0.4, 1.3, -0.2, 1.9, -0.8, 2.5], 3).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| cov_matrix_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| cov_matrix_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "cov_matrix_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let got = x.iter().map(|v| ordered_constrain(v)).collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| ordered_constrain(v)).collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("ordered_constrain(array vec): {got:?} vs {want:?}"));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let got = x
            .iter()
            .map(|a1| a1.iter().map(|v| ordered_constrain(v)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| ordered_constrain(v)).collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "ordered_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| a2.iter().map(|v| ordered_constrain(v)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| ordered_constrain(v)).collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "ordered_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| ordered_jacobian(v, &mut lp))
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| ordered_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!("ordered_jacobian(array vec) lp: {got_lp} vs {lp}"));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("ordered_jacobian(array vec): {got:?} vs {want:?}"));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| ordered_jacobian(v, &mut lp))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| ordered_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "ordered_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "ordered_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| ordered_jacobian(v, &mut lp))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| ordered_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "ordered_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "ordered_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            ordered_constrain(&vec![0.3, -0.6, 0.9]),
            ordered_constrain(&vec![-0.2, 0.7, -0.8]),
        ];
        let flat_x: Vec<_> = vec![
            ordered_constrain(&vec![0.3, -0.6, 0.9]),
            ordered_constrain(&vec![-0.2, 0.7, -0.8]),
        ];
        let got = x
            .iter()
            .map(|v| ordered_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| ordered_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "ordered_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                ordered_constrain(&vec![0.3, -0.6, 0.9]),
                ordered_constrain(&vec![-0.2, 0.7, -0.8]),
            ],
            vec![
                ordered_constrain(&vec![0.5, -0.4, 1.1]),
                ordered_constrain(&vec![0.0, 0.9, -0.6]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            ordered_constrain(&vec![0.3, -0.6, 0.9]),
            ordered_constrain(&vec![-0.2, 0.7, -0.8]),
            ordered_constrain(&vec![0.5, -0.4, 1.1]),
            ordered_constrain(&vec![0.0, 0.9, -0.6]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| ordered_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| ordered_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "ordered_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    ordered_constrain(&vec![0.3, -0.6, 0.9]),
                    ordered_constrain(&vec![-0.2, 0.7, -0.8]),
                ],
                vec![
                    ordered_constrain(&vec![0.5, -0.4, 1.1]),
                    ordered_constrain(&vec![0.0, 0.9, -0.6]),
                ],
            ],
            vec![
                vec![
                    ordered_constrain(&vec![0.7, -0.2, 1.3]),
                    ordered_constrain(&vec![0.2, 1.1, -0.4]),
                ],
                vec![
                    ordered_constrain(&vec![0.9, 0.0, 1.5]),
                    ordered_constrain(&vec![0.4, 1.3, -0.2]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            ordered_constrain(&vec![0.3, -0.6, 0.9]),
            ordered_constrain(&vec![-0.2, 0.7, -0.8]),
            ordered_constrain(&vec![0.5, -0.4, 1.1]),
            ordered_constrain(&vec![0.0, 0.9, -0.6]),
            ordered_constrain(&vec![0.7, -0.2, 1.3]),
            ordered_constrain(&vec![0.2, 1.1, -0.4]),
            ordered_constrain(&vec![0.9, 0.0, 1.5]),
            ordered_constrain(&vec![0.4, 1.3, -0.2]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| ordered_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| ordered_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "ordered_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let got = x
            .iter()
            .map(|v| positive_ordered_constrain(v))
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_constrain(v))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_constrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| positive_ordered_constrain(v))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_constrain(v))
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| positive_ordered_constrain(v))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_constrain(v))
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| positive_ordered_jacobian(v, &mut lp))
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "positive_ordered_jacobian(array vec) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_jacobian(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| positive_ordered_jacobian(v, &mut lp))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "positive_ordered_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| positive_ordered_jacobian(v, &mut lp))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "positive_ordered_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            positive_ordered_constrain(&vec![0.3, -0.6, 0.9]),
            positive_ordered_constrain(&vec![-0.2, 0.7, -0.8]),
        ];
        let flat_x: Vec<_> = vec![
            positive_ordered_constrain(&vec![0.3, -0.6, 0.9]),
            positive_ordered_constrain(&vec![-0.2, 0.7, -0.8]),
        ];
        let got = x
            .iter()
            .map(|v| positive_ordered_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                positive_ordered_constrain(&vec![0.3, -0.6, 0.9]),
                positive_ordered_constrain(&vec![-0.2, 0.7, -0.8]),
            ],
            vec![
                positive_ordered_constrain(&vec![0.5, -0.4, 1.1]),
                positive_ordered_constrain(&vec![0.0, 0.9, -0.6]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            positive_ordered_constrain(&vec![0.3, -0.6, 0.9]),
            positive_ordered_constrain(&vec![-0.2, 0.7, -0.8]),
            positive_ordered_constrain(&vec![0.5, -0.4, 1.1]),
            positive_ordered_constrain(&vec![0.0, 0.9, -0.6]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| positive_ordered_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    positive_ordered_constrain(&vec![0.3, -0.6, 0.9]),
                    positive_ordered_constrain(&vec![-0.2, 0.7, -0.8]),
                ],
                vec![
                    positive_ordered_constrain(&vec![0.5, -0.4, 1.1]),
                    positive_ordered_constrain(&vec![0.0, 0.9, -0.6]),
                ],
            ],
            vec![
                vec![
                    positive_ordered_constrain(&vec![0.7, -0.2, 1.3]),
                    positive_ordered_constrain(&vec![0.2, 1.1, -0.4]),
                ],
                vec![
                    positive_ordered_constrain(&vec![0.9, 0.0, 1.5]),
                    positive_ordered_constrain(&vec![0.4, 1.3, -0.2]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            positive_ordered_constrain(&vec![0.3, -0.6, 0.9]),
            positive_ordered_constrain(&vec![-0.2, 0.7, -0.8]),
            positive_ordered_constrain(&vec![0.5, -0.4, 1.1]),
            positive_ordered_constrain(&vec![0.0, 0.9, -0.6]),
            positive_ordered_constrain(&vec![0.7, -0.2, 1.3]),
            positive_ordered_constrain(&vec![0.2, 1.1, -0.4]),
            positive_ordered_constrain(&vec![0.9, 0.0, 1.5]),
            positive_ordered_constrain(&vec![0.4, 1.3, -0.2]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| positive_ordered_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| positive_ordered_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "positive_ordered_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let got = x.iter().map(|v| simplex_constrain(v)).collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| simplex_constrain(v)).collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("simplex_constrain(array vec): {got:?} vs {want:?}"));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let got = x
            .iter()
            .map(|a1| a1.iter().map(|v| simplex_constrain(v)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| simplex_constrain(v)).collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "simplex_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| a2.iter().map(|v| simplex_constrain(v)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| simplex_constrain(v)).collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "simplex_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| simplex_jacobian(v, &mut lp))
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| simplex_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!("simplex_jacobian(array vec) lp: {got_lp} vs {lp}"));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("simplex_jacobian(array vec): {got:?} vs {want:?}"));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| simplex_jacobian(v, &mut lp))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| simplex_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "simplex_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "simplex_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| simplex_jacobian(v, &mut lp))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| simplex_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "simplex_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "simplex_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            simplex_constrain(&vec![0.3, -0.6, 0.9]),
            simplex_constrain(&vec![-0.2, 0.7, -0.8]),
        ];
        let flat_x: Vec<_> = vec![
            simplex_constrain(&vec![0.3, -0.6, 0.9]),
            simplex_constrain(&vec![-0.2, 0.7, -0.8]),
        ];
        let got = x
            .iter()
            .map(|v| simplex_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| simplex_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "simplex_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                simplex_constrain(&vec![0.3, -0.6, 0.9]),
                simplex_constrain(&vec![-0.2, 0.7, -0.8]),
            ],
            vec![
                simplex_constrain(&vec![0.5, -0.4, 1.1]),
                simplex_constrain(&vec![0.0, 0.9, -0.6]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            simplex_constrain(&vec![0.3, -0.6, 0.9]),
            simplex_constrain(&vec![-0.2, 0.7, -0.8]),
            simplex_constrain(&vec![0.5, -0.4, 1.1]),
            simplex_constrain(&vec![0.0, 0.9, -0.6]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| simplex_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| simplex_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "simplex_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    simplex_constrain(&vec![0.3, -0.6, 0.9]),
                    simplex_constrain(&vec![-0.2, 0.7, -0.8]),
                ],
                vec![
                    simplex_constrain(&vec![0.5, -0.4, 1.1]),
                    simplex_constrain(&vec![0.0, 0.9, -0.6]),
                ],
            ],
            vec![
                vec![
                    simplex_constrain(&vec![0.7, -0.2, 1.3]),
                    simplex_constrain(&vec![0.2, 1.1, -0.4]),
                ],
                vec![
                    simplex_constrain(&vec![0.9, 0.0, 1.5]),
                    simplex_constrain(&vec![0.4, 1.3, -0.2]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            simplex_constrain(&vec![0.3, -0.6, 0.9]),
            simplex_constrain(&vec![-0.2, 0.7, -0.8]),
            simplex_constrain(&vec![0.5, -0.4, 1.1]),
            simplex_constrain(&vec![0.0, 0.9, -0.6]),
            simplex_constrain(&vec![0.7, -0.2, 1.3]),
            simplex_constrain(&vec![0.2, 1.1, -0.4]),
            simplex_constrain(&vec![0.9, 0.0, 1.5]),
            simplex_constrain(&vec![0.4, 1.3, -0.2]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| simplex_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| simplex_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "simplex_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let got = x
            .iter()
            .map(|v| stochastic_column_constrain(v))
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_constrain(v))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_constrain(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| stochastic_column_constrain(v))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_constrain(v))
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_constrain(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                    Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                    Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
                    Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
                    Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
            Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
            Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
            Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| stochastic_column_constrain(v))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_constrain(v))
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_constrain(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| stochastic_column_jacobian(v, &mut lp))
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "stochastic_column_jacobian(array mat) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_jacobian(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| stochastic_column_jacobian(v, &mut lp))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "stochastic_column_jacobian(array array mat) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_jacobian(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                    Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                    Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
                    Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
                    Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
            Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
            Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
            Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| stochastic_column_jacobian(v, &mut lp))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "stochastic_column_jacobian(array array array mat) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_jacobian(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.3, -0.6, 0.9, -1.2],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![-0.2, 0.7, -0.8, 1.3],
            )),
        ];
        let flat_x: Vec<_> = vec![
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.3, -0.6, 0.9, -1.2],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![-0.2, 0.7, -0.8, 1.3],
            )),
        ];
        let got = x
            .iter()
            .map(|v| stochastic_column_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_unconstrain(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                stochastic_column_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.3, -0.6, 0.9, -1.2],
                )),
                stochastic_column_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![-0.2, 0.7, -0.8, 1.3],
                )),
            ],
            vec![
                stochastic_column_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.5, -0.4, 1.1, -1.0],
                )),
                stochastic_column_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.0, 0.9, -0.6, 1.5],
                )),
            ],
        ];
        let flat_x: Vec<_> = vec![
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.3, -0.6, 0.9, -1.2],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![-0.2, 0.7, -0.8, 1.3],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.5, -0.4, 1.1, -1.0],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.0, 0.9, -0.6, 1.5],
            )),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| stochastic_column_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_unconstrain(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    stochastic_column_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.3, -0.6, 0.9, -1.2],
                    )),
                    stochastic_column_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![-0.2, 0.7, -0.8, 1.3],
                    )),
                ],
                vec![
                    stochastic_column_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.5, -0.4, 1.1, -1.0],
                    )),
                    stochastic_column_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.0, 0.9, -0.6, 1.5],
                    )),
                ],
            ],
            vec![
                vec![
                    stochastic_column_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.7, -0.2, 1.3, -0.8],
                    )),
                    stochastic_column_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.2, 1.1, -0.4, 1.7],
                    )),
                ],
                vec![
                    stochastic_column_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.9, 0.0, 1.5, -0.6],
                    )),
                    stochastic_column_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.4, 1.3, -0.2, 1.9],
                    )),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.3, -0.6, 0.9, -1.2],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![-0.2, 0.7, -0.8, 1.3],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.5, -0.4, 1.1, -1.0],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.0, 0.9, -0.6, 1.5],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.7, -0.2, 1.3, -0.8],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.2, 1.1, -0.4, 1.7],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.9, 0.0, 1.5, -0.6],
            )),
            stochastic_column_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.4, 1.3, -0.2, 1.9],
            )),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| stochastic_column_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_column_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_column_unconstrain(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let got = x
            .iter()
            .map(|v| stochastic_row_constrain(v))
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| stochastic_row_constrain(v)).collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_constrain(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| stochastic_row_constrain(v))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| stochastic_row_constrain(v)).collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_constrain(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                    Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                    Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
                    Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
                    Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
            Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
            Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
            Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| stochastic_row_constrain(v))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| stochastic_row_constrain(v)).collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_constrain(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| stochastic_row_jacobian(v, &mut lp))
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_row_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "stochastic_row_jacobian(array mat) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_jacobian(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| stochastic_row_jacobian(v, &mut lp))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_row_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "stochastic_row_jacobian(array array mat) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_jacobian(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                    Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                    Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
                    Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
                    Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
            Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
            Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
            Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| stochastic_row_jacobian(v, &mut lp))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_row_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "stochastic_row_jacobian(array array array mat) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_jacobian(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3])),
        ];
        let flat_x: Vec<_> = vec![
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3])),
        ];
        let got = x
            .iter()
            .map(|v| stochastic_row_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_row_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_unconstrain(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                stochastic_row_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.3, -0.6, 0.9, -1.2],
                )),
                stochastic_row_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![-0.2, 0.7, -0.8, 1.3],
                )),
            ],
            vec![
                stochastic_row_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.5, -0.4, 1.1, -1.0],
                )),
                stochastic_row_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.0, 0.9, -0.6, 1.5],
                )),
            ],
        ];
        let flat_x: Vec<_> = vec![
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5])),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| stochastic_row_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_row_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_unconstrain(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    stochastic_row_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.3, -0.6, 0.9, -1.2],
                    )),
                    stochastic_row_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![-0.2, 0.7, -0.8, 1.3],
                    )),
                ],
                vec![
                    stochastic_row_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.5, -0.4, 1.1, -1.0],
                    )),
                    stochastic_row_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.0, 0.9, -0.6, 1.5],
                    )),
                ],
            ],
            vec![
                vec![
                    stochastic_row_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.7, -0.2, 1.3, -0.8],
                    )),
                    stochastic_row_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.2, 1.1, -0.4, 1.7],
                    )),
                ],
                vec![
                    stochastic_row_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.9, 0.0, 1.5, -0.6],
                    )),
                    stochastic_row_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.4, 1.3, -0.2, 1.9],
                    )),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6])),
            stochastic_row_constrain(&Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9])),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| stochastic_row_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| stochastic_row_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "stochastic_row_unconstrain(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let got = x
            .iter()
            .map(|v| sum_to_zero_constrain(v))
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| sum_to_zero_constrain(v)).collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_constrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| sum_to_zero_constrain(v))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| sum_to_zero_constrain(v)).collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| sum_to_zero_constrain(v))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x.iter().map(|v| sum_to_zero_constrain(v)).collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| sum_to_zero_jacobian(v, &mut lp))
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "sum_to_zero_jacobian(array vec) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_jacobian(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| sum_to_zero_jacobian(v, &mut lp))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "sum_to_zero_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| sum_to_zero_jacobian(v, &mut lp))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "sum_to_zero_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            sum_to_zero_constrain(&vec![0.3, -0.6, 0.9]),
            sum_to_zero_constrain(&vec![-0.2, 0.7, -0.8]),
        ];
        let flat_x: Vec<_> = vec![
            sum_to_zero_constrain(&vec![0.3, -0.6, 0.9]),
            sum_to_zero_constrain(&vec![-0.2, 0.7, -0.8]),
        ];
        let got = x
            .iter()
            .map(|v| sum_to_zero_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                sum_to_zero_constrain(&vec![0.3, -0.6, 0.9]),
                sum_to_zero_constrain(&vec![-0.2, 0.7, -0.8]),
            ],
            vec![
                sum_to_zero_constrain(&vec![0.5, -0.4, 1.1]),
                sum_to_zero_constrain(&vec![0.0, 0.9, -0.6]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            sum_to_zero_constrain(&vec![0.3, -0.6, 0.9]),
            sum_to_zero_constrain(&vec![-0.2, 0.7, -0.8]),
            sum_to_zero_constrain(&vec![0.5, -0.4, 1.1]),
            sum_to_zero_constrain(&vec![0.0, 0.9, -0.6]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| sum_to_zero_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    sum_to_zero_constrain(&vec![0.3, -0.6, 0.9]),
                    sum_to_zero_constrain(&vec![-0.2, 0.7, -0.8]),
                ],
                vec![
                    sum_to_zero_constrain(&vec![0.5, -0.4, 1.1]),
                    sum_to_zero_constrain(&vec![0.0, 0.9, -0.6]),
                ],
            ],
            vec![
                vec![
                    sum_to_zero_constrain(&vec![0.7, -0.2, 1.3]),
                    sum_to_zero_constrain(&vec![0.2, 1.1, -0.4]),
                ],
                vec![
                    sum_to_zero_constrain(&vec![0.9, 0.0, 1.5]),
                    sum_to_zero_constrain(&vec![0.4, 1.3, -0.2]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            sum_to_zero_constrain(&vec![0.3, -0.6, 0.9]),
            sum_to_zero_constrain(&vec![-0.2, 0.7, -0.8]),
            sum_to_zero_constrain(&vec![0.5, -0.4, 1.1]),
            sum_to_zero_constrain(&vec![0.0, 0.9, -0.6]),
            sum_to_zero_constrain(&vec![0.7, -0.2, 1.3]),
            sum_to_zero_constrain(&vec![0.2, 1.1, -0.4]),
            sum_to_zero_constrain(&vec![0.9, 0.0, 1.5]),
            sum_to_zero_constrain(&vec![0.4, 1.3, -0.2]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| sum_to_zero_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let got = x
            .iter()
            .map(|v| sum_to_zero_matrix_constrain(v))
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_constrain(v))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_constrain(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| sum_to_zero_matrix_constrain(v))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_constrain(v))
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_constrain(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                    Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                    Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
                    Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
                    Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
            Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
            Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
            Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| sum_to_zero_matrix_constrain(v))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_constrain(v))
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_constrain(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| sum_to_zero_matrix_jacobian(v, &mut lp))
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "sum_to_zero_matrix_jacobian(array mat) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_jacobian(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            ],
            vec![
                Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| sum_to_zero_matrix_jacobian(v, &mut lp))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "sum_to_zero_matrix_jacobian(array array mat) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_jacobian(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
                    Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
                    Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
                ],
            ],
            vec![
                vec![
                    Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
                    Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
                ],
                vec![
                    Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
                    Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            Matrix::from_column_major(2, 2, vec![0.3, -0.6, 0.9, -1.2]),
            Matrix::from_column_major(2, 2, vec![-0.2, 0.7, -0.8, 1.3]),
            Matrix::from_column_major(2, 2, vec![0.5, -0.4, 1.1, -1.0]),
            Matrix::from_column_major(2, 2, vec![0.0, 0.9, -0.6, 1.5]),
            Matrix::from_column_major(2, 2, vec![0.7, -0.2, 1.3, -0.8]),
            Matrix::from_column_major(2, 2, vec![0.2, 1.1, -0.4, 1.7]),
            Matrix::from_column_major(2, 2, vec![0.9, 0.0, 1.5, -0.6]),
            Matrix::from_column_major(2, 2, vec![0.4, 1.3, -0.2, 1.9]),
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| sum_to_zero_matrix_jacobian(v, &mut lp))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_jacobian(v, &mut lp))
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "sum_to_zero_matrix_jacobian(array array array mat) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_jacobian(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.3, -0.6, 0.9, -1.2],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![-0.2, 0.7, -0.8, 1.3],
            )),
        ];
        let flat_x: Vec<_> = vec![
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.3, -0.6, 0.9, -1.2],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![-0.2, 0.7, -0.8, 1.3],
            )),
        ];
        let got = x
            .iter()
            .map(|v| sum_to_zero_matrix_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_unconstrain(array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.3, -0.6, 0.9, -1.2],
                )),
                sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![-0.2, 0.7, -0.8, 1.3],
                )),
            ],
            vec![
                sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.5, -0.4, 1.1, -1.0],
                )),
                sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                    2,
                    2,
                    vec![0.0, 0.9, -0.6, 1.5],
                )),
            ],
        ];
        let flat_x: Vec<_> = vec![
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.3, -0.6, 0.9, -1.2],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![-0.2, 0.7, -0.8, 1.3],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.5, -0.4, 1.1, -1.0],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.0, 0.9, -0.6, 1.5],
            )),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| sum_to_zero_matrix_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_unconstrain(array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.3, -0.6, 0.9, -1.2],
                    )),
                    sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![-0.2, 0.7, -0.8, 1.3],
                    )),
                ],
                vec![
                    sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.5, -0.4, 1.1, -1.0],
                    )),
                    sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.0, 0.9, -0.6, 1.5],
                    )),
                ],
            ],
            vec![
                vec![
                    sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.7, -0.2, 1.3, -0.8],
                    )),
                    sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.2, 1.1, -0.4, 1.7],
                    )),
                ],
                vec![
                    sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.9, 0.0, 1.5, -0.6],
                    )),
                    sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                        2,
                        2,
                        vec![0.4, 1.3, -0.2, 1.9],
                    )),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.3, -0.6, 0.9, -1.2],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![-0.2, 0.7, -0.8, 1.3],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.5, -0.4, 1.1, -1.0],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.0, 0.9, -0.6, 1.5],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.7, -0.2, 1.3, -0.8],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.2, 1.1, -0.4, 1.7],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.9, 0.0, 1.5, -0.6],
            )),
            sum_to_zero_matrix_constrain(&Matrix::from_column_major(
                2,
                2,
                vec![0.4, 1.3, -0.2, 1.9],
            )),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| sum_to_zero_matrix_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| sum_to_zero_matrix_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "sum_to_zero_matrix_unconstrain(array array array mat): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let got = x
            .iter()
            .map(|v| unit_vector_constrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_constrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_constrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| unit_vector_constrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_constrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_constrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| unit_vector_constrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_constrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_constrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let flat_x: Vec<_> = vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|v| unit_vector_jacobian(v, &mut lp).unwrap())
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_jacobian(v, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "unit_vector_jacobian(array vec) lp: {got_lp} vs {lp}"
            ));
        }
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_jacobian(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
            vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| unit_vector_jacobian(v, &mut lp).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_jacobian(v, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "unit_vector_jacobian(array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_jacobian(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![vec![0.3, -0.6, 0.9], vec![-0.2, 0.7, -0.8]],
                vec![vec![0.5, -0.4, 1.1], vec![0.0, 0.9, -0.6]],
            ],
            vec![
                vec![vec![0.7, -0.2, 1.3], vec![0.2, 1.1, -0.4]],
                vec![vec![0.9, 0.0, 1.5], vec![0.4, 1.3, -0.2]],
            ],
        ];
        let flat_x: Vec<_> = vec![
            vec![0.3, -0.6, 0.9],
            vec![-0.2, 0.7, -0.8],
            vec![0.5, -0.4, 1.1],
            vec![0.0, 0.9, -0.6],
            vec![0.7, -0.2, 1.3],
            vec![0.2, 1.1, -0.4],
            vec![0.9, 0.0, 1.5],
            vec![0.4, 1.3, -0.2],
        ];
        let mut lp = 0.0;
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| unit_vector_jacobian(v, &mut lp).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let got_lp = lp;
        let mut lp = 0.0;
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_jacobian(v, &mut lp).unwrap())
            .collect();
        if (got_lp - lp).abs() > 1e-12 {
            failures.push(format!(
                "unit_vector_jacobian(array array array vec) lp: {got_lp} vs {lp}"
            ));
        }
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_jacobian(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            unit_vector_constrain(&vec![0.3, -0.6, 0.9]).unwrap(),
            unit_vector_constrain(&vec![-0.2, 0.7, -0.8]).unwrap(),
        ];
        let flat_x: Vec<_> = vec![
            unit_vector_constrain(&vec![0.3, -0.6, 0.9]).unwrap(),
            unit_vector_constrain(&vec![-0.2, 0.7, -0.8]).unwrap(),
        ];
        let got = x
            .iter()
            .map(|v| unit_vector_unconstrain(v).unwrap())
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_unconstrain(v).unwrap())
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_unconstrain(array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                unit_vector_constrain(&vec![0.3, -0.6, 0.9]).unwrap(),
                unit_vector_constrain(&vec![-0.2, 0.7, -0.8]).unwrap(),
            ],
            vec![
                unit_vector_constrain(&vec![0.5, -0.4, 1.1]).unwrap(),
                unit_vector_constrain(&vec![0.0, 0.9, -0.6]).unwrap(),
            ],
        ];
        let flat_x: Vec<_> = vec![
            unit_vector_constrain(&vec![0.3, -0.6, 0.9]).unwrap(),
            unit_vector_constrain(&vec![-0.2, 0.7, -0.8]).unwrap(),
            unit_vector_constrain(&vec![0.5, -0.4, 1.1]).unwrap(),
            unit_vector_constrain(&vec![0.0, 0.9, -0.6]).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|v| unit_vector_unconstrain(v).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_unconstrain(array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let x = vec![
            vec![
                vec![
                    unit_vector_constrain(&vec![0.3, -0.6, 0.9]).unwrap(),
                    unit_vector_constrain(&vec![-0.2, 0.7, -0.8]).unwrap(),
                ],
                vec![
                    unit_vector_constrain(&vec![0.5, -0.4, 1.1]).unwrap(),
                    unit_vector_constrain(&vec![0.0, 0.9, -0.6]).unwrap(),
                ],
            ],
            vec![
                vec![
                    unit_vector_constrain(&vec![0.7, -0.2, 1.3]).unwrap(),
                    unit_vector_constrain(&vec![0.2, 1.1, -0.4]).unwrap(),
                ],
                vec![
                    unit_vector_constrain(&vec![0.9, 0.0, 1.5]).unwrap(),
                    unit_vector_constrain(&vec![0.4, 1.3, -0.2]).unwrap(),
                ],
            ],
        ];
        let flat_x: Vec<_> = vec![
            unit_vector_constrain(&vec![0.3, -0.6, 0.9]).unwrap(),
            unit_vector_constrain(&vec![-0.2, 0.7, -0.8]).unwrap(),
            unit_vector_constrain(&vec![0.5, -0.4, 1.1]).unwrap(),
            unit_vector_constrain(&vec![0.0, 0.9, -0.6]).unwrap(),
            unit_vector_constrain(&vec![0.7, -0.2, 1.3]).unwrap(),
            unit_vector_constrain(&vec![0.2, 1.1, -0.4]).unwrap(),
            unit_vector_constrain(&vec![0.9, 0.0, 1.5]).unwrap(),
            unit_vector_constrain(&vec![0.4, 1.3, -0.2]).unwrap(),
        ];
        let got = x
            .iter()
            .map(|a1| {
                a1.iter()
                    .map(|a2| {
                        a2.iter()
                            .map(|v| unit_vector_unconstrain(v).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let want: Vec<_> = flat_x
            .iter()
            .map(|v| unit_vector_unconstrain(v).unwrap())
            .collect();
        let got = got.into_iter().flatten().flatten().collect::<Vec<_>>();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "unit_vector_unconstrain(array array array vec): {got:?} vs {want:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
