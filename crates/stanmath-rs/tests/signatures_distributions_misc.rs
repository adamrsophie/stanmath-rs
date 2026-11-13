//! The Stan signatures of distributions that no generator reaches: the Poisson binomial, the
//! categorical forms with an array response, the Gaussian dynamic linear model with a vector of
//! observation variances, the container forms of `wiener_lpdf`, and two RNGs. Each block names its
//! Stan signatures in `// stan:` comments and checks the call against a brute force sum or an
//! equivalent call; the scalar forms are compared with Stan Math by `stan_reference.rs`.
#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::needless_range_loop
)]

#[cfg(feature = "rand")]
use rand::SeedableRng;
#[cfg(feature = "rand")]
use rand::rngs::StdRng;
use stanmath_rs::prelude::*;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * b.abs().max(1.0)
}

/// The Poisson binomial probabilities of 0..=n successes, by enumerating every outcome.
fn poisson_binomial_pmf(theta: &[f64]) -> Vec<f64> {
    let n = theta.len();
    let mut p = vec![0.0; n + 1];
    for mask in 0..1_u32 << n {
        let pr: f64 = (0..n)
            .map(|i| {
                if mask >> i & 1 == 1 {
                    theta[i]
                } else {
                    1.0 - theta[i]
                }
            })
            .product();
        p[mask.count_ones() as usize] += pr;
    }
    p
}

#[test]
fn poisson_binomial() {
    let theta = [0.2, 0.7, 0.45, 0.9];
    let p = poisson_binomial_pmf(&theta);
    let cdf = |y: usize| p[..=y].iter().sum::<f64>();
    // stan: poisson_binomial_lpmf(int, vector)
    // stan: poisson_binomial_lpmf(int, row_vector)
    // stan: poisson_binomial_lpmf(int, array[] real)
    // stan: poisson_binomial_lcdf(int, vector)
    // stan: poisson_binomial_lcdf(int, row_vector)
    // stan: poisson_binomial_lcdf(int, array[] real)
    // stan: poisson_binomial_lccdf(int, vector)
    // stan: poisson_binomial_lccdf(int, row_vector)
    // stan: poisson_binomial_lccdf(int, array[] real)
    for y in 0..=3_usize {
        let yi = y as i64;
        assert!(
            close(poisson_binomial_lpmf(yi, &theta).unwrap(), p[y].ln()),
            "{y}"
        );
        assert!(
            close(poisson_binomial_lcdf(yi, &theta).unwrap(), cdf(y).ln()),
            "{y}"
        );
        assert!(
            close(
                poisson_binomial_lccdf(yi, &theta).unwrap(),
                (1.0 - cdf(y)).ln()
            ),
            "{y}"
        );
    }
    // stan: poisson_binomial_lpmf(array[] int, vector)
    // stan: poisson_binomial_lpmf(array[] int, row_vector)
    // stan: poisson_binomial_lpmf(array[] int, array[] real)
    // stan: poisson_binomial_lcdf(array[] int, vector)
    // stan: poisson_binomial_lcdf(array[] int, row_vector)
    // stan: poisson_binomial_lcdf(array[] int, array[] real)
    // stan: poisson_binomial_lccdf(array[] int, vector)
    // stan: poisson_binomial_lccdf(array[] int, row_vector)
    // stan: poisson_binomial_lccdf(array[] int, array[] real)
    // An array response shares theta: the sum over its elements.
    let ys = [1_i64, 3, 2];
    let sum = |f: &dyn Fn(i64) -> f64| ys.iter().map(|&y| f(y)).sum::<f64>();
    assert!(close(
        poisson_binomial_lpmf(&ys[..], &theta).unwrap(),
        sum(&|y| poisson_binomial_lpmf(y, &theta).unwrap())
    ));
    assert!(close(
        poisson_binomial_lcdf(&ys[..], &theta).unwrap(),
        sum(&|y| poisson_binomial_lcdf(y, &theta).unwrap())
    ));
    assert!(close(
        poisson_binomial_lccdf(&ys[..], &theta).unwrap(),
        sum(&|y| poisson_binomial_lccdf(y, &theta).unwrap())
    ));
}

#[test]
fn categorical() {
    let theta = [0.2, 0.5, 0.3];
    let beta = [0.4, -1.0, 1.3];
    let z: f64 = beta.iter().map(|b: &f64| b.exp()).sum();
    // stan: categorical_lpmf(int, vector)
    // stan: categorical_logit_lpmf(int, vector)
    for n in 1..=3_i64 {
        let k = usize::try_from(n - 1).unwrap();
        assert!(close(categorical_lpmf(n, &theta).unwrap(), theta[k].ln()));
        assert!(close(
            categorical_logit_lpmf(n, &beta).unwrap(),
            beta[k] - z.ln()
        ));
    }
    // stan: categorical_lpmf(array[] int, vector)
    // stan: categorical_logit_lpmf(array[] int, vector)
    let ns = [3_i64, 1, 3];
    assert!(close(
        categorical_lpmf(&ns[..], &theta).unwrap(),
        2.0 * 0.3_f64.ln() + 0.2_f64.ln()
    ));
    assert!(close(
        categorical_logit_lpmf(&ns[..], &beta).unwrap(),
        2.0 * 1.3 + 0.4 - 3.0 * z.ln()
    ));
}

#[test]
fn gaussian_dlm_obs() {
    // Two observed series, a two dimensional state, three time points.
    let y = Matrix::from_column_major(2, 3, vec![0.3, -0.2, 0.8, 0.1, -0.4, 0.6]);
    let f = Matrix::from_column_major(2, 2, vec![1.0, 0.2, 0.0, 1.0]);
    let g = Matrix::from_column_major(2, 2, vec![0.9, 0.0, 0.1, 0.8]);
    let v = [0.5, 0.7];
    let w = Matrix::from_column_major(2, 2, vec![0.3, 0.05, 0.05, 0.2]);
    let m0 = [0.0, 0.1];
    let c0 = Matrix::from_column_major(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    // stan: gaussian_dlm_obs_lpdf(matrix, matrix, matrix, matrix, matrix, vector, matrix)
    // stan: gaussian_dlm_obs_lpdf(matrix, matrix, matrix, vector, matrix, vector, matrix)
    // Stan's vector V is the diagonal of the observation covariance; it is diag_matrix(V) here.
    let full = gaussian_dlm_obs_lpdf(&y, &f, &g, &diag_matrix(&v), &w, &m0, &c0).unwrap();
    // The Kalman filter by hand: the sum of the one step ahead predictive normal densities.
    let mut m = m0.to_vec();
    let mut c = c0.clone();
    let mut want = 0.0;
    let mm = |a: &Matrix<f64>, b: &Matrix<f64>| multiply(a, b).unwrap();
    for t in 0..3 {
        let gm = mm(&g, &Matrix::column_vector(m.clone()));
        let r = add(&mm(&mm(&g, &c), &g.transpose()), &w).unwrap();
        let fa = mm(&f.transpose(), &gm);
        let q = add(&mm(&mm(&f.transpose(), &r), &f), &diag_matrix(&v)).unwrap();
        let e: Vec<f64> = (0..2).map(|i| y[(i, t)] - fa[(i, 0)]).collect();
        want += multi_normal_lpdf(&[y[(0, t)], y[(1, t)]][..], fa.as_slice(), &q).unwrap();
        let k = mm(&mm(&r, &f), &inverse(&q).unwrap());
        let ke = mm(&k, &Matrix::column_vector(e));
        m = (0..2).map(|i| gm[(i, 0)] + ke[(i, 0)]).collect();
        c = subtract(&r, &mm(&mm(&k, &q), &k.transpose())).unwrap();
    }
    assert!(close(full, want), "{full} vs {want}");
}

#[test]
fn wiener_containers() {
    let y = [1.2, 0.9, 1.6];
    let a = [1.5, 1.8, 1.2];
    let t0 = [0.2, 0.25, 0.3];
    let w = [0.5, 0.4, 0.6];
    let v = [0.8, -0.3, 1.1];
    let sv = [0.2, 0.0, 0.4];
    let sw = [0.1, 0.05, 0.0];
    let st0 = [0.05, 0.1, 0.0];
    let sum = |f: &dyn Fn(usize) -> f64| (0..3).map(f).sum::<f64>();
    // stan: wiener_lpdf(vector, vector, vector, vector, vector, vector)
    let got = wiener5_lpdf(&y[..], &a[..], &t0[..], &w[..], &v[..], &sv[..]).unwrap();
    let want = sum(&|i| wiener5_lpdf(y[i], a[i], t0[i], w[i], v[i], sv[i]).unwrap());
    assert!(close(got, want), "{got} vs {want}");
    // stan: wiener_lpdf(vector, vector, vector, vector, vector, vector, real)
    let got = wiener5_prec_lpdf(&y[..], &a[..], &t0[..], &w[..], &v[..], &sv[..], 1e-8).unwrap();
    let want = sum(&|i| wiener5_prec_lpdf(y[i], a[i], t0[i], w[i], v[i], sv[i], 1e-8).unwrap());
    assert!(close(got, want), "{got} vs {want}");
    // stan: wiener_lpdf(vector, vector, vector, vector, vector, vector, vector, vector)
    let got = wiener_full_lpdf(
        &y[..],
        &a[..],
        &t0[..],
        &w[..],
        &v[..],
        &sv[..],
        &sw[..],
        &st0[..],
    )
    .unwrap();
    let want =
        sum(&|i| wiener_full_lpdf(y[i], a[i], t0[i], w[i], v[i], sv[i], sw[i], st0[i]).unwrap());
    assert!(close(got, want), "{got} vs {want}");
    // stan: wiener_lpdf(vector, vector, vector, vector, vector, vector, vector, vector, real)
    let got = wiener_full_prec_lpdf(
        &y[..],
        &a[..],
        &t0[..],
        &w[..],
        &v[..],
        &sv[..],
        &sw[..],
        &st0[..],
        1e-8,
    )
    .unwrap();
    let want = sum(&|i| {
        wiener_full_prec_lpdf(y[i], a[i], t0[i], w[i], v[i], sv[i], sw[i], st0[i], 1e-8).unwrap()
    });
    assert!(close(got, want), "{got} vs {want}");
}

#[cfg(feature = "rand")]
#[test]
fn samplers() {
    let mut rng = StdRng::seed_from_u64(11);
    // stan: hypergeometric_rng(int, int, int)
    // Draws of 5 from 4 successes and 6 failures: a count in 0..=4 with mean 2.
    let draws: Vec<i64> = (0..4000)
        .map(|_| hypergeometric_rng(5, 4, 6, &mut rng).unwrap())
        .collect();
    assert!(draws.iter().all(|d| (0..=4).contains(d)));
    let mean = draws.iter().sum::<i64>() as f64 / 4000.0;
    assert!((mean - 2.0).abs() < 0.1, "{mean}");
    // stan: categorical_rng(vector)
    let theta = [0.1, 0.6, 0.3];
    let mut counts = [0_usize; 3];
    for _ in 0..4000 {
        let k = categorical_rng(&theta, &mut rng).unwrap();
        counts[usize::try_from(k - 1).unwrap()] += 1;
    }
    for (c, t) in counts.iter().zip(theta) {
        assert!((*c as f64 / 4000.0 - t).abs() < 0.04, "{counts:?}");
    }
}
