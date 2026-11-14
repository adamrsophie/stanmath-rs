//! Mathematical identities that tie the densities, CDFs, and samplers together.

use stanmath_rs_autodiff::{Dual64, Real, derivative, grad, hessian, hessian_forward};
use stanmath_rs_core::Error;
use stanmath_rs_distributions::*;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + b.abs())
}

/// `d/dy exp(lcdf(y)) = exp(lpdf(y))` and `exp(lcdf) + exp(lccdf) = 1`.
#[test]
fn cdf_derivative_is_density() {
    type Fns = (
        &'static str,
        fn(Dual64) -> Dual64,
        fn(f64) -> f64,
        fn(f64) -> f64,
        Vec<f64>,
    );
    let cases: Vec<Fns> = vec![
        (
            "normal",
            |y| normal_lcdf(y, 0.3, 1.7).unwrap().exp(),
            |y| normal_lpdf(y, 0.3, 1.7).unwrap(),
            |y| normal_lccdf(y, 0.3, 1.7).unwrap(),
            vec![-4.0, -0.5, 0.3, 2.0, 6.0],
        ),
        (
            "student_t",
            |y| student_t_lcdf(y, 3.5, 0.3, 1.7).unwrap().exp(),
            |y| student_t_lpdf(y, 3.5, 0.3, 1.7).unwrap(),
            |y| student_t_lccdf(y, 3.5, 0.3, 1.7).unwrap(),
            vec![-10.0, -0.5, 0.3, 0.31, 2.0, 25.0],
        ),
        (
            "cauchy",
            |y| cauchy_lcdf(y, -1.0, 0.5).unwrap().exp(),
            |y| cauchy_lpdf(y, -1.0, 0.5).unwrap(),
            |y| cauchy_lccdf(y, -1.0, 0.5).unwrap(),
            vec![-50.0, -1.2, 0.0, 9.0],
        ),
        (
            "logistic",
            |y| logistic_lcdf(y, 2.0, 0.7).unwrap().exp(),
            |y| logistic_lpdf(y, 2.0, 0.7).unwrap(),
            |y| logistic_lccdf(y, 2.0, 0.7).unwrap(),
            vec![-3.0, 1.0, 2.0, 8.0],
        ),
        (
            "exponential",
            |y| exponential_lcdf(y, 1.3).unwrap().exp(),
            |y| exponential_lpdf(y, 1.3).unwrap(),
            |y| exponential_lccdf(y, 1.3).unwrap(),
            vec![0.01, 0.5, 3.0],
        ),
        (
            "gamma",
            |y| gamma_lcdf(y, 2.5, 1.5).unwrap().exp(),
            |y| gamma_lpdf(y, 2.5, 1.5).unwrap(),
            |y| gamma_lccdf(y, 2.5, 1.5).unwrap(),
            vec![0.05, 1.0, 1.66, 4.0, 12.0],
        ),
        (
            "inv_gamma",
            |y| inv_gamma_lcdf(y, 3.0, 2.0).unwrap().exp(),
            |y| inv_gamma_lpdf(y, 3.0, 2.0).unwrap(),
            |y| inv_gamma_lccdf(y, 3.0, 2.0).unwrap(),
            vec![0.2, 0.7, 3.0],
        ),
        (
            "beta",
            |y| beta_lcdf(y, 2.5, 4.0).unwrap().exp(),
            |y| beta_lpdf(y, 2.5, 4.0).unwrap(),
            |y| beta_lccdf(y, 2.5, 4.0).unwrap(),
            vec![0.02, 0.3, 0.6, 0.97],
        ),
        (
            "lognormal",
            |y| lognormal_lcdf(y, 0.2, 0.8).unwrap().exp(),
            |y| lognormal_lpdf(y, 0.2, 0.8).unwrap(),
            |y| lognormal_lccdf(y, 0.2, 0.8).unwrap(),
            vec![0.1, 1.0, 5.0],
        ),
        (
            "weibull",
            |y| weibull_lcdf(y, 1.7, 2.2).unwrap().exp(),
            |y| weibull_lpdf(y, 1.7, 2.2).unwrap(),
            |y| weibull_lccdf(y, 1.7, 2.2).unwrap(),
            vec![0.1, 1.0, 5.0],
        ),
        (
            "uniform",
            |y| uniform_lcdf(y, -1.0, 3.0).unwrap().exp(),
            |y| uniform_lpdf(y, -1.0, 3.0).unwrap(),
            |y| uniform_lccdf(y, -1.0, 3.0).unwrap(),
            vec![-0.5, 0.0, 2.9],
        ),
    ];
    for (name, cdf, lpdf, lccdf, ys) in cases {
        for y in ys {
            let density = derivative(cdf, y);
            let want = lpdf(y).exp();
            assert!(
                close(density, want, 1e-12),
                "{name} at {y}: dF/dy {density} vs pdf {want}"
            );
            let total = cdf(Dual64::constant(y)).re + lccdf(y).exp();
            assert!(
                close(total, 1.0, 1e-14),
                "{name} at {y}: F + (1 - F) = {total}"
            );
        }
    }
}

/// `exp(lcdf(n)) - exp(lcdf(n - 1)) = exp(lpmf(n))` and `exp(lcdf) + exp(lccdf) = 1`.
#[test]
fn discrete_cdf_differences_are_masses() {
    for n in 1..30i64 {
        let step = poisson_lcdf(n, 7.5).unwrap().exp() - poisson_lcdf(n - 1, 7.5).unwrap().exp();
        assert!(
            close(step, poisson_lpmf(n, 7.5).unwrap().exp(), 1e-13),
            "poisson {n}"
        );
        let total = poisson_lcdf(n, 7.5).unwrap().exp() + poisson_lccdf(n, 7.5).unwrap().exp();
        assert!(close(total, 1.0, 1e-14));

        let step = neg_binomial_2_lcdf(n, 6.0, 2.5).unwrap().exp()
            - neg_binomial_2_lcdf(n - 1, 6.0, 2.5).unwrap().exp();
        assert!(
            close(step, neg_binomial_2_lpmf(n, 6.0, 2.5).unwrap().exp(), 1e-13),
            "nb {n}"
        );
        let total = neg_binomial_2_lcdf(n, 6.0, 2.5).unwrap().exp()
            + neg_binomial_2_lccdf(n, 6.0, 2.5).unwrap().exp();
        assert!(close(total, 1.0, 1e-14));
    }
    for n in 1..20i64 {
        let step = binomial_lcdf(n, 20, 0.35).unwrap().exp()
            - binomial_lcdf(n - 1, 20, 0.35).unwrap().exp();
        assert!(
            close(step, binomial_lpmf(n, 20, 0.35).unwrap().exp(), 1e-13),
            "binomial {n}"
        );
    }
}

/// Probability masses sum to one and densities integrate to one.
#[test]
fn distributions_are_normalized() {
    let poisson: f64 = (0..200i64)
        .map(|n| poisson_lpmf(n, 12.0).unwrap().exp())
        .sum();
    assert!(close(poisson, 1.0, 1e-13));
    let nb: f64 = (0..2000i64)
        .map(|n| neg_binomial_2_lpmf(n, 9.0, 1.5).unwrap().exp())
        .sum();
    assert!(close(nb, 1.0, 1e-12));
    let binom: f64 = (0..=40i64)
        .map(|n| binomial_lpmf(n, 40, 0.3).unwrap().exp())
        .sum();
    assert!(close(binom, 1.0, 1e-13));
    let logit: f64 = (0..=40i64)
        .map(|n| binomial_logit_lpmf(n, 40, -0.8).unwrap().exp())
        .sum();
    assert!(close(logit, 1.0, 1e-13));
    let cat: f64 = (1..=3i64)
        .map(|n| categorical_logit_lpmf(n, &[0.2, -1.0, 2.5]).unwrap().exp())
        .sum();
    assert!(close(cat, 1.0, 1e-15));

    // Composite Simpson rule on a wide interval.
    let integrate = |f: &dyn Fn(f64) -> f64, a: f64, b: f64| {
        let n = 20_000;
        let h = (b - a) / f64::from(n);
        let mut s = f(a) + f(b);
        for i in 1..n {
            let x = a + f64::from(i) * h;
            s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
        }
        s * h / 3.0
    };
    type Density = Box<dyn Fn(f64) -> f64>;
    let densities: Vec<(&str, Density, f64, f64)> = vec![
        (
            "normal",
            Box::new(|y| normal_lpdf(y, 1.0, 2.0).unwrap().exp()),
            -30.0,
            30.0,
        ),
        (
            "student_t",
            Box::new(|y| student_t_lpdf(y, 30.0, 0.0, 1.0).unwrap().exp()),
            -60.0,
            60.0,
        ),
        (
            "logistic",
            Box::new(|y| logistic_lpdf(y, 0.0, 1.0).unwrap().exp()),
            -60.0,
            60.0,
        ),
        (
            "gamma",
            Box::new(|y| gamma_lpdf(y, 3.0, 2.0).unwrap().exp()),
            1e-12,
            40.0,
        ),
        (
            "beta",
            Box::new(|y| beta_lpdf(y, 2.0, 3.0).unwrap().exp()),
            0.0,
            1.0,
        ),
        (
            "lognormal",
            Box::new(|y| lognormal_lpdf(y, 0.0, 0.5).unwrap().exp()),
            0.0,
            40.0,
        ),
        (
            "weibull",
            Box::new(|y| weibull_lpdf(y, 2.0, 1.5).unwrap().exp()),
            0.0,
            30.0,
        ),
        (
            "inv_gamma",
            Box::new(|y| inv_gamma_lpdf(y, 4.0, 2.0).unwrap().exp()),
            1e-9,
            400.0,
        ),
    ];
    for (name, f, a, b) in densities {
        let mass = integrate(&*f, a, b);
        assert!(close(mass, 1.0, 1e-6), "{name} integrates to {mass}");
    }
}

/// A vectorized call equals the sum of scalar calls, in value and gradient.
#[test]
fn vectorized_equals_sum_of_scalars() {
    let ys = [0.3, -1.2, 2.5, 0.0];
    let mus = [0.1, 0.2, -0.3, 0.4];
    let g_vec = grad(
        |p| normal_lpdf(&ys[..], &[p[0], p[1], p[2], p[3]][..], p[4]).unwrap(),
        &[mus[0], mus[1], mus[2], mus[3], 1.3],
    );
    let g_sum = grad(
        |p| {
            (0..4)
                .map(|i| normal_lpdf(ys[i], p[i], p[4]).unwrap())
                .fold(p[0] * 0.0, |a, b| a + b)
        },
        &[mus[0], mus[1], mus[2], mus[3], 1.3],
    );
    for (a, b) in g_vec.iter().zip(&g_sum) {
        assert!(close(*a, *b, 1e-15));
    }
    let counts = vec![0i64, 3, 1, 7];
    let v = poisson_log_lpmf(&counts, 0.8).unwrap();
    let s: f64 = counts
        .iter()
        .map(|&n| poisson_log_lpmf(n, 0.8).unwrap())
        .sum();
    assert!(close(v, s, 1e-15));
}

/// Fused kernels give exact second derivatives: forward over reverse equals forward over forward.
#[test]
fn kernel_hessians_agree_across_modes() {
    let y = [0.4, 1.9, 0.7];
    let checks: Vec<(&str, f64, f64)> = {
        let mut v = Vec::new();
        let x = [0.6, 1.4];
        let h1 = hessian(|p| normal_lpdf(&y[..], p[0], p[1]).unwrap(), &x);
        let h2 = hessian_forward(|p| normal_lpdf(&y[..], p[0], p[1]).unwrap(), &x);
        v.push(("normal", h1.max_abs_diff(&h2), h1[(1, 1)]));
        let x = [3.5, 0.2, 1.1];
        let h1 = hessian(|p| student_t_lpdf(&y[..], p[0], p[1], p[2]).unwrap(), &x);
        let h2 = hessian_forward(|p| student_t_lpdf(&y[..], p[0], p[1], p[2]).unwrap(), &x);
        v.push(("student_t", h1.max_abs_diff(&h2), h1[(0, 0)]));
        let x = [2.2, 1.3];
        let h1 = hessian(|p| gamma_lpdf(&y[..], p[0], p[1]).unwrap(), &x);
        let h2 = hessian_forward(|p| gamma_lpdf(&y[..], p[0], p[1]).unwrap(), &x);
        v.push(("gamma", h1.max_abs_diff(&h2), h1[(0, 0)]));
        let yb = [0.2, 0.55, 0.9];
        let x = [1.7, 2.4];
        let h1 = hessian(|p| beta_lpdf(&yb[..], p[0], p[1]).unwrap(), &x);
        let h2 = hessian_forward(|p| beta_lpdf(&yb[..], p[0], p[1]).unwrap(), &x);
        v.push(("beta", h1.max_abs_diff(&h2), h1[(0, 1)]));
        let n = [0i64, 4, 9];
        let x = [5.0, 2.0];
        let h1 = hessian(|p| neg_binomial_2_lpmf(&n[..], p[0], p[1]).unwrap(), &x);
        let h2 = hessian_forward(|p| neg_binomial_2_lpmf(&n[..], p[0], p[1]).unwrap(), &x);
        v.push(("neg_binomial_2", h1.max_abs_diff(&h2), h1[(1, 1)]));
        let x = [2.5, 1.5];
        let h1 = hessian(|p| gamma_lcdf(1.7, p[0], p[1]).unwrap(), &x);
        let h2 = hessian_forward(|p| gamma_lcdf(1.7, p[0], p[1]).unwrap(), &x);
        v.push(("gamma_lcdf", h1.max_abs_diff(&h2), h1[(0, 0)]));
        v
    };
    for (name, diff, scale) in checks {
        assert!(
            diff <= 1e-12 * (1.0 + scale.abs()),
            "{name}: Hessian modes differ by {diff}"
        );
    }
}

#[test]
fn errors_distinguish_parameters_from_observations() {
    assert!(matches!(
        normal_lpdf(1.0, 0.0, -1.0),
        Err(Error::InvalidParameter { name: "sigma", .. })
    ));
    assert!(matches!(
        normal_lpdf(f64::NAN, 0.0, 1.0),
        Err(Error::InvalidObservation { name: "y", .. })
    ));
    assert!(matches!(
        poisson_lpmf(-1i64, 2.0),
        Err(Error::InvalidObservation { name: "n", .. })
    ));
    assert!(matches!(
        normal_lpdf(&[1.0, 2.0][..], &[0.0, 1.0, 2.0][..], 1.0),
        Err(Error::DimensionMismatch { .. })
    ));
    let empty: [f64; 0] = [];
    assert_eq!(normal_lpdf(&empty[..], 0.0, 1.0).unwrap(), 0.0);
    // Stan returns -inf instead of an error at these boundaries.
    assert_eq!(lognormal_lpdf(0.0, 0.0, 1.0).unwrap(), f64::NEG_INFINITY);
    assert_eq!(poisson_lpmf(3i64, 0.0).unwrap(), f64::NEG_INFINITY);
    assert_eq!(poisson_lpmf(0i64, 0.0).unwrap(), 0.0);
}

#[cfg(feature = "rand")]
#[test]
fn samplers_have_the_right_moments() {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(20261111);
    let n = 40_000;
    let mut check =
        |name: &str, draw: &mut dyn FnMut(&mut rand::rngs::StdRng) -> f64, mean: f64, var: f64| {
            let xs: Vec<f64> = (0..n).map(|_| draw(&mut rng)).collect();
            let m = xs.iter().sum::<f64>() / n as f64;
            let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
            let se = (var / n as f64).sqrt();
            assert!((m - mean).abs() < 5.0 * se, "{name}: mean {m} vs {mean}");
            assert!(
                (v - var).abs() < 0.05 * var,
                "{name}: variance {v} vs {var}"
            );
        };
    check(
        "normal",
        &mut |r| normal_rng(1.0, 2.0, r).unwrap(),
        1.0,
        4.0,
    );
    check("gamma", &mut |r| gamma_rng(3.0, 2.0, r).unwrap(), 1.5, 0.75);
    check(
        "beta",
        &mut |r| beta_rng(2.0, 5.0, r).unwrap(),
        2.0 / 7.0,
        10.0 / (49.0 * 8.0),
    );
    check(
        "exponential",
        &mut |r| exponential_rng(0.5, r).unwrap(),
        2.0,
        4.0,
    );
    check(
        "poisson",
        &mut |r| poisson_rng(4.5, r).unwrap() as f64,
        4.5,
        4.5,
    );
    check(
        "neg_binomial_2",
        &mut |r| neg_binomial_2_rng(6.0, 3.0, r).unwrap() as f64,
        6.0,
        18.0,
    );
    check(
        "binomial",
        &mut |r| binomial_rng(30, 0.2, r).unwrap() as f64,
        6.0,
        4.8,
    );
    check(
        "student_t",
        &mut |r| student_t_rng(10.0, 0.0, 1.0, r).unwrap(),
        0.0,
        1.25,
    );
    check(
        "weibull",
        &mut |r| weibull_rng(2.0, 1.0, r).unwrap(),
        0.886_226_925_452_758,
        0.214_601_836_602_552,
    );
    check(
        "categorical",
        &mut |r| categorical_rng(&[0.2, 0.3, 0.5], r).unwrap() as f64,
        2.3,
        0.61,
    );
    assert!(normal_rng(0.0, -1.0, &mut rng).is_err());
}

/// The LKJ normalizing constant, checked independently for `K = 2`: with `L = [[1, 0], [r, s]]` and
/// `s = sqrt(1 - r^2)`, the density integrates to one over `r` in `(-1, 1)`.
#[test]
fn lkj_corr_cholesky_is_normalized() {
    use stanmath_rs_core::Matrix;
    for eta in [0.7, 1.0, 2.5] {
        let n = 200_000;
        let h = 2.0 / f64::from(n);
        let mut mass = 0.0;
        for i in 0..n {
            let r = -1.0 + (f64::from(i) + 0.5) * h;
            let l = Matrix::from_rows(&[[1.0, 0.0], [r, (1.0 - r * r).sqrt()]]);
            mass += lkj_corr_cholesky_lpdf(&l, eta).unwrap().exp() * h;
        }
        assert!(close(mass, 1.0, 1e-4), "eta {eta}: mass {mass}");
    }
}
