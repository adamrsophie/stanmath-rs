//! Every vectorized `_rng` signature of the Stan language, through `apply_broadcast1` to
//! `apply_broadcast4`.
//!
//! Written by `tools/reference/gen_rng.py` from the CmdStan 2.40.0 signature dump: 1995 Stan
//! signatures in 124 Rust shapes (vector, row vector, and array forms of one argument are the
//! same Rust slice). Each call must return the scalar sampler's draws at each index, in order from
//! the same seed; the scalar samplers are checked against their CDFs or moments by `samplers.rs`.
//!
//! Not vectorized in Stan, or with container parameters: bernoulli_logit_glm_rng, categorical_rng, categorical_logit_rng, dirichlet_rng, dirichlet_multinomial_rng, hmm_latent_rng, hypergeometric_rng, inv_wishart_rng, inv_wishart_cholesky_rng, lkj_corr_rng, lkj_corr_cholesky_rng, multi_normal_rng, multi_normal_cholesky_rng, multi_student_t_rng, multi_student_t_cholesky_rng, multinomial_rng, multinomial_logit_rng, ordered_logistic_rng, ordered_probit_rng, poisson_binomial_rng, std_normal_rng, wishart_rng, wishart_cholesky_rng.
#![cfg(feature = "rand")]
#![allow(
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::unnecessary_cast
)]

use rand::SeedableRng;
use rand::rngs::StdRng;
use stanmath_rs_autodiff::functions::{
    apply_broadcast1, apply_broadcast2, apply_broadcast3, apply_broadcast4,
};
use stanmath_rs_core::Result;
use stanmath_rs_distributions::*;

#[test]
fn vectorized_rng_signatures() {
    let mut failures = Vec::new();
    {
        let a = [0.3, 0.306, 0.312];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("bernoulli_rng", &a[..], |a| bernoulli_rng(a as _, &mut rng));
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3).map(|i| bernoulli_rng(a[i] as _, &mut rng)).collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("bernoulli_rng(vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("bernoulli_logit_rng", &a[..], |a| {
            bernoulli_logit_rng(a as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| bernoulli_logit_rng(a[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("bernoulli_logit_rng(vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [5.0, 5.1, 5.2];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("beta_rng", 2.0, &b[..], |a, b| {
            beta_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_rng(2.0 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("beta_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("beta_rng", &a[..], 5.0, |a, b| {
            beta_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_rng(a[i] as _, 5.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("beta_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.0, 2.04, 2.08];
        let b = [5.0, 5.1, 5.2];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("beta_rng", &a[..], &b[..], |a, b| {
            beta_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("beta_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let c = [0.7, 0.714, 0.728];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_binomial_rng", 10_i64, 2.5, &c[..], |a, b, c| {
            beta_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_binomial_rng(10_i64 as _, 2.5 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_binomial_rng(scalar, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_binomial_rng", 10_i64, &b[..], 0.7, |a, b, c| {
            beta_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_binomial_rng(10_i64 as _, b[i] as _, 0.7 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_binomial_rng(scalar, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [2.5, 2.55, 2.6];
        let c = [0.7, 0.714, 0.728];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_binomial_rng", 10_i64, &b[..], &c[..], |a, b, c| {
            beta_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_binomial_rng(10_i64 as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_binomial_rng(scalar, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [10_i64, 10_i64, 10_i64];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_binomial_rng", &a[..], 2.5, 0.7, |a, b, c| {
            beta_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_binomial_rng(a[i] as _, 2.5 as _, 0.7 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_binomial_rng(vec, scalar, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [10_i64, 10_i64, 10_i64];
        let c = [0.7, 0.714, 0.728];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_binomial_rng", &a[..], 2.5, &c[..], |a, b, c| {
            beta_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_binomial_rng(a[i] as _, 2.5 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_binomial_rng(vec, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [10_i64, 10_i64, 10_i64];
        let b = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_binomial_rng", &a[..], &b[..], 0.7, |a, b, c| {
            beta_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_binomial_rng(a[i] as _, b[i] as _, 0.7 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_binomial_rng(vec, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [10_i64, 10_i64, 10_i64];
        let b = [2.5, 2.55, 2.6];
        let c = [0.7, 0.714, 0.728];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_binomial_rng", &a[..], &b[..], &c[..], |a, b, c| {
            beta_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_binomial_rng(a[i] as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_binomial_rng(vec, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let c = [1.5, 1.53, 1.56];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_neg_binomial_rng", 2.5, 3.7, &c[..], |a, b, c| {
            beta_neg_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_neg_binomial_rng(2.5 as _, 3.7 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_neg_binomial_rng(scalar, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [3.7, 3.774, 3.848];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_neg_binomial_rng", 2.5, &b[..], 1.5, |a, b, c| {
            beta_neg_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_neg_binomial_rng(2.5 as _, b[i] as _, 1.5 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_neg_binomial_rng(scalar, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [3.7, 3.774, 3.848];
        let c = [1.5, 1.53, 1.56];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_neg_binomial_rng", 2.5, &b[..], &c[..], |a, b, c| {
            beta_neg_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_neg_binomial_rng(2.5 as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_neg_binomial_rng(scalar, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_neg_binomial_rng", &a[..], 3.7, 1.5, |a, b, c| {
            beta_neg_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_neg_binomial_rng(a[i] as _, 3.7 as _, 1.5 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_neg_binomial_rng(vec, scalar, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let c = [1.5, 1.53, 1.56];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_neg_binomial_rng", &a[..], 3.7, &c[..], |a, b, c| {
            beta_neg_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_neg_binomial_rng(a[i] as _, 3.7 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_neg_binomial_rng(vec, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let b = [3.7, 3.774, 3.848];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("beta_neg_binomial_rng", &a[..], &b[..], 1.5, |a, b, c| {
            beta_neg_binomial_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_neg_binomial_rng(a[i] as _, b[i] as _, 1.5 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_neg_binomial_rng(vec, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let b = [3.7, 3.774, 3.848];
        let c = [1.5, 1.53, 1.56];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3(
            "beta_neg_binomial_rng",
            &a[..],
            &b[..],
            &c[..],
            |a, b, c| beta_neg_binomial_rng(a as _, b as _, c as _, &mut rng),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_neg_binomial_rng(a[i] as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_neg_binomial_rng(vec, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [5.0, 5.1, 5.2];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("beta_proportion_rng", 0.4, &b[..], |a, b| {
            beta_proportion_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_proportion_rng(0.4 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_proportion_rng(scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.4, 0.408, 0.416];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("beta_proportion_rng", &a[..], 5.0, |a, b| {
            beta_proportion_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_proportion_rng(a[i] as _, 5.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_proportion_rng(vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.4, 0.408, 0.416];
        let b = [5.0, 5.1, 5.2];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("beta_proportion_rng", &a[..], &b[..], |a, b| {
            beta_proportion_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| beta_proportion_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "beta_proportion_rng(vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [0.4, 0.408, 0.416];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("binomial_rng", 10_i64, &b[..], |a, b| {
            binomial_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| binomial_rng(10_i64 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("binomial_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [10_i64, 10_i64, 10_i64];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("binomial_rng", &a[..], 0.4, |a, b| {
            binomial_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| binomial_rng(a[i] as _, 0.4 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("binomial_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [10_i64, 10_i64, 10_i64];
        let b = [0.4, 0.408, 0.416];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("binomial_rng", &a[..], &b[..], |a, b| {
            binomial_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| binomial_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("binomial_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("cauchy_rng", 0.5, &b[..], |a, b| {
            cauchy_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| cauchy_rng(0.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("cauchy_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("cauchy_rng", &a[..], 2.0, |a, b| {
            cauchy_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| cauchy_rng(a[i] as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("cauchy_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("cauchy_rng", &a[..], &b[..], |a, b| {
            cauchy_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| cauchy_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("cauchy_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [3.0, 3.06, 3.12];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("chi_square_rng", &a[..], |a| {
            chi_square_rng(a as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| chi_square_rng(a[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("chi_square_rng(vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [5_i64, 5_i64, 5_i64];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("discrete_range_rng", 1_i64, &b[..], |a, b| {
            discrete_range_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| discrete_range_rng(1_i64 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "discrete_range_rng(scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [1_i64, 1_i64, 1_i64];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("discrete_range_rng", &a[..], 5_i64, |a, b| {
            discrete_range_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| discrete_range_rng(a[i] as _, 5_i64 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "discrete_range_rng(vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [1_i64, 1_i64, 1_i64];
        let b = [5_i64, 5_i64, 5_i64];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("discrete_range_rng", &a[..], &b[..], |a, b| {
            discrete_range_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| discrete_range_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("discrete_range_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("double_exponential_rng", 0.5, &b[..], |a, b| {
            double_exponential_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| double_exponential_rng(0.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "double_exponential_rng(scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("double_exponential_rng", &a[..], 1.2, |a, b| {
            double_exponential_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| double_exponential_rng(a[i] as _, 1.2 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "double_exponential_rng(vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("double_exponential_rng", &a[..], &b[..], |a, b| {
            double_exponential_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| double_exponential_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "double_exponential_rng(vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let c = [0.8, 0.816, 0.832];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("exp_mod_normal_rng", 0.5, 1.2, &c[..], |a, b, c| {
            exp_mod_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| exp_mod_normal_rng(0.5 as _, 1.2 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "exp_mod_normal_rng(scalar, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("exp_mod_normal_rng", 0.5, &b[..], 0.8, |a, b, c| {
            exp_mod_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| exp_mod_normal_rng(0.5 as _, b[i] as _, 0.8 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "exp_mod_normal_rng(scalar, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let c = [0.8, 0.816, 0.832];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("exp_mod_normal_rng", 0.5, &b[..], &c[..], |a, b, c| {
            exp_mod_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| exp_mod_normal_rng(0.5 as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "exp_mod_normal_rng(scalar, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("exp_mod_normal_rng", &a[..], 1.2, 0.8, |a, b, c| {
            exp_mod_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| exp_mod_normal_rng(a[i] as _, 1.2 as _, 0.8 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "exp_mod_normal_rng(vec, scalar, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let c = [0.8, 0.816, 0.832];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("exp_mod_normal_rng", &a[..], 1.2, &c[..], |a, b, c| {
            exp_mod_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| exp_mod_normal_rng(a[i] as _, 1.2 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "exp_mod_normal_rng(vec, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("exp_mod_normal_rng", &a[..], &b[..], 0.8, |a, b, c| {
            exp_mod_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| exp_mod_normal_rng(a[i] as _, b[i] as _, 0.8 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "exp_mod_normal_rng(vec, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let c = [0.8, 0.816, 0.832];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("exp_mod_normal_rng", &a[..], &b[..], &c[..], |a, b, c| {
            exp_mod_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| exp_mod_normal_rng(a[i] as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "exp_mod_normal_rng(vec, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("exponential_rng", &a[..], |a| {
            exponential_rng(a as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| exponential_rng(a[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("exponential_rng(vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("frechet_rng", 2.0, &b[..], |a, b| {
            frechet_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| frechet_rng(2.0 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("frechet_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("frechet_rng", &a[..], 1.2, |a, b| {
            frechet_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| frechet_rng(a[i] as _, 1.2 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("frechet_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.0, 2.04, 2.08];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("frechet_rng", &a[..], &b[..], |a, b| {
            frechet_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| frechet_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("frechet_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("gamma_rng", 2.5, &b[..], |a, b| {
            gamma_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| gamma_rng(2.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("gamma_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("gamma_rng", &a[..], 0.5, |a, b| {
            gamma_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| gamma_rng(a[i] as _, 0.5 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("gamma_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let b = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("gamma_rng", &a[..], &b[..], |a, b| {
            gamma_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| gamma_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("gamma_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("gumbel_rng", 0.5, &b[..], |a, b| {
            gumbel_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| gumbel_rng(0.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("gumbel_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("gumbel_rng", &a[..], 1.2, |a, b| {
            gumbel_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| gumbel_rng(a[i] as _, 1.2 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("gumbel_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("gumbel_rng", &a[..], &b[..], |a, b| {
            gumbel_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| gumbel_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("gumbel_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [3.0, 3.06, 3.12];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("inv_chi_square_rng", &a[..], |a| {
            inv_chi_square_rng(a as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| inv_chi_square_rng(a[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("inv_chi_square_rng(vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("inv_gamma_rng", 2.5, &b[..], |a, b| {
            inv_gamma_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| inv_gamma_rng(2.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("inv_gamma_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("inv_gamma_rng", &a[..], 0.5, |a, b| {
            inv_gamma_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| inv_gamma_rng(a[i] as _, 0.5 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("inv_gamma_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let b = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("inv_gamma_rng", &a[..], &b[..], |a, b| {
            inv_gamma_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| inv_gamma_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("inv_gamma_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("logistic_rng", 0.5, &b[..], |a, b| {
            logistic_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| logistic_rng(0.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("logistic_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("logistic_rng", &a[..], 2.0, |a, b| {
            logistic_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| logistic_rng(a[i] as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("logistic_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("logistic_rng", &a[..], &b[..], |a, b| {
            logistic_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| logistic_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("logistic_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [3.0, 3.06, 3.12];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("loglogistic_rng", 2.0, &b[..], |a, b| {
            loglogistic_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| loglogistic_rng(2.0 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("loglogistic_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("loglogistic_rng", &a[..], 3.0, |a, b| {
            loglogistic_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| loglogistic_rng(a[i] as _, 3.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("loglogistic_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.0, 2.04, 2.08];
        let b = [3.0, 3.06, 3.12];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("loglogistic_rng", &a[..], &b[..], |a, b| {
            loglogistic_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| loglogistic_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("loglogistic_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [0.7, 0.714, 0.728];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("lognormal_rng", 0.2, &b[..], |a, b| {
            lognormal_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| lognormal_rng(0.2 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("lognormal_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.2, 0.204, 0.208];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("lognormal_rng", &a[..], 0.7, |a, b| {
            lognormal_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| lognormal_rng(a[i] as _, 0.7 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("lognormal_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.2, 0.204, 0.208];
        let b = [0.7, 0.714, 0.728];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("lognormal_rng", &a[..], &b[..], |a, b| {
            lognormal_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| lognormal_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("lognormal_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [0.7, 0.714, 0.728];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_rng", 2.5, &b[..], |a, b| {
            neg_binomial_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_rng(2.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "neg_binomial_rng(scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_rng", &a[..], 0.7, |a, b| {
            neg_binomial_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_rng(a[i] as _, 0.7 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "neg_binomial_rng(vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let b = [0.7, 0.714, 0.728];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_rng", &a[..], &b[..], |a, b| {
            neg_binomial_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("neg_binomial_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [4.0, 4.08, 4.16];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_2_rng", 2.5, &b[..], |a, b| {
            neg_binomial_2_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_2_rng(2.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "neg_binomial_2_rng(scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_2_rng", &a[..], 4.0, |a, b| {
            neg_binomial_2_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_2_rng(a[i] as _, 4.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "neg_binomial_2_rng(vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let b = [4.0, 4.08, 4.16];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_2_rng", &a[..], &b[..], |a, b| {
            neg_binomial_2_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_2_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("neg_binomial_2_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [4.0, 4.08, 4.16];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_2_log_rng", 0.9, &b[..], |a, b| {
            neg_binomial_2_log_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_2_log_rng(0.9 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "neg_binomial_2_log_rng(scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.9, 0.918, 0.936];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_2_log_rng", &a[..], 4.0, |a, b| {
            neg_binomial_2_log_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_2_log_rng(a[i] as _, 4.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "neg_binomial_2_log_rng(vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.9, 0.918, 0.936];
        let b = [4.0, 4.08, 4.16];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("neg_binomial_2_log_rng", &a[..], &b[..], |a, b| {
            neg_binomial_2_log_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| neg_binomial_2_log_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "neg_binomial_2_log_rng(vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("normal_rng", 0.5, &b[..], |a, b| {
            normal_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| normal_rng(0.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("normal_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("normal_rng", &a[..], 1.2, |a, b| {
            normal_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| normal_rng(a[i] as _, 1.2 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("normal_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("normal_rng", &a[..], &b[..], |a, b| {
            normal_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| normal_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("normal_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("pareto_rng", 0.5, &b[..], |a, b| {
            pareto_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_rng(0.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("pareto_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("pareto_rng", &a[..], 1.2, |a, b| {
            pareto_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_rng(a[i] as _, 1.2 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("pareto_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("pareto_rng", &a[..], &b[..], |a, b| {
            pareto_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("pareto_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("pareto_type_2_rng", 0.5, 1.2, &c[..], |a, b, c| {
            pareto_type_2_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_type_2_rng(0.5 as _, 1.2 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "pareto_type_2_rng(scalar, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("pareto_type_2_rng", 0.5, &b[..], 2.0, |a, b, c| {
            pareto_type_2_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_type_2_rng(0.5 as _, b[i] as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "pareto_type_2_rng(scalar, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("pareto_type_2_rng", 0.5, &b[..], &c[..], |a, b, c| {
            pareto_type_2_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_type_2_rng(0.5 as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "pareto_type_2_rng(scalar, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("pareto_type_2_rng", &a[..], 1.2, 2.0, |a, b, c| {
            pareto_type_2_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_type_2_rng(a[i] as _, 1.2 as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "pareto_type_2_rng(vec, scalar, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("pareto_type_2_rng", &a[..], 1.2, &c[..], |a, b, c| {
            pareto_type_2_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_type_2_rng(a[i] as _, 1.2 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "pareto_type_2_rng(vec, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("pareto_type_2_rng", &a[..], &b[..], 2.0, |a, b, c| {
            pareto_type_2_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_type_2_rng(a[i] as _, b[i] as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "pareto_type_2_rng(vec, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("pareto_type_2_rng", &a[..], &b[..], &c[..], |a, b, c| {
            pareto_type_2_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| pareto_type_2_rng(a[i] as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "pareto_type_2_rng(vec, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("poisson_rng", &a[..], |a| poisson_rng(a as _, &mut rng));
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3).map(|i| poisson_rng(a[i] as _, &mut rng)).collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("poisson_rng(vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.9, 0.918, 0.936];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("poisson_log_rng", &a[..], |a| {
            poisson_log_rng(a as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| poisson_log_rng(a[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("poisson_log_rng(vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("rayleigh_rng", &a[..], |a| rayleigh_rng(a as _, &mut rng));
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3).map(|i| rayleigh_rng(a[i] as _, &mut rng)).collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("rayleigh_rng(vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [0.8, 0.816, 0.832];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("scaled_inv_chi_square_rng", 3.0, &b[..], |a, b| {
            scaled_inv_chi_square_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| scaled_inv_chi_square_rng(3.0 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "scaled_inv_chi_square_rng(scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [3.0, 3.06, 3.12];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("scaled_inv_chi_square_rng", &a[..], 0.8, |a, b| {
            scaled_inv_chi_square_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| scaled_inv_chi_square_rng(a[i] as _, 0.8 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "scaled_inv_chi_square_rng(vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [3.0, 3.06, 3.12];
        let b = [0.8, 0.816, 0.832];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("scaled_inv_chi_square_rng", &a[..], &b[..], |a, b| {
            scaled_inv_chi_square_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| scaled_inv_chi_square_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "scaled_inv_chi_square_rng(vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let c = [0.3, 0.306, 0.312];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3(
            "skew_double_exponential_rng",
            0.5,
            1.2,
            &c[..],
            |a, b, c| skew_double_exponential_rng(a as _, b as _, c as _, &mut rng),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_double_exponential_rng(0.5 as _, 1.2 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_double_exponential_rng(scalar, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3(
            "skew_double_exponential_rng",
            0.5,
            &b[..],
            0.3,
            |a, b, c| skew_double_exponential_rng(a as _, b as _, c as _, &mut rng),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_double_exponential_rng(0.5 as _, b[i] as _, 0.3 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_double_exponential_rng(scalar, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let c = [0.3, 0.306, 0.312];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3(
            "skew_double_exponential_rng",
            0.5,
            &b[..],
            &c[..],
            |a, b, c| skew_double_exponential_rng(a as _, b as _, c as _, &mut rng),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_double_exponential_rng(0.5 as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_double_exponential_rng(scalar, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3(
            "skew_double_exponential_rng",
            &a[..],
            1.2,
            0.3,
            |a, b, c| skew_double_exponential_rng(a as _, b as _, c as _, &mut rng),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_double_exponential_rng(a[i] as _, 1.2 as _, 0.3 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_double_exponential_rng(vec, scalar, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let c = [0.3, 0.306, 0.312];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3(
            "skew_double_exponential_rng",
            &a[..],
            1.2,
            &c[..],
            |a, b, c| skew_double_exponential_rng(a as _, b as _, c as _, &mut rng),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_double_exponential_rng(a[i] as _, 1.2 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_double_exponential_rng(vec, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3(
            "skew_double_exponential_rng",
            &a[..],
            &b[..],
            0.3,
            |a, b, c| skew_double_exponential_rng(a as _, b as _, c as _, &mut rng),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_double_exponential_rng(a[i] as _, b[i] as _, 0.3 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_double_exponential_rng(vec, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let c = [0.3, 0.306, 0.312];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3(
            "skew_double_exponential_rng",
            &a[..],
            &b[..],
            &c[..],
            |a, b, c| skew_double_exponential_rng(a as _, b as _, c as _, &mut rng),
        );
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_double_exponential_rng(a[i] as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_double_exponential_rng(vec, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("skew_normal_rng", 0.5, 1.2, &c[..], |a, b, c| {
            skew_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_normal_rng(0.5 as _, 1.2 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_normal_rng(scalar, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("skew_normal_rng", 0.5, &b[..], 2.0, |a, b, c| {
            skew_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_normal_rng(0.5 as _, b[i] as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_normal_rng(scalar, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("skew_normal_rng", 0.5, &b[..], &c[..], |a, b, c| {
            skew_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_normal_rng(0.5 as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_normal_rng(scalar, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("skew_normal_rng", &a[..], 1.2, 2.0, |a, b, c| {
            skew_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_normal_rng(a[i] as _, 1.2 as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_normal_rng(vec, scalar, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("skew_normal_rng", &a[..], 1.2, &c[..], |a, b, c| {
            skew_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_normal_rng(a[i] as _, 1.2 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_normal_rng(vec, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("skew_normal_rng", &a[..], &b[..], 2.0, |a, b, c| {
            skew_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_normal_rng(a[i] as _, b[i] as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_normal_rng(vec, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("skew_normal_rng", &a[..], &b[..], &c[..], |a, b, c| {
            skew_normal_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| skew_normal_rng(a[i] as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "skew_normal_rng(vec, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("student_t_rng", 3.0, 0.5, &c[..], |a, b, c| {
            student_t_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| student_t_rng(3.0 as _, 0.5 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "student_t_rng(scalar, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("student_t_rng", 3.0, &b[..], 2.0, |a, b, c| {
            student_t_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| student_t_rng(3.0 as _, b[i] as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "student_t_rng(scalar, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let b = [0.5, 0.51, 0.52];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("student_t_rng", 3.0, &b[..], &c[..], |a, b, c| {
            student_t_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| student_t_rng(3.0 as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "student_t_rng(scalar, vec, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [3.0, 3.06, 3.12];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("student_t_rng", &a[..], 0.5, 2.0, |a, b, c| {
            student_t_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| student_t_rng(a[i] as _, 0.5 as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "student_t_rng(vec, scalar, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [3.0, 3.06, 3.12];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("student_t_rng", &a[..], 0.5, &c[..], |a, b, c| {
            student_t_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| student_t_rng(a[i] as _, 0.5 as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "student_t_rng(vec, scalar, vec): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [3.0, 3.06, 3.12];
        let b = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("student_t_rng", &a[..], &b[..], 2.0, |a, b, c| {
            student_t_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| student_t_rng(a[i] as _, b[i] as _, 2.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!(
                "student_t_rng(vec, vec, scalar): {got:?} vs {want:?}"
            ));
        }
    }
    {
        let a = [3.0, 3.06, 3.12];
        let b = [0.5, 0.51, 0.52];
        let c = [2.0, 2.04, 2.08];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast3("student_t_rng", &a[..], &b[..], &c[..], |a, b, c| {
            student_t_rng(a as _, b as _, c as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| student_t_rng(a[i] as _, b[i] as _, c[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("student_t_rng(vec, vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [1.0, 1.02, 1.04];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("uniform_rng", 0.0, &b[..], |a, b| {
            uniform_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| uniform_rng(0.0 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("uniform_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.0, 0.0, 0.0];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("uniform_rng", &a[..], 1.0, |a, b| {
            uniform_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| uniform_rng(a[i] as _, 1.0 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("uniform_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.02, 1.04];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("uniform_rng", &a[..], &b[..], |a, b| {
            uniform_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| uniform_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("uniform_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("von_mises_rng", 0.5, &b[..], |a, b| {
            von_mises_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| von_mises_rng(0.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("von_mises_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("von_mises_rng", &a[..], 1.2, |a, b| {
            von_mises_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| von_mises_rng(a[i] as _, 1.2 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("von_mises_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [0.5, 0.51, 0.52];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("von_mises_rng", &a[..], &b[..], |a, b| {
            von_mises_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| von_mises_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("von_mises_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("weibull_rng", 2.5, &b[..], |a, b| {
            weibull_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| weibull_rng(2.5 as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("weibull_rng(scalar, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("weibull_rng", &a[..], 1.2, |a, b| {
            weibull_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| weibull_rng(a[i] as _, 1.2 as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("weibull_rng(vec, scalar): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let b = [1.2, 1.224, 1.248];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast2("weibull_rng", &a[..], &b[..], |a, b| {
            weibull_rng(a as _, b as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| weibull_rng(a[i] as _, b[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("weibull_rng(vec, vec): {got:?} vs {want:?}"));
        }
    }
    {
        let a = [2.5, 2.55, 2.6];
        let mut rng = StdRng::seed_from_u64(7);
        let got = apply_broadcast1("yule_simon_rng", &a[..], |a| {
            yule_simon_rng(a as _, &mut rng)
        });
        let mut rng = StdRng::seed_from_u64(7);
        let want: Result<Vec<_>> = (0..3)
            .map(|i| yule_simon_rng(a[i] as _, &mut rng))
            .collect();
        if format!("{got:?}") != format!("{want:?}") {
            failures.push(format!("yule_simon_rng(vec): {got:?} vs {want:?}"));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
