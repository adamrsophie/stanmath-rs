//! The 256 `student_t_qf` signatures of the Stan language: each of the four arguments a scalar or
//! a vector, row vector, or array (one Rust slice), through `apply_broadcast4`. Each call must equal
//! the scalar quantile function at each index.

use stanmath_rs::autodiff::Arg;
use stanmath_rs::prelude::*;

fn check<P, N, M, S>(p: P, nu: N, mu: M, sigma: S)
where
    P: Arg<Elem = f64> + Copy,
    N: Arg<Elem = f64> + Copy,
    M: Arg<Elem = f64> + Copy,
    S: Arg<Elem = f64> + Copy,
{
    let got = apply_broadcast4("student_t_qf", p, nu, mu, sigma, student_t_qf::<f64>).unwrap();
    let n = [p.arg_len(), nu.arg_len(), mu.arg_len(), sigma.arg_len()]
        .into_iter()
        .flatten()
        .next()
        .unwrap_or(1);
    assert_eq!(got.len(), n);
    for (i, g) in got.iter().enumerate() {
        let want = student_t_qf(p.arg_at(i), nu.arg_at(i), mu.arg_at(i), sigma.arg_at(i)).unwrap();
        assert_eq!(g.to_bits(), want.to_bits());
    }
}

#[test]
fn student_t_qf_signatures() {
    check(0.3, 4.5, 0.2, 1.3);
    check(0.3, 4.5, 0.2, &[1.3, 1.326, 1.352][..]);
    check(0.3, 4.5, &[0.2, 0.204, 0.208][..], 1.3);
    check(0.3, 4.5, &[0.2, 0.204, 0.208][..], &[1.3, 1.326, 1.352][..]);
    check(0.3, &[4.5, 4.59, 4.68][..], 0.2, 1.3);
    check(0.3, &[4.5, 4.59, 4.68][..], 0.2, &[1.3, 1.326, 1.352][..]);
    check(0.3, &[4.5, 4.59, 4.68][..], &[0.2, 0.204, 0.208][..], 1.3);
    check(
        0.3,
        &[4.5, 4.59, 4.68][..],
        &[0.2, 0.204, 0.208][..],
        &[1.3, 1.326, 1.352][..],
    );
    check(&[0.3, 0.306, 0.312][..], 4.5, 0.2, 1.3);
    check(&[0.3, 0.306, 0.312][..], 4.5, 0.2, &[1.3, 1.326, 1.352][..]);
    check(&[0.3, 0.306, 0.312][..], 4.5, &[0.2, 0.204, 0.208][..], 1.3);
    check(
        &[0.3, 0.306, 0.312][..],
        4.5,
        &[0.2, 0.204, 0.208][..],
        &[1.3, 1.326, 1.352][..],
    );
    check(&[0.3, 0.306, 0.312][..], &[4.5, 4.59, 4.68][..], 0.2, 1.3);
    check(
        &[0.3, 0.306, 0.312][..],
        &[4.5, 4.59, 4.68][..],
        0.2,
        &[1.3, 1.326, 1.352][..],
    );
    check(
        &[0.3, 0.306, 0.312][..],
        &[4.5, 4.59, 4.68][..],
        &[0.2, 0.204, 0.208][..],
        1.3,
    );
    check(
        &[0.3, 0.306, 0.312][..],
        &[4.5, 4.59, 4.68][..],
        &[0.2, 0.204, 0.208][..],
        &[1.3, 1.326, 1.352][..],
    );
}
