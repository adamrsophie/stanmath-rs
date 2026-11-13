//! The `propto` forms of the multivariate and the other non-kernel densities against Stan Math
//! 5.4.0, as printed by `tools/stan-reference/stan_probe`. Matrix arguments share the scalar type,
//! so each case makes them autodiff variables, as the probe does. Every case is checked before the
//! test fails, so one run reports each mismatch.

// Reference values keep the 17 significant digits that Stan Math printed.
#![allow(clippy::excessive_precision)]

use stanmath_rs_autodiff::{Var, value_and_grad};
use stanmath_rs_core::Matrix;
use stanmath_rs_distributions::*;
use stanmath_rs_linalg::cholesky_decompose;

const Y: [f64; 3] = [0.3, -0.5, 1.2];
const MU: [f64; 3] = [0.1, 0.2, -0.3];

fn sigma() -> Matrix<f64> {
    Matrix::from_rows(&[[2.0, 0.3, 0.1], [0.3, 1.5, -0.2], [0.1, -0.2, 1.0]])
}

fn scale() -> Matrix<f64> {
    Matrix::from_rows(&[[1.0, 0.2, 0.0], [0.2, 1.2, 0.1], [0.0, 0.1, 0.8]])
}

fn corr() -> Matrix<f64> {
    Matrix::from_rows(&[[1.0, 0.3, -0.2], [0.3, 1.0, 0.4], [-0.2, 0.4, 1.0]])
}

fn gp_y() -> Matrix<f64> {
    Matrix::from_rows(&[[0.3, -0.5, 1.2], [0.8, 0.1, -0.4]])
}

fn chol(m: &Matrix<f64>) -> Matrix<f64> {
    cholesky_decompose(m).unwrap()
}

/// The entries of `ms`, row by row, followed by `extra`.
fn pack(ms: &[&Matrix<f64>], extra: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    for m in ms {
        for i in 0..m.nrows() {
            for j in 0..m.ncols() {
                out.push(m[(i, j)]);
            }
        }
    }
    out.extend_from_slice(extra);
    out
}

/// The `r` by `c` matrix stored row by row in `v` from `*at`, advancing `*at` past it.
fn take<'t>(v: &[Var<'t>], at: &mut usize, r: usize, c: usize) -> Matrix<Var<'t>> {
    let o = *at;
    *at += r * c;
    Matrix::from_fn(r, c, |i, j| v[o + i * c + j])
}

/// The value of `f` at `x` with every entry of `x` an autodiff variable.
fn at(x: &[f64], f: impl for<'t> Fn(&[Var<'t>]) -> Var<'t>) -> f64 {
    value_and_grad(|v| f(v), x).value
}

fn report(cases: &[(&str, f64, f64)], tol: f64) {
    let failures: Vec<String> = cases
        .iter()
        .filter(|&&(_, got, want)| !((got - want).abs() <= tol * (1.0 + want.abs())))
        .map(|(name, got, want)| format!("{name}: {got:e} vs {want:e}"))
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn multivariate_normal_and_student_t_propto_match_stan() {
    let (s, l) = (sigma(), chol(&sigma()));
    report(
        &[
            (
                "multi_normal, Sigma a variable",
                at(&pack(&[&s], &[]), |v| {
                    multi_normal_lupdf(&Y, &MU, &take(v, &mut 0, 3, 3)).unwrap()
                }),
                -1.7000757818280363,
            ),
            (
                "multi_normal full",
                multi_normal_lpdf(&Y, &MU, &s).unwrap(),
                -4.4568913814420545,
            ),
            (
                "multi_normal_cholesky, L a variable",
                at(&pack(&[&l], &[]), |v| {
                    multi_normal_cholesky_lupdf(&Y, &MU, &take(v, &mut 0, 3, 3)).unwrap()
                }),
                -1.7000757818280361,
            ),
            (
                "multi_normal_prec, Omega a variable",
                at(&pack(&[&s], &[]), |v| {
                    multi_normal_prec_lupdf(&Y, &MU, &take(v, &mut 0, 3, 3)).unwrap()
                }),
                -1.2151548639086742,
            ),
            (
                "multi_normal_prec full",
                multi_normal_prec_lpdf(&Y, &MU, &s).unwrap(),
                -3.9719704635226929,
            ),
            (
                "multi_student_t, Sigma a variable",
                at(&pack(&[&s], &[]), |v| {
                    multi_student_t_lupdf(&Y, 4.5, &MU, &take(v, &mut 0, 3, 3)).unwrap()
                }),
                -2.1016262301564734,
            ),
            (
                "multi_student_t, Sigma and nu variables",
                at(&pack(&[&s], &[4.5]), |v| {
                    multi_student_t_lupdf(&Y, v[9], &MU, &take(v, &mut 0, 3, 3)).unwrap()
                }),
                -2.9957984616198639,
            ),
            (
                "multi_student_t full",
                multi_student_t_lpdf(&Y, 4.5, &MU, &s).unwrap(),
                -4.7128932903939651,
            ),
            (
                "multi_student_t_cholesky, L a variable",
                at(&pack(&[&l], &[]), |v| {
                    multi_student_t_cholesky_lupdf(&Y, 4.5, &MU, &take(v, &mut 0, 3, 3)).unwrap()
                }),
                -2.1016262301564734,
            ),
            (
                "multi_student_t_cholesky, L and nu variables",
                at(&pack(&[&l], &[4.5]), |v| {
                    multi_student_t_cholesky_lupdf(&Y, v[9], &MU, &take(v, &mut 0, 3, 3)).unwrap()
                }),
                -2.9957984616198643,
            ),
        ],
        1e-10,
    );
    assert_eq!(multi_normal_lupdf(&Y, &MU, &s).unwrap(), 0.0);
    assert_eq!(multi_student_t_lupdf(&Y, 4.5, &MU, &s).unwrap(), 0.0);
}

#[test]
fn gaussian_process_and_dynamic_linear_model_propto_match_stan() {
    let (gy, s, l) = (gp_y(), sigma(), chol(&sigma()));
    let w = [1.5, 0.7];
    let m11 = |x: f64| Matrix::from_rows(&[[x]]);
    let dlm_y = Matrix::from_rows(&[[0.3, -0.1, 0.4]]);
    report(
        &[
            (
                "multi_gp, y and Sigma variables",
                at(&pack(&[&gy, &s], &[]), |v| {
                    let mut o = 0;
                    let (y, k) = (take(v, &mut o, 2, 3), take(v, &mut o, 3, 3));
                    multi_gp_lupdf(&y, &k, &w).unwrap()
                }),
                -2.3507473538808319,
            ),
            (
                "multi_gp, every argument a variable",
                at(&pack(&[&gy, &s], &w), |v| {
                    let mut o = 0;
                    let (y, k) = (take(v, &mut o, 2, 3), take(v, &mut o, 3, 3));
                    multi_gp_lupdf(&y, &k, &v[o..]).unwrap()
                }),
                -2.2775621076266841,
            ),
            (
                "multi_gp full",
                multi_gp_lpdf(&gy, &s, &w).unwrap(),
                -7.7911933068547201,
            ),
            (
                "multi_gp_cholesky, y and L variables",
                at(&pack(&[&gy, &l], &[]), |v| {
                    let mut o = 0;
                    let (y, k) = (take(v, &mut o, 2, 3), take(v, &mut o, 3, 3));
                    multi_gp_cholesky_lupdf(&y, &k, &w).unwrap()
                }),
                -2.3507473538808323,
            ),
            (
                "multi_gp_cholesky, every argument a variable",
                at(&pack(&[&gy, &l], &w), |v| {
                    let mut o = 0;
                    let (y, k) = (take(v, &mut o, 2, 3), take(v, &mut o, 3, 3));
                    multi_gp_cholesky_lupdf(&y, &k, &v[o..]).unwrap()
                }),
                -2.2775621076266845,
            ),
            (
                "gaussian_dlm_obs, every argument a variable",
                at(&pack(&[&dlm_y], &[1.0, 0.9, 0.5, 0.2, 0.0, 1.0]), |v| {
                    let y = take(v, &mut 0, 1, 3);
                    let m = |i: usize| Matrix::from_rows(&[[v[i]]]);
                    gaussian_dlm_obs_lupdf(&y, &m(3), &m(4), &m(5), &m(6), &v[7..8], &m(8)).unwrap()
                }),
                -0.27923180099763883,
            ),
            (
                "gaussian_dlm_obs full",
                gaussian_dlm_obs_lpdf(
                    &dlm_y,
                    &m11(1.0),
                    &m11(0.9),
                    &m11(0.5),
                    &m11(0.2),
                    &[0.0],
                    &m11(1.0),
                )
                .unwrap(),
                -3.0360474006116567,
            ),
        ],
        1e-10,
    );
}

/// The Wishart family density `$f` at `w`, `nu = 5.5`, `s`, with the matrices variables and `nu`
/// data or a variable.
macro_rules! wishart_cases {
    ($f:ident, $full:ident, $w:expr, $s:expr, $nu_data:expr, $nu_var:expr) => {{
        let x = pack(&[$w, $s], &[5.5]);
        [
            (
                concat!(stringify!($f), ", nu data"),
                at(&x, |v| {
                    let mut o = 0;
                    let (a, b) = (take(v, &mut o, 3, 3), take(v, &mut o, 3, 3));
                    $f(&a, 5.5, &b).unwrap()
                }),
                $nu_data,
            ),
            (
                concat!(stringify!($f), ", nu a variable"),
                at(&x, |v| {
                    let mut o = 0;
                    let (a, b) = (take(v, &mut o, 3, 3), take(v, &mut o, 3, 3));
                    $f(&a, v[o], &b).unwrap()
                }),
                $nu_var,
            ),
            (stringify!($full), $full($w, 5.5, $s).unwrap(), $nu_var),
        ]
    }};
}

#[test]
fn wishart_family_propto_matches_stan() {
    let (s, sc) = (sigma(), scale());
    let (l, lsc) = (chol(&s), chol(&sc));
    let mut rows = Vec::new();
    rows.extend(wishart_cases!(
        wishart_lupdf,
        wishart_lpdf,
        &s,
        &sc,
        -1.2836371006898339,
        -9.234881429870331
    ));
    rows.extend(wishart_cases!(
        wishart_cholesky_lupdf,
        wishart_cholesky_lpdf,
        &l,
        &lsc,
        0.11235816627316433,
        -5.7594446212274963
    ));
    rows.extend(wishart_cases!(
        inv_wishart_lupdf,
        inv_wishart_lpdf,
        &s,
        &sc,
        -11.943712027479529,
        -14.176492117040477
    ));
    rows.extend(wishart_cases!(
        inv_wishart_cholesky_lupdf,
        inv_wishart_cholesky_lpdf,
        &l,
        &lsc,
        -4.8292525208969828,
        -10.701055308397644
    ));
    report(&rows, 1e-10);
    assert_eq!(wishart_lupdf(&s, 5.5, &sc).unwrap(), 0.0);
}

#[test]
fn lkj_and_matrix_normal_propto_match_stan() {
    let (s, r) = (sigma(), corr());
    let rl = chol(&r);
    let (lmu, lsd) = ([0.1, 0.0, -0.2], [1.0, 0.8, 1.2]);
    let mn_mu = Matrix::from_rows(&[[0.1, 0.0, 0.2], [-0.1, 0.3, 0.0]]);
    let mn_s = Matrix::from_rows(&[[1.5, 0.2], [0.2, 0.9]]);
    report(
        &[
            (
                "lkj_corr_cholesky, L a variable",
                at(&pack(&[&rl], &[]), |v| {
                    lkj_corr_cholesky_lupdf(&take(v, &mut 0, 3, 3), 2.5).unwrap()
                }),
                -0.66588992430331384,
            ),
            (
                "lkj_corr_cholesky, L and eta variables",
                at(&pack(&[&rl], &[2.5]), |v| {
                    lkj_corr_cholesky_lupdf(&take(v, &mut 0, 3, 3), v[9]).unwrap()
                }),
                -0.95886759941612998,
            ),
            (
                "lkj_corr_cholesky full",
                lkj_corr_cholesky_lpdf(&rl, 2.5).unwrap(),
                -0.95886759941612998,
            ),
            (
                "lkj_corr, y a variable",
                at(&pack(&[&r], &[]), |v| {
                    lkj_corr_lupdf(&take(v, &mut 0, 3, 3), 2.5).unwrap()
                }),
                -0.61873458456769326,
            ),
            (
                "lkj_corr, y and eta variables",
                at(&pack(&[&r], &[2.5]), |v| {
                    lkj_corr_lupdf(&take(v, &mut 0, 3, 3), v[9]).unwrap()
                }),
                -0.9117122596805094,
            ),
            (
                "lkj_corr full",
                lkj_corr_lpdf(&r, 2.5).unwrap(),
                -0.9117122596805094,
            ),
            (
                "lkj_cov, y a variable",
                at(&pack(&[&s], &[]), |v| {
                    lkj_cov_lupdf(&take(v, &mut 0, 3, 3), &lmu, &lsd, 2.0).unwrap()
                }),
                -3.4504416736539572,
            ),
            (
                "lkj_cov, every argument a variable",
                at(&pack(&[&s], &[0.1, 0.0, -0.2, 1.0, 0.8, 1.2, 2.0]), |v| {
                    lkj_cov_lupdf(&take(v, &mut 0, 3, 3), &v[9..12], &v[12..15], v[15]).unwrap()
                }),
                -4.0251030172608315,
            ),
            (
                "lkj_cov full",
                lkj_cov_lpdf(&s, &lmu, &lsd, 2.0).unwrap(),
                -4.0251030172608315,
            ),
            (
                "matrix_normal_prec, every argument a variable",
                at(&pack(&[&gp_y(), &mn_mu, &mn_s, &s], &[]), |v| {
                    let mut o = 0;
                    let y = take(v, &mut o, 2, 3);
                    let m = take(v, &mut o, 2, 3);
                    let sg = take(v, &mut o, 2, 2);
                    let d = take(v, &mut o, 3, 3);
                    matrix_normal_prec_lupdf(&y, &m, &sg, &d).unwrap()
                }),
                -0.15178745451296272,
            ),
            (
                "matrix_normal_prec full",
                matrix_normal_prec_lpdf(&gp_y(), &mn_mu, &mn_s, &s).unwrap(),
                -5.6654186537409998,
            ),
        ],
        1e-10,
    );
    assert_eq!(lkj_cov_lupdf(&s, &lmu, &lsd, 2.0).unwrap(), 0.0);
}

#[test]
fn simplex_count_and_ordinal_propto_match_stan() {
    let (theta, alpha, beta) = ([0.2, 0.5, 0.3], [1.5, 2.0, 0.7], [0.3, -0.2, 0.8]);
    let (y, ns) = ([2i64, 3, 1], [2i64, 0, 5]);
    let (eta, cuts) = ([0.1, -0.2, 0.3], [-0.5, 0.7]);
    report(
        &[
            (
                "dirichlet, theta a variable",
                at(&theta, |v| dirichlet_lupdf(v, &alpha).unwrap()),
                -1.1366742954792146,
            ),
            (
                "dirichlet, theta and alpha variables",
                at(&pack(&[], &[theta, alpha].concat()), |v| {
                    dirichlet_lupdf(&v[..3], &v[3..]).unwrap()
                }),
                0.77179633258495395,
            ),
            (
                "dirichlet full",
                dirichlet_lpdf(&theta, &alpha).unwrap(),
                0.77179633258495395,
            ),
            (
                "categorical",
                at(&theta, |v| categorical_lupmf(&y[..], v).unwrap()),
                -3.5065578973199818,
            ),
            (
                "categorical_logit",
                at(&beta, |v| categorical_logit_lupmf(&y[..], v).unwrap()),
                -3.5408090119252038,
            ),
            (
                "multinomial",
                at(&theta, |v| multinomial_lupmf(&ns, v).unwrap()),
                -9.2387398464978805,
            ),
            (
                "multinomial full",
                multinomial_lpmf(&ns, &theta).unwrap(),
                -6.1942174087744579,
            ),
            (
                "multinomial_logit",
                at(&beta, |v| multinomial_logit_lupmf(&ns, v).unwrap()),
                -5.7618876944921418,
            ),
            (
                "multinomial_logit full",
                multinomial_logit_lpmf(&ns, &beta).unwrap(),
                -2.7173652567687188,
            ),
            (
                "dirichlet_multinomial",
                at(&alpha, |v| dirichlet_multinomial_lupmf(&ns, v).unwrap()),
                -4.782045197420798,
            ),
            (
                "dirichlet_multinomial full",
                dirichlet_multinomial_lpmf(&ns, &alpha).unwrap(),
                -5.1387201413595305,
            ),
            (
                "ordered_logistic, c a variable",
                at(&cuts, |v| {
                    ordered_logistic_lupmf(&y[..], &eta[..], v).unwrap()
                }),
                -3.6456128595120711,
            ),
            (
                "ordered_probit, c a variable",
                at(&cuts, |v| {
                    ordered_probit_lupmf(&y[..], &eta[..], v).unwrap()
                }),
                -4.0395378426011348,
            ),
            (
                "poisson_binomial",
                at(&theta, |v| poisson_binomial_lupmf(2i64, v).unwrap()),
                -1.5141277326297755,
            ),
            (
                "hypergeometric full",
                hypergeometric_lpmf(2i64, 5i64, 4i64, 6i64).unwrap(),
                -0.74193734472937756,
            ),
        ],
        1e-10,
    );
    // Integer arguments are data, so Stan returns zero, even outside the range.
    assert_eq!(hypergeometric_lupmf(2i64, 5i64, 4i64, 6i64).unwrap(), 0.0);
    assert_eq!(discrete_range_lupmf(3i64, 1i64, 5i64).unwrap(), 0.0);
    assert_eq!(discrete_range_lupmf(7i64, 1i64, 5i64).unwrap(), 0.0);
    assert_eq!(dirichlet_lupdf(&theta, &alpha).unwrap(), 0.0);
    assert_eq!(categorical_lupmf(&y[..], &theta).unwrap(), 0.0);
}

#[test]
fn wiener_propto_keeps_every_term() {
    // No term of the Wiener densities depends only on data unless all of it does.
    let w5 = at(&[1.5], |v| {
        wiener5_lupdf(1.2, v[0], 0.3, 0.5, 0.4, 0.2).unwrap()
    });
    let w7 = at(&[1.5], |v| {
        wiener_full_lupdf(1.2, v[0], 0.3, 0.5, 0.4, 0.2, 0.1, 0.1).unwrap()
    });
    report(
        &[
            ("wiener5, a a variable", w5, -1.4268689049947876),
            (
                "wiener5 full",
                wiener5_lpdf(1.2, 1.5, 0.3, 0.5, 0.4, 0.2).unwrap(),
                -1.4268689049947876,
            ),
            ("wiener_full, a a variable", w7, -1.3136644075808659),
        ],
        1e-6,
    );
    assert_eq!(wiener5_lupdf(1.2, 1.5, 0.3, 0.5, 0.4, 0.2).unwrap(), 0.0);
    let full7 = wiener_full_lpdf(1.2, 1.5, 0.3, 0.5, 0.4, 0.2, 0.1, 0.1).unwrap();
    report(&[("wiener_full keeps every term", w7, full7)], 1e-12);
}
