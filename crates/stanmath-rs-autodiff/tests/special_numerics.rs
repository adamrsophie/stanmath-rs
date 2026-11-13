//! Bessel functions, Owen's T, the Lambert W branches, and the hypergeometric functions against
//! Stan Math 5.4.0, as printed by `tools/stan-reference/stan_probe`. Every case is checked before
//! the test fails, so one run reports the relative error of each.

// Reference values keep the 17 significant digits that Stan Math printed.
#![allow(clippy::excessive_precision)]

use stanmath_rs_autodiff::functions::{
    bessel_first_kind, bessel_second_kind, hypergeometric_2F1, hypergeometric_3F2,
    hypergeometric_pFq, lambert_w0, lambert_wm1, log_modified_bessel_first_kind,
    modified_bessel_first_kind, modified_bessel_second_kind, owens_t,
};
use stanmath_rs_autodiff::{Real, grad};

const TOL: f64 = 1e-10;

fn report(cases: &[(&str, f64, f64)]) {
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|&(name, got, want)| {
            let rel = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
            (!(rel <= TOL))
                .then(|| format!("{name}: {got:e} vs {want:e} (relative error {rel:.1e})"))
        })
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn bessel_values_match_stan() {
    report(&[
        ("J0(0.5)", bessel_first_kind(0, 0.5), 0.93846980724081286),
        (
            "I0(0.5)",
            modified_bessel_first_kind(0, 0.5),
            1.0634833707413236,
        ),
        ("Y0(0.5)", bessel_second_kind(0, 0.5), -0.44451873350670656),
        (
            "K0(0.5)",
            modified_bessel_second_kind(0, 0.5),
            0.92441907122766587,
        ),
        ("J1(3.7)", bessel_first_kind(1, 3.7), 0.053833987745461789),
        (
            "I1(3.7)",
            modified_bessel_first_kind(1, 3.7),
            7.4357457965353371,
        ),
        ("Y1(3.7)", bessel_second_kind(1, 3.7), 0.41667437268380747),
        (
            "K1(3.7)",
            modified_bessel_second_kind(1, 3.7),
            0.017628035102223261,
        ),
        ("J2(30)", bessel_first_kind(2, 30.0), 0.078451246073265354),
        (
            "I2(30)",
            modified_bessel_first_kind(2, 30.0),
            730436828561.38037,
        ),
        ("Y2(30)", bessel_second_kind(2, 30.0), 0.12292410306411385),
        (
            "K2(30)",
            modified_bessel_second_kind(2, 30.0),
            2.2769929632558262e-14,
        ),
        ("J5(10)", bessel_first_kind(5, 10.0), -0.23406152818679363),
        (
            "I5(10)",
            modified_bessel_first_kind(5, 10.0),
            777.18828640326001,
        ),
        ("Y5(10)", bessel_second_kind(5, 10.0), 0.13540304768936232),
        (
            "K5(10)",
            modified_bessel_second_kind(5, 10.0),
            5.7541849985312281e-05,
        ),
        (
            "J3(0.01)",
            bessel_first_kind(3, 0.01),
            2.0833203125325523e-08,
        ),
        (
            "I3(0.01)",
            modified_bessel_first_kind(3, 0.01),
            2.083346354199219e-08,
        ),
        ("Y3(0.01)", bessel_second_kind(3, 0.01), -5093021.8417137368),
        (
            "K3(0.01)",
            modified_bessel_second_kind(3, 0.01),
            7999900.0012498824,
        ),
        (
            "J0(-2.5)",
            bessel_first_kind(0, -2.5),
            -0.048383776468197998,
        ),
        (
            "I0(-2.5)",
            modified_bessel_first_kind(0, -2.5),
            3.2898391440501231,
        ),
        ("J4(60)", bessel_first_kind(4, 60.0), -0.097064754699832931),
        (
            "I4(60)",
            modified_bessel_first_kind(4, 60.0),
            5.1527718571754953e+24,
        ),
        ("Y4(60)", bessel_second_kind(4, 60.0), 0.03481436008081968),
        (
            "K4(60)",
            modified_bessel_second_kind(4, 60.0),
            1.6137249034821196e-27,
        ),
        (
            "log I0(0.5)",
            log_modified_bessel_first_kind(0.0, 0.5),
            0.061549719185481327,
        ),
        (
            "log I1.5(3.7)",
            log_modified_bessel_first_kind(1.5, 3.7),
            1.8128774772330947,
        ),
        (
            "log I2.3(50)",
            log_modified_bessel_first_kind(2.3, 50.0),
            47.074144568602435,
        ),
        (
            "log I0.5(200)",
            log_modified_bessel_first_kind(0.5, 200.0),
            196.43190278351997,
        ),
        (
            "log I7(0.2)",
            log_modified_bessel_first_kind(7.0, 0.2),
            -24.642007098814823,
        ),
        (
            "log I0.25(1e-3)",
            log_modified_bessel_first_kind(0.25, 1e-3),
            -1.8019535784637164,
        ),
    ]);
}

#[test]
fn bessel_derivatives_match_stan() {
    let lv = grad(|v| log_modified_bessel_first_kind(v[0], v[1]), &[1.5, 3.7]);
    report(&[
        (
            "J1'(3.7)",
            grad(|v| bessel_first_kind(1, v[0]), &[3.7])[0],
            -0.4137799297888835,
        ),
        (
            "Y2'(5.5)",
            grad(|v| bessel_second_kind(2, v[0]), &[5.5])[0],
            -0.14406414196053713,
        ),
        ("Y2(5.5)", bessel_second_kind(2, 5.5), 0.33084123326140574),
        (
            "I1'(3.7)",
            grad(|v| modified_bessel_first_kind(1, v[0]), &[3.7])[0],
            6.7289564980787659,
        ),
        (
            "K2'(1.5)",
            grad(|v| modified_bessel_second_kind(2, v[0]), &[1.5])[0],
            -1.0555957514657115,
        ),
        (
            "K2(1.5)",
            modified_bessel_second_kind(2, 1.5),
            0.58365596325665081,
        ),
        ("d log I / dv", lv[0], -0.45652864085921158),
        ("d log I / dz", lv[1], 0.96267164373434067),
    ]);
}

#[test]
fn owens_t_matches_stan() {
    let g = grad(|v| owens_t(v[0], v[1]), &[1.2, 2.0]);
    report(&[
        ("T(0.5, 0.3)", owens_t(0.5, 0.3), 0.040786707344250106),
        ("T(1.2, 2)", owens_t(1.2, 2.0), 0.057315639682100065),
        ("T(-0.7, 0.9)", owens_t(-0.7, 0.9), 0.086456260898305992),
        ("T(3, 0.5)", owens_t(3.0, 0.5), 0.0006051213785851949),
        ("T(0.1, 10)", owens_t(0.1, 10.0), 0.22679969643349898),
        ("T(6, 0.99)", owens_t(6.0, 0.99), 4.9329382180895675e-10),
        ("T(0, 1)", owens_t(0.0, 1.0), 0.125),
        ("dT/dh", g[0], -0.095501180329825991),
        ("dT/da", g[1], 0.00086974109823151845),
    ]);
}

#[test]
fn lambert_w_matches_stan() {
    report(&[
        ("W0(0.5)", lambert_w0(0.5), 0.35173371124919584),
        ("W0(5)", lambert_w0(5.0), 1.3267246652422002),
        ("W0(-0.3)", lambert_w0(-0.3), -0.48940222718021481),
        ("W0(100)", lambert_w0(100.0), 3.3856301402900506),
        ("W0(-0.36)", lambert_w0(-0.36), -0.80608431597081742),
        ("W0(1e-8)", lambert_w0(1e-8), 9.9999999000000018e-09),
        ("W-1(-0.3)", lambert_wm1(-0.3), -1.7813370234216279),
        ("W-1(-0.01)", lambert_wm1(-0.01), -6.4727751243940048),
        ("W-1(-0.36)", lambert_wm1(-0.36), -1.2227701339785064),
        ("W-1(-1e-5)", lambert_wm1(-1e-5), -14.163600815810183),
        (
            "W0'(0.5)",
            grad(|v| lambert_w0(v[0]), &[0.5])[0],
            0.52041864210687394,
        ),
        (
            "W-1'(-0.3)",
            grad(|v| lambert_wm1(v[0]), &[-0.3])[0],
            -7.5995247890204229,
        ),
    ]);
}

#[test]
fn hypergeometric_functions_match_stan() {
    report(&[
        (
            "2F1(1, 2; 3.5; 0.4)",
            hypergeometric_2F1(1.0, 2.0, 3.5, 0.4),
            1.3153281771695777,
        ),
        (
            "2F1(0.5, 1.5; 2; -0.6)",
            hypergeometric_2F1(0.5, 1.5, 2.0, -0.6),
            0.83417182185325989,
        ),
        (
            "2F1(1.2, -0.7; 2.1; 0.9)",
            hypergeometric_2F1(1.2, -0.7, 2.1, 0.9),
            0.58453949006671879,
        ),
        (
            "d 2F1 / dz",
            grad(
                |v| {
                    let c = Real::from_f64;
                    hypergeometric_2F1(c(1.0), c(2.0), c(3.5), v[0])
                },
                &[0.4],
            )[0],
            1.0997587450488244,
        ),
        (
            "3F2(1, 1.5, 2; 2.5, 3; 0.3)",
            hypergeometric_3F2([1.0, 1.5, 2.0], [2.5, 3.0], 0.3),
            1.1438300798086294,
        ),
        (
            "2F2(0.5, 1.2; 1.7, 2.4; 0.8)",
            hypergeometric_pFq(&[0.5, 1.2], &[1.7, 2.4], 0.8),
            1.1370661527903918,
        ),
        (
            "2F2(0.5, 1.2; 1.7, 2.4; -3.5)",
            hypergeometric_pFq(&[0.5, 1.2], &[1.7, 2.4], -3.5),
            0.68765146983404712,
        ),
    ]);
}
