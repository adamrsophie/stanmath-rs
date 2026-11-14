//! The device functions against values and gradients that Stan Math computed, from the matrix
//! reference rows of the main crate (`crates/stanmath-rs/tests/data/stan_reference_matrix.csv`,
//! written by `tools/stan-reference/stan_reference_matrix.cpp`). Each result is reduced as that
//! test does, to `sum_k (1 + k / 10) r_k` over its entries in column major order.

use stanmath_rs_autodiff::{Var, value_and_grad};
use stanmath_rs_core::Matrix;
use stanmath_rs_opencl::{
    cholesky_decompose_on_device, gp_exp_quad_cov_on_device, mdivide_left_tri_low_on_device,
    mdivide_right_tri_low_on_device, multiply_on_device,
};

fn weighted<'t>(m: &Matrix<Var<'t>>) -> Var<'t> {
    m.as_slice()
        .iter()
        .enumerate()
        .map(|(k, &v)| v * (1.0 + 0.1 * k as f64))
        .sum()
}

fn mat<'t>(v: &[Var<'t>], r: usize, c: usize) -> Matrix<Var<'t>> {
    Matrix::from_column_major(r, c, v[..r * c].to_vec())
}

fn low<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    let z = Var::constant(0.0);
    Matrix::from_rows(&[[v[0], z, z], [v[1], v[2], z], [v[3], v[4], v[5]]])
}

fn sym<'t>(v: &[Var<'t>]) -> Matrix<Var<'t>> {
    Matrix::from_rows(&[[v[0], v[1], v[3]], [v[1], v[2], v[4]], [v[3], v[4], v[5]]])
}

fn eval<'t>(name: &str, x: &[Var<'t>]) -> Var<'t> {
    let m = match name {
        "cholesky_decompose" => cholesky_decompose_on_device(&sym(x)),
        "multiply" => multiply_on_device(&mat(x, 3, 3), &mat(&x[9..], 3, 2)),
        "mdivide_left_tri_low" => mdivide_left_tri_low_on_device(&low(x), &mat(&x[6..], 3, 2)),
        "mdivide_right_tri_low" => mdivide_right_tri_low_on_device(&mat(&x[6..], 2, 3), &low(x)),
        "gp_exp_quad_cov" => {
            let pts: Vec<Vec<Var<'t>>> = x[0..3].iter().map(|&p| vec![p]).collect();
            gp_exp_quad_cov_on_device(&pts, x[3], x[4])
        }
        other => panic!("no device function for {other}"),
    };
    weighted(&m.unwrap())
}

#[test]
fn device_functions_match_stan() {
    let text = include_str!("../../../crates/stanmath-rs/tests/data/stan_reference_matrix.csv");
    let names = [
        "cholesky_decompose",
        "multiply",
        "mdivide_left_tri_low",
        "mdivide_right_tri_low",
        "gp_exp_quad_cov",
    ];
    let list = |s: &str| -> Vec<f64> { s.split(';').map(|v| v.parse().unwrap()).collect() };
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * b.abs().max(1.0);
    let mut failures = Vec::new();
    let mut seen = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split(',').collect();
        if !names.contains(&f[0]) {
            continue;
        }
        seen += 1;
        let r = value_and_grad(|x| eval(f[0], x), &list(f[2]));
        let (want, grad) = (f[3].parse::<f64>().unwrap(), list(f[4]));
        if !close(r.value, want)
            || r.gradient.len() != grad.len()
            || r.gradient.iter().zip(&grad).any(|(a, b)| !close(*a, *b))
        {
            failures.push(format!(
                "{}: ours {} {:?}, Stan {want} {grad:?}",
                f[0], r.value, r.gradient
            ));
        }
    }
    assert!(seen >= names.len(), "found {seen} rows");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
