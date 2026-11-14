//! Values and gradients against Stan Math 5.4.0 for the complex scalar functions of the Stan
//! language, at points that include the branch cuts approached from both signed zeros.
//! `tests/data/stan_reference_complex.csv` is produced by
//! `tools/stan-reference/stan_reference_complex.cpp`. Each output (real and imaginary part, or the
//! one real) is differentiated with respect to the real and imaginary part of each input.
//!
//! Where Stan and stanmath-rs disagree, stanmath-rs must match
//! `tests/data/c99_reference_complex.csv` (`tools/reference/complex.py`): the C99 Annex G value,
//! which keeps the sign of a zero, and the gradient on the side of the branch cut that the input's
//! signed zero picks. Stan differs from C99 in the sign of some zero outputs, and its gradients at
//! points on a cut come from the other side of the cut than its values.

use stanmath_rs::autodiff::Complex;
use stanmath_rs::prelude::*;

// stan: exp(complex)
// stan: log(complex)
// stan: log10(complex)
// stan: sqrt(complex)
// stan: sin(complex)
// stan: cos(complex)
// stan: tan(complex)
// stan: sinh(complex)
// stan: cosh(complex)
// stan: tanh(complex)
// stan: asin(complex)
// stan: acos(complex)
// stan: atan(complex)
// stan: asinh(complex)
// stan: acosh(complex)
// stan: atanh(complex)
// stan: conj(complex)
// stan: proj(complex)
// stan: minus(complex)
// stan: abs(complex)
// stan: arg(complex)
// stan: norm(complex)
// stan: pow(complex, complex)
// stan: divide(complex, complex)
// stan: multiply(complex, complex)
// stan: elt_divide(complex, complex)
// stan: elt_multiply(complex, complex)

/// The outputs of `name` at the inputs (real and imaginary parts, in order).
fn eval<'t>(name: &str, x: &[Var<'t>]) -> Vec<Var<'t>> {
    let z = Complex::new(x[0], x[1]);
    let parts = |w: Complex<Var<'t>>| vec![w.re, w.im];
    match name {
        "exp" => parts(z.exp()),
        "log" => parts(z.log()),
        "log10" => parts(z.log10()),
        "sqrt" => parts(z.sqrt()),
        "sin" => parts(z.sin()),
        "cos" => parts(z.cos()),
        "tan" => parts(z.tan()),
        "sinh" => parts(z.sinh()),
        "cosh" => parts(z.cosh()),
        "tanh" => parts(z.tanh()),
        "asin" => parts(z.asin()),
        "acos" => parts(z.acos()),
        "atan" => parts(z.atan()),
        "asinh" => parts(z.asinh()),
        "acosh" => parts(z.acosh()),
        "atanh" => parts(z.atanh()),
        "conj" => parts(z.conj()),
        "proj" => parts(z.proj()),
        "minus" => parts(-z),
        "abs" => vec![z.abs()],
        "arg" => vec![z.arg()],
        "norm" => vec![z.norm()],
        "pow" => parts(z.pow(Complex::new(x[2], x[3]))),
        "divide" => parts(z / Complex::new(x[2], x[3])),
        "multiply" => parts(z * Complex::new(x[2], x[3])),
        _ => panic!("no complex function {name}"),
    }
}

fn nums(s: &str) -> Vec<f64> {
    s.split(';').map(|v| v.parse().unwrap()).collect()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= tol * b.abs().max(1.0)
}

/// Whether `value` and `gradient` match output `k` of a reference row. A zero output keeps its
/// sign: it says which side of a branch cut the point is on.
fn matches(
    value: f64,
    gradient: &[f64],
    want: &[f64],
    want_grad: &[f64],
    k: usize,
    tol: f64,
) -> bool {
    let w = want[k];
    let sign_ok = w != 0.0 || value.is_sign_negative() == w.is_sign_negative();
    let n = gradient.len();
    let g = &want_grad[k * n..(k + 1) * n];
    close(value, w, 1e-13) && sign_ok && gradient.iter().zip(g).all(|(a, b)| close(*a, *b, tol))
}

#[test]
fn complex_functions_match_stan() {
    let data = include_str!("data/stan_reference_complex.csv");
    let c99: Vec<&str> = include_str!("data/c99_reference_complex.csv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect();
    let mut failures = Vec::new();
    let (mut rows, mut adjudicated) = (0, 0);
    for (line, c_line) in data.lines().filter(|l| !l.starts_with('#')).zip(&c99) {
        let f: Vec<&str> = line.split(',').collect();
        let c: Vec<&str> = c_line.split(',').collect();
        assert_eq!(
            f[..2],
            c[..2],
            "the two reference files list the same points"
        );
        let (name, inputs, want, want_grad) = (f[0], nums(f[1]), nums(f[2]), nums(f[3]));
        let (c_want, c_grad) = (nums(c[2]), nums(c[3]));
        rows += 1;
        for k in 0..want.len() {
            let ours = value_and_grad(|x| eval(name, x)[k], &inputs);
            if matches(ours.value, &ours.gradient, &want, &want_grad, k, 1e-11) {
                continue;
            }
            if matches(ours.value, &ours.gradient, &c_want, &c_grad, k, 1e-9) {
                adjudicated += 1;
                continue;
            }
            failures.push(format!(
                "{name}{inputs:?} output {k}: {} {:?} vs Stan {want:?} {want_grad:?}, C99 {c_want:?} {c_grad:?}",
                ours.value, ours.gradient
            ));
        }
    }
    assert!(rows > 250);
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    // The outputs where Stan differs from C99 (signed zeros, and gradients on a cut).
    assert!(adjudicated <= 30, "{adjudicated} outputs differ from Stan");
}
