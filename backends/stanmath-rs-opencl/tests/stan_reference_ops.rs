//! Every device elementwise function and density against values and gradients that Stan Math
//! computed, from the scalar and distribution reference rows of the main crates. Each operation
//! runs on every row with its name and arity; `fabs` uses the `abs` rows. Where Stan Math is less
//! accurate than we are (`docs/compatibility.md`, "Measured differences from Stan Math"), the row
//! is decided by the mpmath value instead, as in the main crates' comparisons.

use stanmath_rs_autodiff::{Var, try_value_and_grad};
use stanmath_rs_core::Result;
use stanmath_rs_opencl::{
    Binary, Density, Unary, binary_on_device, lpmf_on_device, unary_on_device,
};

const DATA: [&str; 4] = [
    include_str!("../../../crates/stanmath-rs/tests/data/stan_reference_scalar.csv"),
    include_str!("../../../crates/stanmath-rs/tests/data/stan_reference_scalar_extra.csv"),
    include_str!("../../../crates/stanmath-rs-distributions/tests/data/stan_reference.csv"),
    include_str!("../../../crates/stanmath-rs-distributions/tests/data/stan_reference_extra.csv"),
];

/// High precision values for the rows where Stan Math and stanmath-rs disagree.
const EXACT: [&str; 3] = [
    include_str!("../../../crates/stanmath-rs/tests/data/mpmath_reference_scalar.csv"),
    include_str!("../../../crates/stanmath-rs-distributions/tests/data/mpmath_reference.csv"),
    include_str!("../../../crates/stanmath-rs-distributions/tests/data/mpmath_reference_extra.csv"),
];

/// `(key, ints, reals, value, gradient)` of one reference row; the key is its first three fields.
type Row = (String, Vec<i64>, Vec<f64>, f64, Vec<f64>);

/// The mpmath `(value, gradient)` for the row with `key`.
fn exact(key: &str) -> Option<(f64, Vec<f64>)> {
    EXACT.iter().flat_map(|t| t.lines()).find_map(|l| {
        let f: Vec<&str> = l.split(',').collect();
        if f.len() < 5 || f[..3].join(",") != key {
            return None;
        }
        let g = if f[4].is_empty() {
            vec![]
        } else {
            f[4].split(';')
                .map(|v| v.parse().ok())
                .collect::<Option<_>>()?
        };
        Some((f[3].parse().ok()?, g))
    })
}

/// The rows named `name`.
fn rows(name: &str) -> Vec<Row> {
    let list = |s: &str| -> Option<Vec<f64>> {
        if s.is_empty() {
            return Some(vec![]);
        }
        s.split(';').map(|v| v.parse().ok()).collect()
    };
    DATA.iter()
        .flat_map(|t| t.lines())
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            if f.len() != 5 || f[0] != name {
                return None;
            }
            let ints = if f[1].is_empty() {
                vec![]
            } else {
                f[1].split(';')
                    .map(|v| v.parse().ok())
                    .collect::<Option<_>>()?
            };
            Some((
                f[..3].join(","),
                ints,
                list(f[2])?,
                f[3].parse().ok()?,
                list(f[4])?,
            ))
        })
        .collect()
}

fn close(a: f64, b: f64, rtol: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= rtol * b.abs().max(1.0)
}

fn agrees(value: f64, grad: &[f64], want: f64, want_grad: &[f64], rtol: f64) -> bool {
    close(value, want, rtol)
        && grad.len() == want_grad.len()
        && grad.iter().zip(want_grad).all(|(a, b)| close(*a, *b, rtol))
}

/// Compares `f` on every row of `name` with `arity` reals; returns the number of rows compared.
fn compare(
    name: &str,
    ints: usize,
    arity: usize,
    f: impl for<'t> Fn(&[i64], &[Var<'t>]) -> Result<Var<'t>>,
    failures: &mut Vec<String>,
) -> usize {
    let mut n = 0;
    for (key, i, x, want, grad) in rows(name) {
        if i.len() != ints || x.len() != arity || grad.len() != arity {
            continue;
        }
        n += 1;
        match try_value_and_grad(|v| f(&i, v), &x) {
            Err(e) => failures.push(format!("{name} {i:?} {x:?}: we reject: {e}")),
            Ok(r) => {
                if agrees(r.value, &r.gradient, want, &grad, 1e-9) {
                    continue;
                }
                match exact(&key) {
                    Some((v, g)) if agrees(r.value, &r.gradient, v, &g, 1e-12) => {}
                    _ => failures.push(format!(
                        "{name} {i:?} {x:?}: ours {} {:?}, Stan {want} {grad:?}",
                        r.value, r.gradient
                    )),
                }
            }
        }
    }
    n
}

#[test]
fn every_device_operation_matches_stan() {
    let mut failures = Vec::new();
    let mut unmatched = Vec::new();
    for &op in Unary::ALL {
        let name = if op.name() == "fabs" {
            "abs"
        } else {
            op.name()
        };
        let n = compare(
            name,
            0,
            1,
            |_, v| Ok(unary_on_device(op, v)?[0]),
            &mut failures,
        );
        if n == 0 {
            unmatched.push(op.name());
        }
    }
    for &op in Binary::ALL {
        let n = compare(
            op.name(),
            0,
            2,
            |_, v| Ok(binary_on_device(op, &v[..1], &v[1..])?[0]),
            &mut failures,
        );
        if n == 0 {
            unmatched.push(op.name());
        }
    }
    for &op in Density::ALL {
        let n = compare(
            op.name(),
            op.int_args().len(),
            op.args().len(),
            |i, v| {
                let ints: Vec<&[i64]> = i.iter().map(std::slice::from_ref).collect();
                let args: Vec<&[Var<'_>]> = v.iter().map(std::slice::from_ref).collect();
                lpmf_on_device(op, &ints, &args)
            },
            &mut failures,
        );
        if n == 0 {
            unmatched.push(op.name());
        }
    }
    assert!(unmatched.is_empty(), "no Stan rows for {unmatched:?}");
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
