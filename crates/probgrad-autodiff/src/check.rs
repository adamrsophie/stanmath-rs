//! Finite-difference gradient checking.
//!
//! Finite differences are a sanity check, not a reference: they lose about half the significant
//! digits to cancellation and fail near kinks and boundaries. The test suites therefore compare
//! derivatives against closed forms and high-precision references first, and use these helpers
//! as an additional net.

use crate::drivers::value_and_grad;
use crate::real::Real;
use crate::tape::Var;

/// Gradient by sixth-order central differences.
///
/// The step for coordinate `i` is `h * max(1, |x[i]|)`; `h = 1e-3` balances truncation error
/// (order `h^6`) against rounding error (order `eps / h`).
pub fn finite_diff_gradient(f: impl Fn(&[f64]) -> f64, x: &[f64], h: f64) -> Vec<f64> {
    const W: [(f64, f64); 6] = [
        (-3.0, -1.0 / 60.0),
        (-2.0, 9.0 / 60.0),
        (-1.0, -45.0 / 60.0),
        (1.0, 45.0 / 60.0),
        (2.0, -9.0 / 60.0),
        (3.0, 1.0 / 60.0),
    ];
    let mut xp = x.to_vec();
    (0..x.len())
        .map(|i| {
            let step = h * x[i].abs().max(1.0);
            let mut acc = 0.0;
            for &(k, w) in &W {
                xp[i] = x[i] + k * step;
                acc += w * f(&xp);
            }
            xp[i] = x[i];
            acc / step
        })
        .collect()
}

/// Outcome of comparing an automatic gradient against finite differences.
#[derive(Clone, Debug)]
pub struct GradientCheck {
    /// Function value.
    pub value: f64,
    /// Reverse-mode gradient.
    pub autodiff: Vec<f64>,
    /// Finite-difference gradient.
    pub finite_diff: Vec<f64>,
}

impl GradientCheck {
    /// Largest absolute difference between the two gradients.
    pub fn max_abs_error(&self) -> f64 {
        self.autodiff
            .iter()
            .zip(&self.finite_diff)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max)
    }

    /// True when every component satisfies `|ad - fd| <= atol + rtol * |fd|`.
    pub fn passes(&self, rtol: f64, atol: f64) -> bool {
        self.autodiff
            .iter()
            .zip(&self.finite_diff)
            .all(|(a, b)| (a - b).abs() <= atol + rtol * b.abs())
    }
}

/// Compares the reverse-mode gradient of `f` at `x` with finite differences.
///
/// The same closure is used for both: evaluating it on constant variables (no tape) gives plain
/// `f64` values for the finite differences.
///
/// ```
/// use probgrad_autodiff::{check_gradient, Real};
///
/// let c = check_gradient(|x| (x[0] * x[1]).sin() + x[0].exp(), &[0.3, -1.2]);
/// assert!(c.passes(1e-8, 1e-10));
/// ```
pub fn check_gradient<F>(f: F, x: &[f64]) -> GradientCheck
where
    F: for<'t> Fn(&[Var<'t>]) -> Var<'t>,
{
    let r = value_and_grad(&f, x);
    let fd = finite_diff_gradient(
        |p| {
            let c: Vec<Var<'_>> = p.iter().map(|&v| Var::constant(v)).collect();
            f(&c).value()
        },
        x,
        1e-3,
    );
    GradientCheck {
        value: r.value,
        autodiff: r.gradient,
        finite_diff: fd,
    }
}
