//! Differentiable numerical solvers: scalar root finding, systems of nonlinear equations, and
//! one-dimensional integration.
//!
//! # Derivatives of solutions
//!
//! A root `x(θ)` of `f(x, θ) = 0` is found in `f64`. Its derivatives with respect to `θ` then come
//! from Newton steps taken in the caller's scalar type, starting from the `f64` root:
//! `x ← x - f(x, θ) / f_x(x, θ)`. At an exact root one step gives the implicit-function-theorem
//! derivative `dx/dθ = -f_θ / f_x`, and each further step doubles the order of derivatives that are
//! exact (Newton's quadratic convergence carried over to Taylor coefficients). The solvers take the
//! smallest number of steps that covers [`Real::ORDER`], so gradients, Hessians, and higher
//! derivatives of solutions are exact up to the accuracy of the root itself. For systems the same
//! scheme uses the Jacobian.
//!
//! Equations and integrands are written once, generically over [`Real`], by implementing
//! [`Equation`], [`EquationSystem`], or [`Integrand`].
//!
//! ```
//! use probgrad_autodiff::{grad, Real};
//! use probgrad_solvers::{Equation, RootOptions, solve_root};
//!
//! /// x^3 + x - theta = 0
//! struct Cubic;
//! impl Equation for Cubic {
//!     fn eval<S: Real>(&self, x: S, theta: &[S]) -> S {
//!         x.powi(3) + x - theta[0]
//!     }
//! }
//!
//! let opts = RootOptions::bracket(-10.0, 10.0);
//! let g = grad(|t| solve_root(&Cubic, t, &opts).unwrap(), &[2.0]);
//! // The root is x = 1, and dx/dtheta = 1 / (3 x^2 + 1) = 1/4.
//! assert!((g[0] - 0.25).abs() < 1e-14);
//! ```

mod quadrature;
mod root;
mod system;

pub use quadrature::{Integrand, QuadratureOptions, integrate};
pub use root::{Equation, RootOptions, solve_root};
pub use system::{EquationSystem, SystemOptions, solve_system};

#[doc(no_inline)]
pub use probgrad_autodiff::Real;

/// Newton steps needed in a type of derivative order `order`: the smallest `k` with
/// `2^k - 1 >= order`.
fn newton_steps(order: usize) -> usize {
    let mut k = 0;
    while (1usize << k) - 1 < order {
        k += 1;
    }
    k
}

#[cfg(test)]
mod tests {
    #[test]
    fn newton_step_counts() {
        assert_eq!(super::newton_steps(0), 0);
        assert_eq!(super::newton_steps(1), 1);
        assert_eq!(super::newton_steps(2), 2);
        assert_eq!(super::newton_steps(3), 2);
        assert_eq!(super::newton_steps(4), 3);
    }
}
