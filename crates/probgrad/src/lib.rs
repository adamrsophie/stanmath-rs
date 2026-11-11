//! probgrad: differentiable mathematics for Bayesian computation in Rust.
//!
//! This umbrella crate re-exports the workspace crates so that a single dependency and a single
//! import cover the common cases:
//!
//! ```
//! use probgrad::prelude::*;
//!
//! fn log_density<T: Real>(p: &[T]) -> T {
//!     let (mu, sigma) = (p[0], p[1]);
//!     let z = (-mu + 1.3) / sigma;
//!     -(z * z) * 0.5 - log(sigma)
//! }
//!
//! let r = value_and_grad(|p| log_density(p), &[0.5, 1.2]);
//! println!("value {} gradient {:?}", r.value, r.gradient);
//! ```
//!
//! | Module | Crate | Contents |
//! | --- | --- | --- |
//! | [`autodiff`] | `probgrad-autodiff` | scalar types, tape, drivers |
//! | [`distributions`] | `probgrad-distributions` | log densities, CDFs, random number generation |
//! | [`linalg`] | `probgrad-linalg` | matrix products, Cholesky, triangular solves, log determinants |
//! | [`transforms`] | `probgrad-transforms` | constraining transforms with log Jacobians |
//! | [`solvers`] | `probgrad-solvers` | root finding, nonlinear systems, quadrature, with exact derivatives |
//! | [`special`], [`Matrix`], [`Error`] | `probgrad-core` | `f64` special functions, matrix container, errors |

pub use probgrad_autodiff as autodiff;
pub use probgrad_distributions as distributions;
pub use probgrad_linalg as linalg;
pub use probgrad_solvers as solvers;
pub use probgrad_transforms as transforms;

pub use probgrad_autodiff::functions;
pub use probgrad_core::special;
pub use probgrad_core::{Error, Matrix, Result};

/// The common imports.
pub mod prelude {
    pub use probgrad_autodiff::functions::*;
    pub use probgrad_autodiff::{
        Dual, Dual64, GradientWorkspace, Real, Tape, ValueAndGrad, Var, check_gradient, derivative,
        grad, hessian, hessian_vector_product, jacobian, jvp, try_grad, try_value_and_grad,
        value_and_grad, vjp,
    };
    pub use probgrad_core::{Error, Matrix, Result};
    pub use probgrad_distributions::*;
    pub use probgrad_linalg::*;
    pub use probgrad_solvers::{
        Equation, EquationSystem, Integrand, QuadratureOptions, RootOptions, SystemOptions,
        integrate, solve_root, solve_system,
    };
    pub use probgrad_transforms::*;
}

/// The README's code blocks, compiled and run as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct ReadmeDoctests;
