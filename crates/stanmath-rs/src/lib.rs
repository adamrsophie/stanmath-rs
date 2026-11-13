//! stanmath-rs: differentiable mathematics for Bayesian computation in Rust.
//!
//! This umbrella crate re-exports the workspace crates so that a single dependency and a single
//! import cover the common cases:
//!
//! ```
//! use stanmath_rs::prelude::*;
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
//! | [`autodiff`] | `stanmath-rs-autodiff` | scalar types, tape, drivers |
//! | [`distributions`] | `stanmath-rs-distributions` | log densities, CDFs, random number generation |
//! | [`linalg`] | `stanmath-rs-linalg` | matrix products, Cholesky, triangular solves, log determinants |
//! | [`transforms`] | `stanmath-rs-transforms` | constraining transforms with log Jacobians |
//! | [`solvers`] | `stanmath-rs-solvers` | root finding, nonlinear systems, quadrature, cubature, ODEs, DAEs, Laplace approximation |
//! | [`special`], [`Matrix`], [`Error`] | `stanmath-rs-core` | `f64` special functions, matrix container, errors |

pub use stanmath_rs_autodiff as autodiff;
pub use stanmath_rs_distributions as distributions;
pub use stanmath_rs_linalg as linalg;
pub use stanmath_rs_solvers as solvers;
pub use stanmath_rs_transforms as transforms;

pub use stanmath_rs_autodiff::functions;
pub use stanmath_rs_core::special;
pub use stanmath_rs_core::{Error, Matrix, Result};

/// The common imports.
pub mod prelude {
    pub use stanmath_rs_autodiff::functions::*;
    pub use stanmath_rs_autodiff::{
        Complex, Dual, Dual64, GradHessian, GradientWorkspace, Real, Tape, ValueAndGrad, Var,
        check_gradient, derivative, finite_diff_grad_hessian, finite_diff_gradient_auto,
        finite_diff_hessian, finite_diff_hessian_auto, grad, grad_hessian,
        grad_tr_mat_times_hessian, gradient_dot_vector, hessian, hessian_times_vector,
        hessian_vector_product, jacobian, jvp, partial_derivative, try_grad, try_value_and_grad,
        value_and_grad, vjp,
    };
    pub use stanmath_rs_core::{Error, Matrix, Result};
    pub use stanmath_rs_distributions::*;
    pub use stanmath_rs_linalg::*;
    pub use stanmath_rs_solvers::*;
    pub use stanmath_rs_transforms::*;
}

/// The README's code blocks, compiled and run as documentation tests.
#[cfg(doctest)]
#[doc = include_str!("../../../README.md")]
pub struct ReadmeDoctests;
