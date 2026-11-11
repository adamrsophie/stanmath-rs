//! Forward-mode and reverse-mode automatic differentiation for probgrad.
//!
//! # Scalar types
//!
//! | Type | Mode | Use |
//! | --- | --- | --- |
//! | `f64` | none | plain evaluation |
//! | [`Dual<f64>`](Dual) | forward | derivatives, directional derivatives, Jacobian vector products |
//! | [`Var`] | reverse | gradients of scalar functions, vector Jacobian products |
//! | `Dual<Var>` | forward over reverse | Hessians, Hessian vector products |
//! | `Dual<Dual<f64>>` | forward over forward | second derivatives, Hessian cross-checks |
//!
//! All of them implement [`Real`]. Write a model once as a generic function and evaluate it in
//! whichever mode you need.
//!
//! # Quick start
//!
//! ```
//! use probgrad_autodiff::{value_and_grad, Real};
//!
//! // Normal log density of y = 1.3 as a function of (mu, sigma).
//! fn log_density<T: Real>(p: &[T]) -> T {
//!     let (mu, sigma) = (p[0], p[1]);
//!     let z = (-mu + 1.3) / sigma;
//!     -(z * z) * 0.5 - sigma.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
//! }
//!
//! let r = value_and_grad(|p| log_density(p), &[0.5, 1.2]);
//! assert!((r.gradient[0] - 0.8 / 1.44).abs() < 1e-15);
//! ```
//!
//! # Extending
//!
//! New differentiable functions implement [`UnaryFn`], [`BinaryFn`], or [`NaryFn`] and are applied
//! with [`Real::map_unary`] and friends. Operations with structured adjoints (matrix algebra) use
//! [`Tape::custom_op`].

mod check;
mod drivers;
mod dual;
pub mod functions;
mod incomplete;
pub mod primitives;
mod real;
pub mod special;
mod tape;
mod vectorize;

pub use check::{GradientCheck, check_gradient, finite_diff_gradient};
pub use drivers::{
    GradientWorkspace, ValueAndGrad, ValueGradHessian, derivative, directional_derivative, grad,
    hessian, hessian_forward, hessian_vector_product, jacobian, jacobian_forward, jvp,
    second_derivative, try_grad, try_value_and_grad, value_and_derivative, value_and_grad,
    value_grad_hessian, vjp,
};
pub use dual::{Dual, Dual64};
pub use real::{BinaryFn, MultiFn, NaryFn, Real, UnaryFn};
pub use tape::{Gradients, Tape, TapeStats, Var};
pub use vectorize::{Arg, Broadcast, IntArg, Lift, Promote, Promoted, Promoted2, broadcast_len};
