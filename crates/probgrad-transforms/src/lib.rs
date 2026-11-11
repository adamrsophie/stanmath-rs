//! Constraining transforms for Bayesian inference, following Stan Math 5.4.
//!
//! Each transform maps unconstrained values `y` to constrained values `x` and has three forms:
//!
//! * `*_constrain(y, ...)` returns `x`;
//! * `*_constrain_lp(y, ..., &mut lp)` returns `x` and adds `ln |det J|` to `lp`, the log
//!   absolute Jacobian determinant needed to turn a density on `x` into one on `y`;
//! * `*_free(x, ...)` inverts the transform after checking that `x` is in the constrained set.
//!
//! All are generic over [`Real`], so a log density written with them can be differentiated in any
//! mode. The Jacobian terms are verified in the tests by comparing them with the log determinant of
//! the transform's Jacobian computed by forward-mode differentiation.
//!
//! ```
//! use probgrad_autodiff::grad;
//! use probgrad_transforms::lb_constrain_lp;
//!
//! // A positive parameter sigma = exp(y) with its Jacobian, inside a log density.
//! let g = grad(
//!     |y| {
//!         let mut lp = y[0] * 0.0;
//!         let sigma = lb_constrain_lp(y[0], 0.0, &mut lp);
//!         lp - sigma // exponential(1) density on sigma
//!     },
//!     &[0.3],
//! );
//! assert!((g[0] - (1.0 - 0.3f64.exp())).abs() < 1e-15);
//! ```
//!
//! # Conventions for vector-valued constraints
//!
//! The sum-to-zero transform of Stan Math 5.4 is an isometry onto the hyperplane, so it adds no
//! Jacobian term: densities on the sum-to-zero vector are relative to the hyperplane's surface
//! measure, which is `sqrt(N + 1)` times Lebesgue measure on the first `N` coordinates. The
//! simplex transform's term is relative to Lebesgue measure on the first `N` coordinates, the
//! measure in which the Dirichlet density is written (the tests verify both). `unit_vector_constrain_lp`
//! adds `-|y|^2 / 2`, which is not a Jacobian but makes the induced distribution on the sphere
//! uniform, as in Stan.

mod matrix;
mod scalar;
mod vector;

pub use matrix::*;
pub use scalar::*;
pub use vector::*;

#[doc(no_inline)]
pub use probgrad_autodiff::Real;
