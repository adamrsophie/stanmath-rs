//! Differentiable numerical solvers: scalar root finding, systems of nonlinear equations,
//! one-dimensional integration, ordinary differential equations, and differential algebraic
//! equations.
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
//! use stanmath_rs_autodiff::{grad, Real};
//! use stanmath_rs_solvers::{Equation, RootOptions, solve_root};
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

mod adjoint;
mod deprecated;
mod fixed_point;
mod functors;
mod hcubature;
mod laplace;
#[cfg(feature = "rand")]
mod laplace_rng;
mod limits;
mod ode;
mod ode_times;
mod parallel;
mod powell;
mod quadrature;
mod quadrature_wave2;
mod root;
mod system;

pub use adjoint::{AdjointOptions, ode_adjoint_tol_ctl};
pub use deprecated::{AlgebraSystemData, algebra_solver, algebra_solver_newton, integrate_ode};
pub use fixed_point::{FixedPointOptions, solve_fixed_point, solve_fixed_point_tol};
pub use functors::{
    OdeSystemData, integrate_ode_adams, integrate_ode_bdf, integrate_ode_rk45, map_rect,
    reduce_sum, reduce_sum_static,
};
pub use hcubature::{CubatureIntegrand, hcubature};
pub use laplace::{
    LaplaceCovariance, LaplaceLikelihood, LaplaceOptions, generate_laplace_options,
    generate_laplace_options_from, laplace_latent_solve, laplace_latent_solve_tol,
    laplace_marginal, laplace_marginal_bernoulli_logit_lpmf,
    laplace_marginal_bernoulli_logit_lupmf, laplace_marginal_neg_binomial_2_log_lpmf,
    laplace_marginal_neg_binomial_2_log_lupmf, laplace_marginal_poisson_log_lpmf,
    laplace_marginal_poisson_log_lupmf, laplace_marginal_tol,
    laplace_marginal_tol_bernoulli_logit_lpmf, laplace_marginal_tol_bernoulli_logit_lupmf,
    laplace_marginal_tol_neg_binomial_2_log_lpmf, laplace_marginal_tol_neg_binomial_2_log_lupmf,
    laplace_marginal_tol_poisson_log_lpmf, laplace_marginal_tol_poisson_log_lupmf,
};
#[cfg(feature = "rand")]
pub use laplace_rng::{
    laplace_latent_bernoulli_logit_rng, laplace_latent_neg_binomial_2_log_rng,
    laplace_latent_poisson_log_rng, laplace_latent_rng, laplace_latent_tol_bernoulli_logit_rng,
    laplace_latent_tol_neg_binomial_2_log_rng, laplace_latent_tol_poisson_log_rng,
    laplace_latent_tol_rng,
};
pub use limits::integrate_1d;
pub use ode::{
    DaeSystem, OdeOptions, OdeSystem, dae, dae_tol, ode_adams, ode_adams_tol, ode_bdf, ode_bdf_tol,
    ode_ckrk, ode_ckrk_tol, ode_rk45, ode_rk45_tol,
};
pub use ode_times::{OdeMethod, ode_tol_times};
pub use parallel::{MapRectJob, PartialSum, map_rect_parallel, reduce_sum_parallel};
pub use powell::{solve_newton, solve_newton_tol, solve_powell, solve_powell_tol};
pub use quadrature::{Integrand, QuadratureOptions, integrate};
pub use quadrature_wave2::{
    integrate_1d_double_exponential, integrate_1d_double_exponential_tol,
    integrate_1d_gauss_kronrod, integrate_1d_gauss_kronrod_tol,
};
pub use root::{Equation, RootOptions, solve_root};
pub use system::{EquationSystem, SystemOptions, solve_system};

#[doc(no_inline)]
pub use stanmath_rs_autodiff::Real;

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
