//! Stan language names for the constraining transforms.
//!
//! Stan 2.37 added `X_constrain`, `X_jacobian`, and `X_unconstrain` to the language. Here they are
//! aliases of the Stan Math names used elsewhere in this crate: `X_jacobian` is `X_constrain_lp`
//! (constrain and add the log Jacobian to `lp`), and `X_unconstrain` is `X_free`. Stan overloads
//! one name over vectors and matrices; the matrix form of `sum_to_zero` is `sum_to_zero_matrix_*`.

use stanmath_rs_autodiff::Real;

pub use crate::matrix::{
    cholesky_factor_corr_constrain_lp as cholesky_factor_corr_jacobian,
    cholesky_factor_corr_free as cholesky_factor_corr_unconstrain,
    cholesky_factor_cov_constrain_lp as cholesky_factor_cov_jacobian,
    cholesky_factor_cov_free as cholesky_factor_cov_unconstrain,
    corr_matrix_constrain_lp as corr_matrix_jacobian, corr_matrix_free as corr_matrix_unconstrain,
    cov_matrix_constrain_lp as cov_matrix_jacobian, cov_matrix_free as cov_matrix_unconstrain,
    stochastic_column_constrain_lp as stochastic_column_jacobian,
    stochastic_column_free as stochastic_column_unconstrain,
    stochastic_row_constrain_lp as stochastic_row_jacobian,
    stochastic_row_free as stochastic_row_unconstrain,
    sum_to_zero_matrix_constrain_lp as sum_to_zero_matrix_jacobian,
    sum_to_zero_matrix_free as sum_to_zero_matrix_unconstrain,
};
pub use crate::scalar::{
    lb_constrain as lower_bound_constrain, lb_constrain_lp as lower_bound_jacobian,
    lb_free as lower_bound_unconstrain, lub_constrain as lower_upper_bound_constrain,
    lub_constrain_lp as lower_upper_bound_jacobian, lub_free as lower_upper_bound_unconstrain,
    offset_multiplier_constrain_lp as offset_multiplier_jacobian,
    offset_multiplier_free as offset_multiplier_unconstrain, ub_constrain as upper_bound_constrain,
    ub_constrain_lp as upper_bound_jacobian, ub_free as upper_bound_unconstrain,
};
pub use crate::vector::{
    ordered_constrain_lp as ordered_jacobian, ordered_free as ordered_unconstrain,
    positive_ordered_constrain_lp as positive_ordered_jacobian,
    positive_ordered_free as positive_ordered_unconstrain,
    simplex_constrain_lp as simplex_jacobian, simplex_free as simplex_unconstrain,
    unit_vector_constrain_lp as unit_vector_jacobian, unit_vector_free as unit_vector_unconstrain,
};

use crate::vector::sum_to_zero_constrain;
pub use crate::vector::sum_to_zero_free as sum_to_zero_unconstrain;

/// `sum_to_zero_constrain` under Stan's `_jacobian` name. The transform is an isometry, so it adds
/// nothing to `lp`; the argument is kept so all `_jacobian` functions share one shape.
pub fn sum_to_zero_jacobian<T: Real>(y: &[T], _lp: &mut T) -> Vec<T> {
    sum_to_zero_constrain(y)
}
