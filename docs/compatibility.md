# Compatibility with Stan Math

This inventory compares stanmath-rs with Stan Math 5.4.0 (shipped with CmdStan 2.40.0). It is updated
with every feature. Status values:

* **tested**: implemented and verified against independent references. For scalar functions and
  distributions these are 60 to 320 digit mpmath values and, where Stan has the function, Stan
  Math output. For linear algebra, transforms, and solvers they are closed forms, forward-mode
  Jacobians of the generic algorithm, and finite differences (see `docs/numerical-accuracy.md`);
  these have not yet been compared with Stan Math output;
* **experimental**: implemented and unit tested, but not yet cross-checked against references;
* **planned**: on the roadmap;
* **not supported**: deliberately out of scope for now.

stanmath-rs does not claim compatibility for anything not marked **tested**.

## Automatic differentiation

| Stan Math | stanmath-rs | Status |
| --- | --- | --- |
| `var` (reverse mode) | `Var<'t>` on an explicit `Tape` | tested |
| `fvar<T>` (forward mode) | `Dual<T>` | tested |
| `fvar<var>` (forward over reverse) | `Dual<Var>` | tested |
| `fvar<fvar<T>>` | `Dual<Dual<T>>` | tested |
| `gradient` | `grad`, `value_and_grad`, `try_*`, `GradientWorkspace` | tested |
| `jacobian` | `jacobian` (reverse), `jacobian_forward` | tested |
| `hessian` | `hessian` (forward over reverse), `hessian_forward` | tested |
| `hessian_times_vector` | `hessian_vector_product` | tested |
| `gradient_dot_vector` | `directional_derivative`, `jvp` | tested |
| `grad_hessian` (third order) | `Dual<Dual<Var>>` by hand; no driver | experimental |
| `finite_diff_gradient` | `finite_diff_gradient`, `check_gradient` | tested |
| nested `start_nested` / `recover_memory_nested` | separate tapes; closures cannot capture other tapes | tested |
| `operands_and_partials` | `Var::from_partials`, `NaryFn`, fused distribution kernels | tested |
| custom reverse-mode functions (`reverse_pass_callback`) | `Tape::custom_op` | tested |
| `var` over `fvar` (reverse over forward) | not provided | not supported |
| OpenCL, TBB `reduce_sum`, `map_rect` | not provided | not supported |

## Scalar functions

Every function below is tested in all five modes (value, forward, reverse, forward over reverse,
forward over forward) against mpmath.

| Group | Functions | Status |
| --- | --- | --- |
| arithmetic | `+ - * /`, `fma`, `fmax`, `fmin`, `fdim`, `abs`, `fabs` | tested |
| powers | `pow`, `powi`, `powf`, `sqrt`, `cbrt`, `square`, `inv`, `inv_sqrt`, `inv_square`, `hypot` | tested |
| exponential | `exp`, `exp2`, `expm1`, `log`, `log1p`, `log1m`, `log2`, `log10` | tested |
| trigonometric | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2` | tested |
| hyperbolic | `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh` | tested |
| logistic | `inv_logit`, `logit`, `log_inv_logit`, `log1m_inv_logit`, `log1p_exp`, `log1m_exp` | tested |
| log domain | `log_sum_exp` (2 and n arguments), `log_diff_exp`, `softmax`, `log_softmax` | tested |
| gamma family | `lgamma`, `tgamma`, `digamma`, `trigamma`, `lbeta`, `lchoose`, `lgamma1p` | tested |
| polygamma | `polygamma(n, x)` (`x > 0`, or negative non-integer by recurrence) | tested |
| error functions | `erf`, `erfc`, `erfcx` | tested |
| normal | `Phi`, `log_Phi`, `inv_Phi` | tested |
| incomplete gamma | `gamma_p`, `gamma_q` with derivatives in both arguments | tested |
| incomplete beta | `beta_inc` (`inc_beta` in Stan) with derivatives in all three arguments | tested |
| other Stan functions | `Phi_approx`, `owens_t`, `log_modified_bessel_first_kind`, Bessel functions, `lambert_w`, `binary_log_loss`, `log_falling_factorial`, `log_rising_factorial`, `lmgamma`, `hypergeometric_*` | planned |
| integer functions | `choose`, `factorial`, `bit operations` | not supported |

## Distributions

Log densities (`_lpdf` / `_lpmf`), log CDFs (`_lcdf`), log complementary CDFs (`_lccdf`), and
random number generators (`_rng`) with Stan's parameterizations and argument checks. All
arguments are vectorized. `propto` (dropping constants) is not implemented; stanmath-rs always returns
the full density.

| Distribution | lpdf / lpmf | lcdf / lccdf | rng | Status |
| --- | --- | --- | --- | --- |
| `normal` | yes | yes | yes | tested |
| `std_normal` | yes | yes | no | tested |
| `student_t` | yes | yes | yes | tested |
| `cauchy` | yes | yes | yes | tested |
| `logistic` | yes | yes | yes | tested |
| `exponential` | yes | yes | yes | tested |
| `gamma` (shape, rate) | yes | yes | yes | tested |
| `inv_gamma` (shape, scale) | yes | yes | yes | tested |
| `beta` | yes | yes | yes | tested |
| `lognormal` | yes | yes | yes | tested |
| `uniform` | yes | yes | yes | tested |
| `weibull` (shape, scale) | yes | yes | yes | tested |
| `bernoulli` | yes | yes | yes | tested |
| `bernoulli_logit` | yes | no | yes | tested |
| `binomial` | yes | yes | yes | tested |
| `binomial_logit` | yes | no | no | tested |
| `poisson` | yes | yes | yes | tested |
| `poisson_log` | yes | no | yes | tested |
| `neg_binomial_2` (mean, dispersion) | yes | yes | yes | tested |
| `neg_binomial_2_log` | yes | no | no | tested |
| `categorical` | yes | not in Stan | yes | tested |
| `categorical_logit` | yes | not in Stan | no | tested |
| `dirichlet` | yes | not in Stan | no | tested |
| `multi_normal` (single vector) | yes | not in Stan | no | tested |
| `multi_normal_cholesky` (single vector) | yes | not in Stan | no | tested |
| `lkj_corr_cholesky` | yes | not in Stan | no | tested |
| `lkj_corr`, `multi_normal_prec`, `multi_student_t`, `wishart`, `inv_wishart`, Gaussian processes | | | | planned |
| remaining Stan distributions (`skew_normal`, `von_mises`, `wiener`, `beta_binomial`, `ordered_logistic`, ...) | | | | planned |

### `propto`

Stan's `propto` template flag, which the language exposes as `_lupdf` and `_lupmf`, is a separate
function in stanmath-rs: `normal_lupdf`, `poisson_lupmf`, and so on, for every density built on a
fused kernel. An argument is data when its elements are `f64`; `Var` and `Dual` arguments are
parameters. A `propto` function is zero when every argument is data, and otherwise drops the terms
that depend only on data, so its gradient equals that of the full function. The normal, standard
normal, Student's t, Cauchy, logistic, exponential, gamma, inverse gamma, beta, lognormal, uniform,
Gumbel, binomial, Poisson, and both negative binomial 2 kernels drop their terms as Stan does, and
their values match Stan Math 5.4.0 (probed with one argument a variable). As in Stan, the
lognormal keeps its constant `-ln(2 pi) / 2`. The other kernels keep every term when any argument
is a parameter, which differs from Stan's value by a constant only. The seven GLMs have `propto`
forms that match Stan's values; since `beta` fixes their scalar type, a GLM whose `phi` or `sigma`
is a parameter treats `beta` as one too. Multivariate densities and the full Wiener model have no
`propto` form yet.

### Boundary conventions

stanmath-rs follows Stan's checks: the same arguments are validated, with the same requirements.
Observations that fail a check return `Error::InvalidObservation`; parameters return
`Error::InvalidParameter`. Where Stan returns `-inf` instead of throwing (for example
`lognormal_lpdf(0 | ...)`, `poisson_lpmf(n > 0 | 0)`, `uniform_lpdf` outside the interval),
stanmath-rs does the same.

When a sequence argument is empty, the result is zero, but the scalar arguments are still
validated, as in Stan: `normal_lpdf([] | 0, -1)` is an error. Checks that relate two arguments,
such as `y > tau` for `wiener_lpdf`, apply only when both are present.
A few functions follow Stan in returning zero for an empty response before any other check:
`poisson_log_glm`, `bernoulli_logit_glm`, `binomial_logit_glm`, `categorical_logit_glm`, and
`ordered_logistic`. `ordered_probit`, `ordered_logistic_glm` (the cutpoints), `categorical`,
`normal_id_glm` (a scalar `sigma`), and `neg_binomial_2_log_glm` (a scalar `alpha`, but not
`phi`) check those arguments first, also as in Stan Math 5.4.0.

Deliberate differences:

* `beta_lpdf` and `weibull_lpdf` use `0 ln 0 = 0`, so for example `beta_lpdf(0 | 1, b)` is finite.

## Linear algebra

| Stan Math | stanmath-rs | Status |
| --- | --- | --- |
| `multiply` (matrix, matrix-vector) | `multiply`, `multiply_vector`, with structured adjoint | tested |
| `cholesky_decompose` | `cholesky_decompose`, with structured adjoint | tested |
| `mdivide_left_tri_low` | `mdivide_left_tri_low`, `mdivide_left_tri_low_transpose` | tested |
| `mdivide_left_spd`, `inverse_spd`, `log_determinant_spd` | same names | tested |
| `quad_form`, `multiply_lower_tri_self_transpose` | same names | tested |
| `add`, `subtract`, scalar multiplication | `add`, `subtract`, `scale` | tested |
| general `mdivide_left`, `inverse`, `log_determinant` (LU), eigen and singular value decompositions, matrix exponential, sparse matrices | | planned |

## Constraining transforms

All with `*_constrain`, `*_constrain_lp` (log Jacobian), and `*_free`.

| Stan Math 5.4 | Status |
| --- | --- |
| `lb`, `ub`, `lub` (including infinite bounds), `offset_multiplier` | tested |
| `ordered`, `positive_ordered` | tested |
| `sum_to_zero` (vector), `simplex` (softmax of `sum_to_zero`) | tested |
| `unit_vector` | tested |
| `cholesky_factor_corr`, `cholesky_factor_cov`, `cov_matrix` | tested |
| `corr_matrix` | planned (Stan orders its partial correlations differently from `cholesky_factor_corr`; a compatible implementation needs that order) |
| `sum_to_zero` matrices, `stochastic_row` and `stochastic_column` matrices | planned |

## Solvers

| Stan Math | stanmath-rs | Status |
| --- | --- | --- |
| scalar root finding (`root_finder`) | `solve_root` (bracketed safeguarded Newton), derivatives exact to the order of the scalar type | tested |
| `algebra_solver_newton` | `solve_system` (Newton), derivatives exact to the order of the scalar type | tested |
| `integrate_1d` | `integrate` (tanh-sinh, finite and infinite limits), derivatives with respect to parameters | tested |
| derivatives with respect to integration limits | | planned |
| `algebra_solver_powell`, fixed point solvers | | planned |
| `ode_rk45`, `ode_ckrk`, `ode_adams`, `ode_bdf`, adjoint ODE solvers, DAE solvers | Dormand and Prince, Cash and Karp, variable order Adams (orders 1 to 12, PECE), and Radau IIA; Stan uses CVODES for Adams and BDF, so results agree to the tolerance, not to rounding error. Derivatives are those of the numerical solution | tested |

The LKJ difference above is a Stan Math gradient bug: at `eta = 1` Stan evaluates the normalizing
constant with a special-case formula that does not depend on `eta`, so its derivative is lost.
stanmath-rs uses the general formula, which `tests/identities.rs` checks independently (the density
integrates to one for `K = 2`).

## Measured differences from Stan Math

`crates/stanmath-rs-distributions/tests/stan_reference.rs` compares 150 cases (values and gradients)
with Stan Math 5.4.0 at a relative tolerance of `1e-12`. Where the two disagree, the test requires
stanmath-rs to be closer to a 320-digit mpmath reference. At the time of writing, 45 values or gradient
components disagree, and all 45 are resolved in stanmath-rs's favor. The largest:

| Case | Stan relative error | stanmath-rs relative error |
| --- | --- | --- |
| `lkj_corr_cholesky_lpdf(L \| 1)`, d/d eta | 2.1 (drops the normalizing constant's derivative) | 7e-16 |
| `gamma_lccdf(50 \| 40, 1)`, d/d alpha | 1.6e-2 | below 1e-16 |
| `logistic_lccdf(-100 \| 0, 1)`, value | 1 (returns 0) | exact |
| `logistic_lccdf(-30 \| 0, 1)`, value | 1.7e-4 | exact |
| `inv_gamma_lcdf(0.2 \| 3, 1)`, d/d alpha | 1.4e-4 | 1e-16 |
| `normal_lcdf(-40 \| 0, 1)`, gradient | 7.4e-6 | 2e-16 |
| `std_normal_lccdf(30)`, gradient | 7.7e-6 | 2e-16 |
| `lognormal_lccdf(0.001 \| 0, 1)`, value | 3.1e-6 | 1.6e-15 |
| `student_t_lcdf(3 \| 1e4, 0, 1)`, d/d nu | 1.1e-6 | 6e-10 |

Run `cargo test -p stanmath-rs-distributions --test stan_reference -- --nocapture` for the full list.
