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
| `reduce_sum`, `map_rect` (TBB) | `reduce_sum`, `map_rect`, and threaded `reduce_sum_parallel`, `map_rect_parallel` | experimental |
| `map_rect` over MPI | `map_rect_mpi` in `backends/stanmath-rs-mpi` (CI with OpenMPI) | experimental |
| OpenCL | `backends/stanmath-rs-opencl`, Stan's OpenCL function list (CI with POCL) | experimental |

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
| other Stan functions | `Phi_approx`, `owens_t`, `log_modified_bessel_first_kind`, Bessel functions, `lambert_w`, `binary_log_loss`, `log_falling_factorial`, `log_rising_factorial`, `lmgamma`, `hypergeometric_*` | tested |
| integer functions | `choose`, `falling_factorial`, `rising_factorial`, `int_step`, `logical_*` | supported |

## Distributions

Log densities (`_lpdf` / `_lpmf`), log CDFs (`_lcdf`), log complementary CDFs (`_lccdf`), and
random number generators (`_rng`) with Stan's parameterizations and argument checks. All
arguments are vectorized. Each log density has a `propto` form; see below.

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
function in stanmath-rs: `normal_lupdf`, `poisson_lupmf`, and so on, for every log density and every
overload (`multi_normal_lupdf_array`, `poisson_log_glm_lupmf_row`, ...). An argument is data when
its elements are `f64`; `Var` and `Dual` arguments are parameters. A `propto` function is zero when
every argument is data, and otherwise drops the terms that depend only on data, so its gradient
equals that of the full function. Every kernel groups its terms as Stan does, and its values match
Stan Math 5.4.0, probed with one argument a variable at a time. Stan's grouping is not always the
finest one: the lognormal keeps its constant `-ln(2 pi) / 2`, the log-logistic keeps `(beta - 1) ln
y` while `alpha` is a parameter, the beta negative binomial drops only `-ln Gamma(n + 1)`, and the
Yule-Simon drops `ln Gamma(n)`. The seven GLMs have `propto` forms that match Stan's values. Their
result type is the promotion of every argument's, as in Stan, so `beta` can be data while `x`,
`alpha`, `phi`, or `sigma` is a parameter, and data `beta` stays data under `propto`.

The multivariate densities, the Dirichlet, categorical, multinomial, Dirichlet multinomial,
ordered, Poisson binomial, hypergeometric, and discrete range masses, and the five and seven
parameter Wiener densities also match Stan's `propto` values. Matrix arguments share the scalar
type, so they are parameters whenever the result is differentiable; vector arguments such as `y`,
`mu`, `nu`, `w`, and `eta` keep their own flags. Stan's grouping shows in a few places: the
Cholesky forms of the Wishart and inverse Wishart drop their Jacobian constant `K ln 2` only with
`nu`, the inverse Wishart keeps `-nu K / 2 ln 2` while its scale is a parameter, and the
hypergeometric and discrete range masses, whose arguments are all integers, are zero under
`propto`, even outside the support. The Laplace marginal likelihoods have no `propto` form.

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
* `matrix_normal_prec_lpdf` of an `N` by `P` matrix weights `ln det Sigma` by `P / 2` and
  `ln det D` by `N / 2`, as the matrix normal density does. Stan Math 5.4.0 swaps the two weights,
  so the values differ unless `N = P` or `ln det Sigma = ln det D`.
* The built in Laplace marginals (`laplace_marginal_*_lpmf`) use every observation. Stan Math
  5.4.0 uses only the first `m.size()` of them: with `y_index` longer than `m`, later observations
  do not change its value. The `_lupmf` forms equal the `_lpmf` forms, as in Stan, which keeps the
  constants. The negative binomial takes a scalar `eta`, as the Functions Reference documents;
  stanc 2.40 types it as a vector, and Stan Math 5.4.0 rejected every vector length probed.

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

`tests/stan_reference_extra.rs` adds 189 cases for 34 more families: chi-square, inverse and scaled
inverse chi-square, double exponential, exponentially modified normal, Frechet, Gumbel, Pareto,
Pareto type 2, Rayleigh, skew normal, skew double exponential, von Mises, beta proportion,
log-logistic, negative binomial, beta binomial, beta negative binomial, and Poisson binomial (each
`_lpdf`/`_lpmf` and, where Stan has them, `_lcdf` and `_lccdf`), and the ordered logistic and
probit, multinomial, multinomial logit, Dirichlet multinomial, multi normal precision, multi
Student t and its Cholesky form, Wishart, inverse Wishart and their Cholesky forms, LKJ correlation
and covariance, and multi GP and its Cholesky form. They agree with Stan to `1e-10` except 29 rows, where stanmath-rs matches a 40 digit mpmath reference (`tools/reference/extra.py`)
to `1e-12` and Stan does not:

| Case | Stan relative error | stanmath-rs |
| --- | --- | --- |
| `chi_square_lcdf`, `chi_square_lccdf`, inverse and scaled inverse chi-square CDFs, d/d nu | 1e-7 to 1.6e-5 | about 1e-15 |
| `beta_neg_binomial_lcdf`, `beta_neg_binomial_lccdf`, gradient | 5e-7 to 6e-5 | about 1e-15 |
| `exp_mod_normal_lcdf(-2 \| 0, 0.5, 3)`, value and gradient | 1.6e-10 to 2e-8 | about 1e-15 |
| `poisson_binomial_lccdf(3 \| [0.6, 0.4, 0.8])`, value | NaN (the exact value is `log 0`) | `-inf`, zero gradient |
| `ordered_probit_lpmf`, d/d eta and cutpoints | 7e-7 to 3e-6 | about 1e-16 |

`crates/stanmath-rs/tests/stan_reference_scalar.rs` covers every function of one or two reals in the
Stan language (from `stanc --dump-stan-math-signatures`) except the `_rng` and bound transform
functions: 103 functions and 574 cases, written by `tools/stan-reference/gen_scalar.py`. Where Stan
throws, stanmath-rs must return an error or NaN. They agree to `1e-10` except 43 rows, where
stanmath-rs matches a 40 digit mpmath reference (`tools/reference/scalar.py`) to `1e-12`:

| Case | Stan relative error | stanmath-rs |
| --- | --- | --- |
| `digamma` derivative and `trigamma` | 1e-10 to 3e-9 | about 1e-16 |
| `std_normal_lcdf`, `std_normal_lccdf`, derivative | 1.7e-7 to 4e-5 | about 1e-16 |
| `gamma_q`, d/d a | 1.7e-7 to 1.3e-5 | about 1e-15 |
| chi-square and inverse chi-square CDFs, d/d nu | 4e-9 to 4e-6 | about 1e-15 |

`crates/stanmath-rs/tests/stan_reference_matrix.rs` checks 97 cases of the matrix functions
(decompositions, inverses, solves, products, reductions, `matrix_exp`, `svd_*`, `qr_*`, sorting and
softmax) and every constraining transform's `_constrain`, `_jacobian`, and `_unconstrain` form
against Stan to `1e-10`; a matrix result is reduced to a weighted sum of its entries, so the
gradient covers each one. `svd_U`, `svd_V`, and the `qr_*` factors match Stan's signs.
`eigenvectors_sym` signs are arbitrary (Stan's follow Eigen's solver), so that case checks
`V diag(1, 2, 3) Vᵀ`. The comparison changed `corr_matrix_unconstrain` to read the diagonal as Stan
does, which changes only its derivative with respect to the diagonal.

`crates/stanmath-rs-distributions/tests/signatures.rs`, written by `tools/reference/gen_signatures.py`
from the signature dump, maps every vectorized distribution signature (11953 Stan signatures over
int, real, vector, row vector, and one dimensional arrays, in 1273 Rust shapes) to its Rust call,
and requires each broadcast call to equal the sum (the product for `_cdf`) of the scalar calls in
value and gradient. `tests/signatures_multivariate.rs` does the same for the `array[] vector` forms
of the multivariate normal, Student t, Dirichlet, and ordered densities.

Stan vectorizes scalar functions over containers; in Rust, `apply_scalar_unary` and
`apply_scalar_binary` apply a scalar function over vectors, matrices, and arrays of any depth.
`crates/stanmath-rs/tests/signatures_elementwise.rs`, written by `tools/reference/gen_elementwise.py`,
calls every elementwise signature of 106 functions this way (2844 signatures up to three array
levels, and a depth 8 check for the 3763 deeper ones, which share one recursive impl). `add`,
`subtract`, `multiply`, `elt_multiply`, and `elt_divide` are Rust's operators.

The vectorized `_rng` signatures go through `apply_broadcast1` to `apply_broadcast4`, which apply a
scalar sampler at each index of scalar or slice arguments. `tests/signatures_rng.rs`, written by
`tools/reference/gen_rng.py`, calls all 1995 of them (124 Rust shapes) and requires the draws to
equal the scalar sampler's, in order from the same seed.

The bound transforms (`lower_bound`, `upper_bound`, `lower_upper_bound`, `offset_multiplier`, each
`_constrain`, `_jacobian`, `_unconstrain`) apply over containers the same way, with
`apply_scalar_ternary` when both bounds are containers. `tests/signatures_bounds.rs`, written by
`tools/reference/gen_bounds.py`, calls 468 of their container signatures (252 Rust shapes; 684
deeper array forms share the recursive impl) and requires the scalar transform of every element and,
for `_jacobian`, the summed log Jacobian. `tests/signatures_qf.rs` covers the 256 broadcast forms of
`student_t_qf` through `apply_broadcast4`.

`tests/signatures_containers.rs`, written by `tools/reference/gen_containers.py`, calls 302
signatures of `dims`, `num_elements`, `size`, `rep_array`, `reverse`, `append_array`, `head`,
`tail`, and `segment` over every element type, complex ones included, and checks the shape of each
result (239 deeper array forms run the same generic code).
`tests/signatures_complex.rs`, written by `tools/reference/gen_complex.py`, calls 84 signatures of
`to_complex`, `get_real`, `get_imag`, and `conj` over complex containers (`ScalarContainer` now
treats `Complex<T>` as a leaf) and checks each against the definition (111 deeper array forms share
the recursive impl).
`tests/signatures_utilities.rs`, written by `tools/reference/gen_utilities.py`, calls 187
signatures of `sum`, `prod`, `cumulative_sum`, `to_vector`, `to_row_vector`, `to_array_1d`,
`to_array_2d`, `to_vector_array`, `to_row_vector_array`, `to_matrix`, `append_col`, `append_row`,
`fma`, `log_mix`, `hypergeometric_3F2`, and `hypergeometric_pFq`, each through its Rust form, and
checks the result against the definition written out from the input's elements. The fill order of
`to_matrix` and the array form of `log_mix` were probed against Stan Math. Stan's one-element
concatenations (`append_col(real, row_vector)`, `append_row(vector, vector)` and similar) are
`append_array` on slices, `sum(array[] int)` is `iter().sum()`, and `to_array_1d` of an array of
any depth is `to_array_1d_nested`.
`tests/signatures_operators.rs`, written by `tools/reference/gen_operators.py`, calls the 72
container signatures of `add`, `subtract`, `elt_multiply`, and `elt_divide`: two real matrices
through the library functions, every other form (vectors, complex containers, a scalar with a
container) as the Rust operator through `apply_scalar_unary` or `apply_scalar_binary`.
`tests/signatures_misc.rs` calls the signatures no generator covers (samplers with container
arguments, comparisons, constructors, row and column dot products, eigen and Schur
decompositions, the Wiener CDFs, `poisson_binomial_cdf`, the hidden Markov functions, and the
deprecated functional solvers), each labeled with a `// stan:` comment and checked against its
definition, a reconstruction, or its support.

`tools/reference/coverage.py` sorts every line of the signature dump by how it is checked: called
by a signature test (by a generator's ledger or a `// stan:` label), compared with Stan values in
the reference CSVs, skipped only because its arrays nest deeper than the generators write calls
for, or checked by name only. A function whose signatures differ only in spelling a vector
(vector, row vector, or one dimensional real array) is one Rust call, so a reference comparison of
that call counts for each spelling. Run against CmdStan 2.40.0 it reports 18993 called, 210
compared with Stan values, 5088 deeper than the calls, none by name, and no unchecked line.
The 5088 deeper lines are array forms nested past the depth the generators write calls for (the
same nested map as the shallower forms); they are not called one by one.

`tests/signatures_linalg_misc.rs` and `tests/signatures_distributions_misc.rs` call the last
signatures no generator reached: products and quotients with a scalar, the vector forms of the
solvers, the complex forms of the matrix functions, the decompositions (checked by
reconstruction), the FFTs (against a direct DFT), the CSR functions, the small constructors, the
Poisson binomial (against enumeration), the categorical forms with an array response, the
Gaussian dynamic linear model (against a Kalman filter written in the test), the container forms
of `wiener_lpdf` (against the sum of scalar calls), and `hypergeometric_rng` and
`categorical_rng`.

`tests/signatures_glm.rs`, written by `tools/reference/gen_glm.py`, calls all 58 `_glm_lpdf` and
`_glm_lpmf` signatures and compares each, in value and gradient, with the sum of its response
distribution over the observations. As in Stan, a one row `x` (Stan's `row_vector x`) is shared
by every observation of a longer response, and `normal_id_glm_lpdf` with a real response (a one
element slice) shares it across the rows of `x`. Stan Math 5.4.0 differs in one case:
`poisson_log_glm_lpmf` with an int response and several rows of `x` adds `-lgamma(y + 1)` once
rather than once per observation; stanmath-rs returns the sum over the observations.

`tests/signatures_gp.rs`, written by `tools/reference/gen_gp.py`, calls all 28 signatures of the
stationary kernels (`gp_exp_quad_cov`, `gp_matern32_cov`, `gp_matern52_cov`,
`gp_exponential_cov`, `gp_periodic_cov`) through `gp_points_1d`, the `_cross` functions, and the
`_ard` functions, and compares every entry with the kernel formula. The elementwise test also
calls the forms that pair a two dimensional int array with a matrix (`ldexp(matrix, array[,]
int)`), which Stan pairs by position.

`tests/stan_reference_scalar_extra.rs` compares with Stan Math 5.4.0 the scalar signatures the
one and two real sweep cannot reach (`tools/stan-reference/stan_reference_scalar_extra.cpp`): the
constants, the Bessel functions, `binary_log_loss`, `lmgamma`, `ldexp`, and the factorials with
their int arguments, the int functions (`choose`, `modulus`, the logical operators, `to_int`),
`hypergeometric_2F1`, `inc_beta`, `inv_inc_beta`, `log_mix`, the scalar bound transforms, and
the 6 to 9 argument Wiener densities. At Stan's default precision (1e-4) the eight argument
Wiener density differs from ours by about 2e-6; at precision 1e-8 the two agree to 1e-15.
`tests/stan_reference_matrix.rs` names the Stan signatures each of its cases calls, and
`signatures_complex.rs` also calls the complex container forms of `abs`, `minus`, and `pow`.

`tests/signatures_transforms.rs`, written by `tools/reference/gen_transforms.py`, calls the array
forms of the structural transforms (`_constrain`, `_jacobian`, and `_unconstrain` of simplex,
sum_to_zero, ordered, positive_ordered, unit_vector, the stochastic matrices, cov_matrix,
corr_matrix, and the Cholesky factors) to three array levels. Stan's array forms apply the
transform to each element; in Rust they are an iterator map, and a `_jacobian` map adds each
element's log Jacobian to one `lp`. `tests/signatures_multivariate.rs` labels the multivariate
normal, Student t, Dirichlet, and ordered logistic and probit signatures it calls, and now also
calls the ordered forms with one cutpoint vector for every observation and (probit) a real
linear predictor with one cutpoint vector per observation.

`tests/stan_reference_complex.rs` compares the complex scalar functions (the elementary and
inverse trigonometric and hyperbolic functions, `exp`, `log`, `log10`, `sqrt`, `pow`, `abs`, `arg`,
`norm`, `conj`, `proj`, and the arithmetic operators) with Stan Math 5.4.0 in value and gradient,
at points on and off the branch cuts, approached from both signed zeros
(`tools/stan-reference/stan_reference_complex.cpp`). Stan differs from C99 Annex G in the sign of
some zero outputs, and at points on a cut its gradient is taken on the other side of the cut than
its value. Where they differ, stanmath-rs follows `tests/data/c99_reference_complex.csv`
(`tools/reference/complex.py`: Python's `cmath` values and mpmath gradients on the side the
input's signed zero picks). The sweep fixed `atanh` (so also `atan`) on the cuts: `1 - z` was
built as `0 - z + 1`, which turned an imaginary `-0` into `+0` and put the result on the wrong side.

The scalar sweep also fixed four conventions to match Stan: `fmax` and `fmin` return the second argument
on ties (and take its derivative); `beta(a, b)` with a nonpositive argument is
`|Γ(a) Γ(b) / Γ(a + b)|`; `log_falling_factorial` and `log_rising_factorial` reject `x <= 0`; and
`log_modified_bessel_first_kind` accepts orders in `[-1, 0)`.
