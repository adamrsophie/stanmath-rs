# Numerical accuracy

This document records how accuracy is measured, the accuracy achieved, and the known limitations.
Every number here comes from a test in the repository.

## Method

* **References.** `tools/reference/elementary.py`, `special.py`, and `distributions.py` compute
  values and first and second derivatives with mpmath at 60 to 320 significant digits. Derivatives
  are central differences at that precision with a step relative to the argument, so their error
  is far below double precision. The fixtures are checked in, so tests do not need Python.
* **Stan Math.** `tools/stan-reference` records Stan Math 5.4.0 values and gradients for 150
  distribution cases; see `docs/compatibility.md`.
* **Every mode.** Each function is evaluated as `f64`, by forward mode, by reverse mode, by
  forward over reverse, and by forward over forward. All must match the reference.
* **Identities.** Tests also check relations that the implementation does not encode: the
  derivative of every CDF equals its density, CDF and complementary CDF sum to one, discrete CDF
  differences equal the mass function, densities integrate to one, and both Hessian modes agree.

Tolerances are relative unless stated. Where a tolerance is looser than a few ulps, the test file
states why, and the reason is repeated below.

## Elementary functions

`crates/stanmath-rs-autodiff/tests/reference.rs`, 126 unary and 17 binary points including extreme
arguments (`log(1e-300)`, `log1p_exp(700)`, `log1m_exp(-1e-10)`, `tanh(10)`, `hypot(1e5, 1e-3)`).

| Quantity | Tolerance |
| --- | --- |
| value | 4e-15 |
| first derivative, forward and reverse | 8e-15 |
| second derivative, both modes | 3e-14 |

Second derivatives required care. A derivative rule that is exact as a first derivative can cancel
catastrophically when it is itself differentiated. The reference suite found and fixed:

| Function | Original rule | Problem | Rule now |
| --- | --- | --- | --- |
| `expm1` | `y + 1` | 1e-3 relative error at `x = -30` | `exp(x)` |
| `tanh` | `1 - y^2` | 1e-8 error at `x = 10` | `1 / cosh(x)^2` |
| `atanh`, `asin`, `acos` | product rule on `(1 - x)(1 + x)` | 5e-10 error near zero | primitive with closed-form derivative |
| `hypot` | quotient rule on `a / h` | relative error 2.4 in d²/da² at `hypot(1e5, 1e-3)` | `DivHypot` primitive |
| `log_sum_exp` | `exp(a - y)` | `1 - exp(b - y)` cancels; 1.6e-14 at large `y` | `inv_logit(a - b)` |

## Special functions

`special_unary_reference.csv` (95 points), `special_binary_reference.csv` (30), and
`beta_inc_reference.csv` (9).

| Function | Value | First derivative | Second derivative | Notes |
| --- | --- | --- | --- | --- |
| `lgamma`, `digamma`, `trigamma`, `polygamma` | 1e-14 + 2e-15 absolute | same | 3e-14 | absolute floor because recurrences sum O(1) terms; matters only near the roots of digamma |
| `erf`, `erfc`, `Phi`, `log_Phi` | 4e-15 | 8e-15 | 3e-14 | `log_Phi` accurate to `x = -1e3` and beyond; `Phi` corrects the rounding of `x / sqrt(2)` |
| `inv_Phi` | 4e-15 | 3e-13 | 3e-13 | the derivative `1 / φ(y)` has condition number `y^2`, about 1400 at `p = 1e-300` |
| `erfcx` | 4e-15 | 8e-15 | 3e-14 | derivative loses `log10(2 x^2)` digits for large `x` (not used internally) |
| `lbeta`, `lgamma_stirling_diff` | 1e-14 | 1e-13 | 1e-12 | Stirling differences avoid cancellation for large arguments |
| `gamma_p`, `gamma_q` | 1e-14 | 1e-13 | 1e-12 | shape derivatives by forward mode through the series or continued fraction |
| `beta_inc` | 1e-14 | 1e-13 | Hessian modes agree to 1e-9 | as above |

### Derivatives of iterative algorithms

The incomplete gamma and beta functions differentiate their own series and continued fractions
in forward mode. Two pitfalls were found and fixed:

1. **Value-only stopping tests.** At integer shape parameters the continued fractions terminate
   exactly, so the value converges after finitely many terms while the derivative still depends on
   the tail. A first version returned `d/da P(2, 5)` with a relative error of 5e-5. Stopping tests
   now use `Real::magnitude`, which includes every forward-mode tangent.
2. **Hidden reverse-mode components.** When the shape partial is itself differentiated in reverse
   mode (Hessians), the tape's components are invisible to the stopping test. A hyper-dual `f64`
   evaluation first determines how many iterations second derivatives need, and the nested
   evaluation runs that many plus a margin.

Other fixes from the reference suite: `ψ(a) - ψ(a + b)` is computed by `digamma_diff` without
cancellation (the naive difference lost 5e-8 at `a = 1e-8`); `Q(a, x)` for small `a` uses a
dedicated expansion instead of `1 - P`; `lgamma1p` avoids forming `1 + a`.

## Distributions

`crates/stanmath-rs-distributions/tests/stan_reference.rs` against 320-digit mpmath:

| Group | Value | Gradient |
| --- | --- | --- |
| densities and mass functions | 1e-13 | 1e-12 |
| CDFs and complementary CDFs | 1e-13 | 1e-11 |
| Student's t CDF with `nu >= 1000` | 1e-13 | 1e-9 |

plus an absolute floor of `1e-14` times the largest argument scale, for gradients that are exactly
zero (`1 / beta - y` at `beta y = 1`).

CDFs choose the tail: `ln F` is computed directly when `F <= 1/2` and as `ln(1 - Q)` otherwise, so
neither tail loses the digits of its complement. Student's t CDF uses the complement
`t^2 / (nu + t^2)` directly and a Taylor branch near its center, where the beta-function form is
not differentiable.

## Linear algebra, transforms, and solvers

| Area | Check | Tolerance |
| --- | --- | --- |
| matrix reverse rules | vector-Jacobian product against the generic algorithm in forward mode, over every output of every operation | 1e-12 |
| matrix reverse rules | against sixth-order finite differences | 1e-8 |
| `log_determinant_spd` | gradient equals the inverse (closed form) | 1e-14 |
| Hessians through Cholesky | forward over reverse against forward over forward | 1e-13 absolute |
| transforms | round trips | 1e-12 to 1e-14 |
| transforms | Jacobian term against the log determinant of a forward-mode Jacobian | 1e-12 |
| `solve_root` | first and second derivatives against closed forms, both Hessian modes | 1e-14 |
| `solve_system` | gradient against finite differences of the closed form; Hessian modes agree | 1e-9 and 1e-12 |
| `integrate` | values and derivatives against closed forms, including `x^(-1/2)` and `ln x` at an endpoint and infinite limits | 1e-10 to 1e-12 |
| LKJ | density integrates to one for `K = 2` | 1e-4 (midpoint rule) |

## Known limitations

* `log_sum_exp_slice` second derivatives lose relative accuracy in entries where one softmax weight
  is within about `1e-16` of one; the absolute error stays at rounding level.
* Incomplete gamma and beta functions use series and continued fractions without uniform
  asymptotic expansions, so very large shapes (above about `1e4`) converge slowly and lose a few
  digits.
* `polygamma(n, x)` for negative `x` uses the upward recurrence, costing `O(|x|)`; it returns `NaN`
  below `-1e5`.
* `erfcx` overflows to `+inf` below `x = -26.6`, where the true value exceeds `f64::MAX`.
* Derivatives of solver solutions are exact for the computed root; the root itself has the solver's
  tolerance (relative `1e-15` for scalar roots, `1e-14` for systems by default).
* `integrate` does not differentiate with respect to its limits, and singularities inside the
  interval (rather than at an endpoint) need the interval split by the caller.
