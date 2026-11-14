# Architecture

This document records the design of stanmath-rs and the alternatives that were considered. It is
written for contributors who need to extend the library without redesigning it.

## Goals and priorities

1. Mathematical correctness, including higher-order derivatives.
2. Memory safety without `unsafe` (every crate sets `unsafe_code = "forbid"`).
3. Performance, measured before optimizing.
4. A small, idiomatic Rust API that serves both hand-written models and generated code.

Stan Math is the mathematical specification and the correctness and performance baseline. It is
not translated: the C++ template machinery (`operands_and_partials`, `return_type_t`, Eigen
expression templates, the global arena allocator) has Rust-native replacements described below.

## Workspace layout

| Crate | Depends on | Contents |
| --- | --- | --- |
| `stanmath-rs-core` | `libm` | `Error`, `Matrix<T>` container, `f64` special functions |
| `stanmath-rs-autodiff` | core | `Real` trait, `Dual`, `Var`/`Tape`, drivers, differentiable functions, broadcasting |
| `stanmath-rs-linalg` | autodiff | matrix products, Cholesky, triangular solves, log determinants |
| `stanmath-rs-distributions` | autodiff, linalg, `rand`, `rand_distr` | densities, CDFs, samplers |
| `stanmath-rs-transforms` | autodiff, linalg | constraining transforms with log Jacobians |
| `stanmath-rs-solvers` | autodiff | root finding, nonlinear systems, quadrature |
| `stanmath-rs` | all | umbrella crate, prelude, examples, benchmarks, sampler integration test |

Crates are created when their milestone starts; there are no placeholder crates.

## Dependencies

Every dependency was checked for maintenance, license compatibility with GPL-3.0, and whether it
does something stanmath-rs should not reimplement.

| Crate | Used for | License | Notes |
| --- | --- | --- | --- |
| `libm` | `lgamma`, `tgamma`, `erf`, `erfc` | MIT | rust-lang maintained port of musl; values verified against mpmath |
| `rand`, `rand_distr` | `*_rng` samplers (optional feature) | MIT or Apache-2.0 | rust-random project |
| `criterion` | benchmarks (dev only) | MIT or Apache-2.0 | |
| `proptest` | property tests (dev only) | MIT or Apache-2.0 | |
| `nuts-rs` | sampler integration test (dev only) | MIT | brings `faer` into the dev build only |

Considered and not adopted:

* **`num-dual`**: its `DualNum` trait requires `'static` and lacks gamma, digamma, and error
  functions, so it cannot nest with a tape-borrowing `Var`.
* **`nalgebra`**: `Scalar` requires `'static`, excluding `Var<'t>` as an element type.
* **`ndarray`**: no `'static` bound, but its arithmetic and linear algebra live in other crates and
  a small column-major container was enough.
* **`faer`**: the strongest candidate for fast `f64` kernels inside the matrix reverse rules;
  deferred until benchmarks on an idle machine show where time goes.
* **`statrs`**: its special functions are not differentiable in shape parameters and could not
  feed the generic derivative rules; stanmath-rs's own implementations are tested against mpmath.
* **Enzyme** (LLVM-level automatic differentiation): not evaluated. It requires a nightly compiler
  and a custom toolchain, which conflicts with the stable-Rust requirement.

## The scalar abstraction: `Real`

Every differentiable function is written once, generically over `T: Real`. Implementations:

| Type | Meaning |
| --- | --- |
| `f64` | plain values |
| `Dual<T>` | forward mode over any `Real` `T` |
| `Var<'t>` | reverse mode on tape `'t` |

Nesting gives higher orders: `Dual<Var>` (forward over reverse) for Hessians and Hessian vector
products, `Dual<Dual<f64>>` for forward over forward, `Dual<Dual<Var>>` for third derivatives.
Reverse over forward is not provided: tape partials are `f64`, as in Stan, so `Var` does not nest
inside itself.

### Primitives carry derivative rules written against `Real`

Functions are implemented as small structs implementing `UnaryFn`, `BinaryFn`, or `NaryFn`. Each
has an `f64` value function and a derivative rule that is generic over `S: Real`:

```rust
impl UnaryFn for Lgamma {
    fn value(&self, x: f64) -> f64 { /* f64 algorithm */ }
    fn derivative<S: Real>(&self, x: S, _y: S) -> S { digamma(x) }
}
```

Every scalar type implements three hooks (`map_unary`, `map_binary`, `map_nary`) once:

* `f64` calls `value`;
* `Var` calls `value` and the rule with `S = f64`, then records one node;
* `Dual<T>` recurses into `T` for the value and calls the rule with `S = T` for the tangent.

Because the rule is evaluated in the inner scalar type, nested modes differentiate the rule
itself. A first-order-only shortcut (an `f64` partial inside `Dual<Var>`) is impossible by
construction; this was the main reason to reject a design with per-type hand-written partials.

A consequence is that derivative rules must be numerically good *as functions*, not only at first
order. The mpmath reference suite (below) found several rules that were exact for the first
derivative but cancelled catastrophically in the second (`1 - tanh(x)^2`, `y + 1` for `expm1`,
`exp(x - lse)` for `log_sum_exp`); each was replaced with a cancellation-free form, sometimes by
making the derivative a primitive of its own (`InvOneMinusSquare`, `DivHypot`).

### Alternatives rejected

* **`num-dual` for forward mode.** Its `DualNum` trait requires `'static` and has no gamma,
  digamma, or error functions; a reverse-mode variable that borrows its tape cannot implement it,
  so `Dual<Var>` would be impossible. Our `Dual<T>` is about 400 lines.
* **Per-type method implementations.** Writing `exp`, `lgamma`, ... separately for `Dual` and
  `Var` doubles the code and makes it easy for the two to diverge. The primitive structs give a
  single source of truth.

## Reverse mode: explicit tape with borrowed handles

```rust
pub struct Var<'t> { val: f64, idx: u32, tape: Option<&'t Tape> }   // 24 bytes, Copy
```

The tape stores, per node, a range into two edge arrays (`parents: Vec<u32>`,
`partials: Vec<f64>`). The reverse sweep is one backward loop over these vectors. Constants are
`Var`s without a tape; operations on constants record nothing, and an operation whose local partial
is exactly zero records nothing either.

Safety properties, all without `unsafe`:

| Property | Mechanism |
| --- | --- |
| A `Var` cannot outlive its tape | the `'t` borrow |
| `Tape::clear` cannot run while a `Var` is alive | `clear(&mut self)` |
| Mixing two tapes is detected | pointer comparison on every binary operation, always on, panics |
| Closures passed to drivers cannot capture another tape's `Var` | `for<'t> Fn(&[Var<'t>]) -> Var<'t>` makes it a compile error |
| One tape per thread | `Tape: Send + !Sync`, so `Var: !Send` |
| Repeated sweeps start clean | adjoints live in a per-sweep buffer, never on the tape |

These are pinned by `compile_fail` doctests and unit tests.

Custom operations (`Tape::custom_op`) record a closure from output adjoints to input adjoints.
They exist for operations whose adjoint is cheaper than their dense Jacobian (matrix products,
Cholesky), so the linear algebra layer never materializes Jacobians.

### Alternatives rejected

* **Dorset's design.** Dorset (2018) converts `Rc<*mut Vari>` to `&mut Vari` with `unsafe`,
  which creates aliasing mutable references; its `Var` is not `Copy`, so expressions need
  explicit borrows. Nothing was reused.
* **Thread-local global tape (Stan's model).** Smaller handles (16 bytes) and no lifetime, but
  stale variables after a tape reset can only be caught at run time, and nested differentiation
  needs a manual stack discipline. The borrowed tape turns both into compile errors.
* **Branded lifetimes (generativity) on every tape.** Would make all cross-tape mixing a compile
  error, but forces every use of a tape into a closure. The closure drivers get this guarantee;
  the explicit `Tape` API uses the run-time check instead.
* **Reference counted graph nodes.** Allocation and reference counting per operation; slower and
  no safer than indices.

## Vectorized arguments and scalar promotion

Probability functions accept scalars, slices, `Vec`s, and arrays, mixing `f64` data and
differentiable parameters, without turbofish annotations. The design:

* `Arg` describes an argument (`type Elem`, length or broadcast);
* `Promote<B>` computes the common scalar type (`f64` with `T` gives `T`);
* `Lift<T>` converts an element;
* `Broadcast<N>` combines `N` arguments and returns elements as `[Out; N]`.

`Real` has `Promote<f64, Out = Self> + Promote<Self, Out = Self> + Lift<Self>` as supertraits so
that generic code (`T: Real`) can call `normal_lpdf(&data, mu, sigma)` and get `T` back. The
array return type matters: a tuple-valued associated type does not normalize inside generic code,
while `[Out; N]` destructures everywhere.

The obvious alternative, `impl<T: Real> Arg<T> for T` plus `impl<T: Real> Arg<T> for f64`,
overlaps at `T = f64` and is rejected by coherence. That was verified with a spike before the
design was fixed.

## Limitations of the type design

* A generic `fn model<T: Real>(x: &[T]) -> T` must be passed to reverse-mode drivers through a
  closure, `grad(|x| model(x), &x0)`, because the driver needs a function that is generic over the
  tape lifetime. Forward-mode drivers accept the function directly.
* `2.0 * x` works for concrete types but not for a generic `T` (a bound on `f64` cannot be a
  supertrait); write `x * 2.0`.
* Comparisons between AD values compare primal values only.

## Fused distribution kernels

A log density over `n` observations written with elementary operations records several tape
nodes per observation. Stan avoids this with `operands_and_partials`. stanmath-rs's equivalent is the
`Kernel` trait in `stanmath-rs-distributions`: one function per distribution that returns the log
density of one element and, on request, its partial derivatives, both generic over `Real`.
`sum_kernel` validates every element, lays the arguments out as operands (a broadcast scalar is
one operand, a sequence is `n`), and calls `Real::map_nary` with an `NaryFn` that sums the kernel.

* `f64`: evaluates the sum.
* `Var`: one tape node with one edge per non-constant operand, partials computed in `f64` in the
  same pass as the value (`NaryFn::value_and_partials`).
* `Dual<T>`: the partials are evaluated in `T`, so forward over reverse differentiates them and
  Hessians are exact.

Measured effect on the tape of a 10,000 observation normal likelihood: 7 nodes and 8 edges fused,
versus 20,005 nodes for a loop of scalar calls and 80,008 nodes for the elementary formula.

CDFs are composed from differentiable special functions instead, because their shape partials
come from iterative algorithms.

## Derivatives of iterative algorithms

The regularized incomplete gamma and beta functions have no closed-form shape derivatives. The
primitive's derivative rule runs the same series or continued fraction (DLMF 8.7.1, 8.9.2,
8.17.22, evaluated by a generic modified Lentz routine) on `Dual<S>` seeded in the shape
parameter, which yields the derivative as an ordinary `S` expression that nested modes can
differentiate again.

Two details make this exact rather than approximately right:

* Stopping tests use `Real::magnitude`, the largest absolute value over the value and all
  forward-mode tangents, so derivatives must converge too. At integer shapes the continued
  fractions terminate exactly in value while the derivatives still depend on the tail.
* When the partial will itself be differentiated in reverse mode (`S::ORDER > 0`), the tape's
  components are invisible to the stopping test. A hyper-dual `f64` evaluation first finds the
  iteration count that second derivatives need, and the nested evaluation runs that many plus a
  margin (the convergence oracle in `incomplete.rs`).

## Linear algebra

`nalgebra::DMatrix<T>` requires `T: 'static`, which excludes `Var<'t>`, so matrices use
stanmath-rs's column-major `Matrix<T>`. A separate matrix-scalar trait would have added a second bound
to every model, transform, and multivariate density, because Rust has no specialization. Instead
`Real` has a fourth hook, `map_multi`, taking a `MultiFn`:

* `eval<S: Real>`: one generic scalar algorithm, used for `f64` and every forward-mode type;
* `reverse`: an `f64` adjoint rule, recorded by `Var` as a single `Tape::custom_op` with a
  `needed` mask so that data matrices cost nothing in the sweep.

A product of two `n` by `n` matrices therefore adds `n^2` tape nodes (a test pins this), and no
dense Jacobian is formed. Core operations are `multiply`, `cholesky_decompose`, and the two
triangular solves; `log_determinant_spd`, `mdivide_left_spd`, `inverse_spd`, `quad_form`, and
`multiply_lower_tri_self_transpose` are compositions.

Cholesky convention: only the lower triangle is read (symmetry is checked on values with Stan's
absolute tolerance `1e-8`), and the reverse rule (Murray 2016, symmetrized for the lower-triangle
parameterization) assigns derivatives to the lower triangle. Generic and reverse paths agree
entry by entry, which the tests check through forward-mode Jacobians.

Kernels are plain loops. Using `faer` or `nalgebra` for the `f64` kernels is deferred until a
benchmark on an unloaded machine shows they matter.

## Transforms

Transforms follow Stan Math 5.4, read from its headers: `lb`, `ub`, `lub` (stable Jacobian
`ln(ub - lb) - |y| - 2 log1p_exp(-|y|)`), `offset_multiplier`, `ordered`, `positive_ordered`,
the isometric `sum_to_zero`, `simplex` as the softmax of `sum_to_zero`, `unit_vector`,
`cholesky_factor_corr` (canonical partial correlations through `tanh`), `cholesky_factor_cov`, and
`cov_matrix`. Every Jacobian term is tested against the log determinant of a forward-mode
Jacobian of the map onto the constrained value's free coordinates. Those tests established the
reference measures: the sum-to-zero term (zero) is relative to the hyperplane's surface measure,
while the simplex term is relative to Lebesgue measure on the first `N` coordinates.

## Solvers

Implicit functions are differentiated without adjoint systems. The solution is found in `f64`;
then Newton steps are taken in the caller's scalar type from the `f64` solution. At an exact root
one step reproduces the implicit-function-theorem derivative, and each further step doubles the
number of exact Taylor coefficients, so `solve_root` and `solve_system` take the smallest number of
steps that covers `Real::ORDER`. Gradients cost one extra residual and Jacobian evaluation in the
AD type; Hessians cost two. The equation is a user type implementing a trait with a generic
`eval<S: Real>` method, because closures cannot be generic.

Quadrature uses tanh-sinh nodes, which are constants, so derivatives with respect to parameters
are the same rule applied to the integrand's derivative; the stopping test includes forward-mode
tangents. Each endpoint's nodes stop independently, because nodes approach `0` far more closely
than they approach `1` in floating point.

## Testing strategy

* **High-precision references.** `tools/reference/*.py` computes values and first and second
  derivatives with mpmath at 60 to 320 digits. Tests check every function in every mode (`f64`,
  forward, reverse, forward over reverse, forward over forward).
* **Stan Math.** `tools/stan-reference` compiles a generator against an installed Stan Math (path
  from the `STAN_MATH` make variable) and records values and gradients; fixtures are checked in so
  CI does not need Stan. Where stanmath-rs and Stan disagree, a test requires stanmath-rs to be closer to
  the mpmath reference.
* **Cross-mode consistency.** Forward and reverse gradients, both Hessian modes, Jacobians, JVPs,
  VJPs, and matrix reverse rules against the generic algorithm.
* **Identities.** CDF derivatives equal densities, densities integrate to one, transform
  Jacobians equal forward-mode log determinants.
* **External sampler.** `crates/stanmath-rs/tests/nuts.rs` runs NUTS from `nuts-rs` on stanmath-rs log
  densities and checks the draws' moments.
* **Property tests** (`proptest`) and **finite differences** as additional nets.
