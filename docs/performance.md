# Performance

## Status

Benchmarks exist and run (`cargo bench -p probgrad`), but **no trustworthy timing results have been
recorded yet**. The only runs so far were made on a development machine whose load average was
between 100 and 230 on 8 cores, caused by unrelated processes. Timings from those runs vary by a
factor of two to three between repetitions and are not reported as performance claims. They should
be rerun on an idle machine (see below), and this file updated.

What does not depend on machine load is reported: tape sizes, and the structure of the work.

## Tape size

Normal log likelihood with two parameters (`mu`, `log_sigma`) and `n` observations, as recorded by
`Tape::stats()` (printed at the start of `cargo bench`):

| Formulation | `n = 100` | `n = 10,000` |
| --- | --- | --- |
| fused `normal_lpdf(&y, mu, sigma)` | 7 nodes, 8 edges, 128 bytes | 7 nodes, 8 edges, 128 bytes |
| loop of scalar `normal_lpdf` calls | 205 nodes, 404 edges, 7 KB | 20,005 nodes, 40,004 edges, 0.9 MB |
| elementary operations | 808 nodes, 1,107 edges, 28 KB | 80,008 nodes, 110,007 edges, 2.1 MB |

The fused kernel's tape is independent of `n`: the whole likelihood is one node whose partials are
computed analytically in the forward pass. A product of two `n` by `n` matrices adds exactly `n^2`
nodes (tested), and its reverse sweep runs one `f64` kernel instead of `n^3` scalar operations.

## Ratios from the loaded runs

Absolute times from the loaded machine are not reported, but the variants of one workload were
measured in the same run, so their ratios are indicative. Gradient of the normal likelihood with a
reused workspace:

| `n` | fused vs elementary formula | scalar-call loop vs elementary formula |
| --- | --- | --- |
| 100 | fused about 3x faster | loop about 2.7x slower |
| 1,000 | fused about 7x faster | loop about 2.9x slower |
| 10,000 | fused about 6x faster | loop about 2.4x slower |

A loop of scalar calls is slower than the elementary formula because every fused call allocates
its operand and partial vectors and validates its arguments. Adding an allocation-free scalar path
to `sum_kernel` is the first performance task.

`tools/stan-reference/stan_bench` was built but has not been run, so there is no comparison with
Stan Math yet.

## Benchmarks

`crates/probgrad/benches/probgrad.rs` measures, for each workload, plain `f64` evaluation, a gradient
on a fresh tape, and a gradient with a reused `GradientWorkspace` (the steady state of a sampler):

| Group | Workloads |
| --- | --- |
| `scalar_expression` | a small expression with `log_sum_exp`, `sin`, `exp`, `lgamma`: value, forward, reverse, Hessian |
| `normal_likelihood` | `n` = 10, 100, 1,000, 10,000: fused, scalar loop, elementary; gradient and Hessian |
| `logistic_regression` | `n` = 100, 1,000, 10,000 with 10 covariates |
| `linalg` | `multiply` and `log_determinant_spd` for `n` = 10, 50, 100: value and gradient |

`tools/stan-reference/stan_bench.cpp` times the normal likelihood and logistic regression gradients
in Stan Math 5.4.0 on the same data, built with Stan's standalone makefile flags.

To reproduce on an idle machine:

```sh
cargo bench -p probgrad --bench probgrad
cd tools/stan-reference
make STAN_MATH=$CMDSTAN/stan/lib/stan_math/ stan_bench
./stan_bench
```

Record the CPU, operating system, `rustc --version`, and the compiler of the Stan build with the
results.

## Known performance limitations

* Matrix kernels are plain loops; `faer` or `nalgebra` kernels for `f64` are a planned optimization.
* Logistic regression builds the linear predictor with scalar operations (two tape nodes per
  covariate and observation); a fused matrix-vector product with a structured adjoint would reduce
  that to one custom operation.
* Hessians of fused kernels record the partial expressions on the tape (`n` times the kernel's
  operation count). A vector-valued forward mode would let one forward pass carry all directions.
* Fused kernels allocate operand and partial vectors on every call, which makes many scalar calls
  slower than the plain formula (see the ratios above).
