//! Criterion benchmarks. Run with `cargo bench -p probgrad`; results are summarized in
//! `docs/performance.md`.
//!
//! Each workload is measured as plain `f64` evaluation, as a gradient on a fresh tape, and as a
//! gradient with a reused `GradientWorkspace` (the steady state of a sampler), so that model
//! evaluation, tape construction, and the reverse sweep can be told apart.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use probgrad::prelude::*;

fn data(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as f64) * 0.7).sin() * 2.0 + 1.0)
        .collect()
}

/// Normal log likelihood with priors, vectorized (one fused node in reverse mode).
fn normal_model<T: Real>(p: &[T], y: &[f64]) -> T {
    let sigma = p[1].exp();
    normal_lpdf(y, p[0], sigma).unwrap() + normal_lpdf(p[0], 0.0, 10.0).unwrap() + p[1]
}

/// The same model written as a loop of scalar calls.
fn normal_model_scalar_loop<T: Real>(p: &[T], y: &[f64]) -> T {
    let sigma = p[1].exp();
    let mut lp = normal_lpdf(p[0], 0.0, 10.0).unwrap() + p[1];
    for &yi in y {
        lp += normal_lpdf(yi, p[0], sigma).unwrap();
    }
    lp
}

/// The same model written with elementary operations only (no fused kernel).
fn normal_model_elementary<T: Real>(p: &[T], y: &[f64]) -> T {
    let mu = p[0];
    let log_sigma = p[1];
    let inv_sigma = (-log_sigma).exp();
    let mut lp = -(mu.square()) * 0.005 + log_sigma;
    for &yi in y {
        let z = (-mu + yi) * inv_sigma;
        lp += -(z.square()) * 0.5 - log_sigma;
    }
    lp
}

fn bench_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_likelihood");
    for n in [10usize, 100, 1000, 10_000] {
        let y = data(n);
        let x = [0.8, 0.1];
        group.bench_with_input(BenchmarkId::new("f64_value", n), &y, |b, y| {
            b.iter(|| normal_model(black_box(&x), y))
        });
        group.bench_with_input(BenchmarkId::new("grad_fresh_tape", n), &y, |b, y| {
            b.iter(|| grad(|p| normal_model(p, y), black_box(&x)))
        });
        let mut ws = GradientWorkspace::new();
        let mut g = [0.0; 2];
        group.bench_with_input(BenchmarkId::new("grad_workspace", n), &y, |b, y| {
            b.iter(|| ws.value_and_grad_into(|p| normal_model(p, y), black_box(&x), &mut g))
        });
        group.bench_with_input(
            BenchmarkId::new("grad_workspace_scalar_loop", n),
            &y,
            |b, y| {
                b.iter(|| {
                    ws.value_and_grad_into(
                        |p| normal_model_scalar_loop(p, y),
                        black_box(&x),
                        &mut g,
                    )
                })
            },
        );
        group.bench_with_input(
            BenchmarkId::new("grad_workspace_elementary", n),
            &y,
            |b, y| {
                b.iter(|| {
                    ws.value_and_grad_into(|p| normal_model_elementary(p, y), black_box(&x), &mut g)
                })
            },
        );
        group.bench_with_input(BenchmarkId::new("hessian", n), &y, |b, y| {
            b.iter(|| hessian(|p| normal_model(p, y), black_box(&x)))
        });
    }
    group.finish();
}

fn logistic_model<T: Real>(beta: &[T], x: &[f64], y: &[i64], k: usize) -> T {
    let eta: Vec<T> = x
        .chunks(k)
        .map(|row| {
            let mut s = T::zero();
            for (v, b) in row.iter().zip(beta) {
                s += *b * *v;
            }
            s
        })
        .collect();
    bernoulli_logit_lpmf(y, &eta).unwrap() + normal_lpdf(beta, 0.0, 2.5).unwrap()
}

fn bench_logistic(c: &mut Criterion) {
    let mut group = c.benchmark_group("logistic_regression");
    let k = 10;
    for n in [100usize, 1000, 10_000] {
        let x: Vec<f64> = (0..n * k).map(|i| ((i as f64) * 0.37).cos()).collect();
        let y: Vec<i64> = (0..n)
            .map(|i| i64::from(((i as f64) * 1.3).sin() > 0.0))
            .collect();
        let beta = vec![0.1; k];
        let mut ws = GradientWorkspace::new();
        let mut g = vec![0.0; k];
        group.bench_function(BenchmarkId::new("f64_value", n), |b| {
            b.iter(|| logistic_model(black_box(&beta), &x, &y, k))
        });
        group.bench_function(BenchmarkId::new("grad_workspace", n), |b| {
            b.iter(|| {
                ws.value_and_grad_into(|p| logistic_model(p, &x, &y, k), black_box(&beta), &mut g)
            })
        });
    }
    group.finish();
}

fn spd(n: usize) -> Matrix<f64> {
    Matrix::from_fn(n, n, |i, j| {
        if i == j {
            n as f64 + 1.0
        } else {
            1.0 / (1.0 + (i as f64 - j as f64).abs())
        }
    })
}

fn bench_linalg(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg");
    group.sample_size(20);
    for n in [10usize, 50, 100] {
        let a = spd(n);
        let flat = a.as_slice().to_vec();
        group.bench_function(BenchmarkId::new("multiply_f64", n), |b| {
            b.iter(|| multiply(black_box(&a), black_box(&a)).unwrap())
        });
        group.bench_function(BenchmarkId::new("multiply_grad", n), |b| {
            b.iter(|| {
                grad(
                    |p| {
                        let m = Matrix::from_column_major(n, n, p.to_vec());
                        multiply(&m, &m).unwrap().as_slice().iter().copied().sum()
                    },
                    black_box(&flat),
                )
            })
        });
        group.bench_function(BenchmarkId::new("log_det_spd_f64", n), |b| {
            b.iter(|| log_determinant_spd(black_box(&a)).unwrap())
        });
        group.bench_function(BenchmarkId::new("log_det_spd_grad", n), |b| {
            b.iter(|| {
                grad(
                    |p| log_determinant_spd(&Matrix::from_column_major(n, n, p.to_vec())).unwrap(),
                    black_box(&flat),
                )
            })
        });
    }
    group.finish();
}

fn bench_scalar(c: &mut Criterion) {
    fn f<T: Real>(x: &[T]) -> T {
        let s = log_sum_exp(x[0], x[1] * 2.0);
        s * x[0].sin() + (x[1] * x[2]).exp() / (x[0].square() + 1.0) + lgamma(x[2] + 2.0)
    }
    let x = [0.3, -0.7, 1.1];
    let mut group = c.benchmark_group("scalar_expression");
    group.bench_function("f64", |b| b.iter(|| f(black_box(&x))));
    group.bench_function("forward_directional", |b| {
        b.iter(|| probgrad::autodiff::directional_derivative(f, black_box(&x), &[1.0, 0.0, 0.0]))
    });
    let mut ws = GradientWorkspace::new();
    let mut g = [0.0; 3];
    group.bench_function("reverse_workspace", |b| {
        b.iter(|| ws.value_and_grad_into(|p| f(p), black_box(&x), &mut g))
    });
    group.bench_function("hessian", |b| b.iter(|| hessian(|p| f(p), black_box(&x))));
    group.finish();
}

/// Prints tape sizes for the benchmarked workloads (memory is reported this way because a counting
/// allocator would need `unsafe`, which the workspace forbids).
fn report_tape_sizes(_: &mut Criterion) {
    for n in [100usize, 10_000] {
        let y = data(n);
        for (name, stats) in [
            ("fused", tape_stats(|p| normal_model(p, &y))),
            (
                "scalar_loop",
                tape_stats(|p| normal_model_scalar_loop(p, &y)),
            ),
            ("elementary", tape_stats(|p| normal_model_elementary(p, &y))),
        ] {
            eprintln!(
                "tape normal_likelihood n={n} {name}: {} nodes, {} edges, {} bytes",
                stats.nodes, stats.edges, stats.bytes
            );
        }
    }
}

fn tape_stats(f: impl for<'t> Fn(&[Var<'t>]) -> Var<'t>) -> probgrad::autodiff::TapeStats {
    let tape = Tape::new();
    let x = tape.vars(&[0.8, 0.1]);
    let _ = f(&x);
    tape.stats()
}

criterion_group!(
    benches,
    report_tape_sizes,
    bench_scalar,
    bench_normal,
    bench_logistic,
    bench_linalg
);
criterion_main!(benches);
