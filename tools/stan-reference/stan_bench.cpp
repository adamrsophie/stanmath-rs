// Timing of the stanmath-rs benchmark workloads in Stan Math, on the same data.
//
//   make STAN_MATH=/path/to/stan_math/ stan_bench && ./stan_bench
//
// Prints one line per workload: name, n, nanoseconds per gradient evaluation (the median of 15
// batches, each at least 20 ms long). The workloads mirror crates/stanmath-rs/benches/stanmath_rs.rs.

#include <stan/math.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using stan::math::var;

namespace {

std::vector<double> data(std::size_t n) {
  std::vector<double> y(n);
  for (std::size_t i = 0; i < n; ++i) y[i] = std::sin(static_cast<double>(i) * 0.7) * 2.0 + 1.0;
  return y;
}

template <typename F>
double time_ns(F&& f) {
  // Calibrate a batch size that takes at least 20 ms.
  std::size_t reps = 1;
  for (;;) {
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < reps; ++r) f();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (ms >= 20.0) break;
    reps *= 2;
  }
  std::vector<double> samples;
  for (int s = 0; s < 15; ++s) {
    auto t0 = std::chrono::steady_clock::now();
    for (std::size_t r = 0; r < reps; ++r) f();
    double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
    samples.push_back(ns / static_cast<double>(reps));
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

volatile double sink = 0.0;

}  // namespace

int main() {
  using namespace stan::math;
  for (std::size_t n : {10, 100, 1000, 10000}) {
    std::vector<double> y_std = data(n);
    Eigen::VectorXd y = Eigen::Map<Eigen::VectorXd>(y_std.data(), n);
    double ns = time_ns([&] {
      var mu = 0.8, log_sigma = 0.1;
      var sigma = exp(log_sigma);
      var lp = normal_lpdf(y, mu, sigma) + normal_lpdf(mu, 0.0, 10.0) + log_sigma;
      lp.grad();
      sink += mu.adj() + log_sigma.adj();
      recover_memory();
    });
    std::printf("normal_likelihood,%zu,%.1f\n", n, ns);
  }
  const std::size_t k = 10;
  for (std::size_t n : {100, 1000, 10000}) {
    Eigen::MatrixXd x(n, k);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < k; ++j)
        x(i, j) = std::cos(static_cast<double>(i * k + j) * 0.37);
    std::vector<int> yv(n);
    for (std::size_t i = 0; i < n; ++i) yv[i] = std::sin(static_cast<double>(i) * 1.3) > 0.0;
    double ns = time_ns([&] {
      Eigen::Matrix<var, -1, 1> beta = Eigen::VectorXd::Constant(k, 0.1).cast<var>();
      Eigen::Matrix<var, -1, 1> eta = x * beta;
      var lp = bernoulli_logit_lpmf(yv, eta) + normal_lpdf(beta, 0.0, 2.5);
      lp.grad();
      sink += beta(0).adj();
      recover_memory();
    });
    std::printf("logistic_regression,%zu,%.1f\n", n, ns);
  }
  return 0;
}
