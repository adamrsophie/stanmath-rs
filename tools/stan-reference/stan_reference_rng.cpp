// Draws from every _rng function of Stan Math, for a two sample comparison with stanmath-rs's
// samplers. Stan's generator (boost::ecuyer1988) cannot be matched draw for draw, so the test
// compares distributions: each output element of each function is one feature, and the
// stanmath-rs test draws its own sample and runs a two sample Kolmogorov Smirnov test against the
// draws here.
//
// Each output line is
//   function,feature,draw;draw;...
// with N draws, where feature numbers the output elements (column major for a matrix).

#include <stan/math.hpp>

#include <boost/random/additive_combine.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

constexpr int N = 1000;
boost::ecuyer1988 rng(20261114);

std::vector<double> flat(double x) { return {x}; }
std::vector<double> flat(int x) { return {static_cast<double>(x)}; }
std::vector<double> flat(const std::vector<int>& x) { return {x.begin(), x.end()}; }
std::vector<double> flat(const VectorXd& x) { return {x.data(), x.data() + x.size()}; }
std::vector<double> flat(const MatrixXd& x) { return {x.data(), x.data() + x.size()}; }

template <typename F>
void emit(const std::string& name, F f) {
  std::vector<std::vector<double>> draws;
  for (int n = 0; n < N; ++n) {
    std::vector<double> d = flat(f());
    if (draws.empty()) draws.resize(d.size());
    for (std::size_t k = 0; k < d.size(); ++k) draws[k].push_back(d[k]);
  }
  for (std::size_t k = 0; k < draws.size(); ++k) {
    std::cout << name << ',' << k << ',';
    for (int n = 0; n < N; ++n) std::cout << (n ? ";" : "") << draws[k][n];
    std::cout << '\n';
  }
}

VectorXd vec(std::initializer_list<double> x) {
  VectorXd v(x.size());
  int k = 0;
  for (double e : x) v(k++) = e;
  return v;
}

}  // namespace

int main() {
  using namespace stan::math;
  std::cout << std::setprecision(8);
  const VectorXd theta3 = vec({0.2, 0.5, 0.3});
  const VectorXd beta3 = vec({0.4, -1.0, 1.3});
  const VectorXd mu2 = vec({0.5, -1.0});
  MatrixXd s2(2, 2);
  s2 << 1.5, 0.4, 0.4, 0.8;
  const MatrixXd l2 = cholesky_decompose(s2);
  MatrixXd x32(3, 2);
  x32 << 0.5, -1.0, 1.2, 0.3, -0.7, 0.9;
  const VectorXd cuts = vec({-0.5, 0.4, 1.3});

  // Scalar continuous.
  emit("normal_rng", [&] { return normal_rng(0.5, 1.3, rng); });
  emit("std_normal_rng", [&] { return std_normal_rng(rng); });
  emit("cauchy_rng", [&] { return cauchy_rng(0.5, 1.3, rng); });
  emit("logistic_rng", [&] { return logistic_rng(0.5, 1.3, rng); });
  emit("gumbel_rng", [&] { return gumbel_rng(0.5, 1.3, rng); });
  emit("double_exponential_rng", [&] { return double_exponential_rng(0.5, 1.3, rng); });
  emit("lognormal_rng", [&] { return lognormal_rng(0.5, 0.7, rng); });
  emit("student_t_rng", [&] { return student_t_rng(4.5, 0.5, 1.3, rng); });
  emit("skew_normal_rng", [&] { return skew_normal_rng(0.5, 1.3, 2.0, rng); });
  emit("skew_double_exponential_rng", [&] { return skew_double_exponential_rng(0.5, 1.3, 0.3, rng); });
  emit("exp_mod_normal_rng", [&] { return exp_mod_normal_rng(0.5, 1.3, 0.8, rng); });
  emit("gamma_rng", [&] { return gamma_rng(2.5, 1.5, rng); });
  emit("inv_gamma_rng", [&] { return inv_gamma_rng(3.5, 1.5, rng); });
  emit("exponential_rng", [&] { return exponential_rng(1.7, rng); });
  emit("chi_square_rng", [&] { return chi_square_rng(3.5, rng); });
  emit("inv_chi_square_rng", [&] { return inv_chi_square_rng(5.5, rng); });
  emit("scaled_inv_chi_square_rng", [&] { return scaled_inv_chi_square_rng(5.5, 0.8, rng); });
  emit("beta_rng", [&] { return beta_rng(2.5, 1.5, rng); });
  emit("beta_proportion_rng", [&] { return beta_proportion_rng(0.3, 6.0, rng); });
  emit("weibull_rng", [&] { return weibull_rng(2.5, 1.5, rng); });
  emit("frechet_rng", [&] { return frechet_rng(3.5, 1.5, rng); });
  emit("rayleigh_rng", [&] { return rayleigh_rng(1.3, rng); });
  emit("pareto_rng", [&] { return pareto_rng(1.2, 3.5, rng); });
  emit("pareto_type_2_rng", [&] { return pareto_type_2_rng(0.5, 1.5, 3.5, rng); });
  emit("loglogistic_rng", [&] { return loglogistic_rng(1.5, 4.0, rng); });
  emit("uniform_rng", [&] { return uniform_rng(-0.5, 2.0, rng); });
  emit("von_mises_rng", [&] { return von_mises_rng(0.5, 2.0, rng); });

  // Scalar discrete.
  emit("bernoulli_rng", [&] { return bernoulli_rng(0.3, rng); });
  emit("bernoulli_logit_rng", [&] { return bernoulli_logit_rng(-0.4, rng); });
  emit("binomial_rng", [&] { return binomial_rng(12, 0.35, rng); });
  emit("beta_binomial_rng", [&] { return beta_binomial_rng(12, 2.5, 1.5, rng); });
  emit("hypergeometric_rng", [&] { return hypergeometric_rng(5, 4, 6, rng); });
  emit("poisson_rng", [&] { return poisson_rng(3.5, rng); });
  emit("poisson_log_rng", [&] { return poisson_log_rng(1.2, rng); });
  emit("neg_binomial_rng", [&] { return neg_binomial_rng(2.5, 0.7, rng); });
  emit("neg_binomial_2_rng", [&] { return neg_binomial_2_rng(3.5, 2.0, rng); });
  emit("neg_binomial_2_log_rng", [&] { return neg_binomial_2_log_rng(1.2, 2.0, rng); });
  emit("beta_neg_binomial_rng", [&] { return beta_neg_binomial_rng(3.0, 12.0, 2.0, rng); });
  emit("yule_simon_rng", [&] { return yule_simon_rng(2.5, rng); });
  emit("discrete_range_rng", [&] { return discrete_range_rng(-2, 4, rng); });
  emit("categorical_rng", [&] { return categorical_rng(theta3, rng); });
  emit("categorical_logit_rng", [&] { return categorical_logit_rng(beta3, rng); });
  emit("ordered_logistic_rng", [&] { return ordered_logistic_rng(0.3, cuts, rng); });
  emit("ordered_probit_rng", [&] { return ordered_probit_rng(0.3, cuts, rng); });
  emit("poisson_binomial_rng", [&] { return poisson_binomial_rng(vec({0.2, 0.7, 0.45, 0.9}), rng); });

  // Multivariate.
  emit("multinomial_rng", [&] { return multinomial_rng(theta3, 10, rng); });
  emit("multinomial_logit_rng", [&] { return multinomial_logit_rng(beta3, 10, rng); });
  emit("dirichlet_multinomial_rng", [&] { return dirichlet_multinomial_rng(vec({1.5, 2.5, 0.8}), 10, rng); });
  emit("dirichlet_rng", [&] { return dirichlet_rng(vec({1.5, 2.5, 0.8}), rng); });
  emit("multi_normal_rng", [&] { return multi_normal_rng(mu2, s2, rng); });
  emit("multi_normal_cholesky_rng", [&] { return multi_normal_cholesky_rng(mu2, l2, rng); });
  emit("multi_student_t_rng", [&] { return multi_student_t_rng(5.5, mu2, s2, rng); });
  emit("multi_student_t_cholesky_rng", [&] { return multi_student_t_cholesky_rng(5.5, mu2, l2, rng); });
  emit("wishart_rng", [&] { return wishart_rng(4.5, s2, rng); });
  emit("wishart_cholesky_rng", [&] { return wishart_cholesky_rng(4.5, l2, rng); });
  emit("inv_wishart_rng", [&] { return inv_wishart_rng(6.5, s2, rng); });
  emit("inv_wishart_cholesky_rng", [&] { return inv_wishart_cholesky_rng(6.5, l2, rng); });
  emit("lkj_corr_rng", [&] { return lkj_corr_rng(3, 1.5, rng); });
  emit("lkj_corr_cholesky_rng", [&] { return lkj_corr_cholesky_rng(3, 1.5, rng); });
  emit("bernoulli_logit_glm_rng", [&] { return bernoulli_logit_glm_rng(x32, vec({0.2, -0.3, 0.1}), vec({0.8, -0.5}), rng); });
  // Two states over three time points: log densities by column, transitions by row.
  MatrixXd log_omegas(2, 3);
  log_omegas << -0.5, -1.2, -0.3, -1.1, -0.4, -0.9;
  MatrixXd gamma(2, 2);
  gamma << 0.7, 0.3, 0.4, 0.6;
  emit("hmm_latent_rng", [&] { return hmm_latent_rng(log_omegas, gamma, vec({0.6, 0.4}), rng); });
  return 0;
}
