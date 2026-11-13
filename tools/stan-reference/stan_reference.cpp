// Reference values and gradients from Stan Math for stanmath-rs's distribution tests.
//
// Each output line is
//   function,ints,reals,value,gradient
// where ints, reals, and gradient are ';' separated lists, and the gradient is taken with respect
// to every real argument in order. A case that Stan rejects prints "error" as its value.

#include <stan/math.hpp>

#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using stan::math::var;
using Vars = std::vector<var>;

namespace {

std::string join(const std::vector<double>& v) {
  std::ostringstream os;
  os << std::setprecision(17);
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) os << ';';
    os << v[i];
  }
  return os.str();
}

std::string join_int(const std::vector<int>& v) {
  std::ostringstream os;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) os << ';';
    os << v[i];
  }
  return os.str();
}

void emit(const std::string& name, const std::vector<int>& ints,
          const std::vector<double>& x,
          const std::function<var(const std::vector<int>&, const Vars&)>& f) {
  Vars xv(x.begin(), x.end());
  std::cout << name << ',' << join_int(ints) << ',' << join(x) << ',';
  try {
    var lp = f(ints, xv);
    lp.grad();
    std::vector<double> g;
    for (auto& v : xv) g.push_back(v.adj());
    std::cout << std::setprecision(17) << lp.val() << ',' << join(g) << '\n';
  } catch (const std::exception&) {
    std::cout << "error,\n";
  }
  stan::math::recover_memory();
}

using Pts = std::vector<std::vector<double>>;

template <typename F>
void continuous(const std::string& name, const Pts& pts, F f) {
  for (const auto& p : pts) {
    emit(name, {}, p, [&](const std::vector<int>&, const Vars& v) { return f(v); });
  }
}

template <typename F>
void discrete(const std::string& name,
              const std::vector<std::pair<std::vector<int>, std::vector<double>>>& pts, F f) {
  for (const auto& p : pts) {
    emit(name, p.first, p.second, f);
  }
}

}  // namespace

int main() {
  using namespace stan::math;

  const Pts loc_scale = {{1.3, 0.5, 1.2}, {-40.0, 0.0, 1.0}, {5.0, -2.0, 0.1}, {0.1, 0.0, 30.0}};
  continuous("normal_lpdf", loc_scale, [](const Vars& v) { return normal_lpdf(v[0], v[1], v[2]); });
  continuous("normal_lcdf", loc_scale, [](const Vars& v) { return normal_lcdf(v[0], v[1], v[2]); });
  continuous("normal_lccdf", loc_scale, [](const Vars& v) { return normal_lccdf(v[0], v[1], v[2]); });
  const Pts std_pts = {{-3.0}, {0.5}, {30.0}};
  continuous("std_normal_lpdf", std_pts, [](const Vars& v) { return std_normal_lpdf(v[0]); });
  continuous("std_normal_lcdf", std_pts, [](const Vars& v) { return std_normal_lcdf(v[0]); });
  continuous("std_normal_lccdf", std_pts, [](const Vars& v) { return std_normal_lccdf(v[0]); });

  const Pts t_pts = {{1.5, 3.0, 0.5, 2.0}, {-20.0, 0.7, 0.0, 1.0}, {0.1, 50.0, 0.2, 0.5},
                     {3.0, 1e4, 0.0, 1.0}};
  continuous("student_t_lpdf", t_pts,
             [](const Vars& v) { return student_t_lpdf(v[0], v[1], v[2], v[3]); });
  continuous("student_t_lcdf", t_pts,
             [](const Vars& v) { return student_t_lcdf(v[0], v[1], v[2], v[3]); });
  continuous("student_t_lccdf", t_pts,
             [](const Vars& v) { return student_t_lccdf(v[0], v[1], v[2], v[3]); });

  const Pts heavy = {{1.5, 0.5, 2.0}, {-100.0, 0.0, 1.0}, {-30.0, 0.0, 1.0}};
  continuous("cauchy_lpdf", heavy, [](const Vars& v) { return cauchy_lpdf(v[0], v[1], v[2]); });
  continuous("cauchy_lcdf", heavy, [](const Vars& v) { return cauchy_lcdf(v[0], v[1], v[2]); });
  continuous("cauchy_lccdf", heavy, [](const Vars& v) { return cauchy_lccdf(v[0], v[1], v[2]); });
  continuous("logistic_lpdf", heavy, [](const Vars& v) { return logistic_lpdf(v[0], v[1], v[2]); });
  continuous("logistic_lcdf", heavy, [](const Vars& v) { return logistic_lcdf(v[0], v[1], v[2]); });
  continuous("logistic_lccdf", heavy,
             [](const Vars& v) { return logistic_lccdf(v[0], v[1], v[2]); });

  const Pts exp_pts = {{0.5, 2.0}, {10.0, 0.1}, {0.0, 3.0}};
  continuous("exponential_lpdf", exp_pts, [](const Vars& v) { return exponential_lpdf(v[0], v[1]); });
  continuous("exponential_lcdf", {{0.5, 2.0}, {10.0, 0.1}},
             [](const Vars& v) { return exponential_lcdf(v[0], v[1]); });
  continuous("exponential_lccdf", exp_pts,
             [](const Vars& v) { return exponential_lccdf(v[0], v[1]); });

  const Pts gamma_pts = {{1.5, 2.5, 0.5}, {0.01, 0.3, 2.0}, {50.0, 40.0, 1.0}};
  continuous("gamma_lpdf", gamma_pts, [](const Vars& v) { return gamma_lpdf(v[0], v[1], v[2]); });
  continuous("gamma_lcdf", gamma_pts, [](const Vars& v) { return gamma_lcdf(v[0], v[1], v[2]); });
  continuous("gamma_lccdf", gamma_pts, [](const Vars& v) { return gamma_lccdf(v[0], v[1], v[2]); });
  const Pts ig_pts = {{1.5, 2.5, 0.5}, {0.2, 3.0, 1.0}};
  continuous("inv_gamma_lpdf", ig_pts, [](const Vars& v) { return inv_gamma_lpdf(v[0], v[1], v[2]); });
  continuous("inv_gamma_lcdf", ig_pts, [](const Vars& v) { return inv_gamma_lcdf(v[0], v[1], v[2]); });
  continuous("inv_gamma_lccdf", ig_pts,
             [](const Vars& v) { return inv_gamma_lccdf(v[0], v[1], v[2]); });

  const Pts beta_pts = {{0.3, 2.0, 5.0}, {0.999, 0.5, 0.5}, {0.01, 30.0, 2.0}};
  continuous("beta_lpdf", beta_pts, [](const Vars& v) { return beta_lpdf(v[0], v[1], v[2]); });
  continuous("beta_lcdf", beta_pts, [](const Vars& v) { return beta_lcdf(v[0], v[1], v[2]); });
  continuous("beta_lccdf", beta_pts, [](const Vars& v) { return beta_lccdf(v[0], v[1], v[2]); });

  const Pts ln_pts = {{1.5, 0.2, 0.7}, {1e-3, 0.0, 1.0}};
  continuous("lognormal_lpdf", ln_pts, [](const Vars& v) { return lognormal_lpdf(v[0], v[1], v[2]); });
  continuous("lognormal_lcdf", ln_pts, [](const Vars& v) { return lognormal_lcdf(v[0], v[1], v[2]); });
  continuous("lognormal_lccdf", ln_pts,
             [](const Vars& v) { return lognormal_lccdf(v[0], v[1], v[2]); });

  const Pts unif_pts = {{0.3, 0.0, 1.0}, {1.5, -1.0, 2.0}};
  continuous("uniform_lpdf", unif_pts, [](const Vars& v) { return uniform_lpdf(v[0], v[1], v[2]); });
  continuous("uniform_lcdf", unif_pts, [](const Vars& v) { return uniform_lcdf(v[0], v[1], v[2]); });
  continuous("uniform_lccdf", unif_pts, [](const Vars& v) { return uniform_lccdf(v[0], v[1], v[2]); });

  const Pts wb_pts = {{1.5, 2.5, 1.2}, {0.01, 0.5, 2.0}};
  continuous("weibull_lpdf", wb_pts, [](const Vars& v) { return weibull_lpdf(v[0], v[1], v[2]); });
  continuous("weibull_lcdf", wb_pts, [](const Vars& v) { return weibull_lcdf(v[0], v[1], v[2]); });
  continuous("weibull_lccdf", wb_pts, [](const Vars& v) { return weibull_lccdf(v[0], v[1], v[2]); });

  using IP = std::vector<std::pair<std::vector<int>, std::vector<double>>>;
  const IP bern = {{{1}, {0.3}}, {{0}, {0.3}}};
  discrete("bernoulli_lpmf", bern, [](auto& n, auto& v) { return bernoulli_lpmf(n[0], v[0]); });
  discrete("bernoulli_lcdf", bern, [](auto& n, auto& v) { return bernoulli_lcdf(n[0], v[0]); });
  discrete("bernoulli_lccdf", {{{0}, {0.3}}},
           [](auto& n, auto& v) { return bernoulli_lccdf(n[0], v[0]); });
  discrete("bernoulli_logit_lpmf", {{{1}, {2.5}}, {{0}, {-30.0}}, {{1}, {-30.0}}},
           [](auto& n, auto& v) { return bernoulli_logit_lpmf(n[0], v[0]); });

  const IP binom = {{{3, 10}, {0.4}}, {{0, 5}, {0.01}}, {{20, 20}, {0.9}}};
  discrete("binomial_lpmf", binom, [](auto& n, auto& v) { return binomial_lpmf(n[0], n[1], v[0]); });
  discrete("binomial_lcdf", {{{3, 10}, {0.4}}, {{0, 5}, {0.01}}},
           [](auto& n, auto& v) { return binomial_lcdf(n[0], n[1], v[0]); });
  discrete("binomial_lccdf", {{{3, 10}, {0.4}}, {{0, 5}, {0.01}}},
           [](auto& n, auto& v) { return binomial_lccdf(n[0], n[1], v[0]); });
  discrete("binomial_logit_lpmf", {{{3, 10}, {-0.4}}},
           [](auto& n, auto& v) { return binomial_logit_lpmf(n[0], n[1], v[0]); });

  const IP pois = {{{3}, {2.5}}, {{0}, {0.1}}, {{100}, {80.0}}};
  discrete("poisson_lpmf", pois, [](auto& n, auto& v) { return poisson_lpmf(n[0], v[0]); });
  discrete("poisson_lcdf", pois, [](auto& n, auto& v) { return poisson_lcdf(n[0], v[0]); });
  discrete("poisson_lccdf", pois, [](auto& n, auto& v) { return poisson_lccdf(n[0], v[0]); });
  discrete("poisson_log_lpmf", {{{3}, {0.9}}, {{0}, {-5.0}}},
           [](auto& n, auto& v) { return poisson_log_lpmf(n[0], v[0]); });

  const IP nb = {{{3}, {2.5, 4.0}}, {{0}, {0.5, 0.1}}, {{50}, {40.0, 1e3}}};
  discrete("neg_binomial_2_lpmf", nb,
           [](auto& n, auto& v) { return neg_binomial_2_lpmf(n[0], v[0], v[1]); });
  discrete("neg_binomial_2_lcdf", nb,
           [](auto& n, auto& v) { return neg_binomial_2_lcdf(n[0], v[0], v[1]); });
  discrete("neg_binomial_2_lccdf", nb,
           [](auto& n, auto& v) { return neg_binomial_2_lccdf(n[0], v[0], v[1]); });
  discrete("neg_binomial_2_log_lpmf", {{{3}, {0.9, 4.0}}, {{0}, {-2.0, 0.5}}},
           [](auto& n, auto& v) { return neg_binomial_2_log_lpmf(n[0], v[0], v[1]); });

  discrete("categorical_lpmf", {{{2}, {0.2, 0.5, 0.3}}}, [](auto& n, auto& v) {
    Eigen::Matrix<var, -1, 1> theta(3);
    theta << v[0], v[1], v[2];
    return categorical_lpmf(n[0], theta);
  });
  discrete("categorical_logit_lpmf", {{{3}, {0.1, -1.0, 2.0}}}, [](auto& n, auto& v) {
    Eigen::Matrix<var, -1, 1> beta(3);
    beta << v[0], v[1], v[2];
    return categorical_logit_lpmf(n[0], beta);
  });

  // Multivariate densities. Reals: y (3), mu (3), then the lower triangle of L or Sigma row by
  // row (6); symmetric matrices use each lower entry in both positions.
  const std::vector<double> mvn = {0.3, -1.2, 0.8, 0.1, -0.5, 0.4, 1.3, 0.4, 0.9, -0.2, 0.3, 1.1};
  emit("multi_normal_cholesky_lpdf", {}, mvn, [](auto&, const Vars& v) {
    Eigen::Matrix<var, -1, 1> y(3), mu(3);
    y << v[0], v[1], v[2];
    mu << v[3], v[4], v[5];
    Eigen::Matrix<var, -1, -1> L = Eigen::Matrix<var, -1, -1>::Zero(3, 3);
    L(0, 0) = v[6]; L(1, 0) = v[7]; L(1, 1) = v[8]; L(2, 0) = v[9]; L(2, 1) = v[10]; L(2, 2) = v[11];
    return multi_normal_cholesky_lpdf(y, mu, L);
  });
  const std::vector<double> mvn_cov = {0.3, -1.2, 0.8, 0.1, -0.5, 0.4, 2.0, 0.4, 1.5, -0.3, 0.2, 1.2};
  emit("multi_normal_lpdf", {}, mvn_cov, [](auto&, const Vars& v) {
    Eigen::Matrix<var, -1, 1> y(3), mu(3);
    y << v[0], v[1], v[2];
    mu << v[3], v[4], v[5];
    Eigen::Matrix<var, -1, -1> S(3, 3);
    S(0, 0) = v[6]; S(1, 0) = S(0, 1) = v[7]; S(1, 1) = v[8];
    S(2, 0) = S(0, 2) = v[9]; S(2, 1) = S(1, 2) = v[10]; S(2, 2) = v[11];
    return multi_normal_lpdf(y, mu, S);
  });
  // L is the Cholesky factor of a correlation matrix; reals: lower triangle row by row, then eta.
  for (double eta : {1.0, 2.5}) {
    const double l10 = 0.6, l11 = 0.8, l20 = -0.3, l21 = 0.4, l22 = std::sqrt(1.0 - 0.09 - 0.16);
    emit("lkj_corr_cholesky_lpdf", {}, {1.0, l10, l11, l20, l21, l22, eta},
         [](auto&, const Vars& v) {
           Eigen::Matrix<var, -1, -1> L = Eigen::Matrix<var, -1, -1>::Zero(3, 3);
           L(0, 0) = v[0]; L(1, 0) = v[1]; L(1, 1) = v[2]; L(2, 0) = v[3]; L(2, 1) = v[4];
           L(2, 2) = v[5];
           return lkj_corr_cholesky_lpdf(L, v[6]);
         });
  }
  emit("dirichlet_lpdf", {}, {0.2, 0.5, 0.3, 1.5, 2.0, 0.7}, [](auto&, const Vars& v) {
    Eigen::Matrix<var, -1, 1> theta(3), alpha(3);
    theta << v[0], v[1], v[2];
    alpha << v[3], v[4], v[5];
    return dirichlet_lpdf(theta, alpha);
  });

  // Invalid arguments: Stan throws, stanmath-rs must return an error.
  continuous("normal_lpdf", {{1.0, 0.0, -1.0}}, [](const Vars& v) { return normal_lpdf(v[0], v[1], v[2]); });
  continuous("gamma_lpdf", {{-1.0, 2.0, 1.0}}, [](const Vars& v) { return gamma_lpdf(v[0], v[1], v[2]); });
  continuous("beta_lpdf", {{1.5, 2.0, 1.0}}, [](const Vars& v) { return beta_lpdf(v[0], v[1], v[2]); });
  discrete("poisson_lpmf", {{{-1}, {2.0}}}, [](auto& n, auto& v) { return poisson_lpmf(n[0], v[0]); });
  return 0;
}
