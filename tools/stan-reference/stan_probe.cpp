// Black-box probes of Stan Math behavior that the Functions Reference leaves open.
//
// Each output line is
//   name,result
// where result is a value, a ';' separated list of values, or "error: <message>". The probes only
// call Stan Math's public functions; they settle questions of convention for stanmath-rs.

#include <stan/math.hpp>

#include <complex>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string clean(std::string s) {
  for (auto& c : s) {
    if (c == '\n' || c == ',') c = ' ';
  }
  return s;
}

void probe(const std::string& name, const std::function<std::vector<double>()>& f) {
  std::cout << name << ',';
  try {
    const auto v = f();
    std::ostringstream os;
    os << std::setprecision(17);
    for (std::size_t i = 0; i < v.size(); ++i) {
      if (i) os << ';';
      os << v[i];
    }
    std::cout << os.str() << '\n';
  } catch (const std::exception& e) {
    std::cout << "error: " << clean(e.what()) << '\n';
  }
}

void probe1(const std::string& name, const std::function<double()>& f) {
  probe(name, [&] { return std::vector<double>{f()}; });
}

std::vector<double> lower_rows(const Eigen::MatrixXd& m) {
  std::vector<double> out;
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j <= i && j < m.cols(); ++j) out.push_back(m(i, j));
  }
  return out;
}

}  // namespace

int main() {
  using namespace stan::math;
  using V = Eigen::VectorXd;
  using M = Eigen::MatrixXd;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const std::vector<double> none;
  const std::vector<int> none_int;

  // Validation of the other arguments when a sequence argument is empty.
  probe1("empty_y.bad_scalar_sigma", [&] { return normal_lpdf(none, 0.0, -1.0); });
  probe1("empty_y.nan_scalar_mu", [&] { return normal_lpdf(none, nan, 1.0); });
  probe1("empty_y.vector_mu", [&] { return normal_lpdf(none, std::vector<double>{0.0, 1.0}, 1.0); });
  probe1("empty_y.empty_mu.bad_sigma", [&] { return normal_lpdf(none, none, -1.0); });
  probe1("empty_y.uniform_reversed_bounds", [&] { return uniform_lpdf(none, 2.0, 1.0); });
  probe1("scalar_y.uniform_reversed_bounds", [&] { return uniform_lpdf(1.5, 2.0, 1.0); });
  probe1("empty_n.binomial_bad_N", [&] { return binomial_lpmf(none_int, -1, 0.3); });
  probe1("empty_n.binomial_bad_theta", [&] { return binomial_lpmf(none_int, 5, 1.5); });
  probe1("empty_theta.binomial_n_above_N", [&] { return binomial_lpmf(7, 5, none); });
  probe1("empty_y.normal_lcdf_bad_sigma", [&] { return normal_lcdf(none, 0.0, -1.0); });
  probe1("empty_y.poisson_bad_lambda", [&] { return poisson_lpmf(none_int, -1.0); });
  probe1("empty_y.student_t_bad_nu", [&] { return student_t_lpdf(none, -1.0, 0.0, 1.0); });
  probe1("empty_y.gamma_lccdf_bad_beta", [&] { return gamma_lccdf(none, 2.0, -1.0); });
  probe1("empty_n.neg_binomial_2_bad_phi", [&] { return neg_binomial_2_lpmf(none_int, 1.0, -1.0); });
  probe1("empty_y.pareto_y_below_min", [&] { return pareto_lpdf(none, 2.0, 1.5); });
  probe1("scalar_y.pareto_y_below_min", [&] { return pareto_lpdf(1.0, 2.0, 1.5); });
  probe1("empty_y.wiener_tau_above_y", [&] { return wiener_lpdf(none, 1.5, 2.0, 0.5, 0.3); });
  const M x0(0, 2);
  V beta2(2);
  beta2 << 0.4, -0.3;
  V beta2_nan(2);
  beta2_nan << nan, 0.1;
  V cuts_bad(2);
  cuts_bad << 1.0, 0.0;
  V eta0(0);
  V theta_bad(2);
  theta_bad << 0.5, 0.6;
  probe1("empty_y.poisson_log_glm_nan_alpha", [&] { return poisson_log_glm_lpmf(none_int, x0, nan, beta2); });
  probe1("empty_y.bernoulli_logit_glm_nan_beta", [&] { return bernoulli_logit_glm_lpmf(none_int, x0, 0.0, beta2_nan); });
  probe1("empty_y.normal_id_glm_bad_sigma", [&] { return normal_id_glm_lpdf(V(0), x0, 0.0, beta2, -1.0); });
  probe1("empty_y.ordered_logistic_bad_cuts", [&] { return ordered_logistic_lpmf(none_int, eta0, cuts_bad); });
  probe1("empty_y.ordered_probit_bad_cuts", [&] { return ordered_probit_lpmf(none_int, eta0, cuts_bad); });
  probe1("empty_y.categorical_bad_theta", [&] { return categorical_lpmf(none_int, theta_bad); });
  M beta23(2, 3);
  beta23 << 0.1, -0.2, 0.3, 0.4, 0.0, -0.1;
  V alpha3_nan(3);
  alpha3_nan << nan, 0.0, 0.1;
  probe1("empty_y.binomial_logit_glm_nan_alpha", [&] { return binomial_logit_glm_lpmf(none_int, none_int, x0, nan, beta2); });
  probe1("empty_y.neg_binomial_2_log_glm_bad_phi", [&] { return neg_binomial_2_log_glm_lpmf(none_int, x0, 0.0, beta2, -1.0); });
  probe1("empty_y.neg_binomial_2_log_glm_nan_alpha", [&] { return neg_binomial_2_log_glm_lpmf(none_int, x0, nan, beta2, 1.0); });
  probe1("empty_y.categorical_logit_glm_nan_alpha", [&] { return categorical_logit_glm_lpmf(none_int, x0, alpha3_nan, beta23); });
  probe1("empty_y.ordered_logistic_glm_bad_cuts", [&] { return ordered_logistic_glm_lpmf(none_int, x0, beta2, cuts_bad); });
  probe1("empty_y.discrete_range_reversed", [&] { return discrete_range_lpmf(none_int, 5, 2); });
  probe1("scalar_y.discrete_range_reversed", [&] { return discrete_range_lpmf(3, 5, 2); });

  // propto: which terms are dropped, by which arguments are autodiff variables.
  const std::vector<double> yp = {1.0, 2.5, -0.5, 3.0};
  const std::vector<int> np = {2, 0, 5};
  const auto with_var = [](double v, const std::function<var(const var&)>& f) {
    const var x = v;
    const double out = f(x).val();
    recover_memory();
    return out;
  };
  probe1("propto.normal.all_data", [&] { return normal_lpdf<true>(yp, 0.7, 1.3); });
  probe1("propto.normal.var_mu", [&] { return with_var(0.7, [&](const var& m) { return normal_lpdf<true>(yp, m, 1.3); }); });
  probe1("propto.normal.var_sigma", [&] { return with_var(1.3, [&](const var& s) { return normal_lpdf<true>(yp, 0.7, s); }); });
  probe1("propto.normal.full", [&] { return normal_lpdf<false>(yp, 0.7, 1.3); });
  probe1("propto.poisson.var_lambda", [&] { return with_var(2.2, [&](const var& l) { return poisson_lpmf<true>(np, l); }); });
  probe1("propto.poisson.full", [&] { return poisson_lpmf<false>(np, 2.2); });
  probe1("propto.gamma.var_alpha", [&] { return with_var(2.5, [&](const var& a) { return gamma_lpdf<true>(1.7, a, 0.8); }); });
  probe1("propto.gamma.var_beta", [&] { return with_var(0.8, [&](const var& b) { return gamma_lpdf<true>(1.7, 2.5, b); }); });
  probe1("propto.gamma.var_y", [&] { return with_var(1.7, [&](const var& y) { return gamma_lpdf<true>(y, 2.5, 0.8); }); });
  probe1("propto.gamma.full", [&] { return gamma_lpdf<false>(1.7, 2.5, 0.8); });
  probe1("propto.binomial.var_theta", [&] { return with_var(0.3, [&](const var& t) { return binomial_lpmf<true>(4, 10, t); }); });
  probe1("propto.binomial.full", [&] { return binomial_lpmf<false>(4, 10, 0.3); });
  probe1("propto.student_t.var_mu", [&] { return with_var(0.2, [&](const var& m) { return student_t_lpdf<true>(1.1, 4.0, m, 1.5); }); });
  probe1("propto.student_t.full", [&] { return student_t_lpdf<false>(1.1, 4.0, 0.2, 1.5); });
  probe1("propto.std_normal.var_y", [&] { return with_var(0.4, [&](const var& y) { return std_normal_lpdf<true>(y); }); });
  probe1("propto.cauchy.var_sigma", [&] { return with_var(1.3, [&](const var& s) { return cauchy_lpdf<true>(0.5, -0.2, s); }); });
  probe1("propto.cauchy.var_mu", [&] { return with_var(-0.2, [&](const var& m) { return cauchy_lpdf<true>(0.5, m, 1.3); }); });
  probe1("propto.logistic.var_mu", [&] { return with_var(0.1, [&](const var& m) { return logistic_lpdf<true>(0.5, m, 0.8); }); });
  probe1("propto.exponential.var_y", [&] { return with_var(0.7, [&](const var& y) { return exponential_lpdf<true>(y, 1.9); }); });
  probe1("propto.exponential.var_beta", [&] { return with_var(1.9, [&](const var& b) { return exponential_lpdf<true>(0.7, b); }); });
  probe1("propto.inv_gamma.var_beta", [&] { return with_var(0.8, [&](const var& b) { return inv_gamma_lpdf<true>(1.2, 2.5, b); }); });
  probe1("propto.inv_gamma.var_alpha", [&] { return with_var(2.5, [&](const var& a) { return inv_gamma_lpdf<true>(1.2, a, 0.8); }); });
  probe1("propto.beta.var_a", [&] { return with_var(2.0, [&](const var& a) { return beta_lpdf<true>(0.3, a, 3.5); }); });
  probe1("propto.beta.var_y", [&] { return with_var(0.3, [&](const var& y) { return beta_lpdf<true>(y, 2.0, 3.5); }); });
  probe1("propto.lognormal.var_y", [&] { return with_var(1.4, [&](const var& y) { return lognormal_lpdf<true>(y, 0.2, 0.6); }); });
  probe1("propto.lognormal.var_sigma", [&] { return with_var(0.6, [&](const var& s) { return lognormal_lpdf<true>(1.4, 0.2, s); }); });
  probe1("propto.uniform.var_y", [&] { return with_var(0.4, [&](const var& y) { return uniform_lpdf<true>(y, 0.1, 1.5); }); });
  probe1("propto.uniform.var_alpha", [&] { return with_var(0.1, [&](const var& a) { return uniform_lpdf<true>(0.4, a, 1.5); }); });
  probe1("propto.poisson_log.var_alpha", [&] { return with_var(0.8, [&](const var& a) { return poisson_log_lpmf<true>(3, a); }); });
  probe1("propto.binomial_logit.var_alpha", [&] { return with_var(-0.4, [&](const var& a) { return binomial_logit_lpmf<true>(4, 10, a); }); });
  probe1("propto.neg_binomial_2.var_mu", [&] { return with_var(2.0, [&](const var& m) { return neg_binomial_2_lpmf<true>(3, m, 1.5); }); });
  probe1("propto.neg_binomial_2.var_phi", [&] { return with_var(1.5, [&](const var& p) { return neg_binomial_2_lpmf<true>(3, 2.0, p); }); });
  probe1("propto.neg_binomial_2_log.var_eta", [&] { return with_var(0.6, [&](const var& e) { return neg_binomial_2_log_lpmf<true>(3, e, 1.5); }); });
  probe1("propto.weibull.var_alpha", [&] { return with_var(1.5, [&](const var& a) { return weibull_lpdf<true>(1.2, a, 0.9); }); });
  probe1("propto.gumbel.var_mu", [&] { return with_var(1.0, [&](const var& m) { return gumbel_lpdf<true>(0.5, m, 2.0); }); });
  probe1("propto.bernoulli_logit.var_alpha", [&] { return with_var(0.3, [&](const var& a) { return bernoulli_logit_lpmf<true>(1, a); }); });

  // normal_id_glm with a vector x.
  V y(3);
  y << 1.0, -0.5, 2.0;
  V xv(3);
  xv << 0.3, 1.2, -0.7;
  V beta1(1);
  beta1 << 0.8;
  V beta3(3);
  beta3 << 0.8, -0.2, 0.4;
  const M xm = xv;
  probe1("normal_id_glm.vector_y.vector_x.beta1", [&] { return normal_id_glm_lpdf(y, xv, 0.5, beta1, 1.2); });
  probe1("normal_id_glm.vector_y.vector_x.beta3", [&] { return normal_id_glm_lpdf(y, xv, 0.5, beta3, 1.2); });
  probe1("normal_id_glm.vector_y.n_by_1_x.beta1", [&] { return normal_id_glm_lpdf(y, xm, 0.5, beta1, 1.2); });
  probe1("normal_id_glm.real_y.vector_x.beta1", [&] { return normal_id_glm_lpdf(1.0, xv, 0.5, beta1, 1.2); });
  probe1("normal_id_glm.real_y.vector_x.beta3", [&] { return normal_id_glm_lpdf(1.0, xv, 0.5, beta3, 1.2); });

  // gaussian_dlm_obs with a vector V against the diagonal matrix.
  M ydlm(2, 3);
  ydlm << 0.5, 1.0, -0.3, 1.2, 0.1, 0.4;
  M F(2, 2);
  F << 1.0, 0.3, 0.2, 1.0;
  M G(2, 2);
  G << 0.9, 0.1, 0.0, 0.8;
  V Vd(2);
  Vd << 0.5, 1.5;
  M W(2, 2);
  W << 0.3, 0.05, 0.05, 0.2;
  V m0(2);
  m0 << 0.0, 0.5;
  M C0(2, 2);
  C0 << 1.0, 0.2, 0.2, 2.0;
  const M Vm = Vd.asDiagonal();
  probe1("gaussian_dlm_obs.vector_V", [&] { return gaussian_dlm_obs_lpdf(ydlm, F, G, Vd, W, m0, C0); });
  probe1("gaussian_dlm_obs.diagonal_matrix_V", [&] { return gaussian_dlm_obs_lpdf(ydlm, F, G, Vm, W, m0, C0); });

  // normal_sufficient_lpdf: is s_squared the sum of squared deviations or the sample variance?
  const std::vector<double> ys = {1.0, 2.5, -0.5, 3.0};  // mean 1.5, sum of squares 7.5
  probe1("normal_sufficient.normal_lpdf_sum", [&] { return normal_lpdf(ys, 0.7, 1.3); });
  probe1("normal_sufficient.s_squared_7.5", [&] { return normal_sufficient_lpdf(1.5, 7.5, 4, 0.7, 1.3); });
  probe1("normal_sufficient.s_squared_2.5", [&] { return normal_sufficient_lpdf(1.5, 2.5, 4, 0.7, 1.3); });

  // Complex dot product: conjugated or not?
  Eigen::VectorXcd ca(2), cb(2);
  ca << std::complex<double>(1.0, 2.0), std::complex<double>(3.0, -1.0);
  cb << std::complex<double>(2.0, -1.0), std::complex<double>(0.5, 1.0);
  probe("dot_product.complex", [&] {
    const std::complex<double> d = dot_product(ca, cb);
    return std::vector<double>{d.real(), d.imag()};
  });
  const auto parts = [](const Eigen::VectorXcd& v) {
    std::vector<double> out;
    for (int i = 0; i < v.size(); ++i) {
      out.push_back(v(i).real());
      out.push_back(v(i).imag());
    }
    return out;
  };
  probe("dot_self.complex", [&] {
    const std::complex<double> d = dot_self(ca);
    return std::vector<double>{d.real(), d.imag()};
  });
  probe("multiply.complex_row_vector_vector", [&] {
    const Eigen::RowVectorXcd r = ca.transpose();
    const std::complex<double> d = multiply(r, cb);
    return std::vector<double>{d.real(), d.imag()};
  });
  Eigen::MatrixXcd cm(2, 2), cn(2, 2);
  cm << std::complex<double>(1.0, 2.0), std::complex<double>(0.0, -1.0),
      std::complex<double>(3.0, -1.0), std::complex<double>(2.0, 0.5);
  cn << std::complex<double>(2.0, -1.0), std::complex<double>(1.0, 1.0),
      std::complex<double>(0.5, 1.0), std::complex<double>(-1.0, 2.0);
  probe("columns_dot_product.complex", [&] {
    const Eigen::VectorXcd v = columns_dot_product(cm, cn).transpose();
    return parts(v);
  });
  probe("rows_dot_product.complex", [&] {
    const Eigen::VectorXcd v = rows_dot_product(cm, cn);
    return parts(v);
  });
  probe("columns_dot_self.complex", [&] {
    const Eigen::VectorXcd v = columns_dot_self(cm).transpose();
    return parts(v);
  });
  probe("rows_dot_self.complex", [&] {
    const Eigen::VectorXcd v = rows_dot_self(cm);
    return parts(v);
  });
  probe("multiply.complex_matrix_vector", [&] {
    const Eigen::VectorXcd v = multiply(cm, cb);
    return parts(v);
  });
  probe("multiply.complex_row_vector_matrix", [&] {
    const Eigen::RowVectorXcd r = ca.transpose();
    const Eigen::VectorXcd v = multiply(r, cm).transpose();
    return parts(v);
  });

  // Order of the canonical partial correlations.
  V cpcs(6);
  cpcs << 0.3, -0.6, 0.15, 0.45, -0.2, 0.7;
  probe("read_corr_L.k4", [&] { return lower_rows(read_corr_L(cpcs, 4)); });
  V xc(6);
  xc << 0.3, -1.2, 0.8, 0.05, 2.0, -0.4;
  probe("corr_matrix_constrain.k4", [&] { return lower_rows(corr_matrix_constrain(xc, 4)); });
  probe("cholesky_corr_constrain.k4", [&] { return lower_rows(cholesky_corr_constrain(xc, 4)); });

  // Edge cases of the newer densities.
  probe1("rayleigh_lpdf.y0", [&] { return rayleigh_lpdf(0.0, 1.0); });
  probe1("gumbel_lpdf.beta_inf", [&] { return gumbel_lpdf(0.5, 0.0, inf); });
  probe1("loglogistic_lpdf.y0", [&] { return loglogistic_lpdf(0.0, 1.0, 2.0); });
  probe1("loglogistic_lpdf.y_negative", [&] { return loglogistic_lpdf(-0.5, 1.0, 2.0); });
  probe1("loglogistic_cdf.interior", [&] { return loglogistic_cdf(1.3, 0.9, 2.5); });
  probe1("rayleigh_lcdf.y0", [&] { return rayleigh_lcdf(0.0, 1.0); });
  probe1("rayleigh_lccdf.y0", [&] { return rayleigh_lccdf(0.0, 1.0); });
  probe1("loglogistic_cdf.y0", [&] { return loglogistic_cdf(0.0, 1.0, 2.0); });
  probe1("loglogistic_cdf.y_negative", [&] { return loglogistic_cdf(-0.5, 1.0, 2.0); });

  // lkj_cov_lpdf at an interior point.
  M sigma(3, 3);
  sigma << 2.0, 0.4, -0.3, 0.4, 1.5, 0.2, -0.3, 0.2, 1.2;
  V lmu(3);
  lmu << 0.1, -0.2, 0.3;
  V lsd(3);
  lsd << 1.0, 0.8, 1.2;
  probe1("lkj_cov_lpdf.interior", [&] { return lkj_cov_lpdf(sigma, lmu, lsd, 1.5); });

  // Special numerics: values, and the value with its gradient as value;d1;d2.
  const auto grad1 = [](double a, const std::function<var(const var&)>& f) {
    const var x = a;
    var r = f(x);
    r.grad();
    std::vector<double> out{r.val(), x.adj()};
    recover_memory();
    return out;
  };
  const auto grad2 = [](double a, double b, const std::function<var(const var&, const var&)>& f) {
    const var x = a;
    const var y = b;
    var r = f(x, y);
    r.grad();
    std::vector<double> out{r.val(), x.adj(), y.adj()};
    recover_memory();
    return out;
  };
  for (const auto& pr : std::vector<std::pair<int, double>>{{0, 0.5}, {1, 3.7}, {2, 30.0}, {5, 10.0}, {3, 0.01}, {0, -2.5}, {4, 60.0}}) {
    const int v = pr.first;
    const double z = pr.second;
    const std::string at = std::to_string(v) + "_" + std::to_string(z);
    probe1("bessel_first_kind." + at, [&] { return bessel_first_kind(v, z); });
    probe1("modified_bessel_first_kind." + at, [&] { return modified_bessel_first_kind(v, z); });
    if (z > 0) {
      probe1("bessel_second_kind." + at, [&] { return bessel_second_kind(v, z); });
      probe1("modified_bessel_second_kind." + at, [&] { return modified_bessel_second_kind(v, z); });
    }
  }
  probe("bessel_first_kind.grad_1_3.7", [&] { return grad1(3.7, [](const var& z) { return bessel_first_kind(1, z); }); });
  probe("bessel_second_kind.grad_2_5.5", [&] { return grad1(5.5, [](const var& z) { return bessel_second_kind(2, z); }); });
  probe("modified_bessel_first_kind.grad_1_3.7", [&] { return grad1(3.7, [](const var& z) { return modified_bessel_first_kind(1, z); }); });
  probe("modified_bessel_second_kind.grad_2_1.5", [&] { return grad1(1.5, [](const var& z) { return modified_bessel_second_kind(2, z); }); });
  for (const auto& pr : std::vector<std::pair<double, double>>{{0.0, 0.5}, {1.5, 3.7}, {2.3, 50.0}, {0.5, 200.0}, {7.0, 0.2}, {0.25, 1e-3}}) {
    const double v = pr.first;
    const double z = pr.second;
    probe1("log_modified_bessel_first_kind." + std::to_string(v) + "_" + std::to_string(z), [&] { return log_modified_bessel_first_kind(v, z); });
  }
  probe("log_modified_bessel_first_kind.grad_1.5_3.7", [&] { return grad2(1.5, 3.7, [](const var& v, const var& z) { return log_modified_bessel_first_kind(v, z); }); });
  for (const auto& pr : std::vector<std::pair<double, double>>{{0.5, 0.3}, {1.2, 2.0}, {-0.7, 0.9}, {3.0, 0.5}, {0.1, 10.0}, {6.0, 0.99}, {0.0, 1.0}}) {
    const double h = pr.first;
    const double a = pr.second;
    probe1("owens_t." + std::to_string(h) + "_" + std::to_string(a), [&] { return owens_t(h, a); });
  }
  probe("owens_t.grad_1.2_2.0", [&] { return grad2(1.2, 2.0, [](const var& h, const var& a) { return owens_t(h, a); }); });
  for (double x : {0.5, 5.0, -0.3, 100.0, -0.36, 1e-8}) {
    probe1("lambert_w0." + std::to_string(x), [&] { return lambert_w0(x); });
  }
  for (double x : {-0.3, -0.01, -0.36, -1e-5}) {
    probe1("lambert_wm1." + std::to_string(x), [&] { return lambert_wm1(x); });
  }
  probe("lambert_w0.grad_0.5", [&] { return grad1(0.5, [](const var& x) { return lambert_w0(x); }); });
  probe("lambert_wm1.grad_-0.3", [&] { return grad1(-0.3, [](const var& x) { return lambert_wm1(x); }); });
  probe1("hypergeometric_2F1.1_2_3.5_0.4", [&] { return hypergeometric_2F1(1.0, 2.0, 3.5, 0.4); });
  probe1("hypergeometric_2F1.0.5_1.5_2_-0.6", [&] { return hypergeometric_2F1(0.5, 1.5, 2.0, -0.6); });
  probe1("hypergeometric_2F1.1.2_-0.7_2.1_0.9", [&] { return hypergeometric_2F1(1.2, -0.7, 2.1, 0.9); });
  probe("hypergeometric_2F1.grad_z_1_2_3.5_0.4", [&] { return grad1(0.4, [](const var& z) { return hypergeometric_2F1(1.0, 2.0, 3.5, z); }); });
  V a3(3);
  a3 << 1.0, 1.5, 2.0;
  V b2(2);
  b2 << 2.5, 3.0;
  probe1("hypergeometric_3F2.1_1.5_2__2.5_3__0.3", [&] { return hypergeometric_3F2(a3, b2, 0.3); });
  V pa(2);
  pa << 0.5, 1.2;
  V pb(2);
  pb << 1.7, 2.4;
  probe1("hypergeometric_pFq.0.5_1.2__1.7_2.4__0.8", [&] { return hypergeometric_pFq(pa, pb, 0.8); });
  probe1("hypergeometric_pFq.0.5_1.2__1.7_2.4__-3.5", [&] { return hypergeometric_pFq(pa, pb, -3.5); });
  return 0;
}
