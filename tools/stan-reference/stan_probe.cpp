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

  // lkj_cov_lpdf at an interior point.
  M sigma(3, 3);
  sigma << 2.0, 0.4, -0.3, 0.4, 1.5, 0.2, -0.3, 0.2, 1.2;
  V lmu(3);
  lmu << 0.1, -0.2, 0.3;
  V lsd(3);
  lsd << 1.0, 0.8, 1.2;
  probe1("lkj_cov_lpdf.interior", [&] { return lkj_cov_lpdf(sigma, lmu, lsd, 1.5); });
  return 0;
}
