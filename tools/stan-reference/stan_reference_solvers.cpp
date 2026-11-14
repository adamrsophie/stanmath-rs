// Reference values and gradients from Stan Math for the deprecated functional solvers of the
// Stan language: integrate_ode_rk45, integrate_ode_adams, integrate_ode_bdf, algebra_solver,
// algebra_solver_newton, and integrate_1d. Each solution is reduced to the weighted sum
// sum_k (1 + k / 10) r_k over its entries (time major for the ODEs), so one gradient checks every
// entry. The solvers run at tight tolerances, so the values agree to about 1e-8.
//
// Each output line is
//   function,ints,reals,value,gradient
// as in stan_reference_extra.cpp.

// The Laplace functions live in the mixed mode header, as in stanc's models.
#include <stan/math/mix.hpp>

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

template <typename F>
void emit(const std::string& name, const std::vector<double>& x, F f) {
  Vars xv(x.begin(), x.end());
  std::cout << name << ",," << join(x) << ',';
  try {
    var s = f(xv);
    s.grad();
    std::vector<double> g;
    for (auto& v : xv) g.push_back(v.adj());
    std::cout << std::setprecision(17) << s.val() << ',' << join(g) << '\n';
  } catch (const std::exception& e) {
    std::cout << "error,\n";
  }
  stan::math::recover_memory();
}

var wsum(const std::vector<std::vector<var>>& r) {
  var s = 0;
  int k = 0;
  for (const auto& row : r)
    for (const auto& v : row) s += (1.0 + 0.1 * k++) * v;
  return s;
}

template <typename E>
var wsum(const E& e) {
  var s = 0;
  for (int k = 0; k < e.size(); ++k) s += (1.0 + 0.1 * k) * e(k);
  return s;
}

// Lotka Volterra: y0' = t0 y0 - t1 y0 y1, y1' = -t2 y1 + t1 y0 y1.
struct Lv {
  template <typename T0, typename T1, typename T2>
  std::vector<stan::return_type_t<T1, T2>> operator()(const T0&, const std::vector<T1>& y,
                                                       const std::vector<T2>& th,
                                                       const std::vector<double>&,
                                                       const std::vector<int>&,
                                                       std::ostream*) const {
    return {th[0] * y[0] - th[1] * y[0] * y[1], -th[2] * y[1] + th[1] * y[0] * y[1]};
  }
};

// x0^2 + x1 - t0 = 0, x0 - x1^3 + t1 = 0.
struct Sys {
  template <typename T0, typename T1>
  Eigen::Matrix<stan::return_type_t<T0, T1>, -1, 1> operator()(
      const Eigen::Matrix<T0, -1, 1>& x, const Eigen::Matrix<T1, -1, 1>& th,
      const std::vector<double>&, const std::vector<int>&, std::ostream*) const {
    Eigen::Matrix<stan::return_type_t<T0, T1>, -1, 1> r(2);
    r << x(0) * x(0) + x(1) - th(0), x(0) - x(1) * x(1) * x(1) + th(1);
    return r;
  }
};

// exp(-t0 x) cos(t1 x).
struct Damped {
  template <typename T0, typename T1, typename T2>
  stan::return_type_t<T0, T1, T2> operator()(const T0& x, const T1&, const std::vector<T2>& th,
                                             const std::vector<double>&,
                                             const std::vector<int>&, std::ostream*) const {
    return stan::math::exp(-th[0] * x) * stan::math::cos(th[1] * x);
  }
};

// map_rect job: [phi0 theta0 + x_r0, phi1 exp(theta1) x_i0].
struct Job {
  template <typename T1, typename T2>
  Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> operator()(
      const Eigen::Matrix<T1, -1, 1>& phi, const Eigen::Matrix<T2, -1, 1>& th,
      const std::vector<double>& xr, const std::vector<int>& xi, std::ostream*) const {
    Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> r(2);
    r << phi(0) * th(0) + xr[0], phi(1) * stan::math::exp(th(1)) * xi[0];
    return r;
  }
};

// The variadic forms take Eigen vectors and pass the parameters as trailing arguments.
struct LvV {
  template <typename T0, typename T1, typename T2>
  Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> operator()(const T0&, const T1& y,
                                                              std::ostream*, const T2& th) const {
    Eigen::Matrix<stan::return_type_t<T1, T2>, -1, 1> r(2);
    r << th(0) * y(0) - th(1) * y(0) * y(1), -th(2) * y(1) + th(1) * y(0) * y(1);
    return r;
  }
};

struct SysV {
  template <typename T0, typename T1>
  Eigen::Matrix<stan::return_type_t<T0, T1>, -1, 1> operator()(
      const Eigen::Matrix<T0, -1, 1>& x, std::ostream*, const Eigen::Matrix<T1, -1, 1>& th) const {
    Eigen::Matrix<stan::return_type_t<T0, T1>, -1, 1> r(2);
    r << x(0) * x(0) + x(1) - th(0), x(0) - x(1) * x(1) * x(1) + th(1);
    return r;
  }
};

// Fixed point map G(x) = [cos(x1) / 2 + th0, 0.3 sin(x0) th1 + 0.2], a contraction.
struct Gfp {
  template <typename T0, typename T1>
  Eigen::Matrix<stan::return_type_t<T0, T1>, -1, 1> operator()(
      const Eigen::Matrix<T0, -1, 1>& x, const Eigen::Matrix<T1, -1, 1>& th,
      const std::vector<double>&, const std::vector<int>&, std::ostream*) const {
    Eigen::Matrix<stan::return_type_t<T0, T1>, -1, 1> r(2);
    r << 0.5 * stan::math::cos(x(1)) + th(0), 0.3 * stan::math::sin(x(0)) * th(1) + 0.2;
    return r;
  }
};

// DAE residual: y0' + k y0 = 0, y1 - 2 y0 = 0, so y0 = c exp(-k t) and y1 = 2 y0.
struct Res {
  template <typename T0, typename T1, typename T2, typename T3>
  Eigen::Matrix<stan::return_type_t<T1, T2, T3>, -1, 1> operator()(
      const T0&, const Eigen::Matrix<T1, -1, 1>& y, const Eigen::Matrix<T2, -1, 1>& yp,
      std::ostream*, const T3& k) const {
    Eigen::Matrix<stan::return_type_t<T1, T2, T3>, -1, 1> r(2);
    r << yp(0) + k * y(0), y(1) - 2 * y(0);
    return r;
  }
};

// reduce_sum partial sum: normal_lpdf(slice | mu, sigma).
struct Part {
  template <typename T1, typename T2>
  stan::return_type_t<T1, T2> operator()(const std::vector<double>& s, int, int, std::ostream*,
                                         const T1& mu, const T2& sigma) const {
    return stan::math::normal_lpdf(s, mu, sigma);
  }
};

// Laplace covariance: gp_exp_quad_cov(x, alpha, rho) at x = [0, 0.5, 1.2], plus 1e-8 on the
// diagonal.
struct Kf {
  template <typename T1, typename T2>
  Eigen::Matrix<stan::return_type_t<T1, T2>, -1, -1> operator()(const T1& a, const T2& rho,
                                                               std::ostream*) const {
    std::vector<double> x = {0.0, 0.5, 1.2};
    return stan::math::add_diag(stan::math::gp_exp_quad_cov(x, a, rho), 1e-8);
  }
};

// Laplace likelihood: y ~ poisson_log(theta + eta[1]).
struct Ll {
  template <typename T0, typename T1>
  stan::return_type_t<T0, T1> operator()(const T0& theta,
                                         const std::vector<int>& y, const T1& eta,
                                         std::ostream*) const {
    return stan::math::poisson_log_lpmf(y, stan::math::add(theta, eta));
  }
};

var wsumv(const std::vector<Eigen::Matrix<var, -1, 1>>& r) {
  var s = 0;
  int k = 0;
  for (const auto& row : r)
    for (int i = 0; i < row.size(); ++i) s += (1.0 + 0.1 * k++) * row(i);
  return s;
}

}  // namespace

STAN_REGISTER_MAP_RECT(1, Job)

int main() {
  using namespace stan::math;
  std::cout << "# Written by tools/stan-reference/stan_reference_solvers.cpp\n";
  const std::vector<double> ts = {0.5, 1.0, 2.0};
  const std::vector<double> ode = {1.0, 0.5, 1.1, 0.4, 0.6};
  const std::vector<double> xr;
  const std::vector<int> xi;
  auto y0 = [](const Vars& v) { return Vars(v.begin(), v.begin() + 2); };
  auto th = [](const Vars& v) { return Vars(v.begin() + 2, v.begin() + 5); };
  emit("integrate_ode_rk45", ode, [&](const Vars& v) {
    return wsum(integrate_ode_rk45(Lv(), y0(v), 0.0, ts, th(v), xr, xi, nullptr, 1e-10, 1e-10, 1000000)); });
  emit("integrate_ode_adams", ode, [&](const Vars& v) {
    return wsum(integrate_ode_adams(Lv(), y0(v), 0.0, ts, th(v), xr, xi, nullptr, 1e-10, 1e-10, 1000000)); });
  emit("integrate_ode_bdf", ode, [&](const Vars& v) {
    return wsum(integrate_ode_bdf(Lv(), y0(v), 0.0, ts, th(v), xr, xi, nullptr, 1e-10, 1e-10, 1000000)); });
  Eigen::VectorXd guess(2);
  guess << 1.0, 1.0;
  auto theta = [](const Vars& v) { Eigen::Matrix<var, -1, 1> t(2); t << v[0], v[1]; return t; };
  emit("algebra_solver", {2.0, 0.5}, [&](const Vars& v) {
    return wsum(algebra_solver(Sys(), guess, theta(v), xr, xi, nullptr, 1e-12, 1e-12, 1000)); });
  emit("algebra_solver_newton", {2.0, 0.5}, [&](const Vars& v) {
    return wsum(algebra_solver_newton(Sys(), guess, theta(v), xr, xi, nullptr, 1e-12, 1e-12, 1000)); });
  emit("integrate_1d", {2.5, 0.7, 1.3}, [&](const Vars& v) {
    return integrate_1d(Damped(), var(0.0), v[0], Vars{v[1], v[2]}, xr, xi, nullptr, 1e-10); });
  // stanc compiles integrate_ode to integrate_ode_rk45 at its default tolerances (1e-6); the
  // sensitivities then agree to about 1e-4.
  emit("integrate_ode", ode, [&](const Vars& v) {
    return wsum(integrate_ode_rk45(Lv(), y0(v), 0.0, ts, th(v), xr, xi, nullptr)); });
  // Two jobs; reals are phi (2), then theta for each job (2 each).
  emit("map_rect", {1.3, 0.7, 0.4, -0.2, 1.1, 0.5}, [&](const Vars& v) {
    Eigen::Matrix<var, -1, 1> phi(2);
    phi << v[0], v[1];
    std::vector<Eigen::Matrix<var, -1, 1>> jobs(2, Eigen::Matrix<var, -1, 1>(2));
    jobs[0] << v[2], v[3];
    jobs[1] << v[4], v[5];
    std::vector<std::vector<double>> jr = {{0.5}, {0.25}};
    std::vector<std::vector<int>> ji = {{2}, {3}};
    return wsum(map_rect<1, Job>(phi, jobs, jr, ji, nullptr)); });
  // Variadic ODE solvers. The default tolerances (1e-6) agree to about 1e-4; the _tol forms and
  // the adjoint solver run at 1e-10.
  auto yv = [](const Vars& v) { Eigen::Matrix<var, -1, 1> r(2); r << v[0], v[1]; return r; };
  auto tv = [](const Vars& v) { Eigen::Matrix<var, -1, 1> r(3); r << v[2], v[3], v[4]; return r; };
  emit("ode_rk45", ode, [&](const Vars& v) { return wsumv(ode_rk45(LvV(), yv(v), 0.0, ts, nullptr, tv(v))); });
  emit("ode_ckrk", ode, [&](const Vars& v) { return wsumv(ode_ckrk(LvV(), yv(v), 0.0, ts, nullptr, tv(v))); });
  emit("ode_adams", ode, [&](const Vars& v) { return wsumv(ode_adams(LvV(), yv(v), 0.0, ts, nullptr, tv(v))); });
  emit("ode_bdf", ode, [&](const Vars& v) { return wsumv(ode_bdf(LvV(), yv(v), 0.0, ts, nullptr, tv(v))); });
  emit("ode_rk45_tol", ode, [&](const Vars& v) {
    return wsumv(ode_rk45_tol(LvV(), yv(v), 0.0, ts, 1e-10, 1e-10, 1000000, nullptr, tv(v))); });
  emit("ode_ckrk_tol", ode, [&](const Vars& v) {
    return wsumv(ode_ckrk_tol(LvV(), yv(v), 0.0, ts, 1e-10, 1e-10, 1000000, nullptr, tv(v))); });
  emit("ode_adams_tol", ode, [&](const Vars& v) {
    return wsumv(ode_adams_tol(LvV(), yv(v), 0.0, ts, 1e-10, 1e-10, 1000000, nullptr, tv(v))); });
  emit("ode_bdf_tol", ode, [&](const Vars& v) {
    return wsumv(ode_bdf_tol(LvV(), yv(v), 0.0, ts, 1e-10, 1e-10, 1000000, nullptr, tv(v))); });
  Eigen::VectorXd atol(2);
  atol << 1e-10, 1e-10;
  emit("ode_adjoint_tol_ctl", ode, [&](const Vars& v) {
    return wsumv(ode_adjoint_tol_ctl(LvV(), yv(v), 0.0, ts, 1e-10, atol, 1e-10, atol, 1e-10, 1e-10,
                                     1000000, 150, 1, 2, 2, nullptr, tv(v))); });
  // DAE with consistent initial values y = [1, 2], y' = [-k, -2k]; the parameter is k.
  auto dy = [](const Vars& v) { Eigen::Matrix<var, -1, 1> r(2); r << 1.0, 2.0; return r; };
  auto dyp = [](const Vars& v) { Eigen::Matrix<var, -1, 1> r(2); r << -v[0], -2 * v[0]; return r; };
  emit("dae", {0.7}, [&](const Vars& v) { return wsumv(dae(Res(), dy(v), dyp(v), 0.0, ts, nullptr, v[0])); });
  emit("dae_tol", {0.7}, [&](const Vars& v) {
    return wsumv(dae_tol(Res(), dy(v), dyp(v), 0.0, ts, 1e-10, 1e-10, 1000000, nullptr, v[0])); });
  // Variadic algebraic solvers; the default tolerances agree to about 1e-6.
  emit("solve_newton", {2.0, 0.5}, [&](const Vars& v) { return wsum(solve_newton(SysV(), guess, nullptr, theta(v))); });
  emit("solve_powell", {2.0, 0.5}, [&](const Vars& v) { return wsum(solve_powell(SysV(), guess, nullptr, theta(v))); });
  emit("solve_newton_tol", {2.0, 0.5}, [&](const Vars& v) {
    return wsum(solve_newton_tol(SysV(), guess, 1e-12, 1e-12, 1000, nullptr, theta(v))); });
  emit("solve_powell_tol", {2.0, 0.5}, [&](const Vars& v) {
    return wsum(solve_powell_tol(SysV(), guess, 1e-12, 1e-12, 1000, nullptr, theta(v))); });
  // Stan Math's internal fixed point solver (not in the Stan language), unit scales.
  emit("algebra_solver_fp", {0.4, 1.3}, [&](const Vars& v) {
    Eigen::VectorXd g0 = Eigen::VectorXd::Zero(2);
    return wsum(algebra_solver_fp(Gfp(), g0, theta(v), std::vector<double>{}, std::vector<int>{},
                                  std::vector<double>{1, 1}, std::vector<double>{1, 1}, nullptr,
                                  1e-12, 200)); });
  const std::vector<double> rs = {0.1, 0.4, -0.3, 1.2, 0.8};
  emit("reduce_sum", {0.3, 1.4}, [&](const Vars& v) { return reduce_sum<Part>(rs, 2, nullptr, v[0], v[1]); });
  emit("reduce_sum_static", {0.3, 1.4}, [&](const Vars& v) {
    return reduce_sum_static<Part>(rs, 2, nullptr, v[0], v[1]); });
  // Laplace marginals: phi = [alpha, rho] (and the offset or dispersion last), y_index one based.
  const std::vector<int> yp = {1, 0, 3}, yb = {1, 0, 1}, idx = {1, 2, 3};
  auto opts = generate_laplace_options(3);
  auto mv = [](const var& c) { Eigen::Matrix<var, -1, 1> r(3); r << c, c, c; return r; };
  emit("laplace_marginal_poisson_log_lpmf", {1.2, 0.7, 0.3}, [&](const Vars& v) {
    return laplace_marginal_poisson_log_lpmf<false>(yp, idx, mv(v[2]), 1, Kf(), std::forward_as_tuple(v[0], v[1]), nullptr); });
  emit("laplace_marginal_poisson_log_lupmf", {1.2, 0.7, 0.3}, [&](const Vars& v) {
    return laplace_marginal_poisson_log_lpmf<true>(yp, idx, mv(v[2]), 1, Kf(), std::forward_as_tuple(v[0], v[1]), nullptr); });
  emit("laplace_marginal_tol_poisson_log_lpmf", {1.2, 0.7, 0.3}, [&](const Vars& v) {
    return laplace_marginal_tol_poisson_log_lpmf<false>(yp, idx, mv(v[2]), 1, Kf(), std::forward_as_tuple(v[0], v[1]), opts, nullptr); });
  emit("laplace_marginal_bernoulli_logit_lpmf", {1.2, 0.7, 0.3}, [&](const Vars& v) {
    return laplace_marginal_bernoulli_logit_lpmf<false>(yb, idx, mv(v[2]), 1, Kf(), std::forward_as_tuple(v[0], v[1]), nullptr); });
  emit("laplace_marginal_bernoulli_logit_lupmf", {1.2, 0.7, 0.3}, [&](const Vars& v) {
    return laplace_marginal_bernoulli_logit_lpmf<true>(yb, idx, mv(v[2]), 1, Kf(), std::forward_as_tuple(v[0], v[1]), nullptr); });
  emit("laplace_marginal_tol_bernoulli_logit_lpmf", {1.2, 0.7, 0.3}, [&](const Vars& v) {
    return laplace_marginal_tol_bernoulli_logit_lpmf<false>(yb, idx, mv(v[2]), 1, Kf(), std::forward_as_tuple(v[0], v[1]), opts, nullptr); });
  emit("laplace_marginal_neg_binomial_2_log_lpmf", {1.2, 0.7, 2.5}, [&](const Vars& v) {
    return laplace_marginal_neg_binomial_2_log_lpmf<false>(yp, idx, v[2], mv(0.0), 1, Kf(), std::forward_as_tuple(v[0], v[1]), nullptr); });
  emit("laplace_marginal_neg_binomial_2_log_lupmf", {1.2, 0.7, 2.5}, [&](const Vars& v) {
    return laplace_marginal_neg_binomial_2_log_lpmf<true>(yp, idx, v[2], mv(0.0), 1, Kf(), std::forward_as_tuple(v[0], v[1]), nullptr); });
  emit("laplace_marginal_tol_neg_binomial_2_log_lpmf", {1.2, 0.7, 2.5}, [&](const Vars& v) {
    return laplace_marginal_tol_neg_binomial_2_log_lpmf<false>(yp, idx, v[2], mv(0.0), 1, Kf(), std::forward_as_tuple(v[0], v[1]), opts, nullptr); });
  emit("laplace_marginal", {1.2, 0.7, 0.3}, [&](const Vars& v) {
    return laplace_marginal(Ll(), std::forward_as_tuple(yp, v[2]), 1, Kf(), std::forward_as_tuple(v[0], v[1]), nullptr); });
  emit("laplace_marginal_tol", {1.2, 0.7, 0.3}, [&](const Vars& v) {
    return laplace_marginal_tol(Ll(), std::forward_as_tuple(yp, v[2]), 1, Kf(), std::forward_as_tuple(v[0], v[1]), opts, nullptr); });
  // generate_laplace_options: each field of the tuple as its own row.
  Eigen::VectorXd t0(2);
  t0 << 1.5, 0.2;
  for (const auto& o : {generate_laplace_options(3), generate_laplace_options(t0)}) {
    const Eigen::VectorXd& init = std::get<0>(o);
    double s = 0;
    for (int k = 0; k < init.size(); ++k) s += (1.0 + 0.1 * k) * init(k);
    std::cout << std::setprecision(17) << "generate_laplace_options," << init.size() << ",,"
              << s << ',' << std::get<1>(o) << ';' << std::get<2>(o) << ';' << std::get<3>(o)
              << ';' << std::get<4>(o) << ';' << std::get<5>(o) << '\n';
  }
  return 0;
}
