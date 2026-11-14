// Reference values and gradients from Stan Math for the deprecated functional solvers of the
// Stan language: integrate_ode_rk45, integrate_ode_adams, integrate_ode_bdf, algebra_solver,
// algebra_solver_newton, and integrate_1d. Each solution is reduced to the weighted sum
// sum_k (1 + k / 10) r_k over its entries (time major for the ODEs), so one gradient checks every
// entry. The solvers run at tight tolerances, so the values agree to about 1e-8.
//
// Each output line is
//   function,ints,reals,value,gradient
// as in stan_reference_extra.cpp.

#include <stan/math.hpp>

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

}  // namespace

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
  return 0;
}
