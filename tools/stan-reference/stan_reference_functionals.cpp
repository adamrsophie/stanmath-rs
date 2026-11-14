// Values from Stan Math's C++ derivative functionals (gradient, hessian, grad_hessian,
// hessian_times_vector, gradient_dot_vector, grad_tr_mat_times_hessian, jacobian, derivative,
// partial_derivative, finite_diff_gradient, finite_diff_grad_hessian), for comparison with
// stanmath-rs's drivers. Each output line is
//   function,value;value;...
// with matrices column major and the grad_hessian slices in order of the differentiated input.
//
//   f(x) = x0^2 x1 + exp(x0 x1) sin(x2) + x2^3 / x1 at x = [0.7, 1.3, -0.4]
//   g(x) = [x0 x1, sin(x0) + x2^2, exp(x1) x2]
//   h(t) = t^3 sin(t) at t = 0.9

#include <stan/math/mix.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {

struct F {
  template <typename T>
  T operator()(const Eigen::Matrix<T, -1, 1>& x) const {
    using stan::math::exp;
    using stan::math::sin;
    return x(0) * x(0) * x(1) + exp(x(0) * x(1)) * sin(x(2)) + x(2) * x(2) * x(2) / x(1);
  }
};

struct G {
  template <typename T>
  Eigen::Matrix<T, -1, 1> operator()(const Eigen::Matrix<T, -1, 1>& x) const {
    using stan::math::exp;
    using stan::math::sin;
    Eigen::Matrix<T, -1, 1> r(3);
    r << x(0) * x(1), sin(x(0)) + x(2) * x(2), exp(x(1)) * x(2);
    return r;
  }
};

struct H {
  template <typename T>
  T operator()(const T& t) const {
    return t * t * t * stan::math::sin(t);
  }
};

void out(const std::string& name, const std::vector<double>& v) {
  std::cout << name << ',';
  for (std::size_t i = 0; i < v.size(); ++i) std::cout << (i ? ";" : "") << v[i];
  std::cout << '\n';
}

std::vector<double> flat(double s, const MatrixXd& m) {
  std::vector<double> r = {s};
  r.insert(r.end(), m.data(), m.data() + m.size());
  return r;
}

}  // namespace

int main() {
  using namespace stan::math;
  std::cout << std::setprecision(17);
  VectorXd x(3), v(3);
  x << 0.7, 1.3, -0.4;
  v << 0.5, -1.0, 2.0;
  MatrixXd m(3, 3);
  m << 1.0, 0.2, -0.3, 0.4, 2.0, 0.1, -0.5, 0.6, 1.5;
  double fx;
  VectorXd g;
  MatrixXd hm;
  gradient(F(), x, fx, g);
  out("gradient", flat(fx, g));
  hessian(F(), x, fx, g, hm);
  out("hessian", flat(fx, hm));
  std::vector<MatrixXd> gh;
  grad_hessian(F(), x, fx, hm, gh);
  std::vector<double> r = flat(fx, hm);
  for (const auto& s : gh) r.insert(r.end(), s.data(), s.data() + s.size());
  out("grad_hessian", r);
  VectorXd hv;
  hessian_times_vector(F(), x, v, fx, hv);
  out("hessian_times_vector", flat(fx, hv));
  double gv;
  gradient_dot_vector(F(), x, v, fx, gv);
  out("gradient_dot_vector", {fx, gv});
  VectorXd gtr;
  grad_tr_mat_times_hessian(F(), x, m, gtr);
  out("grad_tr_mat_times_hessian", flat(0.0, gtr));
  VectorXd jx;
  MatrixXd jm;
  jacobian(G(), x, jx, jm);
  r = flat(0.0, jx);
  r.insert(r.end(), jm.data(), jm.data() + jm.size());
  out("jacobian", r);
  double d;
  derivative(H(), 0.9, fx, d);
  out("derivative", {fx, d});
  partial_derivative(F(), x, 2, fx, d);
  out("partial_derivative", {fx, d});
  finite_diff_gradient(F(), x, fx, g);
  out("finite_diff_gradient", flat(fx, g));
  finite_diff_grad_hessian(F(), x, fx, hm, gh);
  r = flat(fx, hm);
  for (const auto& s : gh) r.insert(r.end(), s.data(), s.data() + s.size());
  out("finite_diff_grad_hessian", r);
  return 0;
}
