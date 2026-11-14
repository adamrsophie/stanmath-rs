// Reference values and gradients from Stan Math for the scalar signatures of the Stan language
// that stan_reference_scalar.cpp does not reach: constants, functions with an int argument, three
// and four argument functions, the scalar bound transforms, and the 6 to 9 argument Wiener forms.
//
//   make STAN_MATH=/path/to/stan_math/ stan_reference_scalar_extra
//   ./stan_reference_scalar_extra > ../../crates/stanmath-rs/tests/data/stan_reference_scalar_extra.csv
//
// Each output line is
//   function,ints,reals,value,gradient
// with the gradient taken with respect to every real argument in order. A name with a suffix
// (_int, _6, ...) picks one overload. A case that Stan rejects prints "error" as its value.

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
  std::cout << "# Written by tools/stan-reference/stan_reference_scalar_extra.cpp\n";
  using I = std::vector<int>;
  using D = std::vector<double>;
  using Cases = std::vector<std::pair<I, D>>;
  auto c = [](var v) { return v; };
  auto k = [](double v) { return var(v); };
  // Constants.
  continuous("e", {{}}, [&](const Vars&) { return k(e()); });
  continuous("pi", {{}}, [&](const Vars&) { return k(pi()); });
  continuous("sqrt2", {{}}, [&](const Vars&) { return k(sqrt2()); });
  continuous("machine_precision", {{}}, [&](const Vars&) { return k(machine_precision()); });
  continuous("not_a_number", {{}}, [&](const Vars&) { return k(not_a_number()); });
  continuous("positive_infinity", {{}}, [&](const Vars&) { return k(positive_infinity()); });
  continuous("negative_infinity", {{}}, [&](const Vars&) { return k(negative_infinity()); });
  // An int and a real.
  const Cases order_z = {{{0}, {1.3}}, {{1}, {2.7}}, {{2}, {0.6}}};
  discrete("bessel_first_kind", order_z, [&](const I& i, const Vars& v) { return c(bessel_first_kind(i[0], v[0])); });
  discrete("bessel_second_kind", order_z, [&](const I& i, const Vars& v) { return c(bessel_second_kind(i[0], v[0])); });
  discrete("modified_bessel_first_kind", order_z, [&](const I& i, const Vars& v) { return c(modified_bessel_first_kind(i[0], v[0])); });
  discrete("modified_bessel_second_kind", order_z, [&](const I& i, const Vars& v) { return c(modified_bessel_second_kind(i[0], v[0])); });
  discrete("binary_log_loss", {{{0}, {0.3}}, {{1}, {0.3}}, {{1}, {0.85}}}, [&](const I& i, const Vars& v) { return c(binary_log_loss(i[0], v[0])); });
  discrete("lmgamma", {{{1}, {2.5}}, {{3}, {4.1}}, {{2}, {1.7}}}, [&](const I& i, const Vars& v) { return c(lmgamma(i[0], v[0])); });
  discrete("ldexp", {{{3}, {1.5}}, {{-2}, {-0.7}}}, [&](const I& i, const Vars& v) { return c(ldexp(v[0], i[0])); });
  discrete("falling_factorial", {{{2}, {4.5}}, {{3}, {6.2}}}, [&](const I& i, const Vars& v) { return c(falling_factorial(v[0], i[0])); });
  discrete("rising_factorial", {{{2}, {4.5}}, {{3}, {0.7}}}, [&](const I& i, const Vars& v) { return c(rising_factorial(v[0], i[0])); });
  // Ints only.
  const Cases nk = {{{5, 2}, {}}, {{7, 3}, {}}, {{4, 0}, {}}};
  discrete("falling_factorial_int", nk, [&](const I& i, const Vars&) { return k(falling_factorial(i[0], i[1])); });
  discrete("rising_factorial_int", nk, [&](const I& i, const Vars&) { return k(rising_factorial(i[0], i[1])); });
  discrete("choose", {{{5, 2}, {}}, {{10, 3}, {}}, {{4, 6}, {}}}, [&](const I& i, const Vars&) { return k(choose(i[0], i[1])); });
  discrete("modulus", {{{7, 3}, {}}, {{-7, 3}, {}}, {{7, -3}, {}}}, [&](const I& i, const Vars&) { return k(modulus(i[0], i[1])); });
  const Cases bits = {{{1, 0}, {}}, {{1, 1}, {}}, {{0, 0}, {}}};
  discrete("logical_and", bits, [&](const I& i, const Vars&) { return k(logical_and(i[0], i[1])); });
  discrete("logical_or", bits, [&](const I& i, const Vars&) { return k(logical_or(i[0], i[1])); });
  discrete("logical_lt_int_int", {{{1, 2}, {}}, {{2, 2}, {}}}, [&](const I& i, const Vars&) { return k(logical_lt(i[0], i[1])); });
  discrete("logical_lt_int_real", {{{1}, {1.5}}, {{2}, {1.5}}}, [&](const I& i, const Vars& v) { return k(logical_lt(i[0], v[0].val())); });
  discrete("logical_lt_real_int", {{{1}, {0.5}}, {{1}, {1.5}}}, [&](const I& i, const Vars& v) { return k(logical_lt(v[0].val(), i[0])); });
  continuous("logical_lt_real_real", {{0.5, 0.7}, {0.7, 0.5}}, [&](const Vars& v) { return k(logical_lt(v[0].val(), v[1].val())); });
  continuous("is_nan", {{0.3}, {-2.0}}, [&](const Vars& v) { return k(is_nan(v[0].val())); });
  continuous("to_int", {{2.7}, {-3.9}}, [&](const Vars& v) { return k(to_int(v[0].val())); });
  discrete("to_int_int", {{{5}, {}}}, [&](const I& i, const Vars&) { return k(to_int(i[0])); });
  // Three and four reals.
  continuous("hypergeometric_2F1", {{1.0, 1.5, 2.5, 0.3}, {0.5, 0.7, 1.3, -0.4}}, [&](const Vars& v) { return c(hypergeometric_2F1(v[0], v[1], v[2], v[3])); });
  continuous("inc_beta", {{2.0, 3.0, 0.4}, {0.7, 1.5, 0.8}}, [&](const Vars& v) { return c(inc_beta(v[0], v[1], v[2])); });
  continuous("inv_inc_beta", {{2.0, 3.0, 0.4}, {0.7, 1.5, 0.8}}, [&](const Vars& v) { return c(inv_inc_beta(v[0], v[1], v[2])); });
  continuous("log_mix", {{0.3, -1.2, -0.4}, {0.9, -3.0, 0.5}}, [&](const Vars& v) { return c(log_mix(v[0], v[1], v[2])); });
  // Bound transforms; a _jacobian case is the constrained value plus the log Jacobian.
  continuous("lower_bound_constrain", {{0.4, -1.0}, {-2.1, 0.5}}, [&](const Vars& v) { return c(lb_constrain(v[0], v[1])); });
  continuous("lower_bound_jacobian", {{0.4, -1.0}, {-2.1, 0.5}}, [&](const Vars& v) { var lp = 0; var y = lb_constrain(v[0], v[1], lp); return y + lp; });
  continuous("lower_bound_unconstrain", {{0.4, -1.0}, {2.1, 0.5}}, [&](const Vars& v) { return c(lb_free(v[0], v[1])); });
  continuous("upper_bound_constrain", {{0.4, 2.0}, {-2.1, 0.5}}, [&](const Vars& v) { return c(ub_constrain(v[0], v[1])); });
  continuous("upper_bound_jacobian", {{0.4, 2.0}, {-2.1, 0.5}}, [&](const Vars& v) { var lp = 0; var y = ub_constrain(v[0], v[1], lp); return y + lp; });
  continuous("upper_bound_unconstrain", {{0.4, 2.0}, {-2.1, 0.5}}, [&](const Vars& v) { return c(ub_free(v[0], v[1])); });
  continuous("lower_upper_bound_constrain", {{0.4, -1.0, 2.0}, {-2.1, 0.5, 3.0}}, [&](const Vars& v) { return c(lub_constrain(v[0], v[1], v[2])); });
  continuous("lower_upper_bound_jacobian", {{0.4, -1.0, 2.0}, {-2.1, 0.5, 3.0}}, [&](const Vars& v) { var lp = 0; var y = lub_constrain(v[0], v[1], v[2], lp); return y + lp; });
  continuous("lower_upper_bound_unconstrain", {{0.4, -1.0, 2.0}, {2.1, 0.5, 3.0}}, [&](const Vars& v) { return c(lub_free(v[0], v[1], v[2])); });
  continuous("offset_multiplier_constrain", {{0.4, -1.0, 2.0}, {-2.1, 0.5, 0.3}}, [&](const Vars& v) { return c(offset_multiplier_constrain(v[0], v[1], v[2])); });
  continuous("offset_multiplier_jacobian", {{0.4, -1.0, 2.0}, {-2.1, 0.5, 0.3}}, [&](const Vars& v) { var lp = 0; var y = offset_multiplier_constrain(v[0], v[1], v[2], lp); return y + lp; });
  continuous("offset_multiplier_unconstrain", {{0.4, -1.0, 2.0}, {-2.1, 0.5, 0.3}}, [&](const Vars& v) { return c(offset_multiplier_free(v[0], v[1], v[2])); });
  // The Wiener forms with 6 to 9 arguments (y | a, t0, w, v, sv[, sw, st0][, precision]).
  const Pts w6 = {{1.2, 2.0, 0.3, 0.5, 0.8, 0.4}, {0.9, 1.4, 0.2, 0.45, -0.6, 0.9}};
  continuous("wiener_lpdf_6", w6, [&](const Vars& v) { return c(wiener_lpdf(v[0], v[1], v[2], v[3], v[4], v[5])); });
  continuous("wiener_lpdf_7", w6, [&](const Vars& v) { return c(wiener_lpdf(v[0], v[1], v[2], v[3], v[4], v[5], 1e-8)); });
  const Pts w8 = {{1.2, 2.0, 0.3, 0.5, 0.8, 0.4, 0.2, 0.1}, {0.9, 1.4, 0.2, 0.45, -0.6, 0.9, 0.3, 0.25}};
  continuous("wiener_lpdf_8", w8, [&](const Vars& v) { return c(wiener_lpdf(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7])); });
  continuous("wiener_lpdf_9", w8, [&](const Vars& v) { return c(wiener_lpdf(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], 1e-8)); });
}
