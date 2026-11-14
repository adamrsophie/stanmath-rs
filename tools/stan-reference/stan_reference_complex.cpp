// Stan values and gradients of the complex scalar functions of the Stan language, at points that
// include the branch cuts (approached from both signed zeros).
//
//   make STAN_MATH=/path/to/stan_math/ stan_reference_complex
//   ./stan_reference_complex > ../../crates/stanmath-rs/tests/data/stan_reference_complex.csv
//
// Each row: name, the inputs (re, im of each complex argument) separated by ';', the outputs
// (re, im, or one real), then the gradient: for each output, its partials with respect to each
// input in order.

#include <stan/math.hpp>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using stan::math::var;
using cvar = std::complex<var>;

static void row(const std::string& name, const std::vector<double>& in,
                const std::function<std::vector<var>(const std::vector<var>&)>& f) {
  std::vector<var> x(in.begin(), in.end());
  std::vector<var> out = f(x);
  std::printf("%s,", name.c_str());
  for (size_t i = 0; i < in.size(); ++i) std::printf("%s%.17g", i ? ";" : "", in[i]);
  std::printf(",");
  for (size_t i = 0; i < out.size(); ++i) std::printf("%s%.17g", i ? ";" : "", out[i].val());
  std::printf(",");
  bool first = true;
  for (auto& o : out) {
    stan::math::set_zero_all_adjoints();
    o.grad();
    for (auto& xi : x) {
      std::printf("%s%.17g", first ? "" : ";", xi.adj());
      first = false;
    }
  }
  std::printf("\n");
  stan::math::recover_memory();
}

int main() {
  std::printf("# name,inputs,outputs,gradient (Stan Math %d.%d.%d)\n", STAN_MATH_MAJOR,
              STAN_MATH_MINOR, STAN_MATH_PATCH);
  const std::vector<std::pair<double, double>> pts = {
      {0.7, 0.3}, {-1.5, 0.0}, {-1.5, -0.0}, {2.0, 0.5}, {0.0, 1.5}, {-0.0, -2.0},
      {-0.4, -0.9}, {1.3, 0.0}, {1.3, -0.0}, {-2.5, 1.1}, {0.25, -0.0}};
  using U = std::function<cvar(const cvar&)>;
  const std::vector<std::pair<std::string, U>> unary = {
      {"exp", [](const cvar& z) { return stan::math::exp(z); }},
      {"log", [](const cvar& z) { return stan::math::log(z); }},
      {"log10", [](const cvar& z) { return stan::math::log10(z); }},
      {"sqrt", [](const cvar& z) { return stan::math::sqrt(z); }},
      {"sin", [](const cvar& z) { return stan::math::sin(z); }},
      {"cos", [](const cvar& z) { return stan::math::cos(z); }},
      {"tan", [](const cvar& z) { return stan::math::tan(z); }},
      {"sinh", [](const cvar& z) { return stan::math::sinh(z); }},
      {"cosh", [](const cvar& z) { return stan::math::cosh(z); }},
      {"tanh", [](const cvar& z) { return stan::math::tanh(z); }},
      {"asin", [](const cvar& z) { return stan::math::asin(z); }},
      {"acos", [](const cvar& z) { return stan::math::acos(z); }},
      {"atan", [](const cvar& z) { return stan::math::atan(z); }},
      {"asinh", [](const cvar& z) { return stan::math::asinh(z); }},
      {"acosh", [](const cvar& z) { return stan::math::acosh(z); }},
      {"atanh", [](const cvar& z) { return stan::math::atanh(z); }},
      {"conj", [](const cvar& z) { return stan::math::conj(z); }},
      {"proj", [](const cvar& z) { return stan::math::proj(z); }},
      {"minus", [](const cvar& z) { return -z; }},
  };
  for (auto& [name, f] : unary)
    for (auto [re, im] : pts)
      row(name, {re, im}, [&](const std::vector<var>& x) {
        cvar w = f(cvar(x[0], x[1]));
        return std::vector<var>{w.real(), w.imag()};
      });
  using R = std::function<var(const cvar&)>;
  const std::vector<std::pair<std::string, R>> real_valued = {
      {"abs", [](const cvar& z) { return stan::math::abs(z); }},
      {"arg", [](const cvar& z) { return stan::math::arg(z); }},
      {"norm", [](const cvar& z) { return stan::math::norm(z); }},
  };
  for (auto& [name, f] : real_valued)
    for (auto [re, im] : pts)
      row(name, {re, im}, [&](const std::vector<var>& x) {
        return std::vector<var>{f(cvar(x[0], x[1]))};
      });
  const std::vector<std::pair<std::pair<double, double>, std::pair<double, double>>> pairs = {
      {{0.7, 0.3}, {1.5, -0.4}}, {{-1.5, 0.0}, {0.5, 0.0}}, {{-1.5, -0.0}, {0.5, 0.0}},
      {{2.0, 0.5}, {-0.3, 1.2}}, {{0.0, 1.5}, {2.0, 0.0}}, {{-0.4, -0.9}, {0.25, 0.75}}};
  for (auto [z, w] : pairs)
    row("pow", {z.first, z.second, w.first, w.second}, [](const std::vector<var>& x) {
      cvar p = stan::math::pow(cvar(x[0], x[1]), cvar(x[2], x[3]));
      return std::vector<var>{p.real(), p.imag()};
    });
  for (auto [z, w] : pairs) {
    row("divide", {z.first, z.second, w.first, w.second}, [](const std::vector<var>& x) {
      cvar p = cvar(x[0], x[1]) / cvar(x[2], x[3]);
      return std::vector<var>{p.real(), p.imag()};
    });
    row("multiply", {z.first, z.second, w.first, w.second}, [](const std::vector<var>& x) {
      cvar p = cvar(x[0], x[1]) * cvar(x[2], x[3]);
      return std::vector<var>{p.real(), p.imag()};
    });
  }
}
