# Builds the Stan Math reference generator with Stan Math's standalone build rules.
#
#   make STAN_MATH=/path/to/stan_math/ stan_reference
#   ./stan_reference > ../../crates/stanmath-rs-distributions/tests/data/stan_reference.csv
#
# stan_probe (built the same way) prints black-box answers to questions of convention; the
# "Stan Math probe" workflow builds and runs it on CI.
#
# STAN_MATH must end with a slash. With CmdStan installed it is
# $CMDSTAN/stan/lib/stan_math/.

MATH := $(STAN_MATH)
include $(MATH)make/standalone

# The ODE and Newton solvers link the SUNDIALS libraries bundled with Stan Math.
SUNDIALS_LIB = $(wildcard $(MATH)lib/sundials_*/lib)
stan_reference_solvers: LDLIBS += $(addprefix $(SUNDIALS_LIB)/libsundials_,cvodes.a idas.a kinsol.a nvecserial.a)
