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
