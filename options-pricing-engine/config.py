"""
Standard test case for the pricing engine.

This is the at-the-money, one-year European call you'll use all of Week 1 so
that your two pricing methods (Black-Scholes and Monte Carlo) are always
comparing the same option. Keep BS_CALL_REFERENCE taped to your monitor: when
both pricers land near it, Week 1 is done.
"""

# --- Standard option parameters ---
S0 = 100.0      # current stock price (spot)
K = 100.0       # strike price
R = 0.05        # risk-free rate (annual, continuously compounded)
SIGMA = 0.20    # volatility (annual)
T = 1.0         # time to maturity (years)

# --- Validation target ---
# The Black-Scholes price of the standard CALL above is ~10.4506.
# Your closed-form pricer should return this; your Monte Carlo pricer should
# converge to it as the number of paths grows.
BS_CALL_REFERENCE = 10.4506

# The matching PUT (via put-call parity) is ~5.5735, if you want a second check.
BS_PUT_REFERENCE = 5.5735

# --- Reproducibility ---
# Use this seed so your simulation results are repeatable run-to-run.
RANDOM_SEED = 42

# --- Confidence interval reporting ---
# Every Monte Carlo price should come with an error bound, not just a point
# estimate. A 95% CI uses z = 1.96 (the standard-normal critical value).
CONFIDENCE_LEVEL = 0.95
Z_SCORE = 1.96
