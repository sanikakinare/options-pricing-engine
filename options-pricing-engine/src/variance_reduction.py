"""
Variance reduction techniques.

WEEK 3. These make your Monte Carlo estimates converge faster (smaller standard
error for the same number of paths). This is the most "quant" week and the
centerpiece of your writeup.

  - Antithetic variates: for each random draw Z, also use -Z. The paired paths
    partially cancel each other's noise.
  - Control variates: use a quantity with a KNOWN answer (the geometric-average
    Asian, which has a closed form) to correct the noisy estimate of the
    quantity you actually want (the arithmetic-average Asian).

Left as placeholders for now.
"""


def price_asian_antithetic(*args, **kwargs):
    """Week 3 -- antithetic-variates version of the Asian pricer."""
    raise NotImplementedError("Week 3.")


def price_asian_control_variate(*args, **kwargs):
    """Week 3 -- control-variate version using the geometric-Asian closed form."""
    raise NotImplementedError("Week 3.")
