import numpy as np
"""
Exotic options.

WEEK 2. This is where you price your first option with no clean closed form --
an Asian option, whose payoff depends on the AVERAGE price over the life of the
option, not just the final price. That's why you'll need to track the full
price path here, unlike the terminal-only simulation in gbm.py.

Left as a placeholder for now; you'll build it out next week.
"""


def price_asian_option(S0, K, r, sigma, T, n_paths, n_steps,
                       option_type="call", average="arithmetic", seed=None):
    """
    Price an Asian option by Monte Carlo (Week 2).

    Unlike the European case, you must simulate the FULL path (n_steps points)
    so you can average the price over time. The payoff uses that average in
    place of S_T:

        arithmetic average call: max(mean(S_path) - K, 0)

    A sanity check for Week 2: an Asian option should be CHEAPER than the
    equivalent vanilla, because averaging dampens volatility.
    """
    S_path = simulate_gbm_path(S0, r, sigma, T, n_paths, n_steps, seed=seed)
    if average == "arithmetic":
        S_average = np.mean(S_path, axis=1)
        if option_type == "call":
            payoffs = np.maximum(S_average - K, 0)
        else:
            payoffs = np.maximum(K - S_average, 0)

    final_payoff = np.mean(payoffs)
    discounted_price = final_payoff * np.exp(-r * T)
    return discounted_price
    

def simulate_gbm_path(S0, r, sigma, T, n_paths, n_steps, seed=None):
    """Function to simulate a full path instead of the terminal price"""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_steps))

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    wobble = sigma * np.sqrt(dt) * Z
    S_path = S0 * np.exp(np.cumsum(drift + wobble, axis=1))
    return S_path