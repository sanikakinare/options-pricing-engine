"""
Week 1 driver script.

Once you've implemented black_scholes_price, simulate_gbm_terminal, and
monte_carlo_price, run this to produce your headline result: the Monte Carlo
estimate converging to the Black-Scholes price, saved as a figure.

    python run_week1.py

Until the functions are implemented this will raise NotImplementedError -- that
is expected. It's here so that the moment your Saturday code works, your first
README figure pops out with zero extra effort.
"""

import config
from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price
from src.plotting import plot_convergence


def main():
    bs = black_scholes_price(config.S0, config.K, config.R,
                             config.SIGMA, config.T, "call")
    print(f"Black-Scholes price: {bs:.4f}")

    path_counts = [1_000, 10_000, 100_000, 1_000_000]
    mc_estimates = []
    for n in path_counts:
        mc = monte_carlo_price(config.S0, config.K, config.R, config.SIGMA,
                               config.T, n_paths=n, option_type="call",
                               seed=config.RANDOM_SEED)
        mc_estimates.append(mc)
        print(f"  n_paths={n:>9,}  MC price={mc:.4f}  (error {mc - bs:+.4f})")

    plot_convergence(path_counts, mc_estimates, reference_price=bs)


if __name__ == "__main__":
    main()
