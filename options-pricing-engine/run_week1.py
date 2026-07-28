"""
Week 1 driver script.

Once you've implemented black_scholes_price, simulate_gbm_terminal,
monte_carlo_price, and monte_carlo_price_with_ci, run this to produce your
headline result: the Monte Carlo estimate converging to the Black-Scholes
price, WITH confidence intervals that visibly shrink as paths grow.

    python run_week1.py

Until the functions are implemented this will raise NotImplementedError -- that
is expected. It's here so that the moment your Saturday code works, your first
README figure (with error bars) pops out with zero extra effort.
"""

import config
from src.black_scholes import black_scholes_price
from src.monte_carlo import monte_carlo_price_with_ci
from src.plotting import plot_convergence


def main():
    bs = black_scholes_price(config.S0, config.K, config.R,
                             config.SIGMA, config.T, "call")
    print(f"Black-Scholes price: {bs:.4f}\n")

    path_counts = [1_000, 10_000, 100_000, 1_000_000]
    estimates, half_widths = [], []

    print(f"{'n_paths':>10}  {'MC price':>10}  {'95% CI':>22}  {'error vs BS':>12}")
    print("-" * 60)
    for n in path_counts:
        res = monte_carlo_price_with_ci(
            config.S0, config.K, config.R, config.SIGMA, config.T,
            n_paths=n, option_type="call",
            z_score=config.Z_SCORE, seed=config.RANDOM_SEED,
        )
        estimates.append(res["price"])
        half_widths.append(res["half_width"])
        ci = f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]"
        print(f"{n:>10,}  {res['price']:>10.4f}  {ci:>22}  {res['price']-bs:>+12.4f}")

    # Headline figure: convergence with shrinking error bars.
    plot_convergence(path_counts, estimates, reference_price=bs,
                     half_widths=half_widths)


if __name__ == "__main__":
    main()
