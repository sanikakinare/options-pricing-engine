"""
Plotting utilities (plumbing -- fully implemented, no need to edit).

This gives you the Week 1 convergence figure for free once your pricers work.
"""

import os
import matplotlib.pyplot as plt


def plot_convergence(path_counts, mc_estimates, reference_price,
                     half_widths=None,
                     title="Monte Carlo convergence to Black-Scholes",
                     save_path="figures/convergence.png"):
    """
    Plot Monte Carlo price estimates against the analytic reference.

    Parameters
    ----------
    path_counts : list[int]
        The n_paths values you tested (x-axis, log scale).
    mc_estimates : list[float]
        The Monte Carlo price at each n_paths.
    reference_price : float
        The Black-Scholes price (drawn as a horizontal reference line).
    half_widths : list[float] or None
        Optional. The +/- confidence-interval half-width at each n_paths. If
        given, the plot shows error bars that visibly SHRINK as paths grow --
        this is the "narrow error bound" story made visual, and a great README
        figure. Pass the "half_width" values from monte_carlo_price_with_ci.
    title : str
    save_path : str
        Where to save the PNG. The folder is created if needed.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    if half_widths is not None:
        ax.errorbar(path_counts, mc_estimates, yerr=half_widths, fmt="o-",
                    capsize=4, label="Monte Carlo estimate (95% CI)")
    else:
        ax.plot(path_counts, mc_estimates, "o-", label="Monte Carlo estimate")
    ax.axhline(reference_price, color="crimson", linestyle="--",
               label=f"Black-Scholes = {reference_price:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Option price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved figure to {save_path}")
    return fig, ax