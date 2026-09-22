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

# --- Week 2: the volatility smile -------------------------------------------

# Categorical slots 1 and 2 of a CVD-validated palette; spot is an annotation,
# not a series, so it wears muted ink rather than a third hue.
_SERIES = {"put": "#2a78d6", "call": "#eb6834"}
_INK, _INK_MUTED, _GRID = "#0b0b0b", "#52514e", "#d8d7d2"


def plot_smile(smile, title=None, save_path="figures/smile.png"):
    """
    Plot implied volatility against strike for one expiry.

    If Black-Scholes were true this would be a flat line -- sigma is a property
    of the underlying, not of the contract. The curve you get instead is the
    market pricing in a fatter left tail than a lognormal allows, so the deep
    downside puts imply a far higher vol than the at-the-money options.

    Parameters
    ----------
    smile : pandas.DataFrame
        Output of build_smile: columns strike, option_type, mid, iv. Spot,
        expiry and T are read from `smile.attrs` when present.
    title : str or None
        Defaults to a title naming the expiry and time to maturity.
    save_path : str
        Where to save the PNG. The folder is created if needed.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    spot = smile.attrs.get("spot")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for option_type in ("put", "call"):
        leg = smile[smile["option_type"] == option_type]
        if leg.empty:
            continue
        ax.plot(leg["strike"], leg["iv"] * 100, lw=2, color=_SERIES[option_type],
                label=f"{option_type.capitalize()}s (out of the money)")

    if spot is not None:
        ax.axvline(spot, color=_INK_MUTED, lw=1, ls="--", zorder=0)
        ax.annotate(f"spot {spot:.2f}", xy=(spot, ax.get_ylim()[1]),
                    xytext=(4, -10), textcoords="offset points",
                    color=_INK_MUTED, fontsize=9, va="top")

    if title is None:
        expiry, T = smile.attrs.get("expiry", "?"), smile.attrs.get("T")
        title = f"Implied volatility smile -- expiry {expiry}"
        if T is not None:
            title += f"  ({T * 365.25:.0f} days)"

    ax.set_title(title, color=_INK, fontsize=13, pad=12)
    ax.set_xlabel("Strike", color=_INK_MUTED)
    ax.set_ylabel("Implied volatility (%)", color=_INK_MUTED)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)
    ax.grid(True, color=_GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.legend(frameon=False, labelcolor=_INK_MUTED, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved figure to {save_path}")
    return fig, ax
