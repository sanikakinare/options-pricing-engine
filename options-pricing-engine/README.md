# Options Pricing Engine

A derivatives pricing engine built up from vanilla European options to a
path-dependent exotic, validated by pricing the same instrument two independent
ways and accelerated with variance-reduction techniques.

> **Status:** Week 1 in progress. Sections below fill in as the project grows —
> the README is written *as* the work happens, not bolted on at the end.

---

## Overview

This project prices options two ways and shows they agree:

1. **Black-Scholes** — the closed-form analytic price (exact).
2. **Monte Carlo** — simulate many risk-neutral price paths, average the
   discounted payoffs. Converges to the analytic price as paths grow.

It then moves to an **Asian option** (payoff depends on the *average* price over
the option's life), which has no clean closed form and so *requires* simulation —
and applies **variance reduction** (antithetic and control variates) to make
that simulation converge faster.

## Results

*(Fill in as you go.)*

- [ ] **Validation:** Monte Carlo converges to Black-Scholes (Figure 1).
- [ ] **Error bounds:** every Monte Carlo price is reported with a 95%
      confidence interval, and the interval narrows as √n paths are added — the
      estimate comes with a stated uncertainty, not just a point value.
- [ ] **Exotic:** Asian option priced by simulation; comes out cheaper than the
      vanilla equivalent, as expected.
- [ ] **Variance reduction:** standard error vs. paths for naive MC vs.
      antithetic vs. control variate (Figure 2) — quantified speedup, shown as
      a *narrower CI for the same number of paths*.

## Repo structure

```
options-pricing-engine/
├── config.py                 # standard test case + validation targets
├── run_week1.py              # driver: produces the convergence figure
├── requirements.txt
├── src/
│   ├── black_scholes.py      # closed-form pricer          (Week 1, Mon)
│   ├── gbm.py                # risk-neutral path simulation (Week 1, Wed)
│   ├── monte_carlo.py        # Monte Carlo pricer           (Week 1, Sat)
│   ├── exotics.py            # Asian option                 (Week 2)
│   ├── variance_reduction.py # antithetic + control variate (Week 3)
│   └── plotting.py           # figure helpers
├── tests/
│   └── test_pricing.py       # run `pytest -v` — your definition of done
└── notebooks/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
pytest -v          # run the test suite (tests go green as you implement)
python run_week1.py  # produce the Week 1 convergence figure
```

## Method notes

*A few sentences, in your own words, on the ideas that made this work — the
risk-neutral drift, why the two methods agree, why variance reduction helps.
This is the section that becomes your SOP sentence, so write it for a human.*

## What I'd do next

The natural extension is to grow this into a fuller pricing engine covering a
wider family of exotics and a richer set of variance-reduction methods. In
rough order of what I'd add:

- **Barrier options (knock-in / knock-out).** Path-dependent options that
  activate or extinguish when the underlying crosses a level. These build
  directly on the path-tracking already used for the Asian option.
- **Importance sampling.** The key technique for *rare-payoff* options — e.g. a
  far out-of-the-money barrier that almost never triggers, where naive Monte
  Carlo wastes nearly all its paths on zero payoffs. Importance sampling shifts
  the simulated distribution toward the region that matters, then reweights, so
  the estimator concentrates its effort where the payoff actually lives.
- **A unified variance-reduction layer.** Combine antithetic variates, control
  variates, and importance sampling, and compare which delivers the tightest
  confidence interval per unit of compute for each option type.
- **Greeks and a volatility surface.** Sensitivities (delta, vega, …) via
  finite differences or pathwise methods, and calibration to a market-implied
  vol surface.

The through-line is the same discipline used here: don't just estimate the
price, quantify the uncertainty and drive the error bound down.
