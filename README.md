# Options Pricer

A complete options pricing library in Python  analytical and numerical models, Greeks, strategies, exotic options, portfolio management, interactive hedging, and a Streamlit interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Pricing Methods](#pricing-methods)
- [Option Types](#option-types)
- [Greeks](#greeks)
- [Strategies](#strategies)
- [Interface  Tab Guide](#interface--tab-guide)
- [Database](#database)
- [Market Data](#market-data)
- [Configuration](#configuration)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

This project is an end-to-end options pricer built to apply quantitative finance concepts in practice: valuation of vanilla and exotic options, Greek computation, strategy construction, unbalanced portfolio management, and dynamic hedging.

It is built around five independent pricing methods, an analytical and numerical Greeks engine, a portfolio generator that pulls real market data from Yahoo Finance (SPY options chain), and a full Streamlit interface with a dark theme.

The core idea is to price the same option with every available method simultaneously, compute a consensus price, measure the spread of uncertainty across methods, compare it to the market mid price, and use the resulting Greeks to make hedging decisions  all from a single interface.

---

## Features

### Pricing
- Five methods implemented and compared on every option
- Multi-method consensus price (weighted average) with inter-method uncertainty spread
- Mispricing measure: consensus price versus market mid price
- Support for European, American, and exotic options

### Greeks
- Closed-form Black-Scholes formulas for European options
- Central finite differences (bump and reprice) for any other pricer
- Delta, Gamma, Vega, Theta, Rho  computed per position and aggregated at portfolio level

### Strategies
- Call Spread, Put Spread, Straddle, Strangle, Butterfly, Iron Condor
- Automatic computation of net premium, breakevens, maximum profit and maximum loss
- Payoff diagram at expiry

### Exotic Options
- Asian: arithmetic or geometric averaging on price or strike
- Barrier: down-and-out, up-and-out, down-and-in, up-and-in
- Lookback: floating or fixed strike
- All priced via Monte Carlo with Longstaff-Schwartz for early exercise

### Portfolio and Hedging
- Random portfolio generation from the live SPY options chain
- Portfolio-level aggregated Greeks
- Delta-Gamma-Vega stress test: heatmap of P&L impact from simultaneous S and sigma shocks
- Automatic hedge recommendations for delta, gamma, and vega
- Interactive hedge simulator: add SPY shares, ATM calls, puts, or straddles and see the immediate impact on portfolio Greeks

### Heatmaps
- Price and Greeks heatmaps as a function of any two configurable parameters
- P&L heatmap with configurable entry price, green for profit and red for loss
- Every calculation persisted to a SQLite database

### Database
- Full calculation history in SQLite via a two-table relational schema
- Reload and re-visualize any past calculation
- CSV export of inputs and outputs per calculation ID

---

## Project Structure

```
Pricer_Project/
│
├── option_types/
│   ├── base_option.py             Base class for all options
│   ├── european_option.py         EuropeanOption
│   ├── american_option.py         AmericanOption
│   └── exotic_option.py           AsianOption, BarrierOption, LookbackOption
│
├── pricing_methods/
│   ├── black_scholes.py           Black-Scholes-Merton closed form
│   ├── binomial.py                Cox-Ross-Rubinstein (vectorised)
│   ├── trinomial.py               Boyle trinomial tree
│   ├── montecarlo.py              Monte Carlo with antithetic variates and LSM
│   └── finite_diff.py             Crank-Nicolson finite difference scheme
│
├── greeks/
│   └── greeks.py                  Greeks class  analytical and numerical
│
├── strategies/
│   └── strategies.py              Multi-leg option strategies
│
├── visuals/
│   ├── heatmap.py                 Price and Greeks heatmaps (Plotly)
│   └── pnl_heatmap.py             P&L stress test heatmap
│
├── data/
│   ├── market_data.py             MarketDataLoader  yfinance with synthetic fallback
│   ├── portfolio_generator.py     Random portfolio generation from market data
│   ├── database.py                SQLite layer (SQLAlchemy)  inputs and outputs tables
│   ├── pricer.db                  SQLite database (auto-created on first run)
│   ├── input_data.csv             Legacy CSV
│   └── output_data.csv            Legacy CSV
│
├── pnl/
│   ├── portfolio_pricer.py        Multi-method pricing, Greek aggregation, hedge recommendations
│   └── pnl_calculator.py         Mark-to-market P&L calculator
│
├── interface/
│   └── app.py                     Streamlit application  7 tabs, dark theme
│
├── main.py                        CLI entry point  quick test of all modules
├── requirements.txt
└── pyrightconfig.json             Pylance / Pyright import resolution config
```

Every directory contains an `__init__.py` file, making the project a standard Python package importable from the root.

---

## Installation

**Requirements:** Python 3.10 or higher.

Create a virtual environment, activate it, and install the dependencies listed in `requirements.txt`:

```
numpy >= 1.24
scipy >= 1.10
pandas >= 2.0
plotly >= 5.18
streamlit >= 1.30
sqlalchemy >= 2.0
yfinance >= 0.2
```

---

## Getting Started

Open a terminal at the project root. To launch the Streamlit interface, run the following command and open the URL printed in the terminal (defaults to `http://localhost:8501`):

    streamlit run interface/app.py

To run a quick CLI test of all modules against a sample SPY ATM option, run:

    python main.py

This prints pricing results from all five methods, the full Greeks table, strategy summaries, and a portfolio P&L breakdown.

**VS Code import resolution.** Open the `Pricer_Project/` folder as the workspace root via File > Open Folder. The `pyrightconfig.json` at the root sets `extraPaths` to `.`, which tells Pylance to resolve all imports relative to the project root. If Pylance still underlines imports, verify that the workspace root is correct.

---

## Pricing Methods

### Black-Scholes-Merton

Closed-form analytical solution for European options with continuous dividend yield. Inputs are the five standard parameters: spot price, strike, time to maturity, risk-free rate, and implied volatility. Does not support American or exotic options.

The call price is given by:

    C = S * exp(-y*T) * N(d1) - K * exp(-r*T) * N(d2)

where:

    d1 = ( ln(S/K) + (r - y + 0.5 * sigma^2) * T ) / ( sigma * sqrt(T) )
    d2 = d1 - sigma * sqrt(T)

### Binomial  Cox-Ross-Rubinstein

Recombining binomial tree with N time steps. Fully vectorised via NumPy (no nested Python loops). Supports both European and American exercise. The up and down factors are:

    u = exp(sigma * sqrt(dt)),   d = 1/u

Risk-neutral probability:

    p = ( exp((r - y) * dt) - d ) / ( u - d )

Early exercise is enforced at each backward induction step for American options.

### Trinomial  Boyle

Three-branch recombining tree using the Boyle (1986) parametrisation. Converges faster than the binomial at equal N because each time step covers a larger price range. Up factor:

    u = exp(sigma * sqrt(2 * dt))

Probabilities derived from matching the first two moments of the log-normal distribution. Supports European and American options.

### Monte Carlo with Longstaff-Schwartz

Simulates asset price paths under the risk-neutral measure. Variance reduction via antithetic variates (pairs Z and -Z). Returns a price estimate with standard error and 95% confidence interval.

American early exercise is handled by the Longstaff-Schwartz least-squares Monte Carlo algorithm: at each time step, the continuation value is estimated by regressing the discounted future cash flows on a polynomial basis of the in-the-money paths (Laguerre polynomials of degree 2). Early exercise is triggered whenever the intrinsic value exceeds the estimated continuation value.

Exotic options (Asian, Barrier, Lookback) are priced by recording the full path and applying the relevant payoff function.

### Finite Differences  Crank-Nicolson

Solves the Black-Scholes PDE on a two-dimensional (S, t) grid using the Crank-Nicolson scheme. The scheme is unconditionally stable and second-order accurate in both time and space. At each backward time step, a tridiagonal linear system is solved via the banded solver from SciPy.

For American options, a projection step is applied at each node to enforce the early exercise constraint (the option value cannot fall below the intrinsic value). The final price is recovered by linear interpolation at the spot price.

---

## Option Types

All option classes inherit from `BaseOption` and share the same parameter interface: spot price S, strike K, time to maturity t (in years), risk-free rate r, volatility sigma, and continuous dividend yield y.

**EuropeanOption**  exercise at maturity only. Compatible with all five pricing methods.

**AmericanOption**  exercise possible at any time before maturity. Compatible with Binomial, Trinomial, Monte Carlo, and Finite Differences.

**AsianOption**  payoff depends on the average of the underlying price over the option life. Parameters: `averaging` (arithmetic or geometric) and `avg_type` (price or strike). Priced by Monte Carlo only.

**BarrierOption**  the option is activated or extinguished when the underlying crosses a barrier level. Parameter `barrier_type` accepts: down-and-out, up-and-out, down-and-in, up-and-in. Priced by Monte Carlo.

**LookbackOption**  payoff depends on the minimum or maximum price over the option life. Parameter `lookback_type`: floating (call pays S_T minus the minimum, put pays the maximum minus S_T) or fixed (standard strike, extremum substituted). Priced by Monte Carlo.

---

## Greeks

The `Greeks` class provides a unified interface regardless of the underlying pricer. For European options without a custom pricing function, it uses closed-form Black-Scholes formulas. For all other cases (American options, exotic options, or any numerical pricer passed as argument), it falls back to central finite differences.

**Delta** measures the sensitivity of the option price to a one-unit change in the underlying. For a European call, it equals exp(-y*T) * N(d1), which approximates the risk-neutral probability of the option expiring in the money.

**Gamma** measures the rate of change of Delta with respect to the underlying. It quantifies the convexity of the position and indicates how frequently a delta hedge must be rebalanced.

**Vega** measures the sensitivity to a one-percentage-point change in implied volatility. Expressed per 1% move in vol (divided by 100).

**Theta** measures the daily time decay of the option value, expressed per calendar day (divided by 365). Theta is negative for long option positions.

**Rho** measures the sensitivity to a one-percentage-point change in the risk-free rate, divided by 100.

At portfolio level, each Greek is the sum of individual position Greeks weighted by signed quantity (positive for long, negative for short).

---

## Strategies

All strategies take an option object (used as a parameter template), the relevant strikes, and a pricing function. They return a `StrategyResult` object containing the net premium, breakeven levels, maximum profit, maximum loss, and numpy arrays for the payoff diagram.

**Call Spread (Bull)**  long call at K1, short call at K2 with K1 < K2. Directional bullish view with limited cost and limited gain. Breakeven at K1 plus net debit.

**Put Spread (Bear)**  long put at K2, short put at K1 with K1 < K2. Directional bearish view with limited cost and limited gain. Breakeven at K2 minus net debit.

**Straddle**  long call and long put at the same strike. Bets on a large move in either direction. Maximum loss is the total premium paid.

**Strangle**  long OTM put at K_put and long OTM call at K_call. Cheaper than a straddle but requires a larger move to be profitable.

**Butterfly**  long K1, short 2x K2, long K3 where K1 < K2 < K3 and K2 is the midpoint. Profits if the underlying stays near K2. Available for calls or puts.

**Iron Condor**  short put spread combined with a short call spread. Collects a net credit. Profits if the underlying stays within the range [K2, K3] until expiry. Maximum loss is capped by the width of either spread.

---

## Interface  Tab Guide

### Pricing

The sidebar controls all input parameters: spot price, strike, risk-free rate, dividend yield, volatility, time to maturity, option style, option type, and numerical method with its specific parameters (number of steps for trees, number of simulations for Monte Carlo, grid size for finite differences).

The tab displays the option price, intrinsic value, time value, moneyness (ITM / ATM / OTM), and a comparison table of all available methods with their prices and the spread versus the Black-Scholes reference. A payoff diagram at expiry shows the gross payoff as a function of the underlying price at maturity.

### Greeks

Displays the five Greeks with colour coding (green for positive, red for negative). An interactive chart shows how a selected Greek evolves as a function of the underlying price, computed numerically across the full range.

### Heatmaps

Seven sub-tabs: Price, Delta, Gamma, Vega, Theta, Rho, and an overview showing all five Greeks side by side in a two-column grid. The X and Y axes are configurable from the set {S, sigma, t, r}. Grid resolution is adjustable from 6 to 20 points per axis.

### Strategies

Select a strategy, configure the relevant strikes, and the tab computes the full P&L profile. It displays the leg breakdown (type, strike, position, individual premium), the key metrics (net premium, breakeven levels, maximum profit, maximum loss), and the payoff diagram with breakeven levels highlighted.

### P&L Heatmap

Shock the spot price (configurable range, default -40% to +40%) and the volatility (configurable range, default -20 to +20 vol points) on a grid. The entry price is configurable and defaults to the current option price. Each cell shows the P&L equal to the shocked option price minus the entry price. Green cells indicate profit, red cells indicate loss, white is the breakeven. Summary metrics include maximum P&L, minimum P&L, median P&L, and the percentage of scenarios that are profitable. Every calculation is automatically saved to the SQLite database.

### Historical Database

Lists the 100 most recent saved calculations. Selecting a calculation ID reconstructs both the P&L heatmap and the raw price heatmap from the stored data. The inputs and outputs for any selected calculation can be downloaded as separate CSV files. A clear button purges the entire database.

### Portfolio

The central tab of the project. It has two modes for building a portfolio.

Manual entry allows adding any option type including all exotic variants: European, American, Asian arithmetic, Asian geometric, Barrier down-and-out, Barrier up-and-out, Barrier down-and-in, Lookback floating, and Lookback fixed. Each position has its own parameters (S, K, t, r, sigma, y, quantity, entry price, barrier level for barrier options).

Random generation pulls the SPY options chain from Yahoo Finance if a connection is available, and falls back to a realistic synthetic chain otherwise. The mix of strategies is configurable by category weight: vanilla long/short, call and put spreads, volatility strategies (straddle, strangle, butterfly, iron condor), and exotic options.

After clicking the pricing button, each position is priced with all compatible methods, a consensus price is computed, and the results are saved to the database. The visualisation section then shows the following.

A horizontal bar chart of P&L by position, sorted from worst to best, with green bars for profitable positions and red bars for losing ones. A donut chart breaking down positions by strategy type, with the total portfolio P&L displayed at the centre. A cumulative P&L curve across all positions. A scatter plot of consensus price versus market mid price, where each dot is coloured by the position P&L.

Below the charts, the five aggregated portfolio Greeks are displayed. An automatic hedge recommendation table suggests actions for delta (number of SPY shares to buy or sell), gamma (number of short-term ATM options needed), and vega (number of three-month ATM options needed), with an urgency rating for each. A delta-gamma-vega stress test heatmap shows the estimated P&L impact of simultaneous shocks to S and sigma using the second-order Taylor approximation. An interactive hedge simulator lets the user add hedging instruments (SPY shares, ATM call, ATM put, or ATM straddle at a chosen maturity) and immediately see the before/after Greeks comparison with percentage reduction for each.

---

## Database

The SQLite database at `data/pricer.db` is created automatically on the first run. It follows a two-table relational schema linked by a short UUID called `calc_id`.

The `inputs` table stores one row per calculation with the option parameters: style, option type, S, K, t, r, sigma, y, entry price, and pricing method used.

The `outputs` table stores one row per scenario (one per grid cell in the heatmap), with the shock applied to S (as a percentage), the shock applied to sigma (in absolute vol points), the resulting shocked values of S and sigma, the option price at those shocked parameters, and the P&L relative to the entry price. Each output row references its parent calculation via `calc_id`.

This schema makes it straightforward to query all scenarios for a given set of parameters, reconstruct any heatmap from history, or export a calculation to CSV for further analysis.

---

## Market Data

The `MarketDataLoader` class attempts to retrieve the SPY options chain from Yahoo Finance. If the connection fails or the data is unavailable, it falls back to a synthetic chain generated internally.

The synthetic fallback reproduces the main features of a real equity options chain. The at-the-money volatility decreases with time to maturity (inverted term structure typical of equity markets in stressed regimes). The volatility skew is negative: puts struck below the forward are more expensive than equidistant calls, reflecting the empirical equity risk premium in volatility space. An additional symmetric smile component is added to reflect the wings. The bid-ask spread widens for deep out-of-the-money strikes and for short-dated options, consistent with reduced liquidity in those regions. Open interest and volume are drawn from a log-normal distribution calibrated to typical SPY figures, with higher values near the at-the-money strike.

The chain covers four expiry dates (one week, one month, three months, six months) and approximately 110 strikes spaced five dollars apart across a range from 75% to 125% of the spot price.

---

## Configuration

**Pylance import resolution.** The file `pyrightconfig.json` at the project root sets `pythonVersion` to 3.10, `typeCheckingMode` to basic, and `extraPaths` to `["."]`. This tells Pylance to look for modules starting from the project root, so all imports of the form `from option_types.european_option import EuropeanOption` resolve correctly without any `sys.path` manipulation. VS Code must open the `Pricer_Project/` folder as the workspace root for this to work.

**Monte Carlo reproducibility.** Every Monte Carlo call accepts a `seed` parameter passed to `numpy.random.default_rng`. Setting a fixed seed guarantees identical results across runs. The interface exposes this parameter in the sidebar.

**Pricing method compatibility.** The `STYLE_COMPAT` dictionary in `pnl/portfolio_pricer.py` controls which methods are used for each option style during portfolio pricing. Asian, Barrier, and Lookback options are restricted to Monte Carlo. American options exclude Black-Scholes. Adjusting this dictionary allows adding or removing methods from the consensus computation.

---

## Roadmap

**Implied volatility surface calibration.** Invert the Black-Scholes formula for each market price using Brent's root-finding algorithm to extract the implied volatility, then interpolate or parametrise the resulting surface (SVI, SABR, or a spline).

**Stochastic volatility models.** Implement the Heston model (closed-form characteristic function with numerical Fourier inversion) and the SABR model (Hagan et al. analytical approximation for the implied volatility smile).

**Analytical approximation for American options.** Implement the Barone-Adesi and Whaley quadratic approximation to price American options without a numerical grid, significantly faster for a single price query.

**Higher-order Greeks.** Add Vanna (sensitivity of Delta to volatility), Volga (second derivative with respect to volatility), and Charm (sensitivity of Delta to time).

**Backtesting engine.** Replay historical OHLCV data combined with a historical implied volatility surface to simulate the P&L of a strategy or portfolio over a given period.

**Multi-asset options.** Extend the framework to basket options and best-of / worst-of payoffs, requiring a correlation matrix and multi-dimensional Monte Carlo.

**Database migration.** Replace SQLite with PostgreSQL for multi-user deployments or production use, with connection pooling managed by SQLAlchemy.

**Deployment.** Package the application with Docker and deploy to Streamlit Cloud or Hugging Face Spaces for browser-based access without a local Python installation.

---

## References

Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637–654.

Cox, J., Ross, S. and Rubinstein, M. (1979). Option Pricing: A Simplified Approach. Journal of Financial Economics, 7(3), 229–263.

Boyle, P. (1986). Option Valuation Using a Three-Jump Process. International Options Journal, 3, 7–12.

Longstaff, F. and Schwartz, E. (2001). Valuing American Options by Simulation: A Simple Least-Squares Approach. Review of Financial Studies, 14(1), 113–147.

Crank, J. and Nicolson, P. (1947). A Practical Method for Numerical Evaluation of Solutions of Partial Differential Equations of the Heat-Conduction Type. Mathematical Proceedings of the Cambridge Philosophical Society, 43(1), 50–67.

Wilmott, P., Howison, S. and Dewynne, J. (1995). The Mathematics of Financial Derivatives. Cambridge University Press.

Barone-Adesi, G. and Whaley, R. (1987). Efficient Analytic Approximation of American Option Values. Journal of Finance, 42(2), 301–320.
