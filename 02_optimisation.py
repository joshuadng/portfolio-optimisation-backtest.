"""Step 2: equal-weight vs minimum-variance portfolios and the efficient frontier.

Needs data/prices.csv from step 1.   Run:  python 02_optimisation.py

NOTE: everything here is IN-SAMPLE (we use the whole history to pick weights, then
judge them on that same history). That flatters the results. Step 3 fixes this with
a walk-forward backtest.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252
FIG_DIR = Path("figures")


def load_returns() -> pd.DataFrame:
    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    return prices.pct_change().dropna()


def portfolio_stats(w, mu, cov):
    """Annualised return and volatility of a portfolio with weights w."""
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    return ret, vol


def min_variance(cov, target_return=None, mu=None):
    """Long-only weights (sum to 1) with the lowest volatility.

    If target_return is given, the portfolio must also achieve that return.
    """
    n = len(cov)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target_return is not None:
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target_return})
    result = minimize(
        lambda w: w @ cov @ w,          # minimise variance
        x0=np.full(n, 1 / n),           # start from equal weights
        bounds=[(0, 1)] * n,            # no short selling, no leverage
        constraints=constraints,
        method="SLSQP",
    )
    return result.x if result.success else None


def efficient_frontier(mu, cov, n_points=40):
    """Trace the lowest-risk portfolio for each target return."""
    w_min = min_variance(cov)
    targets = np.linspace(w_min @ mu, mu.max(), n_points)
    points = []
    for t in targets:
        w = min_variance(cov, target_return=t, mu=mu.values)
        if w is not None:
            points.append(portfolio_stats(w, mu.values, cov))
    return np.array(points)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)

    returns = load_returns()
    mu = returns.mean() * TRADING_DAYS            # annualised mean returns
    cov = returns.cov().values * TRADING_DAYS     # annualised covariance matrix
    n = len(mu)

    w_equal = np.full(n, 1 / n)
    w_minvar = min_variance(cov)

    print("Weights (%):")
    print(pd.DataFrame({"equal": w_equal, "min_var": w_minvar}, index=mu.index).mul(100).round(1))

    for name, w in [("Equal-weight", w_equal), ("Min-variance", w_minvar)]:
        ret, vol = portfolio_stats(w, mu.values, cov)
        print(f"{name}: return {ret:.2%}, volatility {vol:.2%}, return/risk {ret / vol:.2f}")

    frontier = efficient_frontier(mu, cov)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(np.sqrt(np.diag(cov)), mu, label="Individual ETFs", color="grey")
    for name in mu.index:
        ax.annotate(name, (np.sqrt(cov[mu.index.get_loc(name)][mu.index.get_loc(name)]), mu[name]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.plot(frontier[:, 1], frontier[:, 0], label="Efficient frontier (long-only)")
    ax.scatter(*portfolio_stats(w_equal, mu.values, cov)[::-1], marker="*", s=200, label="Equal-weight")
    ax.scatter(*portfolio_stats(w_minvar, mu.values, cov)[::-1], marker="D", s=80, label="Min-variance")
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised return")
    ax.set_title("Efficient frontier (in-sample)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "efficient_frontier.png", dpi=150)
    print("Saved figures/efficient_frontier.png")
