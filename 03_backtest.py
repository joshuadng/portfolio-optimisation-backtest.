"""Step 3: walk-forward backtest of equal-weight vs minimum-variance portfolios.

Needs data/prices.csv (step 1) and 02_optimisation.py (step 2) in the same folder.
The first run also downloads the US T-bill yield (risk-free rate) and caches it.
Run:  python 03_backtest.py

How it works
------------
On the first trading day of each month the strategy chooses new weights using ONLY
returns from days BEFORE that day. It then holds those weights (letting them drift
with market moves) until the next month, paying a transaction cost on whatever it
has to trade. This removes the look-ahead bias of step 2.
"""
import importlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Step 2's file name starts with a digit, so a normal `import` statement can't be used.
opt = importlib.import_module("02_optimisation")   # reuse min_variance()

# --- Settings -----------------------------------------------------------
TRADING_DAYS = 252
COST_RATE = 0.001      # 0.1% of every unit of money traded (10 basis points)
MIN_HISTORY = 504      # ~2 years of data needed before the first trade
ROLLING_WINDOW = 756   # ~3 years, for the rolling-window strategy
START = "2007-01-01"
RF_TICKER = "^IRX"     # US 13-week Treasury bill yield, quoted in % per year
RF_FILE = Path("data/risk_free.csv")
FIG_DIR = Path("figures")

STRESS_PERIODS = {
    "COVID crash (19 Feb-23 Mar 2020)": ("2020-02-19", "2020-03-23"),
    "2022 (bonds and equities both fell)": ("2022-01-01", "2022-12-31"),
}


# --- Weight rules: each takes past returns only and returns a weight vector ---
def equal_weight(history: pd.DataFrame) -> np.ndarray:
    n = history.shape[1]
    return np.full(n, 1 / n)


def make_min_variance(window=None):
    """Minimum-variance rule. window=None uses all past data (expanding window);
    window=756 uses only the most recent 756 days (rolling window)."""
    def weights(history: pd.DataFrame) -> np.ndarray:
        if window is not None:
            history = history.iloc[-window:]
        cov = history.cov().values * TRADING_DAYS
        w = opt.min_variance(cov)
        return w if w is not None else equal_weight(history)   # fall back if solver fails
    return weights


# --- Risk-free rate -----------------------------------------------------
def load_risk_free(dates: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free return for each date, from the US 13-week T-bill yield.

    The yield is quoted as an annual percentage (e.g. 4.5 means 4.5% a year), so we
    convert it to a daily rate. Downloaded once and cached to CSV.
    """
    if RF_FILE.exists():
        annual_pct = pd.read_csv(RF_FILE, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        data = yf.download(RF_TICKER, start=START, auto_adjust=True, threads=False, progress=False)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        annual_pct = close.dropna()
        if annual_pct.empty:
            raise RuntimeError(f"No data downloaded for {RF_TICKER}. Run the script again.")
        annual_pct.to_csv(RF_FILE, header=["rf_annual_pct"])

    annual = (annual_pct / 100).reindex(dates).ffill().bfill()     # align to our trading days
    return (1 + annual) ** (1 / TRADING_DAYS) - 1                  # annual rate -> daily rate


# --- The backtest -------------------------------------------------------
def backtest(returns: pd.DataFrame, weight_fn):
    """Walk forward through history. Returns (daily net returns, weights, turnover)."""
    dates = returns.index
    n_days, n_assets = returns.shape

    month = dates.month
    is_rebalance = np.zeros(n_days, dtype=bool)
    is_rebalance[1:] = month[1:] != month[:-1]            # first trading day of each month
    start = int(np.argmax(is_rebalance & (np.arange(n_days) >= MIN_HISTORY)))

    w = np.zeros(n_assets)                                # start fully in cash
    net_returns, rebal_dates, targets, turnovers = [], [], [], []

    for i in range(start, n_days):
        cost = 0.0
        if i == start or is_rebalance[i]:
            history = returns.iloc[:i]                    # strictly BEFORE today: no look-ahead
            target = weight_fn(history)
            traded = np.abs(target - w).sum()             # how much we have to buy and sell
            cost = COST_RATE * traded
            w = target
            rebal_dates.append(dates[i])
            targets.append(target.copy())
            turnovers.append(traded)

        r = returns.iloc[i].values
        gross = w @ r                                     # today's portfolio return
        net_returns.append(gross - cost)
        w = w * (1 + r) / (1 + gross)                     # weights drift as prices move

    net = pd.Series(net_returns, index=dates[start:])
    weights = pd.DataFrame(targets, index=rebal_dates, columns=returns.columns)
    turnover = pd.Series(turnovers, index=rebal_dates)
    return net, weights, turnover


# --- Performance measures -----------------------------------------------
def performance(r: pd.Series, rf: pd.Series, turnover: pd.Series) -> dict:
    """rf is the daily risk-free return on the same dates as r."""
    years = len(r) / TRADING_DAYS
    growth = (1 + r).cumprod()
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    excess = r - rf                                        # return earned above cash
    cutoff = np.percentile(r, 5)
    return {
        "Ann. return": growth.iloc[-1] ** (1 / years) - 1,
        "Ann. volatility": ann_vol,
        "Sharpe": excess.mean() / excess.std() * np.sqrt(TRADING_DAYS),
        "Sharpe (rf=0)": r.mean() / r.std() * np.sqrt(TRADING_DAYS),
        "Max drawdown": (growth / growth.cummax() - 1).min(),
        "Daily VaR 95%": -cutoff,
        "Daily ES 95%": -r[r <= cutoff].mean(),
        "Annual turnover": turnover.iloc[1:].sum() / years,   # excludes the initial purchase
    }


def period_return(r: pd.Series, start: str, end: str) -> float:
    return (1 + r.loc[start:end]).prod() - 1


# --- Main ---------------------------------------------------------------
if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)

    prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()

    strategies = {
        "Equal-weight": equal_weight,
        "Min-var (expanding)": make_min_variance(),
        "Min-var (3y rolling)": make_min_variance(ROLLING_WINDOW),
    }

    results = {name: backtest(returns, fn) for name, fn in strategies.items()}

    rf = load_risk_free(returns.index)

    first = min(r.index[0] for r, _, _ in results.values())
    last = returns.index[-1]
    print(f"Backtest period: {first.date()} to {last.date()} "
          f"(first {MIN_HISTORY} days used only to estimate weights)")
    avg_rf = ((1 + rf.loc[first:]).prod() ** (TRADING_DAYS / len(rf.loc[first:])) - 1)
    print(f"Transaction cost: {COST_RATE:.2%} of each unit traded; rebalanced monthly")
    print(f"Risk-free rate: US 13-week T-bill, average {avg_rf:.2%} a year over the backtest\n")

    table = pd.DataFrame({name: performance(r, rf.loc[r.index], t) for name, (r, _, t) in results.items()}).T
    pct_cols = ["Ann. return", "Ann. volatility", "Max drawdown", "Daily VaR 95%", "Daily ES 95%"]
    shown = table.copy()
    shown[pct_cols] = shown[pct_cols].map(lambda x: f"{x:.2%}")
    for col in ["Sharpe", "Sharpe (rf=0)"]:
        shown[col] = shown[col].map(lambda x: f"{x:.2f}")
    shown["Annual turnover"] = shown["Annual turnover"].map(lambda x: f"{x:.0%}")
    print(shown.to_string())

    print("\nStress periods (total return):")
    stress = pd.DataFrame(
        {name: {label: period_return(r, a, b) for label, (a, b) in STRESS_PERIODS.items()}
         for name, (r, _, _) in results.items()}
    )
    print(stress.map(lambda x: f"{x:.1%}").to_string())

    # Chart 1: growth of 1 unit
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, (r, _, _) in results.items():
        (1 + r).cumprod().plot(ax=ax, label=name)
    ax.set_title("Backtest: growth of 1 unit invested (after costs)")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "backtest_growth.png", dpi=150)
    plt.close()

    # Chart 2: drawdowns
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (r, _, _) in results.items():
        growth = (1 + r).cumprod()
        (growth / growth.cummax() - 1).plot(ax=ax, label=name)
    ax.set_title("Backtest: drawdown from previous peak")
    ax.set_ylabel("Drawdown")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "backtest_drawdown.png", dpi=150)
    plt.close()

    # Chart 3: how the minimum-variance weights moved over time
    weights = results["Min-var (expanding)"][1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(weights.index, weights.T.values, labels=weights.columns)
    ax.set_title("Min-variance (expanding window): target weights each month")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "backtest_weights.png", dpi=150)
    plt.close()

    print("\nSaved charts to", FIG_DIR)
