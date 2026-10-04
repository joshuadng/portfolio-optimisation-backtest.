"""Step 1: download ETF prices, compute returns, and look at risk and correlation.

Run:  python 01_data_and_returns.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# --- Settings -----------------------------------------------------------
# A small, diversified universe: equities, bonds, credit, gold, commodities, property.
TICKERS = {
    "SPY": "US large-cap equity",
    "IWM": "US small-cap equity",
    "EFA": "Developed ex-US equity",
    "EEM": "Emerging-market equity",
    "AGG": "US aggregate bonds",
    "TLT": "Long-dated US Treasuries",
    "LQD": "Investment-grade credit",
    "GLD": "Gold",
    "DBC": "Broad commodities",
    "VNQ": "US real estate",
}
START = "2007-01-01"   # early enough to include the 2008 crisis
TRADING_DAYS = 252     # used to annualise daily statistics

DATA_FILE = Path("data/prices.csv")
FIG_DIR = Path("figures")


# --- Functions ----------------------------------------------------------
def load_prices() -> pd.DataFrame:
    """Return adjusted close prices, downloading once and caching to CSV."""
    if DATA_FILE.exists():
        cached = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        if not cached.empty:
            return cached
        print("Cached price file is empty, downloading again...")

    # threads=False avoids a 'database is locked' error on some Windows setups
    prices = yf.download(list(TICKERS), start=START, auto_adjust=True, threads=False)["Close"]

    missing = [t for t in TICKERS if t not in prices.columns or prices[t].isna().all()]
    if missing:
        raise RuntimeError(f"No data downloaded for {missing}. Run the script again.")

    prices = prices.dropna()          # keep only dates where every ETF has a price
    if prices.empty:
        raise RuntimeError("Download returned no usable rows. Run the script again.")
    prices.to_csv(DATA_FILE)          # only saved once we know the data is good
    return prices


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns: (P_t / P_{t-1}) - 1."""
    return prices.pct_change().dropna()


def summary_table(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised return, volatility and a simple return/risk ratio per asset."""
    ann_return = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    return pd.DataFrame(
        {"ann_return": ann_return, "ann_vol": ann_vol, "return_per_risk": ann_return / ann_vol}
    )


def plot_cumulative(returns: pd.DataFrame) -> None:
    growth = (1 + returns).cumprod()
    ax = growth.plot(figsize=(10, 6), logy=True, title="Growth of 1 unit invested (log scale)")
    ax.set_ylabel("Value")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cumulative_growth.png", dpi=150)
    plt.close()


def plot_correlation(returns: pd.DataFrame) -> None:
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45)
    ax.set_yticks(range(len(corr)), corr.columns)
    fig.colorbar(im, label="Correlation")
    ax.set_title("Correlation of daily returns")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation.png", dpi=150)
    plt.close()


# --- Main ---------------------------------------------------------------
if __name__ == "__main__":
    DATA_FILE.parent.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    prices = load_prices()
    returns = daily_returns(prices)

    print(f"Prices: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    print(summary_table(returns).round(3))

    plot_cumulative(returns)
    plot_correlation(returns)
    print("Saved figures to", FIG_DIR)