"""Check that the backtest has no look-ahead bias.

Idea: scramble every return AFTER a chosen date. If the strategies only use past data,
every weight and return BEFORE that date must be exactly unchanged.

Run:  python test_no_lookahead.py      (needs data/prices.csv from step 1)
"""
import importlib

import numpy as np
import pandas as pd

bt = importlib.import_module("03_backtest")

prices = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
returns = prices.pct_change().dropna()

cut = returns.index[len(returns) // 2]          # a date roughly halfway through history
scrambled = returns.copy()
future = scrambled.index > cut
scrambled.loc[future] = np.random.default_rng(0).normal(0, 0.05, scrambled.loc[future].shape)

strategies = {
    "Equal-weight": bt.equal_weight,
    "Min-var (expanding)": bt.make_min_variance(),
    "Min-var (3y rolling)": bt.make_min_variance(bt.ROLLING_WINDOW),
}

for name, fn in strategies.items():
    ret_a, w_a, _ = bt.backtest(returns, fn)
    ret_b, w_b, _ = bt.backtest(scrambled, fn)
    same_weights = np.allclose(w_a[w_a.index <= cut].values, w_b[w_b.index <= cut].values)
    same_returns = np.allclose(ret_a[ret_a.index <= cut].values, ret_b[ret_b.index <= cut].values)
    assert same_weights and same_returns, f"{name}: results before {cut.date()} changed, so the future leaked in"
    print(f"PASS  {name}: decisions before {cut.date()} are unaffected by future data")

# Sanity checks: weights are valid and trading only starts after the estimation period
_, w, _ = bt.backtest(returns, bt.make_min_variance())
assert np.allclose(w.sum(axis=1), 1) and w.values.min() >= -1e-9, "weights must sum to 1 and be non-negative"
assert returns.index.get_loc(w.index[0]) >= bt.MIN_HISTORY, "first trade happened too early"
print("PASS  weights sum to 1, are non-negative, and the first trade follows the estimation period")
