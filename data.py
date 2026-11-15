"""
Data loading and preprocessing.

Fetches historical adjusted close prices via yfinance, computes log returns,
and provides train/test splits for out-of-sample evaluation.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",             # Technology
    "JPM",  "GS",   "BAC",                       # Financials
    "JNJ",  "PFE",                               # Healthcare
    "XOM",  "CVX",                               # Energy
    "PG",   "KO",   "WMT",                       # Consumer Staples
]


@dataclass
class MarketData:
    """Container for processed market data."""
    tickers: list[str]
    prices: pd.DataFrame          # adjusted close, shape (T_total, n)
    returns: pd.DataFrame         # log returns, shape (T_total-1, n)
    train_returns: np.ndarray     # shape (T_train, n)
    test_returns: np.ndarray      # shape (T_test,  n)
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def fetch_data(
    tickers: list[str] = TICKERS,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    train_end: str = "2019-12-31",
) -> MarketData:
    """
    Download adjusted close prices and compute log returns.

    Parameters
    ----------
    tickers   : list of Yahoo Finance ticker symbols
    start     : start date (inclusive)
    end       : end date (inclusive)
    train_end : last date of training window; everything after is test

    Returns
    -------
    MarketData dataclass with prices, returns, and train/test splits.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Keep only "Close" (auto_adjust already applies splits & dividends)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(how="all")
    else:
        prices = raw.dropna(how="all")

    # Drop tickers with too many missing values (> 5 %)
    missing_frac = prices.isna().mean()
    prices = prices.loc[:, missing_frac < 0.05]
    prices = prices.ffill().dropna()

    tickers_clean = prices.columns.tolist()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    train_mask = log_ret.index <= pd.Timestamp(train_end)
    test_mask  = log_ret.index >  pd.Timestamp(train_end)
    train_ret = log_ret.loc[train_mask]
    test_ret  = log_ret.loc[test_mask]

    return MarketData(
        tickers=tickers_clean,
        prices=prices,
        returns=log_ret,
        train_returns=train_ret.values,
        test_returns=test_ret.values,
        train_dates=train_ret.index,
        test_dates=test_ret.index,
    )


def annualise_return(r: float, freq: int = 252) -> float:
    """Daily log-return → annualised arithmetic return."""
    return np.expm1(r * freq)


def annualise_vol(sigma: float, freq: int = 252) -> float:
    """Daily volatility → annualised."""
    return sigma * np.sqrt(freq)
