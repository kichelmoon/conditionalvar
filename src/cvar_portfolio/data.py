"""
Data loading and preprocessing.

Prices come from Yahoo Finance (via yfinance) and are cached locally as
parquet (or CSV if pyarrow is unavailable), so that every later run is
reproducible and works offline.

Important modelling choice: the optimisation works with *simple* returns
R_t = P_t / P_{t-1} - 1, because only those aggregate linearly across assets:
the portfolio return is R_t^T w. For log-returns, log(1 + R_t^T w) is NOT
sum_i w_i log(1 + R_{t,i}), so a loss L(w) = -r_t^T w built from log-returns
would be wrong.

Known limitation: the ticker universe is a set of *today's* large caps
(survivorship / look-ahead bias). This is acceptable for illustrating the
mathematics, but not for drawing conclusions about real performance.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN",             # Technology
    "JPM",  "GS",   "BAC",                       # Financials
    "JNJ",  "PFE",                               # Healthcare
    "XOM",  "CVX",                               # Energy
    "PG",   "KO",   "WMT",                       # Consumer Staples
]

# 100 US large caps from all 11 GICS sectors, each listed before 2010 and still
# trading at the end of 2025 (Yahoo Finance symbols). Contains TICKERS as a subset,
# so the 14-stock problem is a restriction of the 100-stock problem.
# Caveat: selecting today's survivors makes the survivorship bias *stronger*.
TICKERS_100 = [
    # Information technology (17)
    "AAPL", "MSFT", "NVDA", "ORCL", "CSCO", "INTC", "IBM", "QCOM", "TXN", "ADBE",
    "CRM", "AMD", "ACN", "INTU", "ADP", "GOOGL", "AMZN",
    # Communication services (5)
    "VZ", "T", "CMCSA", "DIS", "NFLX",
    # Financials (14)
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "USB", "PNC", "BLK",
    "SCHW", "BK", "MET", "BRK-B",
    # Health care (13)
    "JNJ", "PFE", "MRK", "UNH", "ABT", "AMGN", "BMY", "LLY", "MDT", "GILD",
    "CVS", "TMO", "DHR",
    # Consumer staples (10)
    "PG", "KO", "PEP", "WMT", "COST", "MO", "CL", "KMB", "GIS", "SYY",
    # Consumer discretionary (10)
    "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "TJX", "F", "YUM",
    # Industrials (12)
    "BA", "CAT", "GE", "HON", "MMM", "UNP", "UPS", "LMT", "DE", "EMR",
    "FDX", "GD",
    # Energy (6)
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
    # Utilities (5)
    "NEE", "DUK", "SO", "D", "AEP",
    # Materials (4)
    "APD", "SHW", "ECL", "NEM",
    # Real estate (4)
    "SPG", "AMT", "PSA", "EQR",
]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"


@dataclass
class MarketData:
    """Container for processed market data (simple returns)."""
    tickers: list[str]
    prices: pd.DataFrame          # adjusted close, shape (T_total + 1, n)
    returns: pd.DataFrame         # simple returns, shape (T_total, n)
    train_returns: np.ndarray     # shape (T_train, n)
    test_returns: np.ndarray      # shape (T_test,  n)
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex

    @property
    def log_returns(self) -> pd.DataFrame:
        """Log-returns, for descriptive statistics only (not for optimisation)."""
        return np.log1p(self.returns)


def _cache_path(tickers: list[str], start: str, end: str, cache_dir: Path) -> Path:
    key = hashlib.sha1(f"{sorted(tickers)}|{start}|{end}".encode()).hexdigest()[:12]
    return cache_dir / f"prices_{key}"


def _read_cache(base: Path) -> pd.DataFrame | None:
    pq, csv = base.with_suffix(".parquet"), base.with_suffix(".csv")
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, index_col=0, parse_dates=True)
    return None


def _write_cache(prices: pd.DataFrame, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        prices.to_parquet(base.with_suffix(".parquet"))
    except ImportError:
        prices.to_csv(base.with_suffix(".csv"))


def load_prices(
    tickers: list[str] = TICKERS,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Adjusted close prices, read from the local cache or downloaded once."""
    base = _cache_path(tickers, start, end, Path(cache_dir))
    if not refresh:
        cached = _read_cache(base)
        if cached is not None:
            return cached

    import yfinance as yf  # imported lazily: not needed when the cache exists

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    prices = prices.dropna(how="all")
    if prices.empty:
        raise RuntimeError("Download returned no data (network or ticker problem).")
    _write_cache(prices, base)
    return prices


def market_data_from_prices(prices: pd.DataFrame, train_end: str) -> MarketData:
    """Clean prices, compute simple returns and split into train / test."""
    missing_frac = prices.isna().mean()
    prices = prices.loc[:, missing_frac < 0.05].ffill().dropna()

    returns = prices.pct_change().dropna()
    train_mask = returns.index <= pd.Timestamp(train_end)
    train, test = returns.loc[train_mask], returns.loc[~train_mask]

    return MarketData(
        tickers=prices.columns.tolist(),
        prices=prices,
        returns=returns,
        train_returns=train.to_numpy(),
        test_returns=test.to_numpy(),
        train_dates=train.index,
        test_dates=test.index,
    )


def fetch_data(
    tickers: list[str] = TICKERS,
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    train_end: str = "2019-12-31",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> MarketData:
    """Load (cached) prices and return a train/test split of simple returns."""
    prices = load_prices(tickers, start, end, cache_dir=cache_dir, refresh=refresh)
    return market_data_from_prices(prices, train_end)
