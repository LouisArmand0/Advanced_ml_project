import pandas as pd
import numpy as np
from typing import List
import os

def compute_returns(prices: pd.DataFrame, return_type: str) -> pd.DataFrame:
    """Compute simple or log returns."""
    if return_type == "simple":
        ret = prices.pct_change()
    elif return_type == "log":
        ret = np.log(prices / prices.shift(1))
    else:
        raise ValueError("return_type must be 'simple' or 'log'")
    return ret.dropna()


def compute_rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation volatility."""
    return returns.rolling(window).std().dropna()


def compute_ewma_vol(returns: pd.DataFrame, lam: float) -> pd.DataFrame:
    """EWMA volatility using RiskMetrics formulation."""

    var = returns.pow(2).ewm(alpha=(1 - lam), adjust=False).mean()
    vol = np.sqrt(var)
    return vol.dropna()


def compute_vol_adjusted_returns(
    prices: pd.DataFrame,
    return_type: str = "log",   # "simple" or "log"
    vol_model: str = "ewma",    # "rolling" or "ewma"
    window: int = 40,
    ewma_lambda: float = 0.94
):
    """
    Compute returns, volatility, and vol-adjusted returns.
    """

    returns = compute_returns(prices, return_type)

    if vol_model == "rolling":
        vol = compute_rolling_vol(returns, window)
    elif vol_model == "ewma":
        vol = compute_ewma_vol(returns, ewma_lambda)
    else:
        raise ValueError("vol_model must be 'rolling' or 'ewma'")

    returns = returns.loc[vol.index]

    adj_returns = returns / vol

    return adj_returns


def compute_beta_to_market(ret_df: pd.DataFrame,
                           tickers_stock: list,
                           ticker_market: str,
                           lookback: int) -> pd.DataFrame:
    """
    Compute beta to market returns.
    """
    beta_dict = {}
    for stock in tickers_stock:
        stock_col = f'{stock.lower()}_close'

        # compute rolling covariance
        cov = ret_df[stock_col].rolling(lookback, min_periods=lookback).cov(ret_df[f'{ticker_market}_close'])
        var = ret_df[f'{ticker_market}_close'].rolling(lookback, min_periods=lookback).var()
        beta = cov / var

        beta_dict[f'beta_to_market_{stock.lower()}'] = beta

    return pd.DataFrame(beta_dict)

def compute_alpha_to_market(ret_df: pd.DataFrame,
                             tickers_stock: List[str],
                             ticker_market: str,
                             lookback: int) -> pd.DataFrame:

    stock_col = [f'{ticker.lower()}_close' for ticker in tickers_stock]
    market_col = f'{ticker_market.lower()}_close'
    alpha = ret_df[stock_col].sub(ret_df[market_col], axis=0).rolling(lookback, min_periods=lookback).mean()

    # rename columns
    alpha.columns = [f'alpha_{t.lower()}_{lookback}' for t in tickers_stock]

    return alpha

def compute_alpha_variance_ratio(ret_df: pd.DataFrame,
                                tickers_stock: List[str],
                                ticker_market: str,
                                lookback: int) -> pd.DataFrame:
    stock_col = [f'{ticker.lower()}_close' for ticker in tickers_stock]
    market_col = f'{ticker_market.lower()}_close'

    # alpha
    alpha = ret_df[stock_col].sub(ret_df[market_col], axis=0)

    # rolling variances
    rolling_var_alpha = alpha.rolling(lookback).var()
    rolling_var_stock = ret_df[stock_col].rolling(lookback).var()

    # alpha variance ratio
    avr = rolling_var_alpha / rolling_var_stock
    avr.columns = [f'alpha_var_ratio_{t.lower()}_{lookback}' for t in tickers_stock]

    return avr

def compute_cti(ret_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Compute the Correlation Trend Indicator (CTI) for multiple assets.
    ret_df : DataFrame of returns (each column = one asset)
    """

    # Time index for regression: [1 .. lookback]
    t = np.arange(1, lookback + 1)
    Sx  = t.sum()
    Sxx = (t * t).sum()
    n = lookback

    roll = ret_df.rolling(lookback)

    # Rolling sums
    Sy  = roll.sum()
    Syy = roll.apply(lambda x: np.sum(x * x), raw=True)
    Sxy = roll.apply(lambda x: np.sum(t * x), raw=True)

    # CTI
    num = Sx * Sy - n * Sxy
    den = np.sqrt((Sx**2 - n * Sxx) * (Sy**2 - n * Syy))
    cti = num / den

    clean_names = [
        col.lower().replace("close_", "").replace("_close", "")
        for col in ret_df.columns
    ]
    cti.columns = [f"cti_{name}_{lookback}" for name in clean_names]

    return cti

def compute_macd(ret_df: pd.DataFrame,
                tickers_stock: List[str],
                short_window: int,
                long_window: int) -> pd.DataFrame:
    """
    Compute MACD for multiple tickers vectorized.
    """
    macd_dict = {}
    for ticker in tickers_stock:
        col = f'{ticker.lower()}_close'
        price = ret_df[col].astype(float)

        ema_short = price.ewm(span=short_window, adjust=False).mean()
        ema_long = price.ewm(span=long_window, adjust=False).mean()

        macd = ema_short - ema_long
        macd_dict[f'macd_{short_window}_{long_window}_{ticker.lower()}'] = macd

    return pd.DataFrame(macd_dict, index=ret_df.index)

if __name__ == "__main__":

    file_path_ticker = os.path.join(os.path.dirname(__file__), '../data/raw/trading_universe.csv')
    tickers_stock = pd.read_csv(file_path_ticker)['symbol'].to_list()

    file_path_prices = os.path.join(os.path.dirname(__file__), '../data/raw/historical_prices.csv')
    prices = pd.read_csv("src/data/raw/historical_prices.csv").set_index("Date").filter(like="_close")

    adj_returns = compute_vol_adjusted_returns(prices)
    beta_to_mkt = compute_beta_to_market(
        ret_df=adj_returns,
        tickers_stock=tickers_stock,
        ticker_market="spy",
        lookback=40
    )

    alpha_to_mkt = compute_alpha_to_market(
        ret_df=adj_returns,
        tickers_stock=tickers_stock,
        ticker_market="spy",
        lookback=5
    )
    alpha_variance_ratio = compute_alpha_variance_ratio(
        ret_df=adj_returns,
        tickers_stock=tickers_stock,
        ticker_market="spy",
        lookback=5
    )

    cti = compute_cti(ret_df=adj_returns, lookback=5)
    macd = compute_macd(
        ret_df=adj_returns,
        tickers_stock=tickers_stock,
        short_window=5,
        long_window=16
    )
