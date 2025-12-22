import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List


def getting_data_for_ticker_list(ticker_list: List[str]) -> pd.DataFrame:
    ROOT = Path(__file__).resolve().parents[2]
    print(ROOT)
    TICKERS_DIR = ROOT / "data/tickers"
    l = []
    for ticker in ticker_list:
        df = pd.read_parquet(TICKERS_DIR / f"{ticker.upper()}.parquet")
        df_close = df.xs('Close', level='Price', axis=1).copy()
        df_close.columns = [f'price_at_close']
        df_close.loc[: ,'stock_name'] = ticker.upper()
        df_close = df_close.reset_index()
        df_close.columns = [c.lower() for c in df_close.columns]
        l.append(df_close)

    df = pd.concat(l, axis=0)
    return df

def getting_trading_universe():
    ROOT = Path(__file__).resolve().parents[2]
    print(ROOT)
    DATA_DIR = ROOT / "data"
    return pd.read_parquet(DATA_DIR / "trading_universe.parquet")


def compute_returns(prices: pd.DataFrame, return_type: str) -> pd.DataFrame:
    """
    Compute simple or log returns per stock.
    Expects columns: ['date', 'stock_name', 'price_at_close'].
    """
    prices = prices.sort_values(['stock_name', 'date']).copy()

    if return_type == "simple":
        prices['ret'] = prices.groupby('stock_name')['price_at_close'].pct_change()
    elif return_type == "log":
        prices['ret'] = prices.groupby('stock_name')['price_at_close'].apply(
            lambda x: np.log(x / x.shift(1))
        ).reset_index(level=0, drop=True)
    else:
        raise ValueError("return_type must be 'simple' or 'log'")

    prices = prices.rename(columns={'ret': f'{return_type}_ret'})
    return prices[['date', 'stock_name', f'{return_type}_ret']].dropna()

def compute_rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling standard deviation volatility."""
    returns = returns.sort_values(['stock_name', 'date']).copy()
    ret_col = [c for c in returns.columns if c.endswith('_ret')]
    returns[f'{window}_volatility'] = returns.groupby('stock_name')[ret_col].rolling(window=window).std().reset_index(level=0, drop=True)
    return returns[['date', 'stock_name', f'{window}_volatility']].dropna()

def compute_ewma_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """EWMA volatility using RiskMetrics formulation."""
    returns = returns.sort_values(['stock_name', 'date']).copy()
    ret_col = [c for c in returns.columns if c.endswith('_ret')]
    returns[f'{window}_ewma_volatility'] = (
            returns.groupby('stock_name')[ret_col]
            .apply(lambda x: x.pow(2).ewm(adjust=False, span=window).mean())
            .reset_index(level=0, drop=True)
        )
    returns[f'{window}_ewma_volatility'] = np.sqrt(returns[f'{window}_ewma_volatility'])
    return returns[['date', 'stock_name', f'{window}_ewma_volatility']].dropna()

def compute_vol_adjusted_returns(
    prices: pd.DataFrame,
    return_type: str ,   # "simple" or "log"
    vol_model: str ,    # "rolling" or "ewma"
    window: int ,

):
    """
    Compute returns, volatility, and vol-adjusted returns.
    """

    returns = compute_returns(prices, return_type)

    ret_col = [c for c in returns.columns if c.endswith('_ret')][0]

    if vol_model == "rolling":
        vol = compute_rolling_vol(returns, window)  # ['date','stock_name','vol']
    elif vol_model == "ewma":
        vol = compute_ewma_vol(returns, window=window)
    else:
        raise ValueError("vol_model must be 'rolling' or 'ewma'")

    df = returns.merge(vol, on=['date', 'stock_name'], how='inner')
    df['vol_adjusted_ret'] = df[ret_col] / df[f'{window}_{vol_model}_volatility']

    return df[['date', 'stock_name', 'vol_adjusted_ret']]

def compute_beta_to_market(ret_df: pd.DataFrame,
                           ticker_market: str,
                           lookback: int) -> pd.DataFrame:

    tickers_stock = ret_df['stock_name'].unique().tolist()
    tickers_stock = [_ for _ in tickers_stock if _ != ticker_market]
    ret_col = [_ for _ in ret_df.columns if _.endswith('_ret')][0]

    market = ret_df[ret_df['stock_name'] == ticker_market.upper()][['date', ret_col]].rename(columns={ret_col: 'market_ret'})
    df = ret_df[ret_df['stock_name'].isin(tickers_stock)].copy()
    df = df.merge(market, on='date', how='left')
    df = df.sort_values(['stock_name', 'date'])

    def rolling_beta(x):
        cov = x[ret_col].rolling(lookback, min_periods=lookback).cov(x['market_ret'])
        var = x['market_ret'].rolling(lookback, min_periods=lookback).var()
        return cov / var

    df[f'beta_to_market_{lookback}'] = df.groupby('stock_name').apply(
    rolling_beta
    ).reset_index(level=0, drop=True)

    return df[['date', 'stock_name', f'beta_to_market_{lookback}']].dropna().reset_index(drop=True)

def compute_alpha_to_market(
        ret_df: pd.DataFrame,
        ticker_market: str,
        lookback: int,
        ) -> pd.DataFrame:
    tickers_stock = ret_df['stock_name'].unique().tolist()
    tickers_stock = [_ for _ in tickers_stock if _ != ticker_market]
    ret_col = [_ for _ in ret_df.columns if _.endswith('_ret')][0]

    market = ret_df[ret_df['stock_name'] == ticker_market.upper()][['date', ret_col]].rename(columns={ret_col: 'market_ret'})
    beta = compute_beta_to_market(ret_df, ticker_market, lookback)
    ret_df = ret_df[ret_df['stock_name'].isin(tickers_stock)].copy()
    df = pd.merge(ret_df, beta, on=['date', 'stock_name'], how='left')
    df = pd.merge(df, market, on='date', how='left')
    df = df.sort_values(['stock_name', 'date'])

    df[f'alpha_to_market_{lookback}'] = df[ret_col] - df[f'beta_to_market_{lookback}'] * df['market_ret']
    return df[['date', 'stock_name', f'alpha_to_market_{lookback}']].dropna().reset_index(drop=True)

def compute_alpha_variance_ratio(
        ret_df: pd.DataFrame,
        ticker_market: str,
        window: int) -> pd.DataFrame:
    ret_col = [_ for _ in ret_df.columns if _.endswith('_ret')][0]
    alpha = compute_alpha_to_market(ret_df, ticker_market, window)
    df = pd.merge(ret_df, alpha, on=['date', 'stock_name'], how='inner')
    alpha_var = df.groupby('stock_name')[f'alpha_to_market_{window}'].rolling(window, min_periods=window).var().reset_index(level=0, drop=True)
    ret_var = df.groupby('stock_name')[ret_col].rolling(window, min_periods=window).var().reset_index(level=0, drop=True)
    df[f'alpha_variance_ratio_{window}'] = alpha_var / ret_var
    return df[['date', 'stock_name', f'alpha_variance_ratio_{window}']].dropna().reset_index(drop=True)

def compute_cti(ret_df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Compute the Correlation Trend Indicator (CTI) per stock in long-format returns.

    """
    df = ret_df.sort_values(['stock_name', 'date']).copy()
    t = np.arange(1, lookback + 1)
    ret_col = [_ for _ in ret_df.columns if _.endswith('_ret')][0]
    def rolling_cti(x):
        # x: Series of returns for one stock
        res = pd.Series(index=x.index, dtype=float)
        for i in range(lookback - 1, len(x)):
            y = x.iloc[i - lookback + 1: i + 1].values
            Sx  = t.sum()
            Sxx = (t**2).sum()
            Sy  = y.sum()
            Syy = (y**2).sum()
            Sxy = np.sum(t * y)
            num = Sx * Sy - lookback * Sxy
            den = np.sqrt((Sx**2 - lookback * Sxx) * (Sy**2 - lookback * Syy))
            res.iloc[i] = num / den if den != 0 else np.nan
        return res

    # Apply per stock
    df[f'cti_{lookback}'] = df.groupby('stock_name', group_keys=False)[ret_col].transform(rolling_cti)

    return df[['date', 'stock_name', f'cti_{lookback}']].dropna().reset_index(drop=True)

def compute_macd(ret_df: pd.DataFrame,
                  short_window: int,
                  long_window: int) -> pd.DataFrame:

    df = ret_df.sort_values(['stock_name', 'date'])
    ret_col = [_ for _ in ret_df.columns if _.endswith('_ret')][0]
    def macd_calc(x):
        ret = x[ret_col].astype(float)
        ema_short = ret.ewm(span=short_window, adjust=False).mean()
        ema_long = ret.ewm(span=long_window, adjust=False).mean()
        return ema_short - ema_long

    df[f'macd_{short_window}_{long_window}'] = df.groupby('stock_name', group_keys=False).apply(macd_calc)

    return df[['date', 'stock_name', f'macd_{short_window}_{long_window}']].dropna().reset_index(drop=True)

