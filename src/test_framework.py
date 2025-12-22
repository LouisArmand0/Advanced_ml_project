
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from utils.features import (compute_vol_adjusted_returns,
                               compute_alpha_to_market,
                               compute_beta_to_market,
                               compute_macd, compute_cti,
                               compute_alpha_variance_ratio,
                            getting_trading_universe,
                            getting_data_for_ticker_list, compute_returns,)
from utils.metrics import sharpe_ratio, drawdown
from utils.estimators import LinearRegression, MultiLGBMRegressor
from utils.backtesting import Backtester
from utils.mv_estimator import MeanVariance



import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def compute_features(ret, wide=True):
    beta_to_mkt_list = []
    for lookback in [126, 252]:
        temp = compute_beta_to_market(
            ret_df=ret,
            ticker_market="spy",
            lookback=lookback
        )
        beta_to_mkt_list.append(temp)

    beta_to_mkt = beta_to_mkt_list[0]

    for beta_df in beta_to_mkt_list[1:]:
        beta_to_mkt = pd.merge(beta_to_mkt, beta_df, on=['date', 'stock_name'], how='inner')

    alpha_to_mkt_list = []
    for lookback in [5, 21, 63, 126, 252]:
        alpha_to_mkt_list.append(compute_alpha_to_market(
            ret_df=ret,
            ticker_market="spy",
            lookback=lookback
        ))
    alpha_to_mkt = alpha_to_mkt_list[0]
    for alpha_df in alpha_to_mkt_list[1:]:
        alpha_to_mkt = pd.merge(alpha_to_mkt, alpha_df, on=['date', 'stock_name'], how='inner')

    macd_list = []
    for c in [(8, 24), (16, 48), (32, 64)]:
        macd_list.append(compute_macd(
            ret_df=ret,
            short_window=c[0],
            long_window=c[1]
        ))
    macd = macd_list[0]
    for macd_df in macd_list[1:]:
        macd = pd.merge(macd, macd_df, on=['date', 'stock_name'], how='inner')

    cti_list = []
    for c in [63]:
        cti_list.append(
            compute_cti(
                ret_df=ret,
                lookback=c,
            )
        )
    cti = cti_list[0]
    for cti_df in cti_list[1:]:
        cti = pd.merge(cti, cti_df, on=['date', 'stock_name'], how='inner')

    alpha_var_list = []
    for lookback in [5, 21, 63, 126, 252]:
        alpha_var_list.append(compute_alpha_variance_ratio(
            ret_df=ret,
            ticker_market="spy",
            window=lookback,
        ))

    alpha_var = alpha_var_list[0]
    for alpha_var_df in alpha_var_list[1:]:
        alpha_var = pd.merge(alpha_var, alpha_var_df, on=['date', 'stock_name'], how='inner')

    # Setting data
    l = [beta_to_mkt, alpha_to_mkt, alpha_var, cti, macd]
    X_full = l[0]

    for c in l[1:]:
        X_full = pd.merge(X_full, c, on=['date', 'stock_name'], how='inner')
    X = X_full.sort_values(by=['date'], ascending=True).reset_index(drop=True)

    if wide:
        features = [c for c in X_full.columns if c not in ["date", "stock_name"]]

        # Pivot each feature and prefix columns
        wide_list = []

        for feat in features:
            temp = X_full.pivot(index="date", columns="stock_name", values=feat)
            temp.columns = [f"{feat}_{c}" for c in temp.columns]
            wide_list.append(temp)

        # Combine all wide matrices
        X = pd.concat(wide_list, axis=1).dropna(axis=1, how='any')
    return X



if __name__ == '__main__':
    trading_universe = getting_trading_universe()['symbol'].to_list()
    ticker_list = [ticker for ticker in trading_universe if ticker != 'SPY']
    market_ticker = ["SPY"]

    df = getting_data_for_ticker_list(ticker_list + market_ticker)
    df = df.dropna(axis=1, how='any')
    ticker_list = [col.split('_')[0] for col in df.columns if col != 't_close']
    logger.info('Raw data loaded')
    logger.info('Computing features...')
    vol_adj_ret = compute_vol_adjusted_returns(
        df,
        return_type='simple',
        vol_model='ewma',
        window=20
        )

    X = compute_features(vol_adj_ret, wide=True)

    y = vol_adj_ret.pivot(index='date', columns='stock_name', values='vol_adjusted_ret')
    y = y.copy().iloc[1:].dropna(axis=1, how='any')
    y = y.loc[X.index]
    y = y.shift(-1)

    ret = compute_returns(
        df,
        return_type='simple',
    )
    ret = ret.pivot(index='date', columns='stock_name', values='simple_ret')
    ret = ret.loc[X.index]


    benchmark = (
        Backtester(MeanVariance(), name="benchmark")
        .compute_holdings(y,y,ret)
        .compute_pnl(ret)
    )
    pnl = pd.DataFrame(benchmark.pnl_)

    lr = make_pipeline(StandardScaler(), LinearRegression(), MeanVariance())
    m = (
        Backtester(lr, name="linear_regression")
        .compute_holdings(X, y)
        .compute_pnl(ret)
    )
    pnl = pd.merge(pnl, pd.DataFrame(m.pnl_), right_index=True, left_index=True)

    lgbm = make_pipeline(StandardScaler(), MultiLGBMRegressor(min_child_samples=5, n_estimators=25), MeanVariance())

    m = (Backtester(lgbm, name="lgbm")
         .compute_holdings(X, y)
         .compute_pnl(ret))

    pnl = pd.merge(pnl, pd.DataFrame(m.pnl_), right_index=True, left_index=True)

    plt.figure(figsize=(12, 8))
    plt.plot(pnl.cumsum(), label=f'lr: {sharpe_ratio(pnl)}')
    plt.plot(pnl_benchmark.cumsum(), label='benchmark')
    plt.legend()
    plt.title("Cumulative Sum")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.grid(True)
    plt.show()

