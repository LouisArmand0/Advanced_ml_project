
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from utils.features import (compute_vol_adjusted_returns,
                            getting_trading_universe,
                            getting_data_for_ticker_list,
                            compute_returns,
                            FeatureParams,
                            compute_features)
from utils.metrics import sharpe_ratio, drawdown
from utils.estimators import LinearRegression, MultiLGBMRegressor
from utils.backtesting import Backtester
from utils.mv_estimator import MeanVariance



import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()




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

    params = FeatureParams(
        beta_to_market=[5, 21, 63, 126, 252],
        alpha_to_market=[5, 21, 63, 126, 252],
        alpha_variance_ratio=[5, 21, 63, 126, 252],
        macd=[(8, 24), (16, 48), (32, 96)],
        cti=[5, 21, 64, 126, 252],

    )
    X = compute_features(vol_adj_ret, params=params, wide=True)

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



    plt.figure(figsize=(12, 8))
    plt.plot(pnl['linear_regression'].cumsum(), label=f'lr')
    plt.plot(pnl['benchmark'].cumsum(), label='benchmark')
    plt.legend()
    plt.title("Cumulative Sum")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Value")
    plt.grid(True)
    plt.show()

