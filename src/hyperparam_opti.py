
import re
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from gnn_wrapper_multiple import GNNRegressor_Multiple
from utils.mv_estimator import MeanVariance

from utils.backtesting import WalkForwardBacktester
from utils.custom_loss import SharpeLoss
from utils.features import (getting_trading_universe,
                            getting_data_for_ticker_list,
                            compute_returns,
                            compute_features)

#setting random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def parse_feature(name):
    parts = re.split(r"[_-]", name)
    ticker = parts[-1]              # last part is ticker
    feature = "_".join(parts[:-1])  # everything before ticker

    # optional numeric order if it exists
    m = re.search(r"\d+", feature)
    feature_id = int(m.group()) if m else float("inf")

    return ticker, feature_id, feature


trading_universe = getting_trading_universe()['symbol'].to_list()
ticker_list = [ticker for ticker in trading_universe if ticker != 'SPY']
market_ticker = ["SPY"]

df = getting_data_for_ticker_list(ticker_list + market_ticker)
df = df.dropna(axis=1, how='any')
ticker_list = [col.split('_')[0] for col in df.columns if col != 't_close']

#computing the returns
returns = compute_returns(df, 'simple')
returns = returns.dropna(axis=1)
#computing the features based on returns
X = compute_features(returns, wide=True)
grouped_cols = sorted(X.columns.to_list(), key=parse_feature)
X = X[grouped_cols]
X = X[X.index > "2023-01-01"]

#target
returns = returns.pivot(index='date', columns='stock_name', values='simple_ret')
returns = returns.dropna(axis=1, how='any')
y = returns.shift(-1)[:-1]

#Aligning on all the dates
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]
returns = returns.loc[common_index]
pnl_list = []
sharpe_list = []

for h in [32, 64, 128, 256]:
    for n in [_ for _ in range(1, 4)]:
        print(f'Starting backtesting for {h} hidden dimensions and {n} layers on the LSTM...')
        model = GNNRegressor_Multiple(
                epochs=1,
                window_size=20,
                hidden_dim=h,
                corr_threshold=0.5,
                num_heads=5,
                lr=0.05,
                loss=SharpeLoss(),
                nb_features_per_stock=16,
                drop_out=0.0,
                num_layers_lstm=2,
            )

        bt = WalkForwardBacktester(
            model=model,
            scaler=StandardScaler(),
            mvo=MeanVariance(),
            name=f'{h}_hidden_dim'
        )

        bt.run(X,y,returns)
        pnl_list.append(pd.DataFrame(bt.pnl_))
        sharpe_list.append(bt.pnl_.mean() / bt.pnl_.std())
        print('Backtesting finished')

pnl = pd.concat(pnl_list, axis=1)
cumsum_pnl = pnl.cumsum()

plt.figure(figsize=(12, 6))
for i, (c, s) in enumerate(zip(cumsum_pnl.columns, sharpe_list)):
    plt.plot(
        cumsum_pnl.index,
        cumsum_pnl[c].values,
        label=f"Model {i} – Sharpe: {s:.2f}"
    )
plt.xlabel("Date")
plt.ylabel("PnL")
plt.title("Walk-Forward GNN + MVO Cumulative PnL")
plt.legend()
plt.grid(True)
plt.show()