import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.features import (
                               compute_alpha_to_market,
                               compute_beta_to_market,
                               compute_macd, compute_cti,
                               compute_alpha_variance_ratio,
                            )
# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")
from src.graph_nn.gnn_wrapper import GNNRegressor
from utils.features import getting_data_for_ticker_list, getting_trading_universe, compute_returns
from utils.mv_estimator import MeanVariance

import torch
import torch.nn as nn

def Neg_Sharpe(portfolio):
    return -torch.mean(portfolio) / torch.std(portfolio)

class SharpeLoss(nn.Module):
    def __init__(self):
        super(SharpeLoss, self).__init__()
    def forward(self, outputs, future_rets):
        portfolio = outputs * future_rets
        loss = Neg_Sharpe(portfolio)
        return loss


def compute_decile_portfolio(preds, y_test, decile=0.1):
    n_samples, n_stocks = preds.shape
    portfolio_returns = []

    for t in range(n_samples):
        pred_t = preds[t]
        y_t = y_test[t]

        # Rank stocks by predicted return (descending)
        ranks = np.argsort(-pred_t)
        n_top = int(n_stocks * decile)
        n_bottom = int(n_stocks * decile)

        # Select top and bottom decile
        top = ranks[:n_top]
        bottom = ranks[-n_bottom:]

        # Portfolio return: long top, short bottom
        ret = y_t[top].mean() - y_t[bottom].mean()
        portfolio_returns.append(ret)

    portfolio_returns = np.array(portfolio_returns)
    cumulative_returns = np.cumsum(portfolio_returns)
    sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)

    return portfolio_returns, cumulative_returns, sharpe

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


trading_universe = getting_trading_universe()

# NOTE: We select the first 50 stocks for the demonstration
tickers = trading_universe['symbol'].to_list()[:50]

# Download historical prices
df_prices = getting_data_for_ticker_list(tickers)

# Compute Log-Returns (Stationary data is required for ML)
print("      Computing Log-Returns...")
returns = compute_returns(df_prices, return_type='log')

# Pivot to create the Matrix X (Row=Date, Col=Stock)
# This format is essential for the Backtester.
X = returns.pivot(index='date', columns='stock_name', values='log_ret').dropna()

y = X.shift(-1).dropna()
X = X.iloc[:-1]

print(f"Data Ready: {X.shape[0]} days, {X.shape[1]} stocks.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

hidden_dims = [32]
cum_returns_dict = {}
sharpe_dict = {}

for h in hidden_dims:
    print(f"Training model with hidden_dim = {h}")

    model = GNNRegressor(
        epochs=10,
        window_size=20,
        hidden_dim=h,
        corr_threshold=0.7,
        lr=0.001,
        loss=SharpeLoss(),
    )

    model.fit(X_train, y_train, X_train)
    preds = model.predict(X_test)

    if isinstance(X_test, pd.DataFrame):
        y_test_array = X_test.values[-preds.shape[0]:]
    else:
        y_test_array = X_test[-preds.shape[0]:]

    _, cumulative_returns, sharpe = compute_decile_portfolio(preds, y_test_array)
    print(f"Hidden dim {h}: Sharpe = {sharpe:.2f}")
    cum_returns_dict[h] = cumulative_returns
    sharpe_dict[h] = sharpe

    preds_df = pd.DataFrame(preds, index=y_test.index)
    benchmark = (
        Backtester(MeanVariance(), name="benchmark")
        .compute_holdings(preds_df.iloc[20:], preds_df.iloc[20:],pd.DataFrame(y_test_array, index=y_test.index).iloc[20:])
        .compute_pnl(pd.DataFrame(y_test_array, index=y_test.index).iloc[20:])
    )
    pnl = pd.DataFrame(benchmark.pnl_)

plt.figure(figsize=(12, 6))
for h, cum_ret in cum_returns_dict.items():
    plt.plot(cum_ret, label=f"Hidden dim {h} - Sharpe = {sharpe_dict[h]:.2f}")

plt.xlabel("Time Step")
plt.ylabel("Cumulative Returns")
plt.title("Cross-Sectional Momentum: Compare Hidden Dimensions")
plt.legend()
plt.grid(True)
plt.show()
