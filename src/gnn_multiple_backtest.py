from utils.features import (getting_trading_universe,
                            getting_data_for_ticker_list,
                            compute_returns,
                            compute_features)
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gnn_wrapper_multiple import GNNRegressor_Multiple
import matplotlib.pyplot as plt
import pandas as pd
from utils.custom_loss import SharpeLoss
import torch.nn as nn
import numpy as np
import torch
import random

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
#target
returns = returns.pivot(index='date', columns='stock_name', values='simple_ret')
returns = returns.dropna(axis=1, how='any')
y = returns.shift(-1)[:-1]

#Aligning on all the dates
common_index = X.index.intersection(y.index)
X = X.loc[common_index]
y = y.loc[common_index]
returns = returns.loc[common_index]

print(returns.shape, X.shape, y.shape)
print(f"      Data Ready: {returns.shape[0]} days, {returns.shape[1]} stocks.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
ret_train, ret_test = train_test_split(returns, test_size=0.2, random_state=42, shuffle=False)

scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    index=X_train.index,
    columns=X_train.columns
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    index=X_test.index,
    columns=X_test.columns
)

hidden_dims = [32]
cum_returns_dict = {}
sharpe_dict = {}

for h in hidden_dims:
    print(f"Training model with hidden_dim = {h}")

    model = GNNRegressor_Multiple(
        epochs=100,
        window_size=20,
        hidden_dim=h,
        corr_threshold=0.5,
        num_heads=5,
        lr=0.05,
        loss=SharpeLoss(),
        nb_features_per_stock=16,
        drop_out=0.0,
        num_layers_lstm=3,
    )

    model.fit(X_train, y_train, returns, X_test, y_test)
    preds = model.predict(X_test)

    if isinstance(ret_test, pd.DataFrame) or isinstance(ret_test, pd.Series):
        ret_test_array = ret_test.values[-preds.shape[0]:]
    else:
        ret_test_array = ret_test[-preds.shape[0]:]

    _, cumulative_returns, sharpe = compute_decile_portfolio(preds, ret_test_array, decile=0.1)
    print(f"Hidden dim {h}: Sharpe = {sharpe:.2f}")
    cum_returns_dict[h] = cumulative_returns
    sharpe_dict[h] = sharpe

    plt.figure(figsize=(12, 8))
    plt.title(f"Training vs Validation for {h} hidden dim")
    plt.plot(model.train_losses, label="Training Loss")
    plt.plot(model.val_losses, label="Validation Loss")
    plt.legend()
    plt.show()


plt.figure(figsize=(12, 6))
for h, cum_ret in cum_returns_dict.items():
    plt.plot(cum_ret, label=f"Hidden dim {h} - Sharpe = {sharpe_dict[h]:.2f}")

plt.xlabel("Time Step")
plt.ylabel("Cumulative Returns")
plt.title("Cross-Sectional Momentum: Compare Hidden Dimensions")
plt.legend()
plt.grid(True)
plt.show()


