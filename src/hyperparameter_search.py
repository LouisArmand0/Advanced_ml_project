import re
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


from sklearn.preprocessing import StandardScaler

from graph_nn.gnn_wrapper import GNNRegressor_Multiple
from utils.mv_estimator import MeanVariance

from utils.backtesting import WalkForwardBacktester
from utils.custom_loss import SharpeLoss
from utils.features import (getting_trading_universe,
                            getting_data_for_ticker_list,
                            compute_returns,
                            compute_features)


def parse_feature(name):
    parts = re.split(r"[_-]", name)
    ticker = parts[-1]  # last part is ticker
    feature = "_".join(parts[:-1])  # everything before ticker

    # optional numeric order if it exists
    m = re.search(r"\d+", feature)
    feature_id = int(m.group()) if m else float("inf")

    return ticker, feature_id, feature


def run_backtest(h, n, X, y, returns):
    logger.info(f"[START] gnn_{h}_{n}")

    torch.manual_seed(42)
    np.random.seed(42)

    model = GNNRegressor_Multiple(
        epochs=150,
        window_size=21,
        hidden_dim=h,
        corr_threshold=0.5,
        num_heads=5,
        lr=0.05,
        loss=SharpeLoss(),
        nb_features_per_stock=21,
        drop_out=0.0,
        num_layers_lstm=n,
    )

    bt = WalkForwardBacktester(
        model=model,
        scaler=StandardScaler(),
        mvo=MeanVariance(),
        name=f"gnn_{h}_{n}"
    )

    bt.run(X, y, returns)

    pnl = pd.DataFrame(bt.pnl_, columns=[f"gnn_{h}_{n}"])
    sharpe = bt.pnl_.mean() / bt.pnl_.std()

    logger.info(f"[END] gnn_{h}_{n} | Sharpe={sharpe:.3f}")

    return h, n, pnl, sharpe

# setting random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# init logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    logger.info('Loading data...')
    trading_universe = getting_trading_universe()['symbol'].to_list()
    ticker_list = [ticker for ticker in trading_universe if ticker != 'SPY']
    market_ticker = ["SPY"]

    df = getting_data_for_ticker_list(ticker_list + market_ticker)
    df = df.dropna(axis=1, how='any')
    ticker_list = [col.split('_')[0] for col in df.columns if col != 't_close']
    logger.info(f'Data loaded for {len(ticker_list)} tickers')

    # computing the returns
    returns = compute_returns(df, 'simple')
    returns = returns.dropna(axis=1)

    # computing the features based on returns
    logger.info('Computing the features...')
    X = compute_features(df, returns, wide=True)
    grouped_cols = sorted(X.columns.to_list(), key=parse_feature)
    X = X[grouped_cols]
    X = X[X.index > "2012-01-01"]

    # target
    logger.info(f'Computing the target')
    returns = returns.pivot(index='date', columns='stock_name', values='simple_ret')
    returns = returns.dropna(axis=1, how='any')

    #VOLATILITY TARGET
    TARGET_VOL = 0.15
    returns = (returns / returns.std()) * TARGET_VOL

    y = returns.shift(-1)[:-1]

    # Aligning on all the dates
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    returns = returns.loc[common_index]


    params = [(h, n) for h in [32] for n in range(3, 4)]

    pnl_list = []
    sharpe_dict = {}

    # limit workers if RAM is tight (e.g. max_workers=4)
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_backtest, h, n, X, y, returns)
            for h, n in params
        ]

        for future in as_completed(futures):
            h, n, pnl, sharpe = future.result()
            pnl_list.append(pnl)
            sharpe_dict[(h, n)] = sharpe

    pnl = pd.concat(pnl_list, axis=1)
    cumsum_pnl = pnl.cumsum()

    plt.figure(figsize=(12, 6))

    for (h, n), sharpe in sharpe_dict.items():
        col = f"gnn_{h}_{n}"
        plt.plot(
            cumsum_pnl.index,
            cumsum_pnl[col].values,
            label=f"{col} – Sharpe: {sharpe:.2f}"
        )

    plt.xlabel("Date")
    plt.ylabel("PnL")
    plt.title("Walk-Forward GNN + MVO Cumulative PnL")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cumulative_pnl.png", dpi=300, bbox_inches="tight")
    plt.close()