import pandas as pd
import numpy as np
import xgboost as xgb
from functools import reduce
from src.utils.features import (compute_vol_adjusted_returns,
                               compute_alpha_to_market,
                               compute_beta_to_market,
                               compute_macd, compute_cti,
                               compute_alpha_variance_ratio,
                               getting_trading_universe,
                               getting_data_for_ticker_list)
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

def negative_portfolio_sharpe_loss(R: np.ndarray):
    """
    Custom loss to minimize portfolio sharpe loss
    """
    # To ensure numerical stability and avoid division by zero
    eps = 1e-8

    T,N = R.shape

    def obj(preds_flat, dtrain):
        preds = preds_flat.reshape(T,N)
        p = (preds * R).sum(axis=1) # vector of portfolio returns

        mean = np.mean(p)
        var = ((p - mean) ** 2).sum() / (T - 1) # unbiased variance estimator
        std = np.sqrt(var + eps)

        # Gradient wrt p_t
        dL_dp = (-1 / (T * std)) + (mean * (p - mean)) / (T * (std ** 3))

        # Chain rule: grad wrt s_{t,i}
        grad = dL_dp.to_numpy()[:, None] * R.values # shape (T, N)

        # Approximate hessian
        hess = np.ones_like(grad)

        return grad, hess

    return obj

def softmax_longshort(x, temp=1.0):
    x = x - x.mean()
    z = (x - np.max(x)) / temp
    e = np.exp(z)
    pos = e / e.sum()

    z_neg = (-x - np.max(-x)) / temp
    e_neg = np.exp(z_neg)
    neg = e_neg / e_neg.sum()

    w = pos - neg
    return w / np.sum(np.abs(w))

if __name__ == "__main__":

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

    beta_to_mkt_list = []
    for lookback in [126, 252]:
        temp = compute_beta_to_market(
            ret_df=vol_adj_ret,
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
            ret_df=vol_adj_ret,
            ticker_market="spy",
            lookback=lookback
        ))
    alpha_to_mkt = alpha_to_mkt_list[0]
    for alpha_df in alpha_to_mkt_list[1:]:
        alpha_to_mkt = pd.merge(alpha_to_mkt, alpha_df, on=['date', 'stock_name'], how='inner')

    macd_list = []
    for c in [(8, 24), (16, 48), (32, 64)]:
        macd_list.append(compute_macd(
            ret_df=vol_adj_ret,
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
                ret_df=vol_adj_ret,
                lookback=c,
            )
        )
    cti = cti_list[0]
    for cti_df in cti_list[1:]:
        cti = pd.merge(cti, cti_df, on=['date', 'stock_name'], how='inner')

    alpha_var_list = []
    for lookback in [5, 21, 63, 126, 252]:
        alpha_var_list.append(compute_alpha_variance_ratio(
            ret_df=vol_adj_ret,
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
    X_full = X_full.sort_values(by=['date'], ascending=True).reset_index(drop=True)

    y_full = pd.merge(vol_adj_ret, X_full, on=['date', 'stock_name'], how='inner')[['date', 'stock_name', 'vol_adjusted_ret']]
    y_full = y_full.sort_values(by=['date', 'stock_name']).copy()
    y_full['target'] = y_full.groupby('stock_name')['vol_adjusted_ret'].shift(-1)
    y_full = y_full.dropna().reset_index(level=0, drop=True)

    X_full = X_full[X_full['date'] <= np.max(y_full['date'])]

    dates = y_full['date'].sort_values().unique()
    T_total, N, d = len(dates), len(X_full.stock_name.unique()), len(X_full.columns)
    np.random.seed(42)

    # Total window size
    val_size = 252     # last 5 days used as validation
    step = 252
    num_round = 100
    TARGET_VOL = 0.15
    params = {
        'objective': 'rank:pairwise',  # predict relative returns
        'eta': 0.05,
        'max_depth': 3,
        'eval_metric': 'auc',  # for continuous returns; use 'ndcg' if labels are integer grades
        'tree_method': 'hist',
        'verbosity': 1
    }
    logger.info('Features computed')
    logger.info('Starting the training loop...')
    val_sharpe = []
    for start_idx in range(252, T_total - val_size, step):
        train_end_idx = start_idx
        val_start_idx = train_end_idx
        val_end_idx = val_start_idx + val_size

        # Select dates
        train_dates = dates[:train_end_idx]  # expanding window
        val_dates = dates[val_start_idx:val_end_idx]

        # Training data
        train_mask = y_full['date'].isin(train_dates)
        X_train = X_full.loc[train_mask].drop(columns=['date', 'stock_name'], axis=1)
        y_train = y_full.loc[train_mask]['target'].values

        # Group sizes per date
        train_group = y_full.loc[train_mask].groupby('date').size().to_list()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(train_group)

        # Validation data
        val_mask = y_full['date'].isin(val_dates)
        X_val = X_full.loc[val_mask].drop(columns=['date', 'stock_name'], axis=1)
        y_val = y_full.loc[val_mask]['target'].values

        val_group = y_full.loc[val_mask].groupby('date').size().to_list()
        dval = xgb.DMatrix(X_val, label=y_val)
        dval.set_group(val_group)

        # Train model
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=num_round,
            evals=[(dtrain, "train"), (dval, "val")],
            verbose_eval=False
        )

        # Predictions on validation set
        preds = bst.predict(dval)
        val_df = y_full.loc[val_mask][['date','target']]
        val_df['pred'] = preds

        # Compute weights per date

        weights = []
        for date, group in val_df.groupby('date'):
            w = softmax_longshort(group['pred'].values)
            weights.append(pd.Series(w, index=group.index))

        val_df['weight'] = pd.concat(weights).sort_index()
        portfolio_returns = (val_df['weight'] * val_df['target']).groupby(val_df['date']).sum()

        realized_vol = portfolio_returns.std(ddof=1) * np.sqrt(252)
        scaling = TARGET_VOL / (realized_vol + 1e-8)
        portfolio_returns_scaled = portfolio_returns
        sharpe = portfolio_returns_scaled.mean() / (portfolio_returns_scaled.std(ddof=1) + 1e-8)
        val_sharpe.append(sharpe)
        logger.info(f"Fold {start_idx}: Validation Sharpe = {sharpe:.4f}")

    plt.figure(figsize=(12, 8))
    plt.plot(val_sharpe)
    plt.show()