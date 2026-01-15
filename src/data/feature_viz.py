import matplotlib.pyplot as plt
import re
import logging
from src.utils.features import (getting_trading_universe,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger()


if __name__ == "__main__":
    logger.info('Loading data...')
    trading_universe = getting_trading_universe()['symbol'].to_list()
    ticker_list = [ticker for ticker in trading_universe if ticker != 'SPY']
    market_ticker = ["SPY"]

    df = getting_data_for_ticker_list(ticker_list + market_ticker)
    df = df.dropna(axis=1, how='any')

    logger.info(f'Data loaded for {df.stock_name.nunique()} tickers')

    # computing the returns
    returns = compute_returns(df, 'simple')
    returns = returns.dropna(axis=1)

    # computing the features based on returns
    logger.info('Computing the features...')
    X = compute_features(returns, wide=True)
    grouped_cols = sorted(X.columns.to_list(), key=parse_feature)
    X = X[grouped_cols]
    X = X[X.index > "2012-01-01"]

    # target
    logger.info(f'Computing the target')
    returns = returns.pivot(index='date', columns='stock_name', values='simple_ret')
    returns = returns.dropna(axis=1, how='any')
    y = returns.shift(-1)[:-1]

    # Aligning on all the dates
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    returns = returns.loc[common_index]

    plt.rcParams['font.family'] =  'serif'

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    axs[0, 0].plot(X.index, X['alpha_to_market_5_MSFT'])
    axs[0, 0].set_title("Alpha to the SP 500 with a 5 days rolling window")

    axs[0, 1].plot(X.index, X['cti_63_MSFT'])
    axs[0, 1].set_title("Correlation Trend Indicator - 63 days")

    axs[1, 0].plot(X.index, X['macd_8_24_MSFT'])
    axs[1, 0].set_title("MACD (8,24)")

    axs[1, 1].plot(X.index, X['beta_to_market_252_MSFT'])
    axs[1, 1].set_title("Beta to SP500 on a 252 rolling window")

    plt.tight_layout()
    plt.show()


