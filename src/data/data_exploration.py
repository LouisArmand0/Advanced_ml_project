import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TICKERS_DIR = ROOT / "data/tickers"



if __name__ == '__main__':
    tickers = ['AAPL', 'MSFT']

    l = []
    for ticker in tickers:
        df = pd.read_parquet(TICKERS_DIR / f"{ticker}.parquet")
        df_close = df.xs('Close', level='Price', axis=1)
        df_close.columns = [f'{ticker.lower()}_close']
        l.append(df_close)

    df = pd.concat(l, axis=1)
    print(df.head())
    print(df.info())
