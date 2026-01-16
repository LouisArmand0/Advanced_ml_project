import numpy as np
import pandas as pd
import wikipedia as wp
import yfinance as yf
from typing import List

import unicodedata
import re
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


DATA_DIR = os.path.join(ROOT, "data")
TICKERS_DIR = os.path.join(DATA_DIR, "tickers")

os.makedirs(TICKERS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def clean_text(s):
    if pd.isna(s):
        return s

    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\[.*?\]", "", s)
    s = s.replace("\xa0", " ")
    return s.strip()

def get_tickers_sector_from_wp(index_name: str = 'S&P 100'):
    """
    :param index_name: Name of index to get stocks tickers from
    :return: DataFrame of ticker, company name, and the sector
    """

    # Getting tickers and sectors from the wikipedia page
    reader = wp.page(index_name).html().encode('UTF-8')
    stocks = pd.read_html(reader)[2]

    # Cleaning and renaming
    stocks = stocks[stocks["Symbol"] != "GOOG"]
    stocks.loc["GOOGL", "Name"] = "Alphabet"
    stocks["Symbol"] = stocks["Symbol"].str.replace(".", "-", regex=False)
    stocks.columns = [col.lower() for col in stocks.columns]

    for col in stocks.columns:
        stocks[col] = stocks[col].apply(clean_text)

    return stocks.dropna(axis=0)


def get_historical_prices(stock_list: List[str], start_date: str, end_date: str):
    """
    :param stock_list: list of stock to get historical prices
    :return: dataframe of historical prices (open, low, high, close, volume)
    for a given tickers list
    """
    df = yf.download(
        tickers=" ".join(stock_list),
        start=start_date,
        end=end_date,
        actions=False,
        group_by="ticker"
    )
    return df

if __name__ == "__main__":

    # Generate and save the csv file of the tickers list and the company sector
    trading_universe = get_tickers_sector_from_wp()
    trading_universe.to_parquet(f"{DATA_DIR}/trading_universe.parquet", index=True)

    # Generate the csv file of historical prices for the considered trading universe
    tickers_list = get_tickers_sector_from_wp()['symbol'].to_list() + ["^GSPC"] # Also download S&P 500 prices

    for ticker in tickers_list:
        historical_prices = get_historical_prices([ticker], start_date='2010-01-01', end_date='2025-12-31')
        if ticker == "^GSPC":
            ticker = "SPY"
        historical_prices.to_parquet(os.path.join(TICKERS_DIR, f"{ticker}.parquet"), index=True)