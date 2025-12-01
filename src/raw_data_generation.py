import numpy as np
import pandas as pd
import wikipedia as wp
import yfinance as yf
from typing import List

import unicodedata
import re

def clean_text(s):
    if pd.isna(s):
        return s

    s = unicodedata.normalize("NFKD", s)
    # Remove footnote markers [1], [a], etc.
    s = re.sub(r"\[.*?\]", "", s)
    # Replace weird spaces like \xa0
    s = s.replace("\xa0", " ")
    # Strip whitespace
    return s.strip()

def get_tickers_sector_from_wp(index_name: str = 'S&P 100'):
    """
    :param index_name: Name of index to get stocks tickers from
    :return: DataFrame of ticker, company name, and the sector
    """

    # Getting tickers and sectors from the wikipedia page
    reader = wp.page(index_name).html().encode('UTF-8')
    stocks = pd.read_html(reader)[2]

    #Cleaning and renaming
    stocks = stocks[stocks["Symbol"] != "GOOG"]
    stocks.loc["GOOGL", "Name"] = "Alphabet"
    stocks["Symbol"] = stocks["Symbol"].str.replace(".", "-", regex=False)
    stocks.columns = [col.lower() for col in stocks.columns]

    for col in stocks.columns:
        stocks[col] = stocks[col].apply(clean_text)

    return stocks.dropna(axis=0)


def get_historical_prices(stock_list: List[str], start_date: str, end_date: str):
    """
    :param stock_list: List of stock to get historical prices
    :return: Dataframe of historical prices (open, low, high, close, volume)
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

    #Generate and safe the csv file of the tickers list and the company sector
    trading_universe = get_tickers_sector_from_wp()
    trading_universe.to_csv("../data/trading_universe.csv", index=False)

    #Generate the csv file of historical prices for the considered trading universe
    tickers_list = get_tickers_sector_from_wp()['symbol'].to_list()
    historical_prices = get_historical_prices(tickers_list, start_date='2025-01-01', end_date='2025-12-31')
    historical_prices.to_csv("../data/raw/historical_prices.csv")