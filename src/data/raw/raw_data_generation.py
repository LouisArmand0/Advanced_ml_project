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
    trading_universe.to_csv("trading_universe.csv")

    #Generate the csv file of historical prices for the considered trading universe
    tickers_list = get_tickers_sector_from_wp()['symbol'].to_list() + ["^GSPC"] #also download S&P 500 prices
    historical_prices = get_historical_prices(tickers_list, start_date='2010-01-01', end_date='2025-12-31')

    if isinstance(historical_prices.index, pd.PeriodIndex):
        historical_prices.index = historical_prices.index.to_timestamp()
    historical_prices = historical_prices.sort_index(axis=1)
    historical_prices.columns = [f"{ticker.lower()}_{field.lower()}" for ticker, field in historical_prices.columns]
    historical_prices.columns = [
        col.replace("^gspc", "spy") for col in historical_prices.columns
    ]
    historical_prices.to_csv("historical_prices.csv")