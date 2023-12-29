# src/data_collection/fetch_data.py

import yfinance as yf
from datetime import datetime

def download_stock_data(ticker, start_date, end_date=datetime.now()):
    """
    Downloads historical data for a single stock ticker.

    :param ticker: Stock ticker symbol as a string.
    :param start_date: Start date for data retrieval as a datetime object.
    :param end_date: End date for data retrieval as a datetime object. Defaults to current date.
    :return: DataFrame with the stock's historical data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def download_multiple_stocks(tickers, start_date, end_date=datetime.now()):
    """
    Downloads historical data for multiple stock tickers.

    :param tickers: List of stock ticker symbols.
    :param start_date: Start date for data retrieval as a datetime object.
    :param end_date: End date for data retrieval as a datetime object. Defaults to current date.
    :return: Dictionary of DataFrames, each containing data for a ticker.
    """
    stock_data = {}
    for ticker in tickers:
        stock_data[ticker] = download_stock_data(ticker, start_date, end_date)
    return stock_data
