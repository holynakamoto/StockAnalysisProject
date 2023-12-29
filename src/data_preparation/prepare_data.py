# src/data_preparation/prepare_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(stock_data):
    """
    Cleans the provided stock data.
    :param stock_data: DataFrame or dictionary of DataFrames containing stock data.
    :return: Cleaned DataFrame or dictionary of DataFrames.
    """
    if isinstance(stock_data, dict):
        return {ticker: _clean_single_df(df) for ticker, df in stock_data.items()}
    else:
        return _clean_single_df(stock_data)

def _clean_single_df(df):
    """
    Cleans a single DataFrame of stock data.
    :param df: DataFrame of stock data.
    :return: Cleaned DataFrame.
    """
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    """
    Adds technical indicators to the stock data.
    :param df: DataFrame of stock data.
    :return: DataFrame with added technical indicators.
    """
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df

def normalize_data(df):
    """
    Normalizes the stock data.
    :param df: DataFrame of stock data.
    :return: Normalized DataFrame.
    """
    df_normalized = (df - df.min()) / (df.max() - df.min())
    return df_normalized
