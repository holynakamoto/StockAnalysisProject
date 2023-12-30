# src/data_preprocessing/preprocess_data.py
import pandas as pd
from data_collection.download_data import download_stock_data

# Get downloaded data
start_date = "2015-01-01"
end_date = "2023-01-01"
all_data = download_stock_data(start_date, end_date)


def prepare_data(df):
    df = clean(df)
    df = add_indicators(df)
    return df


prepared_data = {}
for ticker, df in all_data.items():

    df = prepare_data(df)
    prepared_data[ticker] = df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs and drop remaining"""
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adds technical indicators"""
    df["RSI"] = RSI(df["Close"])
    df["SMA50"] = df["Close"].rolling(50).mean()
    return df


def to_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Formats as supervised learning"""
    # Shift features for input/output format
    df["t+1"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    return df
