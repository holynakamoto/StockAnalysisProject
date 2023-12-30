# src/data_collection/download_data.py
from data_collection.download_data import download_stock_data
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Download data
start_date = "2020-01-01"
all_data = download_stock_data(start_date)

prepared_data = {}


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Scales and cleans dataframe"""

    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    return pd.DataFrame(scaled, columns=df.columns)


for ticker, df in all_data.items():

    df = prepare_data(df)

    prepared_data[ticker] = df




