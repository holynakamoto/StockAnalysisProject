# src/visualization/plot_data.py

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

def plot_stock_prices(df, ticker, figsize=(10, 6)):
    """
    Plots stock prices (Close, Open, High, Low) for a given stock.

    :param df: DataFrame containing stock data.
    :param ticker: Stock ticker symbol as a string.
    :param figsize: Tuple representing figure size.
    """
    df[['Close', 'Open', 'High', 'Low']].plot(figsize=figsize, title=f"Stock Prices for {ticker}")
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.show()

def plot_volume(df, ticker, figsize=(10, 4)):
    """
    Plots trading volume for a given stock.

    :param df: DataFrame containing stock data.
    :param ticker: Stock ticker symbol as a string.
    :param figsize: Tuple representing figure size.
    """
    df['Volume'].plot(figsize=figsize, title=f"Trading Volume for {ticker}")
    plt.ylabel('Volume')
    plt.xlabel('Date')
    plt.show()

def plot_moving_averages(df, ticker, window_sizes=[50, 200], figsize=(10, 6)):
    """
    Plots moving averages for a given stock.

    :param df: DataFrame containing stock data.
    :param ticker: Stock ticker symbol as a string.
    :param window_sizes: List of integers representing moving average window sizes.
    :param figsize: Tuple representing figure size.
    """
    plt.figure(figsize=figsize)
    plt.title(f"Moving Averages for {ticker}")
    plt.plot(df['Close'], label='Close Price', alpha=0.5)
    for window in window_sizes:
        plt.plot(df[f'MA{window}'], label=f'MA for {window} days')
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.show()
