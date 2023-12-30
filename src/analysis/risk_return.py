# src/analysis/risk_return.py

import pandas as pd
import matplotlib.pyplot as plt

def calculate_annualized_return(returns, period='daily'):
    """
    Calculates the annualized return for the given return series.
    :param returns: Pandas Series containing return values.
    :param period: The period of the stock data ('daily', 'monthly', etc.).
    :return: Annualized return.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("Returns should be a pandas Series.")

    if period not in ['daily', 'monthly']:
        raise ValueError("Unsupported period. Choose 'daily' or 'monthly'.")

    df_pct_change = returns.pct_change()
    df_pct_change.dropna(inplace=True)  # Handle NaN values

    if period == 'daily':
        return df_pct_change.mean() * 252
    elif period == 'monthly':
        return df_pct_change.mean() * 12

    """
    Calculates the annualized return for the given DataFrame.
    :param df: DataFrame containing stock prices.
    :param period: The period of the stock data ('daily', 'monthly', etc.).
    :return: Annualized return.
    """
    df_pct_change = df.pct_change()
    df_pct_change.dropna(inplace=True)  # Handle NaN values

    if period == 'daily':
        return df_pct_change.mean() * 252
    elif period == 'monthly':
        return df_pct_change.mean() * 12
    else:
        raise ValueError("Unsupported period. Choose 'daily' or 'monthly'.")

def calculate_volatility(df, period='daily'):
    """
    Calculates the annualized volatility for the given DataFrame.
    :param df: DataFrame containing stock prices.
    :param period: The period of the stock data ('daily', 'monthly', etc.).
    :return: Annualized volatility.
    """
    df_pct_change = df.pct_change()
    df_pct_change.dropna(inplace=True)  # Handle NaN values

    if period == 'daily':
        return df_pct_change.std() * (252 ** 0.5)
    elif period == 'monthly':
        return df_pct_change.std() * (12 ** 0.5)
    else:
        raise ValueError("Unsupported period. Choose 'daily' or 'monthly'.")

def plot_risk_return(annualized_return, volatility):
    """
    Plots a risk-return scatter plot for given annualized return and volatility.
    :param annualized_return: Annualized return as a single float value.
    :param volatility: Annualized volatility as a single float value.
    """
    plt.scatter(volatility, annualized_return)
    plt.title("Risk vs. Return")
    plt.xlabel("Annualized Volatility (Risk)")
    plt.ylabel("Annualized Return")
    plt.show()

