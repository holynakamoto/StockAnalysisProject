# src/analysis/correlation.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_correlation_matrix(df):
    """
    Calculates the correlation matrix for the given DataFrame.
    :param df: DataFrame containing numerical data.
    :return: DataFrame representing the correlation matrix.
    """
    df_clean = df.dropna()  # Ensuring no NaN values
    return df_clean.corr()

def plot_correlation_matrix(df, title="Correlation Matrix", figsize=(10, 8), annot=True):
    """
    Plots a heatmap of the correlation matrix.
    :param df: DataFrame containing the correlation matrix.
    :param title: Title of the plot.
    :param figsize: Size of the figure.
    :param annot: Flag to annotate the heatmap with correlation values.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=annot, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title(title)
    plt.show()
