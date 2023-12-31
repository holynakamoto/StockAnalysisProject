import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from src.model_selection.model_selector import (
    SimpleRNN,
    LSTM,
    ARIMAModel,
    SARIMAXModel,
    RandomForestRegressorModel,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from itertools import product
from scipy import stats
import shap
import datetime
import os
import seaborn as sns

# Load your prepared time series data (replace 'your_data.csv' with the actual data file)
data = pd.read_csv("your_data.csv", parse_dates=["Date"], index_col="Date")

# Set a random seed for reproducibility
random_seed = 42

# Specify the number of time steps to use for the test split
test_steps = 100  # Adjust this number based on your dataset

# Specify the percentage of data for the validation split
validation_split_percentage = 0.2

# Specify the number of lags to analyze for prediction lags
max_lag = 10

# Specify the number of cross-validation folds
n_splits = 5  # Adjust as needed

# Specify exogenous variables if applicable (e.g., 'exog1', 'exog2', ...)
exogenous_vars = []

# Specify the directory for logging evaluation metrics
log_dir = "logs"

# Specify the directory for storing scenario analysis results
scenario_dir = "scenario_results"

# Specify the directory to save the results
results_dir = "results"

# Create the log, scenario, and results directories if they do not exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(scenario_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Split the data into training, validation, and test sets based on time
train_data = data.iloc[:-test_steps]
test_data = data.iloc[-test_steps:]

# Split the training data into features (X_train) and target (y_train)
X_train = train_data.drop(
    columns=["TargetColumn"]
)  # Replace 'TargetColumn' with your actual target column name
y_train = train_data[
    "TargetColumn"
]  # Replace 'TargetColumn' with your actual target column name

# Split the test data into features (X_test) and target (y_test)
X_test = test_data.drop(
    columns=["TargetColumn"]
)  # Replace 'TargetColumn' with your actual target column name
y_test = test_data[
    "TargetColumn"
]  # Replace 'TargetColumn' with your actual target column name

# Perform additional preprocessing, such as scaling (you can customize this for specific models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize TimeSeriesSplit for time series cross-validation
tscv = TimeSeriesSplit(n_splits=n_splits)

# Select and train different models
models = [
    SimpleRNN(),
    LSTM(),
    ARIMAModel(),
    SARIMAXModel(),
    RandomForestRegressorModel(),
]


# Function to analyze prediction lags
def analyze_lags(predictions, actuals, max_lag):
    lags = list(range(1, max_lag + 1))
    mse_lags = []

    for lag in lags:
        shifted_preds = predictions[:-lag]
        shifted_actuals = actuals[lag:]
        mse_lag = mean_squared_error(shifted_actuals, shifted_preds)
        mse_lags.append(mse_lag)

    return lags, mse_lags


# Function to calculate and visualize ACF and PACF plots
def plot_acf_pacf(series):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, lags=max_lag, ax=axes[0])
    plot_pacf(series, lags=max_lag, ax=axes[1])
    plt.show()

# Function to generate new features based on autocorrelation analysis
def generate_autocorrelation_features(series, max_lag):
    acf_values = acf(series, nlags=max_lag)
    pacf_values = pacf(series, nlags=max_lag)

    features = {}
    for lag in range(1, max_lag + 1):
        features[f"ACF_lag{lag}"] = acf_values[lag]
        features[f"PACF_lag{lag}"] = pacf_values[lag]

    return pd.Series(features)


# Function to log evaluation metrics
def log_metrics(model_name, mean_mse, std_mse, optimal_lag):
    log_filename = os.path.join(log_dir, f"{model_name}_metrics.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_filename, "a") as log_file:
        log_file.write(
            f"{timestamp}: Model {model_name} - Mean MSE: {mean_mse}, Std MSE: {std_mse}, Optimal Lag: {optimal_lag}\n"
        )


# Function to plot actual vs. forecasted values for bias analysis
def plot_actual_vs_forecast_scatter(actuals, forecasts, model_name):
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, forecasts, alpha=0.5)
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.title(f"Actual vs. Forecasted Returns for {model_name}")
    plt.grid()
    plt.show()


# Function to plot residuals distribution for error analysis
def plot_residuals_distribution(residuals, model_name):
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Residuals Distribution for {model_name}")
    plt.grid()
    plt.show()


# Function to plot model learning curves (loss vs. epoch)
def plot_learning_curves(history, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Model Learning Curves for {model_name}")
    plt.legend()
    plt.grid()
    plt.show()


# Function to calculate trading metrics
def calculate_trading_metrics(predictions):
    daily_returns = predictions.pct_change().fillna(0)
    cumulative_returns = (1 + daily_returns).cumprod()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    return daily_returns, sharpe_ratio, max_drawdown


# Function to calculate and visualize trading metrics
def calculate_and_visualize_trading_metrics(daily_returns, model_name):
    daily_returns, sharpe_ratio, max_drawdown = calculate_trading_metrics(daily_returns)

    print(f"Sharpe Ratio for {model_name}: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown for {model_name}: {max_drawdown * 100:.2f}%")

    # Visualize cumulative returns and drawdowns
    cumulative_returns = (1 + daily_returns).cumprod()
    drawdowns = (cumulative_returns / cumulative_returns.cummax() - 1) * 100

    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_returns.index, cumulative_returns, label="Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.title(f"Cumulative Returns Over Time for {model_name}")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(drawdowns.index, drawdowns, label="Drawdowns (%)", color="red")
    plt.xlabel("Date")
    plt.ylabel("Drawdowns (%)")
    plt.title(f"Drawdowns Over Time for {model_name}")
    plt.legend()
    plt.grid()
    plt.show()


# Dictionary to store actuals, forecasts, and SHAP values
results = {}

for model in models:
    model_name = model.__class__.__name__
    print(f"Evaluating {model_name}...\n")

    mse_values = []  # To store MSE values for each cross-validation fold
    forecasts = []  # To store forecasts for each fold
    shap_values = []  # To store SHAP values for each fold

    for train_idx, val_idx in tscv.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        print(f"Training {model_name}...")
        history = model.fit(
            X_train_fold,
            y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            early_stopping=True,
        )

        print(f"Predicting {model_name}...")
        predictions = model.predict(X_val_fold)

        mse_fold = mean_squared_error(y_val_fold, predictions)
        mse_values.append(mse_fold)

        # Explain model predictions with SHAP
        explainer = shap.Explainer(model, X_train_fold)
        shap_values_fold = explainer.shap_values(X_val_fold)
        shap_values.extend(shap_values_fold)

        # Store the forecasts for the validation set
        forecasts.extend(predictions)

        # Plot actual vs. forecasted values for bias analysis
        plot_actual_vs_forecast_scatter(y_val_fold, predictions, model_name)

        # Calculate and visualize trading metrics
        daily_returns, sharpe_ratio, max_drawdown = calculate_trading_metrics(
            pd.Series(predictions, index=y_val_fold.index)
        )
        calculate_and_visualize_trading_metrics(daily_returns, model_name)

        # Plot residuals distribution for error analysis
        residuals = y_val_fold - predictions
        plot_residuals_distribution(residuals, model_name)

        # Plot model learning curves (loss vs. epoch)
        plot_learning_curves(history, model_name)

    # Calculate the mean and standard deviation of MSE values across folds
    mean_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)

    print(f"Mean Squared Error for {model_name}: {mean_mse} ± {std_mse}\n")

    # Analyze prediction lags for the full training set
    print(f"Analyzing prediction lags for {model_name} on full training set...")
    predictions = model.predict(X_train_scaled)
    lags, mse_lags = analyze_lags(predictions, y_train, max_lag)
    optimal_lag = lags[np.argmin(mse_lags)]
    print(
        f"Optimal Lag for {model_name}: {optimal_lag} (MSE: {mse_lags[optimal_lag-1]})\n"
    )

    # Plot ACF and PACF plots for the full training set
    print(f"Generating ACF and PACF plots for {model_name} on full training set...")
    plot_acf_pacf(y_train)

    # Generate new features based on autocorrelation analysis
    print(
        f"Generating new features based on autocorrelation analysis for {model_name}..."
    )
    autocorr_features = generate_autocorrelation_features(y_train, max_lag)
    print("Generated Features:")
    print(autocorr_features)

    # Log evaluation metrics
    log_metrics(model_name, mean_mse, std_mse, optimal_lag)

    # Store actuals, forecasts, and SHAP values for further analysis
    results[model_name] = {
        "actuals": y_train,
        "forecasts": predictions,
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "optimal_lag": optimal_lag,
        "autocorr_features": autocorr_features,
        "shap_values": shap_values,
    }

    # Plot actual vs. forecasted values for the full training set
    plot_actual_vs_forecast_scatter(y_train, predictions, model_name)

    # Calculate and visualize trading metrics for the full training set
    daily_returns, sharpe_ratio, max_drawdown = calculate_trading_metrics(
        pd.Series(predictions, index=y_train.index)
    )
    calculate_and_visualize_trading_metrics(daily_returns, model_name)

    # Plot residuals distribution for the full training set
    residuals = y_train - predictions
    plot_residuals_distribution(residuals, model_name)

    # Plot model learning curves (loss vs. epoch) for the full training set
    history = model.fit(X_train_scaled, y_train, epochs=100, verbose=0)
    plot_learning_curves(history, model_name)

# Save the results as a pickle file for use by the training strategy module
results_filename = os.path.join(results_dir, "results.pkl")
pd.to_pickle(results, results_filename)
print(f"Results saved to {results_filename}")
