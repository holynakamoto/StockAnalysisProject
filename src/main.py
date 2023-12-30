import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.lstm_model import create_lstm_model, train_model, evaluate_model
from analysis.risk_return import (
    calculate_annualized_return,
    calculate_volatility,
    plot_risk_return,
)
from data_collection.fetch_data import download_stock_data
from data_preparation.prepare_data import clean_data, add_indicators, to_supervised
from visualization.plot_data import plot_volume, plot_stock_prices, plot_moving_averages
from visualization.plot_settings import set_plot_style, set_figure_size
from analysis.correlation import calculate_correlation_matrix, plot_correlation_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




def propose_trades(actual, predicted):
    """
    Proposes trades based on actual and predicted values.
    """
    trades = pd.DataFrame({"actual": actual, "predicted": predicted})
    trades["signal"] = np.where(
        trades["predicted"] > trades["actual"].shift(1), "Buy", "Sell"
    )
    trades["returns"] = trades["actual"].pct_change()
    trades["strategy_returns"] = trades["returns"] * trades["signal"].shift(1).eq(
        "Buy"
    ).astype(int)
    return trades.dropna()


def main():
    print("Starting the program...")
    ticker = "GBTC"
    start_date = "2020-01-01"
    raw_data = download_stock_data(ticker, start_date)

    raw_data = clean_data(raw_data)
    raw_data = add_indicators(raw_data)

    X = raw_data[["Open", "High", "Low", "Volume"]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(X)

    reframed = to_supervised(scaled, 1, 1)
    train_size = int(len(reframed) * 0.7)
    train = reframed.iloc[:train_size, :]
    test = reframed.iloc[train_size:, :]

    train_X, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    test_X, test_y = test.iloc[:, :-1], test.iloc[:, -1]

    # Correlation Analysis
    correlation_matrix = calculate_correlation_matrix(raw_data)
    set_figure_size(10, 8)  # Setting figure size for better visibility
    plot_correlation_matrix(correlation_matrix)

    # LSTM model creation and training
    model = create_lstm_model((train_X.shape[1], 1))
    history = train_model(model, train_X, train_y, test_X, test_y)

    # Make predictions
    train_predict = model.predict(train_X)
    test_predict = model.predict(test_X)

    # Evaluate model
    train_rmse, test_rmse, train_mae, test_mae = evaluate_model(
        model, train_X, train_y, test_X, test_y
    )

    # Evaluate model performance
    train_rmse = np.sqrt(mean_squared_error(train_y, train_predict))
    test_rmse = np.sqrt(mean_squared_error(test_y, test_predict))
    train_mae = mean_absolute_error(train_y, train_predict)
    test_mae = mean_absolute_error(test_y, test_predict)
    train_r2 = r2_score(train_y, train_predict)
    test_r2 = r2_score(test_y, test_predict)

    print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")
    print(f"Train R2: {train_r2}, Test R2: {test_r2}")

    # Evaluate model with additional metric
    test_r2 = r2_score(actual_test_y, test_predict)
    print(f"Test R2: {test_r2}")

    # Trading strategy
    trades = propose_trades(test_y, model.predict(test_X))
    annualized_return = calculate_annualized_return(trades["strategy_returns"])
    volatility = calculate_volatility(trades["strategy_returns"])
    plot_risk_return(annualized_return, volatility)

    # Visualization
    set_plot_style("ggplot")
    plot_volume(raw_data, ticker)
    plot_stock_prices(raw_data, ticker)
    plot_moving_averages(raw_data, ticker, [20, 50, 100])


if __name__ == "__main__":
    main()
