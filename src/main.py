# src/main.py

import numpy as np 
from datetime import datetime
from src.data_collection.fetch_data import download_stock_data, download_multiple_stocks
from src.data_preparation.prepare_data import clean_data, add_technical_indicators, normalize_data
from src.analysis.correlation import calculate_correlation_matrix, plot_correlation_matrix
from src.analysis.risk_return import calculate_annualized_return, calculate_volatility, plot_risk_return
from src.models.lstm_model import create_lstm_model, train_lstm_model, make_predictions
from src.visualization.plot_data import plot_stock_prices, plot_volume, plot_moving_averages
from src.visualization.plot_settings import set_plot_style, set_figure_size
from sklearn.preprocessing import MinMaxScaler

def main():
    # Set global plot settings
    set_plot_style()
    set_figure_size()

    single_stock_data = download_stock_data("AAPL", datetime(2020, 1, 1))
    print("Single Stock Data fetched.")

    multiple_stocks_data = download_multiple_stocks(["AAPL", "MSFT", "GOOG"], datetime(2020, 1, 1))
    print("Multiple Stocks Data fetched.")

    cleaned_data_single = clean_data(single_stock_data)
    with_indicators_single = add_technical_indicators(cleaned_data_single)
    normalized_data_single = normalize_data(with_indicators_single)
    print("Single Stock Data prepared.")

    cleaned_data_multiple = clean_data(multiple_stocks_data)
    with_indicators_multiple = {ticker: add_technical_indicators(df) for ticker, df in cleaned_data_multiple.items()}
    normalized_data_multiple = {ticker: normalize_data(df) for ticker, df in with_indicators_multiple.items()}
    print("Multiple Stocks Data prepared.")

    correlation_matrix_single = calculate_correlation_matrix(normalized_data_single)
    plot_correlation_matrix(correlation_matrix_single, title="Correlation Matrix - Single Stock")
    print("Correlation analysis for single stock completed.")

    annualized_return_single = calculate_annualized_return(normalized_data_single)
    volatility_single = calculate_volatility(normalized_data_single)
    plot_risk_return(normalized_data_single, period='daily')
    print("Risk-return analysis for single stock completed.")

    # Data Preparation for LSTM Model
    data = single_stock_data.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil(len(dataset) * 0.95))

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape = (len(train), 60 timesteps, 1 feature)
# print(x_train.shape)
# Create the test data set
test_data = scaled_data[training_data_len - 60:, :]
# Split the data into x_test and y_test
x_test = []
y_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])
    if i <= 61:
        print(x_test)
        print(y_test)
        print()
        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        print(x_test.shape)
        print(y_test.shape)
        # Create and train the LSTM model
        model = create_lstm_model(input_shape=x_train.shape[1:])
        trained_model = train_lstm_model(model, x_train, y_train)
        predictions = make_predictions(trained_model, x_test)
        print("LSTM model predictions:", predictions)
        print(x_test.shape)
        print(y_test.shape)
        print(predictions.shape)


    # LSTM Model - Placeholder for actual data preparation
    # x_train, y_train, x_test = prepare_data_for_lstm(normalized_data_single)
    # model = create_lstm_model(input_shape=x_train.shape[1:])
    # trained_model = train_lstm_model(model, x_train, y_train)
    # predictions = make_predictions(trained_model, x_test)
    # print("LSTM model predictions:", predictions)

    # Visualization of Stock Data
    plot_stock_prices(cleaned_data_single, "AAPL")
    plot_volume(cleaned_data_single, "AAPL")
    plot_moving_averages(with_indicators_single, "AAPL")

if __name__ == "__main__":
    main()
