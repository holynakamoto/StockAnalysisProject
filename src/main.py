import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
# Assuming the following modules are correctly implemented
from models.lstm_model import create_lstm_model, train_model, evaluate_model
from data_collection.fetch_data import download_stock_data
from data_preparation.prepare_data import clean_data, add_indicators
from sklearn.model_selection import train_test_split

def main():
    print("Starting the program...")

    # Load data
    ticker = "AAPL"
    start_date = "2020-01-01" 
    raw_data = download_stock_data(ticker, start_date)
    
    # Extract feature columns
    X = raw_data[['Open','High','Low','Volume']]

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(X)

    # Split into train and test sets
    train_size = int(len(scaled) * 0.7)
    train, test = scaled[0:train_size, :], scaled[train_size:len(scaled), :]

    # Convert dataset to supervised learning format
    def to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # Input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # Forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # Put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # Drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # Frame as supervised learning
    reframed = to_supervised(scaled, 1, 1)
    print("Data reframed for supervised learning.")

    # Split into input and outputs
    n_obs = reframed.shape[1] - 1
    train_X, train_y = reframed.iloc[:train_size, :n_obs], reframed.iloc[:train_size, -1]
    test_X, test_y = reframed.iloc[train_size:, :n_obs], reframed.iloc[train_size:, -1]

    # Reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.values.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.values.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print("Input data reshaped for LSTM.")

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    print("LSTM model defined and compiled.")

    # Fit the LSTM model
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # Make predictions
    train_predict = model.predict(train_X)
    test_predict = model.predict(test_X)

    # Convert train_y and test_y from Pandas Series to NumPy arrays and reshape
    train_y_reshaped = train_y.values.reshape(-1, 1)
    test_y_reshaped = test_y.values.reshape(-1, 1)

    # Inverse transform predictions and actual values
    train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], 3))), axis=1))[:, 0]
    test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], 3))), axis=1))[:, 0]
    actual_train_y = scaler.inverse_transform(np.concatenate((train_y_reshaped, np.zeros((train_y_reshaped.shape[0], 3))), axis=1))[:, 0]
    actual_test_y = scaler.inverse_transform(np.concatenate((test_y_reshaped, np.zeros((test_y_reshaped.shape[0], 3))), axis=1))[:, 0]


    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(actual_train_y, train_predict))
    test_rmse = np.sqrt(mean_squared_error(actual_test_y, test_predict))
    print('Train RMSE: %.3f' % train_rmse)
    print('Test RMSE: %.3f' % test_rmse)

    # Propose trade strategy
    def propose_trades(actual, predicted):
        trades = pd.DataFrame({'actual': actual, 'predicted': predicted})
        trades['signal'] = np.where(trades['predicted'] > trades['actual'].shift(1), 'Buy', 'Sell')
        trades['returns'] = trades['actual'].pct_change()
        trades['strategy_returns'] = trades['returns'] * trades['signal'].shift(1).eq('Buy').astype(int)
        return trades.dropna()

    # Apply trade strategy
    trades = propose_trades(actual_test_y, test_predict)
    print(trades.head())

    # Plot trade signals
    plt.figure(figsize=(12, 6))
    plt.plot(trades['actual'], label='Actual Price', alpha=0.5)
    buy_signals = trades[trades['signal'] == 'Buy']
    sell_signals = trades[trades['signal'] == 'Sell']
    plt.plot(buy_signals.index, trades['actual'][buy_signals.index], '^', markersize=10, color='g', label='Buy Signal')
    plt.plot(sell_signals.index, trades['actual'][sell_signals.index], 'v', markersize=10, color='r', label='Sell Signal')
    plt.title('Trade Signals')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Calculate and plot cumulative returns
    trades['cumulative_returns'] = (1 + trades['strategy_returns']).cumprod() - 1
    plt.figure(figsize=(12, 6))
    plt.plot(trades['cumulative_returns'], label='Strategy Returns')
    plt.plot(trades['returns'].cumsum(), label='Buy and Hold Returns')
    plt.title('Cumulative Returns')
    plt.xlabel('Day')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

# ... [Rest of the main function] ...

if __name__=='__main__':
    main()


if __name__=='__main__':
    main()
