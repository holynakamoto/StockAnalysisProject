# src/models/lstm_model.py

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np

def create_lstm_model(input_shape, units=50, dropout=0.2, output_units=1):
    """
    Creates an LSTM model for time series prediction.

    :param input_shape: The shape of the training dataset.
    :param units: The number of neurons in the LSTM layer.
    :param dropout: The dropout rate for regularization.
    :param output_units: The number of output units.
    :return: Compiled LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units))
    model.add(Dense(output_units))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(model, x_train, y_train, batch_size=1, epochs=10):
    """
    Trains the LSTM model.

    :param model: The LSTM model to train.
    :param x_train: Training data features.
    :param y_train: Training data labels.
    :param batch_size: The size of the batch.
    :param epochs: The number of epochs to train for.
    :return: Trained model.
    """
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    return model

def make_predictions(model, x_test):
    """
    Uses the LSTM model to make predictions.

    :param model: The trained LSTM model.
    :param x_test: Test data features.
    :return: Predictions.
    """
    predictions = model.predict(x_test)
    return predictions
