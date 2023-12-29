"""LSTM models for time series forecasting"""

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout  

def create_lstm_model(input_shape, units=50, dropout=0.2):
    """Create LSTM network for sequence forecasting
    
    Arguments:
        input_shape {tuple} -- input dimension shape
        units {int} -- number of LSTM neurons  
        dropout {float} -- dropout rate
        
    Returns:
        model {keras Sequential} -- compiled LSTM model
    """

    model = Sequential()
    model.add(LSTM(units, dropout=dropout, input_shape=input_shape)) 
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    
    return model

def train_model(model, x_train, y_train):
   """Train LSTM model
    
    Arguments:
        model {keras model} -- LSTM model
        x_train {array} -- training features 
        y_train {array} -- training labels
   """
    
   # Validate input data  
   if len(x_train.shape) != 3:
        raise ValueError('Training data must be 3D (samples, timesteps, features)')
    
   # Fit model   
   history = model.fit(x_train, y_train, epochs=10) 
   return history
   
def evaluate_model(model, x_test, y_test):
    """Evaluate model on test data"""
    
    # Generate predictions
    preds = model.predict(x_test)
    
    # Print samples
    print("Predicted:", preds[:5]) 
    print("Actual:", y_test[:5])

    return preds
    
if __name__ == '__main__':
    
    # Sample data
    x_train = np.random.random((1000, 60, 1))
    y_train = np.random.random((1000,))
    
    # Train model
    model = create_lstm_model(input_shape=(60,1)) 
    history = train_model(model, x_train, y_train)
    
    # Plot loss  
    plt.plot(history.history['loss']) 
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")