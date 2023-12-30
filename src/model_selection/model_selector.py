# src/model_selection/model_selector.py
from sklearn.ensemble import RandomForestRegressor 
from keras.models import Sequential
from keras.layers import SimpleRNN as KerasSimpleRNN, LSTM as KerasLSTM
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Base Model
class BaseModel:
    def __init__(self):
        pass 

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

# Various models
class SimpleRNN(BaseModel):
    def __init__(self, units=64):
        self.model = Sequential()
        self.model.add(KerasSimpleRNN(units))

    def fit(self, X, y):
        # You should implement the fitting logic here using X and y
        pass
        
    def predict(self, X):
        # You should implement the prediction logic here using X
        pass

class LSTM(BaseModel):
    def __init__(self, units=64):  
        self.model = Sequential()
        self.model.add(KerasLSTM(units))

    def fit(self, X, y):
        # You should implement the fitting logic here using X and y
        pass
        
    def predict(self, X):
        # You should implement the prediction logic here using X
        pass

class ARIMAModel(BaseModel):  
    def __init__(self, order=(1, 0, 0)):
        self.model = ARIMA(order=order)

    def fit(self, X, y):
        # You should implement the fitting logic here using X and y
        pass
        
    def predict(self, X):
        # You should implement the prediction logic here using X
        pass

class SARIMAXModel(BaseModel):
    def __init__(self, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)):
        self.model = SARIMAX(order=order, seasonal_order=seasonal_order)

    def fit(self, X, y):
        # You should implement the fitting logic here using X and y
        pass
        
    def predict(self, X):
        # You should implement the prediction logic here using X
        pass

class RandomForestRegressorModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
