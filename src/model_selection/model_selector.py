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
        self.model.compile(loss="mean_squared_error", optimizer="adam")
        self.model.fit(X, y, epochs=10)  # You can adjust the number of epochs as needed

    def predict(self, X):
        return self.model.predict(X)


class LSTM(BaseModel):
    def __init__(self, units=64):
        self.model = Sequential()
        self.model.add(KerasLSTM(units))

    def fit(self, X, y):
        self.model.compile(loss="mean_squared_error", optimizer="adam")
        self.model.fit(X, y, epochs=10)  # You can adjust the number of epochs as needed

    def predict(self, X):
        return self.model.predict(X)


class ARIMAModel(BaseModel):
    def __init__(self, order=(1, 0, 0)):
        self.model = ARIMA(order=order)

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        # Assuming X is a pandas DataFrame with appropriate columns, you can use the fitted ARIMA model to predict
        return self.model.predict(start=X.index[0], end=X.index[-1], dynamic=False)


class SARIMAXModel(BaseModel):
    def __init__(self, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)):
        self.model = SARIMAX(order=order, seasonal_order=seasonal_order)

    def fit(self, X, y):
        self.model = self.model.fit(X, y)

    def predict(self, X):
        # Assuming X is a pandas DataFrame with appropriate columns, you can use the fitted SARIMAX model to predict
        return self.model.predict(start=X.index[0], end=X.index[-1], dynamic=False)


class RandomForestRegressorModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
