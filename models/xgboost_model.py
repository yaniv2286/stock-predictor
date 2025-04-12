import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

def prepare_data(df, forecast_days=7, sequence_length=60):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA', 'RSI', 'StochRSI_K', 'StochRSI_D', 'BB_percent']]
    df.dropna(inplace=True)

    data = df.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - forecast_days):
        X.append(scaled_data[i + sequence_length - 1])  # only last step of the sequence
        y.append(scaled_data[i + sequence_length:i + sequence_length + forecast_days, 3])  # Close price

    return np.array(X), np.array(y), scaler, data

def build_model(input_shape=None, forecast_days=7):
    return XGBRegressor(n_estimators=100, objective='reg:squarederror', verbosity=0)

def train_and_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train[:, 0])  # only training for first day forecast
    predictions = model.predict(X_test).reshape(-1, 1)
    return predictions

def predict_future_sequence(model, last_sequence, scaler, data_shape, forecast_days):
    future_pred = model.predict(last_sequence).flatten()
    dummy = np.zeros((forecast_days, data_shape[1]))
    dummy[:, 3] = future_pred
    future_prices = scaler.inverse_transform(dummy)[:, 3]
    return future_prices
