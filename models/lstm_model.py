# models/lstm_model.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def prepare_data(df, forecast_days=7, sequence_length=60):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA', 'RSI', 'StochRSI_K', 'StochRSI_D', 'BB_percent']]
    df.dropna(inplace=True)

    data = df.values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - forecast_days):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length:i + sequence_length + forecast_days, 3])  # Close price

    return (
        np.array(X),
        np.array(y),
        scaler,
        data  # Needed for inverse_transform
    )

def build_model(input_shape, forecast_days):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(forecast_days)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_predict(X_train, y_train, X_test, model):
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    predictions = model.predict(X_test)
    return predictions

def predict_future_sequence(model, last_sequence, scaler, data_shape, forecast_days):
    future_pred = model.predict(last_sequence)[0]
    dummy = np.zeros((forecast_days, data_shape[1]))
    dummy[:, 3] = future_pred
    future_prices = scaler.inverse_transform(dummy)[:, 3]
    return future_prices
