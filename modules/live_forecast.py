import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from .utils import compute_rsi

def run_live_forecast():
    st.title("📈 Live LSTM Stock Forecast")

    ticker = st.text_input("Enter Stock Ticker:", value="AAPL", max_chars=10)
    forecast_days = 7
    sequence_length = 60

    if st.button("Predict"):
        with st.spinner("Fetching and training..."):
            df = yf.download(ticker, start="2000-01-01")
            df['EMA'] = df['Close'].ewm(span=20).mean()
            df['RSI'] = compute_rsi(df['Close'])

            # Stochastic RSI
            rsi = df['RSI']
            min_rsi = rsi.rolling(window=14).min()
            max_rsi = rsi.rolling(window=14).max()
            df['StochRSI_K'] = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
            df['StochRSI_D'] = df['StochRSI_K'].rolling(window=3).mean()

            # Bollinger %B
            bb_mean = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            df['BB_percent'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

            df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA', 'RSI', 'StochRSI_K', 'StochRSI_D', 'BB_percent']]
            df.dropna(inplace=True)

            data = df.values
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)

            def create_sequences(data, seq_len, forecast_len):
                X, y = [], []
                for i in range(len(data) - seq_len - forecast_len):
                    X.append(data[i:i+seq_len])
                    y.append(data[i+seq_len:i+seq_len+forecast_len, 3])
                return np.array(X), np.array(y)

            X, y = create_sequences(scaled_data, sequence_length, forecast_days)
            split = int(0.8 * len(X))
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]

            # First-day forecast evaluation
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(forecast_days)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

            predictions = model.predict(X_test)
            first_day_preds = predictions[:, 0]
            first_day_actuals = y_test[:, 0]

            y_pred_scaled = np.zeros((len(first_day_preds), data.shape[1]))
            y_test_scaled = np.zeros((len(first_day_actuals), data.shape[1]))
            y_pred_scaled[:, 3] = first_day_preds
            y_test_scaled[:, 3] = first_day_actuals

            predicted_prices = scaler.inverse_transform(y_pred_scaled)[:, 3]
            actual_prices = scaler.inverse_transform(y_test_scaled)[:, 3]

            error = np.mean(np.abs(predicted_prices - actual_prices))
            st.success(f"✅ Avg. prediction error: ${error:.2f}")

            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(actual_prices, label='Actual')
            ax.plot(predicted_prices, label='Predicted')
            ax.legend()
            ax.set_title(f"{ticker} Forecast - First Day Accuracy")
            st.pyplot(fig)

            # Predict next 7 days from last known
            last_seq = scaled_data[-sequence_length:].reshape(1, sequence_length, data.shape[1])
            future_pred = model.predict(last_seq)[0]

            dummy = np.zeros((forecast_days, data.shape[1]))
            dummy[:, 3] = future_pred
            future_prices = scaler.inverse_transform(dummy)[:, 3]

            dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
            future_df = pd.DataFrame({'Date': dates, 'Predicted Close Price': future_prices})
            st.subheader(f"📅 {forecast_days}-Day Forecast")
            st.table(future_df)

            st.download_button("📥 Download Forecast CSV", future_df.to_csv(index=False), file_name="forecast.csv")
