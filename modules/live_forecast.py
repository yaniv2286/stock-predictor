# modules/live_forecast.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models import lstm_model, random_forest_model, xgboost_model
from modules.utils import compute_rsi

MODELS = {
    "LSTM": lstm_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model
}

def run_live_forecast(model_name):
    st.title(f"📈 Live Stock Forecast ({model_name})")
    model_module = MODELS[model_name]

    ticker = st.text_input("Enter Stock Ticker:", value="AAPL", max_chars=10)
    forecast_days = 7
    sequence_length = 60

    if st.button("Predict"):
        with st.spinner("Fetching data and training model..."):
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

            df.dropna(inplace=True)

            # Prepare data using the selected model
            X, y, scaler, raw_data = model_module.prepare_data(df, forecast_days, sequence_length)
            split = int(0.8 * len(X))
            X_train, y_train, X_test, y_test = X[:split], y[:split], X[split:], y[split:]

            model = model_module.build_model(X.shape[1:], forecast_days)
            predictions = model_module.train_and_predict(X_train, y_train, X_test, model)


            first_day_preds = predictions[:, 0]
            first_day_actuals = y_test[:, 0]

            y_pred_scaled = np.zeros((len(first_day_preds), raw_data.shape[1]))
            y_test_scaled = np.zeros((len(first_day_actuals), raw_data.shape[1]))
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

            if len(X[-1].shape) == 1:  # Means it's flattened
                last_seq = X[-1].reshape(1, -1)
            else:
                last_seq = X[-1].reshape(1, sequence_length, raw_data.shape[1])
            future_prices = model_module.predict_future_sequence(model, last_seq, scaler, raw_data.shape, forecast_days)

            future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
            forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_prices})
            st.subheader(f"📅 {forecast_days}-Day Forecast")
            st.table(forecast_df)

            st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), file_name="forecast.csv")
