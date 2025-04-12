# modules/backtest_module.py

import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from modules.utils import compute_rsi, add_indicators

def run_backtest(model_name):
    st.title(f"🔁 Backtest: AAPL Strategy (2023–2024) using {model_name}")

    ticker = st.sidebar.text_input("Enter Stock Ticker:", value="AAPL").upper()

    forecast_days = 7
    sequence_length = 60
    initial_cash = 10000

    # Import model module
    from models import lstm_model, random_forest_model, xgboost_model
    model_module = {
        "LSTM": lstm_model,
        "Random Forest": random_forest_model,
        "XGBoost": xgboost_model
    }[model_name]

    # Download data and preprocess
    df = yf.download(ticker, start="2010-01-01", end="2025-01-01")
    df.index = pd.to_datetime(df.index)
    df = add_indicators(df)

    X, y, scaler, raw_data = model_module.prepare_data(df, forecast_days, sequence_length)

    # Split training and testing based on date
    split_index = df.index.get_indexer([df.index[df.index >= "2023-01-01"][0]])[0]
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]
    test_dates = df.index[split_index + sequence_length + forecast_days:]

    model = model_module.build_model(X.shape[1:], forecast_days)
    predictions = model_module.train_and_predict(X_train, y_train, X_test, model)

    portfolio, buy_hold, trades = [], [], []
    cash, stock = initial_cash, 0
    buy_price = None

    predicted_prices = predictions[:, 0]
    actual_prices = y_test[:, 0]

    # Inverse transform for visualization and trading logic
    dummy = np.zeros((len(predicted_prices), raw_data.shape[1]))
    dummy[:, 3] = predicted_prices
    pred_close = scaler.inverse_transform(dummy)[:, 3]

    dummy[:, 3] = actual_prices
    actual_close = scaler.inverse_transform(dummy)[:, 3]

    for i in range(len(pred_close)):
        today_close = actual_close[i]
        pred_price = pred_close[i]

        action = "Hold"
        if pred_price > today_close * 1.01:
            if cash > 0:
                stock = cash / today_close
                cash = 0
                action = "Buy"
        else:
            if stock > 0:
                cash = stock * today_close
                stock = 0
                action = "Sell"

        portfolio_val = cash + stock * today_close
        portfolio.append(portfolio_val)
        if i == 0:
            buy_price = today_close
        buy_hold.append(initial_cash * (today_close / buy_price))

        trades.append({
            "Date": test_dates[i],
            "Today Close": round(today_close, 2),
            "Predicted Close": round(pred_price, 2),
            "Action": action,
            "Portfolio": round(portfolio_val, 2)
        })

    mae = np.mean(np.abs(pred_close - actual_close))
    strategy_final = portfolio[-1]
    buyhold_final = buy_hold[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Strategy", f"${strategy_final:,.2f}")
    col2.metric("📉 Buy & Hold", f"${buyhold_final:,.2f}")
    col3.metric("🎯 MAE", f"${mae:.2f}")

    st.subheader("📊 Portfolio vs Buy & Hold")
    chart_df = pd.DataFrame({
        "Strategy": portfolio,
        "Buy & Hold": buy_hold
    })
    st.line_chart(chart_df)

    st.subheader("📝 Trade Log")
    trade_df = pd.DataFrame(trades)
    st.dataframe(trade_df.set_index("Date"))

    st.download_button(
        label="📥 Download Trade Log",
        data=trade_df.to_csv(index=False),
        file_name="backtest_trades.csv",
        mime="text/csv"
    )
