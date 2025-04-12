import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from .utils import compute_rsi

def run_backtest():
    st.title("🔁 Backtest: AAPL Strategy (2023–2024)")

    ticker = "AAPL"
    forecast_days = 7
    sequence_length = 60
    initial_cash = 10000

    df = yf.download(ticker, start="2010-01-01", end="2025-01-01")
    df.index = pd.to_datetime(df.index)
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df['RSI'] = compute_rsi(df['Close'])

    df.dropna(inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'EMA', 'RSI']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    train_data = scaled_data[df.index < "2023-01-01"]
    test_data = scaled_data[df.index >= "2023-01-01"]
    test_index = df.index[df.index >= "2023-01-01"]

    def create_sequences(data, seq_len, forecast_len):
        X, y = [], []
        for i in range(len(data) - seq_len - forecast_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len:i+seq_len+forecast_len, 3])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, sequence_length, forecast_days)

    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(forecast_days)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    portfolio, buy_hold, trades = [], [], []
    cash, stock = initial_cash, 0
    buy_price = None
    preds, actuals = [], []

    for i in range(sequence_length, len(test_data) - forecast_days):
        input_seq = test_data[i - sequence_length:i].reshape(1, sequence_length, test_data.shape[1])
        pred = model.predict(input_seq, verbose=0)[0]
        actual = test_data[i:i+forecast_days, 3]
        preds.append(pred[0])
        actuals.append(actual[0])

        today_scaled = test_data[i-1, 3]
        today_close = scaler.inverse_transform([[0,0,0,today_scaled,0,0,0]])[0][3]
        pred_close = scaler.inverse_transform([[0,0,0,pred[0],0,0,0]])[0][3]

        action = "Hold"
        if pred_close > today_close * 1.01:
            if cash > 0:
                stock = cash / today_close
                cash = 0
                action = "Buy"
        else:
            if stock > 0:
                cash = stock * today_close
                stock = 0
                action = "Sell"

        portfolio.append(cash + stock * today_close)
        if i == sequence_length:
            buy_price = today_close
        buy_hold.append(initial_cash * (today_close / buy_price))

        trades.append({
            "Date": test_index[i],
            "Today Close": round(today_close, 2),
            "Predicted Close": round(pred_close, 2),
            "Action": action,
            "Portfolio": round(cash + stock * today_close, 2)
        })

    mae = np.mean(np.abs(np.array(preds) - np.array(actuals)))
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
