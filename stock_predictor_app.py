import streamlit as st
from modules.live_forecast import run_live_forecast
from modules.backtest_module import run_backtest

st.set_page_config(page_title="📈 Stock Predictor", layout="wide")

st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose Mode:", ["📈 Live Forecast", "🔁 Backtest"])

if page == "📈 Live Forecast":
    run_live_forecast()

elif page == "🔁 Backtest":
    run_backtest()
