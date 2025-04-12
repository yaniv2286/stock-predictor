import streamlit as st
from modules.live_forecast import run_live_forecast
from modules.backtest_module import run_backtest
from models import lstm_model, random_forest_model, xgboost_model

# Model mapping
MODELS = {
    "LSTM": lstm_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model,
}

# Streamlit page settings
st.set_page_config(page_title="📈 Stock Predictor", layout="wide")

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox("Choose Mode:", ["📈 Live Forecast", "🔁 Backtest"])

# Model selector
model_name = st.sidebar.selectbox("Select Model:", list(MODELS.keys()))

# Routing
if page == "📈 Live Forecast":
    run_live_forecast(model_name)

elif page == "🔁 Backtest":
    run_backtest(model_name)
