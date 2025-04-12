from . import lstm_model as run_lstm_forecast
from . import random_forest_model as run_rf_forecast
from . import xgboost_model as run_xgb_forecast


models = {
    "LSTM": run_lstm_forecast,
    "Random Forest": run_rf_forecast,
    "XGBoost": run_xgb_forecast,
}
