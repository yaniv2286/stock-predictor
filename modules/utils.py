import pandas as pd

def compute_rsi(series, period=14):
    """
    Computes the Relative Strength Index (RSI) for a given price series.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_indicators(df):
    """
    Adds common technical indicators to the DataFrame:
    EMA, RSI, Stochastic RSI (%K, %D), and Bollinger %B.
    """
    df['EMA'] = df['Close'].ewm(span=20).mean()
    df['RSI'] = compute_rsi(df['Close'])

    # Stochastic RSI
    rsi = df['RSI']
    min_rsi = rsi.rolling(window=14).min()
    max_rsi = rsi.rolling(window=14).max()
    df['StochRSI_K'] = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)
    df['StochRSI_D'] = df['StochRSI_K'].rolling(window=3).mean()

    # Bollinger Bands %B
    bb_mean = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    df['BB_percent'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)

    df.dropna(inplace=True)
    return df
