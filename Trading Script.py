import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the fetch_data function from fetching_yfinance_data.py
from fetching_yfinance_data import fetch_data

def calculate_indicators(df):
    """
    Calculate VWAP and EMA for the given DataFrame.
    """
    df['Cumulative Price x Volume'] = (df['Close'] * df['Volume']).cumsum()
    df['Cumulative Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative Price x Volume'] / df['Cumulative Volume']
    
    ema_period = 8
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    
    return df

def determine_orb_levels(daily_data):
    """
    Determine Open Range Breakout (ORB) levels based on the first 15 minutes (9:30 - 9:45 AM) 
    for each trading day in `daily_data` dictionary.
    """
    for date, df in daily_data.items():
        # Extract the first 15 rows (first 15 minutes of trading)
        orb_window = df.iloc[:15].dropna()  # Drop NaN rows to prevent errors

        if orb_window.empty:
            print(f"⚠️ ORB skipped for {date} (missing data).")
            continue

        # Calculate ORB levels
        orb_high = orb_window["High"].max()  # Max high in first 15 minutes
        orb_low = orb_window["Low"].min()    # Min low in first 15 minutes

        # Fix misalignment issue (remove "Ticker" label if present)
        df["ORB_High"] = orb_high.values[0] if isinstance(orb_high, pd.Series) else orb_high
        df["ORB_Low"] = orb_low.values[0] if isinstance(orb_low, pd.Series) else orb_low

        print(f"📊 ORB for {date}: High = {df['ORB_High'].iloc[0]}, Low = {df['ORB_Low'].iloc[0]}")

    return daily_data

def generate_signals(df):
    """
    Generate buy/sell signals based on ORB breakout levels.
    """
    required_columns = ['Close', 'ORB_High', 'ORB_Low']
    if df.empty or any(col not in df.columns for col in required_columns):
        print("❌ ORB levels missing. Cannot generate signals.")
        df['Signal'] = 0
        return df

    df = df.copy()  # Prevent chained assignment warnings

    # Fill forward ORB values to avoid NaNs (fix deprecated method warning)
    df['ORB_High'] = df['ORB_High'].ffill()
    df['ORB_Low'] = df['ORB_Low'].ffill()

    df['Signal'] = 0
    df.loc[(df['Close'] > df['ORB_High']) & (df['Close'].shift(1) <= df['ORB_High']), 'Signal'] = 1
    df.loc[(df['Close'] < df['ORB_Low']) & (df['Close'].shift(1) >= df['ORB_Low']), 'Signal'] = -1

    return df

def plot_signals(df, stock, date):
    """
    Plot the stock price and highlight buy/sell signals.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
    
    # Plot buy signals
    plt.scatter(df.index[df['Signal'] == 1], df['Close'][df['Signal'] == 1], label='Buy Signal', marker='^', color='green', alpha=1, lw=3)

    # Plot sell signals
    plt.scatter(df.index[df['Signal'] == -1], df['Close'][df['Signal'] == -1], label='Sell Signal', marker='v', color='red', alpha=1, lw=3)

    plt.title(f'{stock} Price Chart with ORB Signals ({date})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def main():
    stock = 'SPY'

    # Fetch Data (now returns a dictionary of DataFrames, one per trading day)
    daily_data = fetch_data(stock)

    if not daily_data:
        print("❌ No data fetched. Exiting program.")
        return

    # Apply ORB Calculation
    daily_data = determine_orb_levels(daily_data)

    for date, df in daily_data.items():
        print(f"\n🔍 Processing Data for {date}:")
        df = calculate_indicators(df)
        df = generate_signals(df)
        print(df[['Close', 'VWAP', 'EMA', 'ORB_High', 'ORB_Low', 'Signal']].head(20))
        plot_signals(df, stock, date)

if __name__ == "__main__":
    main()
