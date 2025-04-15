import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data():
    print("📈 Fetching 1m data for SPY over the past 5d.")
    df = yf.download('SPY', period='5d', interval='1m')
    print(f"✅ {df.index.date[-1]} - {len(df)} rows fetched.\n")
    return df

def calculate_orb(df):
    # Convert index to local time (assuming NY time zone for SPY)
    df.index = df.index.tz_convert('America/New_York')
    
    print(f"📅 First Trading Day: {df.index[0]}")

    # Calculate the ORB for the first 15 minutes of trading (9:30 to 9:45)
    open_range = df.between_time('09:30', '9:45')
    print("Open range data:\n", open_range.head())

    # Check if 'High' and 'Low' columns exist
    if 'High' not in open_range.columns or 'Low' not in open_range.columns:
        print("Error: 'High' or 'Low' columns are missing in the open range data.")
        return df

    if open_range.empty:
        print("Warning: No data available for the first 15 minutes of trading.")
        return df  # Returning without setting ORB_High/ORB_Low if no data is available

    # Debugging: print open_range data
    print("Open range data:\n", open_range)

    # Get the first 15 rows of data from the open range data
    open_range_first_15 = open_range.iloc[:15]

    print(open_range_first_15.head())
    # Check if 'High' and 'Low' columns exist in the first 15 rows

    # Grab only the 'High' and 'Low' columns
    high_df = open_range_first_15[['High']]
    low_df = open_range_first_15[['Low']]

    print("High DataFrame:\n", high_df)
    print("Low DataFrame:\n", low_df)

    # Find the max from the high_df and min from the low_df
    orb_high = high_df['High'].max()
    orb_low = low_df['Low'].min()

    print(f"ORB_High: {orb_high}, \n ORB_Low: {orb_low}")
    
    # Fill the ORB values for the rest of the day <- we have big error here dawg
    df['ORB_High'] = orb_high
    df['ORB_Low'] = orb_low
    
    # Debugging: print ORB values for review
    print("ORB_High before fill:\n", df['ORB_High'].head())
    print("ORB_Low before fill:\n", df['ORB_Low'].head())

    # Fill missing values in ORB_High and ORB_Low with forward fill
    df[['ORB_High', 'ORB_Low']] = df[['ORB_High', 'ORB_Low']].ffill()

    # Debugging: print ORB values after fill
    print("ORB_High after fill:\n", df['ORB_High'].head())
    print("ORB_Low after fill:\n", df['ORB_Low'].head())

    return df

def generate_signals(df):
    # Generating buy signals when price crosses above ORB_High
    df.loc[(df['Close'] > df['ORB_High']) & (df['Close'].shift(1) <= df['ORB_High']), 'Signal'] = 'BUY'
    
    # Generating sell signals when price crosses below ORB_Low
    df.loc[(df['Close'] < df['ORB_Low']) & (df['Close'].shift(1) >= df['ORB_Low']), 'Signal'] = 'SELL'

    return df

def main():
    # Fetching data
    df = fetch_data()
    
    # Calculating ORB values
    df = calculate_orb(df)
    
    # Check if ORB_High and ORB_Low exist before calling dropna
    if 'ORB_High' in df.columns and 'ORB_Low' in df.columns:
        df = df.dropna(subset=['ORB_High', 'ORB_Low'])
    else:
        print("Error: ORB_High or ORB_Low columns are missing.")
        return

    # Generating signals based on ORB
    df = generate_signals(df)

    # Printing the signals dataframe
    print(df[['Close', 'ORB_High', 'ORB_Low', 'Signal']].head())

if __name__ == "__main__":
    main()