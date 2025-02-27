import yfinance as yf
import pandas as pd

def fetch_data(stock="SPY", interval="1m", period="5d"):
    """
    Fetches 1-minute historical stock data for the last 5 trading days.
    Ensures each trading day has exactly 390 rows (one per trading min).
    Returns a dictionary of DataFrames, one for each trading day.
    """

    print(f"📈 Fetching {interval} data for {stock} over the past {period}.")

    # Fetch data from Yahoo Finance
    data = yf.download(
        tickers=stock,
        period=period,
        interval=interval,
        group_by="column",
        auto_adjust=True,
        prepost=False,  # Only regular market hours
        threads=True
    )

    if data.empty:
        print("❌ No data fetched. Check ticker, period, or interval.")
        return {}

    # Convert timestamps from UTC to New York time (instead of localizing)
    data.index = data.index.tz_convert('America/New_York')

    # Filter for regular trading hours (9:30 AM - 4:00 PM)
    data = data.between_time("09:30", "16:00")

    # Ensure we have at least 5 unique trading days
    unique_days = data.index.normalize().unique()
    if len(unique_days) < 5:
        print(f"⚠️ Warning: Only {len(unique_days)} trading days detected instead of 5.")

    # Create a dictionary to hold DataFrames for each trading day
    daily_dataframes = {}

    for date in unique_days:
        date_str = date.strftime('%Y-%m-%d')  # Convert date to string to avoid timezone issues

        # Extract data for the specific trading day
        daily_df = data[data.index.normalize() == date].copy()

        # Ensure 390 rows exist
        expected_times = pd.date_range(
            start=f"{date_str} 09:30:00",
            end=f"{date_str} 16:00:00",
            freq="1min",
            tz="America/New_York"
        )

        # Reindex to ensure all time slots exist
        daily_df = daily_df.reindex(expected_times)

        # Fix .fillna() deprecation warning
        daily_df.ffill(inplace=True)  # Use .ffill() instead of fillna(method="ffill")

        # Store in dictionary
        daily_dataframes[date] = daily_df

        print(f"✅ {date_str} - {len(daily_df)} rows fetched.")

    return daily_dataframes

def fetch_premarket_data(stock="SPY", period="5d"):
    """
    Fetches premarket high and low for the last 5 trading days.
    Returns a dictionary with dates as keys and a tuple (premarket_high, premarket_low) as values.
    """
    print(f"📈 Fetching premarket data for {stock} over the past {period}.")

    # Fetch data from Yahoo Finance
    data = yf.download(
        tickers=stock,
        period=period,
        interval="1m",
        group_by="column",
        auto_adjust=True,
        prepost=True,  # Include pre/post market hours
        threads=True
    )

    if data.empty:
        print("❌ No data fetched. Check ticker, period, or interval.")
        return {}

    # Convert timestamps from UTC to New York time (instead of localizing)
    data.index = data.index.tz_convert('America/New_York')

    # Ensure we have at least 5 unique trading days
    unique_days = data.index.normalize().unique()
    if len(unique_days) < 5:
        print(f"⚠️ Warning: Only {len(unique_days)} trading days detected instead of 5.")

    # Create a dictionary to hold premarket high and low for each trading day
    premarket_data = {}

    for date in unique_days:
        # Extract premarket data (4:00 AM - 9:30 AM)
        premarket_window = data.between_time("04:00", "09:30").loc[date.strftime('%Y-%m-%d')]

        if premarket_window.empty:
            print(f"⚠️ Premarket data skipped for {date} (missing data).")
            continue

        # Calculate premarket high and low
        premarket_high = premarket_window["High"].max()
        premarket_low = premarket_window["Low"].min()

        premarket_data[date] = (premarket_high, premarket_low)

    return premarket_data

# Example usage
if __name__ == "__main__":
    daily_data = fetch_data("SPY")
    premarket_data = fetch_premarket_data("SPY")

    if daily_data:
        first_date = list(daily_data.keys())[0]  # Extract the first date key
        first_day_df = daily_data[first_date]    # Get the DataFrame for the first date

        print(f"\n📅 First Trading Day: {first_date}")
        print(first_day_df.head(15))  # Print first 15 rows
    else:
        print("❌ No data available.")

    if premarket_data:
        print("\n📊 Premarket Data:")
        print(f"{'Date':<12} {'High':<10} {'Low':<10}")
        print("-" * 32)
        for date, (high, low) in premarket_data.items():
            high_value = high if not isinstance(high, pd.Series) else high.iloc[0]
            low_value = low if not isinstance(low, pd.Series) else low.iloc[0]
            print(f"{date.strftime('%Y-%m-%d'):<12} {high_value:<10.2f} {low_value:<10.2f}")
    else:
        print("❌ No premarket data available.")