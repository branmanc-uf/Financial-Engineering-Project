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


#first_date = list(daily_data.keys())[0]  # Extract the first date key
#first_day_df = daily_data[first_date]
#print(f"\n📅 First Trading Day: {first_date}")
#print(first_day_df.head(20))
# Fetch data
daily_data = fetch_data("SPY")

# Check if data was returned
if daily_data:
    first_date = list(daily_data.keys())[0]  # Extract the first date key
    first_day_df = daily_data[first_date]    # Get the DataFrame for the first date

    print(f"\n📅 First Trading Day: {first_date}")
    print(first_day_df.head(15))  # Print first 15 rows
else:
    print("❌ No data available.")

daily_dataframes.head(5)