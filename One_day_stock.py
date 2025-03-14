# Import modules
from polygon import RESTClient
import pandas as pd
import pytz
from Stock_API import main

# API key (Replace with your actual API key)
API_KEY = "N5TPWLXWaGR0IN3_ttrmMROH_q9b2yBi"

# Create client and authenticate
client = RESTClient(API_KEY)

# Define stock symbol and date range
stockTicker = "SPY"
start_date = "2025-02-27"  # Ensure market was open
end_date = "2025-02-27"

# Fetch 1-minute interval bars
dataRequest = client.get_aggs(
    ticker=stockTicker,
    multiplier=1,
    timespan="minute",
    from_=start_date,
    to=end_date,
    adjusted=True
)

# Convert API response to DataFrame
df = pd.DataFrame(dataRequest)
def classify_missing_minutes(df):
    """
    Ensures the timestamp is properly decoded and detects missing minutes.

    Parameters:
    - df (DataFrame): DataFrame containing a 'timestamp' column in milliseconds.

    Returns:
    - DataFrame: DataFrame with correctly decoded timestamps and missing time classification.
    """
    # 🔹 Convert 'timestamp' from milliseconds to UTC DateTime
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # 🔹 Convert UTC to Eastern Time (ET)
    eastern = pytz.timezone("US/Eastern")
    df["datetime"] = df["datetime_utc"].dt.tz_convert(eastern)

    # 🔹 Ensure timestamps are sorted correctly
    df = df.sort_values(by="datetime").reset_index(drop=True)

    # 🔹 Calculate time differences between consecutive rows
    df["time_diff"] = df["datetime"].diff().dt.total_seconds() / 60  # Convert to minutes

    # 🔹 Classify missing minutes: If the gap is greater than 1 minute, flag it
    df["missing_minutes"] = df["time_diff"] > 1

    return df[['close', 'datetime']]  # Return DataFrame with properly decoded timestamps

df_2 = classify_missing_minutes(df)

def fill_missing_minutes(df):
    """
    Adds missing minute timestamps to the DataFrame and fills missing prices
    with the previous row's value.

    Parameters:
    - df (DataFrame): DataFrame with a 'datetime' index and 'close' prices.

    Returns:
    - DataFrame: Complete DataFrame with all 1-minute timestamps and missing prices filled.
    """
    # 🔹 Ensure 'datetime' is set as the index
    df = df.set_index("datetime")

    # 🔹 Generate a full range of timestamps from the first to the last recorded timestamp
    full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="min")

    # 🔹 Reindex the DataFrame to include all minutes and fill missing values using forward fill
    df = df.reindex(full_time_range, method="ffill")

    # 🔹 Reset index and rename back to 'datetime'
    df = df.reset_index().rename(columns={"index": "datetime"})

    return df

df_3 = fill_missing_minutes(df_2)

import time  # Needed for adding delay
from datetime import datetime, timedelta

























def fetch_multiple_days(symbol: str, start_date: str, end_date: str):
    """
    Fetches stock data for multiple days and combines the results.

    Parameters:
    - symbol (str): Stock ticker symbol (e.g., "SPY").
    - start_date (str): Start date in YYYY-MM-DD format.
    - end_date (str): End date in YYYY-MM-DD format.

    Returns:
    - DataFrame: Combined DataFrame with all dates.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    combined_df = pd.DataFrame()

    current_date = start_dt
    while current_date <= end_dt:
        date_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching data for {symbol} on {date_str}...")

        try:
            df = main(symbol, date_str)

            if df is not None:
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        except Exception as e:
            print(f"⚠️ Error fetching data for {symbol} on {date_str}: {e}")

        # 🔹 Avoid hitting the API rate limit (wait 15 seconds)
        time.sleep(15)  

        current_date += timedelta(days=1)  # Move to the next day

    return combined_df

# ✅ Ensure function is defined BEFORE calling it
df_week = fetch_multiple_days("SPY", "2025-02-20", "2025-02-28")

# ✅ Check results
print(df_week.head())
print(f"\n✅ Final DataFrame contains {df_week.shape[0]} rows across multiple days.")
