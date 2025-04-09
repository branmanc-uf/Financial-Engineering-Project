# Import modules
from polygon import RESTClient
import pandas as pd
import pytz

from datetime import datetime, timedelta


import time  # Needed for adding delay
from datetime import datetime, timedelta

# API key (Replace with your actual API key)
API_KEY = "N5TPWLXWaGR0IN3_ttrmMROH_q9b2yBi"

def fetch_stock_data(symbol: str, date: str):
    """
    Fetches stock data from Polygon.io for a given symbol and date.

    Parameters:
    - symbol (str): Stock ticker symbol (e.g., "SPY").
    - date (str): Date in YYYY-MM-DD format.

    Returns:
    - DataFrame: Raw API response as a DataFrame.
    """
    client = RESTClient(API_KEY)

    # Fetch 1-minute interval bars
    dataRequest = client.get_aggs(
        ticker=symbol,
        multiplier=1,  # Always fetch 1-minute intervals
        timespan="minute",
        from_=date,
        to=date,
        adjusted=True
    )

    # Convert API response to DataFrame
    return pd.DataFrame(dataRequest)

def classify_missing_minutes(df):
    """
    Converts the timestamp column to Eastern Time (ET) and detects missing minutes.

    Parameters:
    - df (DataFrame): DataFrame with a 'timestamp' column in milliseconds.

    Returns:
    - DataFrame: DataFrame with properly decoded timestamps.
    """
    # Convert 'timestamp' from milliseconds to UTC DateTime
    df["datetime_utc"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    # Convert UTC to Eastern Time (ET)
    eastern = pytz.timezone("US/Eastern")
    df["datetime"] = df["datetime_utc"].dt.tz_convert(eastern)

    # Ensure timestamps are sorted correctly
    df = df.sort_values(by="datetime").reset_index(drop=True)

    # Calculate time differences between consecutive rows
    df["time_diff"] = df["datetime"].diff().dt.total_seconds() / 60  # Convert to minutes

    # Classify missing minutes: If the gap is greater than 1 minute, flag it
    df["missing_minutes"] = df["time_diff"] > 1

    return df[['close', 'datetime', 'open', 'high', 'low', 'volume']]  # Return DataFrame with properly decoded timestamps

def fill_missing_minutes(df):
    """
    Adds missing minute timestamps to the DataFrame and fills missing prices
    with the previous row's value.

    Parameters:
    - df (DataFrame): DataFrame with 'datetime' and 'close' prices.

    Returns:
    - DataFrame: Complete DataFrame with all 1-minute timestamps filled.
    """
    # Ensure 'datetime' is set as the index
    df = df.set_index("datetime")

    # Generate a full range of timestamps from the first to the last recorded timestamp
    full_time_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="min")

    # Reindex the DataFrame to include all minutes and fill missing values using forward fill
    df = df.reindex(full_time_range, method="ffill")

    # Reset index and rename back to 'datetime'
    df = df.reset_index().rename(columns={"index": "datetime"})

    return df

def main(symbol: str, date: str):
    """
    Fetch, classify, and fill stock data for a given symbol and date.

    Parameters:
    - symbol (str): Stock ticker symbol (e.g., "SPY").
    - date (str): Date in YYYY-MM-DD format.

    Returns:
    - DataFrame: Cleaned DataFrame with all timestamps filled.
    """
    df = fetch_stock_data(symbol, date)
    
    if df is None or df.empty:
        print(f"⚠️ No data found for {symbol} on {date}.")
        return None

    df = classify_missing_minutes(df)
    return fill_missing_minutes(df)

df_2 = main('SPY', '2025-02-27')





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

