#ALL HERE ARE PENDING TESTING
#author: Antonio de Guzman

import openpyxl
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from strategy_test import test_strategy
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def preprocess_data(file_path, random_dates, market_open, pre_market_start, market_close):
    """
    Preprocesses the data from the CSV file and formats it for use in the report.

    Parameters:
    - file_path: Path to the CSV file containing stock data.
    - random_dates: List of random dates to filter the data.
    - market_open: Market open time (HH:MM:SS).
    - pre_market_start: Pre-market start time (HH:MM:SS).
    - market_close: Market close time (HH:MM:SS).

    Returns:
    - Preprocessed DataFrame with calculated indicators and session information.
    """
    df = pd.read_csv(file_path)
    df.columns = [col.capitalize() for col in df.columns]
    df['Datetime'] = df['Datetime'].astype(str)
    formatted_random_dates = [date.strftime('%Y-%m-%d') for date in random_dates]
    retrieved_data = df[df['Datetime'].str.startswith(tuple(formatted_random_dates))]
    retrieved_data['8EMA'] = retrieved_data['Close'].ewm(span=8, adjust=False).mean()
    retrieved_data['VWAP'] = (retrieved_data['Close'] * retrieved_data['Volume']).cumsum() / retrieved_data['Volume'].cumsum()
    retrieved_data['Datetime'] = pd.to_datetime(retrieved_data['Datetime'])
    retrieved_data.rename(columns={'Datetime': 'Timestamp'}, inplace=True)
    retrieved_data['Day'] = retrieved_data['Timestamp'].dt.strftime('%Y-%m-%d')
    day_mapping = {day: idx + 1 for idx, day in enumerate(sorted(retrieved_data['Day'].unique()))}
    retrieved_data['Day'] = retrieved_data['Day'].map(day_mapping).astype(int)

    orb_highs, orb_lows, pm_highs, pm_lows, yest_highs, yest_lows = [], [], [], [], [], []
    for day in retrieved_data['Day'].unique():
        daily_data = retrieved_data[retrieved_data['Day'] == day]
        orb_data = daily_data[
            (daily_data['Timestamp'].dt.time >= datetime.strptime(market_open, '%H:%M:%S').time()) &
            (daily_data['Timestamp'].dt.time < (datetime.strptime(market_open, '%H:%M:%S') + timedelta(minutes=15)).time())
        ]
        orb_highs.append(orb_data['High'].max())
        orb_lows.append(orb_data['Low'].min())
        pm_data = daily_data[
            (daily_data['Timestamp'].dt.time >= datetime.strptime(pre_market_start, '%H:%M:%S').time()) &
            (daily_data['Timestamp'].dt.time < datetime.strptime(market_open, '%H:%M:%S').time())
        ]
        pm_highs.append(pm_data['High'].max())
        pm_lows.append(pm_data['Low'].min())
        if len(yest_highs) == 0:
            yest_highs.append(None)
            yest_lows.append(None)
        else:
            prev_day_data = retrieved_data[retrieved_data['Day'] == (day - 1)]
            yest_highs.append(prev_day_data['High'].max())
            yest_lows.append(prev_day_data['Low'].min())

    retrieved_data['ORB_High'] = retrieved_data['Day'].map(dict(zip(retrieved_data['Day'].unique(), orb_highs)))
    retrieved_data['ORB_Low'] = retrieved_data['Day'].map(dict(zip(retrieved_data['Day'].unique(), orb_lows)))
    retrieved_data['PM_High'] = retrieved_data['Day'].map(dict(zip(retrieved_data['Day'].unique(), pm_highs)))
    retrieved_data['PM_Low'] = retrieved_data['Day'].map(dict(zip(retrieved_data['Day'].unique(), pm_lows)))
    retrieved_data['Yest_High'] = retrieved_data['Day'].map(dict(zip(retrieved_data['Day'].unique(), yest_highs)))
    retrieved_data['Yest_Low'] = retrieved_data['Day'].map(dict(zip(retrieved_data['Day'].unique(), yest_lows)))

    retrieved_data['Session'] = retrieved_data['Timestamp'].dt.time.apply(
        lambda t: 'PM' if datetime.strptime(pre_market_start, '%H:%M:%S').time() <= t < datetime.strptime(market_open, '%H:%M:%S').time()
        else 'Regular Market' if datetime.strptime(market_open, '%H:%M:%S').time() <= t < datetime.strptime(market_close, '%H:%M:%S').time()
        else None
    )
    return retrieved_data

def generate_excel_report(symbol, start_date, end_date, output_file, csv_file, random_dates, market_open, pre_market_start, market_close):
    """
    Generates an Excel report with strategy performance metrics and a candlestick chart using real data.

    Parameters:
    - symbol: Stock ticker (e.g., "SPY").
    - start_date: Start date for backtest (YYYY-MM-DD).
    - end_date: End date for backtest (YYYY-MM-DD).
    - output_file: Path to save the Excel report.
    - csv_file: Path to the CSV file containing stock data.
    - random_dates: List of random dates to filter the data.
    - market_open: Market open time (HH:MM:SS).
    - pre_market_start: Pre-market start time (HH:MM:SS).
    - market_close: Market close time (HH:MM:SS).
    """
    # Preprocess the data
    real_data = preprocess_data(csv_file, random_dates, market_open, pre_market_start, market_close)

    # Run the strategy test and get performance metrics
    performance_metrics = test_strategy(symbol, start_date, end_date)

    if performance_metrics is None:
        print("⚠️ No data available or strategy test failed.")
        return

    # Create a new workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Performance Metrics"

    # Write performance metrics to the sheet
    ws.append(["Metric", "Value"])
    for key, value in performance_metrics.items():
        ws.append([key, value])

    # Generate and save the candlestick chart
    from generate_candlestick_df import plot_candlestick_with_indicators
    candlestick_data = real_data  # Use real data for candlestick chart
    plot_candlestick_with_indicators(candlestick_data, real_data)
    chart_path = "candlestick_chart.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    # Embed the chart in the Excel report
    ws_chart = wb.create_sheet(title="Candlestick Chart")
    img = Image(chart_path)
    img.anchor = "A1"
    ws_chart.add_image(img)

    # Save the workbook
    wb.save(output_file)
    print(f"Excel report generated: {output_file}")

def generate_text_report(symbol, start_date, end_date, output_file):
    """
    Generates a text file summarizing the results of the strategy test.

    Parameters:
    - symbol: Stock ticker (e.g., "SPY").
    - start_date: Start date for backtest (YYYY-MM-DD).
    - end_date: End date for backtest (YYYY-MM-DD).
    - output_file: Path to save the text report.
    """
    # Run the strategy test and get performance metrics
    performance_metrics = test_strategy(symbol, start_date, end_date)

    if performance_metrics is None:
        print("⚠️ No data available or strategy test failed.")
        return

    # Write performance metrics to a text file
    with open(output_file, "w") as file:
        file.write(f"Strategy Test Results for {symbol} ({start_date} to {end_date})\n")
        file.write("=" * 50 + "\n")
        for key, value in performance_metrics.items():
            file.write(f"{key}: {value}\n")
    print(f"Text report generated: {output_file}")

# Example usage
random_dates = [datetime(2023, 1, 3), datetime(2023, 1, 4)]  # Replace with actual random dates
generate_excel_report(
    "SPY", "2023-01-01", "2023-01-31", "strategy_report.xlsx",
    "df_2022_2024.csv", random_dates, "09:30:00", "07:00:00", "16:00:00"
)
generate_text_report("SPY", "2023-01-01", "2023-01-31", "strategy_report.txt")
