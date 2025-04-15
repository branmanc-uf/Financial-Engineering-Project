# ALL HERE ARE PENDING TESTING
# author: Antonio de Guzman

import openpyxl
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from strategy_test_df import test_strategy_from_df
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from getting_options import black_scholes_dataframe

def generate_reports(symbol, df, output_file, strategy_params):
    """
    Generates an Excel report with strategy performance metrics and a candlestick chart using the new strategy function.

    Parameters:
    - symbol: Stock ticker (e.g., "SPY").
    - df: DataFrame containing stock data.
    - output_file: Path to save the Excel report.
    - strategy_params: Dictionary of parameters to pass to test_strategy_from_df.
    """

    performance_metrics = test_strategy_from_df(
        df,
        **strategy_params
    )

    if performance_metrics is None:
        print("⚠️ No data available or strategy test failed.")
        return

    # Extract metrics and trade log
    metrics = {k: v for k, v in performance_metrics.items() if k != 'DataFrame' and k != 'Trade Log'}
    trade_log = performance_metrics['Trade Log']

    # Create a new workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Performance Metrics"

    # Write performance metrics to the sheet
    ws.append(["Metric", "Value"])
    for key, value in metrics.items():
        ws.append([key, value])

    # Write trade log to a new sheet
    ws_trades = wb.create_sheet(title="Trade Log")

    # Strip timezone from datetime before exporting to Excel
    trade_log_export = trade_log.copy()

    # Strip timezone from any datetime-like columns
    for col in trade_log_export.columns:
        if pd.api.types.is_datetime64_any_dtype(trade_log_export[col]):
            trade_log_export[col] = trade_log_export[col].dt.tz_localize(None)


    for row in dataframe_to_rows(trade_log_export, index=False, header=True):
        ws_trades.append(row)
    '''
    # Generate and save the candlestick chart
    plot_candlestick_with_indicators(candlestick_data, retrieved_data)
    chart_path = "candlestick_chart.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    # Embed the chart in the Excel report
    ws_chart = wb.create_sheet(title="Candlestick Chart")
    img = Image(chart_path)
    img.anchor = "A1"
    ws_chart.add_image(img)
    '''

    # Save the workbook
    wb.save(output_file)
    print(f"Excel report generated: {output_file}")

    # Write performance metrics to a text file
    text_output_file = output_file.replace(".xlsx", ".txt")
    with open(text_output_file, "w") as file:
        file.write(f"Strategy Test Results for {symbol}\n")
        file.write("=" * 50 + "\n")
        for key, value in metrics.items():
    # Write performance metrics to a text file
    text_output_file = output_file.replace(".xlsx", ".txt")
    with open(text_output_file, "w") as file:
        file.write(f"Strategy Test Results for {symbol}\n")
        file.write("=" * 50 + "\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")
        file.write("\nTrade Log:\n")
        file.write(trade_log.to_string(index=False))
    print(f"Text report generated: {text_output_file}")
