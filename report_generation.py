#ALL HERE ARE PENDING TESTING
#author: Antonio de Guzman

import openpyxl
from openpyxl.drawing.image import Image
from openpyxl.utils.dataframe import dataframe_to_rows
from strategy_test import test_strategy
import matplotlib.pyplot as plt
import pandas as pd

def generate_excel_report(symbol, start_date, end_date, output_file):
    """
    Generates an Excel report with strategy performance metrics and a candlestick chart.

    Parameters:
    - symbol: Stock ticker (e.g., "SPY").
    - start_date: Start date for backtest (YYYY-MM-DD).
    - end_date: End date for backtest (YYYY-MM-DD).
    - output_file: Path to save the Excel report.
    """
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
    from presentation_dashboard import plot_candlestick_with_indicators
    simulated_data = pd.DataFrame()  # Replace with actual simulated data
    candlestick_data = pd.DataFrame()  # Replace with actual candlestick data
    plot_candlestick_with_indicators(candlestick_data, simulated_data)
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
generate_excel_report("SPY", "2023-01-01", "2023-01-31", "strategy_report.xlsx")
generate_text_report("SPY", "2023-01-01", "2023-01-31", "strategy_report.txt")
