import numpy as np  # Importing numpy for numerical operations
import pandas as pd
from Gen_SPY_With_Indicators import simulate_stock  # Importing simulation function
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import matplotlib.ticker as mticker  # For proper Y-axis formatting

def generate_signals(df, stop_loss_pct=0.02):
    """ Generates trade signals based on ORB, VWAP, 8EMA, PM High/Low, and Yest High/Low """

    df['Signal'] = ''
    df['Stop_Loss'] = np.nan  # Initialize Stop_Loss column

    # Buy Call: Price breaks ORB High & supported by 8EMA & VWAP
    df.loc[
        (df['Close'] > df['ORB_High']) & 
        (df['Close'] > df['VWAP']) & 
        (df['Close'] > df['8EMA']),
        'Signal'
    ] = 'BUY CALL'

    # Sell Put: Price breaks ORB Low & rejected by 8EMA & VWAP
    df.loc[
        (df['Close'] < df['ORB_Low']) & 
        (df['Close'] < df['VWAP']) & 
        (df['Close'] < df['8EMA']),
        'Signal'
    ] = 'BUY PUT'

    # Boost Call Confidence if PM High or Yesterday's High is broken
    df.loc[
        (df['Signal'] == 'BUY CALL') & 
        ((df['Close'] > df['PM_High']) | (df['Close'] > df['Yest_High'])),
        'Signal'
    ] = 'STRONG BUY CALL'

    # Boost Put Confidence if PM Low or Yesterday's Low is broken
    df.loc[
        (df['Signal'] == 'BUY PUT') & 
        ((df['Close'] < df['PM_Low']) | (df['Close'] < df['Yest_Low'])),
        'Signal'
    ] = 'STRONG BUY PUT'

    # Add stop loss levels
    df.loc[df['Signal'].isin(['BUY CALL', 'STRONG BUY CALL']), 'Stop_Loss'] = df['Close'] * (1 - stop_loss_pct)
    df.loc[df['Signal'].isin(['BUY PUT', 'STRONG BUY PUT']), 'Stop_Loss'] = df['Close'] * (1 + stop_loss_pct)

    return df

# Simulate Stock Data for 2 Days
simulated_data, yesterday_high, yesterday_low = simulate_stock(2)

# Apply Signal Generation
simulated_data = generate_signals(simulated_data)

# **Plot with Pre-Market, Regular Market, and Indicators**
plt.figure(figsize=(12, 6))

for day in simulated_data["Day"].unique():
    day_data = simulated_data[simulated_data["Day"] == day]
    pm_data = day_data[day_data["Session"] == "PM"]
    market_data = day_data[day_data["Session"] == "Regular Market"]

    plt.plot(pm_data['Timestamp'], pm_data['Close'], label=f"PM Day {day}", color="black", linestyle="dotted", alpha=0.7)
    plt.plot(market_data['Timestamp'], market_data['Close'], label=f"Regular Market Day {day}", color="steelblue", alpha=0.8)

plt.plot(simulated_data['Timestamp'], simulated_data['8EMA'], label="8EMA", color="orange", linestyle="-")
plt.plot(simulated_data['Timestamp'], simulated_data['VWAP'], label="VWAP", color="blue", linestyle="-")

for day in simulated_data['Day'].unique():
    day_data = simulated_data[simulated_data['Day'] == day]
    plt.step(day_data['Timestamp'], day_data['ORB_High'], where='post', color='green', linestyle='--', label="ORB High" if day == 1 else "")
    plt.step(day_data['Timestamp'], day_data['ORB_Low'], where='post', color='red', linestyle='--', label="ORB Low" if day == 1 else "")
    plt.step(day_data['Timestamp'], day_data['PM_High'], where='post', color='green', linestyle='dotted', label="PM High" if day == 1 else "")
    plt.step(day_data['Timestamp'], day_data['PM_Low'], where='post', color='red', linestyle='dotted', label="PM Low" if day == 1 else "")
    market_open_time = day_data['Timestamp'].iloc[0]
    market_close_time = day_data['Timestamp'].iloc[-1]
    if day == 1:
        plt.hlines(y=yesterday_high, xmin=market_open_time, xmax=market_close_time, color='gray', linestyle='-.', linewidth=1.5, label="Yesterday's High")
        plt.hlines(y=yesterday_low, xmin=market_open_time, xmax=market_close_time, color='brown', linestyle='-.', linewidth=1.5, label="Yesterday's Low")
    else:
        plt.hlines(y=day_data['Yest_High'].iloc[0], xmin=market_open_time, xmax=market_close_time, color='gray', linestyle='-.', linewidth=1.5, label="Yesterday's High" if day == 2 else "")
        plt.hlines(y=day_data['Yest_Low'].iloc[0], xmin=market_open_time, xmax=market_close_time, color='brown', linestyle='-.', linewidth=1.5, label="Yesterday's Low" if day == 2 else "")

plt.title("Simulated Stock Price with PM, 8EMA, VWAP, ORB, & Yesterday's High/Low")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Print any part of the simulated data that contains BUY CALL or BUY PUT signals during Regular Market session
print(simulated_data[(simulated_data['Signal'].isin(['BUY CALL', 'BUY PUT', 'STRONG BUY CALL', 'STRONG BUY PUT'])) & 
                     (simulated_data['Session'] == 'Regular Market')])