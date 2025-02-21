import numpy as np  # Importing numpy for numerical operations
import pandas as pd
from Gen_SPY_With_Indicators import simulate_stock  # Importing simulation function
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import matplotlib.ticker as mticker  # For proper Y-axis formatting

def generate_signals(df):
    """ Generates trade signals based on ORB, VWAP, 8EMA, PM High/Low, and Yest High/Low """

    df['Signal'] = ''

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

    return df

# Simulate Stock Data for 5 Days
simulated_data = simulate_stock(2)

# Apply Signal Generation
simulated_data = generate_signals(simulated_data)

# **Plot with Pre-Market, Regular Market, and Indicators**
plt.figure(figsize=(12, 6))

for day in simulated_data["Day"].unique():
    day_data = simulated_data[simulated_data["Day"] == day]
    pm_data = day_data[day_data["Session"] == "PM"]
    market_data = day_data[day_data["Session"] == "Regular Market"]

    plt.plot(pm_data['Timestamp'], pm_data['Close'], label=f"PM Day {day}", color="black", linestyle="dotted", alpha=0.7)
    plt.plot(market_data['Timestamp'], market_data['Close'], label=f"Regular Market Day {day}", alpha=0.8)

plt.plot(simulated_data['Timestamp'], simulated_data['8EMA'], label="8EMA", color="orange", linestyle="-")
plt.plot(simulated_data['Timestamp'], simulated_data['VWAP'], label="VWAP", color="blue", linestyle="-")

for day in simulated_data['Day'].unique():
    day_data = simulated_data[simulated_data['Day'] == day]
    plt.step(day_data['Timestamp'], day_data['ORB_High'], where='post', color='green', linestyle='--', label="ORB High" if day == 1 else "")
    plt.step(day_data['Timestamp'], day_data['ORB_Low'], where='post', color='red', linestyle='--', label="ORB Low" if day == 1 else "")
    plt.step(day_data['Timestamp'], day_data['PM_High'], where='post', color='green', linestyle='dotted', label="PM High" if day == 1 else "")
    plt.step(day_data['Timestamp'], day_data['PM_Low'], where='post', color='red', linestyle='dotted', label="PM Low" if day == 1 else "")

    # Draw one big green dot for the day when stock price breaks through ORB high after 9:30 AM
    market_open_data = day_data[day_data['Session'] == 'Regular Market']
    if not market_open_data[market_open_data['Close'] > market_open_data['ORB_High']].empty:
        first_above_orb_high = market_open_data[market_open_data['Close'] > market_open_data['ORB_High']].iloc[0]
        plt.scatter(first_above_orb_high['Timestamp'], first_above_orb_high['Close'], color='green', s=75, zorder=5)

    # Draw one big red dot for the day when stock price breaks below ORB low after 9:30 AM
    if not market_open_data[market_open_data['Close'] < market_open_data['ORB_Low']].empty:
        first_below_orb_low = market_open_data[market_open_data['Close'] < market_open_data['ORB_Low']].iloc[0]
        plt.scatter(first_below_orb_low['Timestamp'], first_below_orb_low['Close'], color='red', s=75, zorder=5)

plt.title("Simulated Stock Price with PM, 8EMA, VWAP, ORB, & Yesterday's High/Low")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

# Print any part of the simulated data that contains BUY CALL or BUY PUT signals
print(simulated_data[simulated_data['Signal'].isin(['BUY CALL', 'BUY PUT', 'STRONG BUY CALL', 'STRONG BUY PUT'])])