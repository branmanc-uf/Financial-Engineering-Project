import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # For proper Y-axis formatting

def simulate_stock(days, S0=610.00, mu_annual=0.10, sigma_annual=0.1592):
    """
    Simulates stock price movements over multiple trading days, including pre-market hours.
    Adds 8EMA, VWAP, ORB, PM High/Low, and Yesterday's High/Low.
    
    Returns:
        stock_df (DataFrame): A DataFrame containing the simulated stock price movements with indicators.
    """
    N_pre_market_minutes = 330  # Pre-market minutes (4:00 AM - 9:30 AM)
    N_market_minutes = 390  # Regular trading minutes (9:30 AM - 4:00 PM)
    dt = 1 / (252 * (N_pre_market_minutes + N_market_minutes))  # Time step per minute

    mu_per_min = mu_annual / (252 * (N_pre_market_minutes + N_market_minutes))
    sigma_per_min = sigma_annual / np.sqrt(252)

    stock_df = pd.DataFrame()
    trading_day = pd.Timestamp("2023-02-20")

    prev_day_high, prev_day_low = None, None

    for day in range(days):
        pre_market_timestamps = pd.date_range(
            start=trading_day.replace(hour=4, minute=0), periods=N_pre_market_minutes, freq="min"
        ) + pd.Timedelta(days=day)

        market_timestamps = pd.date_range(
            start=trading_day.replace(hour=9, minute=30), periods=N_market_minutes, freq="min"
        ) + pd.Timedelta(days=day)

        dW_pre_market = np.random.normal(0, 1, N_pre_market_minutes)
        dW_market = np.random.normal(0, 1, N_market_minutes)

        volatility_factor_pre_market = np.ones(N_pre_market_minutes) * 2.5  
        volatility_factor_market = np.ones(N_market_minutes)
        volatility_factor_market[:120] = 2  

        S_pre_market = S0 * np.exp(np.cumsum((mu_per_min - 0.5 * sigma_per_min**2) * dt +
                                             (sigma_per_min * volatility_factor_pre_market * np.sqrt(dt)) * dW_pre_market))

        S_market = S_pre_market[-1] * np.exp(np.cumsum((mu_per_min - 0.5 * sigma_per_min**2) * dt +
                                                       (sigma_per_min * volatility_factor_market * np.sqrt(dt)) * dW_market))

        volume_pre_market = np.random.randint(100, 5000, size=N_pre_market_minutes)
        volume_market = np.random.randint(5000, 20000, size=N_market_minutes)

        pre_market_df = pd.DataFrame({'Timestamp': pre_market_timestamps, 'Close': S_pre_market, 
                                      'Volume': volume_pre_market, 'Session': 'PM', 'Day': day + 1})

        market_df = pd.DataFrame({'Timestamp': market_timestamps, 'Close': S_market, 
                                  'Volume': volume_market, 'Session': 'Regular Market', 'Day': day + 1})

        day_df = pd.concat([pre_market_df, market_df])

        day_df['8EMA'] = day_df['Close'].ewm(span=8, adjust=False).mean()
        day_df['VWAP'] = (day_df['Close'] * day_df['Volume']).cumsum() / day_df['Volume'].cumsum()

        open_range = day_df[(day_df['Timestamp'].dt.time >= pd.Timestamp("09:30:00").time()) & 
                            (day_df['Timestamp'].dt.time < pd.Timestamp("09:45:00").time())]

        orb_high = open_range['Close'].max()
        orb_low = open_range['Close'].min()
        day_df.loc[:, 'ORB_High'] = orb_high
        day_df.loc[:, 'ORB_Low'] = orb_low
        day_df[['ORB_High', 'ORB_Low']] = day_df[['ORB_High', 'ORB_Low']].fillna(method='ffill')

        pm_range = day_df[day_df['Session'] == 'PM']
        pm_high = pm_range['Close'].max()
        pm_low = pm_range['Close'].min()
        day_df.loc[:, 'PM_High'] = pm_high
        day_df.loc[:, 'PM_Low'] = pm_low
        day_df[['PM_High', 'PM_Low']] = day_df[['PM_High', 'PM_Low']].fillna(method='ffill')

        if prev_day_high is not None and prev_day_low is not None:
            day_df.loc[:, 'Yest_High'] = prev_day_high
            day_df.loc[:, 'Yest_Low'] = prev_day_low
        else:
            day_df.loc[:, 'Yest_High'] = None
            day_df.loc[:, 'Yest_Low'] = None

        prev_day_high = day_df['Close'].max()
        prev_day_low = day_df['Close'].min()

        stock_df = pd.concat([stock_df, day_df], ignore_index=True)
        S0 = S_market[-1]

    return stock_df

# Generate Fake Stock Data for 5 Days
simulated_data = simulate_stock(1)

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
