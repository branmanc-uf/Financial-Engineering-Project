import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # For proper Y-axis formatting

def simulate_stock(days=5, S0=610.00, mu_annual=0.10, sigma_annual=0.1592):
    """
    Simulates stock price movements over multiple trading days using GBM.
    Adds 8EMA, VWAP, ORB, PM High/Low, and Yesterday's High/Low.
    
    Returns:
        stock_df (DataFrame): Simulated stock price movements with indicators.
    """
    N_minutes_per_day = 390  
    dt = 1 / (252 * 390)  

    mu_per_min = mu_annual / (252 * 390)
    sigma_per_min = sigma_annual / np.sqrt(252) * 1.8

    stock_df = pd.DataFrame()
    trading_day = pd.Timestamp("2023-02-20")

    for day in range(days):
        timestamps = pd.date_range(
            start=trading_day.replace(hour=9, minute=30),
            periods=N_minutes_per_day,
            freq="min"
        ) + pd.Timedelta(days=day)

        dW = np.random.normal(0, 1, N_minutes_per_day)
        volatility_factor = np.ones(N_minutes_per_day)
        volatility_factor[:120] = 2  

        S = S0 * np.exp(np.cumsum((mu_per_min - 0.5 * sigma_per_min**2) * dt + 
                                  (sigma_per_min * volatility_factor * np.sqrt(dt)) * dW))
        
        volume = np.random.randint(100, 10000, size=N_minutes_per_day)

        temp_df = pd.DataFrame({
            'Timestamp': timestamps, 
            'Close': S, 
            'Volume': volume,
            'Day': day + 1
        })

        # **Calculate 8EMA & VWAP**
        temp_df['8EMA'] = temp_df['Close'].ewm(span=8, adjust=False).mean()
        temp_df['VWAP'] = (temp_df['Close'] * temp_df['Volume']).cumsum() / temp_df['Volume'].cumsum()

        # **Calculate ORB High/Low (First 15 Minutes)**
        open_range = temp_df[temp_df['Timestamp'].dt.time < pd.Timestamp("09:45:00").time()]
        orb_high = open_range['Close'].max()
        orb_low = open_range['Close'].min()
        temp_df.loc[:, 'ORB_High'] = orb_high
        temp_df.loc[:, 'ORB_Low'] = orb_low
        temp_df[['ORB_High', 'ORB_Low']] = temp_df[['ORB_High', 'ORB_Low']].fillna(method='ffill')

        # **Simulate PM High/Low & Yesterday’s High/Low**
        temp_df.loc[:, 'PM_High'] = S[0] * (1 + np.random.uniform(0.002, 0.008))  
        temp_df.loc[:, 'PM_Low'] = S[0] * (1 - np.random.uniform(0.002, 0.008))  
        temp_df.loc[:, 'Yest_High'] = S[0] * (1 + np.random.uniform(0.01, 0.02))
        temp_df.loc[:, 'Yest_Low'] = S[0] * (1 - np.random.uniform(0.01, 0.02))

        stock_df = pd.concat([stock_df, temp_df], ignore_index=True)
        S0 = S[-1]
    
    return stock_df

# Generate Fake Stock Data for 10 Days with Indicators
simulated_data = simulate_stock(1)

# Plot the results with indicators and dynamic ORB levels
plt.figure(figsize=(12, 6))

# Plot Stock Price
plt.plot(simulated_data['Timestamp'], simulated_data['Close'], label="Stock Price", color="blue", alpha=0.6)

# Plot 8EMA
plt.plot(simulated_data['Timestamp'], simulated_data['8EMA'], label="8EMA", color="orange", linestyle="--")

# Plot VWAP
plt.plot(simulated_data['Timestamp'], simulated_data['VWAP'], label="VWAP", color="green", linestyle="--")

# Iterate through unique days to plot ORB dynamically
unique_days = simulated_data['Day'].unique()
for day in unique_days:
    day_data = simulated_data[simulated_data['Day'] == day]
    plt.step(day_data['Timestamp'], day_data['ORB_High'], where='post', color='red', linestyle='--', label="ORB High" if day == 1 else "")
    plt.step(day_data['Timestamp'], day_data['ORB_Low'], where='post', color='purple', linestyle='--', label="ORB Low" if day == 1 else "")

# Formatting the Plot
plt.title("Simulated Stock Price with Dynamic ORB Levels, 8EMA & VWAP")
plt.xlabel("Time")
plt.ylabel("Price")

# Format X-axis to show timestamps in a readable way
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(10))  # Show fewer X-ticks for readability

# Format Y-axis
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

# Show legend and grid
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
