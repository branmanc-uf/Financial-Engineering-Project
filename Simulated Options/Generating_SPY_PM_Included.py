import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # For proper Y-axis formatting

def simulate_stock(days, S0=610.00, mu_annual=0.10, sigma_annual=0.1592):
    """
    Simulates stock price movements over multiple trading days, including pre-market hours.
    
    Parameters:
        days (int): Number of trading days to simulate.
        S0 (float): Initial stock price.
        mu_annual (float): Annual drift.
        sigma_annual (float): Annual volatility.
    
    Returns:
        stock_df (DataFrame): A DataFrame containing the simulated stock price movements for all days.
    """
    # Set trading parameters
    N_pre_market_minutes = 330  # Pre-market minutes (4:00 AM - 9:30 AM)
    N_market_minutes = 390  # Regular trading minutes (9:30 AM - 4:00 PM)
    dt = 1 / (252 * (N_pre_market_minutes + N_market_minutes))  # Time step per minute

    # Convert annual drift and volatility to per-minute scale
    mu_per_min = mu_annual / (252 * (N_pre_market_minutes + N_market_minutes))
    sigma_per_min = sigma_annual / np.sqrt(252)

    # Initialize an empty DataFrame to store results
    stock_df = pd.DataFrame()

    # Set initial timestamp
    trading_day = pd.Timestamp("2023-02-20")

    # Iterate over the number of trading days
    for day in range(days):
        # Generate pre-market timestamps (4:00 AM - 9:30 AM)
        pre_market_timestamps = pd.date_range(
            start=trading_day.replace(hour=4, minute=0),
            periods=N_pre_market_minutes,
            freq="min"
        ) + pd.Timedelta(days=day)

        # Generate regular market timestamps (9:30 AM - 4:00 PM)
        market_timestamps = pd.date_range(
            start=trading_day.replace(hour=9, minute=30),
            periods=N_market_minutes,
            freq="min"
        ) + pd.Timedelta(days=day)

        # Generate random Brownian motion increments
        dW_pre_market = np.random.normal(0, 1, N_pre_market_minutes)
        dW_market = np.random.normal(0, 1, N_market_minutes)

        # **Higher Volatility in Pre-Market**
        volatility_factor_pre_market = np.ones(N_pre_market_minutes) * 2.5  # More volatile pre-market
        volatility_factor_market = np.ones(N_market_minutes)
        volatility_factor_market[:120] = 2  # Higher volatility in first 2 hours of market open

        # Compute stock price path for the pre-market using GBM formula
        S_pre_market = S0 * np.exp(np.cumsum((mu_per_min - 0.5 * sigma_per_min**2) * dt + 
                                             (sigma_per_min * volatility_factor_pre_market * np.sqrt(dt)) * dW_pre_market))

        # Compute stock price path for the regular market using GBM formula
        S_market = S_pre_market[-1] * np.exp(np.cumsum((mu_per_min - 0.5 * sigma_per_min**2) * dt + 
                                                       (sigma_per_min * volatility_factor_market * np.sqrt(dt)) * dW_market))

        # Create DataFrames for pre-market and market sessions
        pre_market_df = pd.DataFrame({'Timestamp': pre_market_timestamps, 'Stock_Price': S_pre_market, 'Session': 'Pre-Market', 'Day': day + 1})
        market_df = pd.DataFrame({'Timestamp': market_timestamps, 'Stock_Price': S_market, 'Session': 'Regular Market', 'Day': day + 1})

        # Append to main DataFrame
        stock_df = pd.concat([stock_df, pre_market_df, market_df], ignore_index=True)

        # Update S0 to be the last price of the previous day
        S0 = S_market[-1]

    return stock_df

# Generate Fake Stock Data for 5 Days with Pre-Market
simulated_data = simulate_stock(5)

# **Plot with Pre-Market & Regular Market**
plt.figure(figsize=(12, 6))

# Plot Pre-Market Data
for day in simulated_data["Day"].unique():
    day_data = simulated_data[simulated_data["Day"] == day]
    pre_market_data = day_data[day_data["Session"] == "Pre-Market"]
    market_data = day_data[day_data["Session"] == "Regular Market"]

    plt.plot(pre_market_data['Timestamp'], pre_market_data['Stock_Price'], label=f"Pre-Market Day {day}", color="gray", linestyle="--", alpha=0.7)
    plt.plot(market_data['Timestamp'], market_data['Stock_Price'], label=f"Regular Market Day {day}", alpha=0.8)

plt.title("Simulated Stock Price with Pre-Market and Regular Market Volatility")
plt.xlabel("Time")
plt.ylabel("Stock Price")

# 🔹 Format X-axis to show timestamps properly
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(10))  # Show fewer X-ticks for readability

# 🔹 Fix Y-Axis Formatting (No Scientific Notation)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.legend()
plt.grid(True)

# Display the plot
plt.show()

