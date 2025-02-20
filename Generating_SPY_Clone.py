import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # For proper Y-axis formatting

def simulate_stock(days, S0=610.00, mu_annual=0.10, sigma_annual=0.1592):
    """
    Simulates stock price movements over multiple trading days using GBM.
    
    Parameters:
        days (int): Number of trading days to simulate.
        S0 (float): Initial stock price.
        mu_annual (float): Annual drift.
        sigma_annual (float): Annual volatility.
    
    Returns:
        stock_df (DataFrame): A DataFrame containing the simulated stock price movements for all days.
    """
    # Set trading parameters
    N_minutes_per_day = 390  # Trading minutes per day
    dt = 1 / (252 * 390)  # Time step per minute

    # Convert annual drift and volatility to per-minute scale
    mu_per_min = mu_annual / (252 * 390)
    sigma_per_min = sigma_annual / np.sqrt(252) * 1.8

    # Initialize an empty DataFrame to store results
    stock_df = pd.DataFrame()

    # Set initial timestamp
    trading_day = pd.Timestamp("2023-02-20")

    # Iterate over the number of trading days
    for day in range(days):
        # Generate timestamps that are continuous across days
        timestamps = pd.date_range(
            start=trading_day.replace(hour=9, minute=30),
            periods=N_minutes_per_day,
            freq="min"
        ) + pd.Timedelta(days=day)  # Shift each day's timestamps forward

        # Generate random Brownian motion increments
        dW = np.random.normal(0, 1, N_minutes_per_day)

        # Modify volatility based on the time of day (higher in the first 2 hours)
        volatility_factor = np.ones(N_minutes_per_day)
        volatility_factor[:120] = 2  # Higher volatility in the first two hours

        # Compute stock price path for the day using GBM formula
        S = S0 * np.exp(np.cumsum((mu_per_min - 0.5 * sigma_per_min**2) * dt + 
                                  (sigma_per_min * volatility_factor * np.sqrt(dt)) * dW))

        # Create a DataFrame for the current day
        temp_df = pd.DataFrame({'Timestamp': timestamps, 'Stock_Price': S, 'Day': day + 1})

        # Append to main DataFrame
        stock_df = pd.concat([stock_df, temp_df], ignore_index=True)

        # Update S0 to be the last price of the previous day
        S0 = S[-1]

    return stock_df

# Run the function for 5 trading days
simulated_data = simulate_stock(days=5)

# Plot the results with a continuous X-axis
unique_days = simulated_data["Day"].unique()

for day in unique_days:  # Loop over each unique day
    day_data = simulated_data[simulated_data["Day"] == day]
    plt.plot(day_data["Timestamp"], day_data["Stock_Price"], label=f"Day {day}")

plt.title("Simulated Intraday Stock Price Over Multiple Days (Fixed Time Axis)")
plt.xlabel("Time")
plt.ylabel("Stock Price")

# 🔹 Format X-axis to show dates and times continuously
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(10))  # Show fewer X-ticks for readability

# 🔹 Fix Y-Axis Formatting (No Scientific Notation)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.legend()
plt.grid(True)
plt.show()

print(len(simulated_data))