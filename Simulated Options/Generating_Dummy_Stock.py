import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # For proper Y-axis formatting

# 📌 Step 1: Set GBM Parameters
S0 = 610.00  # Initial stock price
mu_annual = 0.10  # Annual drift (10%)
sigma_annual = 0.1592  # Annual volatility (15.92%)

N_minutes_per_day = 390  # Trading minutes per day (9:30 AM - 4:00 PM)
N = N_minutes_per_day  # Total steps (minutes) for one day

# Convert annual drift and volatility to per-minute scale
mu_per_min = mu_annual / (252 * 390)  # Drift per minute

# ✅ Adjusted Volatility Scaling (Fix)
sigma_per_min = sigma_annual / np.sqrt(252)  # Convert to daily volatility
sigma_per_min = sigma_per_min  # Base volatility, will scale dynamically during the open period

dt = 1 / (252 * 390)  # Time step per minute

# 📌 Step 2: Generate Trading Timestamps for One Day
trading_day = pd.Timestamp("2023-02-20")  # Fixed single day
timestamps = pd.date_range(start=trading_day.replace(hour=9, minute=30), periods=N_minutes_per_day, freq="min")  # ✅ FIXED WARNING

# 📌 Step 3: Generate GBM Stock Price Path for One Day
np.random.seed(42)  # For reproducibility
dW = np.random.normal(0, 1, N)  # Standard normal random increments

# Modify volatility based on the time of day (first 2 hours more volatile)
volatility_factor = np.ones(N)  # Default volatility scaling (1 for normal)

# Increase volatility during the first 2 hours (120 minutes)
volatility_factor[:120] = 2  # Double volatility during the open period

# Compute stock price path using GBM formula with modified volatility factor
S = S0 * np.exp(np.cumsum((mu_per_min - 0.5 * sigma_per_min**2) * dt + (sigma_per_min * volatility_factor * np.sqrt(dt)) * dW))

# 📌 Step 4: Create DataFrame and Store Results
stock_df = pd.DataFrame({'Timestamp': timestamps, 'Stock_Price': S})

# 📌 Step 5: Plot the Simulated Stock Prices for One Day
plt.figure(figsize=(12, 5))
plt.plot(stock_df['Timestamp'], stock_df['Stock_Price'], label="Simulated Stock Price", color="blue")
plt.title("Simulated Intraday Stock Price (1 Trading Day) with Increased Volatility in Open Period")
plt.xlabel("Time")
plt.xticks(stock_df['Time'][::60], [t.strftime("%H:%M") for t in stock_df['Time'][::60]], rotation=45)
plt.ylabel("Stock Price")

# 🔹 Fix Y-Axis Formatting (No Scientific Notation)
plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

plt.legend()
plt.show()

# 📌 Step 6: Display First 10 Rows of the Data
print(f"Number of timestamps generated: {len(stock_df)}")  # Should be 390
print(stock_df.head(10))  # Show first 10 rows
