import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qfin as qf  # Run locally where qfin is installed

# Define GBM Parameters with Correct Scaling
S0 = 610  # Initial stock price (e.g., SPY)
mu = 0.000001  # Realistic minute-level expected drift (~10% annual return)
sigma = 0.005  # Realistic minute-level volatility (~15% annual volatility)
T = 100  # Total simulation time (trading days)
N = 390 * T  # Number of time steps (390 minutes/day * 10 days)
dt = 1 / (252 * 390)  # Time step per minute (252 trading days per year)

# Creating a dummy stock that mimics SPY for 10 trading days
path = qf.simulations.GeometricBrownianMotion(S0, mu, sigma, dt, T)

# Creating timestamps for 10 trading days (minute-by-minute)
timestamps = pd.date_range(start='2023-02-20 09:30', periods=len(path.simulated_path), freq='min')

# Convert to DataFrame
stock_df = pd.DataFrame({'Timestamp': timestamps, 'Stock_Price': path.simulated_path})

# Plot the fixed GBM simulation
plt.figure(figsize=(12, 5))
plt.plot(stock_df['Timestamp'], stock_df['Stock_Price'], label='Simulated Stock Price', color='blue')
plt.title('Simulated Stock Price Over 10 Trading Days (Realistic Movement)')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Stock Price')
plt.legend()
plt.show()