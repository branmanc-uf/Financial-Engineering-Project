import numpy as np
from scipy.stats import norm
def black_scholes_dataframe(df, strike_offset=5, r=0.05, sigma=0.20, expiration_days=1):
    """
    Adds Black-Scholes option pricing to a DataFrame.

    Parameters:
    - df: Pandas DataFrame containing stock data with 'close' price.
    - strike_offset: Difference between stock price and strike price (default: $5).
    - r: Risk-free interest rate (default: 5%).
    - sigma: Implied volatility (default: 20%).
    - expiration_days: Days until expiration (default: 30 days).

    Returns:
    - Updated DataFrame with Call & Put option prices + Greeks.
    """

    df = df.copy()

    # Convert expiration days into years
    T = expiration_days / 365

    # Define Strike Price (ATM or ITM)
    df['Strike_Price'] = np.round(df['close'] / strike_offset) * strike_offset

    # Calculate d1 and d2
    df['d1'] = (np.log(df['close'] / df['Strike_Price']) + (r + (sigma ** 2) / 2) * T) / (sigma * np.sqrt(T))
    df['d2'] = df['d1'] - sigma * np.sqrt(T)

    # Call & Put Prices
    df['Call_Price'] = df['close'] * norm.cdf(df['d1']) - df['Strike_Price'] * np.exp(-r * T) * norm.cdf(df['d2'])
    df['Put_Price'] = df['Strike_Price'] * np.exp(-r * T) * norm.cdf(-df['d2']) - df['close'] * norm.cdf(-df['d1'])

    # Greeks
    df['Call_Delta'] = norm.cdf(df['d1'])
    df['Put_Delta'] = -norm.cdf(-df['d1'])
    df['Gamma'] = norm.pdf(df['d1']) / (df['close'] * sigma * np.sqrt(T))
    df['Theta'] = (- (df['close'] * norm.pdf(df['d1']) * sigma) / (2 * np.sqrt(T))) - (r * df['Strike_Price'] * np.exp(-r * T) * norm.cdf(df['d2']))
    df['Vega'] = df['close'] * norm.pdf(df['d1']) * np.sqrt(T)
    df['Rho'] = df['Strike_Price'] * T * np.exp(-r * T) * norm.cdf(df['d2'])

    return df
