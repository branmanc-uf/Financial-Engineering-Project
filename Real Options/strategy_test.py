from getting_options import black_scholes_dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_strategy(symbol, start_date, end_date, 
                  initial_capital=10000, risk_per_trade=0.02, 
                  open_range_min=15, stop_loss_pct=0.10, take_profit_pct=0.20, 
                  expiration_days=7, sigma=0.20):
    """
    Backtests an intraday options breakout strategy using the Black-Scholes model.
    
    Parameters:
    - symbol: Stock ticker (e.g., "SPY").
    - start_date: Start date for backtest (YYYY-MM-DD).
    - end_date: End date for backtest (YYYY-MM-DD).
    - initial_capital: Starting capital for backtest (default $10,000).
    - risk_per_trade: % of capital risked per trade (default 2%).
    - open_range_min: Open range breakout period (default 15 minutes).
    - stop_loss_pct: Stop-loss percentage (default 10%).
    - take_profit_pct: Take-profit percentage (default 20%).
    - expiration_days: Days until expiration (default 7 days).
    - sigma: Implied volatility (default 20%).

    Returns:
    - Aggregated performance metrics (Sharpe Ratio, Max Drawdown, Win Rate, Final Return).
    """

    # Fetch stock data
    df = pd.read_csv('df_2023_2025.csv', parse_dates=['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/New_York')
    
    if df is None or df.empty:
        print("⚠️ No data available for the given date range.")
        return None

    # Identify breakout levels and trade direction
    df = identify_breakout_levels(df, open_range_min=open_range_min)
    df = determine_trade_direction(df)

    # Add Black-Scholes option pricing to the dataframe
    df = black_scholes_dataframe(df, sigma=sigma, expiration_days=expiration_days)

    # Portfolio Tracking
    portfolio = initial_capital
    trade_size = risk_per_trade * portfolio  # 2% risk per trade
    position = None  # Track open trade
    entry_price = None
    df['Portfolio Value'] = portfolio
    df['Realized PnL'] = 0  # Track profits & losses

    # Iterate through trades and adjust portfolio value
    for i in range(1, len(df)):
        trade_type = df.loc[df.index[i], 'Trade_Type']
        price = df.loc[df.index[i], 'close']

        if trade_type in ["CALL", "PUT"] and position is None:
            # Use Black-Scholes estimated price instead of real market data
            entry_price = df.loc[df.index[i], 'Call_Price'] if trade_type == "CALL" else df.loc[df.index[i], 'Put_Price']
            position = trade_type  # Track active position

        elif position is not None:
            # Get current option price based on Black-Scholes model
            current_price = df.loc[df.index[i], 'Call_Price'] if position == "CALL" else df.loc[df.index[i], 'Put_Price']

            # Calculate Stop-Loss & Take-Profit levels
            stop_loss_level = entry_price * (1 - stop_loss_pct)
            take_profit_level = entry_price * (1 + take_profit_pct)

            # Check Stop-Loss or Take-Profit Execution
            if current_price <= stop_loss_level or current_price >= take_profit_level:
                trade_profit = trade_size * ((current_price - entry_price) / entry_price)
                portfolio += trade_profit  # Update portfolio with realized P&L
                df.loc[df.index[i], 'Realized PnL'] = trade_profit

                # Close Position
                position = None
                entry_price = None

        df.loc[df.index[i], 'Portfolio Value'] = portfolio

    # Compute Final Performance Metrics
    df['Market Return'] = df['close'].pct_change()
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod()
    df['Cumulative Strategy Return'] = df['Portfolio Value'] / initial_capital

    sharpe_ratio = df['Realized PnL'].mean() / df['Realized PnL'].std() * np.sqrt(252)
    rolling_max = df['Cumulative Strategy Return'].cummax()
    drawdown = (df['Cumulative Strategy Return'] / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    win_rate = (df['Realized PnL'] > 0).mean()

    # Plot Portfolio Growth
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", linestyle="dashed")
    plt.plot(df.index, df['Cumulative Strategy Return'], label="Strategy Return", color='green')
    plt.legend()
    plt.title(f"Strategy Performance ({start_date} to {end_date})\nSharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2%}")
    plt.show()

    return {
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Max Drawdown': round(max_drawdown * 100, 2),
        'Win Rate': round(win_rate * 100, 2),
        'Final Portfolio Value': round(portfolio, 2),
        'Final Strategy Return': round(df['Cumulative Strategy Return'].iloc[-1] * 100, 2)
    }


def identify_breakout_levels(df, open_range_min=15):
    """
    Identifies key breakout levels for intraday trading:
    - Open range high/low based on user-defined minutes.
    - Pre-market High/Low.
    - Previous Day High/Low.
    
    Parameters:
    - df: DataFrame containing intraday price data.
    - open_range_min: Number of minutes to define the open range.

    Returns:
    - DataFrame with breakout levels.
    """
    
    df = pd.read_csv('df_2023_2025.csv', parse_dates=['datetime'])
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/New_York')
    df["Volume"] = pd.read_csv('df_2023_2025_volumes.csv', usecols=['volume'])["volume"]

    # Define market open time
    market_open = pd.to_datetime('09:30:00').time()  
    open_range_end = (pd.to_datetime('09:30:00') + pd.Timedelta(minutes=open_range_min)).time()

    # Extract open range (first X minutes)
    open_range = df[(df['datetime'].dt.time >= market_open) & (df['datetime'].dt.time <= open_range_end)]
    df['OpenRange_High'] = open_range['high'].max()
    df['OpenRange_Low'] = open_range['low'].min()

    # Pre-market high/low
    pre_market = df[df['datetime'].dt.time < market_open]
    df['PreMarket_High'] = pre_market['high'].max()
    df['PreMarket_Low'] = pre_market['low'].min()

    # Previous day high/low
    df['date'] = df['datetime'].dt.date
    previous_day = df[df['date'] == (df['date'].min() - pd.Timedelta(days=1))]
    df['PrevDay_High'] = previous_day['high'].max()
    df['PrevDay_Low'] = previous_day['low'].min()

    return df

def determine_trade_direction(df):
    """
    Determines if the trade should be bullish (buy CALL) or bearish (buy PUT).
    Also identifies failed breakouts and breakout flips.
    
    Returns:
    - Updated DataFrame with 'Trade_Type' column (CALL, PUT, EXIT, NONE)
    """
    df = df.copy()

    # Identify breakout conditions
    df['Breakout_Long'] = df['close'] > df['OpenRange_High']
    df['Breakout_Short'] = df['close'] < df['OpenRange_Low']

    # EMA Trend Confirmation
    df['EMA_Fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=21, adjust=False).mean()
    df['EMA_Trend'] = np.where(df['EMA_Fast'] > df['EMA_Slow'], 1, -1)

    # **Scenario 3: Confirm Breakout using 5-min rolling window**
    df['Confirmed_Breakout'] = df['Breakout_Long'] & (df['close'].rolling(5).mean() > df['OpenRange_High'])

    # **Scenario 2: Handle Breakout Failures**
    df['Breakout_Failure'] = df['Breakout_Long'] & (df['close'] < df['OpenRange_High'])

    # **Scenario 2: Flip to Bearish if Breakout Fails & Price Drops Below OpenRange_Low**
    df['Breakout_Failure_Short'] = df['Breakout_Failure'] & (df['close'] < df['OpenRange_Low']) & (df['EMA_Fast'] < df['EMA_Slow'])

    # **Define Trade Signals**
    df['Trade_Type'] = 'NONE'
    df.loc[df['Confirmed_Breakout'], 'Trade_Type'] = 'CALL'  # Only enter long if confirmed
    df.loc[df['Breakout_Failure'], 'Trade_Type'] = 'EXIT'  # Exit if breakout fails
    df.loc[df['Breakout_Failure_Short'], 'Trade_Type'] = 'PUT'  # Flip to PUT if it breaks down

    return df
