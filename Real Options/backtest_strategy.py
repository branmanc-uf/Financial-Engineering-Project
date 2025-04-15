import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from ML.Price_Pred_30min import fetch_data, prepare_features, train_model, make_prediction
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.dates as mdates

class TradingStrategy:
    def __init__(self, config):
        self.config = config
        self.initial_capital = config.initial_capital
        self.capital = config.initial_capital
        self.position = 0  # 1 for long, -1 for short, 0 for no position
        self.trades = []
        self.portfolio_value = []
        self.current_price = None
        self.open_positions = 0
        
        # Adjust parameters based on risk level
        self._adjust_risk_parameters()
    
    def _adjust_risk_parameters(self):
        """Adjust trading parameters based on risk level"""
        if self.config.risk_level == "Conservative":
            self.config.position_size *= 0.5
            self.config.stop_loss *= 0.8
            self.config.take_profit *= 0.8
        elif self.config.risk_level == "Aggressive":
            self.config.position_size *= 2
            self.config.stop_loss *= 1.2
            self.config.take_profit *= 1.2
            self.config.leverage = min(self.config.leverage * 1.5, 3)  # Cap leverage at 3x
    
    def execute_trade(self, predicted_direction, current_price, timestamp):
        """
        Execute trade based on predicted direction and strategy parameters
        """
        # Check if we can open new positions
        if self.position == 0 and self.open_positions < self.config.max_positions:
            # Calculate position size based on capital and risk
            position_value = self.capital * self.config.position_size * self.config.leverage
            
            # Check stop loss and take profit levels
            if predicted_direction == 1:  # Long position
                stop_loss_price = current_price * (1 - self.config.stop_loss)
                take_profit_price = current_price * (1 + self.config.take_profit)
            else:  # Short position
                stop_loss_price = current_price * (1 + self.config.stop_loss)
                take_profit_price = current_price * (1 - self.config.take_profit)
            
            # Open new position
            self._open_position(predicted_direction, current_price, timestamp, 
                              position_value, stop_loss_price, take_profit_price)
        
        # Check if we need to close existing positions
        elif self.position != 0:
            # Close if direction changes or stop loss/take profit hit
            if (self.position != predicted_direction or
                (self.position == 1 and current_price <= self.stop_loss_price) or
                (self.position == 1 and current_price >= self.take_profit_price) or
                (self.position == -1 and current_price >= self.stop_loss_price) or
                (self.position == -1 and current_price <= self.take_profit_price)):
                self._close_position(current_price, timestamp)
    
    def _open_position(self, direction, price, timestamp, position_value, stop_loss_price, take_profit_price):
        """Open a new trading position"""
        self.position = direction
        self.current_price = price
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.open_positions += 1
        
        # Apply commission
        commission = position_value * self.config.commission
        self.capital -= commission
        
        self.trades.append({
            'timestamp': timestamp,
            'type': 'LONG' if direction == 1 else 'SHORT',
            'price': price,
            'position_value': position_value,
            'commission': commission,
            'capital': self.capital,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price
        })
    
    def _close_position(self, price, timestamp):
        """Close an existing trading position"""
        if self.position == 0:
            return
            
        # Calculate returns
        price_change = (price - self.current_price) / self.current_price
        if self.position == -1:  # If short position
            price_change = -price_change
            
        # Calculate position value and returns
        position_value = self.capital * self.config.position_size * self.config.leverage
        trade_return = position_value * price_change
        
        # Apply commission
        commission = position_value * self.config.commission
        trade_return -= commission
        
        # Update capital
        self.capital += trade_return
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'type': 'CLOSE',
            'price': price,
            'return': trade_return,
            'commission': commission,
            'capital': self.capital
        })
        
        self.position = 0
        self.current_price = None
        self.open_positions -= 1

def calculate_performance_metrics(trades, portfolio_values):
    """Calculate trading strategy performance metrics"""
    if not trades:
        return {}
        
    # Convert trades to DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate basic metrics
    total_trades = len(trades_df[trades_df['type'] == 'CLOSE'])
    winning_trades = len(trades_df[trades_df['return'] > 0])
    losing_trades = len(trades_df[trades_df['return'] <= 0])
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate returns
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    
    # Calculate daily returns for Sharpe ratio
    portfolio_df = pd.DataFrame({'value': portfolio_values})
    daily_returns = portfolio_df['value'].pct_change()
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() - 0.02/252) / daily_returns.std() if len(daily_returns) > 1 else 0
    
    # Calculate maximum drawdown
    portfolio_df['peak'] = portfolio_df['value'].cummax()
    portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['peak']) / portfolio_df['peak']
    max_drawdown = portfolio_df['drawdown'].min()
    
    # Calculate total commission paid
    total_commission = trades_df['commission'].sum()
    
    return {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Win Rate': win_rate,
        'Total Return': total_return,
        'Sharpe Ratio': sharpe_ratio,
        'Maximum Drawdown': max_drawdown,
        'Total Commission': total_commission
    }

def plot_performance(portfolio_values, trades, config):
    """Plot portfolio performance, candlestick chart, market behavior, and trade points"""
    plt.figure(figsize=(15, 20))
    
    # Convert trades to DataFrame for easier handling
    trades_df = pd.DataFrame(trades)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    # Create time index for portfolio values
    portfolio_times = trades_df[trades_df['type'] == 'LONG']['timestamp']
    portfolio_df = pd.DataFrame({
        'timestamp': portfolio_times,
        'value': portfolio_values
    })
    
    # Plot portfolio value
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(portfolio_df['timestamp'], portfolio_df['value'], label='Portfolio Value')
    ax1.set_title(f'Portfolio Performance ({config.risk_level} Risk)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot candlestick chart
    ax2 = plt.subplot(4, 1, 2)
    
    # Get the price data from the trades
    price_data = pd.DataFrame({
        'Open': trades_df[trades_df['type'] == 'LONG']['price'],
        'High': trades_df[trades_df['type'] == 'LONG']['price'] * 1.001,  # Approximate high
        'Low': trades_df[trades_df['type'] == 'LONG']['price'] * 0.999,   # Approximate low
        'Close': trades_df[trades_df['type'] == 'CLOSE']['price']
    }, index=trades_df[trades_df['type'] == 'LONG']['timestamp'])
    
    # Plot candlesticks
    for idx in price_data.index:
        # Plot the candlestick body
        if price_data.loc[idx, 'Close'] >= price_data.loc[idx, 'Open']:
            color = 'green'
            body_bottom = price_data.loc[idx, 'Open']
            body_height = price_data.loc[idx, 'Close'] - price_data.loc[idx, 'Open']
        else:
            color = 'red'
            body_bottom = price_data.loc[idx, 'Close']
            body_height = price_data.loc[idx, 'Open'] - price_data.loc[idx, 'Close']
            
        ax2.bar(idx, body_height, bottom=body_bottom, color=color, width=0.6)
        
        # Plot the wicks
        ax2.plot([idx, idx], [price_data.loc[idx, 'Low'], price_data.loc[idx, 'High']], 
                color=color, linewidth=1)
    
    ax2.set_title('Market Price (Candlestick)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price ($)')
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot market behavior for the test day
    ax3 = plt.subplot(4, 1, 3)
    spy = yf.Ticker("SPY")
    test_day_data = spy.history(start=config.test_date, 
                              end=config.test_date + timedelta(days=1), 
                              interval='1m')
    
    if not test_day_data.empty:
        # Plot price movement
        ax3.plot(test_day_data.index, test_day_data['Close'], label='Price', color='blue')
        
        # Add volume bars at the bottom
        volume_ax = ax3.twinx()
        volume_ax.bar(test_day_data.index, test_day_data['Volume'], 
                     alpha=0.3, color='gray', label='Volume')
        volume_ax.set_ylabel('Volume')
        
        ax3.set_title(f'Market Behavior on {config.test_date.strftime("%Y-%m-%d")}')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Price ($)')
        ax3.legend(loc='upper left')
        ax3.grid(True)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot trade points
    ax4 = plt.subplot(4, 1, 4)
    long_entries = trades_df[trades_df['type'] == 'LONG']
    short_entries = trades_df[trades_df['type'] == 'SHORT']
    exits = trades_df[trades_df['type'] == 'CLOSE']
    
    ax4.scatter(long_entries['timestamp'], long_entries['price'], 
               color='green', marker='^', label='Long Entry', s=100)
    ax4.scatter(short_entries['timestamp'], short_entries['price'], 
               color='red', marker='v', label='Short Entry', s=100)
    ax4.scatter(exits['timestamp'], exits['price'], 
               color='black', marker='x', label='Exit', s=100)
    
    ax4.set_title('Trade Entry and Exit Points')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Price ($)')
    ax4.legend()
    ax4.grid(True)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

def backtest_strategy(config):
    """Run backtest of trading strategy with custom configuration"""
    print("Fetching historical data...")
    # Fetch data for the specified test date and historical period
    end_date = config.test_date + timedelta(days=1)  # Include the full test day
    start_date = config.test_date - timedelta(days=config.historical_days)
    df = fetch_data(start_date=start_date, end_date=end_date)
    
    if len(df) == 0:
        print("Error: No data available for the specified date range")
        return None, None
        
    print(f"Fetched {len(df)} data points from {df.index[0]} to {df.index[-1]}")
    
    print("Preparing features...")
    X, y = prepare_features(df)
    
    # Split data into training (historical) and testing (test date) sets
    test_date_mask = df.index.date == config.test_date.date()
    X_train = X[~test_date_mask]
    y_train = y[~test_date_mask]
    X_test = X[test_date_mask]
    y_test = y[test_date_mask]
    
    print(f"Training data points: {len(X_train)}")
    print(f"Testing data points: {len(X_test)}")
    
    print("Training model...")
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model with more trees and different random state
    model = RandomForestRegressor(n_estimators=200, 
                                max_depth=10,
                                min_samples_split=5,
                                random_state=int(datetime.now().timestamp()))
    model.fit(X_train_scaled, y_train)
    
    # Initialize strategy with config
    strategy = TradingStrategy(config)
    portfolio_values = [strategy.initial_capital]
    
    print(f"\nStarting backtest with configuration:")
    print(f"Test Date: {config.test_date.strftime('%Y-%m-%d')}")
    print(f"Historical Data Days: {config.historical_days}")
    print(f"Risk Level: {config.risk_level}")
    print(f"Position Size: {config.position_size*100}%")
    print(f"Stop Loss: {config.stop_loss*100}%")
    print(f"Take Profit: {config.take_profit*100}%")
    print(f"Leverage: {config.leverage}x")
    
    # Iterate through each 30-minute interval on the test date
    for i in range(len(X_test) - 1):
        try:
            # Get current data point
            current_data = X_test.iloc[i:i+1]
            if current_data.empty:
                continue
                
            current_price = df[test_date_mask]['Close'].iloc[i]
            next_price = df[test_date_mask]['Close'].iloc[i+1]
            timestamp = df[test_date_mask].index[i]
            
            # Make prediction
            current_data_scaled = scaler.transform(current_data)
            predicted_price = model.predict(current_data_scaled)[0]
            predicted_direction = 1 if predicted_price > current_price else -1
            
            # Execute trade based on prediction
            strategy.execute_trade(predicted_direction, current_price, timestamp)
            
            # Update portfolio value
            portfolio_values.append(strategy.capital)
            
        except Exception as e:
            print(f"Error at index {i}: {str(e)}")
            continue
    
    # Close any open position at the end
    if strategy.position != 0:
        try:
            strategy._close_position(df[test_date_mask]['Close'].iloc[-1], 
                                   df[test_date_mask].index[-1])
        except Exception as e:
            print(f"Error closing final position: {str(e)}")
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(strategy.trades, portfolio_values)
    
    # Print results
    print("\nBacktest Results:")
    print(f"Initial Capital: ${strategy.initial_capital:,.2f}")
    print(f"Final Capital: ${strategy.capital:,.2f}")
    print(f"Total Return: {metrics['Total Return']*100:.2f}%")
    print(f"Total Trades: {metrics['Total Trades']}")
    print(f"Win Rate: {metrics['Win Rate']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
    print(f"Maximum Drawdown: {metrics['Maximum Drawdown']*100:.2f}%")
    
    # Plot results
    plot_performance(portfolio_values, strategy.trades, config)
    
    return strategy, metrics

if __name__ == "__main__":
    from strategy_config import StrategyConfigGUI
    gui = StrategyConfigGUI()
    gui.run() 