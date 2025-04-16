from getting_options import black_scholes_dataframe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def preprocess_intraday_df(df):
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def test_strategy_from_df(df,
                          initial_capital,               # Starting cash balance for the strategy
                          risk_per_trade,               # Fraction of portfolio to risk per trade (e.g. 0.1 = 10%)
                          open_range_min,               # Number of minutes after open to define breakout range
                          stop_loss_pct,                # Percentage loss that triggers a stop on a trade
                          take_profit_pct,              # Percentage gain that triggers profit-taking
                          expiration_days,              # Days to expiration used in Black-Scholes option pricing
                          sigma,                        # Assumed/implied volatility used in Black-Scholes model
                          max_trades=3,                 # Max number of trades allowed per day
                          max_minutes_in_trade=60,      # Max time (in minutes) a position can be held
                          max_daily_loss_pct=0.15,      # Max % of portfolio you can lose in a single day before halting trades
                          max_portfolio_cap=4,          # Cap portfolio growth (e.g. 4x starting capital) to avoid unrealistic compounding
                          max_drawdown_pct=0.70,        # Max intraday drawdown allowed (reset daily)
                          max_consecutive_losses=7,     # Number of losing trades in a row before trading halts for the day
                          trailing_stop_pct=0.25,       # Trailing stop to protect profits if price falls from recent high
                          decay_threshold=0.03):        # If no price movement after max_minutes, this % threshold triggers an exit

    df = preprocess_intraday_df(df)

    if df is None or df.empty:
        print("\u26a0\ufe0f No data provided.")
        return None

    df = df.copy()
    df = identify_breakout_levels(df, open_range_min=open_range_min)
    df = determine_trade_direction(df)
    df = black_scholes_dataframe(df, sigma=sigma, expiration_days=expiration_days)

    df.set_index('datetime', inplace=True)
    portfolio = float(initial_capital)
    position = None
    entry_price = None
    entry_time = None
    trade_size = 0.0
    contracts = 0
    total_cost = 0.0
    df['Portfolio Value'] = float(portfolio)
    df['Realized PnL'] = 0.0

    df['range'] = df['high'] - df['low']
    df['avg_range'] = df['range'].rolling(15).mean()
    volatility_threshold = df['avg_range'].quantile(0.5)
    df['Vol_Filter'] = df['avg_range'] > volatility_threshold

    trade_log = []
    current_day = None
    trade_count = 0
    consecutive_losses = 0
    daily_portfolio_high = portfolio

    for i in range(1, len(df)):
        now = df.index[i]
        trade_time = now.time()
        if trade_time < pd.to_datetime("09:30:00").time() or trade_time > pd.to_datetime("16:00:00").time():
            continue

        new_day = now.date()
        if new_day != current_day:
            current_day = new_day
            trade_count = 0
            consecutive_losses = 0
            daily_portfolio_high = portfolio

        day_trades = [log for log in trade_log if pd.to_datetime(log['Exit Time']).date() == current_day]
        daily_pnl = sum([log['PnL'] for log in day_trades])
        if daily_pnl < -max_daily_loss_pct * portfolio:
            continue

        # Daily drawdown based on daily portfolio high
        daily_drawdown = (portfolio / daily_portfolio_high) - 1
        if daily_drawdown < -max_drawdown_pct:
            continue

        if consecutive_losses >= max_consecutive_losses:
            continue

        if trade_count >= max_trades:
            continue

        if not df.iloc[i]['Vol_Filter']:
            continue

        trade_type = df.iloc[i]['Trade_Type']

        if trade_type in ["CALL", "PUT"] and position is None:
            entry_price = df.iloc[i]['Call_Price'] if trade_type == "CALL" else df.iloc[i]['Put_Price']
            if np.isnan(entry_price) or entry_price <= 0 or entry_price > 100:
                continue

            drawdown_ratio = (portfolio / daily_portfolio_high) - 1
            adjusted_risk = max(0.01, risk_per_trade * (1 + drawdown_ratio))
            trade_size = adjusted_risk * portfolio

            contracts = int(trade_size // (entry_price * 100))
            if contracts == 0:
                continue

            position = trade_type
            entry_time = now
            total_cost = contracts * entry_price * 100
            peak_price = entry_price

        elif position is not None:
            is_eod = (i + 1 == len(df)) or (df.index[i + 1].date() != current_day)

            current_price = df.iloc[i]['Call_Price'] if position == "CALL" else df.iloc[i]['Put_Price']
            if np.isnan(current_price):
                continue

            peak_price = max(peak_price, current_price)
            trailing_stop_triggered = current_price <= peak_price * (1 - trailing_stop_pct)

            stop_loss_level = entry_price * (1 - stop_loss_pct)
            take_profit_level = entry_price * (1 + take_profit_pct)
            time_in_trade = (now - entry_time).total_seconds() / 60 if entry_time else 0
            decay_exit = time_in_trade >= max_minutes_in_trade and abs(current_price - entry_price) / entry_price < decay_threshold

            exit_condition = (
                current_price <= stop_loss_level or
                current_price >= take_profit_level or
                trailing_stop_triggered or
                is_eod or
                decay_exit
            )

            if exit_condition:
                if current_price <= stop_loss_level:
                    reason = 'Stop Loss'
                elif current_price >= take_profit_level:
                    reason = 'Take Profit'
                elif trailing_stop_triggered:
                    reason = 'Trailing Stop'
                elif is_eod:
                    reason = 'EOD Force Exit'
                else:
                    reason = 'Time/Price Decay'

                trade_profit = contracts * (current_price - entry_price) * 100

                '''
                if abs(trade_profit) > 0.5 * portfolio:
                    print(f"Large trade: {trade_profit:.2f} at {now}")
                '''
                
                portfolio += trade_profit
                daily_portfolio_high = max(daily_portfolio_high, portfolio)
                df.iloc[i, df.columns.get_loc('Realized PnL')] = float(trade_profit)

                trade_log.append({
                    'Entry Time': entry_time,
                    'Exit Time': now,
                    'Type': position,
                    'Entry Price': entry_price,
                    'Exit Price': current_price,
                    'PnL': trade_profit,
                    'Contracts': contracts,
                    'Total Cost': total_cost,
                    'Reason': reason
                })

                position = None
                entry_price = None
                trade_size = 0.0
                contracts = 0
                total_cost = 0.0
                entry_time = None
                trade_count += 1

                if trade_profit < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

        if portfolio <= 0:
            print("Portfolio wiped out")
            break

        portfolio = min(portfolio, max_portfolio_cap * initial_capital)
        df.iloc[i, df.columns.get_loc('Portfolio Value')] = float(portfolio)

    df['Market Return'] = df['close'].pct_change()
    df['Cumulative Market Return'] = (1 + df['Market Return']).cumprod()
    df['Cumulative Strategy Return'] = df['Portfolio Value'] / initial_capital

    realized = df['Realized PnL']
    valid_pnls = realized[realized != 0]
    win_rate = (valid_pnls > 0).mean() if not valid_pnls.empty else 0.0

    sharpe_ratio = realized.mean() / realized.std() * np.sqrt(252) if realized.std() > 0 else 0
    rolling_max = df['Cumulative Strategy Return'].cummax()
    drawdown = (df['Cumulative Strategy Return'] / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    reason_summary = pd.DataFrame(trade_log).groupby("Reason")['PnL'].agg(['count', 'mean', 'sum'])
    print("\nExit Reason Summary:\n", reason_summary)

    # Plot and save the return graph
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative Market Return'], label="Market Return", linestyle="dashed")
    plt.plot(df.index, df['Cumulative Strategy Return'], label="Strategy Return", color='green')
    plt.legend()
    date_range = f"{df.index[0].date()} to {df.index[-1].date()}"
    plt.title(f"Strategy Performance ({date_range})\nSharpe: {sharpe_ratio:.2f}, Max DD: {max_drawdown:.2%}")
    plt.savefig("Return_Graph.png", bbox_inches="tight")
    plt.show()

    # Plot and save the PnL distribution graph
    plt.figure(figsize=(8, 4))
    pd.Series([t['PnL'] for t in trade_log]).hist(bins=100)
    plt.title("PnL Distribution")
    plt.xlabel("Profit/Loss per Trade")
    plt.ylabel("Frequency")
    plt.savefig("PnL_Distribution_Graph.png", bbox_inches="tight")
    plt.show()
    
    total_trades = len(trade_log)

    return {
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Max Drawdown': round(max_drawdown * 100, 2),
        'Win Rate': round(win_rate * 100, 2),
        'Final Portfolio Value': round(portfolio, 2),
        'Final Strategy Return': round(df['Cumulative Strategy Return'].iloc[-1] * 100, 2),
        'Total Trades': total_trades,
        'DataFrame': df.reset_index(),
        'Trade Log': pd.DataFrame(trade_log)
    }


# identify_breakout_levels and determine_trade_direction stay the same

def determine_trade_direction(df):
    df = df.copy()
    df['Breakout_Long'] = df['high'] > df['OpenRange_High']
    df['Breakout_Short'] = df['close'] < df['OpenRange_Low']

    df['Confirmed_Breakout'] = df['Breakout_Long']
    df['Breakout_Failure'] = df['Breakout_Long'] & (df['close'] < df['OpenRange_High'])
    df['Breakout_Failure_Short'] = df['Breakout_Failure'] & (df['close'] < df['OpenRange_Low'])

    df['Trade_Type'] = 'NONE'
    df.loc[df['Confirmed_Breakout'], 'Trade_Type'] = 'CALL'
    df.loc[df['Breakout_Failure_Short'], 'Trade_Type'] = 'PUT'

    return df


def identify_breakout_levels(df, open_range_min=15):
    df = df.copy()
    df['date'] = df['datetime'].dt.date

    breakout_levels = []
    for day, group in df.groupby('date'):
        market_open = pd.to_datetime(f"{day} 09:30:00-04:00")
        open_range_end = market_open + pd.Timedelta(minutes=open_range_min)

        open_range = group[(group['datetime'] >= market_open) & (group['datetime'] <= open_range_end)]
        pre_market = group[group['datetime'] < market_open]
        prev_day = df[df['date'] == (pd.to_datetime(day) - pd.Timedelta(days=1)).date()]

        high = open_range['high'].max() if not open_range.empty else np.nan
        low = open_range['low'].min() if not open_range.empty else np.nan
        pm_high = pre_market['high'].max() if not pre_market.empty else np.nan
        pm_low = pre_market['low'].min() if not pre_market.empty else np.nan
        prev_high = prev_day['high'].max() if not prev_day.empty else np.nan
        prev_low = prev_day['low'].min() if not prev_day.empty else np.nan

        group = group.copy()
        group['OpenRange_High'] = high
        group['OpenRange_Low'] = low
        group['PreMarket_High'] = pm_high
        group['PreMarket_Low'] = pm_low
        group['PrevDay_High'] = prev_high
        group['PrevDay_Low'] = prev_low

        breakout_levels.append(group)

    return pd.concat(breakout_levels)