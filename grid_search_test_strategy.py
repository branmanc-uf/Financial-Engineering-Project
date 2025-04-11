from strategy_test_df import test_strategy_from_df
from itertools import product
import pandas as pd
import gc

def grid_search_strategy(df, param_grid):
    """
    Performs a grid search over the parameter grid for the strategy test.

    Parameters:
    - df: DataFrame containing stock data.
    - param_grid: Dictionary containing parameter ranges for grid search.

    Returns:
    - DataFrame containing results for each parameter combination.
    """
    # Generate all combinations of parameters
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    results = []

    for params in param_combinations:
        # Map parameter combination to dictionary
        param_dict = dict(zip(param_names, params))

        # Run the strategy test with the current parameter combination
        metrics = test_strategy_from_df(
            df,
            initial_capital=param_dict['initial_capital'],
            risk_per_trade=param_dict['risk_per_trade'],
            open_range_min=param_dict['open_range_min'],
            stop_loss_pct=param_dict['stop_loss_pct'],
            take_profit_pct=param_dict['take_profit_pct'],
            expiration_days=param_dict['expiration_days'],
            sigma=param_dict['sigma'],
            max_trades=param_dict['max_trades'],
            max_minutes_in_trade=param_dict['max_minutes_in_trade'],
            max_daily_loss_pct=param_dict['max_daily_loss_pct'],
            max_portfolio_cap=param_dict['max_portfolio_cap'],
            max_drawdown_pct=param_dict['max_drawdown_pct'],
            max_consecutive_losses=param_dict['max_consecutive_losses'],
            trailing_stop_pct=param_dict['trailing_stop_pct'],
            decay_threshold=param_dict['decay_threshold']
        )

        # If the strategy test returned results, store only the required metrics
        if metrics:
            filtered_metrics = {
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Max Drawdown': metrics['Max Drawdown'],
                'Win Rate': metrics['Win Rate'],
                'Final Portfolio Value': metrics['Final Portfolio Value'],
                'Final Strategy Return': metrics['Final Strategy Return'],
                'Total Trades': metrics['Total Trades']
            }
            results.append({**param_dict, **filtered_metrics})

        # Clean up memory to prevent kernel crashes
        del metrics
        gc.collect()

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

'''param_grid = {
    'initial_capital': [25000],
    'risk_per_trade': [0.1, 0.2],
    'open_range_min': [10, 15, 20],
    'stop_loss_pct': [0.05, 0.1, 0.15],
    'take_profit_pct': [0.1, 0.2],
    'expiration_days': [1, 2, 3],
    'sigma': [0.2],  # You can expand this if you want to test different volatilities
    'max_trades': [1, 3],
    'max_minutes_in_trade': [60],
    'max_daily_loss_pct': [0.15],
    'max_portfolio_cap': [4],
    'max_drawdown_pct': [0.70],
    'max_consecutive_losses': [7],
    'trailing_stop_pct': [0.25],
    'decay_threshold': [0.03]
}'''

param_grid = {
    'initial_capital': [25000],
    'risk_per_trade': [0.2],
    'open_range_min': [20],
    'stop_loss_pct': [0.15],
    'take_profit_pct': [0.2],
    'expiration_days': [3],
    'sigma': [0.2],
    'max_trades': [3],
    'max_minutes_in_trade': [60],
    'max_daily_loss_pct': [0.15],
    'max_portfolio_cap': [4],
    'max_drawdown_pct': [0.70],
    'max_consecutive_losses': [7],
    'trailing_stop_pct': [0.25],
    'decay_threshold': [0.03]
}
