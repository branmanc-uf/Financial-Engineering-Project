from strategy_test_df import test_strategy_from_df
from itertools import product
import pandas as pd
import gc
import random

def generate_combination_csv(param_grid, output_file="combinations.csv"):
    """
    Generates a CSV file containing all possible parameter combinations.

    Parameters:
    - param_grid: Dictionary containing parameter ranges for grid search.
    - output_file: Path to the CSV file where combinations will be saved.

    Returns:
    - None. The combinations are saved to the specified CSV file.
    """
    # Generate all combinations of parameters
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # Create a DataFrame for the combinations
    combinations_df = pd.DataFrame(param_combinations, columns=param_names)

    # Add an index column for reference
    combinations_df.reset_index(inplace=True)
    combinations_df.rename(columns={"index": "Combination_Index"}, inplace=True)

    # Save to CSV
    combinations_df.to_csv(output_file, index=False)
    print(f"Combination CSV generated: {output_file}")

def grid_search_strategy(df, param_grid_csv="combinations.csv", batch_start=0, batch_end=20, output_file="grid_search_results.csv"):
    """
    Performs a grid search over the parameter grid for the strategy test in batches.

    Parameters:
    - df: DataFrame containing stock data.
    - param_grid_csv: Path to the CSV file containing parameter combinations.
    - batch_start: Starting index of the batch to process.
    - batch_end: Ending index of the batch to process.
    - output_file: Path to the CSV file where results will be appended.

    Returns:
    - None. Results are appended to the specified CSV file.
    """
    # Load parameter combinations from the CSV file
    param_combinations = pd.read_csv(param_grid_csv)

    # Select the batch of combinations to process
    batch_combinations = param_combinations.iloc[batch_start:batch_end]

    results = []

    for _, row in batch_combinations.iterrows():
        # Convert the row to a dictionary
        param_dict = row.to_dict()

        try:
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

        except Exception as e:
            # Log the error and skip the current parameter combination
            print(f"Error with parameters {param_dict}: {e}")
            continue

        # Clean up memory to prevent kernel crashes
        del metrics
        gc.collect()

    # Append batch results to the CSV file
    if results:
        results_df = pd.DataFrame(results)
        if batch_start == 0 and not pd.io.common.file_exists(output_file):
            # Write header only if the file doesn't exist
            results_df.to_csv(output_file, mode='w', index=False)
        else:
            # Append without writing the header
            results_df.to_csv(output_file, mode='a', index=False, header=False)

    print(f"Processed combinations from index {batch_start} to {batch_end}.")

def grid_search_strategy_random(df, param_grid, num_random_combinations=3):
    """
    Performs a grid search over a random subset of parameter combinations for the strategy test.

    Parameters:
    - df: DataFrame containing stock data.
    - param_grid: Dictionary containing parameter ranges for grid search.
    - num_random_combinations: Number of random parameter combinations to test.

    Returns:
    - DataFrame containing results for the selected parameter combinations.
    """
    # Generate all combinations of parameters
    param_combinations = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())

    # Select random combinations
    random_combinations = random.sample(param_combinations, min(num_random_combinations, len(param_combinations)))

    results = []

    for idx, params in enumerate(random_combinations):
        # Map parameter combination to dictionary
        param_dict = dict(zip(param_names, params))

        try:
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

                # Stop the grid search early if Sharpe Ratio exceeds 0.6
                if filtered_metrics['Sharpe Ratio'] > 0.6:
                    print(f"Stopping early: Sharpe Ratio exceeded 0.6 with parameters {param_dict}")
                    break

        except Exception as e:
            # Log the error and skip the current parameter combination
            print(f"Error with parameters {param_dict}: {e}")
            continue

        # Clean up memory to prevent kernel crashes
        del metrics
        gc.collect()

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Example usage
param_grid = {
    'initial_capital': [25000],
    'risk_per_trade': [0.1, 0.2],
    'open_range_min': [10, 15, 20],
    'stop_loss_pct': [0.05, 0.1, 0.15],
    'take_profit_pct': [0.1, 0.2],
    'expiration_days': [1, 2, 3],
    'sigma': [0.2],
    'max_trades': [1, 3],
    'max_minutes_in_trade': [60],
    'max_daily_loss_pct': [0.15],
    'max_portfolio_cap': [4],
    'max_drawdown_pct': [0.70],
    'max_consecutive_losses': [7],
    'trailing_stop_pct': [0.25],
    'decay_threshold': [0.03]
}

# Generate the combination CSV
generate_combination_csv(param_grid, output_file="combinations.csv")

# Perform grid search for a specific batch
df = pd.read_csv('df_2023_2025.csv', parse_dates=['datetime'])
df['datetime'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert('America/New_York')

grid_search_strategy(df, param_grid_csv="combinations.csv", batch_start=0, batch_end=20, output_file="grid_search_results.csv")

# Perform random grid search
random_results = grid_search_strategy_random(df, param_grid, num_random_combinations=3)
random_results.to_csv("random_grid_search_results.csv", index=False)
print("Random grid search completed. Results saved to random_grid_search_results.csv.")