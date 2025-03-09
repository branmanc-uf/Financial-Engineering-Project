import pandas as pd
from Gen_SPY_With_Indicators import simulate_stock

def generate_candlestick(simulated_data):
    # Simulate stock data
    df = simulated_data


    print(df)

    # Initialize lists to store the aggregated data
    open_list = []
    close_list = []
    high_list = []
    low_list = []
    volume_list = []
    day_list = []

    # Iterate through the dataframe in chunks of 5
    for i in range(0, len(df), 5):
        chunk = df.iloc[i:i+5]
        if not chunk.empty:
            open_list.append(chunk['Close'].iloc[0])
            close_list.append(chunk['Close'].iloc[-1])
            high_list.append(chunk['Close'].max())
            low_list.append(chunk['Close'].min())
            volume_list.append(chunk['Volume'].sum())
            day_list.append(chunk['Day'].sum() // 5)  # Extract the day from the index

    # Create a new dataframe with the aggregated data
    df_resampled = pd.DataFrame({
        'Open': open_list,
        'Close': close_list,
        'High': high_list,
        'Low': low_list,
        'Volume': volume_list,
        'Day': day_list
    })

    return df_resampled

# Example usage
if __name__ == "__main__":
    days = 2  # Specify the number of days for simulation
    candlestick_df = generate_candlestick(simulate_stock(days))
    print("this is the candlestick df")
    print(candlestick_df)