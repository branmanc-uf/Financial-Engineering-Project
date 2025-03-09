import pandas as pd
from Gen_SPY_With_Indicators import simulate_stock

def generate_candlestick(simulated_data):
    # Simulate stock data
    df = simulated_data

    # Initialize lists to store the aggregated data
    open_list = []
    close_list = []
    high_list = []
    low_list = []
    volume_list = []
    day_list = []
    timestamp_list = []

    # Iterate through the dataframe in chunks of 5
    for i in range(0, len(df), 30):
        chunk = df.iloc[i:i+30]
        if not chunk.empty:
            open_list.append(chunk['Close'].iloc[0])
            close_list.append(chunk['Close'].iloc[-1])
            high_list.append(chunk['Close'].max())
            low_list.append(chunk['Close'].min())
            volume_list.append(chunk['Volume'].sum())
            day_list.append(chunk['Day'].sum() // 5)  # Extract the day from the index
            timestamp_list.append(chunk['Timestamp'].iloc[0])

    # Create a new dataframe with the aggregated data
    df_resampled = pd.DataFrame({
        'Timestamp': timestamp_list,
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
    fake_stock, nothingOne, nothingTwo = simulate_stock(days)
    candlestick_df = generate_candlestick(fake_stock)
    print("this is the candlestick df")
    print(candlestick_df)