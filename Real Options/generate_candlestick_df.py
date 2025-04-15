import pandas as pd
from Gen_SPY_With_Indicators import simulate_stock

def generate_candlestick(simulated_data, interval):
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
    for i in range(0, len(df), interval):
        chunk = df.iloc[i:i+interval]
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
    candlestick_df = generate_candlestick(fake_stock, 30)
    print("this is the candlestick df")
    print(candlestick_df)

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

def plot_candlestick_with_indicators(candlestick_data, simulated_data):
    # Create a figure with a single subplot
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=("Candlestick Plot",))
    # Add candlestick traces
    fig.add_trace(go.Candlestick(
        x=candlestick_data.index,
        open=candlestick_data['Open'],
        high=candlestick_data['High'],
        low=candlestick_data['Low'],
        close=candlestick_data['Close'],
        name='Candlestick',
        opacity=1
    ), row=1, col=1)

    # Add indicators to the candlestick plot
    fig.add_trace(go.Scatter(
        x=simulated_data['Timestamp'],
        y=simulated_data['ORB_High'],
        mode='lines',
        name='ORB High',
        line=dict(color='green', dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=simulated_data['Timestamp'],
        y=simulated_data['ORB_Low'],
        mode='lines',
        name='ORB Low',
        line=dict(color='red', dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=simulated_data['Timestamp'],
        y=simulated_data['Yest_High'],
        mode='lines',
        name="Yesterday's High",
        line=dict(color='gray', dash='dashdot')
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=simulated_data['Timestamp'],
        y=simulated_data['Yest_Low'],
        mode='lines',
        name="Yesterday's Low",
        line=dict(color='brown', dash='dashdot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=simulated_data['Timestamp'],
        y=simulated_data['PM_High'],
        mode='lines',
        name='PM High',
        line=dict(color='green', dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=simulated_data['Timestamp'],
        y=simulated_data['PM_Low'],
        mode='lines',
        name='PM Low',
        line=dict(color='red', dash='dot')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=simulated_data['Timestamp'],
        y=simulated_data['8EMA'],
        mode='lines',
        name='8EMA',
        line=dict(color='orange')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=simulated_data['Timestamp'],
        y=simulated_data['VWAP'],
        mode='lines',
        name='VWAP',
        line=dict(color='blue')
    ), row=1, col=1)

    # Add PM and Regular Market values
    fig.add_trace(go.Scatter(
        x=simulated_data[simulated_data['Session'] == 'PM']['Timestamp'],
        y=simulated_data[simulated_data['Session'] == 'PM']['Close'],
        mode='lines',
        name='Pre-Market Value',
        line=dict(color='black', dash='dot'),
        opacity=0.7
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=simulated_data[simulated_data['Session'] == 'Regular Market']['Timestamp'],
        y=simulated_data[simulated_data['Session'] == 'Regular Market']['Close'],
        mode='lines',
        name='Regular Market Value',
        line=dict(color='steelblue'),
        opacity=0.8
    ), row=1, col=1)

    # Update layout for better appearance
    fig.update_layout(
        title="Simulated Stock Price with Indicators",
        xaxis_title="Timestamp",
           yaxis_title="Price",
        legend_title="Legend",
        xaxis=dict(tickangle=45),
        template="seaborn",
        hovermode='x unified',
        height=800  # Make the graph taller
    )

    # Show the plot
    fig.show()