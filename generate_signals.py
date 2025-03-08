import numpy as np  # Importing numpy for numerical operations
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
from Gen_SPY_With_Indicators import simulate_stock  # Importing simulation function
import matplotlib.pyplot as plt  # Importing matplotlib for plotting
import matplotlib.ticker as mticker  # For proper Y-axis formatting

def generate_signals(df):
    """ Generates trade signals based on ORB, VWAP, 8EMA, PM High/Low, and Yest High/Low """

    df['Signal'] = ''

    # Buy Call: Price breaks ORB High & supported by 8EMA & VWAP
    df.loc[
        (df['Close'] > df['ORB_High']) & 
        (df['Close'] > df['VWAP']) & 
        (df['Close'] > df['8EMA']),
        'Signal'
    ] = 'BUY CALL'

    # Sell Put: Price breaks ORB Low & rejected by 8EMA & VWAP
    df.loc[
        (df['Close'] < df['ORB_Low']) & 
        (df['Close'] < df['VWAP']) & 
        (df['Close'] < df['8EMA']),
        'Signal'
    ] = 'BUY PUT'

    # Boost Call Confidence if PM High or Yesterday's High is broken
    
    df.loc[
        (df['Signal'] == 'BUY CALL') & 
        ((df['Close'] > df['PM_High']) | (df['Close'] > df['Yest_High'])),
        'Signal'
    ] = 'STRONG BUY CALL'
    

    
    # Boost Put Confidence if PM Low or Yesterday's Low is broken
    df.loc[
        (df['Signal'] == 'BUY PUT') & 
        ((df['Close'] < df['PM_Low']) | (df['Close'] < df['Yest_Low'])),
        'Signal'
    ] = 'STRONG BUY PUT'

    return df

# Simulate Stock Data for 2 Days
simulated_data, yesterday_high, yesterday_low = simulate_stock(2)

# Apply Signal Generation
simulated_data = generate_signals(simulated_data)

import plotly.express as px
import plotly.graph_objects as go

# Set seaborn style
sns.set(style="whitegrid")

# Create a subplot
fig = make_subplots(rows=1, cols=1)

# Add traces for each day
for day in simulated_data["Day"].unique():
    day_data = simulated_data[simulated_data["Day"] == day]
    pm_data = day_data[day_data["Session"] == "PM"]
    market_data = day_data[day_data["Session"] == "Regular Market"]

    fig.add_trace(go.Scatter(x=pm_data['Timestamp'], y=pm_data['Close'], mode='lines', name=f"PM Day {day}", line=dict(color="black", dash="dot"), opacity=0.7))
    fig.add_trace(go.Scatter(x=market_data['Timestamp'], y=market_data['Close'], mode='lines', name=f"Regular Market Day {day}", line=dict(color="steelblue"), opacity=0.8))

# Add traces for indicators
fig.add_trace(go.Scatter(x=simulated_data['Timestamp'], y=simulated_data['8EMA'], mode='lines', name="8EMA", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=simulated_data['Timestamp'], y=simulated_data['VWAP'], mode='lines', name="VWAP", line=dict(color="blue")))

# Add traces for ORB and PM High/Low
for day in simulated_data['Day'].unique():
    day_data = simulated_data[simulated_data['Day'] == day]
    fig.add_trace(go.Scatter(x=day_data['Timestamp'], y=day_data['ORB_High'], mode='lines', name="ORB High" if day == 1 else "", line=dict(color="green", dash="dash")))
    fig.add_trace(go.Scatter(x=day_data['Timestamp'], y=day_data['ORB_Low'], mode='lines', name="ORB Low" if day == 1 else "", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(x=day_data['Timestamp'], y=day_data['PM_High'], mode='lines', name="PM High" if day == 1 else "", line=dict(color="green", dash="dot")))
    fig.add_trace(go.Scatter(x=day_data['Timestamp'], y=day_data['PM_Low'], mode='lines', name="PM Low" if day == 1 else "", line=dict(color="red", dash="dot")))

    market_open_time = day_data['Timestamp'].iloc[0]
    market_close_time = day_data['Timestamp'].iloc[-1]
    if day == 1:
        fig.add_trace(go.Scatter(x=[market_open_time, market_close_time], y=[yesterday_high, yesterday_high], mode='lines', name="Yesterday's High", line=dict(color="gray", dash="dashdot"), opacity=0.7))
        fig.add_trace(go.Scatter(x=[market_open_time, market_close_time], y=[yesterday_low, yesterday_low], mode='lines', name="Yesterday's Low", line=dict(color="brown", dash="dashdot"), opacity=0.7))
    else:
        fig.add_trace(go.Scatter(x=[market_open_time, market_close_time], y=[day_data['Yest_High'].iloc[0], day_data['Yest_High'].iloc[0]], mode='lines', name="Yesterday's High" if day == 2 else "", line=dict(color="gray", dash="dashdot"), opacity=0.7))
        fig.add_trace(go.Scatter(x=[market_open_time, market_close_time], y=[day_data['Yest_Low'].iloc[0], day_data['Yest_Low'].iloc[0]], mode='lines', name="Yesterday's Low" if day == 2 else "", line=dict(color="brown", dash="dashdot"), opacity=0.7))

# Update layout for better appearance
fig.update_layout(
    title="Simulated Stock Price with PM, 8EMA, VWAP, ORB, & Yesterday's High/Low",
    xaxis_title="Time",
    yaxis_title="Stock Price",
    legend_title="Legend",
    xaxis=dict(tickangle=45),
    template="seaborn"
)

# Show the plot
fig.show()

# Print any part of the simulated data that contains BUY CALL or BUY PUT signals during Regular Market session
print(simulated_data[(simulated_data['Signal'].isin(['BUY CALL', 'BUY PUT', 'STRONG BUY CALL', 'STRONG BUY PUT'])) & 
                     (simulated_data['Session'] == 'Regular Market')])