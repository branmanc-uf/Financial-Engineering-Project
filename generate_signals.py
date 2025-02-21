import pandas as pd
from Gen_SPY_With_Indicators import simulate_stock  # Importing simulation function

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

# Simulate Stock Data for 5 Days
simulated_data = simulate_stock(1)

# Apply Signal Generation
simulated_data = generate_signals(simulated_data)

# Print the head of the simulated data
print(simulated_data.head())


