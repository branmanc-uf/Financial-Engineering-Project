import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import seaborn as sns
from tabulate import tabulate

def fetch_data():
    # Fetch SPY data with 1-minute intervals
    spy = yf.Ticker("SPY")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)  # Get 5 days of 1-minute data
    df = spy.history(start=start_date, end=end_date, interval='1m')
    return df

def prepare_features(df):
    # Create technical indicators
    df['Returns'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    # Add more technical indicators
    df['MACD'] = calculate_macd(df['Close'])
    df['MACD_Signal'] = calculate_macd_signal(df['Close'])
    df['Bollinger_Upper'] = calculate_bollinger_bands(df['Close'])['upper']
    df['Bollinger_Lower'] = calculate_bollinger_bands(df['Close'])['lower']
    
    # Create target variable (next 30 minutes closing price)
    df['Target'] = df['Close'].shift(-30)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features for the model
    features = ['Open', 'High', 'Low', 'Volume', 
               'Returns', 'SMA_5', 'SMA_20', 'SMA_50', 'RSI',
               'MACD', 'MACD_Signal', 'Bollinger_Upper', 'Bollinger_Lower']
    
    X = df[features]
    y = df['Target']
    
    return X, y

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

def calculate_macd_signal(prices, fast=12, slow=26, signal=9):
    macd = calculate_macd(prices, fast, slow)
    return macd.ewm(span=signal, adjust=False).mean()

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return {'upper': upper, 'lower': lower}

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    direction_correct = np.sum(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    total_predictions = len(y_true) - 1
    directional_accuracy = (direction_correct / total_predictions) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'Directional Accuracy': directional_accuracy
    }

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, scaler, X_test_scaled, y_test, feature_importance

def make_prediction(model, scaler, latest_data):
    # Scale the latest data
    latest_data_scaled = scaler.transform(latest_data)
    
    # Make prediction
    prediction = model.predict(latest_data_scaled)
    return prediction[0]

def plot_results(y_test, y_pred, feature_importance, df):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Get the dates for the test set
    test_dates = df.index[-len(y_test):]
    
    # Plot actual vs predicted prices
    ax1.plot(test_dates, y_test.values, label='Actual', alpha=0.7)
    ax1.plot(test_dates, y_pred, label='Predicted', alpha=0.7)
    ax1.set_title('SPY Price Prediction (30 minutes ahead)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    
    # Format x-axis to show dates nicely
    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot feature importance
    feature_importance.plot(kind='bar', x='feature', y='importance', ax=ax2)
    ax2.set_title('Feature Importance')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Importance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def visualize_prediction_accuracy(y_test, y_pred, test_dates):
    # Create a DataFrame with actual and predicted prices
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual Price': y_test,
        'Predicted Price': y_pred
    })
    
    # Calculate price changes
    results_df['Actual Change'] = results_df['Actual Price'].diff()
    results_df['Predicted Change'] = results_df['Predicted Price'].diff()
    
    # Determine if prediction was correct (same direction)
    results_df['Correct Direction'] = (
        np.sign(results_df['Actual Change']) == np.sign(results_df['Predicted Change'])
    )
    
    # Calculate accuracy percentage
    accuracy = (results_df['Correct Direction'].sum() / len(results_df)) * 100
    
    # Create styled DataFrame for display
    styled_df = results_df.copy()
    styled_df['Date'] = styled_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
    styled_df['Actual Price'] = styled_df['Actual Price'].round(2)
    styled_df['Predicted Price'] = styled_df['Predicted Price'].round(2)
    styled_df['Actual Change'] = styled_df['Actual Change'].round(2)
    styled_df['Predicted Change'] = styled_df['Predicted Change'].round(2)
    
    # Create color-coded table
    print("\nPrediction Accuracy Analysis:")
    print(f"Directional Accuracy: {accuracy:.2f}%")
    print("\nDetailed Results (Green: Correct Direction, Red: Incorrect Direction):")
    
    # Print table with colors
    for idx, row in styled_df.iterrows():
        color = '\033[92m' if row['Correct Direction'] else '\033[91m'  # Green for correct, Red for incorrect
        print(f"{color}{row['Date']} | Actual: ${row['Actual Price']} ({row['Actual Change']:+.2f}) | "
              f"Predicted: ${row['Predicted Price']} ({row['Predicted Change']:+.2f})\033[0m")
    
    # Create a summary plot
    plt.figure(figsize=(10, 6))
    correct_direction = results_df['Correct Direction'].sum()
    incorrect_direction = len(results_df) - correct_direction
    
    plt.bar(['Correct Direction', 'Incorrect Direction'], 
            [correct_direction, incorrect_direction],
            color=['green', 'red'])
    plt.title('Directional Prediction Accuracy')
    plt.ylabel('Number of Predictions')
    plt.text(0, correct_direction/2, f'{accuracy:.1f}%', 
             ha='center', va='center', color='white', fontweight='bold')
    plt.show()

def main():
    # Fetch data
    print("Fetching SPY data...")
    df = fetch_data()
    
    # Prepare features
    print("Preparing features...")
    X, y = prepare_features(df)
    
    # Train model
    print("Training model...")
    model, scaler, X_test, y_test, feature_importance = train_model(X, y)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Get test dates
    test_dates = df.index[-len(y_test):]
    
    # Evaluate model
    metrics = evaluate_model(y_test, y_pred)
    
    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    print(f"Root Mean Square Error: ${metrics['RMSE']:.2f}")
    print(f"Mean Absolute Error: ${metrics['MAE']:.2f}")
    print(f"R-squared Score: {metrics['R2']:.4f}")
    print(f"Directional Accuracy: {metrics['Directional Accuracy']:.2f}%")
    
    # Visualize prediction accuracy
    visualize_prediction_accuracy(y_test, y_pred, test_dates)
    
    # Get latest data for prediction
    latest_data = X.iloc[-1:].copy()
    
    # Make prediction
    print("\nMaking prediction for next 30 minutes...")
    predicted_price = make_prediction(model, scaler, latest_data)
    
    # Print results
    current_price = df['Close'].iloc[-1]
    print(f"\nCurrent SPY Price: ${current_price:.2f}")
    print(f"Predicted SPY Price (30 minutes ahead): ${predicted_price:.2f}")
    print(f"Predicted Change: {((predicted_price - current_price) / current_price * 100):.2f}%")
    
    # Plot results
    plot_results(y_test, y_pred, feature_importance, df)


main() 