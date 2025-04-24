import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from loadData import load_or_fetch_data  # Import the function from your previous script

def preprocess_data(df, sequence_length=60):
    """Preprocess the data for training."""
    print("Normalizing and preparing the data for training...")

    # Define feature columns (adjust based on your data)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal']
    
    df = df.dropna().reset_index(drop=True)  # Drop NaN values

    # Apply MinMax Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_columns])

    num_samples = len(scaled_data) - sequence_length
    X = np.array([scaled_data[i:i+sequence_length] for i in range(num_samples)])
    y = scaled_data[sequence_length:, feature_columns.index('Close')]  # Predicting 'Close'

    print(f"Data preprocessed: {X.shape[0]} sequences created for model training.")
    return X, y, scaler

def smooth_predictions(predictions, window_size=5):
    """Smooth the predictions with a moving average."""
    return np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')

def main():
    print("Starting the crypto price prediction process...")

    symbol = input("Enter the crypto symbol (e.g., BTC): ").upper()
    print("Select timeframe:")
    print("1: 1 Minute")
    print("2: 5 Minutes")
    print("3: 15 Minutes")
    print("4: 1 Hour")
    print("5: 1 Day")
    interval_map = {'1': '1m', '2': '5m', '3': '15m', '4': '1h', '5': '1d'}
    interval_choice = input("Enter choice (1-5): ")
    interval = interval_map.get(interval_choice, '1h')
    
    start_date = input("Enter start date (YYYY-MM-DD): ")
    
    use_today = input("Use today's date as the end date? (yes/no): ").strip().lower()
    if use_today == 'yes':
        end_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        end_date = input("Enter end date (YYYY-MM-DD): ")

    print(f"Fetching data for {symbol} from {start_date} to {end_date} with {interval} interval...")

    # Use load_or_fetch_data to load data
    data = load_or_fetch_data(symbol, interval, start_date, end_date)
    
    # Preprocess data
    X, y, scaler = preprocess_data(data)

    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: {len(X_train)} training sequences and {len(X_test)} testing sequences.")

    # Reshape X_train and X_test to be 2D (samples, features)
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)  # Flattening into 2D
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)  # Flattening into 2D

    print("Creating and training the XGBoost model...")
    model = XGBRegressor(
        n_estimators=1000,        # Number of trees
        learning_rate=0.01,       # Lower learning rate
        max_depth=6,              # Depth of trees
        subsample=0.8,            # Use 80% of the data for each tree
        colsample_bytree=0.8,     # Randomly sample 80% of features for each tree
        alpha=0.1,                # L1 regularization
        reg_lambda=0.1,           # Use 'reg_lambda' instead of 'lambda_'
        early_stopping_rounds=50  # Stop early if no improvement
    )

    model.fit(X_train_reshaped, y_train, eval_set=[(X_test_reshaped, y_test)], verbose=True)

    print("Model training complete.")

    print("Making predictions on the test data...")
    predictions = model.predict(X_test_reshaped)

    # Rescale the predictions correctly
    # Rescale the predictions and actual values (match columns with 'Close' column)
    predictions_rescaled = scaler.inverse_transform(
        np.column_stack((np.zeros((predictions.shape[0], X.shape[2] - 1)), predictions))
    )[:, -1]  # Get the 'Close' column predictions

    # Rescale actual values
    actual_rescaled = scaler.inverse_transform(
        np.column_stack((np.zeros((y_test.shape[0], X.shape[2] - 1)), y_test))
    )[:, -1]  # Get the 'Close' column actual values

    # Smooth the predictions to reduce fluctuations
    smoothed_predictions = smooth_predictions(predictions_rescaled, window_size=5)

    # Get the corresponding time indices for plotting
    time_indices = data.index[-len(y_test):]  # Match test set length

    # Ensure correct alignment
    if len(smoothed_predictions) < len(time_indices):
        time_indices = time_indices[-len(smoothed_predictions):]

    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(time_indices, actual_rescaled[-len(time_indices):], label="Actual Prices", color="blue", linewidth=2)
    plt.plot(time_indices, smoothed_predictions, label="Predicted Prices", color="red", linestyle="dashed", linewidth=2)
    plt.title(f"Real Prices vs Predicted Prices for {symbol} ({interval} interval)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"Saving the XGBoost model to {symbol}_{interval}_xgboost_model.model...")
    model.save_model(f"models/{symbol}_{interval}_xgboost_model.model")
    print(f"XGBoost Model saved to {symbol}_{interval}_xgboost_model.model")

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    main()
