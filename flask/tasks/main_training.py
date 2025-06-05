import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tasks.loadData import load_or_fetch_data
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_data(df, sequence_length=60):
    logger.info("Normalizing and preparing the data for training...")

    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
        'MACD_line', 'MACD_signal_line', 'ATR_14',
        'Bollinger_Lower', 'Bollinger_Middle', 'Bollinger_Upper',
        'Bollinger_Bandwidth', 'Bollinger_Percent',
        'Stochastic_K', 'Stochastic_D'
    ]

    df = df.dropna().reset_index(drop=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_columns])

    num_samples = len(scaled_data) - sequence_length
    X = np.array([scaled_data[i:i+sequence_length] for i in range(num_samples)])
    y = scaled_data[sequence_length:, feature_columns.index('close')]

    logger.info(f"Data preprocessed: {X.shape[0]} sequences created for model training.")
    return X, y, scaler

def smooth_predictions(predictions, window_size=5):
    return np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')

def run_crypto_prediction(symbol, interval='1h', start_date=None, end_date=None):
    logger.info("Starting the crypto price prediction process...")

    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} with {interval} interval...")
    data = load_or_fetch_data(symbol, interval)

    if data is None or data.empty:
        logger.error("No data returned. Ensure data exists or fetch it using Celery tasks.")
        return

    X, y, scaler = preprocess_data(data)

    logger.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logger.info(f"Data split: {len(X_train)} training sequences and {len(X_test)} testing sequences.")

    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    logger.info("Creating and training the XGBoost model...")
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=20,
        gamma=0.1,
        reg_lambda=1.0,
        alpha=0.2,
        objective='reg:squarederror',
        early_stopping_rounds=30,
        verbosity=1
    )

    model.fit(X_train_reshaped, y_train, eval_set=[(X_test_reshaped, y_test)], verbose=True)

    logger.info("Model training complete.")
    logger.info("Making predictions on the test data...")

    predictions = model.predict(X_test_reshaped)

    predictions_rescaled = scaler.inverse_transform(
        np.column_stack((np.zeros((predictions.shape[0], X.shape[2] - 1)), predictions))
    )[:, -1]

    actual_rescaled = scaler.inverse_transform(
        np.column_stack((np.zeros((y_test.shape[0], X.shape[2] - 1)), y_test))
    )[:, -1]

    smoothed_predictions = smooth_predictions(predictions_rescaled, window_size=5)
    time_indices = data.index[-len(y_test):]

    if len(smoothed_predictions) < len(time_indices):
        time_indices = time_indices[-len(smoothed_predictions):]

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

    model_path = f"tasks/models/{symbol}_{interval}_xgboost_model.json"
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save_model(model_path)
    logger.info(f"XGBoost model saved to {model_path}")

if __name__ == "__main__":
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

    start_date = input("Enter start date (YYYY-MM-DD, default is 5 years ago): ")
    if start_date == "":
        start_date = None

    end_date = input("Enter end date (YYYY-MM-DD, default is today's date): ")
    if end_date == "":
        end_date = None

    run_crypto_prediction(symbol, interval, start_date, end_date)
