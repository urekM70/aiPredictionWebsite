import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from tasks.loadData import load_or_fetch_data
import os
import logging
import json
import sqlite3
from app.db import get_db

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_features(df):
    """Create additional time-based and lag features"""
    df = df.copy()
    
    # Price change features
    df['price_change'] = df['close'].pct_change()
    df['price_change_ma5'] = df['price_change'].rolling(5).mean()
    df['price_volatility'] = df['price_change'].rolling(24).std()  # 24h volatility
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(24).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Time-based features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Lag features (previous values)
    for lag in [1, 2, 3, 6, 12, 24]:  # 1h, 2h, 3h, 6h, 12h, 24h ago
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    return df

def preprocess_data_optimized(df, sequence_length=24, target_steps=1):
    """
    Optimized preprocessing with better feature selection and scaling
    sequence_length: reduced from 60 to 24 for hourly data (1 day lookback)
    target_steps: how many hours ahead to predict
    """
    logger.info("Creating enhanced features...")
    df = create_features(df)
    
    # Core features - more selective
    core_features = [
        'open', 'high', 'low', 'close', 'volume',
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'RSI_14',
        'MACD_line', 'MACD_signal_line', 'ATR_14',
        'Bollinger_Bandwidth', 'Bollinger_Percent',
        'Stochastic_K', 'Stochastic_D'
    ]
    
    # Enhanced features
    enhanced_features = [
        'price_change', 'price_change_ma5', 'price_volatility',
        'volume_ratio', 'close_lag_1', 'close_lag_2', 'close_lag_3',
        'close_lag_6', 'close_lag_12', 'close_lag_24'
    ]
    
    # Time features (if available)
    time_features = []
    if 'hour' in df.columns:
        time_features = ['hour', 'day_of_week', 'is_weekend']
    
    feature_columns = core_features + enhanced_features + time_features
    
    # Remove features that don't exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    logger.info(f"Using {len(available_features)} features: {available_features}")
    
    # Remove rows with NaN values
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Data shape after removing NaN: {df.shape}")
    
    if len(df) < sequence_length + target_steps:
        logger.error(f"Not enough data. Need at least {sequence_length + target_steps} rows, got {len(df)}")
        return None, None, None, None
    
    # Use RobustScaler instead of MinMaxScaler - better for outliers
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(df[available_features])
    
    # Create sequences for LSTM-style input but flattened for XGBoost
    num_samples = len(scaled_data) - sequence_length - target_steps + 1
    X = np.array([scaled_data[i:i+sequence_length].flatten() for i in range(num_samples)])
    
    # Target is 'close' price target_steps ahead
    close_idx = available_features.index('close')
    y = scaled_data[sequence_length + target_steps - 1:, close_idx]
    
    logger.info(f"Data preprocessed: {X.shape[0]} sequences created, each with {X.shape[1]} features")
    return X, y, scaler, available_features

def create_time_series_splits(X, y, n_splits=5):
    """Create time series splits for proper validation"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    return list(tscv.split(X))

def train_optimized_model(X_train, y_train, X_val, y_val):
    """Train XGBoost with optimized hyperparameters for time series"""
    model = XGBRegressor(
        # Reduced complexity to prevent overfitting
        n_estimators=300,  # Reduced from 600
        learning_rate=0.05,  # Increased from 0.02
        max_depth=3,  # Reduced from 4
        subsample=0.8,  # Reduced from 0.9
        colsample_bytree=0.7,  # Reduced from 0.8
        min_child_weight=50,  # Increased from 20
        gamma=0.2,  # Increased from 0.1
        reg_lambda=2.0,  # Increased from 1.0
        alpha=0.5,  # Increased from 0.2
        objective='reg:squarederror',
        early_stopping_rounds=50,  # Increased from 30
        verbosity=0,  # Reduced verbosity
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

def smooth_predictions_advanced(predictions, window_size=5, method='ewm'):
    """Advanced smoothing with exponential weighted moving average"""
    if method == 'ewm':
        # Exponential weighted moving average
        alpha = 2.0 / (window_size + 1)
        smoothed = np.zeros_like(predictions)
        smoothed[0] = predictions[0]
        for i in range(1, len(predictions)):
            smoothed[i] = alpha * predictions[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    else:
        # Simple moving average
        return np.convolve(predictions, np.ones(window_size)/window_size, mode='same')

def evaluate_model(y_true, y_pred, scaler, feature_columns):
    """Evaluate model performance with proper inverse scaling"""
    # Create dummy array for inverse transform
    dummy_features = np.zeros((len(y_true), len(feature_columns)))
    close_idx = feature_columns.index('close')
    dummy_features[:, close_idx] = y_true
    y_true_rescaled = scaler.inverse_transform(dummy_features)[:, close_idx]
    
    dummy_features[:, close_idx] = y_pred
    y_pred_rescaled = scaler.inverse_transform(dummy_features)[:, close_idx]
    
    mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mse)
    
    # Calculate percentage error
    mape = np.mean(np.abs((y_true_rescaled - y_pred_rescaled) / y_true_rescaled)) * 100
    
    logger.info(f"Model Performance:")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    return y_true_rescaled, y_pred_rescaled, rmse, mae, mape

def run_crypto_prediction(symbol, interval='1h', start_date=None, end_date=None):
    logger.info("Starting optimized crypto price prediction process...")

    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=2*365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date} with {interval} interval...")
    data = load_or_fetch_data(symbol, interval)

    if data is None or data.empty:

        logger.error("No data returned. Ensure data exists or fetch it using Celery tasks.")
        return

    logger.info(f"Data loaded: {len(data)} rows")
    
    # Preprocess with optimized parameters
    result = preprocess_data_optimized(data, sequence_length=24, target_steps=1)
    if result[0] is None:
        logger.error("Preprocessing failed")
        return
    
    X, y, scaler, feature_columns = result

    # Use time series split for proper validation
    logger.info("Creating time series splits for validation...")
    splits = create_time_series_splits(X, y, n_splits=3)
    
    best_model = None
    best_score = float('inf')
    
    # Train and validate using time series splits
    for fold, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"Training fold {fold + 1}...")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = train_optimized_model(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        
        logger.info(f"Fold {fold + 1} Validation RMSE: {val_rmse:.6f}")
        
        if val_rmse < best_score:
            best_score = val_rmse
            best_model = model

    logger.info(f"Best model selected with RMSE: {best_score:.6f}")

    # Make predictions on the entire dataset for visualization
    logger.info("Making final predictions...")
    predictions = best_model.predict(X)
    
    # Evaluate model performance
    y_true_rescaled, y_pred_rescaled, rmse, mae, mape = evaluate_model(
        y, predictions, scaler, feature_columns
    )
    
    # Apply advanced smoothing
    smoothed_predictions = smooth_predictions_advanced(y_pred_rescaled, window_size=3, method='ewm')
    
    # Save to database
    logger.info("Saving predictions to database...")
    conn = None
    try:
        predictions_json = json.dumps(smoothed_predictions.tolist())
        actuals_json = json.dumps(y_true_rescaled.tolist())
        
        # Performance metrics
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'n_samples': len(y_true_rescaled)
        }
        metrics_json = json.dumps(metrics)
        
        utc_now = datetime.datetime.utcnow()

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (symbol, interval, predictions, actuals, timestamp, metrics)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol.replace("USDT",""), interval, predictions_json, actuals_json, utc_now, metrics_json))
        conn.commit()
        logger.info(f"Predictions for {symbol} ({interval}) saved successfully.")

    except Exception as e:
        logger.error(f"Database error while saving predictions for {symbol} ({interval}): {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

    # Save model
    model_path = f"tasks/models/{symbol}_{interval}_xgboost_model.json"
    if not os.path.exists('tasks/models'):
        os.makedirs('tasks/models')
    best_model.save_model(model_path)
    logger.info(f"Optimized XGBoost model saved to {model_path}")
    
    return {
        'predictions': smoothed_predictions,
        'actuals': y_true_rescaled,
        'metrics': metrics,
        'model': best_model
    }

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

    start_date = input("Enter start date (YYYY-MM-DD, default is 2 years ago): ")
    if start_date == "":
        start_date = None

    end_date = input("Enter end date (YYYY-MM-DD, default is today's date): ")
    if end_date == "":
        end_date = None

    result = run_crypto_prediction(symbol, interval, start_date, end_date)
    if result:
        print(f"Prediction complete! RMSE: {result['metrics']['rmse']:.4f}, MAPE: {result['metrics']['mape']:.2f}%")