# Module: flask.tasks.loadData
# Description:
# This module is responsible for loading market data, which includes Open, High, Low, Close, Volume (OHLCV)
# and technical indicators. It relies on base OHLCV data being available in CSV files
# (e.g., data/SYMBOL_INTERVAL_essential.csv). These essential CSV files are expected to be
# generated and kept up-to-date by Celery background tasks defined in flask.tasks.data_tasks.
#
# This module then applies a set of technical indicators to this base data based on the global 'mode'
# setting ('simple' or 'complex'). The resulting DataFrames, rich with indicators, are then cached
# in CSV files (e.g., data/SYMBOL_INTERVAL_ASSET-TYPE_MODE_indicators.csv) to speed up subsequent loads.

import os
import pandas as pd
import datetime
import pandas_ta as ta
from tasks.data_tasks import fetch_binance_data, fetch_yfinance_data
import tasks.core
import celery

CACHE_DIR =  "tasks/marketdata/"  # Directory where cached data will be stored

# Define a variable for the mode: 'simple' or 'complex'
mode = 'complex'  # Change to 'complex' to switch to complex mode

def get_freshness_threshold(interval):
    """Return the freshness threshold (in minutes) based on the interval."""
    if interval in ['1m', '5m', '15m']:
        return 30  # For minute-based data, we consider it fresh for 30 minutes
    elif interval == '1h':
        return 60*48  # For hourly data, consider it fresh for 3 hours
    elif interval == '1d':
        return 1440*5  # For daily data, consider it fresh for 1 day
    else:
        return 30  # Default, in case of any other interval

def is_data_fresh(file_path, interval):
    """Check if the cached data is fresh enough based on the interval."""
    if not os.path.exists(file_path):
        return False  # Data doesn't exist, so it's not fresh
    
    file_mod_time = os.path.getmtime(file_path)
    file_mod_datetime = datetime.datetime.fromtimestamp(file_mod_time)
    time_diff = datetime.datetime.now() - file_mod_datetime
    
    freshness_threshold = get_freshness_threshold(interval)
    
    # If the data is older than the freshness threshold, consider it stale
    return time_diff < datetime.timedelta(minutes=freshness_threshold)

# Function: load_or_fetch_data
# Description:
# Loads market data with technical indicators. It first checks a cache for data with indicators.
# If not found or stale, it loads the base OHLCV data (expected to be pre-fetched by data_tasks.py),
# then applies technical indicators according to the set 'mode', and finally caches this new
# DataFrame with indicators before returning it.
def load_or_fetch_data(symbol, interval):
    """Load data from cache or fetch fresh data if needed."""
    # Asset Type Detection
    asset_type = 'crypto' if 'USDT' in symbol.upper() else 'stock'
    file_symbol = symbol.upper().replace('USDT', '')

    # File Path Construction
    base_data_path = f"{CACHE_DIR}{file_symbol}_{interval}_essential.csv"
    indicator_cache_path = f"{CACHE_DIR}{file_symbol}_{interval}_{asset_type}_{mode}_indicators.csv"

    print(f"Checking cache for indicator data: {indicator_cache_path}")
    if is_data_fresh(indicator_cache_path, interval):
        print(f"Loading indicator data from cache: {indicator_cache_path}")
        df = pd.read_csv(indicator_cache_path)
        # Ensure the first column (timestamp or Open Time) is converted to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif 'Open Time' in df.columns: # Fallback for older formats if any
            df['Open Time'] = pd.to_datetime(df['Open Time'])
        return df
    else:
        print(f"Indicator data stale or missing for {indicator_cache_path}")
        print(f"Loading base data from: {base_data_path}")

        if not os.path.exists(base_data_path):
            print(f"Error: Base data file not found: {base_data_path}. Ensure data_tasks.py has run for this symbol and interval.")
            if asset_type == 'crypto':
                fetch_binance_data.delay(symbol, interval)
            elif asset_type == 'stock':
                fetch_yfinance_data.delay(symbol, interval)
            tasks.core.train_model_task.apply_async((symbol,),countdown=20)  # Trigger model training task
            return None # Or raise an exception

        base_df = pd.read_csv(base_data_path)
        
        # Convert 'timestamp' column from Unix seconds to datetime
        if 'timestamp' in base_df.columns:
            base_df['timestamp'] = pd.to_datetime(base_df['timestamp'], unit='s')
        # No specific renaming to 'Open Time' for now, assuming pandas_ta and downstream use 'timestamp' or direct column names.

        print(f"Applying technical indicators (mode: {mode})...")
        if mode == 'simple':
            df_with_indicators = _apply_simple_indicators(base_df.copy())
        elif mode == 'complex':
            df_with_indicators = _apply_complex_indicators(base_df.copy())
        else:
            # This case should ideally not be reached if 'mode' is validated at startup or before this call
            raise ValueError("Invalid mode. Choose 'simple' or 'complex'.")

        print(f"Saving data with indicators to: {indicator_cache_path}")
        df_with_indicators.to_csv(indicator_cache_path, index=False)
        return df_with_indicators

def _apply_simple_indicators(df):
    """
    Applies a set of simple technical indicators to the DataFrame.
    """
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['EMA_12'] = ta.ema(df['close'], length=12)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    
    # Calculate MACD
    macd = ta.macd(df['close'])
    df['MACD_line'] = macd['MACD_12_26_9']
    df['MACD_signal_line'] = macd['MACDs_12_26_9']
    
    # Drop rows with NaN values generated by indicators (especially MACD)
    df = df.iloc[50:]
    
    # Select and return the DataFrame with original and new indicator columns
    # Ensuring all original columns are kept, plus the new ones.
    # The specific columns to keep from simpleMarketData.py were:
    # ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 'MACD_signal']
    # Adapting to lowercase and new indicator names.
    # For now, ensure all original columns are kept, plus the new ones.
    # Example: keep_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close', 'volume']] + ['SMA_20', 'EMA_12', 'RSI_14', 'MACD_line', 'MACD_signal_line']
    # df = df[keep_cols] 
    # For now, returning all columns as per instructions, final column selection can be refined later.
    return df

def _apply_complex_indicators(df):
    """
    Applies a comprehensive set of technical indicators to the DataFrame.
    """
    df['SMA_20'] = ta.sma(df['close'], length=20)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['EMA_12'] = ta.ema(df['close'], length=12)
    df['EMA_26'] = ta.ema(df['close'], length=26)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'])
    df['MACD_line'] = macd['MACD_12_26_9']
    df['MACD_signal_line'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    
    # ATR
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=5, std=2)
    df['Bollinger_Lower'] = bbands['BBL_5_2.0']
    df['Bollinger_Middle'] = bbands['BBM_5_2.0']
    df['Bollinger_Upper'] = bbands['BBU_5_2.0']
    df['Bollinger_Bandwidth'] = bbands['BBB_5_2.0']
    df['Bollinger_Percent'] = bbands['BBP_5_2.0']
    
    # Stochastic Oscillator
    stoch = ta.stoch(df['high'], df['low'], df['close']) # Using default k=14, d=3, smooth_k=3
    df['Stochastic_K'] = stoch['STOCHk_14_3_3']
    df['Stochastic_D'] = stoch['STOCHd_14_3_3']
    
    # Drop rows with NaN values
    df = df.iloc[50:]
    
    return df
