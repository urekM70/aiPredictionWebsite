import os
import pandas as pd
import datetime
from tasks.marketdata import get_historical_data  as get_historical_data_complex  # Assuming this is for simple mode
from tasks.simpleMarketData import get_historical_data as get_historical_data_simple # Assuming this is for complex mode

CACHE_DIR = 'data/'

# Define a variable for the mode: 'simple' or 'complex'
mode = 'simple'  # Change to 'complex' to switch to complex mode

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

def load_or_fetch_data(symbol, interval, start_date, end_date):
    """Load data from cache or fetch fresh data if needed."""
    # File path for cached data
    file_path = f'{CACHE_DIR}{symbol}_{interval}_with_indicators_normalized.csv'

    # If the data is fresh, load it from the cache
    if is_data_fresh(file_path, interval):
        print(f"Loading data from cache: {file_path}")
        return pd.read_csv(file_path)
    else:
        print(f"Data is stale or missing, fetching new data for {symbol}...")

        # Fetch new data based on the mode
        if mode == 'simple':
            df = get_historical_data_simple(symbol, interval, start_date, end_date)
        elif mode == 'complex':
            df = get_historical_data_complex(symbol, interval, start_date, end_date)
        else:
            raise ValueError("Invalid mode. Choose 'simple' or 'complex'.")

        return df
