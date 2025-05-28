import os
import logging
import time
from datetime import datetime, timedelta, timezone
from celery import shared_task
import yfinance as yf
from binance.client import Client
from binance.exceptions import BinanceAPIException
import pandas as pd

from app.celery_instance import celery_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define market data directory
MARKETDATA_DIR = '/data'
CSV_HEADER = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def generate_csv_path(symbol: str, interval: str, directory: str = MARKETDATA_DIR) -> str:
    clean_symbol = symbol.replace("USDT", "")
    filename = f"{clean_symbol}_{interval}_essential.csv"
    return os.path.join(directory, filename)

# Default historical lookback periods
DEFAULT_BINANCE_HISTORICAL_DAYS = 3 * 365
DEFAULT_YFINANCE_HISTORICAL_DAYS = 3 * 365

# --- Helper Functions ---
def _get_latest_timestamp_csv(symbol: str, marketdata_dir: str = MARKETDATA_DIR) -> int | None:
    csv_filepath = generate_csv_path(symbol, "1h", marketdata_dir)
    if not os.path.exists(csv_filepath) or os.path.getsize(csv_filepath) == 0:
        return None
    try:
        df = pd.read_csv(csv_filepath)
        if 'timestamp' in df.columns and not df['timestamp'].empty:
            return int(df['timestamp'].max())
        return None
    except pd.errors.EmptyDataError:
        logger.info(f"CSV file for {symbol} is empty. No latest timestamp.")
        return None
    except Exception as e:
        logger.error(f"Error getting latest timestamp from CSV for {symbol}: {e}")
        return None

def _interval_to_timedelta(interval_str: str) -> timedelta:
    multipliers = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
    value = int(interval_str[:-1])
    unit = interval_str[-1].lower()
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval unit: {unit}")
    return timedelta(**{multipliers[unit]: value})

def _binance_interval_to_milliseconds(interval_str: str) -> int:
    return int(_interval_to_timedelta(interval_str).total_seconds() * 1000)

# --- Celery Tasks ---
@shared_task(name='fetch_binance_data')
def fetch_binance_data(symbol: str, interval: str, batch_size: int = 1000):
    logger.critical("TASK STARTED: fetch_binance_data for symbol %s (CSV)", symbol)
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    client = Client(api_key, api_secret)

    os.makedirs(MARKETDATA_DIR, exist_ok=True)
    csv_filepath = generate_csv_path(symbol, interval)
    total_written_count = 0

    try:
        last_csv_ts_seconds = _get_latest_timestamp_csv(symbol)
        interval_td = _interval_to_timedelta(interval)

        start_dt = datetime.now(timezone.utc) - timedelta(days=DEFAULT_BINANCE_HISTORICAL_DAYS) if last_csv_ts_seconds is None else datetime.fromtimestamp(last_csv_ts_seconds, timezone.utc) + interval_td
        end_dt = datetime.now(timezone.utc)

        if start_dt >= end_dt:
            return f"Data for {symbol} is up to date (CSV)."

        current_fetch_start_dt = start_dt
        while current_fetch_start_dt < end_dt:
            batch_end_dt = min(current_fetch_start_dt + (batch_size * interval_td), end_dt)
            start_ms, end_ms = int(current_fetch_start_dt.timestamp() * 1000), int(batch_end_dt.timestamp() * 1000)

            try:
                klines = client.get_historical_klines(symbol, interval, start_str=str(start_ms), end_str=str(end_ms), limit=batch_size)
            except BinanceAPIException as e:
                logger.error(f"BINANCE ERROR: {e}")
                break

            processed_data, last_kline_ts_ms = [], 0
            for k in klines:
                ts_s = int(k[0] // 1000)
                if last_csv_ts_seconds and ts_s <= last_csv_ts_seconds:
                    continue
                processed_data.append({
                    'timestamp': ts_s,
                    'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]),
                    'close': float(k[4]), 'volume': float(k[5])
                })
                last_kline_ts_ms = int(k[0])

            if processed_data:
                df = pd.DataFrame(processed_data)
                df = df[df['timestamp'] > last_csv_ts_seconds] if last_csv_ts_seconds else df
                if not df.empty:
                    file_exists = os.path.exists(csv_filepath)
                    is_empty = file_exists and os.path.getsize(csv_filepath) == 0
                    df.to_csv(csv_filepath, mode='a', header=not file_exists or is_empty, index=False, columns=CSV_HEADER)
                    total_written_count += len(df)
            else:
                break

            if not last_kline_ts_ms:
                break

            current_fetch_start_dt = datetime.fromtimestamp(last_kline_ts_ms / 1000, timezone.utc) + interval_td
            time.sleep(0.5)

        return f"Wrote {total_written_count} records to CSV for {symbol}"
    except Exception as e:
        logger.error(f"BINANCE CSV: {e}", exc_info=True)
        return f"Failed: {e}"

@shared_task(name='fetch_yfinance_data')
def fetch_yfinance_data(symbol: str, period_unused: str = '3y', interval: str = '1d'):
    logger.critical("TASK STARTED: fetch_yfinance_data for symbol %s (CSV)", symbol)
    os.makedirs(MARKETDATA_DIR, exist_ok=True)
    csv_filepath = generate_csv_path(symbol, interval)
    total_written_count = 0

    try:
        last_csv_ts_seconds = _get_latest_timestamp_csv(symbol)
        now = datetime.now(timezone.utc)

        interval_td = _interval_to_timedelta(interval)
        if last_csv_ts_seconds is None:
            start_dt = now - timedelta(days=DEFAULT_YFINANCE_HISTORICAL_DAYS)
        else:
            start_dt = datetime.fromtimestamp(last_csv_ts_seconds, timezone.utc) + interval_td

        end_dt = now
        if start_dt >= end_dt:
            return f"Data for {symbol} is up to date."

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_dt, end=end_dt, interval=interval)
        if df.empty:
            return f"No new data for {symbol}"

        rows = []
        for ts, row in df.iterrows():
            if row[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any():
                continue
            ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
            ts_s = int(ts.timestamp())
            if last_csv_ts_seconds and ts_s <= last_csv_ts_seconds:
                continue
            rows.append({
                'timestamp': ts_s,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume'])
            })

        if rows:
            df_out = pd.DataFrame(rows)
            file_exists = os.path.exists(csv_filepath)
            is_empty = file_exists and os.path.getsize(csv_filepath) == 0
            df_out.to_csv(csv_filepath, mode='a', header=not file_exists or is_empty, index=False, columns=CSV_HEADER)
            total_written_count += len(df_out)

        return f"Wrote {total_written_count} records to CSV for {symbol}"
    except Exception as e:
        logger.error(f"YFINANCE CSV: {e}", exc_info=True)
        return f"Failed to fetch YFinance data: {str(e)}"
