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

# Setup logger
logger = logging.getLogger("data_fetcher")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Constants
MARKETDATA_DIR = "tasks/marketdata/"
CSV_HEADER = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
DEFAULT_BINANCE_HISTORICAL_DAYS = 3 * 365
DEFAULT_YFINANCE_HISTORICAL_DAYS = 2 * 365


def generate_csv_path(symbol: str, interval: str, directory: str = MARKETDATA_DIR) -> str:
    clean_symbol = symbol.replace("USDT", "")
    return os.path.join(directory, f"{clean_symbol}_{interval}_essential.csv")


def _get_latest_timestamp_csv(symbol: str, marketdata_dir: str = MARKETDATA_DIR) -> int | None:
    csv_filepath = generate_csv_path(symbol, "1h", marketdata_dir)
    if not os.path.exists(csv_filepath) or os.path.getsize(csv_filepath) == 0:
        return None
    try:
        df = pd.read_csv(csv_filepath)
        return int(df['timestamp'].max()) if 'timestamp' in df.columns and not df['timestamp'].empty else None
    except pd.errors.EmptyDataError:
        logger.info(f"CSV file for {symbol} is empty. No latest timestamp.")
    except Exception as e:
        logger.error(f"Error reading CSV for {symbol}: {e}")
    return None


def _interval_to_timedelta(interval_str: str) -> timedelta:
    multipliers = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
    try:
        value = int(interval_str[:-1])
        unit = interval_str[-1].lower()
        return timedelta(**{multipliers[unit]: value})
    except Exception as e:
        logger.error(f"Invalid interval format: {interval_str} — {e}")
        raise

def upToDateChecker(symbol: str, interval: str = "1h") -> bool:
    """
    Check if the data for the given symbol and interval is up to date.
    Returns True if data is fresh, False otherwise.
    """
    csv_filepath = generate_csv_path(symbol, interval)
    if not os.path.exists(csv_filepath) or os.path.getsize(csv_filepath) == 0:
        return False

    try:
        if "USDT" not in symbol.upper():
            return True
        df = pd.read_csv(csv_filepath)
        if 'timestamp' not in df.columns or df['timestamp'].empty:
            return False
        latest_timestamp = int(df['timestamp'].max())
        current_timestamp = int(datetime.now(timezone.utc).timestamp())
        freshness_threshold = 60 * 60 # 1 hour in seconds
        return (current_timestamp - latest_timestamp) < freshness_threshold
    except Exception as e:
        logger.error(f"Error checking data freshness for {symbol} at {interval}: {e}")
        return False

@shared_task(name='fetch_binance_data')
def fetch_binance_data(symbol: str, interval: str, batch_size: int = 1000):
    logger.critical(f"STARTING: fetch_binance_data | Symbol: {symbol} | Interval: {interval}")
    api_key = os.environ.get('BINANCE_API_KEY', '')
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    client = Client(api_key, api_secret)

    os.makedirs(MARKETDATA_DIR, exist_ok=True)
    csv_filepath = generate_csv_path(symbol, interval)
    total_written_count = 0

    try:
        last_ts = _get_latest_timestamp_csv(symbol)
        interval_td = _interval_to_timedelta(interval)

        start_dt = datetime.now(timezone.utc) - timedelta(days=DEFAULT_BINANCE_HISTORICAL_DAYS) if last_ts is None else datetime.fromtimestamp(last_ts, timezone.utc) + interval_td
        end_dt = datetime.now(timezone.utc)

        if start_dt >= end_dt:
            logger.info(f"Data for {symbol} is already up to date. No fetching needed.")
            return

        current_dt = start_dt
        while current_dt < end_dt:
            batch_end = min(current_dt + (batch_size * interval_td), end_dt)
            start_ms, end_ms = int(current_dt.timestamp() * 1000), int(batch_end.timestamp() * 1000)

            try:
                klines = client.get_historical_klines(symbol, interval, start_str=str(start_ms), end_str=str(end_ms), limit=batch_size)
            except BinanceAPIException as e:
                logger.error(f"Binance API Error: {e}")
                break

            processed = []
            last_kline_ts_ms = 0
            for k in klines:
                ts_s = int(k[0] // 1000)
                if last_ts and ts_s <= last_ts:
                    continue
                processed.append({
                    'timestamp': ts_s,
                    'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]),
                    'close': float(k[4]), 'volume': float(k[5])
                })
                last_kline_ts_ms = int(k[0])

            if processed:
                df = pd.DataFrame(processed)
                if last_ts:
                    df = df[df['timestamp'] > last_ts]
                if not df.empty:
                    file_exists = os.path.exists(csv_filepath)
                    is_empty = file_exists and os.path.getsize(csv_filepath) == 0
                    df.to_csv(csv_filepath, mode='a', header=not file_exists or is_empty, index=False, columns=CSV_HEADER)
                    total_written_count += len(df)
            else:
                logger.info(f"No new records to write for {symbol} at interval {interval}.")
                break

            if not last_kline_ts_ms:
                break

            current_dt = datetime.fromtimestamp(last_kline_ts_ms / 1000, timezone.utc) + interval_td
            time.sleep(0.5)

        logger.info(f"COMPLETED: {total_written_count} records written for {symbol} at interval {interval}.")
    except Exception as e:
        logger.exception(f"Exception occurred during Binance data fetch for {symbol}: {e}")


@shared_task(name='fetch_yfinance_data')
def fetch_yfinance_data(symbol: str, period_unused: str = '3y', interval: str = '1d'):
    logger.critical(f"STARTING: fetch_yfinance_data | Symbol: {symbol} | Interval: {interval}")
    os.makedirs(MARKETDATA_DIR, exist_ok=True)
    csv_filepath = generate_csv_path(symbol, interval)
    total_written_count = 0

    try:
        last_ts = _get_latest_timestamp_csv(symbol)
        now = datetime.now(timezone.utc)
        interval_td = _interval_to_timedelta(interval)

        start_dt = now - timedelta(days=DEFAULT_YFINANCE_HISTORICAL_DAYS) if last_ts is None else datetime.fromtimestamp(last_ts, timezone.utc) + interval_td
        end_dt = now

        if start_dt >= end_dt:
            logger.info(f"Data for {symbol} is already up to date. No fetching needed.")
            return

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_dt, end=end_dt, interval=interval)

        if df.empty:
            logger.warning(f"No data returned from YFinance for {symbol}.")
            return

        rows = []
        for ts, row in df.iterrows():
            if row[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any():
                continue
            ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
            ts_s = int(ts.timestamp())
            if last_ts and ts_s <= last_ts:
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

        logger.info(f"COMPLETED: {total_written_count} records written for {symbol} using YFinance.")
    except Exception as e:
        logger.exception(f"Exception occurred during YFinance data fetch for {symbol}: {e}")