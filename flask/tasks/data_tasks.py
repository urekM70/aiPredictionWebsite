import os
import logging
import time
from datetime import datetime, timedelta, timezone
from celery import shared_task
import yfinance as yf
from binance.client import Client
from binance.exceptions import BinanceAPIException

from app.celery_instance import celery_app


from app.db import get_db, insert_market_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default historical lookback periods
DEFAULT_BINANCE_HISTORICAL_DAYS = 3 * 365  # Approx 3 years
DEFAULT_YFINANCE_HISTORICAL_DAYS = 3 * 365 # Approx 3 years

# --- Helper Functions ---

def _get_latest_timestamp(db_conn, symbol: str) -> int | None:
    """
    Fetches the latest (maximum) timestamp for a given symbol from the market_data table.
    Returns Unix timestamp (seconds) or None if no record exists.
    """
    cursor = None
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT MAX(timestamp) FROM market_data WHERE symbol = ?", (symbol,))
        result = cursor.fetchone()
        return int(result[0]) if result and result[0] is not None else None
    except Exception as e:
        logger.error(f"Error getting latest timestamp for {symbol}: {e}")
        return None
    finally:
        if cursor:
            cursor.close()

def _interval_to_timedelta(interval_str: str) -> timedelta:
    """
    Converts a Binance/YFinance interval string to a timedelta object.
    Example: '1h' -> timedelta(hours=1), '1d' -> timedelta(days=1), '5m' -> timedelta(minutes=5)
    """
    multipliers = {'m': 'minutes', 'h': 'hours', 'd': 'days', 'w': 'weeks'}
    try:
        value = int(interval_str[:-1])
        unit = interval_str[-1].lower()
        if unit not in multipliers:
            raise ValueError(f"Unsupported interval unit: {unit}")
        return timedelta(**{multipliers[unit]: value})
    except Exception as e:
        logger.error(f"Failed to parse interval string '{interval_str}': {e}")
        # Default to 1 day if parsing fails, or raise error
        raise ValueError(f"Cannot parse interval string: {interval_str}") from e

def _binance_interval_to_milliseconds(interval_str: str) -> int:
    """Converts Binance interval string to milliseconds."""
    td = _interval_to_timedelta(interval_str)
    return int(td.total_seconds() * 1000)

# --- Celery Tasks ---

@shared_task(name='fetch_binance_data')
def fetch_binance_data(symbol: str, interval: str, batch_size: int = 1000):
    """
    Fetches historical OHLCV klines from Binance incrementally and stores them in the database.
    Fetches data in batches from the last recorded point or up to ~3 years back if no data exists.
    """
    logger.critical("TASK STARTED: fetch_binance_data for symbol %s, interval %s, batch_size %s", symbol, interval, batch_size)
    api_key = os.environ.get('BINANCE_API_KEY', '') # Provide default empty string if not set
    api_secret = os.environ.get('BINANCE_API_SECRET', '')
    client = Client(api_key, api_secret)

    db_conn = None
    total_inserted_count = 0

    try:
        logger.info("Attempting to get DB connection for Binance data for %s", symbol)
        db_conn = get_db()
        logger.info("DB connection object for Binance: %s (Is None: %s)", db_conn, db_conn is None)
        last_ts_seconds = _get_latest_timestamp(db_conn, symbol)
        
        interval_td = _interval_to_timedelta(interval)

        if last_ts_seconds is None:
            # No data for this symbol, fetch from default historical lookback
            start_dt = datetime.now(timezone.utc) - timedelta(days=DEFAULT_BINANCE_HISTORICAL_DAYS)
            logger.info(f"BINANCE: No existing data for {symbol}. Starting fetch from {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}.")
        else:
            # Start from the candle AFTER the last recorded one
            start_dt = datetime.fromtimestamp(last_ts_seconds, timezone.utc) + interval_td
            logger.info(f"BINANCE: Last record for {symbol} at {datetime.fromtimestamp(last_ts_seconds, timezone.utc)}. Resuming fetch from {start_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}.")

        end_dt = datetime.now(timezone.utc)

        if start_dt >= end_dt:
            logger.info(f"BINANCE: Data for {symbol} is already up to date. No new data to fetch.")
            return f"Data for {symbol} is up to date."

        current_fetch_start_dt = start_dt
        
        while current_fetch_start_dt < end_dt:
            # Calculate end of current batch: start + batch_size * interval
            # Binance's get_historical_klines uses start_str and end_str (optional)
            # If end_str is provided, it fetches up to that point, respecting limit.
            # We'll fetch in windows defined by start_dt and a calculated batch_end_dt
            
            batch_end_dt = current_fetch_start_dt + (batch_size * interval_td)
            if batch_end_dt > end_dt:
                batch_end_dt = end_dt
            
            # Convert datetimes to millisecond timestamp strings for Binance API
            start_timestamp_ms_str = str(int(current_fetch_start_dt.timestamp() * 1000))
            # For get_historical_klines, end_str is exclusive, so no need to subtract one interval
            end_timestamp_ms_str = str(int(batch_end_dt.timestamp() * 1000))

            logger.info(f"BINANCE: Fetching {symbol} from {current_fetch_start_dt.strftime('%Y-%m-%d %H:%M:%S')} to {batch_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                klines = client.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_str=start_timestamp_ms_str,
                    end_str=end_timestamp_ms_str, # Fetch up to this point
                    limit=batch_size # Ensure we don't exceed API limits per call
                )
            except BinanceAPIException as e:
                logger.error(f"BINANCE: API error for {symbol} ({current_fetch_start_dt} to {batch_end_dt}): {e}")
                # Depending on error (e.g. rate limits), could implement backoff/retry
                break # Stop fetching for this symbol on API error

            if not klines:
                logger.info(f"BINANCE: No new klines found for {symbol} in range {current_fetch_start_dt} to {batch_end_dt}. Loop might terminate.")
                # This can happen if current_fetch_start_dt is very recent and no new full candle exists
                # Or if batch_end_dt is before current_fetch_start_dt (should not happen with logic)
                break 

            processed_data = []
            last_kline_ts_ms = 0
            for kline in klines:
                timestamp_ms = int(kline[0])
                timestamp_s = timestamp_ms // 1000
                
                # Ensure we don't re-insert the very last point if it was already captured by previous run and start_dt logic
                if last_ts_seconds and timestamp_s <= last_ts_seconds:
                    continue

                processed_data.append({
                    'symbol': symbol,
                    'timestamp': timestamp_s,
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })
                last_kline_ts_ms = timestamp_ms

            if processed_data:
                logger.critical("Attempting to insert %s klines for %s (Binance)", len(processed_data), symbol)
                insert_market_data(db_conn, processed_data)
                logger.info("Called insert_market_data for Binance symbol %s", symbol)
                total_inserted_count += len(processed_data)
                logger.info(f"BINANCE: Inserted {len(processed_data)} records for {symbol}.")
            else:
                logger.warning("No klines to insert for %s (Binance batch)", symbol)
            
            if not last_kline_ts_ms: # No new data points were processed
                 logger.info(f"BINANCE: No new kline timestamps processed for {symbol} in this batch. Terminating to prevent infinite loop.")
                 break

            # Update start_dt for the next batch to be the timestamp of the last kline + 1 interval
            current_fetch_start_dt = datetime.fromtimestamp(last_kline_ts_ms / 1000, timezone.utc) + interval_td
            
            if current_fetch_start_dt >= end_dt:
                logger.info(f"BINANCE: Reached effective end_dt for {symbol}.")
                break

            time.sleep(0.5)  # Be polite to the API

        logger.info(f"BINANCE: Finished fetching for {symbol}. Total new records: {total_inserted_count}.")
        return f"Inserted {total_inserted_count} records for {symbol} from Binance."

    except ValueError as ve: # For interval parsing errors
        logger.error(f"BINANCE: Configuration error for {symbol}: {ve}")
        return f"Configuration error for {symbol}: {str(ve)}"
    except Exception as e:
        logger.error(f"BINANCE: General error fetching data for {symbol}: {e}", exc_info=True)
        return f"Failed to fetch Binance data for {symbol}: {str(e)}"
    finally:
        if db_conn:
            db_conn.close()
        logger.critical("TASK FINISHED: fetch_binance_data for symbol %s", symbol)



@shared_task(name='fetch_yfinance_data')
def fetch_yfinance_data(symbol: str, period_unused: str = '3y', interval: str = '1d'):
    """
    Fetches historical OHLCV data from Yahoo Finance incrementally and stores it.
    Fetches data from the last recorded point or up to ~3 years back if no data exists.
    `period_unused` is kept for compatibility with schedule but not directly used for start/end dates.
    """
    logger.critical("TASK STARTED: fetch_yfinance_data for symbol %s, interval %s", symbol, interval)
    db_conn = None
    total_inserted_count = 0

    try:
        logger.info("Attempting to get DB connection for YFinance data for %s", symbol)
        db_conn = get_db()
        logger.info("DB connection object for YFinance: %s (Is None: %s)", db_conn, db_conn is None)
        last_ts_seconds = _get_latest_timestamp(db_conn, symbol)
        
        # Determine start_date and end_date for yfinance
        # yfinance start_date is inclusive, end_date is exclusive
        end_dt_for_yfinance = datetime.now(timezone.utc) # Fetch up to today
        
        if last_ts_seconds is None:
            # No data, fetch from default historical lookback
            start_dt_for_yfinance = datetime.now(timezone.utc) - timedelta(days=DEFAULT_YFINANCE_HISTORICAL_DAYS)
            logger.info(f"YFINANCE: No existing data for {symbol}. Starting fetch from {start_dt_for_yfinance.strftime('%Y-%m-%d')}.")
        else:
            # Start from the day AFTER the last recorded timestamp to avoid overlap.
            # YFinance typically gives daily data where timestamp is start of day.
            # If interval is finer (e.g. '1h'), this logic might need adjustment or rely on yfinance to handle it.
            # For daily data, this is generally fine.
            start_dt_for_yfinance = datetime.fromtimestamp(last_ts_seconds, timezone.utc) + timedelta(days=1)
            logger.info(f"YFINANCE: Last record for {symbol} at {datetime.fromtimestamp(last_ts_seconds, timezone.utc)}. Resuming fetch from {start_dt_for_yfinance.strftime('%Y-%m-%d')}.")

        # Convert to YYYY-MM-DD string format for yfinance
        start_date_str = start_dt_for_yfinance.strftime('%Y-%m-%d')
        end_date_str = end_dt_for_yfinance.strftime('%Y-%m-%d') # yfinance end is exclusive

        if start_dt_for_yfinance.date() >= end_dt_for_yfinance.date(): # Compare dates to avoid issues with time part
            logger.info(f"YFINANCE: Data for {symbol} ({start_date_str}) is already up to date or ahead of end date ({end_date_str}). No new data to fetch.")
            return f"Data for {symbol} is up to date."

        logger.info(f"YFINANCE: Fetching {symbol} from {start_date_str} to {end_date_str} with interval {interval}.")
        
        ticker = yf.Ticker(symbol)
        hist_data = ticker.history(start=start_date_str, end=end_date_str, interval=interval)

        if hist_data.empty:
            logger.info(f"YFINANCE: No new data returned from yfinance for {symbol} for the period.")
            return f"No new data for {symbol} from yfinance for this period."

        processed_data = []
        for timestamp_dt, row in hist_data.iterrows():
            # Convert pandas Timestamp (DatetimeIndex) to Unix seconds (integer)
            # Ensure timestamp is converted to UTC before taking timestamp() if it's timezone-naive
            if timestamp_dt.tzinfo is None:
                timestamp_dt = timestamp_dt.tz_localize('UTC') # Assume UTC if naive, or use appropriate timezone from yfinance
            else:
                timestamp_dt = timestamp_dt.tz_convert('UTC')
            
            timestamp_s = int(timestamp_dt.timestamp())

            # Prevent re-inserting data if last_ts_seconds exists and new data point is not newer
            if last_ts_seconds and timestamp_s <= last_ts_seconds:
                continue
            
            # Check for NaN values which can occur in yfinance data
            if row[['Open', 'High', 'Low', 'Close', 'Volume']].isnull().any():
                logger.warning(f"YFINANCE: Skipping row for {symbol} at {timestamp_dt} due to NaN values.")
                continue

            processed_data.append({
                'symbol': symbol,
                'timestamp': timestamp_s,
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume'])
            })
        
        if processed_data:
            logger.critical("Attempting to insert %s rows for %s (YFinance)", len(processed_data), symbol)
            insert_market_data(db_conn, processed_data)
            logger.info("Called insert_market_data for YFinance symbol %s", symbol)
            total_inserted_count = len(processed_data)
            logger.info(f"YFINANCE: Successfully inserted {total_inserted_count} records for {symbol} from yfinance.")
        else:
            logger.warning("No data to insert for %s (YFinance)", symbol)
            logger.info(f"YFINANCE: No new, valid data points to insert for {symbol} after processing.")

        return f"Inserted {total_inserted_count} records for {symbol} from yfinance."

    except ValueError as ve: # For interval parsing errors (if _interval_to_timedelta were used more here)
        logger.error(f"YFINANCE: Configuration error for {symbol}: {ve}")
        return f"Configuration error for yfinance {symbol}: {str(ve)}"
    except Exception as e:
        logger.error(f"YFINANCE: Error fetching or processing yfinance data for {symbol}: {e}", exc_info=True)
        return f"Failed to fetch yfinance data for {symbol}: {str(e)}"
    finally:
        if db_conn:
            db_conn.close()
        logger.critical("TASK FINISHED: fetch_yfinance_data for symbol %s", symbol)

if __name__ == '__main__':
    # Local testing examples (requires DB setup and Celery broker for .delay())
    # To test directly without Celery:
    # from flask.app.db import init_db
    # conn = get_db()
    # init_db() # Ensure tables are created
    # conn.close()
    
    # Example direct calls:
    # print(fetch_binance_data("BTCUSDT", "1h", batch_size=100))
    # print(fetch_yfinance_data("MSFT", interval="1d"))
    pass
