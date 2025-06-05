from celery import shared_task
from tasks.main_training import run_crypto_prediction
from tasks.data_tasks import fetch_binance_data, fetch_yfinance_data
import logging
import os
from tasks.data_tasks import upToDateChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@shared_task(name='test_task')
def test_task(crypto):
    run_crypto_prediction(crypto)
    return "Task OK"


def check_data_exists(symbol: str, interval: str = '1h') -> bool:
    base_name = symbol.upper().replace('USDT', '')
    data_dir = 'marketdata'  # or whatever folder you save CSVs in
    filename = f"{base_name}_{interval}_essential.csv"
    path = os.path.join(data_dir, filename)
    return os.path.exists(path) and os.path.getsize(path) > 30


@shared_task(name='train_model_task')
def train_model_task(symbol: str):
    if upToDateChecker(symbol):
        if 'USDT' not in symbol.upper():
            run_crypto_prediction(symbol,"1d")
            return f"Prediction ran for {symbol}"
        run_crypto_prediction(symbol,"1h")
        return f"Prediction ran for {symbol}"

    # Schedule the appropriate fetch task
    if 'USDT' in symbol.upper():
        fetch_binance_data.delay(symbol,"1h")
        delay = 30
    else:
        fetch_yfinance_data.delay(symbol,"1d")
        delay = 15
    
    train_model_task.apply_async((symbol,), countdown=delay)

    return f"Fetch scheduled for {symbol}, prediction will run afterward."
