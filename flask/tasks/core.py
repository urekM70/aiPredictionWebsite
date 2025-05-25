from celery import shared_task
from tasks.main_training import run_crypto_prediction
#from tasks.data_tasks import fetch_binance_ohlcv, fetch_yfinance_ohlcv


@shared_task(name='test_task')
def test_task(crypto):
    run_crypto_prediction(crypto)
    return "Task OK"

@shared_task(name='train_model_task')
def train_model_task(crypto):
    run_crypto_prediction(crypto)



#@shared_task(name='tasks.fetch_binance_data')
def fetch_binance_data(symbol, interval='1h', batch_size=1000):
    """Fetch OHLCV data from Binance and save it to the database."""
    return fetch_binance_ohlcv(symbol, interval, batch_size)

#@shared_task(name='tasks.fetch_yfinance_data')
def fetch_yfinance_data(symbol, period='1mo', interval='1d'):
    """Fetch OHLCV data from Yahoo Finance and save it to the database."""
    return fetch_yfinance_ohlcv(symbol, period, interval)
    
