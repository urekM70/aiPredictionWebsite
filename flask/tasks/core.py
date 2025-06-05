from celery import shared_task
from tasks.main_training import run_crypto_prediction as run_prediction
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@shared_task(name='test_task')
def test_task(crypto):
    run_crypto_prediction(crypto)
    return "Task OK"


def check_data_exists(symbol: str, interval: str = '1h') -> bool:
    base_name = symbol.upper().replace('USDT', '')
    data_dir = 'marketdata'
    filename = f"{base_name}_{interval}_essential.csv"
    path = os.path.join(data_dir, filename)
    return os.path.exists(path) and os.path.getsize(path) > 1000


def stockOrCrypto(symbol: str) -> str:
    cryptos = ["BTC", "ETH", "SOL", "ADA", "XRP"]
    return 'crypto' if symbol in cryptos else 'stock'


@shared_task(name='train_model_task')
def train_model_task(symbol: str):
    if stockOrCrypto(symbol) == 'crypto':
        run_prediction(symbol+"USDT")
    else:
        run_prediction(symbol,"1d")
