from celery import shared_task
from tasks.main_training import run_crypto_prediction

@shared_task(name='test_task')
def test_task():
    run_crypto_prediction(crypto)
    return "Task OK"

@shared_task(name='train_model_task')
def train_model_task(crypto):
    run_crypto_prediction(crypto)