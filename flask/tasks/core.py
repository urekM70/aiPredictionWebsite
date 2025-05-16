from celery import shared_task

@shared_task(name='test_task')
def test_task():
    return "Task OK"

@shared_task(name='train_model_task')
def train_model_task(crypto):
    ...