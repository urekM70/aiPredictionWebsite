from celery import shared_task

@shared_task(name="test_task")
def test_task():
    print("Running test task")
    return "Hello from Celery!"
