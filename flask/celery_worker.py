from app import create_app
from app.celery_instance import celery_app
import tasks.data_tasks
import tasks.core 
import celery

if __name__ == '__main__':
    celery.worker_main()
