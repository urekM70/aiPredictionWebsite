from app import celery
import tasks.core  # to registrira vse @shared_task

if __name__ == '__main__':
    celery.worker_main()
