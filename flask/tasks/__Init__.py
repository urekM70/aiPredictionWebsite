from app.celery_config import make_celery
celery = make_celery(app)