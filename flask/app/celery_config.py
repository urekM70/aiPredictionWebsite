from celery import Celery
from celery.schedules import crontab
celery_app = Celery('flask_app')


def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery



# Celery Beat Schedule
# This will configure Celery Beat to run tasks at specified intervals.
# Make sure your Celery Beat service is running for these schedules to take effect.

celery_app.conf.beat_schedule = {
    'fetch-btcusdt-1h': {
        'task': 'fetch_binance_data', # Short name
        'schedule': crontab(minute=0, hour='*/1'),  # Every hour at minute 0
        'args': ('BTCUSDT', '1h', 1000),             # Arguments for the task: symbol, interval, batch_size
    },
    'fetch-ethusdt-1h': {
        'task': 'fetch_binance_data', # Short name
        'schedule': crontab(minute=5, hour='*/1'),  # Every hour at minute 5
        'args': ('ETHUSDT', '1h', 1000),             # Arguments for the task: symbol, interval, batch_size
    },
    'fetch-aapl-hourly': { 
        'task': 'fetch_yfinance_data', 
        'schedule': crontab(minute=10, hour='*/1'),  # Changed to hourly
        'args': ('AAPL', '1mo', '1d'),
    },
    'fetch-googl-hourly': { # 
        'task': 'fetch_yfinance_data',
        'schedule': crontab(minute=15, hour='*/1'), # Changed to hourly
        'args': ('GOOGL', '1mo', '1d'),
    },
}
# Set timezone for Celery (important for crontab schedules)
celery_app.conf.timezone = 'UTC'