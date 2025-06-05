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

from celery.schedules import crontab

celery_app.conf.beat_schedule = {

    'fetch-btcusdt-1h': {
        'task': 'fetch_binance_data',
        'schedule': crontab(minute=0, hour='*/1'),
        'args': ('BTCUSDT', '1h', 1000),
    },
    'fetch-aapl-hourly': {
        'task': 'fetch_yfinance_data',
        'schedule': crontab(minute=1, hour='*/1'),
        'args': ('AAPL',"1d"),
    },

    'fetch-ethusdt-1h': {
        'task': 'fetch_binance_data',
        'schedule': crontab(minute=2, hour='*/1'),
        'args': ('ETHUSDT', '1h', 1000),
    },
    'fetch-goog-hourly': {
        'task': 'fetch_yfinance_data',
        'schedule': crontab(minute=3, hour='*/1'),
        'args': ('GOOG',"1d"),
    },

    'fetch-solusdt-1h': {
        'task': 'fetch_binance_data',
        'schedule': crontab(minute=4, hour='*/1'),
        'args': ('SOLUSDT', '1h', 1000),
    },
    'fetch-msft-hourly': {
        'task': 'fetch_yfinance_data',
        'schedule': crontab(minute=5, hour='*/1'),
        'args': ('MSFT',"1d"),
    },

    'fetch-adausdt-1h': {
        'task': 'fetch_binance_data',
        'schedule': crontab(minute=6, hour='*/1'),
        'args': ('ADAUSDT', '1h', 1000),
    },
    'fetch-amzn-hourly': {
        'task': 'fetch_yfinance_data',
        'schedule': crontab(minute=7, hour='*/1'),
        'args': ('AMZN',"1d"),
    },

    'fetch-xrpusdt-1h': {
        'task': 'fetch_binance_data',
        'schedule': crontab(minute=8, hour='*/1'),
        'args': ('XRPUSDT', '1h', 1000),
    },
    'fetch-tsla-hourly': {
        'task': 'fetch_yfinance_data',
        'schedule': crontab(minute=9, hour='*/1'),
        'args': ('TSLA',"1d"),
    },
}

# Set timezone for Celery (important for crontab schedules)
celery_app.conf.timezone = 'UTC'