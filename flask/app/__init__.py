from flask import Flask
from flask_bcrypt import Bcrypt
from flask_caching import Cache
from app.celery_config import make_celery
from app import celery_instance

bcrypt = Bcrypt()
cache = Cache()
celery = None

def create_app():
    app = Flask(__name__, template_folder="../templates",static_folder="../static")

    # Konfiguracije
    app.secret_key = "aaaaaaaaaaaaaaaaaaaaaa"
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['CELERY_BROKER_URL'] = 'amqp://guest:guest@localhost:5672//'
    app.config['CELERY_RESULT_BACKEND'] = 'rpc://'
    
    app.config['CACHE_TYPE'] = 'simple'  
    app.config['CACHE_DEFAULT_TIMEOUT'] = 600
    # Inicializacija razširitev
    bcrypt.init_app(app)
    cache.init_app(app)

    # Celery setup
    global celery_app
    celery_instance.celery_app = make_celery(app)
    from tasks import data_tasks
    # Registracija route-ov in API endpointov
    from .routes import setup_routes
    from .api import api_bp
    from .db import init_db # Import init_db
 
    setup_routes(app)
    app.register_blueprint(api_bp)

    # Initialize database tables
    with app.app_context():
        init_db()

    return app