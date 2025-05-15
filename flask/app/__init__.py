from flask import Flask
from flask_bcrypt import Bcrypt
from flask_caching import Cache
from .celery_config import make_celery

bcrypt = Bcrypt()
cache = Cache()
celery = None

def create_app():
    app = Flask(__name__, template_folder="../templates")

    # Konfiguracije
    app.secret_key = "aaaaaaaaaaaaaaaaaaaaaa"
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB
    app.config['UPLOAD_FOLDER'] = 'static/uploads'
    app.config['CELERY_BROKER_URL'] = 'amqp://guest:guest@localhost:5672//'
    app.config['CELERY_RESULT_BACKEND'] = 'rpc://'

    # Inicializacija razširitev
    bcrypt.init_app(app)
    cache.init_app(app)

    # Celery setup
    global celery
    celery = make_celery(app)

    # Registracija route-ov in API endpointov
    from .routes import setup_routes
    from .api import api_bp

    setup_routes(app)
    app.register_blueprint(api_bp)

    return app
