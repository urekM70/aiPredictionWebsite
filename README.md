# FusionPredict

## AI-Powered Trading Predictions for Crypto & Stocks

An intelligent prediction system that leverages machine learning algorithms to forecast price movements in cryptocurrency and stock markets.

### Key Features

* **Advanced ML Models**: Utilizes ensemble learning techniques with gradient boosting and neural networks
* **Multi-Market Analysis**: Specialized in crypto markets with additional support for traditional stocks
* **Pattern Recognition**: Identifies key market patterns and anomalies across timeframes
* **Real-Time Data Integration**: Connects to market APIs for up-to-date prediction calculations
* **Customizable Indicators**: Configure technical indicators that matter most to your trading strategy
* **Performance Tracking**: Measures prediction accuracy against actual market movements

### Technical Stack

* Python backend with XGBoost
* Flask web framework for the user interface
* Data processing via pandas and NumPy
* Interactive visualizations with Plotly
* REST API for integration with trading platforms
* Docker containerization for easy deployment
* Celery and RabbitMQ for asynchronous task processing

---

### Installation & Setup

#### 1. Install Python

Install Python 3.10 or newer.

#### 2. Clone the Repository

```bash
git clone https://github.com/urekM70/aiPredictionWebsite
cd aiPredictionWebsite
```

#### 3. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Set Environment Variables

Replace with your own values:

```bash
export BINANCE_API_KEY=your_key
export BINANCE_API_SECRET=your_secret
export SECRET_KEY=your_flask_secret
```

On Windows (Command Prompt):

```cmd
set BINANCE_API_KEY=your_key
set BINANCE_API_SECRET=your_secret
set SECRET_KEY=your_flask_secret
```

#### 6. Run RabbitMQ

Ensure RabbitMQ is running. You can start the bundled Docker container or use a local RabbitMQ server.

Example with Docker:

```bash
docker run -d --hostname my-rabbit --name some-rabbit -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

#### 7. Start the Celery Worker and Celery Beat

```bash
celery -A flask.celery_worker.celery_app worker --loglevel=info
```
```bash
celery -A flask.celery_worker.celery_app beat --loglevel=info
```

#### 8. Run the Application

Development:

```bash
python flask/main.py
```

Production (with Gunicorn):

```bash
gunicorn -w 4 -b 0.0.0.0:8000 flask.main:app
```

Access the app at: [http://localhost:8000](http://localhost:8000)

---

### Disclaimer

This software is for informational purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.
