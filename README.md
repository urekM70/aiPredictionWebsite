# FusionPredict

## AI-Powered Trading Predictions for Crypto & Stocks

An intelligent prediction system that leverages machine learning algorithms to forecast price movements in cryptocurrency and stock markets.

### Key Features

- **Advanced ML Models**: Utilizes ensemble learning techniques with gradient boosting and neural networks
- **Multi-Market Analysis**: Specialized in crypto markets with additional support for traditional stocks
- **Pattern Recognition**: Identifies key market patterns and anomalies across timeframes
- **Real-Time Data Integration**: Connects to market APIs for up-to-date prediction calculations
- **Customizable Indicators**: Configure technical indicators that matter most to your trading strategy
- **Performance Tracking**: Measures prediction accuracy against actual market movements

### Technical Stack

- Python backend with TensorFlow and PyTorch
- Flask web framework for the user interface
- Data processing via pandas and NumPy
- Interactive visualizations with Plotly
¤ REST API for integration with trading platforms
- Docker containerization for easy deployment
¤ Add celery and rabbit mq generaton queue

### Installation

```bash
# Clone the repository
git clone https://github.com/urekM70/aiPredictionWebsite
cd aiPredictionWebsite

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask application
flask run
```

### Disclaimer

This software is for informational purposes only. Past performance does not guarantee future results. Always conduct your own research before making investment decisions.
