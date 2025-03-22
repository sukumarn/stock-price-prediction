import os
from datetime import datetime, timedelta

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'lookback_period': 60,
    'n_models': 5,
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.1,
    'test_size': 0.2,
    'prediction_days': 5,
    'lstm_units': [100, 50, 50],
    'dense_units': [25],
    'dropout_rate': 0.2,
    'learning_rate': 0.001
}

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    'SMA': [20, 50, 200],
    'RSI': 14,
    'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
    'BB': {'period': 20, 'std_dev': 2},
    'ROC': 10,
    'ATR': 14,
    'Momentum': 10
}

# Data configuration
DATA_CONFIG = {
    'start_date': (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d'),
    'end_date': datetime.now().strftime('%Y-%m-%d'),
    'interval': '1d'
}

# Visualization configuration
VIZ_CONFIG = {
    'figure_size': (15, 8),
    'history_days': 30,
    'confidence_interval': 0.95
} 