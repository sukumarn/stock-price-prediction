# Stock Price Prediction System

A comprehensive stock price prediction system using LSTM-based deep learning models and technical indicators.

## Features

- Data collection from Yahoo Finance
- Technical indicator calculations (SMA, RSI, MACD, Bollinger Bands, etc.)
- Ensemble LSTM model for predictions
- Confidence intervals for predictions
- Detailed visualizations and analysis reports
- Support for multiple stock symbols

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run predictions for a stock symbol:
```bash
python stock_prediction/main.py SYMBOL
```

Example:
```bash
python stock_prediction/main.py AAPL
```

## Project Structure

```
stock_prediction/
├── config/
│   └── config.py         # Configuration settings
├── data/
│   ├── raw/             # Raw data from Yahoo Finance
│   └── processed/       # Processed data with indicators
├── models/              # Saved model files
├── output/             # Predictions and visualizations
└── utils/
    ├── data_collector.py    # Data collection utilities
    ├── technical_indicators.py  # Technical analysis
    └── visualizer.py       # Visualization utilities
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 