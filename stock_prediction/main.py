import argparse
from datetime import datetime
import pandas as pd
from utils.data_collector import StockDataCollector
from utils.technical_indicators import TechnicalIndicators
from models.lstm_model import StockPredictor
from utils.visualizer import StockVisualizer

def predict_stock(symbol, retrain=False):
    """Run the complete stock prediction pipeline"""
    print(f"\nStarting prediction pipeline for {symbol}")
    print("=" * 50)
    
    # 1. Data Collection
    print("\n1. Collecting stock data...")
    collector = StockDataCollector(symbol)
    df = collector.fetch_data()
    
    if df is None or df.empty:
        print(f"Error: Could not fetch data for {symbol}")
        return
    
    # 2. Calculate Technical Indicators
    print("\n2. Calculating technical indicators...")
    ti = TechnicalIndicators(df)
    df = ti.calculate_all()
    
    # 3. Prepare Predictor
    print("\n3. Preparing prediction model...")
    predictor = StockPredictor(symbol)
    
    # Check if we should retrain or load existing models
    if not retrain and predictor.load_models():
        print("Loaded existing models")
    else:
        print("Training new models...")
        X, y = predictor.prepare_data(df)
        predictor.train(X, y)
    
    # 4. Make Predictions
    print("\n4. Generating predictions...")
    X, _ = predictor.prepare_data(df)
    last_sequence = X[-1:]
    predictions = predictor.predict(last_sequence)
    
    # Create predictions DataFrame
    future_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=len(predictions),
        freq='D'
    )
    
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': [p['mean'] for p in predictions],
        'Lower_Bound': [p['lower'] for p in predictions],
        'Upper_Bound': [p['upper'] for p in predictions]
    })
    predictions_df.set_index('Date', inplace=True)
    
    # 5. Visualize Results
    print("\n5. Creating visualizations...")
    visualizer = StockVisualizer(symbol)
    visualizer.plot_predictions(df, predictions_df)
    visualizer.plot_technical_indicators(df)
    visualizer.save_predictions_to_csv(predictions_df)
    
    # 6. Generate Report
    print("\n6. Generating summary report...")
    report = visualizer.create_summary_report(df, predictions_df)
    print("\nSummary Report:")
    print(report)
    
    print("\nPrediction pipeline completed!")
    print(f"Check the 'output' directory for {symbol}_predictions.png, {symbol}_technical.png, and {symbol}_report.txt")

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction')
    parser.add_argument('symbols', nargs='+', help='Stock symbols to predict (e.g., SBIN.NS RELIANCE.NS)')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of models')
    
    args = parser.parse_args()
    
    for symbol in args.symbols:
        predict_stock(symbol, args.retrain)

if __name__ == "__main__":
    main() 