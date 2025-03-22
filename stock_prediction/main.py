import argparse
from utils.data_collector import StockDataCollector
from utils.technical_indicators import TechnicalIndicator
from utils.news_analyzer import NewsAnalyzer
from models.lstm_model import StockPredictor
from utils.visualizer import StockVisualizer
import logging
import pandas as pd
from datetime import datetime

def analyze_stock(symbol: str, retrain: bool = False):
    """
    Analyze stock with price predictions and enhanced news sentiment
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize components
    logger.info(f"\nAnalyzing {symbol}...")
    data_collector = StockDataCollector(symbol)
    news_analyzer = NewsAnalyzer(symbol)
    
    # Collect data and perform technical analysis
    df = data_collector.fetch_data()
    if df is None:
        logger.error(f"Failed to fetch data for {symbol}")
        return
        
    # Get stock information
    stock_info = data_collector.get_stock_info()
    
    # Calculate technical indicators
    tech_ind = TechnicalIndicator(df)
    df = tech_ind.calculate_all_indicators()
    
    # Perform price prediction
    predictor = StockPredictor(df)
    if retrain:
        predictor.train()
    predictions = predictor.predict_next_days(5)
    
    # Get enhanced news sentiment analysis
    sentiment_df = news_analyzer.get_news_sentiment_analysis()
    trading_signal, signal_confidence = news_analyzer.get_trading_signal()
    
    # Save results
    output_dir = "output"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save sentiment analysis
    sentiment_file = f"{output_dir}/{symbol}_news_sentiment_{timestamp}.csv"
    sentiment_df.to_csv(sentiment_file, index=False)
    
    # Save predictions
    predictions_file = f"{output_dir}/{symbol}_predictions_{timestamp}.csv"
    predictions.to_csv(predictions_file, index=True)
    
    # Print comprehensive analysis summary
    logger.info("\n=== Stock Analysis Summary ===")
    logger.info(f"\nStock: {stock_info.get('longName', symbol)} ({symbol})")
    logger.info(f"Current Price: ${stock_info.get('currentPrice', 'N/A')}")
    logger.info(f"52 Week Range: ${stock_info.get('fiftyTwoWeekLow', 'N/A')} - ${stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
    
    logger.info("\n=== Technical Indicators ===")
    logger.info(f"RSI (14): {tech_ind.get_current_rsi():.2f}")
    logger.info(f"MACD: {tech_ind.get_current_macd():.2f}")
    
    logger.info("\n=== Price Predictions ===")
    for date, row in predictions.iterrows():
        logger.info(f"{date.strftime('%Y-%m-%d')}: ${row['Predicted_Price']:.2f} (Confidence: {row['Confidence']:.2%})")
    
    logger.info("\n=== News Sentiment Analysis ===")
    logger.info(f"Trading Signal: {trading_signal} (Confidence: {signal_confidence:.2%})")
    
    # Calculate sentiment statistics
    sentiment_stats = {
        'Very Positive': len(sentiment_df[sentiment_df['Sentiment'] == 'Very Positive']),
        'Positive': len(sentiment_df[sentiment_df['Sentiment'] == 'Positive']),
        'Neutral': len(sentiment_df[sentiment_df['Sentiment'] == 'Neutral']),
        'Negative': len(sentiment_df[sentiment_df['Sentiment'] == 'Negative']),
        'Very Negative': len(sentiment_df[sentiment_df['Sentiment'] == 'Very Negative'])
    }
    
    logger.info("\nSentiment Distribution:")
    for sentiment, count in sentiment_stats.items():
        logger.info(f"{sentiment}: {count}")
    
    logger.info("\n=== Recent News Analysis ===")
    for _, article in sentiment_df.iterrows():
        logger.info(f"\nSource: {article['Source']}")
        logger.info(f"Title: {article['Title']}")
        logger.info(f"Date: {article['Date'].strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Sentiment: {article['Sentiment']}")
        logger.info(f"Confidence: {article['Confidence']:.2%}")
        logger.info(f"Intensity: {article['Intensity']:.2f}")
        logger.info(f"Link: {article['Link']}")
    
    logger.info(f"\nDetailed analysis saved to:")
    logger.info(f"1. Sentiment Analysis: {sentiment_file}")
    logger.info(f"2. Price Predictions: {predictions_file}")

def main():
    parser = argparse.ArgumentParser(description='Stock Analysis with Price Prediction and Enhanced News Sentiment')
    parser.add_argument('symbol', type=str, help='Stock symbol (e.g., AAPL, GOOGL)')
    parser.add_argument('--retrain', action='store_true', help='Retrain the prediction model')
    
    args = parser.parse_args()
    analyze_stock(args.symbol, args.retrain)

if __name__ == "__main__":
    main() 