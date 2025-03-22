import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stock_prediction.config.config import DATA_CONFIG, RAW_DATA_DIR

class StockDataCollector:
    def __init__(self, symbol):
        self.symbol = symbol
        self.start_date = DATA_CONFIG['start_date']
        self.end_date = DATA_CONFIG['end_date']
        self.interval = DATA_CONFIG['interval']
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(self.symbol)
            df = stock.history(
                start=self.start_date,
                end=self.end_date,
                interval=self.interval
            )
            
            # Save raw data
            output_file = os.path.join(RAW_DATA_DIR, f"{self.symbol}_raw_data.csv")
            df.to_csv(output_file)
            
            print(f"Data collected for {self.symbol} from {self.start_date} to {self.end_date}")
            return df
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {str(e)}")
            return None
            
    def get_stock_info(self):
        """Get general stock information"""
        try:
            stock = yf.Ticker(self.symbol)
            info = stock.info
            return info
        except Exception as e:
            print(f"Error fetching stock info for {self.symbol}: {str(e)}")
            return None 