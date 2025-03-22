import pandas as pd
import numpy as np
from ..config.config import TECHNICAL_INDICATORS

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df.copy()
        
    def calculate_all(self):
        """Calculate all technical indicators"""
        self.calculate_sma()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_momentum()
        self.calculate_roc()
        self.calculate_atr()
        return self.df
        
    def calculate_sma(self):
        """Calculate Simple Moving Averages"""
        for period in TECHNICAL_INDICATORS['SMA']:
            self.df[f'SMA_{period}'] = self.df['Close'].rolling(window=period).mean()
            
    def calculate_rsi(self):
        """Calculate Relative Strength Index"""
        period = TECHNICAL_INDICATORS['RSI']
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
    def calculate_macd(self):
        """Calculate MACD"""
        config = TECHNICAL_INDICATORS['MACD']
        exp1 = self.df['Close'].ewm(span=config['fast'], adjust=False).mean()
        exp2 = self.df['Close'].ewm(span=config['slow'], adjust=False).mean()
        self.df['MACD'] = exp1 - exp2
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=config['signal'], adjust=False).mean()
        
    def calculate_bollinger_bands(self):
        """Calculate Bollinger Bands"""
        config = TECHNICAL_INDICATORS['BB']
        self.df['BB_Middle'] = self.df['Close'].rolling(window=config['period']).mean()
        std_dev = self.df['Close'].rolling(window=config['period']).std()
        self.df['BB_Upper'] = self.df['BB_Middle'] + (std_dev * config['std_dev'])
        self.df['BB_Lower'] = self.df['BB_Middle'] - (std_dev * config['std_dev'])
        
    def calculate_momentum(self):
        """Calculate Momentum"""
        period = TECHNICAL_INDICATORS['Momentum']
        self.df['Momentum'] = self.df['Close'] - self.df['Close'].shift(period)
        
    def calculate_roc(self):
        """Calculate Rate of Change"""
        period = TECHNICAL_INDICATORS['ROC']
        self.df['ROC'] = self.df['Close'].pct_change(period) * 100
        
    def calculate_atr(self):
        """Calculate Average True Range"""
        period = TECHNICAL_INDICATORS['ATR']
        high_low = self.df['High'] - self.df['Low']
        high_close = np.abs(self.df['High'] - self.df['Close'].shift())
        low_close = np.abs(self.df['Low'] - self.df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        self.df['ATR'] = true_range.rolling(period).mean() 