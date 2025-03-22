import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import os
from ..config.config import VIZ_CONFIG, OUTPUT_DIR

class StockVisualizer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.config = VIZ_CONFIG
        
    def plot_predictions(self, historical_data, predictions_df):
        """Plot stock predictions with confidence intervals"""
        plt.figure(figsize=self.config['figure_size'])
        
        # Plot historical data
        plt.plot(historical_data.index[-self.config['history_days']:],
                historical_data['Close'][-self.config['history_days']:],
                label='Historical Close Price', color='blue')
        
        # Plot predictions with confidence interval
        plt.plot(predictions_df.index, predictions_df['Predicted_Close'],
                label='Predicted Close Price', color='red', linestyle='--')
        plt.fill_between(predictions_df.index,
                        predictions_df['Lower_Bound'],
                        predictions_df['Upper_Bound'],
                        color='red', alpha=0.1,
                        label=f"{int(self.config['confidence_interval']*100)}% Confidence Interval")
        
        plt.title(f'{self.symbol} Stock Price Prediction for Next {len(predictions_df)} Days')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(OUTPUT_DIR, f'{self.symbol}_predictions.png')
        plt.savefig(output_file)
        plt.close()
        
    def plot_technical_indicators(self, df):
        """Plot technical indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # Price and Moving Averages
        axes[0].plot(df.index, df['Close'], label='Close Price')
        axes[0].plot(df.index, df['SMA_20'], label='SMA 20')
        axes[0].plot(df.index, df['SMA_50'], label='SMA 50')
        axes[0].fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1)
        axes[0].set_title(f'{self.symbol} Price and Moving Averages')
        axes[0].legend()
        axes[0].grid(True)
        
        # RSI
        axes[1].plot(df.index, df['RSI'], label='RSI')
        axes[1].axhline(y=70, color='r', linestyle='--')
        axes[1].axhline(y=30, color='g', linestyle='--')
        axes[1].set_title('Relative Strength Index')
        axes[1].legend()
        axes[1].grid(True)
        
        # MACD
        axes[2].plot(df.index, df['MACD'], label='MACD')
        axes[2].plot(df.index, df['Signal_Line'], label='Signal Line')
        axes[2].set_title('MACD')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(OUTPUT_DIR, f'{self.symbol}_technical.png')
        plt.savefig(output_file)
        plt.close()
        
    def save_predictions_to_csv(self, predictions_df):
        """Save predictions to CSV"""
        output_file = os.path.join(OUTPUT_DIR, f'{self.symbol}_predictions.csv')
        predictions_df.to_csv(output_file)
        
    def create_summary_report(self, df, predictions_df):
        """Create a summary report of predictions and technical indicators"""
        report = []
        
        # Current price and technical indicators
        report.append("Current Market Status:")
        report.append("=" * 50)
        report.append(f"Current Price: ₹{df['Close'].iloc[-1]:.2f}")
        report.append(f"RSI: {df['RSI'].iloc[-1]:.2f}")
        report.append(f"MACD: {df['MACD'].iloc[-1]:.2f}")
        report.append(f"Signal Line: {df['Signal_Line'].iloc[-1]:.2f}")
        report.append(f"20-day SMA: ₹{df['SMA_20'].iloc[-1]:.2f}")
        report.append(f"50-day SMA: ₹{df['SMA_50'].iloc[-1]:.2f}")
        
        # Predictions
        report.append("\nPrice Predictions:")
        report.append("=" * 50)
        for date, row in predictions_df.iterrows():
            report.append(f"{date.strftime('%Y-%m-%d')}:")
            report.append(f"  Predicted: ₹{row['Predicted_Close']:.2f}")
            report.append(f"  Range: ₹{row['Lower_Bound']:.2f} - ₹{row['Upper_Bound']:.2f}")
        
        # Technical Analysis
        report.append("\nTechnical Analysis:")
        report.append("=" * 50)
        
        # RSI Analysis
        rsi = df['RSI'].iloc[-1]
        if rsi > 70:
            rsi_signal = "Overbought"
        elif rsi < 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"
        report.append(f"RSI Signal: {rsi_signal}")
        
        # MACD Analysis
        macd = df['MACD'].iloc[-1]
        signal = df['Signal_Line'].iloc[-1]
        if macd > signal:
            macd_signal = "Bullish"
        else:
            macd_signal = "Bearish"
        report.append(f"MACD Signal: {macd_signal}")
        
        # Trend Analysis
        current_price = df['Close'].iloc[-1]
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]
        
        if current_price > sma_20 > sma_50:
            trend = "Strong Uptrend"
        elif current_price > sma_20 and sma_20 < sma_50:
            trend = "Weak Uptrend"
        elif current_price < sma_20 < sma_50:
            trend = "Strong Downtrend"
        else:
            trend = "Weak Downtrend"
        report.append(f"Trend Analysis: {trend}")
        
        # Save report
        output_file = os.path.join(OUTPUT_DIR, f'{self.symbol}_report.txt')
        with open(output_file, 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report) 