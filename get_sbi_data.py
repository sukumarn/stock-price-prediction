import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, lower_band

# Create a ticker object for SBI (NSE)
sbi = yf.Ticker("SBIN.NS")

# Get dates for 3 years of data
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # 3 years

# Get historical data for the last 3 years
hist_data = sbi.history(start=start_date, end=end_date)

# Calculate SMAs
hist_data['SMA_20'] = hist_data['Close'].rolling(window=20).mean()
hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
hist_data['SMA_200'] = hist_data['Close'].rolling(window=200).mean()

# Calculate RSI
hist_data['RSI'] = calculate_rsi(hist_data['Close'])

# Calculate MACD
hist_data['MACD'], hist_data['Signal_Line'] = calculate_macd(hist_data['Close'])
hist_data['MACD_Histogram'] = hist_data['MACD'] - hist_data['Signal_Line']

# Calculate Bollinger Bands
hist_data['BB_Upper'], hist_data['BB_Lower'] = calculate_bollinger_bands(hist_data['Close'])
hist_data['BB_Middle'] = hist_data['SMA_20']  # Middle band is 20-day SMA

# Calculate Volume Moving Average
hist_data['Volume_MA'] = hist_data['Volume'].rolling(window=20).mean()

# Generate Trading Signals
hist_data['Signal'] = 0  # 0: Hold, 1: Buy, -1: Sell
# SMA Crossover
hist_data.loc[hist_data['SMA_20'] > hist_data['SMA_50'], 'SMA_Cross'] = 1
hist_data.loc[hist_data['SMA_20'] < hist_data['SMA_50'], 'SMA_Cross'] = -1

# RSI Signals
hist_data.loc[hist_data['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold
hist_data.loc[hist_data['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought

# MACD Signals
hist_data.loc[hist_data['MACD'] > hist_data['Signal_Line'], 'MACD_Signal'] = 1
hist_data.loc[hist_data['MACD'] < hist_data['Signal_Line'], 'MACD_Signal'] = -1

# Get current stock info
info = sbi.info

# Print current stock information
print("\nCurrent Stock Information:")
print(f"Current Price: ₹{info.get('currentPrice', 'N/A')}")
print(f"Previous Close: ₹{info.get('previousClose', 'N/A')}")
print(f"Open: ₹{info.get('open', 'N/A')}")
print(f"Day's High: ₹{info.get('dayHigh', 'N/A')}")
print(f"Day's Low: ₹{info.get('dayLow', 'N/A')}")
print(f"52 Week High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
print(f"52 Week Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}")
print(f"Volume: {info.get('volume', 'N/A'):,}")
print(f"Market Cap: ₹{info.get('marketCap', 'N/A'):,}")

# Print current technical indicators
print("\nCurrent Technical Indicators:")
print(f"20-day SMA: ₹{hist_data['SMA_20'].iloc[-1]:.2f}")
print(f"50-day SMA: ₹{hist_data['SMA_50'].iloc[-1]:.2f}")
print(f"200-day SMA: ₹{hist_data['SMA_200'].iloc[-1]:.2f}")
print(f"RSI (14): {hist_data['RSI'].iloc[-1]:.2f}")
print(f"MACD: {hist_data['MACD'].iloc[-1]:.2f}")
print(f"Signal Line: {hist_data['Signal_Line'].iloc[-1]:.2f}")
print(f"Bollinger Bands:")
print(f"  Upper: ₹{hist_data['BB_Upper'].iloc[-1]:.2f}")
print(f"  Middle: ₹{hist_data['BB_Middle'].iloc[-1]:.2f}")
print(f"  Lower: ₹{hist_data['BB_Lower'].iloc[-1]:.2f}")

# Technical Analysis Summary
print("\nTechnical Analysis Summary:")
current_price = hist_data['Close'].iloc[-1]
if current_price > hist_data['BB_Upper'].iloc[-1]:
    print("Price above upper Bollinger Band - Potentially overbought")
elif current_price < hist_data['BB_Lower'].iloc[-1]:
    print("Price below lower Bollinger Band - Potentially oversold")

if hist_data['RSI'].iloc[-1] > 70:
    print("RSI indicates overbought conditions")
elif hist_data['RSI'].iloc[-1] < 30:
    print("RSI indicates oversold conditions")

if hist_data['MACD'].iloc[-1] > hist_data['Signal_Line'].iloc[-1]:
    print("MACD shows bullish momentum")
else:
    print("MACD shows bearish momentum")

# Calculate yearly statistics
yearly_stats = hist_data['Close'].resample('Y').agg(['mean', 'min', 'max', 'std'])
print("\nYearly Statistics:")
print(yearly_stats)

# Calculate monthly returns
monthly_returns = hist_data['Close'].resample('M').last().pct_change()
print("\nAverage Monthly Returns:", monthly_returns.mean() * 100, "%")

# Calculate total return over 3 years
total_return = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[0]) / hist_data['Close'].iloc[0]) * 100
print(f"\nTotal Return over 3 years: {total_return:.2f}%")

# Create subplots for visualization
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})

# Plot 1: Price with SMAs and Bollinger Bands
ax1.plot(hist_data.index, hist_data['Close'], label='Close Price', color='blue')
ax1.plot(hist_data.index, hist_data['SMA_20'], label='20-day SMA', color='orange', alpha=0.7)
ax1.plot(hist_data.index, hist_data['SMA_50'], label='50-day SMA', color='green', alpha=0.7)
ax1.plot(hist_data.index, hist_data['SMA_200'], label='200-day SMA', color='red', alpha=0.7)
ax1.plot(hist_data.index, hist_data['BB_Upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.5)
ax1.plot(hist_data.index, hist_data['BB_Lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(hist_data.index, hist_data['BB_Upper'], hist_data['BB_Lower'], alpha=0.1, color='gray')
ax1.set_title('SBI Stock Price with Technical Indicators')
ax1.set_ylabel('Price (₹)')
ax1.grid(True)
ax1.legend()

# Plot 2: Volume
ax2.bar(hist_data.index, hist_data['Volume'], label='Volume', color='blue', alpha=0.5)
ax2.plot(hist_data.index, hist_data['Volume_MA'], label='20-day Volume MA', color='orange')
ax2.set_ylabel('Volume')
ax2.grid(True)
ax2.legend()

# Plot 3: RSI
ax3.plot(hist_data.index, hist_data['RSI'], label='RSI', color='purple')
ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
ax3.set_ylabel('RSI')
ax3.grid(True)
ax3.legend()

# Plot 4: MACD
ax4.plot(hist_data.index, hist_data['MACD'], label='MACD', color='blue')
ax4.plot(hist_data.index, hist_data['Signal_Line'], label='Signal Line', color='orange')
ax4.bar(hist_data.index, hist_data['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3)
ax4.set_ylabel('MACD')
ax4.set_xlabel('Date')
ax4.grid(True)
ax4.legend()

plt.tight_layout()
plt.savefig('sbi_technical_analysis.png')
plt.close()

# Save complete data to CSV
hist_data.to_csv('sbi_stock_data_3years.csv')
print("\nData has been saved to 'sbi_stock_data_3years.csv'")
print("Technical analysis chart has been saved as 'sbi_technical_analysis.png'") 