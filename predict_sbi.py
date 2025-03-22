import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_technical_indicators(df):
    # Calculate additional technical indicators
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(10) * 100
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    return df

def prepare_data(df, lookback=60):
    # Select features
    features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD',
                'BB_Middle', 'BB_Upper', 'BB_Lower', 'Momentum', 'ROC', 'ATR']
    data = df[features].copy()
    
    # Handle missing values
    data = data.ffill().bfill()
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - lookback):
        X.append(scaled_data[i:(i + lookback)])
        y.append(scaled_data[i + lookback, 0])
    
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# Load and prepare the data
df = pd.read_csv('sbi_stock_data_3years.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Calculate additional technical indicators
df = calculate_technical_indicators(df)

# Prepare sequences
X, y, scaler = prepare_data(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train multiple models for ensemble
n_models = 5
models = []
predictions_list = []

for i in range(n_models):
    print(f"\nTraining model {i+1}/{n_models}")
    model = build_model((X.shape[1], X.shape[2]))
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    models.append(model)

# Prepare last sequence for prediction
last_sequence = X[-1:]

try:
    # Get ensemble predictions
    all_predictions = []
    for _ in range(5):  # Predict next 5 days
        current_predictions = []
        current_sequence = last_sequence.copy()
        
        # Get predictions from all models
        for model in models:
            pred = model.predict(current_sequence, verbose=0)
            current_predictions.append(pred[0, 0])
        
        # Calculate mean and std of predictions
        mean_pred = np.mean(current_predictions)
        std_pred = np.std(current_predictions)
        all_predictions.append({
            'mean': mean_pred,
            'lower': mean_pred - 2*std_pred,
            'upper': mean_pred + 2*std_pred
        })
        
        # Update sequence for next prediction
        new_sequence = current_sequence[0, 1:, :]
        new_point = np.zeros((1, X.shape[2]))
        new_point[0, 0] = mean_pred
        last_sequence = np.vstack([new_sequence, new_point]).reshape(1, X.shape[1], X.shape[2])

    # Transform predictions back to original scale
    last_close_scaler = MinMaxScaler()
    last_close_scaler.fit(df['Close'].values.reshape(-1, 1))
    
    mean_predictions = np.array([p['mean'] for p in all_predictions]).reshape(-1, 1)
    lower_predictions = np.array([p['lower'] for p in all_predictions]).reshape(-1, 1)
    upper_predictions = np.array([p['upper'] for p in all_predictions]).reshape(-1, 1)
    
    mean_prices = last_close_scaler.inverse_transform(mean_predictions)
    lower_prices = last_close_scaler.inverse_transform(lower_predictions)
    upper_prices = last_close_scaler.inverse_transform(upper_predictions)

    # Generate future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(5)]

    # Create prediction DataFrame
    predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': mean_prices.flatten(),
        'Lower_Bound': lower_prices.flatten(),
        'Upper_Bound': upper_prices.flatten()
    })
    predictions_df.set_index('Date', inplace=True)

    # Plot results with confidence intervals
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(df.index[-30:], df['Close'][-30:], label='Historical Close Price', color='blue')
    
    # Plot predictions with confidence interval
    plt.plot(predictions_df.index, predictions_df['Predicted_Close'], 
             label='Predicted Close Price', color='red', linestyle='--')
    plt.fill_between(predictions_df.index, 
                     predictions_df['Lower_Bound'],
                     predictions_df['Upper_Bound'],
                     color='red', alpha=0.1, label='95% Confidence Interval')

    plt.title('SBI Stock Price Prediction for Next 5 Days (with Confidence Intervals)')
    plt.xlabel('Date')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('sbi_predictions_enhanced.png')
    plt.close()

    # Print predictions with confidence intervals
    print("\nStock Price Predictions for Next 5 Days:")
    print("=====================================")
    for date, row in predictions_df.iterrows():
        print(f"{date.strftime('%Y-%m-%d')}:")
        print(f"  Predicted: ₹{row['Predicted_Close']:.2f}")
        print(f"  Range: ₹{row['Lower_Bound']:.2f} - ₹{row['Upper_Bound']:.2f}")

    # Calculate prediction metrics
    print("\nPrediction Metrics:")
    print("=====================================")
    last_close = df['Close'].iloc[-1]
    print(f"Current Price: ₹{last_close:.2f}")
    mean_change = predictions_df['Predicted_Close'][-1] - last_close
    print(f"5-day Price Change: ₹{mean_change:.2f}")
    print(f"Predicted Return: {(mean_change / last_close * 100):.2f}%")
    
    # Print technical indicators
    print("\nCurrent Technical Indicators:")
    print("=====================================")
    print(f"RSI: {df['RSI'].iloc[-1]:.2f}")
    print(f"MACD: {df['MACD'].iloc[-1]:.2f}")
    print(f"Bollinger Bands:")
    print(f"  Upper: ₹{df['BB_Upper'].iloc[-1]:.2f}")
    print(f"  Middle: ₹{df['BB_Middle'].iloc[-1]:.2f}")
    print(f"  Lower: ₹{df['BB_Lower'].iloc[-1]:.2f}")
    print(f"Momentum: {df['Momentum'].iloc[-1]:.2f}")
    print(f"ROC: {df['ROC'].iloc[-1]:.2f}%")
    print(f"ATR: ₹{df['ATR'].iloc[-1]:.2f}")

    # Save predictions to CSV
    predictions_df.to_csv('sbi_predictions_enhanced.csv')
    print("\nPredictions have been saved to 'sbi_predictions_enhanced.csv'")
    print("Enhanced prediction chart has been saved as 'sbi_predictions_enhanced.png'")

except Exception as e:
    print(f"An error occurred during prediction: {str(e)}")
    print("Please check the data and model parameters.") 