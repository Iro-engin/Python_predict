import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
predicted_df = pd.read_csv("/code/USDJPY/predicted_close.csv")
hourly_df = pd.read_csv("/code/USDJPY/1hour.csv")
minute_df = pd.read_csv("/code/USDJPY/15min.csv")

# Merge data
merged_df = pd.merge(hourly_df, predicted_df, left_index=True, right_index=True, how='outer')

# Interpolate missing values using 15min data
def interpolate_with_lstm(df, minute_df, column_name, window_size=50, epochs=10):
    # Prepare data
    minute_values = minute_df[column_name].values
    df[column_name] = df[column_name].interpolate(method='linear')
    df.dropna(inplace=True)
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(minute_values.reshape(-1, 1))
    
    # Create sequences
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X, y, epochs=epochs, batch_size=32)
    
    # Predict missing values
    predictions = []
    if len(df) > window_size:
        for i in range(len(df) - window_size):
            input_seq = scaled_data[i:i+window_size].reshape(1, window_size, 1)
            predicted_value = model.predict(input_seq)
            predictions.append(predicted_value[0, 0])
    
    # Inverse transform predictions
    if predictions:
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        df[column_name].iloc[window_size:] = predictions.flatten()
    
    return df

# Interpolate each column
for col in ['open', 'high', 'low', 'close']:
    merged_df = interpolate_with_lstm(merged_df, minute_df, col)

# Evaluate interpolation accuracy
def evaluate_interpolation(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    corr_coef = np.corrcoef(actual, predicted)[0, 1]
    return mae, rmse, r2, corr_coef

for col in ['open', 'high', 'low', 'close']:
    actual = hourly_df[col].dropna().values
    predicted = merged_df[col].dropna().values
    
    # Ensure the lengths match
    min_length = min(len(actual), len(predicted))
    actual = actual[:min_length]
    predicted = predicted[:min_length]
    
    print(f"Length of actual {col}: {len(actual)}")
    print(f"Length of predicted {col}: {len(predicted)}")
    
    mae, rmse, r2, corr_coef = evaluate_interpolation(actual, predicted)
    print(f"{col.capitalize()} - MAE: {mae}, RMSE: {rmse}, R^2: {r2}, Correlation Coefficient: {corr_coef}")

# Plot results
plt.figure(figsize=(16, 8))
for col in ['open', 'high', 'low', 'close']:
    plt.plot(merged_df['time'], merged_df[col], label=f'Interpolated {col.capitalize()}')
    plt.plot(hourly_df['time'], hourly_df[col], label=f'Actual {col.capitalize()}', linestyle='--')
plt.legend()
plt.title('USD/JPY OHLC Price Interpolation')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.show()
