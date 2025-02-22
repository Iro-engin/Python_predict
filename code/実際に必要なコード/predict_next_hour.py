import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

'''
# Enable GPU usage if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU is available and enabled.")
    except RuntimeError as e:
        print(e)
'''
# Load the existing data
df = pd.read_csv("/code/USDJPY/1hour.csv")

# Preprocess the data
def preprocess_data(df):
    df['diff_open'] = df['open'].diff()
    df['diff_high'] = df['high'].diff()
    df['diff_low'] = df['low'].diff()
    df['diff_close'] = df['close'].diff()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = preprocess_data(df)

# Convert 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'])

# Standardize the data
scaler_open = StandardScaler()
scaler_high = StandardScaler()
scaler_low = StandardScaler()
scaler_close = StandardScaler()

scaled_open = scaler_open.fit_transform(df[['diff_open']])
scaled_high = scaler_high.fit_transform(df[['diff_high']])
scaled_low = scaler_low.fit_transform(df[['diff_low']])
scaled_close = scaler_close.fit_transform(df[['diff_close']])

# Build LSTM model
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Prepare the data for prediction
window_size = 50
last_open = scaled_open[-window_size:].reshape(1, window_size, 1)
last_high = scaled_high[-window_size:].reshape(1, window_size, 1)
last_low = scaled_low[-window_size:].reshape(1, window_size, 1)
last_close = scaled_close[-window_size:].reshape(1, window_size, 1)

# Train and predict for each OHLC component
def train_and_predict(scaled_data, last_data):
    X = []
    y = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])
    X = np.array(X)
    y = np.array(y)

    model = build_lstm_model()
    model.fit(X, y, epochs=50, batch_size=32)
    predicted_diff = model.predict(last_data)
    return predicted_diff

predicted_open_diff = train_and_predict(scaled_open, last_open)
predicted_high_diff = train_and_predict(scaled_high, last_high)
predicted_low_diff = train_and_predict(scaled_low, last_low)
predicted_close_diff = train_and_predict(scaled_close, last_close)

predicted_open_diff = scaler_open.inverse_transform(predicted_open_diff)[0, 0]
predicted_high_diff = scaler_high.inverse_transform(predicted_high_diff)[0, 0]
predicted_low_diff = scaler_low.inverse_transform(predicted_low_diff)[0, 0]
predicted_close_diff = scaler_close.inverse_transform(predicted_close_diff)[0, 0]

# Calculate the predicted OHLC prices
last_open_price = df['open'].iloc[-1]
last_high_price = df['high'].iloc[-1]
last_low_price = df['low'].iloc[-1]
last_close_price = df['close'].iloc[-1]

predicted_open = last_open_price + predicted_open_diff
predicted_high = last_high_price + predicted_high_diff
predicted_low = last_low_price + predicted_low_diff
predicted_close = last_close_price + predicted_close_diff

print(f"Predicted OHLC for the next hour: Open={predicted_open}, High={predicted_high}, Low={predicted_low}, Close={predicted_close}")

# Save the prediction results to a CSV file
prediction_results = pd.DataFrame({
    'predicted_open': [predicted_open],
    'predicted_high': [predicted_high],
    'predicted_low': [predicted_low],
    'predicted_close': [predicted_close]
})
prediction_results.to_csv("/code/USDJPY/predicted_next_hour.csv", index=False)

# Plot the results
plt.figure(figsize=(16, 8))
plt.plot(df['time'].iloc[-window_size:], df['close'].iloc[-window_size:], label='Actual Close')
plt.scatter([df['time'].iloc[-1] + pd.Timedelta(hours=1)], [predicted_close], color='red', label='Predicted Close')
plt.legend()
plt.title('USD/JPY OHLC Prediction for the Next Hour')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.show()
