import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# 1. Read the 4hour.csv file
df_4hour = pd.read_csv("/code/USDJPY/4hour.csv")

# 2. Read the predicted_close.csv file
df_predicted_close = pd.read_csv("/code/USDJPY/predicted_close.csv")

# 3. Add the predicted_close column to the 4hour.csv dataframe
df_4hour['predicted_close'] = df_predicted_close['predicted_close']

# 4. Save the updated dataframe to a new CSV file
df_4hour.to_csv("/code/USDJPY/4hour_with_predicted_close.csv", index=False)

# 5. Read the 1hour.csv file
df_1hour = pd.read_csv("/code/USDJPY/1hour.csv")

# 6. Data preprocessing
df_4hour['time'] = pd.to_datetime(df_4hour['time'])
df_1hour['time'] = pd.to_datetime(df_1hour['time'])

# 7. Create differences for 1-hour data
df_1hour['diff_open'] = df_1hour['open'].diff()
df_1hour['diff_high'] = df_1hour['high'].diff()
df_1hour['diff_low'] = df_1hour['low'].diff()
df_1hour['diff_close'] = df_1hour['close'].diff()
df_1hour.dropna(inplace=True)

# 8. Select features
features = df_1hour[['diff_open', 'diff_high', 'diff_low', 'diff_close']]

# 9. Standardize the data
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# 10. Build LSTM models
window_size = 50  # Use past N periods of data
X = []
y = {'open': [], 'high': [], 'low': [], 'close': []}
for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i])
    for col in ['open', 'high', 'low', 'close']:
        y[col].append(scaled_features[i, features.columns.get_loc(f'diff_{col}')])
X = np.array(X)
for col in y:
    y[col] = np.array(y[col])

models = {}
input_shape = (window_size, scaled_features.shape[1])
for col in ['open', 'high', 'low', 'close']:
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    models[col] = model

# 11. Train the models
for col in models:
    models[col].fit(X, y[col], epochs=100, batch_size=32)

# 12. Make predictions
future_steps = len(df_4hour) * 4  # Interpolate each hour for 4-hour data
predictions = {'open': [], 'high': [], 'low': [], 'close': []}
last_data = scaled_features[-window_size:].reshape(1, window_size, scaled_features.shape[1])
for _ in range(future_steps):
    for col in models:
        predicted_value = models[col].predict(last_data)
        predictions[col].append(predicted_value[0, 0])
        predicted_value_full = np.concatenate([predicted_value, np.zeros((1, scaled_features.shape[1] - 1))], axis=1)
        last_data = np.concatenate([last_data[:, 1:, :], predicted_value_full.reshape(1, 1, scaled_features.shape[1])], axis=1)

# 13. Accumulate differences to revert to original data
initial_values = {col: df_1hour[col].iloc[-1] for col in ['open', 'high', 'low', 'close']}
predicted_values = {col: np.cumsum(predictions[col]) + initial_values[col] for col in predictions}

# 14. Save the predicted data to CSV files
for col in predicted_values:
    predicted_df = pd.DataFrame({f'predicted_{col}': predicted_values[col]})
    predicted_df.to_csv(f"/code/USDJPY/predicted_{col}_1hour.csv", index=False)

# 15. Plot the predictions
for col in ['open', 'high', 'low', 'close']:
    plt.figure(figsize=(16, 8))
    plt.plot(predicted_values[col], label='Predicted', color='red')
    plt.legend()
    plt.title(f'USD/JPY {col.capitalize()} Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{col.capitalize()} Price')
    plt.grid(True)
    plt.savefig(f"/code/USDJPY/{col}_price_prediction_1hour.png")
    plt.show()
