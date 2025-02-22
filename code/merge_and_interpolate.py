import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load data
predicted_df = pd.read_csv("/code/USDJPY/predicted_close.csv")
hourly_df = pd.read_csv("/code/USDJPY/1hour.csv")
minute_df = pd.read_csv("/code/USDJPY/15min.csv")

# Interpolate missing values using LSTM model
def interpolate_with_lstm(df, minute_df, column_name, window_size=50):
    # Prepare data
    minute_values = minute_df[column_name].values
    df[column_name] = df[column_name].interpolate(method='linear')
    df.dropna(inplace=True)
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(minute_values.reshape(-1, 1))
    
    # Load pre-trained LSTM model
    model = load_model(f"c:/Users/稲葉湧大/Desktop/Python予測プロジェクト/Python_trade_learn/code/USDJPY/lstm_model_{column_name}.h5")
    
    # Predict missing values
    predictions = []
    if len(df) > window_size:
        for i in range(len(df) - window_size):
            input_seq = scaled_data[i:i+window_size].reshape(1, window_size, 1)
            predicted_value = model.predict(input_seq)
            predictions.append(predicted_value[0, 0])
            if len(predictions) == 16:  # Stop after 16 predictions
                break
    
    # Inverse transform predictions
    if predictions:
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        df.loc[df.index[window_size:window_size+len(predictions)], column_name] = predictions.flatten()
    
    return df

# Interpolate each column
for col in ['open', 'high', 'low', 'close']:
    hourly_df = interpolate_with_lstm(hourly_df, minute_df, col)

# Merge predicted close prices with hourly data
hourly_df['predicted_close'] = np.nan
hourly_df.loc[hourly_df.index[-len(predicted_df):], 'predicted_close'] = predicted_df['predicted_close'].values

# Interpolate between the last 6 data points of hourly_df and the first data point of predicted_df using LSTM
def interpolate_between_points(hourly_df, predicted_df, column_name, window_size=50):
    last_values = hourly_df[column_name].iloc[-6:].values
    first_predicted = predicted_df['predicted_close'].iloc[0]
    
    # Prepare data for LSTM
    data = np.concatenate([last_values, [first_predicted]])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    # Load pre-trained LSTM model
    model = load_model(f"c:/Users/稲葉湧大/Desktop/Python予測プロジェクト/Python_trade_learn/code/USDJPY/lstm_model_{column_name}.h5")
    
    # Predict missing values
    predictions = []
    for i in range(len(scaled_data) - window_size):
        input_seq = scaled_data[i:i+window_size].reshape(1, window_size, 1)
        predicted_value = model.predict(input_seq)
        predictions.append(predicted_value[0, 0])
    
    # Inverse transform predictions
    if predictions:
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        hourly_df.loc[hourly_df.index[-5:], column_name] = predictions.flatten()
    
    return hourly_df

for col in ['open', 'high', 'low', 'close']:
    hourly_df = interpolate_between_points(hourly_df, predicted_df, col)

# Save the merged dataframe
hourly_df.to_csv("/code/USDJPY/merged_interpolated_data.csv", index=False)

# Evaluate interpolation results
def evaluate_interpolation(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mae, rmse

# Plot results
plt.figure(figsize=(16, 8))
for col in ['open', 'high', 'low', 'close']:
    plt.plot(hourly_df['time'], hourly_df[col], label=f'Interpolated {col.capitalize()}')
plt.plot(hourly_df['time'], hourly_df['predicted_close'], label='Predicted Close', linestyle='--', color='red')
plt.legend()
plt.title('USD/JPY OHLC Price Interpolation and Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Evaluate and plot interpolation results
for col in ['open', 'high', 'low', 'close']:
    actual = hourly_df[col].iloc[-20:].values
    predicted = hourly_df['predicted_close'].iloc[-20:].values
    mae, rmse = evaluate_interpolation(actual, predicted)
    print(f"{col.capitalize()} - MAE: {mae}, RMSE: {rmse}")

    plt.figure(figsize=(16, 8))
    plt.plot(hourly_df['time'].iloc[-20:], actual, label='Actual')
    plt.plot(hourly_df['time'].iloc[-20:], predicted, label='Predicted', linestyle='--', color='red')
    plt.legend()
    plt.title(f'{col.capitalize()} Interpolation Evaluation')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()
