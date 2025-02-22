import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
predicted_df = pd.read_csv("/code/USDJPY/predicted_next_hour.csv")
hourly_df = pd.read_csv("/code/USDJPY/1hour.csv")
minute_df = pd.read_csv("/code/USDJPY/15min.csv")

def preprocess_data(df):
    df['diff_open'] = df['open'].diff()
    df['diff_high'] = df['high'].diff()
    df['diff_low'] = df['low'].diff()
    df['diff_close'] = df['close'].diff()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = preprocess_data(minute_df)

# データの分割
train_size = len(df) - 5
train_df = df[:train_size]
test_df = df[train_size:]

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

# Build LSTM model for 5-step prediction
def build_lstm_model(window_size, predict_size):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(predict_size))  # Output 5 values (for 5 steps)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Prepare the data for prediction
window_size = 50
predict_size = 5

# Train and predict for each OHLC component
def train_and_predict(scaled_data, window_size, predict_size):
    X = []
    y = []
    for i in range(window_size, len(scaled_data) - predict_size + 1):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i:i+predict_size])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    # y = y.reshape(y.shape[0], 5, 1)  # この行は削除
    print(X.shape)
    print(y.shape)

    model = build_lstm_model(window_size, predict_size)
    model.summary()
    model.fit(X, y, epochs=50, batch_size=32)
    last_data = scaled_data[-window_size:].reshape(1, window_size, 1)
    predicted_diff = model.predict(last_data)
    return model, predicted_diff

# 学習データでモデルを学習
model_open, predicted_open_diff_train = train_and_predict(scaled_open[:train_size], window_size, predict_size) # 修正点
model_high, predicted_high_diff_train = train_and_predict(scaled_high[:train_size], window_size, predict_size) # 修正点
model_low, predicted_low_diff_train = train_and_predict(scaled_low[:train_size], window_size, predict_size) # 修正点
model_close, predicted_close_diff_train = train_and_predict(scaled_close[:train_size], window_size, predict_size) # 修正点
                                                          
# テストデータで予測
last_open_test = scaled_open[train_size - window_size:train_size].reshape(1, window_size, 1)
last_high_test = scaled_high[train_size - window_size:train_size].reshape(1, window_size, 1)
last_low_test = scaled_low[train_size - window_size:train_size].reshape(1, window_size, 1)
last_close_test = scaled_close[train_size - window_size:train_size].reshape(1, window_size, 1)

predicted_open_diff_test = model_open.predict(last_open_test) # 修正点
predicted_high_diff_test = model_high.predict(last_high_test) # 修正点
predicted_low_diff_test = model_low.predict(last_low_test) # 修正点
predicted_close_diff_test = model_close.predict(last_close_test) # 修正点

# Inverse transform the predictions (for each of the 5 steps)
predicted_open_diff_train = scaler_open.inverse_transform(predicted_open_diff_train.reshape(-1, 1))
predicted_high_diff_train = scaler_high.inverse_transform(predicted_high_diff_train.reshape(-1, 1))
predicted_low_diff_train = scaler_low.inverse_transform(predicted_low_diff_train.reshape(-1, 1))
predicted_close_diff_train = scaler_close.inverse_transform(predicted_close_diff_train.reshape(-1, 1))

predicted_open_diff_test = scaler_open.inverse_transform(predicted_open_diff_test.reshape(-1, 1)).flatten()
predicted_high_diff_test = scaler_high.inverse_transform(predicted_high_diff_test.reshape(-1, 1)).flatten()
predicted_low_diff_test = scaler_low.inverse_transform(predicted_low_diff_test.reshape(-1, 1)).flatten()
predicted_close_diff_test = scaler_close.inverse_transform(predicted_close_diff_test.reshape(-1, 1)).flatten()

# Calculate the predicted OHLC prices for the next 5 steps
last_open_price = test_df['open'].iloc[0]
last_high_price = test_df['high'].iloc[0]
last_low_price = test_df['low'].iloc[0]
last_close_price = test_df['close'].iloc[0]

predicted_open = last_open_price + predicted_open_diff_test
predicted_high = last_high_price + predicted_high_diff_test
predicted_low = last_low_price + predicted_low_diff_test
predicted_close = last_close_price + predicted_close_diff_test

print(f"Predicted OHLC for the next 5 15min steps: Open={predicted_open}, High={predicted_high}, Low={predicted_low}, Close={predicted_close}")

# 評価
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]  # 相関係数
    return mae, rmse, r2, corr_coef

# 実際の値
actual_open = test_df['open'].values
actual_high = test_df['high'].values
actual_low = test_df['low'].values
actual_close = test_df['close'].values

# 評価指標の計算
mae_open, rmse_open, r2_open, corr_coef_open = evaluate(actual_open, predicted_open.flatten())
mae_high, rmse_high, r2_high, corr_coef_high = evaluate(actual_high, predicted_high.flatten())
mae_low, rmse_low, r2_low, corr_coef_low = evaluate(actual_low, predicted_low.flatten())
mae_close, rmse_close, r2_close, corr_coef_close = evaluate(actual_close, predicted_close.flatten())

# 評価結果の表示
print("Open:")
print(f"  MAE: {mae_open:.4f}")
print(f"  RMSE: {rmse_open:.4f}")
print(f"  R^2: {r2_open:.4f}")
print(f"  相関係数: {corr_coef_open:.4f}")

print("High:")
print(f"  MAE: {mae_high:.4f}")
print(f"  RMSE: {rmse_high:.4f}")
print(f"  R^2: {r2_high:.4f}")
print(f"  相関係数: {corr_coef_high:.4f}")

print("Low:")
print(f"  MAE: {mae_low:.4f}")
print(f"  RMSE: {rmse_low:.4f}")
print(f"  R^2: {r2_low:.4f}")
print(f"  相関係数: {corr_coef_low:.4f}")

print("Close:")
print(f"  MAE: {mae_close:.4f}")
print(f"  RMSE: {rmse_close:.4f}")
print(f"  R^2: {r2_close:.4f}")
print(f"  相関係数: {corr_coef_close:.4f}")