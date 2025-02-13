import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# 1. データ準備
df = pd.read_csv("/code/USDJPY/1hour.csv")

# 2. 欠損値補間 (三次スプライン補間)
def spline_interpolation(df, column_name):
    x = np.arange(len(df))
    y = df[column_name].values
    missing_idx = df[column_name].isnull()
    f = interp1d(x[~missing_idx], y[~missing_idx], kind='cubic', fill_value="extrapolate")
    df.loc[missing_idx, column_name] = f(x[missing_idx])
    return df

df = spline_interpolation(df, 'close')

'''
# 補間後のデータをプロット
plt.figure(figsize=(16, 8))
plt.plot(df['time'], df['close'], label='Interpolated Close Price')
plt.title('USD/JPY Close Price After Spline Interpolation')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
'''

# 3. データ前処理
close_prices = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_close = scaler.fit_transform(close_prices)

window_size = 20  # 過去20期間分のデータを使用
X = []
y = []
for i in range(window_size, len(scaled_close)):
    X.append(scaled_close[i - window_size:i])
    y.append(scaled_close[i])
X = np.array(X)
y = np.array(y)

# 4. LSTMモデル構築
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 5. 学習
model.fit(X, y, epochs=100, batch_size=32)

# 6. 予測
future_steps = 100
last_data = scaled_close[-window_size:].reshape(1, window_size, 1)  # 最新のwindow_size分のデータ
predictions = []
for _ in range(future_steps):
    predicted_price = model.predict(last_data)
    predictions.append(predicted_price[0, 0])
    last_data = np.concatenate([last_data[:, 1:, :], predicted_price.reshape(1, 1, 1)], axis=1)

predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)

# 7. 評価
actual_future = close_prices[-future_steps:]  # 予測期間の実際の値
mae = mean_absolute_error(actual_future, predictions)
rmse = np.sqrt(mean_squared_error(actual_future, predictions))
r2 = r2_score(actual_future, predictions)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# 8. プロット
plt.figure(figsize=(16, 8))
plt.plot(df['time'].iloc[-len(actual_future):], actual_future, label='Actual')  # 実際の値
plt.plot(df['time'].iloc[-len(actual_future):], predictions, label='Predicted', color='red')  # 予測値
plt.legend()
plt.title('USD/JPY Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# 9. モデル保存 (オプション)
# model.save("/code/USDJPY/lstm_model.h5")