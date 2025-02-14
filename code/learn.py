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

# 3. 移動平均の計算
df['MA100'] = df['close'].rolling(window=100).mean()

# MACDをコメントアウト
# df['EMA6'] = df['close'].ewm(span=6, adjust=False).mean()
# df['EMA13'] = df['close'].ewm(span=13, adjust=False).mean()
# df['MACD'] = df['EMA6'] - df['EMA13']
# df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# 欠損値を補完
df.fillna(method='bfill', inplace=True)

# 4. データ前処理
# 特徴量からMACDとSignalを削除
features = df[['close', 'MA100']].values
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

window_size = 20  # 過去20期間分のデータを使用
X = []
y = []
for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i])
    y.append(scaled_features[i, 0])  # close価格を予測対象とする
X = np.array(X)
y = np.array(y)

# 5. LSTMモデル構築
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 6. 学習
model.fit(X, y, epochs=500, batch_size=32)

# 7. 予測
future_steps = 10
last_data = scaled_features[-window_size:].reshape(1, window_size, X.shape[2])
predictions = []
for _ in range(future_steps):
    predicted_price = model.predict(last_data)
    predictions.append(predicted_price[0, 0])
    # predicted_priceの形状を(1, 1, X.shape[2])に変更
    predicted_price_full = np.concatenate([predicted_price, np.zeros((1, X.shape[2] - 1))], axis=1)
    predicted_price_full = predicted_price_full.reshape(1, 1, X.shape[2])
    last_data = np.concatenate([last_data[:, 1:, :], predicted_price_full], axis=1)

predictions = np.array(predictions).reshape(-1, 1)
# 予測値を逆スケーリングする際、元の特徴量の数に合わせて0を追加
predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((future_steps, features.shape[1] - 1))], axis=1))[:, 0]

# 8. 評価
actual_future = df['close'].values[-future_steps:]
mae = mean_absolute_error(actual_future, predictions)
rmse = np.sqrt(mean_squared_error(actual_future, predictions))
r2 = r2_score(actual_future, predictions)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")

# 9. プロット
plt.figure(figsize=(16, 8))
plt.plot(df['time'].iloc[-len(actual_future):], actual_future, label='Actual')
plt.plot(df['time'].iloc[-len(actual_future):], predictions, label='Predicted', color='red')
plt.legend()
plt.title('USD/JPY Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# 10. モデル保存 (オプション)
# model.save("/code/USDJPY/lstm_model.h5")