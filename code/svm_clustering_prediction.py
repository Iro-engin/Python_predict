import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mplfinance as mpf
from scipy.interpolate import interp1d

# 1. データ準備
df = pd.read_csv("/code/USDJPY/15min.csv")

# 2. 欠損値補間 (三次スプライン補間)
def spline_interpolation(df, column_name):
    x = np.arange(len(df))
    y = df[column_name].values
    missing_idx = df[column_name].isnull()
    f = interp1d(x[~missing_idx], y[~missing_idx], kind='cubic', fill_value="extrapolate")
    df.loc[missing_idx, column_name] = f(x[missing_idx])
    return df

for col in ['open', 'high', 'low', 'close']:
    df = spline_interpolation(df, col)

# 3. 特徴量の選択
features = df[['open', 'high', 'low', 'close']].dropna()

# 4. データの標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 5. SVMによるクラスタリング
svm = OneClassSVM(kernel='rbf', gamma='auto')
svm.fit(scaled_features)
df['cluster'] = svm.predict(scaled_features)

# 6. クラスタリング結果のプロット
plt.figure(figsize=(16, 8))
plt.scatter(df.index, df['close'], c=df['cluster'], cmap='viridis', label='Clusters')
plt.title('SVM Clustering')
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# 7. クラスタリング結果を元のデータフレームに追加
df['scaled_close'] = scaled_features[:, 3]  # closeカラムのスケーリングされた値を追加

# 8. LSTMによる予測
window_size = 50  # 過去N期間分のデータを使用
predictions = {}

for col in ['open', 'high', 'low', 'close']:
    X = []
    y = []
    col_index = features.columns.get_loc(col)
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i - window_size:i, col_index].reshape(-1, 1))
        y.append(scaled_features[i, col_index])  # 各カラムを予測対象とする
    X = np.array(X)
    y = np.array(y)

    # LSTMモデル構築
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 学習
    model.fit(X, y, epochs=100, batch_size=32)

    # 予測
    future_steps = 20
    last_data = scaled_features[-window_size:, col_index].reshape(1, window_size, 1)
    col_predictions = []
    for _ in range(future_steps):
        predicted_value = model.predict(last_data)
        col_predictions.append(predicted_value[0, 0])
        last_data = np.concatenate([last_data[:, 1:, :], predicted_value.reshape(1, 1, 1)], axis=1)

    col_predictions = np.array(col_predictions).reshape(-1, 1)
    col_predictions = scaler.inverse_transform(np.concatenate([np.zeros((future_steps, features.shape[1] - 1)), col_predictions], axis=1))[:, -1]
    predictions[col] = col_predictions

# 差分データを累積して元のデータに戻す
initial_values = df[['open', 'high', 'low', 'close']].iloc[-future_steps - 1]
predicted_ohlc = {col: np.cumsum(predictions[col]) + initial_values[col] for col in ['open', 'high', 'low', 'close']}

# 評価
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

actual_future = df[['open', 'high', 'low', 'close']].values[-future_steps:]
mae = {col: mean_absolute_error(actual_future[:, i], predicted_ohlc[col]) for i, col in enumerate(['open', 'high', 'low', 'close'])}
rmse = {col: np.sqrt(mean_squared_error(actual_future[:, i], predicted_ohlc[col])) for i, col in enumerate(['open', 'high', 'low', 'close'])}
r2 = {col: r2_score(actual_future[:, i], predicted_ohlc[col]) for i, col in enumerate(['open', 'high', 'low', 'close'])}
mape = {col: mean_absolute_percentage_error(actual_future[:, i], predicted_ohlc[col]) for i, col in enumerate(['open', 'high', 'low', 'close'])}

for col in ['open', 'high', 'low', 'close']:
    print(f"{col.upper()} - MAE: {mae[col]}, RMSE: {rmse[col]}, R^2: {r2[col]}, MAPE: {mape[col]}%")

# プロット
predicted_df = pd.DataFrame(predicted_ohlc)
predicted_df['time'] = pd.to_datetime(df['time'].iloc[-future_steps:].values)
predicted_df.set_index('time', inplace=True)

actual_df = df[['time', 'open', 'high', 'low', 'close']].iloc[-future_steps:]
actual_df['time'] = pd.to_datetime(actual_df['time'])
actual_df.set_index('time', inplace=True)

fig, ax = plt.subplots(figsize=(16, 8))
mpf.plot(actual_df, type='candle', ax=ax, style='charles', title='Actual vs Predicted OHLC', ylabel='Price')
mpf.plot(predicted_df, type='candle', ax=ax, style='charles', ylabel='Price', secondary_y=True)
plt.show()

# 9. クラスタリング結果を保存
df.to_csv("/code/USDJPY/svm_clustering_results.csv", index=False)
