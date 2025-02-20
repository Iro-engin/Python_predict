import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.interpolate import interp1d

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

for col in ['open', 'high', 'low', 'close']:
    df = spline_interpolation(df, col)

# 3. 差分データの作成
for col in ['open', 'high', 'low', 'close']:
    df[f'{col}_diff'] = df[col].diff()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 4. 50期間で区分
df['segment'] = (df.index // 50).astype(int)

# 5. 分布表の作成
distribution_table = df.groupby('segment').describe()
print(distribution_table)

# 6. 特徴量の選択
features = df[['open_diff', 'high_diff', 'low_diff', 'close_diff']]

# 7. データの標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 8. SVMによるクラスタリング
svm = OneClassSVM(kernel='rbf', gamma='auto')
svm.fit(scaled_features)
df['cluster'] = svm.predict(scaled_features)

# 9. クラスタリング結果のプロット
plt.figure(figsize=(16, 8))
plt.scatter(df.index, df['close'], c=df['cluster'], cmap='viridis', label='Clusters')
plt.title('SVM Clustering')
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# 10. クラスタリング結果を元のデータフレームに追加
for i, col in enumerate(['open_diff', 'high_diff', 'low_diff', 'close_diff']):
    df[f'scaled_{col}'] = scaled_features[:, i]

# 11. LSTMによる予測
window_size = 50  # 過去N期間のデータを使用
future_steps = 5

def lstm_prediction(scaled_features, target_col_index):
    predictions = []
    X = []
    y = []
    for i in range(window_size, len(scaled_features)):
        X.append(scaled_features[i - window_size:i])
        y.append(scaled_features[i, target_col_index])
    X = np.array(X)
    y = np.array(y)

    # LSTMモデル構築
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, 4)))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 学習
    model.fit(X, y, epochs=75, batch_size=32)

    # 予測
    last_data = scaled_features[-window_size:].reshape(1, window_size, 4)
    for _ in range(future_steps):
        predicted_value = model.predict(last_data)
        predictions.append(predicted_value[0, 0])
        new_data = np.zeros((1, 1, 4))
        new_data[0, 0, target_col_index] = predicted_value[0, 0]
        last_data = np.concatenate([last_data[:, 1:, :], new_data], axis=1)
    return predictions

# 各カラムの予測
predicted_data = {}
for i, col in enumerate(['open_diff', 'high_diff', 'low_diff', 'close_diff']):
    predictions = lstm_prediction(scaled_features, i)
    initial_value = df[col.split('_')[0]].iloc[-future_steps - 1]
    predicted_data[col.split('_')[0]] = np.cumsum(predictions) + initial_value

# 元データの為替レートのデータとマージ
for col, predicted_close in predicted_data.items():
    df[f'predicted_{col}'] = np.nan
    df.loc[df.index[-future_steps:], f'predicted_{col}'] = predicted_close

# 予測データをCSVファイルに保存
predicted_df = df[[f'predicted_{col}' for col in predicted_data.keys()]].dropna()
predicted_df.to_csv("/code/USDJPY/predicted_close.csv", index=False)

# 無効な値を確認して処理
actual_future = df[['open', 'high', 'low', 'close']].iloc[-future_steps:].values
predicted_close = np.column_stack([predicted_data[col] for col in ['open', 'high', 'low', 'close']])
predicted_close = np.nan_to_num(predicted_close)

# NaNを含む行を削除
valid_idx = ~np.isnan(actual_future).any(axis=1) & ~np.isnan(predicted_close).any(axis=1)
actual_future = actual_future[valid_idx]
predicted_close = predicted_close[valid_idx]

# actual_futureをCSVファイルに保存
actual_future_df = pd.DataFrame(actual_future, columns=['actual_open', 'actual_high', 'actual_low', 'actual_close'])
actual_future_df.to_csv("/code/USDJPY/actual_future.csv", index=False)
print(len(actual_future_df))

# 評価
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def sign_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

for i, col in enumerate(['open', 'high', 'low', 'close']):
    mae = mean_absolute_error(actual_future[:, i], predicted_close[:, i])
    rmse = np.sqrt(mean_squared_error(actual_future[:, i], predicted_close[:, i]))
    r2 = r2_score(actual_future[:, i], predicted_close[:, i])
    mape = mean_absolute_percentage_error(actual_future[:, i], predicted_close[:, i])
    sign_acc = sign_accuracy(actual_future[:, i], predicted_close[:, i])
    corr_coef = np.corrcoef(actual_future[:, i], predicted_close[:, i])[0, 1]

    print(f"{col.capitalize()} MAE: {mae}")
    print(f"{col.capitalize()} RMSE: {rmse}")
    print(f"{col.capitalize()} R^2: {r2}")
    print(f"{col.capitalize()} MAPE: {mape}%")
    print(f"{col.capitalize()} Sign Accuracy: {sign_acc}")
    print(f"{col.capitalize()} Correlation Coefficient: {corr_coef}")

# プロット
plt.figure(figsize=(16, 8))
for i, col in enumerate(['open', 'high', 'low', 'close']):
    plt.plot(df['time'].iloc[-future_steps:], actual_future[:, i], label=f'Actual {col.capitalize()}')
    plt.plot(df['time'].iloc[-future_steps:], predicted_close[:, i], label=f'Predicted {col.capitalize()}', linestyle='--')
plt.legend()
plt.title('USD/JPY OHLC Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# 12. クラスタリング結果を保存
df.to_csv("/code/USDJPY/svm_clustering_results.csv", index=False)
