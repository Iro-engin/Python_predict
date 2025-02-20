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

# 3. 差分データの作成
df['diff'] = df['close'].diff()
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# 4. 50期間で区分
df['segment'] = (df.index // 50).astype(int)

# 5. 分布表の作成
distribution_table = df.groupby('segment')['diff'].describe()
print(distribution_table)

# 6. 特徴量の選択
features = df[['diff']]

# 7. データの標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 8. SVMによるクラスタリング
svm = OneClassSVM(kernel='rbf', gamma='auto')
svm.fit(scaled_features)
df['cluster'] = svm.predict(scaled_features)

'''
# 9. クラスタリング結果のプロット
plt.figure(figsize=(16, 8))
plt.scatter(df.index, df['close'], c=df['cluster'], cmap='viridis', label='Clusters')
plt.title('SVM Clustering')
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()
'''

# 10. クラスタリング結果を元のデータフレームに追加
df['scaled_close'] = scaled_features[:, 0]  # diffカラムのスケーリングされた値を追加

# 11. LSTMによる予測
window_size = 50  # 過去N期間のデータを使用
predictions = []

X = []
y = []
for i in range(window_size, len(scaled_features)):
    X.append(scaled_features[i - window_size:i])
    y.append(scaled_features[i, 0])  # diffカラムを予測対象とする
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
future_steps = 5
last_data = scaled_features[-window_size:].reshape(1, window_size, 1)
for _ in range(future_steps):
    predicted_value = model.predict(last_data)
    predictions.append(predicted_value[0, 0])
    last_data = np.concatenate([last_data[:, 1:, :], predicted_value.reshape(1, 1, 1)], axis=1)
    # 新規データを元のデータに追加
    new_diff = scaler.inverse_transform(np.concatenate([np.zeros((1, features.shape[1] - 1)), predicted_value.reshape(1, 1)], axis=1))[:, -1]
    new_row = pd.DataFrame({'diff': [new_diff[0]]})
    df = pd.concat([df, new_row], ignore_index=True)
    scaled_features = scaler.fit_transform(df[['diff']])

# 差分データを累積して元のデータに戻す
initial_value = df['close'].iloc[-future_steps - 1]
predicted_close = np.cumsum(predictions) + initial_value

# 元データの為替レートのデータとマージ
df['predicted_close'] = np.nan
df.loc[df.index[-future_steps:], 'predicted_close'] = predicted_close

# 予測データをCSVファイルに保存
predicted_df = df[['predicted_close']].dropna()
predicted_df.to_csv("/code/USDJPY/predicted_close.csv", index=False)

# 無効な値を確認して処理
# actual_futureをCSVファイルに保存
actual_future_df = pd.DataFrame({'actual_future': df['close'].iloc[-future_steps*2:-future_steps].values})
actual_future_df.to_csv("/code/USDJPY/actual_future.csv", index=False)
print(len(actual_future_df))

# 評価とプロットを行う対象をactual_future.csvとpredicted_close.csvに変更
actual_df = pd.read_csv("/code/USDJPY/actual_future.csv")
predicted_df = pd.read_csv("/code/USDJPY/predicted_close.csv")

actual_values = actual_df['actual_future'].values
predicted_values = predicted_df['predicted_close'].values

# 評価
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def sign_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))

mae = mean_absolute_error(actual_values, predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
r2 = r2_score(actual_values, predicted_values)
mape = mean_absolute_percentage_error(actual_values, predicted_values)
sign_acc = sign_accuracy(actual_values, predicted_values)
corr_coef = np.corrcoef(actual_values, predicted_values)[0, 1]

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R^2: {r2}")
print(f"MAPE: {mape}%")
print(f"Sign Accuracy: {sign_acc}")
print(f"Correlation Coefficient: {corr_coef}")

# プロット
plt.figure(figsize=(16, 8))
plt.plot(actual_values, label='Actual')
plt.plot(predicted_values, label='Predicted', color='red')
plt.legend()
plt.title('USD/JPY Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()

# 12. クラスタリング結果を保存
df.to_csv("/code/USDJPY/svm_clustering_results.csv", index=False)