import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def create_lstm_model(df, window_size=20, epochs=50, batch_size=32):
    """
    LSTMモデルを作成し、学習する関数

    Args:
        df (pd.DataFrame): 学習データ
        window_size (int): 過去何日分のデータを使用するか
        epochs (int): 学習回数
        batch_size (int): バッチサイズ

    Returns:
        tuple: 学習済みモデルとスケーラー
    """

    close_prices = df['close'].values.reshape(-1, 1)

    # スケーリング
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_prices)

    # データセット作成
    X = []
    y = []
    for i in range(window_size, len(scaled_close)):
        X.append(scaled_close[i - window_size:i])
        y.append(scaled_close[i])
    X = np.array(X)
    y = np.array(y)

    # モデル構築
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 学習
    model.fit(X, y, epochs=epochs, batch_size=batch_size)

    return model, scaler

def interpolate_missing_data(df, column_name, method='linear'):
    """欠損値を補完する関数 (1次元スプライン補間)"""
    if method == 'linear':
        df[column_name] = df[column_name].interpolate(method='linear')
    elif method == 'spline':  # スプライン補間を追加
        x = np.arange(len(df))
        y = df[column_name].values
        missing_idx = np.isnan(y)
        known_idx = ~missing_idx
        f = interp1d(x[known_idx], y[known_idx], kind='cubic', fill_value="extrapolate") # 3次スプライン補間
        df[column_name] = f(x)
    else:
        raise ValueError("Invalid interpolation method. Choose 'linear' or 'spline'.")
    return df

def predict_future(model, data, scaler, window_size, future_steps):
    """LSTMモデルを用いて未来のデータを予測する関数 (過去数点の平均値を考慮)"""
    predictions = []
    # 過去のデータを複数点取得 (例: 過去5点)
    last_data_points = data[-5:]  # 形状を (5, window_size, 1) にする

    for _ in range(future_steps):
        # 過去のデータ点の平均値を計算
        current_data = np.mean(last_data_points, axis=0).reshape(1, window_size, 1)
        predicted_price = model.predict(current_data)
        predictions.append(predicted_price[0, 0])

        # 予測値を過去のデータに追加 (次の予測のために)
        # last_data_points を (4, 20, 1) に、predicted_price を (1, 20, 1) に変換
        last_data_points = np.concatenate([last_data_points[1:], np.tile(predicted_price.reshape(1, 1, 1), (1, window_size, 1))], axis=0)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    return predictions

# 1. データ読み込み
df = pd.read_csv("/code/USDJPY/1hour.csv")

# 2. 欠損値補完
df = interpolate_missing_data(df, 'close', method='spline')  # スプライン補間を使用

# 3. LSTMモデル学習
model, scaler = create_lstm_model(df)  # ハイパーパラメータチューニング、検証データ分割などを検討

# 4. モデル保存
model.save("/code/USDJPY/lstm_model.h5")

# 5. 未来予測
window_size = 50
future_steps = 100
X = []
close_prices = df['close'].values.reshape(-1, 1)
scaled_close = scaler.transform(close_prices)
for i in range(window_size, len(scaled_close)):
    X.append(scaled_close[i - window_size:i])
X = np.array(X)

future_predictions = predict_future(model, X, scaler, window_size, future_steps)

# 6. 評価
actual_future = close_prices[-future_steps:]
mae = mean_absolute_error(actual_future, future_predictions)
rmse = np.sqrt(mean_squared_error(actual_future, future_predictions))
r2 = r2_score(actual_future, future_predictions)

print(f"平均絶対誤差 (MAE): {mae}")
print(f"二乗平均平方根誤差 (RMSE): {rmse}")
print(f"決定係数 (R²): {r2}")

# 7. プロット
plt.figure(figsize=(14, 7))
plt.plot(df['time'], df['close'], label='Original Close Price')
future_time = pd.date_range(start=df['time'].iloc[-1], periods=future_steps + 1, freq='15min')[1:]
plt.plot(future_time, future_predictions, label='Predicted Close Price', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('USDJPY Close Price Prediction')
plt.legend()
plt.grid()
plt.show()