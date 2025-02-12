import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def load_and_prepare_data(csv_file_path, window_size):
    """
    データを読み込み、LSTMモデル用に準備する関数

    Args:
        csv_file_path (str): CSVファイルのパス
        window_size (int): 過去何日分のデータを使用するか

    Returns:
        tuple: スケーリングされたデータ、スケーラー、元のデータフレーム
    """
    # CSVファイルを取得
    df = pd.read_csv(csv_file_path)

    # 時間をdatetime型に変換
    df['time'] = pd.to_datetime(df['time'])

    # closeカラムを取得
    close_prices = df['close'].values.reshape(-1, 1)

    # スケーリング
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_prices)

    # データセット作成
    X = []
    for i in range(window_size, len(scaled_close)):
        X.append(scaled_close[i - window_size:i])
    X = np.array(X)

    return X, scaler, df

def predict_future(model, data, scaler, window_size, future_steps):
    """
    LSTMモデルを用いて未来のデータを予測する関数

    Args:
        model (keras.Model): 学習済みLSTMモデル
        data (np.array): スケーリングされたデータ
        scaler (MinMaxScaler): スケーラー
        window_size (int): 過去何日分のデータを使用するか
        future_steps (int): 予測する未来のステップ数

    Returns:
        np.array: 予測された未来のデータ
    """
    predictions = []
    current_data = data[-1]

    for _ in range(future_steps):
        current_data = current_data.reshape(1, window_size, 1)
        predicted_price = model.predict(current_data)
        predictions.append(predicted_price[0, 0])
        current_data = np.append(current_data[:, 1:, :], [[predicted_price]], axis=1)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)

    return predictions

def main():
    # パラメータ設定
    csv_file_path = "/code/USDJPY/1hour.csv"
    model_path = "/code/USDJPY/lstm_model.h5"
    window_size = 50
    future_steps = 100

    # データの読み込みと準備
    X, scaler, df = load_and_prepare_data(csv_file_path, window_size)

    # モデルの読み込み
    model = load_model(model_path)

    # 未来のデータを予測
    future_predictions = predict_future(model, X, scaler, window_size, future_steps)

    # モデルの評価
    actual_future = df['close'].values[-future_steps:]
    r2 = r2_score(actual_future, future_predictions)
    mae = mean_absolute_error(actual_future, future_predictions)
    rmse = np.sqrt(mean_squared_error(actual_future, future_predictions))

    print(f"決定係数 (R²): {r2}")
    print(f"平均絶対誤差 (MAE): {mae}")
    print(f"二乗平均平方根誤差 (RMSE): {rmse}")

    # 予測結果のプロット
    plt.figure(figsize=(14, 7))
    plt.plot(df['time'], df['close'], label='Original Close Price')
    future_time = pd.date_range(start=df['time'].iloc[-1], periods=future_steps + 1, freq='H')[1:]
    plt.plot(future_time, future_predictions, label='Predicted Close Price', color='red')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('USDJPY Close Price Prediction')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()