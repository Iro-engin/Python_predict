import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.interpolate import griddata

def create_lstm_model(df, window_size=50, epochs=50, batch_size=32):
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
    """欠損値を補完する関数 (バイキュービック法をgriddataで実装)"""

    if method == 'linear':
        df[column_name] = df[column_name].interpolate(method='linear')
    elif method == 'bicubic':
        # griddataを使用
        x = np.arange(len(df))
        y = df[column_name].values

        # 欠損値のインデックスを取得
        missing_idx = np.isnan(y)
        known_idx = ~missing_idx

        # バイキュービック補間
        y_interp = griddata(x[known_idx], y[known_idx], x, method='cubic')

        # 補間結果をデータフレームに反映
        df[column_name] = y_interp
    else:
        raise ValueError("Invalid interpolation method. Choose 'linear' or 'bicubic'.")

    return df

# 1. データ読み込み
df = pd.read_csv("/code/USDJPY/15min.csv")

# 2. 欠損値の確認
print("Missing values before interpolation:\n", df.isnull().sum())

# 3. 欠損値を補完 (バイキュービック法)
df = interpolate_missing_data(df, 'close', method='bicubic')

# 4. 欠損値の再確認
print("Missing values after interpolation:\n", df.isnull().sum())

# 5. LSTMモデル学習
model, scaler = create_lstm_model(df)

# 6. モデル保存
model.save("/code/USDJPY/lstm_model.h5")

print("LSTMモデルの学習と保存が完了しました。")