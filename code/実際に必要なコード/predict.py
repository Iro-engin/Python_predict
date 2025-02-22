import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# データ読み込み
predicted_next_hour_df = pd.read_csv("/code/USDJPY/predicted_next_hour.csv")
hourly_df = pd.read_csv("/code/USDJPY/1hour.csv")

# 補完対象カラム
columns = ['open', 'high', 'low', 'close']

for column in columns:
    # 最新データを抽出
    last_hourly_data = hourly_df[column].iloc[-1]
    last_predicted_data = predicted_next_hour_df[f'predicted_{column}'].iloc[-1]

    # LSTM入力データ準備
    data = np.array([last_hourly_data, last_predicted_data]).reshape(-1, 1)

    # スケーリング
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # モデル読み込み
    model_path = f"c:/Users/稲葉湧大/Desktop/Python予測プロジェクト/Python_trade_learn/code/USDJPY/lstm_model_{column}.h5"
    model = load_model(model_path)

    # 予測
    input_seq = scaled_data.reshape(1, 2, 1)
    predicted_value = model.predict(input_seq)

    # 逆スケーリング
    predicted_value = scaler.inverse_transform(predicted_value)

    # 結果表示
    print(f"{column.capitalize()} 補完値: {predicted_value[0, 0]}")

    # 1時間足データ更新
    hourly_df[column].iloc[-1] = predicted_value[0, 0]

# 更新された1時間足データを表示
print("\n更新後1時間足データ:")
print(hourly_df.tail())

# 必要であれば、更新されたデータをCSVファイルに保存
# hourly_df.to_csv("updated_1hour.csv", index=False)