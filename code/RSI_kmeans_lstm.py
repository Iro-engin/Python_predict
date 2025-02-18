import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 1. データ準備
df = pd.read_csv("/code/USDJPY/15min.csv")

# 2. RSIの計算 (期間14)
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    return rsi

df['RSI'] = calculate_rsi(df)
df.dropna(inplace=True)

# 3. 移動平均の計算
df['MA100'] = df['close'].rolling(window=100).mean()
df.fillna(method='bfill', inplace=True)

# 4. 多項式近似
poly_order = 3  # 多項式の次数
df['poly_fit'] = np.nan  # 多項式近似の結果を格納する列

for i in range(0, len(df), 100):  # 100個ずつ区切って処理
    segment = df.iloc[i:i+100]
    x = np.arange(len(segment) * 3)  # xの長さを3倍にする
    y = segment[['RSI', 'MA100', 'close']].values.flatten()
    coefficients = np.polyfit(x, y, poly_order)  # 多項式近似
    poly_function = np.poly1d(coefficients)  # 多項式関数を作成
    df.loc[segment.index, 'poly_fit'] = poly_function(np.arange(len(segment)))  # 結果を格納

# 5. エルボー法による適切なクラスタ数の計算
coefficients_list = []
for i in range(0, len(df), 100):
    segment = df.iloc[i:i+100]
    x = np.arange(len(segment) * 3)  # xの長さを3倍にする
    y = segment[['RSI', 'MA100', 'close']].values.flatten()
    coefficients = np.polyfit(x, y, poly_order)
    coefficients_list.append(coefficients)

coefficients_array = np.array(coefficients_list)
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(coefficients_array)
    sse.append(kmeans.inertia_)

# エルボー法のプロット
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.grid(True)
plt.show()

# 適切なクラスタ数を選択（エルボー法の結果に基づく）
optimal_clusters = 6  # ここでは仮に6を選択（プロットを見て適切な値に変更）

# 6. 関数クラスタリング (多項式近似の係数を使用)
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0)
df_coefficients = pd.DataFrame(coefficients_array)
df_coefficients['cluster'] = kmeans.fit_predict(df_coefficients)

# 各区間がどのクラスタに属するかを元のデータフレームにマージ
df['segment_id'] = (df.index // 100).astype(int)  # 各区間のID
df = pd.merge(df, df_coefficients[['cluster']], left_on='segment_id', right_index=True, how='left')

# NaN値を処理
df['cluster'] = df['cluster'].fillna(-1).astype(int)  # NaNを-1に置き換え、整数型に変換
colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'gray']  # クラスタに対応する色 (-1用にgrayを追加)
df['color'] = df['cluster'].map(lambda x: colors[x])

# 7. クラスタごとのXY座標を追加
df['X'] = df.index
df['Y'] = df['RSI']

# クラスタごとのXY座標をリストに格納
cluster_coords = {i: df[df['cluster'] == i][['X', 'Y']].values.tolist() for i in range(optimal_clusters)}

# クラスタの出現パターンを計算
transition_matrix = np.zeros((optimal_clusters, optimal_clusters))
for i in range(1, len(df)):
    prev_cluster = df['cluster'].iloc[i-1]
    curr_cluster = df['cluster'].iloc[i]
    transition_matrix[prev_cluster, curr_cluster] += 1

# 各クラスタから次のクラスタへの遷移確率を計算
transition_probabilities = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

# 8. クラスタごとにLSTMで学習および予測
for cluster, coords in cluster_coords.items():
    cluster_df = df[df['cluster'] == cluster]
    
    # データ前処理
    features = cluster_df[['close', 'MA100', 'RSI']].values
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

    # LSTMモデル構築
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(window_size, X.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 学習
    model.fit(X, y, epochs=50, batch_size=32)

    # 予測
    future_steps = 30
    last_data = scaled_features[-window_size:].reshape(1, window_size, X.shape[2])
    predictions = []
    current_cluster = cluster
    for _ in range(future_steps):
        predicted_price = model.predict(last_data)
        predictions.append(predicted_price[0, 0])
        predicted_price_full = np.concatenate([predicted_price, np.zeros((1, X.shape[2] - 1))], axis=1)
        predicted_price_full = predicted_price_full.reshape(1, 1, X.shape[2])
        last_data = np.concatenate([last_data[:, 1:, :], predicted_price_full], axis=1)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(np.concatenate([predictions, np.zeros((future_steps, features.shape[1] - 1))], axis=1))[:, 0]

    # 最初の10データで評価
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    actual_future = cluster_df['close'].values[-future_steps:]
    initial_predictions = predictions[:10]
    initial_actual = actual_future[:10]
    mae = mean_absolute_error(initial_actual, initial_predictions)
    rmse = np.sqrt(mean_squared_error(initial_actual, initial_predictions))
    r2 = r2_score(initial_actual, initial_predictions)
    mape = mean_absolute_percentage_error(initial_actual, initial_predictions)

    print(f"Cluster {cluster} - Initial 10 Data - MAE: {mae}, RMSE: {rmse}, R^2: {r2}, MAPE: {mape}%")

    # 最初の10データの評価が最も高かったものを選択
    best_model = model
    best_predictions = initial_predictions
    best_actual = initial_actual
    best_mae = mae
    best_rmse = rmse
    best_r2 = r2
    best_mape = mape

    # 残り20データの予測
    remaining_predictions = predictions[10:]
    remaining_actual = actual_future[10:]
    mae_remaining = mean_absolute_error(remaining_actual, remaining_predictions)
    rmse_remaining = np.sqrt(mean_squared_error(remaining_actual, remaining_predictions))
    r2_remaining = r2_score(remaining_actual, remaining_predictions)
    mape_remaining = mean_absolute_percentage_error(remaining_actual, remaining_predictions)

    print(f"Cluster {cluster} - Remaining 20 Data - MAE: {mae_remaining}, RMSE: {rmse_remaining}, R^2: {r2_remaining}, MAPE: {mape_remaining}%")

    # プロット
    plt.figure(figsize=(16, 8))
    plt.plot(cluster_df['time'].iloc[-len(actual_future):], actual_future, label='Actual')
    plt.plot(cluster_df['time'].iloc[-len(actual_future):], predictions, label='Predicted', color='red')
    plt.legend()
    plt.title(f'USD/JPY Price Prediction for Cluster {cluster}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()