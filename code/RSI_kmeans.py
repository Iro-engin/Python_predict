import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

# 1. データ準備
df = pd.read_csv("/code/USDJPY/15min.csv")

# 2. RSIの計算 (期間100)
def calculate_rsi(df, period=100):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    return rsi

df['RSI'] = calculate_rsi(df)
df.dropna(inplace=True)

# 3. 多項式近似
poly_order = 3  # 多項式の次数
df['poly_fit'] = np.nan  # 多項式近似の結果を格納する列

for i in range(0, len(df), 100):  # 100個ずつ区切って処理
    segment = df.iloc[i:i+100]
    x = np.arange(len(segment))
    y = segment['RSI'].values
    coefficients = np.polyfit(x, y, poly_order)  # 多項式近似
    poly_function = np.poly1d(coefficients)  # 多項式関数を作成
    df.loc[segment.index, 'poly_fit'] = poly_function(x)  # 結果を格納

# 4. 関数クラスタリング (多項式近似の係数を使用)
n_clusters = 3  # クラスタ数
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# 多項式近似の係数を特徴量としてクラスタリング
coefficients_list = []
for i in range(0, len(df), 100):
  segment = df.iloc[i:i+100]
  x = np.arange(len(segment))
  y = segment['RSI'].values
  coefficients = np.polyfit(x, y, poly_order)
  coefficients_list.append(coefficients)

coefficients_array = np.array(coefficients_list)
df_coefficients = pd.DataFrame(coefficients_array)

df_coefficients['cluster'] = kmeans.fit_predict(df_coefficients)

# 各区間がどのクラスタに属するかを元のデータフレームにマージ
df['segment_id'] = (df.index // 100).astype(int)  # 各区間のID
df = pd.merge(df, df_coefficients[['cluster']], left_on='segment_id', right_index=True, how='left')

# NaN値を処理
df['cluster'] = df['cluster'].fillna(-1).astype(int)  # NaNを-1に置き換え、整数型に変換
colors = ['red', 'blue', 'green', 'gray']  # クラスタに対応する色 (-1用にgrayを追加)
df['color'] = df['cluster'].map(lambda x: colors[x])

# 5. 色分け
colors = ['red', 'blue', 'green']  # クラスタに対応する色
df['color'] = df['cluster'].map(lambda x: colors[x])

# 6. RSIと多項式近似のプロット
plt.figure(figsize=(16, 8))
#plt.plot(df['time'], df['RSI'], c='black', label='RSI')
#plt.plot(df['time'], df['poly_fit'], c='orange', label='Polynomial Fit')  # 多項式近似の結果をプロット
plt.scatter(df['time'], df['RSI'], c=df['color'], label='Clusters')  # クラスタリング結果を色分け表示
plt.title('RSI and Polynomial Fit with k-means Clustering')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()

# 7. 元データのプロット (クラスタリング反映)
'''
plt.figure(figsize=(16, 8))
plt.plot(df['time'], df['close'], c='black', label='Close Price')
plt.scatter(df['time'], df['close'], c=df['color'], label='Clusters')  # クラスタリング結果を色分け表示
plt.title('USD/JPY Close Price with k-means Clustering')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
'''