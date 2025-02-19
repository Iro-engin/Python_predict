import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

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

# 4. 特徴量の選択
features = df[['close', 'MA100', 'RSI']].dropna()

# 5. データの標準化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 6. PCAによる次元削減（2次元に削減）
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)

# 7. SVMによるクラスタリング
svm = OneClassSVM(kernel='rbf', gamma='auto')
svm.fit(pca_features)
df['cluster'] = svm.predict(pca_features)

# 8. クラスタリング結果のプロット
plt.figure(figsize=(16, 8))
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=df['cluster'], cmap='viridis', label='Clusters')
plt.title('SVM Clustering with PCA')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()

# 9. クラスタリング結果を元のデータフレームに追加
df['PCA1'] = pca_features[:, 0]
df['PCA2'] = pca_features[:, 1]

# 10. クラスタリング結果を保存
df.to_csv("/code/USDJPY/svm_clustering_results.csv", index=False)
