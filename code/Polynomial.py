import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('start Polynomial.py')

# CSVファイルを取得
csv_file_path = "/code/USDJPY/1hour.csv"
rates_df = pd.read_csv(csv_file_path)

# 時間をdatetime型に変換
rates_df['time'] = pd.to_datetime(rates_df['time'])

# closeカラムを取得
close_prices = rates_df['close'].values

# 多項式近似を用いてデータを平滑化
degree = 50  # 多項式の次数
x = np.arange(len(close_prices))
poly_coeffs = np.polyfit(x, close_prices, degree)
poly_fit = np.polyval(poly_coeffs, x)

# プロット
plt.figure(figsize=(14, 7))
plt.plot(rates_df['time'], close_prices, label='Original Close Price')
plt.plot(rates_df['time'], poly_fit, label='Polynomial Fit', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('USDJPY Close Price and Polynomial Fit')
plt.legend()
plt.grid()
plt.show()

print('end Polynomial.py')