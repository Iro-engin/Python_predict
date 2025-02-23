import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

print('start')
minute_df = pd.read_csv("/code/USDJPY/15min.csv")

def preprocess_data(df):
    df['diff_open'] = df['open'].diff()
    df['diff_high'] = df['high'].diff()
    df['diff_low'] = df['low'].diff()
    df['diff_close'] = df['close'].diff()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df = preprocess_data(minute_df)

# データの分割
train_size = len(df) - 5
train_df = df[:train_size]
test_df = df[train_size:]

# Convert 'time' column to datetime format
df['time'] = pd.to_datetime(df['time'])

# Standardize the data
scaler_open = StandardScaler()
scaler_high = StandardScaler()
scaler_low = StandardScaler()
scaler_close = StandardScaler()

scaled_open = scaler_open.fit_transform(df[['diff_open']]) #open
scaled_high = scaler_high.fit_transform(df[['diff_high']])
scaled_low = scaler_low.fit_transform(df[['diff_low']])
scaled_close = scaler_close.fit_transform(df[['diff_close']])

def train_arima(data, p, d, q):
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

def predict_arima(model, n_steps):
    forecast = model.forecast(steps=n_steps)
    return forecast

p = 100
d = 5
q = 0

models = {}
# モデル学習 (修正点)
for col in ['open', 'high', 'low', 'close']:
    train_data = scaled_open[:train_size].flatten() if col == 'open' else \
                 scaled_high[:train_size].flatten() if col == 'high' else \
                 scaled_low[:train_size].flatten() if col == 'low' else \
                 scaled_close[:train_size].flatten()

    model = train_arima(train_data, p, d, q)
    models[col] = model  # 文字列でキーを指定

predictions = {}

# 予測 (修正点)
for col in ['open', 'high', 'low', 'close']:
    test_data = scaled_open[train_size:].flatten() if col == 'open' else \
                scaled_high[train_size:].flatten() if col == 'high' else \
                scaled_low[train_size:].flatten() if col == 'low' else \
                scaled_close[train_size:].flatten()

    forecast = predict_arima(models[col], len(test_data))  # 学習済みモデルを使用
    predictions[col] = forecast  # 予測値を格納
    print([col])
    
# 逆標準化 (修正点)
for col in ['open', 'high', 'low', 'close']:
    scaler = eval(f"scaler_{col}")
    predicted_values = predictions[col]
    predicted_values = predicted_values.reshape(-1, 1)
    predicted_values = scaler.inverse_transform(predicted_values).flatten()
    predictions[col] = predicted_values

last_open_price = test_df['open'].iloc[0]
last_high_price = test_df['high'].iloc[0]
last_low_price = test_df['low'].iloc[0]
last_close_price = test_df['close'].iloc[0]


predicted_open = last_open_price + predictions['open']
predicted_high = last_high_price + predictions['high']
predicted_low = last_low_price + predictions['low']
predicted_close = last_close_price + predictions['close']

'''
predicted_open = predictions['open']
predicted_high = predictions['high']
predicted_low = predictions['low']
predicted_close = predictions['close']
'''
print(f"Predicted OHLC for the next 5 15min steps: Open={predicted_open[:5]}, High={predicted_high[:5]}, Low={predicted_low[:5]}, Close={predicted_close[:5]}") #最初の５つを表示

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    corr_coef = np.corrcoef(y_true, y_pred)[0, 1]
    return mae, rmse, r2, corr_coef

# 実際の値
actual_open = test_df['open'].values
actual_high = test_df['high'].values
actual_low = test_df['low'].values
actual_close = test_df['close'].values

# 評価指標の計算 (最初の5つを評価)
mae_open, rmse_open, r2_open, corr_coef_open = evaluate(actual_open[:5], predicted_open[:5])
mae_high, rmse_high, r2_high, corr_coef_high = evaluate(actual_high[:5], predicted_high[:5])
mae_low, rmse_low, r2_low, corr_coef_low = evaluate(actual_low[:5], predicted_low[:5])
mae_close, rmse_close, r2_close, corr_coef_close = evaluate(actual_close[:5], predicted_close[:5])


print("Open:")
print(f"  MAE: {mae_open:.4f}")
print(f"  RMSE: {rmse_open:.4f}")
print(f"  R^2: {r2_open:.4f}")
print(f"  相関係数: {corr_coef_open:.4f}")

print("High:")
print(f"  MAE: {mae_high:.4f}")
print(f"  RMSE: {rmse_high:.4f}")
print(f"  R^2: {r2_high:.4f}")
print(f"  相関係数: {corr_coef_high:.4f}")

print("Low:")
print(f"  MAE: {mae_low:.4f}")
print(f"  RMSE: {rmse_low:.4f}")
print(f"  R^2: {r2_low:.4f}")
print(f"  相関係数: {corr_coef_low:.4f}")

print("Close:")
print(f"  MAE: {mae_close:.4f}")
print(f"  RMSE: {rmse_close:.4f}")
print(f"  R^2: {r2_close:.4f}")
print(f"  相関係数: {corr_coef_close:.4f}")

plt.figure(figsize=(12, 8))

# 各OHLCの予測結果をプロット
for col in ['open', 'high', 'low', 'close']:
    plt.plot(test_df['time'].values, test_df[col].values, label=f'Actual {col}')  # 実測値
    plt.plot(test_df['time'].values, eval(f'predicted_{col}'), label=f'Predicted {col}')  # 予測値

plt.title('OHLC Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()