import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load actual and predicted data
actual_df = pd.read_csv("/code/USDJPY/actual_future.csv")
predicted_df = pd.read_csv("/code/USDJPY/predicted_close.csv")

# Ensure the lengths match
assert len(actual_df) == len(predicted_df), "The lengths of actual and predicted data do not match."

# Extract values
actual_values = actual_df['actual_future'].values
predicted_values = predicted_df['predicted_close'].values

# Evaluation metrics
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

# Plotting
plt.figure(figsize=(16, 8))
plt.plot(actual_values, label='Actual')
plt.plot(predicted_values, label='Predicted', color='red')
plt.legend()
plt.title('USD/JPY Close Price Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.grid(True)
plt.show()
