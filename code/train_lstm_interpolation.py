import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model

# Load data
hourly_df = pd.read_csv("/code/USDJPY/1hour.csv")
minute_df = pd.read_csv("/code/USDJPY/15min.csv")

# Prepare data for LSTM model
def prepare_lstm_data(df, column_name, window_size=50):
    data = df[column_name].values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Train LSTM model
def train_lstm_model(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32)
    return model

# Interpolate missing values using LSTM model
def interpolate_with_lstm(df, minute_df, column_name, window_size=50):
    # Prepare data
    minute_values = minute_df[column_name].values
    df[column_name] = df[column_name].interpolate(method='linear')
    df.dropna(inplace=True)
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(minute_values.reshape(-1, 1))
    
    # Load pre-trained LSTM model
    model = load_model(f"c:/Users/稲葉湧大/Desktop/Python予測プロジェクト/Python_trade_learn/code/USDJPY/lstm_model_{column_name}.h5")
    
    # Predict missing values
    predictions = []
    if len(df) > window_size:
        for i in range(len(df) - window_size):
            input_seq = scaled_data[i:i+window_size].reshape(1, window_size, 1)
            predicted_value = model.predict(input_seq)
            predictions.append(predicted_value[0, 0])
    
    # Inverse transform predictions
    if predictions:
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        df[column_name].iloc[window_size:] = predictions.flatten()
    
    return df

# Prepare and train LSTM models for each column
for col in ['open', 'high', 'low', 'close']:
    X, y, scaler = prepare_lstm_data(minute_df, col)
    model = train_lstm_model(X, y)
    model.save(f"c:/Users/稲葉湧大/Desktop/Python予測プロジェクト/Python_trade_learn/code/USDJPY/lstm_model_{col}.h5")
