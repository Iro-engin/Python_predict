import pandas as pd

# Read the 4hour.csv file
df_4hour = pd.read_csv("/code/USDJPY/4hour.csv")

# Read the predicted_close.csv file
df_predicted_close = pd.read_csv("/code/USDJPY/predicted_close.csv")

# Add the predicted_close column to the 4hour.csv dataframe
df_4hour['predicted_close'] = df_predicted_close['predicted_close']

# Save the updated dataframe to a new CSV file
df_4hour.to_csv("/code/USDJPY/4hour_with_predicted_close.csv", index=False)
