import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

print('start take_price.py')
# MetaTrader 5に接続
if not mt5.initialize():
    print("MetaTrader5 initialize failed")
    mt5.shutdown()

# MetaTrader 5バージョンについてのデータを表示する
print(f'MetaTrader5 version : {mt5.version()}')
 
terminal_info_dict = mt5.terminal_info()._asdict() # ターミナルの設定とステータスに関する情報を取得
account_info_dict = mt5.account_info()._asdict() # アカウント情報を取得

'''
print(f'\nターミナルの設定とステータスに関する情報\n{terminal_info_dict}')
print(f'\nアカウント情報\n{account_info_dict}')

print(type(terminal_info_dict))
print(type(account_info_dict))

print(terminal_info_dict['company'])
'''

symbol = 'USDJPY'
timeframe = mt5.TIMEFRAME_H1
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
Number_data = 10000

rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

# MetaTrader 5との接続を切断
mt5.shutdown()

#print (rates)
# 取得したレート情報をPandasデータフレームに変換
rates_df = pd.DataFrame(rates, columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')

if len(rates_df) > Number_data:
    rates_df = rates_df.tail(Number_data).copy()

rates_df.reset_index(drop=True, inplace=True)

# print(rates_df)
csv_encoding = 'utf-8'
rates_df.to_csv("/code/USDJPY/1hour.csv", index=False, encoding=csv_encoding)

print('end take_price.py')