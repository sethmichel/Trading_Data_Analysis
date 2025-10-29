import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pygam import LinearGAM, te
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

version = None
success_prob_dir = "Holder_Strat/Parameter_Tuning/Success_Prob_Model"
def Set_Version(passed_version):
    global version
    version = passed_version


def Save_Model_Data(model, scaler, holding_value, holding_sl_value, largest_sl_value):
    file_path = f'{success_prob_dir}/Data/Success_Probability_model_{version}_holding_value_{holding_value}_holding_sl_value_{holding_sl_value}_largest_sl_value_{largest_sl_value}.pkl'

    model_data = {
        'model': model,
        'scaler': scaler,
    }

    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Saved model, scaler to {file_path}")


def Load_Model_Data(holding_value, holding_sl_value, largest_sl_value):
    file_path = f'{success_prob_dir}/Data/Success_Probability_model_{version}_holding_value_{holding_value}_holding_sl_value_{holding_sl_value}_largest_sl_value_{largest_sl_value}.pkl'
    
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    print("Successfully loaded model, and scaler")

    return model, scaler


# save training data so I can quickly get it for tests
def Save_Training_Data(results_df, neither_count, trade_count):
    file_path = f"{success_prob_dir}/Data/Training_Data.json"
    
    data_to_save = {
        'results_df': results_df,
        'neither_count': neither_count,
        'trade_count': trade_count
    }
    
    # Save to txt file
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Saved results_df, neither trade count, and trade count training data to {file_path}")


# load training data so I can quickly get it for tests
def Load_Test_Data():
    file_path = f'{success_prob_dir}/Data/Training_Data.json'
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results_df = data['results_df']
    neither_count = data['neither_count']
    trade_count = data['trade_count']

    print(f"Loaded results_df, neither trade count, and trade count from {file_path}")
    
    return results_df, neither_count, trade_count


def bulk_csv_date_converter(date):
    parts = date.split('-')

    if (len(parts[0]) == 1):
        parts[0] = f"0{parts[0]}"
    if (len(parts[1]) == 1):
        parts[1] = f"0{parts[1]}"
    if (len(parts[2]) == 2):
        parts[2] = f"20{parts[2]}"
    
    return '-'.join(parts)


'''
how this works
-for each trade, record its starting volatility percent and minutes since market open (x features), and the trade id and ticker
-then, look though its roi values until it hits the stop loss or the target (y value). target = 1, stop loss/failure = 0. if we run out of data
   then drop the trade since that's so rare. record the number of dropped trades though
-return results in a easily to read dataframe, the count of skipped trades, and the number of processed trades

parameters
-roi dictionary: each trades second by second roi for the trade duration and beyond
-market_data_dict_by_ticker: market data for each ticker for each date in dataframes. {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
-bulk_df: dataframe of all trades

target variables
x features: minutes since market open, volatility percent
y features: success (1) or failure (0)
ending columns: [trade_id, ticker, minutes since open, volatility percent, result] (where result is y (0 or 1))
'''
def Collect_Training_Data(bulk_df, roi_dictionary, holding_value, stop_loss_value):
    all_trade_data = [] # growing the results dataframe in a loop is super inefficient. just update this dictionary then add it at the end
    skip_dates = []
    trade_count = 0
    neither_count = 0

    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        trade_id = row['Trade Id']
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025

        # if we don't have data for this date
        if (date in skip_dates):
            continue

        # trade is valid if it reaches here

        # 2) process the trade
        roi_list = roi_dictionary[trade_id]
        trade_count += 1
        result = None

        # Iterate through market data starting from entry time
        for roi in roi_list:
            if (roi >= holding_value):
                result = 1
                break
            
            elif (roi <= stop_loss_value):
                result = 0
                break

        else: # for else, this is hit if we don't hit a break in the for loop
            # if here, the trade doesn't reach holding or stop loss or we ran out of market data. so it's NEITHER. skip it
            neither_count += 1
            continue

        entry_time_obj = datetime.strptime(row['Entry Time'] , '%H:%M:%S').time()
        minutes_since_open = (entry_time_obj.hour * 60 + entry_time_obj.minute) - (6 * 60 + 30)  # current minutes - market open minutes

        all_trade_data.append({
            'trade id': trade_id,
            'ticker': ticker,
            'minutes since market open': minutes_since_open, 
            'volatility percent': row['Entry Volatility Percent'],
            'result': result
            })
    
    results_df = pd.DataFrame(all_trade_data)

    return results_df, neither_count, trade_count