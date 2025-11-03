from re import L
import pandas as pd
import numpy as np
import os
import inspect
from datetime import datetime


fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


def Remove_1_Column(csv_path):
    column_to_remove = 'Macd Z-Score'
    
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)
        
        # Check if the column exists
        if column_to_remove in df.columns:
            # Remove the specified column
            df = df.drop(columns=[column_to_remove])
            
            # Overwrite the existing CSV file
            df.to_csv(csv_path, index=False)
            print(f"Successfully removed column '{column_to_remove}' from {csv_path}")
        else:
            print(f"Column '{column_to_remove}' not found in {csv_path}")
            
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")


# vol % has time values, vol ratio has vol% values, time is blank. it's missing vol ratio
def Custom_Fix_1(csv_path):
    df = pd.read_csv(csv_path)
    # capture misplaced columns
    temp_time = df['Volatility Percent'].copy()
    temp_volpct = df['Volatility Ratio'].copy()
    # shift columns to correct positions
    df['Time'] = temp_time
    df['Volatility Percent'] = temp_volpct
    # recalculate and assign volatility ratio
    df['Volatility Ratio'] = (df['Atr14'] / df['Atr28']).round(2)
    # overwrite CSV
    df.to_csv(csv_path, index=False)
    print(f"Successfully applied Custom_Fix_1 to {csv_path}")




# DEPRECATED - incorrect algorithm
''' NOTE: this is wrong, it's using start price vs end price instead of high vs low
-ultimatly this equation doesn't well. it super overreacts when teh minute changes, meaning the value gets larger with 'seconds' then drops at each new minute
-it's going to be better to just offset the actual vol% value. 

'''
def OLD_Fix_Morning_Atr_Issue():
    '''
    notes about how this works. 
    -candle_percent_change_list isn't multiplied by 100 because we later need the average of the list to go to price units instead of
    percents. so we need 0.0045 to mean 0.45%
    -we treat each rows price as the closing price for that candle and only 'lock in' values long term when the minute changes
    -rather than start exactly at market open, many files start a few seconds/minutes late. so we go x minutes after the starting time
    rather than market open
    '''
    dir = os.path.join("Csv_Files", "Raw_Market_Data", "market_data_to_check")
    required_cols = ['Ticker', 'Price', 'Atr14', 'Time']
    csv_files = [f for f in os.listdir(dir) if f.lower().endswith('.csv')]
    minutes = 10 # only find this value for this many minutes

    for csv_file in csv_files:
        file_path = os.path.join(dir, csv_file)
        df = pd.read_csv(file_path)
        df['Early Morning Atr Warmup Fix'] = float('nan')  # Prepare new column with NaNs

        # Process each ticker independently
        def process_ticker(ticker_df):
            ticker_df = ticker_df.sort_values('Time', kind='stable').reset_index()         # Sort by Time
            candle_percent_change = {'minutes': [], 'seconds': []}  # holds the calcualted volatility of the last 3 minutes
            minutes_updated = 0 # each minute iterates this by 1 until it's 3. this is to handle updating the list each second
            minute_starting_prices = []      # last 3 minutes
            prev_row_minutes_since_open = -1 # -1 means it triggers on first row
            prev_rows_price = ticker_df.at[0, 'Price']

            file_starting_time = datetime.strptime(ticker_df.at[0, 'Time'], '%H:%M:%S').time()
            file_starting_minute = file_starting_time.hour * 60 + file_starting_time.minute # since midnight
            
            for idx, row in ticker_df.iterrows():
                row_price = row['Price']
                row_atr14 = row['Atr14']
                row_time = row['Time']

                # get minutes since open
                time_obj = datetime.strptime(row_time, '%H:%M:%S').time()
                current_minutes = time_obj.hour * 60 + time_obj.minute
                minutes_since_open = round(current_minutes - file_starting_minute)

                # check if we're done with this ticker
                if (minutes_since_open >= minutes):
                    break

                # if we reach a new minute
                if (prev_row_minutes_since_open != minutes_since_open):
                    minute_starting_prices.append(row_price)
                    prev_row_minutes_since_open = minutes_since_open

                    if (minutes_since_open != 0):
                        # must be abs
                        candle_percent_change['minutes'].append(abs(((prev_rows_price - minute_starting_prices[-1]) / minute_starting_prices[-1])))
                        if (len(candle_percent_change['minutes']) > 2):
                            candle_percent_change['minutes'].pop(0)
                            minute_starting_prices.pop(0)                                
                
                # do this every single row (must be abs)
                candle_percent_change['seconds'].append(abs(((row_price - minute_starting_prices[-1]) / minute_starting_prices[-1])))

                if (len(candle_percent_change['seconds']) > 1):
                    candle_percent_change['seconds'].pop(0)

                candle_percent_change_list = []
                for value in candle_percent_change['minutes']:
                    candle_percent_change_list.append(value)
                for value in candle_percent_change['seconds']:
                    candle_percent_change_list.append(value)

                w = min(minutes_since_open / 10, 1)
                if (w == 0):
                    w = 0.1 # happens at minute 0, so set it to what it would be for minute 1
                atr3_mimic = np.average(candle_percent_change_list) * row_price # multiple by row_price because it must be in price units
                
                atr_effective = w * row_atr14 + (1 - w) * atr3_mimic
                if (minutes_since_open == 0 and atr_effective == 0): # happens in minute 0 when price hasn't changed from starting price yet
                    vol_percent_estimate = float('nan')
                elif (minutes_since_open != 0 and atr_effective == 0):
                    pass # debug
                else:
                    vol_percent_estimate = round((atr_effective / row_price) * 100, 2)

                ticker_df.at[idx, 'Early Morning Atr Warmup Fix'] = vol_percent_estimate
                prev_rows_price = row_price


            # set original index so we can realign after groupby apply
            ticker_df = ticker_df.set_index('index')
            return ticker_df

        # Apply per-ticker
        processed = (
            df.groupby('Ticker', group_keys=False, as_index=False)
                .apply(process_ticker)
        )

        # Assign back using original index alignment
        df.loc[processed.index, 'Early Morning Atr Warmup Fix'] = processed['Early Morning Atr Warmup Fix']

        # Place the new column immediately after 'Volatility Percent'
        df = df[["Ticker","Price","Val","Avg","Atr14","Atr28","Rsi","Volume","Adx28","Adx14","Adx7","Volatility Percent","Early Morning Atr Warmup Fix","Volatility Ratio","Time"]]

        df.to_csv(file_path, index=False)
        print(f"Saved warmup fix to: {file_path}")


csv_dir = "Csv_Files/raw_Market_Data/market_data_to_check"
csv_path = f"{csv_dir}/Raw_Market_Data_10-22-2025.csv"

#Custom_Fix_1(csv_path)



