import pandas as pd
import os
import inspect
import sys
import shutil
import Main_Globals
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
import os

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


"""
Computes true range values for a single ticker's dataframe.
Tracks high, low, close values by minute and calculates TR for each minute.

Args:
    ticker_df: DataFrame for a single ticker with columns ['Ticker', 'Time', 'Price']

Returns:
    list: directionay of high, low, close TR for ticker
"""
def Find_True_Range(ticker_df):
    try:
        high_values = []
        low_values = []
        close_values = []
        tr_values = []
        current_minute = None
        prev_row_price = None
        
        for idx, row in ticker_df.iterrows():
            curr_price = row['Price']
            curr_time = row['Time']
            
            time_parts = curr_time.split(':')
            minute = int(time_parts[1])
            
            # Check if we're in a new minute
            if current_minute != minute:
                # Start new minute - initialize high and low with current price
                current_minute = minute
                high_values.append(curr_price)
                low_values.append(curr_price)
                # Only add close value if we have a previous price (not first iteration)
                if prev_row_price is not None:
                    close_values.append(prev_row_price)
                
            else:
                # Still in same minute - update high and low if needed
                if curr_price > high_values[-1]:
                    high_values[-1] = curr_price
                elif curr_price < low_values[-1]:
                    low_values[-1] = curr_price
            
            # Store current price as previous for next iteration
            prev_row_price = curr_price
        
        # Add the final close price
        if prev_row_price is not None:
            close_values.append(prev_row_price)
        
        # Calculate True Range for each minute
        for i in range(len(high_values)):
            if i == 0:
                # First minute has no previous close, so TR = None
                tr_values.append(None)
            else:
                high = high_values[i]
                low = low_values[i]
                # Check if we have a previous close value
                if i-1 < len(close_values):
                    prev_close = close_values[i-1]  # Previous minute's close
                    
                    tr = max(
                        high - low,
                        abs(high - prev_close),
                        abs(low - prev_close)
                    )
                    tr_values.append(round(tr, 4))
                else:
                    # No previous close available
                    tr_values.append(None)
        
        return {"high": high_values, 'low': low_values, 'close': close_values, 'true range': tr_values}
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# (BROKEN) DIRECTIONAL BIAS --------------------------------------------------------------------
def directional_bias_helper_get_high_low_close(market_data_df):
    data_holder = {}

    for idx, row in market_data_df.iterrows():
        ticker = row['Ticker']
        curr_price = row['Price']
        curr_time = row['Time']

        if ticker not in data_holder:
            data_holder[ticker] = {
                'high': [],
                'low': [],
                'close': [],
                'prev close': [],
                'TR': [],
                'upTR': [],
                'downTR': [],
                'upAtr': [],
                'downAtr': [],
                'directional bias': [],
                'current_minute': None,
                'prev_row_price': curr_price
            }

        time_parts = curr_time.split(':')
        curr_minute = int(time_parts[1])
        
        # Check if we're in a new minute
        if data_holder[ticker]['current_minute'] != curr_minute:
            # If this isn't the first minute for this ticker, save the close price
            if data_holder[ticker]['current_minute'] != None:
                data_holder[ticker]['close'].append(data_holder[ticker]['prev_row_price'])
            
            # Start new minute - initialize high and low with current price
            data_holder[ticker]['current_minute'] = curr_minute
            data_holder[ticker]['high'].append(curr_price)
            data_holder[ticker]['low'].append(curr_price)
            
        else:
            # Still in same minute - update high and low if needed
            if curr_price > data_holder[ticker]['high'][-1]:
                data_holder[ticker]['high'][-1] = curr_price
            elif curr_price < data_holder[ticker]['low'][-1]:
                data_holder[ticker]['low'][-1] = curr_price
        
        # Store current price as previous for next iteration
        data_holder[ticker]['prev_row_price'] = curr_price

    # After processing all rows, add the final close price for each ticker
    for ticker in data_holder:
        data_holder[ticker]['close'].append(data_holder[ticker]['prev_row_price'])

    return data_holder


def Calcualte_Directional_Bias_V2(market_data_csv_path):
    market_data_df = pd.read_csv(market_data_csv_path)
    ema_length = 7
    
    # 1) get high, low, close for every ticker
    data_holder = directional_bias_helper_get_high_low_close(market_data_df)

    # 2) populate prev close. it's i-1 of close
    for ticker in data_holder:
        close_list = data_holder[ticker]['close']
        prev_close_list = data_holder[ticker]['prev close']
        prev_close_list.append(None)  # first index is non computable
        
        for i in range(len(close_list) - 1):
            prev_close_list.append(close_list[i])

        data_holder[ticker]['prev close'] = prev_close_list

    # 3) compute True range
    '''TR = max(
        High - Low,
        abs(High - PrevClose),
        abs(Low - PrevClose)) '''
    for ticker in data_holder:
        high_list = data_holder[ticker]['high']
        low_list = data_holder[ticker]['low']
        prev_close_list = data_holder[ticker]['prev close']
        tr_list = data_holder[ticker]['TR']
        
        # Compute TR for each index
        for i in range(len(high_list)):
            high = high_list[i]
            low = low_list[i]
            prev_close = prev_close_list[i]
            
            if (prev_close == None):
                tr = round(high - low, 3)
            else:
                tr = round(max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                ), 3)

            tr_list.append(tr)

        data_holder[ticker]['TR'] = tr_list

    # 4) classify TR as upTR or downTR
    # If Close > PrevClose → UpTR = TR, DownTR = 0
    # If Close < PrevClose → DownTR = TR, UpTR = 0
    # else they're both 0
    for ticker in data_holder:
        close_list = data_holder[ticker]['close']
        prev_close_list = data_holder[ticker]['prev close']
        tr_list = data_holder[ticker]['TR']
        up_tr_list = data_holder[ticker]['upTR']
        down_tr_list = data_holder[ticker]['downTR']
        
        # Classify each TR value based on close vs prev_close
        for i in range(len(tr_list)):
            close = close_list[i]
            prev_close = prev_close_list[i]
            tr = tr_list[i]
            
            # prev_close[0] is none
            if  (prev_close == None or (close == prev_close)):
                up_tr_list.append(0)
                down_tr_list.append(0)

            elif close > prev_close:
                up_tr_list.append(tr)
                down_tr_list.append(0)

            else:
                up_tr_list.append(0)
                down_tr_list.append(tr)
                
        data_holder[ticker]['upTR'] = up_tr_list
        data_holder[ticker]['downTR'] = down_tr_list

    # 5) compute wilders EMA using ema_length (use up/downTR for data). The first x entries of up/downAtr should be None due to warmup period
    for ticker in data_holder:
        up_tr_list = data_holder[ticker]['upTR']
        down_tr_list = data_holder[ticker]['downTR']
        up_atr_list = data_holder[ticker]['upAtr']
        down_atr_list = data_holder[ticker]['downAtr']
        
        # Calculate Wilder EMA for both upTR and downTR
        for i in range(len(up_tr_list)):
            if i < ema_length:
                # First 7 indexes should be None
                up_atr_list.append(None)
                down_atr_list.append(None)
            elif i == ema_length:
                # First EMA calculation - simple average of first 7 values
                up_avg = sum(up_tr_list[:ema_length]) / ema_length
                down_avg = sum(down_tr_list[:ema_length]) / ema_length
                up_atr_list.append(round(up_avg, 4))
                down_atr_list.append(round(down_avg, 4))
            else:
                # Subsequent calculations - Wilder EMA formula
                current_up_tr = up_tr_list[i]
                current_down_tr = down_tr_list[i]
                previous_up_ema = up_atr_list[i-1]
                previous_down_ema = down_atr_list[i-1]
                
                new_up_ema = previous_up_ema + (current_up_tr - previous_up_ema) / ema_length
                new_down_ema = previous_down_ema + (current_down_tr - previous_down_ema) / ema_length
                
                up_atr_list.append(round(new_up_ema, 4))
                down_atr_list.append(round(new_down_ema, 4))
        
        data_holder[ticker]['upAtr'] = up_atr_list
        data_holder[ticker]['downAtr'] = down_atr_list

    # 6) compute directional bias
    for ticker in data_holder:
        up_atr_list = data_holder[ticker]['upAtr']
        down_atr_list = data_holder[ticker]['downAtr']
        directional_bias_list = data_holder[ticker]['directional bias']
        
        # Calculate directional bias for each index
        for i in range(len(up_atr_list)):
            if up_atr_list[i] is None or down_atr_list[i] is None:
                # During warmup period, directional bias is None
                directional_bias_list.append(None)
            else:
                up_atr = up_atr_list[i]
                down_atr = down_atr_list[i]
                denominator = up_atr + down_atr
                
                if denominator == 0:
                    # Neutral bias when both are 0
                    directional_bias = 0.5
                else:
                    directional_bias = up_atr / denominator
                
                directional_bias_list.append(round(directional_bias, 3))
        
        data_holder[ticker]['directional bias'] = directional_bias_list
    
    return data_holder
  

# for each ticker in data_holder
# write that minutes data_holder[ticker]['directional bias'] for the whole minute to each line
# when we detect a minute change move to the next index of data_holder[ticker]['directional bias']
# if it's none then write "NaN"
def Add_Directional_Bias_To_Market_Data(data_holder, market_data_csv_path):
    try:
        df = pd.read_csv(market_data_csv_path)
        df['Directional Bias'] = None   # Initialize new directional bias column
        ticker_minute_index = {}        # Dictionary to track current minute index for each ticker
        
        for index, row in df.iterrows():
            ticker = row['Ticker']
            time_str = row['Time']
            
            time_parts = time_str.split(':')
            current_minute = int(time_parts[1])
            
            # Initialize ticker tracking if first time seeing this ticker
            if ticker not in ticker_minute_index:
                ticker_minute_index[ticker] = {
                    'current_minute': current_minute,
                    'minute_index': 0
                }
            
            # Check if we've moved to a new minute for this ticker
            if current_minute != ticker_minute_index[ticker]['current_minute']:
                ticker_minute_index[ticker]['current_minute'] = current_minute
                ticker_minute_index[ticker]['minute_index'] += 1
            
            # Get the directional bias value
            directional_bias_list = data_holder[ticker]['directional bias']
            minute_index = ticker_minute_index[ticker]['minute_index']
            
            # Check if we have data for this minute index
            if minute_index < len(directional_bias_list):
                bias_value = directional_bias_list[minute_index]
                if bias_value is None:
                    df.at[index, 'Directional Bias'] = "NaN"
                else:
                    df.at[index, 'Directional Bias'] = bias_value
        
        # Reorder columns to put Time as the rightmost column
        columns = df.columns.tolist()
        columns.remove('Time')
        columns.append('Time')
        df = df[columns]
        
        # Save the modified dataframe back to the same file
        df.to_csv(market_data_csv_path, index=False)
        
        print(f"Added Directional Bias column to: {market_data_csv_path}\n")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
# -------------------------------------------------------------------------------------

# MY BROKEN ADX --------------------------------------------------------------------------
# basically teh same thing as directional bias. and it's not aligned at all with tos
def Helper_ADX_Find_DM_Values(ticker, ticker_values):
    try:
        high_values = ticker_values[ticker]['high']
        low_values = ticker_values[ticker]['low']
        plus_dm = []
        minus_dm = []
        
        for i in range(len(high_values)):
            if i == 0:
                # First index has no previous values, so DM = 0
                plus_dm.append(None)
                minus_dm.append(None)
            else:
                # Calculate up and down moves
                upMove = high_values[i] - high_values[i-1]
                downMove = low_values[i-1] - low_values[i]  # Positive when low drops
                
                # Calculate +DM and -DM
                if upMove > downMove and upMove > 0:
                    plus_dm.append(round(upMove, 4))
                    minus_dm.append(0)
                elif downMove > upMove and downMove > 0:
                    plus_dm.append(0)
                    minus_dm.append(round(downMove, 4))
                else:
                    # Neither condition met, both are zero
                    plus_dm.append(0)
                    minus_dm.append(0)
        
        # Add DM values to ticker_values dictionary
        ticker_values[ticker]['+DM'] = plus_dm
        ticker_values[ticker]['-DM'] = minus_dm

        return ticker_values

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Helper_ADX_Wilders_Smoothing(ticker, ticker_values):
    try:
        smoothing_period = 7
        
        # Get the raw values
        true_range_values = ticker_values[ticker]['true range']
        plus_dm_values = ticker_values[ticker]['+DM']
        minus_dm_values = ticker_values[ticker]['-DM']
        
        # Initialize smoothed lists
        smoothed_tr = []
        smoothed_plus_dm = []
        smoothed_minus_dm = []
        
        # Apply Wilder's smoothing to True Range. keep in mind index 0 is None, so it takes longer to warm up
        for i in range(len(true_range_values)):
            if i < smoothing_period:
                # Not enough data points yet
                smoothed_tr.append(None)
            elif i == smoothing_period:
                # First smoothed value - sum of first N valid values (indices 1-7)
                tr_sum = sum([val for val in true_range_values[1:smoothing_period+1] if val is not None])
                smoothed_tr.append(round(tr_sum, 4))
            else:
                # Apply Wilder's smoothing formula: smoothed_t = smoothed_t-1 - (smoothed_t-1 / N) + current_value
                current_tr = true_range_values[i] if true_range_values[i] is not None else 0
                previous_smoothed = smoothed_tr[i-1]
                
                if previous_smoothed is None:
                    # If previous smoothed value is None, skip this calculation
                    smoothed_tr.append(None)
                else:
                    new_smoothed = previous_smoothed - (previous_smoothed / smoothing_period) + current_tr
                    smoothed_tr.append(round(new_smoothed, 4))
        
        # Apply Wilder's smoothing to +DM
        for i in range(len(plus_dm_values)):
            if i < smoothing_period:
                # Not enough data points yet
                smoothed_plus_dm.append(None)
            elif i == smoothing_period:
                # First smoothed value - sum of first N valid values (indices 1-7)
                plus_dm_sum = sum([val for val in plus_dm_values[1:smoothing_period+1] if val is not None])
                smoothed_plus_dm.append(round(plus_dm_sum, 4))
            else:
                # Apply Wilder's smoothing formula
                current_plus_dm = plus_dm_values[i]
                previous_smoothed = smoothed_plus_dm[i-1]
                
                if previous_smoothed is None:
                    # If previous smoothed value is None, skip this calculation
                    smoothed_plus_dm.append(None)
                else:
                    new_smoothed = previous_smoothed - (previous_smoothed / smoothing_period) + current_plus_dm
                    smoothed_plus_dm.append(round(new_smoothed, 4))
        
        # Apply Wilder's smoothing to -DM
        for i in range(len(minus_dm_values)):
            if i < smoothing_period:
                # Not enough data points yet
                smoothed_minus_dm.append(None)
            elif i == smoothing_period:
                # First smoothed value - sum of first N valid values (indices 1-7)
                minus_dm_sum = sum([val for val in minus_dm_values[1:smoothing_period+1] if val is not None])
                smoothed_minus_dm.append(round(minus_dm_sum, 4))
            else:
                # Apply Wilder's smoothing formula
                current_minus_dm = minus_dm_values[i]
                previous_smoothed = smoothed_minus_dm[i-1]
                
                if previous_smoothed is None:
                    # If previous smoothed value is None, skip this calculation
                    smoothed_minus_dm.append(None)
                else:
                    new_smoothed = previous_smoothed - (previous_smoothed / smoothing_period) + current_minus_dm
                    smoothed_minus_dm.append(round(new_smoothed, 4))
        
        # Add smoothed values to ticker_values dictionary
        ticker_values[ticker]['smoothed_TR'] = smoothed_tr
        ticker_values[ticker]['smoothed_+DM'] = smoothed_plus_dm
        ticker_values[ticker]['smoothed_-DM'] = smoothed_minus_dm
            
        return ticker_values

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Helper_ADX_Find_DI_Values(ticker, ticker_values):
    try:
        pos_di_vals = []
        neg_di_vals = []

        for i in range(0, len(ticker_values[ticker]['smoothed_TR'])):
            smooth_pos_dm = ticker_values[ticker]['smoothed_+DM'][i]
            smooth_neg_dm = ticker_values[ticker]['smoothed_-DM'][i]
            smooth_tr = ticker_values[ticker]['smoothed_TR'][i]

            if None in [smooth_pos_dm, smooth_neg_dm, smooth_tr]:
                pos_di_vals.append(None)
                neg_di_vals.append(None)
            else:
                pos_di_vals.append(100 * (smooth_pos_dm / smooth_tr))
                neg_di_vals.append(100 * (smooth_neg_dm / smooth_tr))

        ticker_values[ticker]['pos_di'] = pos_di_vals
        ticker_values[ticker]['neg_di'] = neg_di_vals

        return ticker_values
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Helper_ADX_Find_ADX(ticker, ticker_values):
    try:
        dx_list = []
        adx_list = []
        smoothing_period = 7

        for i in range(0, len(ticker_values[ticker]['smoothed_TR'])):
            pos_di_val = ticker_values[ticker]['pos_di'][i]
            neg_di_val = ticker_values[ticker]['neg_di'][i]

            if (None in [pos_di_val, neg_di_val]):
                dx_list.append(None)

            else:
                denominator = pos_di_val + neg_di_val
                
                if (denominator == 0):
                    dx_list.append(None)
                else:
                    dx_list.append(100 * (abs(pos_di_val - neg_di_val) / denominator))

        # Apply Wilder's smoothing to DX values to get ADX
        # First DX value is at index 7, so first ADX is at index 13 (7 + 7 - 1)
        for i in range(len(dx_list)):
            if i < 2 * smoothing_period - 1:
                # Not enough DX values yet (need 7 DX values starting from index 7)
                adx_list.append(None)
            elif i == 2 * smoothing_period - 1:
                # First ADX value - average of first N DX values (indices 7-13)
                valid_dx_values = [val for val in dx_list[smoothing_period:i+1] if val is not None]
                if len(valid_dx_values) >= smoothing_period:
                    first_adx = sum(valid_dx_values) / len(valid_dx_values)
                    adx_list.append(round(first_adx, 4))
                else:
                    adx_list.append(None)
            else:
                # Apply Wilder's smoothing: adx[i] = previous_adx + (current_dx - previous_adx) / N
                current_dx = dx_list[i]
                previous_adx = adx_list[i-1]
                
                if current_dx is None or previous_adx is None:
                    adx_list.append(None)
                else:
                    new_adx = previous_adx + (current_dx - previous_adx) / smoothing_period
                    adx_list.append(round(new_adx, 4))

        ticker_values[ticker]['dx'] = dx_list
        ticker_values[ticker]['adx'] = adx_list

        return ticker_values
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# +/- DI (or DM) tell trend direction. 1 is a value, the other is 0
def Add_ADX(market_data_csv_path):
    try:
        df = pd.read_csv(market_data_csv_path)
        ticker_values = {}
        unique_tickers = df['Ticker'].unique()
        
        # Split dataframe by ticker and process each one
        for ticker in unique_tickers:
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df.reset_index(drop=True, inplace=True)  # Reset index for cleaner processing
            
            # 1) find True Range values for this ticker ({"high": high_values, 'low': low_values, 'close': close_values, 'true range': tr_values})
            ticker_values[ticker] = Find_True_Range(ticker_df)

            # 2) find +/- DM values
            ticker_values = Helper_ADX_Find_DM_Values(ticker, ticker_values)
            
            # 3) smooth TR, DM values
            ticker_values = Helper_ADX_Wilders_Smoothing(ticker, ticker_values)

            # 4) find +/- DI Values
            ticker_values = Helper_ADX_Find_DI_Values(ticker, ticker_values)

            # 5) find DX (directional index) for each row
            ticker_values = Helper_ADX_Find_ADX(ticker, ticker_values)

            # Print ticker_values contents for debugging
            print(f"\n=== {ticker} ticker_values contents ===")
            for key, value in ticker_values[ticker].items():
                if isinstance(value, list):
                    # Round numeric values in lists to 3 decimal places and show only first 10
                    rounded_value = []
                    for item in value[:15]:  # Only process first 10 items
                        if isinstance(item, (int, float)) and item is not None:
                            rounded_value.append(round(item, 3))
                        else:
                            rounded_value.append(item)
                    print(f"{key}: {rounded_value}")
                else:
                    print(f"{key}: {value}")
            print("=" * 50)
            pass

        return ticker_values

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
# -------------------------------------------------------------------------------------


# MACD Z-SCORE ------------------------------------------------------------------------
def Add_Macd_Z_Score(market_data_csv_path):
    try:
        # Load the market data from the CSV file
        df = pd.read_csv(market_data_csv_path)
        
        # Define the lookback period for rolling calculations
        lookback_period = 300  # 300 seconds as specified
        
        # 1. Calculate MACD Histogram (MACD Value - MACD Average/Signal)
        df['MACD_Histogram'] = df['Val'] - df['Avg']
        
        # Define a function to calculate the Z-score for a series (a single ticker's data)
        def calculate_z_score(series):
            # 2. Calculate rolling mean and standard deviation
            rolling_mean = series.rolling(window=lookback_period).mean()
            rolling_std = series.rolling(window=lookback_period).std()
            
            # 3. Compute the Z-Score
            z_score = round((series - rolling_mean) / rolling_std, 4)
            
            # 4. Handle the edge case where standard deviation is zero.
            # This occurs if the histogram value is constant over the lookback period.
            # In this case, the deviation from the mean is zero, so the Z-score should be 0.
            z_score[rolling_std == 0] = 0
            return z_score
            
        # Use groupby().transform() to apply the z-score calculation to each ticker's data.
        # Transform ensures the output is aligned with the original DataFrame's index.
        df['Macd Z-Score'] = df.groupby('Ticker')['MACD_Histogram'].transform(calculate_z_score)

        # Drop the intermediate histogram column as it's no longer needed
        df.drop(columns=['MACD_Histogram'], inplace=True)
        
        # Reorder columns to place the new Z-Score column after the 'Avg' column for readability
        if 'Avg' in df.columns:
            cols = df.columns.tolist()
            avg_index = cols.index('Avg')
            cols.remove('Macd Z-Score')
            cols.insert(avg_index + 1, 'Macd Z-Score')
            df = df[cols]

        # 5. Save the modified dataframe back to the original CSV file
        df.to_csv(market_data_csv_path, index=False)
        
        print(f"Successfully added 'Macd Z-Score' column to: {market_data_csv_path}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, "Add_Macd_Z_Score", str(e), sys.exc_info()[2].tb_lineno)
# -------------------------------------------------------------------------------------


# ADX RATIO ---------------------------------------------------------------------------
def Add_ADX_Ratio(market_data_csv_path):
    try:
        df = pd.read_csv(market_data_csv_path)

        # Check if 'Adx7' or 'Adx28' columns contain any zero values.
        if (df['Adx7'] == 0).any() or (df['Adx28'] == 0).any():
            raise ValueError("ADX values cannot be zero for ratio calculation. Found zeros in 'Adx7' or 'Adx28'.")

        # Calculate the ADX Ratio and round to 3 decimal places.
        df['Adx Ratio'] = (df['Adx7'] / df['Adx28']).round(3)

        # Save the modified dataframe back to the original CSV file
        df.to_csv(market_data_csv_path, index=False)

        print(f"Successfully added 'Adx Ratio' column to: {market_data_csv_path}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, "Add_ADX_Ratio", str(e), sys.exc_info()[2].tb_lineno)
# -------------------------------------------------------------------------------------


# ADX SLOPE ---------------------------------------------------------------------------
def Add_ADX_Slope(market_data_csv_path):
    try:
        # Load the market data from the CSV file
        df = pd.read_csv(market_data_csv_path)

        # Convert 'Time' column to datetime objects for time-based calculations
        df['datetime'] = pd.to_datetime(df['Time'], format='%H:%M:%S')

        # Create a copy of the dataframe to be used for looking up previous values
        df_prev = df[['Ticker', 'datetime', 'Adx7']].copy()
        df_prev.rename(columns={'Adx7': 'Adx7_prev'}, inplace=True)

        # To find the value from 3 minutes ago for each row, we'll shift the time in
        # the original dataframe to create a lookup key.
        df['lookup_time'] = df['datetime'] - pd.Timedelta(minutes=3)

        # Sort dataframes by ticker and time, which is a requirement for merge_asof
        df.sort_values(by=['Ticker', 'datetime'], inplace=True)
        df_prev.sort_values(by=['Ticker', 'datetime'], inplace=True)

        # Use merge_asof to find the closest 'Adx7' value from 3 minutes ago
        # for each ticker. 'direction=backward' ensures we only look back in time.
        merged_df = pd.merge_asof(
            df,
            df_prev,
            left_on='lookup_time',
            right_on='datetime',
            by='Ticker',
            direction='backward'
        )

        # Calculate the 'Adx Slope'. This will be NaN if no data was found 3 minutes ago.
        merged_df['Adx Slope'] = merged_df['Adx7'] - merged_df['Adx7_prev']

        # Clean up by dropping the temporary columns used for the calculation
        merged_df.drop(columns=['datetime', 'Adx7_prev', 'lookup_time'], inplace=True)

        # Save the modified dataframe back to the original CSV file
        merged_df.to_csv(market_data_csv_path, index=False)

        print(f"Successfully added 'Adx Slope' column to: {market_data_csv_path}")

    except Exception as e:
        # Use the existing error handler to log any issues
        Main_Globals.ErrorHandler(fileName, "Add_ADX_Slope", str(e), sys.exc_info()[2].tb_lineno)
# -------------------------------------------------------------------------------------

# ADX 7v14 & 7v28 ---------------------------------------------------------------------
def Add_ADX_Comparisons(market_data_csv_path):
    try:
        # Load the market data from the CSV file
        df = pd.read_csv(market_data_csv_path)

        # Check if the required ADX columns exist
        if 'Adx7' not in df.columns or 'Adx14' not in df.columns or 'Adx28' not in df.columns:
            raise ValueError("CSV file must contain 'Adx7', 'Adx14', and 'Adx28' columns.")

        # Calculate 'Adx7 Cross 14'
        df['Adx7 Cross 14'] = (df['Adx7'] - df['Adx14']).round(2)

        # Calculate 'Adx7 Cross 28'
        df['Adx7 Cross 28'] = (df['Adx7'] - df['Adx28']).round(2)

        # Save the modified dataframe back to the original CSV file
        df.to_csv(market_data_csv_path, index=False)

        print(f"Successfully added ADX comparison columns to: {market_data_csv_path}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, "Add_ADX_Comparisons", str(e), sys.exc_info()[2].tb_lineno)




csv_dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"

# macd z score
'''for filename in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, filename)
    print(f"Processing: {filename}")
    #Add_Macd_Z_Score(file_path)
'''
# adx slope
'''for filename in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, filename)
    print(f"Processing ADX Slope for: {filename}")
    Add_ADX_Slope(file_path)'''

# adx 7v14 & 7v28
for filename in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, filename)
    print(f"Processing ADX Comparisons for: {filename}")
    Add_ADX_Comparisons(file_path)


#Add_ADX("test market csv.csv") # 5/12