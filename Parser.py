import pandas as pd
from datetime import date, timedelta, datetime
import os
import shutil


def CreateDf(csvPath):
    df = pd.read_csv(csvPath, skiprows = 6)

    # delete all the canceled order rows
    index_to_delete = df[df['Unnamed: 0'] == 'Canceled Orders'].index[0]
    df = df[:index_to_delete]

    # drop useless columns
    x = list(df.columns)
    if 'On_Demand' in csvPath:
        columns_to_drop = ["Unnamed: 0", "Spread", "Exp", "Strike", "Type", "Price Improvement", "Order Type"]
    else:
        columns_to_drop = ["Unnamed: 0", "Unnamed: 1", "Spread", "Exp", "Strike", "Type", "Price Improvement", "Order Type"]
    df = df.drop(columns=columns_to_drop)  # 'errors="ignore"' prevents errors if a column doesn't exist

    df.rename(columns={df.columns[0]: "Date"}, inplace = True)   # rename the first column to "Date"
    df.rename(columns={df.columns[4]: "Ticker"}, inplace = True)

    return df


def Move_Processed_Files(raw_trades_name, raw_market_data_name, tos_raw_trades_DONE_dir, market_data_DONE_dir):
    # Move raw trades file to DONE directory
    shutil.move(raw_trades_name, tos_raw_trades_DONE_dir)
    print(f"Moved {raw_trades_name} to {tos_raw_trades_DONE_dir}")
    
    # Move market data file to DONE directory
    shutil.move(raw_market_data_name, market_data_DONE_dir)
    print(f"Moved {raw_market_data_name} to {market_data_DONE_dir}")



def QTYCorrector(qty):
    # Convert qty to a number, remove '+' if it's present
    if qty.startswith("+"):
        qty = qty[1:]
    if isinstance(qty, str):
        qty = float(qty)
    return qty


# Convert price to a number if it's a string
def PriceCorrector(price):
    if isinstance(price, str):
        price = float(price)
    
    return price


def Normalize_Raw_Trades(raw_df):
    normalized_trades = []     # Create a list to store trade summaries
    processed_rows = set()   # Create a set to track processed rows
    ticker_quantities = {}   # Dictionary to track current quantity for each ticker
    
    # Iterate through the dataframe from bottom to top
    for i in range(len(raw_df) - 1, -1, -1):
        # Skip if row already processed
        if i in processed_rows:
            continue
            
        ticker = raw_df.loc[i, "Ticker"]
        current_qty = QTYCorrector(raw_df.loc[i, "Qty"])
        
        # Initialize quantity for ticker if not present
        if ticker not in ticker_quantities:
            ticker_quantities[ticker] = 0
            
        # Skip if this doesn't start a new position (quantity was already non-zero)
        if ticker_quantities[ticker] != 0:
            ticker_quantities[ticker] += current_qty
            continue
            
        # This is a new position (quantity was 0, now changing)
        entry_time = raw_df.loc[i, "Date"].split()[1]  # Get just the time part
        initial_qty = abs(current_qty)  # Store the initial quantity
        current_price = PriceCorrector(raw_df.loc[i, "Net Price"])
        entry_price = current_price  # Store initial entry price
        dollar_change = 0
        total_investment = abs(current_qty * current_price)  # Initial investment amount
        trade_type = "buy" if current_qty > 0 else "short"  # Determine if it's a buy or short trade
        processed_rows.add(i)
        has_multiple_entries = False  # Flag to track if there are multiple entries
        
        # Update the running quantity
        ticker_quantities[ticker] = current_qty
        
        # Look for matching trades going up the dataframe
        for j in range(i - 1, -1, -1):
            if j in processed_rows:
                continue
                
            if raw_df.loc[j, "Ticker"] == ticker:
                new_qty = QTYCorrector(raw_df.loc[j, "Qty"])
                new_price = PriceCorrector(raw_df.loc[j, "Net Price"])
                
                # Check if this is another entry (quantity was 0 before this trade)
                if ticker_quantities[ticker] == 0:
                    has_multiple_entries = True
                
                # Calculate dollar change for this part of the trade
                if current_qty < 0:  # Short position
                    price_diff = current_price - new_price
                else:  # Long position
                    price_diff = new_price - current_price
                    
                # Adjust for the quantity being closed
                closing_qty = abs(new_qty) if new_qty < 0 else new_qty
                dollar_change += price_diff * closing_qty
                
                # Update current quantity and total investment
                current_qty += new_qty
                ticker_quantities[ticker] = current_qty
                
                # Only add to total investment if it's a new entry (quantity was 0)
                if ticker_quantities[ticker] - new_qty == 0:
                    total_investment += abs(new_qty * new_price)
                
                processed_rows.add(j)
                
                # If position is closed (qty = 0), record the trade
                if current_qty == 0:
                    exit_time = raw_df.loc[j, "Date"].split()[1]  # Get just the time part
                    exit_price = new_price  # Store the exit price
                    
                    # Calculate time in trade
                    entry_datetime = pd.to_datetime(raw_df.loc[i, "Date"])
                    exit_datetime = pd.to_datetime(raw_df.loc[j, "Date"])
                    time_diff = exit_datetime - entry_datetime
                    time_str = str(time_diff).split('.')[0]  # Remove microseconds if present
                    time_in_trade = time_str.split('days ')[-1] if 'days' in time_str else time_str # get rid of days
                    
                    # Calculate percentage gain/loss
                    percent_change = (dollar_change / total_investment) * 100
                    
                    # Add trade summary
                    normalized_trades.append({
                        'Date': raw_df.loc[i, "Date"].split()[0].replace("/","-"),  # Just the date part
                        'Ticker': ticker,
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Time in Trade': time_in_trade,
                        'Dollar Change': round(dollar_change, 2),
                        'Percent Change': round(percent_change, 2),
                        'Total Investment': round(total_investment, 2),
                        'Entry Price': round(entry_price, 4),
                        'Exit Price': round(exit_price, 4),
                        'Trade Type': trade_type,
                        'Qty': None if has_multiple_entries else initial_qty,  # Use initial quantity unless multiple entries
                        'Best Exit Price': None,
                        'Best Exit Percent': None,
                        'Worst Exit Price': None,
                        'Worst Exit Percent': None,
                        'Entry Rsi': None,
                        'Entry Macd Val': None, 
                        'Entry Macd Avg': None,
                        'Entry Atr14': None,
                        'Entry Atr28': None,
                        'Entry Volatility Percent': None,
                        'Entry Volatility Ratio': None,
                        'Target': None,
                        'Prev 5 Min Avg Close Volume': None,
                        'Price_Movement': None
                    })
                    break
    
    # Create DataFrame from trade summaries
    noralized_df = pd.DataFrame(normalized_trades)
    
    return noralized_df


# goal: ticker_data_dict['HOOD'] = all the market data rows for only HOOD
# each value is a dataframe of all rows for that ticker
def Add_Market_Data_Helper__Create_Sub_Dfs(normalized_df, raw_market_data_name):
    market_df = pd.read_csv(raw_market_data_name)
    unique_tickers = normalized_df['Ticker'].unique()
    
    ticker_data_dict = {}
    for ticker in unique_tickers:
        ticker_data_dict[ticker] = market_df[market_df['Ticker'] == ticker].copy()

    return ticker_data_dict


def Add_Market_Data_Helper__Find_Start_Row(ticker, ticker_data_dict, trade):
    entry_time = trade['Entry Time']
    ticker_data_df = ticker_data_dict[ticker]
    
    # Convert entry_time to seconds for easier comparison
    entry_time_parts = entry_time.split(':')
    entry_seconds = int(entry_time_parts[0]) * 3600 + int(entry_time_parts[1]) * 60 + int(entry_time_parts[2])
    
    # Find matching row in ticker_data_df (±2 seconds tolerance)
    for idx, row in ticker_data_df.iterrows():
        # convert the rows time to seconds for easier comparison
        curr_row_time = row['Time']
        curr_row_time_parts = curr_row_time.split(':')
        row_seconds = int(curr_row_time_parts[0]) * 3600 + int(curr_row_time_parts[1]) * 60 + int(curr_row_time_parts[2])
        
        # Check if times match within 2 seconds
        if abs(row_seconds - entry_seconds) <= 2:
            return idx, row
    
    return None


# this tracks unique 0.1% movements, does not record duplicates
def Add_Market_Data_Helper__Update_Price_Movement(curr_price_movement, curr_roi_percent):
    # Determine which 0.1% threshold level this ROI represents
    threshold_level = int(curr_roi_percent * 10) / 10  # Floor to nearest 0.1%
    
    # Skip if threshold is 0.0
    if threshold_level == 0.0:
        return curr_price_movement
    
    # Extract current values from dictionaries for comparison
    current_values = [entry['value'] for entry in curr_price_movement]
    
    # Check if ROI has crossed this threshold level for the first time
    if abs(curr_roi_percent) >= abs(threshold_level) and threshold_level not in current_values:
        # Fill in any missing increments between 0 and the current threshold
        if threshold_level < 0:
            # For negative thresholds, start from -0.1 and go down
            increment = -0.1
            while increment >= threshold_level:
                if increment not in current_values:
                    curr_price_movement.append({'value': increment, 'timestamp': None})
                    current_values.append(increment)  # Update our tracking list
                increment -= 0.1
                increment = round(increment, 1)  # Avoid floating point precision issues
        else:
            # For positive thresholds, start from 0.1 and go up
            increment = 0.1
            while increment <= threshold_level:
                if increment not in current_values:
                    curr_price_movement.append({'value': increment, 'timestamp': None})
                    current_values.append(increment)  # Update our tracking list
                increment += 0.1
                increment = round(increment, 1)  # Avoid floating point precision issues
    
    return curr_price_movement


# this tracks all 0.1% movements including 0.0, with duplicate tracking and oscillation detection
# also fills in gaps
# it uses timestamps to reject spam adding values if price oscilates between values over and over
def Add_Market_Data_Helper__Update_Price_Movement_With_Duplicates(curr_price_movement, curr_roi_percent):    
    # Floor to nearest 0.1% like the original function
    threshold_level = int(curr_roi_percent * 10) / 10
    current_time = datetime.now()
    if (len(curr_price_movement) == 3):
        pass
    
    # If this is the first entry, just add it
    if len(curr_price_movement) == 0:
        curr_price_movement.append({'value': threshold_level, 'timestamp': current_time})
        return curr_price_movement
    
    # Rule 1: Cannot be the same value as the most recent value
    if curr_price_movement[-1]['value'] == threshold_level:
        return curr_price_movement
    
    # Check if we need to fill gaps between most recent value and current value
    most_recent_value = curr_price_movement[-1]['value']
    
    # Calculate the gap and fill if necessary
    if most_recent_value != threshold_level:
        # Determine direction and fill gaps
        if most_recent_value < threshold_level:
            # Going up, fill gaps from most_recent + 0.1 to threshold_level
            gap_value = most_recent_value + 0.1
            gap_value = round(gap_value, 1)  # Avoid floating point precision issues
            while gap_value < threshold_level:
                # Calculate interpolated timestamp for gap fill
                most_recent_time = curr_price_movement[-1]['timestamp']
                time_diff = (current_time - most_recent_time).total_seconds()
                gap_timestamp = most_recent_time + timedelta(seconds=time_diff / 2)
                
                curr_price_movement.append({'value': gap_value, 'timestamp': gap_timestamp})
                gap_value += 0.1
                gap_value = round(gap_value, 1)
        else:
            # Going down, fill gaps from most_recent - 0.1 to threshold_level
            gap_value = most_recent_value - 0.1
            gap_value = round(gap_value, 1)
            while gap_value > threshold_level:
                # Calculate interpolated timestamp for gap fill
                most_recent_time = curr_price_movement[-1]['timestamp']
                time_diff = (current_time - most_recent_time).total_seconds()
                gap_timestamp = most_recent_time + timedelta(seconds=time_diff / 2)
                
                curr_price_movement.append({'value': gap_value, 'timestamp': gap_timestamp})
                gap_value -= 0.1
                gap_value = round(gap_value, 1)
    
    # Rule 2: Check oscillation detection with 2nd most recent value
    if len(curr_price_movement) >= 2:
        second_most_recent_value = curr_price_movement[-2]['value']
        if second_most_recent_value == threshold_level:
            # Check if it's been at least 10 seconds since the 2nd most recent entry
            second_most_recent_time = curr_price_movement[-2]['timestamp']
            time_diff = (current_time - second_most_recent_time).total_seconds()
            if time_diff < 30:
                # Less than 10 seconds, ignore this value due to oscillation
                return curr_price_movement
    
    # Add the current value
    curr_price_movement.append({'value': threshold_level, 'timestamp': current_time})
    return curr_price_movement


# removes noise values from list. it's like 230 values per minute otherwise. 80% of which is useless
# we DO NOT save the timestamps
def Post_Process_Price_Movement(price_movement):
    processed_movement = []

    # if too short to process
    if len(price_movement) <= 5:
        # drop all the timestamps
        for i, entry in enumerate(price_movement):
            processed_movement.append(entry['value'])

        return processed_movement  
    
    for i, entry in enumerate(price_movement):
        curr_value = entry['value']
        
        # For the first 5 entries, always add them
        if len(processed_movement) < 5:
            processed_movement.append(curr_value) # no timestamps
            continue
        
        # Get the prev 5 values from processed_movement
        prev_5_values = processed_movement[-5:]
        
        # If current value is NOT in the prev 5, we want to add it
        if curr_value not in prev_5_values:
            # check for gaps
            last_processed_value = processed_movement[-1]
            
            # find the 0.1 distance btw last_processed_value and curr_value. 0.2 and -0.1
            holder_val = last_processed_value
            while (holder_val != curr_value):
                # keep this at the start
                if (last_processed_value > curr_value):
                    holder_val = round(holder_val - 0.1, 1)  # fixes float error
                elif (last_processed_value < curr_value):
                    holder_val = round(holder_val + 0.1, 1)  # fixes float error
                
                # this will add the final value correctly
                processed_movement.append(holder_val)
    
    for i in processed_movement:
        if not isinstance(i, float):
            pass
    return processed_movement


# 0.3
def Was_Target_Hit(processed_price_movement):
    target = 0.3
    stop_loss = -0.3

    for percent in processed_price_movement:
        if (percent == stop_loss):
            return 0
        
        if (percent == target):
            return 1
    
    return 0


def Add_Market_Data_Helper__Best_Worst_Updator(ticker, normalized_df, ticker_data_dict, start_row, idx, start_idx):
    entry_price = start_row['Price']
    curr_best_price = start_row['Price']   # correct, it starts at this
    curr_worst_price = start_row['Price']  # correct, it starts at this
    curr_worst_percent = 0.00
    curr_best_percent = 0.00
    curr_roi_percent = None
    price_movement = []         # list of dictionaries 'value' and 'timestamp'
    exit_time_reached = False   # for exit condition, we stop tracking when the macd re-crosses or the exit time is reached, whichever is later
    macd_cross_reached = False  # for exit condition, edge case described below
    trade_type = normalized_df.at[idx, 'Trade Type']
    ticker_df = ticker_data_dict[ticker]  # Get the rows from the starting index to the end
    
    exit_seconds_parts = normalized_df.at[idx, 'Exit Time'].split(':')
    exit_seconds = int(exit_seconds_parts[0]) * 3600 + int(exit_seconds_parts[1]) * 60 + int(exit_seconds_parts[2])
    
    # Convert original index to position for slicing
    start_position = ticker_df.index.get_loc(start_idx)
    market_data_from_start = ticker_df.iloc[start_position:]

    for idx2, row in market_data_from_start.iterrows():
        row_price = row['Price']
        macd_val = row['Val']
        macd_avg = row["Avg"]
        # check if it's the exit time row
        row_seconds_parts = row['Time'].split(':')
        row_seconds = int(row_seconds_parts[0]) * 3600 + int(row_seconds_parts[1]) * 60 + int(row_seconds_parts[2])

        # 3.2) update best/worst prices/percents
        # Update best/worst prices based on trade direction
        if (trade_type.lower() == 'short'):
            # For shorts, lower price is better (profit), higher price is worse (loss)
            curr_roi_percent = ((entry_price - row_price) / entry_price) * 100

            if row_price < curr_best_price:
                curr_best_price = row_price
                curr_best_percent = curr_roi_percent
            elif row_price > curr_worst_price:
                curr_worst_price = row_price
                curr_worst_percent = curr_roi_percent                    
            
        else:  # buy
            curr_roi_percent = ((row_price - entry_price) / entry_price) * 100

            if row_price > curr_best_price:
                curr_best_price = row_price
                curr_best_percent = curr_roi_percent
            elif row_price < curr_worst_price:
                curr_worst_price = row_price
                curr_worst_percent = curr_roi_percent
        
        # 3.3) track the price movement
        # Round to 2 decimal places
        curr_roi_percent = int(curr_roi_percent * 100) / 100  # Truncate to 2 decimal places # WARNING: DO NOT ROUND, if 1.8957 is rounded to 1.9 then 1.9 won't appear in price movement and you'll think it's a bug
        
        # this version only tracks unique values
        #price_movement = Add_Market_Data_Helper__Update_Price_Movement(price_movement, curr_roi_percent)

        # this version tracks all values, so way more price movement data
        price_movement = Add_Market_Data_Helper__Update_Price_Movement_With_Duplicates(price_movement, curr_roi_percent)

        # 3.4) exit logic - if macd val and avg don't line up with trade_type OR the exit time is reached, whichever is LATER
        # scenarios: exit time reached before end cross
        #            end cross before exit time
        #            end time reached so late that the cross has re-crossed into the trade type
        if (exit_time_reached == False and abs(row_seconds - exit_seconds) <= 2):
            exit_time_reached = True

        if (macd_cross_reached == False and (trade_type == 'buy' and macd_val < macd_avg) or (trade_type == 'short' and macd_val > macd_avg)):
            macd_cross_reached = True

        # edge case. possible to enter and exit a trade but I stop recording data before the macd cross. like if I stop recording at 8 but the macd cross is at 8:05
        elif (macd_cross_reached == False and idx2 == market_data_from_start.index[-1]):
            macd_cross_reached = True

        if (exit_time_reached == True and macd_cross_reached == True):
            processed_price_movement = Post_Process_Price_Movement(price_movement)
            normalized_df.at[idx, 'Target'] = Was_Target_Hit(processed_price_movement)
            normalized_df.at[idx, 'Best Exit Price'] = curr_best_price
            normalized_df.at[idx, 'Worst Exit Price'] = curr_worst_price
            normalized_df.at[idx, 'Best Exit Percent'] = round(curr_best_percent, 2)
            normalized_df.at[idx, 'Worst Exit Percent'] = round(curr_worst_percent, 2)
            normalized_df.at[idx, 'Price_Movement'] = '|'.join(map(str, processed_price_movement)) 

            return normalized_df


def Add_Market_Data(normalized_df, raw_market_data_name):
    # 1) create a dict where keys = the unique tickers, values = dataframe of all csv rows of that ticker
    ticker_data_dict = Add_Market_Data_Helper__Create_Sub_Dfs(normalized_df, raw_market_data_name)
    
    # NOTE: at this point ticker_data_dict['HOOD'] = all the market data rows for only HOOD

    # 2) add basic data: go over each trade in normalized_df, and add atr14, atr28, start rsi, start macd val/avg
    for idx, trade in normalized_df.iterrows():
        ticker = trade['Ticker']
        
        # possible we don't have market data for a ticker we traded. that's fine
        if len(ticker_data_dict[ticker]) == 0:
            print(f"Warning: Empty market data for ticker {ticker}, skipping this trade")
            continue
        
        start_idx, start_row = Add_Market_Data_Helper__Find_Start_Row(ticker, ticker_data_dict, trade)
        if (idx == 15):
            pass # it fails on tsla index 15. it never finds the end of tsla. short. cross fails right away, so val > avg. so the cross exit might be getting skipped

        if start_row is None:
            # TODO: this should NEVER happen, try to play an error sound or something to get the users attention
            print(f"Warning: No matching time found for {ticker} at {trade['Entry Time']}") 
            continue  # Skip this trade if no matching row found
        
        # Add the market data values to normalized_df
        normalized_df.at[idx, 'Entry Atr14'] = start_row['Atr14']
        normalized_df.at[idx, 'Entry Atr28'] = start_row['Atr28']
        normalized_df.at[idx, 'Entry Volatility Percent'] = start_row['Volatility Percent']
        normalized_df.at[idx, 'Entry Volatility Ratio'] = start_row['Volatility Ratio']
        normalized_df.at[idx, 'Entry Rsi'] = start_row['Rsi']
        normalized_df.at[idx, 'Entry Macd Val'] = start_row['Val']
        normalized_df.at[idx, 'Entry Macd Avg'] = start_row['Avg']
    
        # 3) add Best/worst Exit Price, Best/worst Exit Percent. Loop over the data from start time to end time
        # currently have the starting row for the trade as "start_row"
        # 3.0 - 3.4) Get variables, update best/worst values, update price movement for the current trade. updates are written to normalized_df at the correct row
        normalized_df = Add_Market_Data_Helper__Best_Worst_Updator(ticker, normalized_df, ticker_data_dict, start_row, idx, start_idx)

    return normalized_df

