import pandas as pd
from datetime import date, timedelta
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


def MoveProcessedFile(date_str, todoDir, doneDir):
    # Convert date format from MM-DD-YYYY to YYYY-MM-DD
    month, day, year = date_str.split('-')
    year = f'20{year}'
    if (len(month) == 1):
        month = f'0{month}'
    if (len(day) == 1):
        day = f'0{day}'
    formatted_date = f"{year}-{month}-{day}"
    
    # Construct the filename
    filename = f"{formatted_date}-TradeActivity.csv"
    source_path = os.path.join(todoDir, filename)
    dest_path = os.path.join(doneDir, filename)
    
    try:
        # Move the file
        shutil.move(source_path, dest_path)
        print(f"Moved {filename} from {todoDir} to {doneDir}")
    except FileNotFoundError:
        print(f"Warning: Could not find file {filename} in {todoDir}")
    except Exception as e:
        print(f"Error moving file: {str(e)}")


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
        trade_type = "BUY" if current_qty > 0 else "SHORT"  # Determine if it's a buy or short trade
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
                        'Quantity': None if has_multiple_entries else initial_qty,  # Use initial quantity unless multiple entries
                        'Best Exit Price': None,
                        'Best Exit Percent': None,
                        'Worst Exit Price': None,
                        'Worst Exit Percent': None,
                        'ATR14': None,
                        'ATR28': None,
                        'Entry Rsi': None,
                        'Entry Macd Val': None, 
                        'Entry Macd Avg': None,
                        'Prev_5_Min_Avg_Close_Volume': None,
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

        if len(ticker_data_dict[ticker]) == 0:
            raise ValueError(f"Warning: Empty market data for ticker {ticker}")

    return ticker_data_dict


def Add_Makret_Data_Helper__Find_Start_Row(ticker, ticker_data_dict, trade):
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


def update_price_movement_tracking(state_dict, current_price, trade_direction):
    # Calculate ROI for current price
    if trade_direction == 'buy':
        current_roi = (current_price - state_dict['entry_price']) / state_dict['entry_price'] * 100
    else:  # short
        # curr_state['entry_price'] - curr_state['macd_exit_price']) / curr_state['entry_price'] * 100)
        current_roi = (state_dict['entry_price'] - current_price) / state_dict['entry_price'] * 100
    
    # Determine which 0.1% threshold level this ROI represents
    if current_roi >= 0:
        threshold_level = int(current_roi * 10) / 10  # Floor to nearest 0.1%
    else:
        threshold_level = int(current_roi * 10) / 10  # Floor to nearest 0.1% (works for negatives)
    
    # Skip if threshold is 0.0
    if threshold_level == 0.0:
        return
    
    # Initialize price_movement if not set
    if state_dict['price_movement'] is None:
        state_dict['price_movement'] = []
    
    # Check if ROI has crossed this threshold level for the first time
    if abs(current_roi) >= abs(threshold_level) and threshold_level not in state_dict['price_movement']:
        # Fill in any missing increments between 0 and the current threshold
        if threshold_level < 0:
            # For negative thresholds, start from -0.1 and go down
            increment = -0.1
            while increment >= threshold_level:
                if increment not in state_dict['price_movement']:
                    state_dict['price_movement'].append(increment)
                increment -= 0.1
                increment = round(increment, 1)  # Avoid floating point precision issues
        else:
            # For positive thresholds, start from 0.1 and go up
            increment = 0.1
            while increment <= threshold_level:
                if increment not in state_dict['price_movement']:
                    state_dict['price_movement'].append(increment)
                increment += 0.1
                increment = round(increment, 1)  # Avoid floating point precision issues


def Add_Market_Data(normalized_df, raw_market_data_name):
    # 1) create a dict where keys = the unique tickers, values = dataframe of all csv rows of that ticker
    ticker_data_dict = Add_Market_Data_Helper__Create_Sub_Dfs(normalized_df, raw_market_data_name)
    
    # NOTE: at this point ticker_data_dict['HOOD'] = all the market data rows for only HOOD

    # 2) add basic data: go over each trade in normalized_df, and add atr14, atr28, start rsi, start macd val/avg
    for idx, trade in normalized_df.iterrows():
        ticker = trade['Ticker']
        start_idx, start_row = Add_Makret_Data_Helper__Find_Start_Row(ticker, ticker_data_dict, trade)

        if start_row is None:
            # TODO: this should NEVER happen, try to play an error sound or something to get the users attention
            print(f"Warning: No matching time found for {ticker} at {trade['Entry Time']}") 
            continue  # Skip this trade if no matching row found
        
        # Add the market data values to normalized_df
        normalized_df.at[idx, 'ATR14'] = start_row['Atr14']
        normalized_df.at[idx, 'ATR28'] = start_row['Atr28']
        normalized_df.at[idx, 'Entry Rsi'] = start_row['Rsi']
        normalized_df.at[idx, 'Entry Macd Val'] = start_row['Val']
        normalized_df.at[idx, 'Entry Macd Avg'] = start_row['Avg']
    
        # 3) add Best/worst Exit Price, Best/worst Exit Percent. Loop over the data from start time to end time
        # currently have the starting row for the trade as "start_row"
        
        # assign best/worst values at start and note exit time in seconds
        entry_price = normalized_df.at[idx, 'Entry_Price']
        curr_best_price = start_row['Price']
        curr_worst_price = start_row['Price']
        curr_worst_percent = 0.00
        curr_best_percent = 0.00
        curr_roi = None

        exit_seconds_parts = normalized_df.at[idx, 'Exit_Time'].split(':')
        exit_seconds = int(exit_seconds_parts[0]) * 3600 + int(exit_seconds_parts[1]) * 60 + int(exit_seconds_parts[2])

        # Get the rows from the starting index to the end
        ticker_df = ticker_data_dict[ticker]
        market_data_from_start = ticker_df.iloc[start_idx:]

        for idx2, row in market_data_from_start.iterrows():
            trade_type = normalized_df.at[idx, 'Trade Type']
            row_price = row['Price']

            # check if it's the exit time row
            row_seconds_parts = row['Time'].split(':')
            row_seconds = int(row_seconds_parts[0]) * 3600 + int(row_seconds_parts[1]) * 60 + int(row_seconds_parts[2])

            if abs(row_seconds - exit_seconds) <= 2:
                # TODO exit logic
                pass

            else: 
                # Update best/worst prices based on trade direction
                if (trade_type == 'short'):
                    # For shorts, lower price is better (profit), higher price is worse (loss)
                    curr_roi_percent = ((entry_price - curr_best_price) / entry_price) * 100

                    if row_price < curr_best_price:
                        curr_best_price = row_price
                        curr_best_percent = curr_roi_percent
                    if row_price > curr_worst_price:
                        curr_worst_price = row_price
                        curr_worst_percent = curr_roi_percent                    
                    
                else:  # buy
                    curr_roi_percent = ((curr_best_price - entry_price) / entry_price) * 100

                    if row_price > curr_best_price:
                        curr_best_price = row_price
                        curr_best_percent = curr_roi_percent
                    if row_price < curr_worst_price:
                        curr_worst_price = row_price
                        curr_worst_percent = curr_roi_percent

                # now deal with percent tracking
                # Round to 2 decimal places
                current_roi_percent = int(curr_roi_percent * 100) / 100  # Truncate to 2 decimal places # WARNING: DO NOT ROUND, if 1.8957 is rounded to 1.9 then 1.9 won't appear in price movement and you'll think it's a bug
    


            # at the exit time find the percent value
            # save best/worst percent/price to normalized_df



    return normalized_df



