import os
import pandas as pd
import inspect
import sys

# Add parent directory to path so we can import Main_Globals
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Main_Globals

# Alternative approach using relative imports (requires __init__.py files):
# from .. import Main_Globals

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

'''
goal: create a csv with
date, ticker, held, entry time, exit time, time in trade, change percent, change dollar, running sum, total investment, trade type, 
best exit percent, best exit price, best exit time in trade, worst exit percent, worst exit price, worst exit time in trade,
Entry Atr14, Entry Atr28, Entry Volatility Percent, Entry Volatility Ratio, Entry Adx28, Entry Adx14, Entry Adx7, Price Movement

held is bool for if it broke the target
trade type is buy or short
'''

def Check_Dirs(dir_list):
    try:
        print("Step 1: Checking if directories exist...")
        
        for directory in dir_list:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"Created directory: {directory}")
            else:
                print(f"Directory already exists: {directory}")

        for filename in os.listdir(dir_list[2]):
            os.remove(f"{dir_list[2]}/{filename}")
            print(f"Deleted: {filename}")
        print("\n")

        print("Step 1 completed: All directories verified/created. Old summary csv's deleted\n")
        return True

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return False


def Find_Valid_Files(market_data_dir, trade_log_dir):
    try:
        all_trade_dates = []
        all_trade_paths = []
        all_market_data_dates = []
        all_market_data_paths = []
        valid_file_pairs = []  # list of lists [trade file, market file]

        for file in os.listdir(trade_log_dir):
            if file.endswith('.csv'):
                parts = file.split('-')[0:3] # year, month, day
                date = f"{parts[1]}-{parts[2]}-{parts[0]}"
                all_trade_dates.append(date)
                all_trade_paths.append(f"{trade_log_dir}/{file}")

        # now look at market data, if we have a file for the trade date add that date to a list of valid dates
        for file in os.listdir(market_data_dir):
            if file.endswith('.csv'):
                date = (file.split('_'))[3]  # month, day, year
                if ".csv" in date:
                    date = date[:-4] # live data goes ...date.csv
                all_market_data_dates.append(date)
                all_market_data_paths.append(f"{market_data_dir}/{file}")
        
        for i in range(0, len(all_market_data_dates)):
            try:
                trade_index = all_trade_dates.index(all_market_data_dates[i])
            except:
                continue

            valid_file_pairs.append([all_trade_paths[trade_index], all_market_data_paths[i]])
        
        return valid_file_pairs            

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return []


def Create_Df(csv_path):
    try:
        raw_df = pd.read_csv(csv_path, skiprows = 6)

        # delete all the canceled order rows
        index_to_delete = raw_df[raw_df['Unnamed: 0'] == 'Canceled Orders'].index[0]
        raw_df = raw_df[:index_to_delete]

        # drop useless columns
        columns_to_drop = ["Unnamed: 0", "Spread", "Exp", "Strike", "Type", "Price Improvement", "Order Type"]
        if "Unnamed: 1" in raw_df.columns:
            columns_to_drop.append("Unnamed: 1") # idk what causes this to be in it. the original code for this thought it was always in live data, but 8/25/25 live didn't have it

        raw_df = raw_df.drop(columns=columns_to_drop)   # 'errors="ignore"' prevents errors if a column doesn't exist

        raw_df.rename(columns={raw_df.columns[0]: "Date"}, inplace = True)   # rename the first column to "Date"
        raw_df.rename(columns={raw_df.columns[4]: "Ticker"}, inplace = True)

        return raw_df
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


def Create_Summary_Df(raw_df):
    try:
        # actual logs have qty as a string, cross logs have it as a float. no idea why
        def QTYCorrector(qty):
            try:
                # Convert qty to a number, remove '+' if it's present
                if isinstance(qty, str):
                    if qty.startswith("+"):
                        qty = qty[1:]
                    qty = float(qty)
                return qty
            except Exception as e:
                Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)

        # Convert price to a number if it's a string
        def PriceCorrector(price):
            try:
                if isinstance(price, str):
                    price = float(price)
                
                return price
            except Exception as e:
                Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
                
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
                            'Running Percent By Ticker': None,
                            'Running Percent All': None,
                            'Total Investment': round(total_investment, 2),
                            'Entry Price': round(entry_price, 4),
                            'Exit Price': round(exit_price, 4),
                            'Trade Type': trade_type,
                            'Qty': None if has_multiple_entries else initial_qty,  # Use initial quantity unless multiple entries
                            'Best Exit Price': None,
                            'Best Exit Percent': None,
                            'Best Exit Time In Trade': None,
                            'Worst Exit Price': None,
                            'Worst Exit Percent': None,
                            'Worst Exit, Time In Trade': None,
                            'Entry Atr14': None,
                            'Entry Atr28': None,
                            'Entry Volatility Percent': None,
                            'Entry Volatility Ratio': None,
                            'Entry Adx28': None,
                            'Entry Adx14': None,
                            'Entry Adx7': None,
                            'Price Movement': None
                        })
                        break

        # Create DataFrame from trade summaries
        noralized_df = pd.DataFrame(normalized_trades)
        
        return noralized_df
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# NOTE: this does it for 2 columns: by ticker, and in total
def Add_Running_Percents(df):
    try:
        tickers_percent_sum = {}
        total_sum = 0

        for i, row in df.iterrows():
            ticker = row['Ticker']
            percent_change = row['Percent Change']

            if (tickers_percent_sum.get(ticker) == None):
                tickers_percent_sum[ticker] = 0.0

            tickers_percent_sum[ticker] += percent_change
            df.at[i, 'Running Percent By Ticker'] = round(tickers_percent_sum[ticker], 2)

            total_sum += percent_change
            df.at[i, 'Running Percent All'] = round(total_sum, 2)
        
        return df

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None
    

def Create_Ticker_Market_Data_Dfs(df, market_data_path):
    market_df = pd.read_csv(market_data_path)

    # convert each timestamp to seconds for faster comparison
    for idx, row in market_df.iterrows():
        time_parts = row['Time'].split(':')
        market_df.at[idx, 'Time'] = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
    
    unique_tickers = df['Ticker'].unique()
    ticker_data_dict = {}
    for ticker in unique_tickers:
        ticker_data_dict[ticker] = market_df[market_df['Ticker'] == ticker].copy()
    
    return ticker_data_dict


# direction is 'buy' or 'short'
def Add_Best_Worst_Info(ticker_df, idx_2, entry_price, direction, exit_time_seconds):
    try:
        # --- start at idx_2 which is the first rows index of ticker_df. iterate over ticker_df from there
        # --- we stop when we find the last row of the trade, each row has row['Time'] which is the time in seconds. 
        #     we can check the last row by doing some form of row['Time'] < exit_time_seconds.
        # --- for each row, track if it's the new best roi price, and if it's the new worst roi price. This means 
        #     if direction is 'buy' 'best' means prices higher than entry_price, 'worst' means prices lower than entry_price
        #     vice versa for if direction is 'short'. so if the rows price is a new best or worst, record that price along 
        #     with the time stamp (which is in seconds).
        # --- at the end: use investment, entry_price, direction, best price, and worst price to find the % best percent 
        #     change and worst percent change.
        # --- return all of that info

        # Initialize tracking variables
        best_price = entry_price
        worst_price = entry_price
        best_time_seconds = None
        worst_time_seconds = None
        
        # Start iterating from idx_2 (first row of the trade)
        # idx_2 is the dataframe index, so we need to get the position in the dataframe
        start_position = ticker_df.index.get_loc(idx_2)
        for idx in range(start_position, len(ticker_df)):
            row = ticker_df.iloc[idx]
            current_time = row['Time']
            
            # Stop when we reach the exit time
            if current_time > exit_time_seconds:
                break
            
            current_price = row['Price']
            
            if direction == 'buy':
                # For buy trades: best = higher prices, worst = lower prices
                if current_price > best_price:
                    best_price = current_price
                    best_time_seconds = current_time
                if current_price < worst_price:
                    worst_price = current_price
                    worst_time_seconds = current_time
            else:  # direction == 'short'
                # For short trades: best = lower prices, worst = higher prices
                if current_price < best_price:
                    best_price = current_price
                    best_time_seconds = current_time
                if current_price > worst_price:
                    worst_price = current_price
                    worst_time_seconds = current_time
        
        # Calculate percentage changes
        if direction == 'buy':
            best_percent_change = ((best_price - entry_price) / entry_price) * 100
            worst_percent_change = ((worst_price - entry_price) / entry_price) * 100
        else:  # direction == 'short'
            best_percent_change = ((entry_price - best_price) / entry_price) * 100
            worst_percent_change = ((entry_price - worst_price) / entry_price) * 100
        
        # Convert time in seconds to time in trade format (seconds from entry)
        entry_time_seconds = ticker_df.loc[idx_2, 'Time']
        best_time_in_trade = best_time_seconds - entry_time_seconds if best_time_seconds else 0
        worst_time_in_trade = worst_time_seconds - entry_time_seconds if worst_time_seconds else 0
        
        return {
            'Best Exit Price': round(best_price, 4),
            'Best Exit Percent': round(best_percent_change, 2),
            'Best Exit Time In Trade': best_time_in_trade,
            'Worst Exit Price': round(worst_price, 4),
            'Worst Exit Percent': round(worst_percent_change, 2),
            'Worst Exit Time In Trade': worst_time_in_trade
        }

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


# this tracks all 0.1% movements including 0.0, with duplicate tracking and oscillation detection
def Add_Price_Movement(ticker_df, entry_price, start_idx, exit_time_seconds, direction):
    try:
        price_movement = []

        start_position = ticker_df.index.get_loc(start_idx)
        for idx in range(start_position, len(ticker_df)):
            row = ticker_df.iloc[idx]
            current_time = row['Time']
            current_price = row['Price']

            # Stop when we reach the exit time
            if current_time > exit_time_seconds:
                break
            
            if (direction == 'buy'):
                curr_roi_percent = ((current_price - entry_price) / entry_price) * 100
            else: # direction = 'short'
                curr_roi_percent = ((entry_price - current_price) / entry_price) * 100

            # Floor to nearest 0.1%
            threshold_level = int(curr_roi_percent * 10) / 10
            
            # If this is the first entry, just add it
            if len(price_movement) == 0:
                price_movement.append(threshold_level)
                continue
            
            # Rule 1: Cannot be the same value as the most recent value
            most_recent_value = price_movement[-1]
            if most_recent_value == threshold_level:
                continue
            
            # Rule 2: Check if we need to fill gaps between most recent value and current value
            # Calculate the gap and fill if necessary
            # Determine direction and fill gaps
            if most_recent_value < threshold_level:
                # Going up, fill gaps from most_recent + 0.1 to threshold_level
                gap_value = round(most_recent_value + 0.1, 1)
                while gap_value < threshold_level:
                    price_movement.append(gap_value)
                    gap_value = round(gap_value + 0.1, 1)
            else:
                # Going down, fill gaps from most_recent - 0.1 to threshold_level
                gap_value = round(most_recent_value - 0.1, 1)
                while gap_value > threshold_level:
                    price_movement.append(gap_value)
                    gap_value = round(gap_value - 0.1, 1)
            
            # Add the current value
            price_movement.append(threshold_level)

        return price_movement
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Add_Market_Data(df, market_data_path):
    try:
        # find best/worst exist
        # goal: ticker_data_dict['HOOD'] = all the market data rows for only HOOD
        # each value is a dataframe of all rows for that ticker
        ticker_data_dict = Create_Ticker_Market_Data_Dfs(df, market_data_path)

        for idx, trade in df.iterrows():
            ticker = trade['Ticker']
            entry_price = trade['Entry Price']
            direction = trade['Trade Type']
            ticker_df = ticker_data_dict[ticker]
            
            entry_time_parts = trade['Entry Time'].split(':')
            entry_seconds = int(entry_time_parts[0]) * 3600 + int(entry_time_parts[1]) * 60 + int(entry_time_parts[2])

            exit_time_parts = trade['Exit Time'].split(':')
            exit_seconds = int(exit_time_parts[0]) * 3600 + int(exit_time_parts[1]) * 60 + int(exit_time_parts[2])

            # Find the starting index for the trade in ticker_df
            idx_2 = None
            for ticker_idx, row in ticker_df.iterrows():
                # start at the first row with a greater than or equal to entry time
                x = row['Time']
                if (row['Time'] < entry_seconds):
                    continue

                # This is the first row at or after entry time
                if idx_2 is None:
                    idx_2 = ticker_idx
                    df.at[idx, 'Entry Atr14'] = row['Atr14']
                    df.at[idx, 'Entry Atr28'] = row['Atr28']
                    df.at[idx, 'Entry Volatility Percent'] = row['Volatility Percent']
                    df.at[idx, 'Entry Volatility Ratio'] = row['Volatility Ratio']
                    df.at[idx, 'Entry Adx28'] = row['Adx28']
                    df.at[idx, 'Entry Adx14'] = row['Adx14']
                    df.at[idx, 'Entry Adx7'] = row['Adx7']
                    break

            # add Best/worst Exit info. Loop over the data from start time to end time
            if idx_2 is not None:
                best_worst_info = Add_Best_Worst_Info(ticker_df, idx_2, entry_price, direction, exit_seconds)
                if (best_worst_info == None):
                    return None

                df.at[idx, 'Best Exit Price'] = best_worst_info['Best Exit Price']
                df.at[idx, 'Best Exit Percent'] = best_worst_info['Best Exit Percent']
                df.at[idx, 'Best Exit Time In Trade'] = best_worst_info['Best Exit Time In Trade']
                df.at[idx, 'Worst Exit Price'] = best_worst_info['Worst Exit Price']
                df.at[idx, 'Worst Exit Percent'] = best_worst_info['Worst Exit Percent']
                df.at[idx, 'Worst Exit Time In Trade'] = best_worst_info['Worst Exit Time In Trade']

                df.at[idx, 'Price Movement'] = Add_Price_Movement(ticker_df, entry_price, idx_2, exit_seconds, direction)

        return df

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


def Create_Summarized_Info_txt(df):
    try:
        # --- goal: extract summarized results from df and write to a txt file
        # --- df is now a completed normalized trade log dataframe, each line is all details of 1 trade
        # --- write at the top: days % change, days $ change, number of trades where best exit percent is > 0.6,
        #     what's the sum of the percent change of trades who's best exit percent is > 0.6
        # --- do the same as the previous point but for each unique ticker in the 'Ticker' column
        
        # Get the script directory and create output directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'Summary_Text_Files')
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Extract date from the first row and create filename
        if len(df) == 0:
            return False
        
        date = df.iloc[0]['Date']  # Get date from first row
        filename = f'Summary_{date}.txt'
        file_path = os.path.join(output_dir, filename)
        
        # Calculate overall day statistics
        total_percent_change = df['Percent Change'].sum()
        total_dollar_change = df['Dollar Change'].sum()
        trades_with_best_exit_over_06 = df[df['Best Exit Percent'] > 0.6]
        num_trades_over_06 = len(trades_with_best_exit_over_06)
        sum_percent_change_over_06 = trades_with_best_exit_over_06['Percent Change'].sum()
        
        # Start building the output content
        content = []
        content.append(f"=== SUMMARY FOR {date} ===\n")
        content.append("OVERALL DAY STATISTICS:")
        content.append(f"Day's Total % Change: {total_percent_change:.2f}%")
        content.append(f"Day's Total $ Change: ${total_dollar_change:.2f}")
        content.append(f"Number of trades where Best Exit Percent > 0.6%: {num_trades_over_06}")
        content.append(f"Sum of Percent Change for trades with Best Exit > 0.6%: {sum_percent_change_over_06:.2f}%")
        content.append("")  # Empty line for spacing
        
        # Calculate per-ticker statistics
        content.append("PER-TICKER STATISTICS:")
        unique_tickers = df['Ticker'].unique()
        
        for ticker in unique_tickers:
            ticker_df = df[df['Ticker'] == ticker]
            
            ticker_total_percent = ticker_df['Percent Change'].sum()
            ticker_total_dollar = ticker_df['Dollar Change'].sum()
            ticker_trades_over_06 = ticker_df[ticker_df['Best Exit Percent'] > 0.6]
            ticker_num_trades_over_06 = len(ticker_trades_over_06)
            ticker_sum_percent_over_06 = ticker_trades_over_06['Percent Change'].sum()
            
            content.append(f"\n{ticker}:")
            content.append(f"  Total % Change: {ticker_total_percent:.2f}%")
            content.append(f"  Total $ Change: ${ticker_total_dollar:.2f}")
            content.append(f"  Number of trades where Best Exit Percent > 0.6%: {ticker_num_trades_over_06}")
            content.append(f"  Sum of Percent Change for trades with Best Exit > 0.6%: {ticker_sum_percent_over_06:.2f}%")
        
        # Write to file (overwrite if exists)
        with open(file_path, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"Summary written to: {file_path}")
        return True

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return False


def Controller(): 
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    source_log_dir = os.path.join(script_dir, 'Holder_Strat_Trade_Logs')   # where the trade log csv's are
    market_data_dir = os.path.abspath(os.path.join(script_dir, '..', 'Csv_Files', '2_Raw_Market_Data'))
    summary_dir = os.path.join(script_dir, 'Summary_Csvs')

    # step 1) check directories/files, remove everything from the summary dir
    # make the output csv, check the trade logs and market data dir exist
    result = Check_Dirs([source_log_dir, market_data_dir, summary_dir])
    if (result == False):
        return

    # step 2) find all the valid trade days with their market data days. we skip days w/o market days
    # [[trade file path, market data file path], ...]
    file_pairs = Find_Valid_Files(market_data_dir, source_log_dir)
    if (file_pairs == []):
        return

    for files in file_pairs:
        trade_log_path = files[0]
        market_data_path = files[1]

        # step 3) clean the trade logs
        # get rid of the top and bottom areas so it's just the trades
        raw_df = Create_Df(trade_log_path)
        if not isinstance(raw_df, pd.DataFrame):
            return
        
        # step 4) create summary df and add stuff in
        df = Create_Summary_Df(raw_df)
        if not isinstance(df, pd.DataFrame):
            return
        
        df = Add_Running_Percents(df) # running percent by ticker and all together

        df = Add_Market_Data(df, market_data_path)
        if not isinstance(df, pd.DataFrame):
            return
        
        # df is ready to save as a csv, but I'll analyze it more to add final results to a supporting txt file
        result = Create_Summarized_Info_txt(df)
        if (result == False):
            return

        df.to_csv(f"{summary_dir}/Summary_{df.iloc[0]['Date']}.csv", index=False)

Controller()