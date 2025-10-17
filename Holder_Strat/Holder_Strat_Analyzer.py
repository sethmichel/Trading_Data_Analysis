import os
from re import X
import pandas as pd
import inspect
import sys
import glob

# Add parent directory to path so we can import Main_Globals
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Main_Globals

# Alternative approach using relative imports (requires __init__.py files):
# from .. import Main_Globals

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


'''
goal: create a csv with
date, ticker, held, entry time, exit time, time in trade, change percent, change dollar, running sum, total investment, trade type, 
Best Exit Percent, best exit price, best exit time in trade, worst exit percent, worst exit price, worst exit time in trade,
Entry Atr14, Entry Atr28, Entry Volatility Percent, Entry Volatility Ratio, Entry Adx28, Entry Adx14, Entry Adx7, Price Movement

held is bool for if it broke the target
trade type is buy or short
'''

def seconds_to_hms(seconds):
    """Convert seconds to hours:minutes:seconds format"""
    if seconds is None or pd.isna(seconds):
        return None
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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

        print("Step 1 completed: All directories verified/created. Old summary csv's deleted\n")
        return True

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return False


def Find_Valid_Files(market_data_dir, trade_log_dir, do_set_days):
    try:
        all_trade_dates = []
        all_trade_paths = []
        all_market_data_dates = []
        all_market_data_paths = []
        valid_file_pairs = []  # list of lists [trade file, market file]

        for file in os.listdir(trade_log_dir):
            if file.endswith('.csv'):
                parts = file.split('-')[0:3]
                date = f"{parts[0]}-{parts[1]}-{parts[2]}"
                # if we're doing all days, add them
                if (do_set_days == []):
                    all_trade_dates.append(date)
                    all_trade_paths.append(f"{trade_log_dir}/{file}")
                # if we're doing set days, check the date is valid
                elif (date in do_set_days):
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


def Sort_By_Time(raw_df):
    try:
        # Extract time portion from Date column and sort in descending order
        # Date format: "month/day/year hour:minute:second"
        raw_df['temp_time'] = raw_df['Date'].str.split(' ').str[1]
        
        # Sort by time in descending order (newest time first, oldest time last)
        raw_df = raw_df.sort_values('temp_time', ascending=False).reset_index(drop=True)
        
        # Remove the temporary time column
        raw_df = raw_df.drop('temp_time', axis=1)
        
        return raw_df

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


def Create_Df(csv_path):
    try:
        raw_df = pd.read_csv(csv_path, skiprows = 6)

        # delete all the canceled order rows
        try:
            index_to_delete = raw_df[raw_df['Unnamed: 0'] == 'Canceled Orders'].index[0]
            raw_df = raw_df[:index_to_delete]
        except:
            # there's no canceled orders: this happens with backtest logs
            pass

        # drop useless columns
        columns_to_drop = ["Unnamed: 0", "Spread", "Exp", "Strike", "Type", "Price Improvement", "Order Type"]
        if "Unnamed: 1" in raw_df.columns:
            columns_to_drop.append("Unnamed: 1") # idk what causes this to be in it. the original code for this thought it was always in live data, but 8/25/25 live didn't have it

        raw_df = raw_df.drop(columns=columns_to_drop)   # 'errors="ignore"' prevents errors if a column doesn't exist

        raw_df.rename(columns={raw_df.columns[0]: "Date"}, inplace = True)   # rename the first column to "Date"
        raw_df.rename(columns={raw_df.columns[4]: "Ticker"}, inplace = True)

        raw_df = Sort_By_Time(raw_df)

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
                            'Running Percent By Ticker': None,
                            'Running Percent All': None,
                            'Total Investment': round(total_investment, 2),
                            'Entry Price': round(entry_price, 4),
                            'Exit Price': round(exit_price, 4),
                            'Trade Type': trade_type,
                            'Qty': None if has_multiple_entries else initial_qty,  # Use initial quantity unless multiple entries
                            'Best Sl': None,
                            'Best Exit Price': None,
                            'Best Exit Time In Trade': None,
                            'Worst Exit Price': None,
                            'Worst Exit Percent': None,
                            'Worst Exit Time In Trade': None,
                            'Entry Atr14': None,
                            'Entry Atr28': None,
                            'Entry Volatility Percent': None,
                            'Entry Volatility Ratio': None,
                            'Entry Adx28': None,
                            'Entry Adx14': None,
                            'Entry Adx7': None,
                            'Price Movement': None,
                            #'TEST optimal sl result': None,
                            'TEST Optimal sl Hit': None,
                            'Original Holding Reached': None,
                            'TEST Optimal sl Reached Holding': None,
                            'Original Best Exit Percent': None,
                            'TEST Optimal sl Best Exit Percent': None,
                            'Original Percent Change': round(percent_change, 2),
                            'TEST Optimal sl Percent Change': None,
                            'TEST -0.4 sl Benchmark': None,
                            'TEST -0.5 sl Benchmark': None,
                            'TEST -0.6 sl Benchmark': None
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
            percent_change = row['Original Percent Change']

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
    try:
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
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None

# direction is 'buy' or 'short'
def Add_Best_Worst_Info(ticker_df, starting_row, entry_price, direction, exit_time_seconds):
    try:
        # --- start at starting_row which is the first rows index of ticker_df. iterate over ticker_df from there
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
        
        # Start iterating from starting_row (first row of the trade)
        # starting_row is the dataframe index, so we need to get the position in the dataframe
        start_position = ticker_df.index.get_loc(starting_row)
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
        entry_time_seconds = ticker_df.loc[starting_row, 'Time']
        best_time_in_trade = best_time_seconds - entry_time_seconds if best_time_seconds else 0
        worst_time_in_trade = worst_time_seconds - entry_time_seconds if worst_time_seconds else 0
        
        return {
            'Best Exit Price': round(best_price, 4),
            'Original Best Exit Percent': round(best_percent_change, 2),
            'Best Exit Time In Trade': best_time_in_trade,
            'Worst Exit Price': round(worst_price, 4),
            'Worst Exit Percent': round(worst_percent_change, 2),
            'Worst Exit Time In Trade': worst_time_in_trade
        }

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


# this tracks all 0.1% movements including 0.0, with duplicate tracking and oscillation detection
def Add_Price_Movement(ticker_df, entry_price, starting_row, exit_time_seconds, direction):
    try:
        price_movement = []

        start_position = ticker_df.index.get_loc(starting_row)
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
            starting_row = None
            for ticker_idx, row in ticker_df.iterrows():
                # start at the first row with a greater than or equal to entry time
                if (row['Time'] < entry_seconds):
                    continue

                # This is the first row at or after entry time
                if starting_row is None:
                    starting_row = ticker_idx

                    df.at[idx, 'Entry Atr14'] = row['Atr14']
                    df.at[idx, 'Entry Atr28'] = row['Atr28']
                    df.at[idx, 'Entry Volatility Percent'] = row['Volatility Percent']
                    df.at[idx, 'Entry Volatility Ratio'] = row['Volatility Ratio']
                    df.at[idx, 'Entry Adx28'] = row['Adx28']
                    df.at[idx, 'Entry Adx14'] = row['Adx14']
                    df.at[idx, 'Entry Adx7'] = row['Adx7']
                    break

            # add Best/worst Exit info. Loop over the data from start time to end time
            if starting_row is not None:
                best_worst_info = Add_Best_Worst_Info(ticker_df, starting_row, entry_price, direction, exit_seconds)
                if (best_worst_info == None):
                    return None

                df.at[idx, 'Best Exit Price'] = best_worst_info['Best Exit Price']
                df.at[idx, 'Original Best Exit Percent'] = best_worst_info['Original Best Exit Percent']
                df.at[idx, 'Best Exit Time In Trade'] = seconds_to_hms(best_worst_info['Best Exit Time In Trade'])
                df.at[idx, 'Worst Exit Price'] = best_worst_info['Worst Exit Price']
                df.at[idx, 'Worst Exit Percent'] = best_worst_info['Worst Exit Percent']
                df.at[idx, 'Worst Exit Time In Trade'] = seconds_to_hms(best_worst_info['Worst Exit Time In Trade'])
                df.at[idx, 'Best Sl'] = Find_Best_Sl(ticker_df, starting_row, direction) # returns x% or 'not found'

                if (best_worst_info['Original Best Exit Percent'] >= 0.6):
                    df.at[idx, 'Original Holding Reached'] = True
                else:
                    df.at[idx, 'Original Holding Reached'] = False

                price_movement_list = Add_Price_Movement(ticker_df, entry_price, starting_row, exit_seconds, direction)
                df.at[idx, 'Price Movement'] = str(price_movement_list) if price_movement_list is not None else None

                #df = Create_Estimate_Columns(df, idx, ticker_df, exit_seconds)
                if (isinstance(df, pd.DataFrame) == False):
                    return

        return df

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


'''
starting at each trades start row, track worst exit % until 1) +0.6% is reached, 2) end of data is reached
if 1 then write 0.1 worse than worst exit, if 2 then write None
'''
def Find_Best_Sl(ticker_df, starting_row, direction):
    try:
        # Get entry price from the starting row
        entry_price = ticker_df.loc[starting_row, 'Price']
        
        # Initialize tracking variables
        worst_roi_percent = 0.0  # Start at 0% (no loss/gain)
        
        # Start iterating from starting_row
        start_position = ticker_df.index.get_loc(starting_row)
        
        for idx in range(start_position, len(ticker_df)):
            row = ticker_df.iloc[idx]
            current_price = row['Price']
            
            # Calculate current ROI percent based on direction
            if direction == 'buy':
                current_roi_percent = ((current_price - entry_price) / entry_price) * 100
            else:  # direction == 'short'
                current_roi_percent = ((entry_price - current_price) / entry_price) * 100
            
            # Update worst ROI (most negative for both buy and short)
            if current_roi_percent < worst_roi_percent:
                worst_roi_percent = current_roi_percent
            
            # Check if we've reached +0.6% ROI
            if current_roi_percent >= 0.6:
                # Return worst ROI but make it 0.1% worse
                return round(worst_roi_percent - 0.1, 2)
        
        # If we reach end of data without hitting 0.6%
        return 'not found'
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Create_Overall_Summary_Info_Txt(bulk_df):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'Summary_Text_Files')
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        filename = f'Overall_Summary.txt'
        file_path = os.path.join(output_dir, filename)
        
        total_percent_change = round(bulk_df['Original Percent Change'].sum(), 2)
        total_dollar_change = round(bulk_df['Dollar Change'].sum(), 2)
        trades_at_holding = bulk_df[bulk_df['Original Best Exit Percent'] > 0.6]
        num_trades_at_holding = len(trades_at_holding)
        sum_trades_at_holding = round(trades_at_holding['Original Percent Change'].sum(), 2)
        trades_failed = bulk_df[bulk_df['Original Percent Change'] < -0.3]
        num_trades_failed = len(trades_failed)
        sum_trades_failed = round(trades_failed['Original Percent Change'].sum(), 2)
        num_of_days = bulk_df['Date'].nunique()
        
        content = []
        content.append(f"=== SUMMARY ===\n")
        content.append("OVERALL STATISTICS:")
        content.append(f"Total % Change: {total_percent_change}%")
        content.append(f"Avg % Change/Day: {round(total_percent_change/num_of_days, 2)}%")
        content.append(f"Total $ Change: ${total_dollar_change} - fees")
        content.append(f"Number of Holding Trades (0.6%): {num_trades_at_holding}")
        content.append(f"Number of Failed Trades (~-0.4%): {num_trades_failed}")
        content.append(f"Sum of % Change for holding trades: {sum_trades_at_holding}%")
        content.append(f"Sum of % Change for failed trades: {sum_trades_failed}%")
        content.append(f"Avg % Change for holding trades: {round(sum_trades_at_holding/num_of_days, 2)}%")
        content.append(f"Avg % Change for failed trades: {round(sum_trades_failed/num_of_days, 2)}%")

        content.append("PER-TICKER STATISTICS:")
        unique_tickers = bulk_df['Ticker'].unique()
        
        for ticker in unique_tickers:
            ticker_df = bulk_df[bulk_df['Ticker'] == ticker]
            
            ticker_total_percent = round(ticker_df['Original Percent Change'].sum(), 2)
            ticker_total_dollar = round(ticker_df['Dollar Change'].sum(), 2)
            ticker_trades_at_holding = ticker_df[ticker_df['Original Best Exit Percent'] > 0.6]
            ticker_num_trades_at_holding = len(ticker_trades_at_holding)
            ticker_sum_percent_at_holding = round(ticker_trades_at_holding['Original Percent Change'].sum(), 2)
            ticker_trades_failed = ticker_df[ticker_df['Original Percent Change'] < -0.3]
            ticker_num_trades_failed = len(ticker_trades_failed)
            ticker_sum_percent_failed = round(ticker_trades_failed['Original Percent Change'].sum(), 2)
            
            content.append(f"\n{ticker}:")
            content.append(f"  Total % Change: {ticker_total_percent:.2f}%")
            content.append(f"  Avg % Change/Day: {round(ticker_total_percent/num_of_days, 2)}%")
            content.append(f"  Total $ Change: ${ticker_total_dollar:.2f}")
            content.append(f"  Number of Holding Trades (0.6%): {ticker_num_trades_at_holding}")
            content.append(f"  Number of Failed Trades (~-0.4%): {ticker_num_trades_failed}")
            content.append(f"  Sum of % Change for holding trades: {ticker_sum_percent_at_holding:.2f}%")
            content.append(f"  Sum of % Change for failed trades: {ticker_sum_percent_failed:.2f}%")
            content.append(f"  Avg % change for holding trades: {round(ticker_sum_percent_at_holding/num_of_days, 2)}%")
            content.append(f"  Avg % Change for failed trades: {round(ticker_sum_percent_failed/num_of_days, 2)}%")
        
        # Write to file (overwrite if exists)
        with open(file_path, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"Overall summary written to: {file_path}")
        return True

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Create_Summarized_Info_txt(df, date):
    try:
        # --- goal: extract summarized results from df and write to a txt file
        # --- df is now a completed normalized trade log dataframe, each line is all details of 1 trade
        # --- write at the top: days % change, days $ change, number of trades where Best Exit Percent is > 0.6,
        #     what's the sum of the percent change of trades who's Best Exit Percent is > 0.6
        # --- do the same as the previous point but for each unique ticker in the 'Ticker' column
        
        # Get the script directory and create output directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'Summary_Text_Files')
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        filename = f'Summary_{date}.txt'
        file_path = os.path.join(output_dir, filename)
        
        total_percent_change = df['Original Percent Change'].sum()
        total_dollar_change = df['Dollar Change'].sum()
        trades_at_holding = df[df['Original Best Exit Percent'] > 0.6]
        num_trades_at_holding = len(trades_at_holding)
        sum_trades_at_holding = trades_at_holding['Original Percent Change'].sum()
        trades_failed = df[df['Original Percent Change'] < -0.3]
        num_trades_failed = len(trades_failed)
        sum_trades_failed = trades_failed['Original Percent Change'].sum()
        
        content = []
        content.append(f"=== SUMMARY FOR {date} ===\n")
        content.append("OVERALL DAY STATISTICS:")
        content.append(f"Day's Total % Change: {total_percent_change:.2f}%")
        content.append(f"Day's Total $ Change: ${total_dollar_change:.2f} - fees")
        content.append(f"Number of Holding Trades (0.6%): {num_trades_at_holding}")
        content.append(f"Number of Failed Trades (~-0.4%): {num_trades_failed}")
        content.append(f"Sum of % Change for holding trades: {sum_trades_at_holding:.2f}%")
        content.append(f"Sum of % Change for failed trades: {sum_trades_failed:.2f}%")
        
        content.append("PER-TICKER STATISTICS:")
        unique_tickers = df['Ticker'].unique()
        
        for ticker in unique_tickers:
            ticker_df = df[df['Ticker'] == ticker]
            
            ticker_total_percent = ticker_df['Original Percent Change'].sum()
            ticker_total_dollar = ticker_df['Dollar Change'].sum()
            ticker_trades_at_holding = ticker_df[ticker_df['Original Best Exit Percent'] > 0.6]
            ticker_num_trades_at_holding = len(ticker_trades_at_holding)
            ticker_sum_percent_at_holding = ticker_trades_at_holding['Original Percent Change'].sum()
            ticker_trades_failed = ticker_df[ticker_df['Original Percent Change'] < -0.3]
            ticker_num_trades_failed = len(ticker_trades_failed)
            ticker_sum_percent_failed = ticker_trades_failed['Original Percent Change'].sum()
            
            content.append(f"\n{ticker}:")
            content.append(f"  Total % Change: {ticker_total_percent:.2f}%")
            content.append(f"  Total $ Change: ${ticker_total_dollar:.2f}")
            content.append(f"  Number of Holding Trades (0.6%): {ticker_num_trades_at_holding}")
            content.append(f"  Number of Failed Trades (~-0.4%): {ticker_num_trades_failed}")
            content.append(f"  Sum of % Change for holding trades: {ticker_sum_percent_at_holding:.2f}%")
            content.append(f"  Sum of % Change for failed trades: {ticker_sum_percent_failed:.2f}%")
        
        # Write to file (overwrite if exists)
        with open(file_path, 'w') as f:
            f.write('\n'.join(content))
        
        print(f"Summary written to: {file_path}")
        return True

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return False


# if a trade reached 0.6% and is thus a 'holder' trade, estimate roi in various senarios
def Create_Estimate_Columns(df, idx, ticker_df, exit_seconds):
    try:
        # Get the trade row from the dataframe at the given index
        trade = df.iloc[idx]
        
        # step 1) skip trade if it's not a holder trade. A holder trade is one who's 'Original Best Exit Percent' value is at least 0.6
        best_exit_percent = trade['Original Best Exit Percent']
        if best_exit_percent is None or best_exit_percent < 0.6:
            return df  # Not a holder trade, skip analysis but return df
        
        # step 2) if it's a holder trade, iterate over market data to find the price at each target point. market data is organized by ticker, in time ascending order
        # --- When finding these target point values, if the 'Price' column value reaches the 'Entry Price' then record 0
        # --- when finding these target point values, if the market data ends (reached end of file) record the final rows values.
        #        for example, when finding the roi after 1.5 hours, if the price reaches the entry price record 0. or if we reach the end of the 
        #        file record the roi % using the final price value. do the same with finding roi %, if we reach the end of the file record the % 
        #        using the final price
        # --- target points (time): 1 hour, 1.5 hours, 2 hours, 2.5 hours, 3 hours
        # --- target points (% roi): 1%, 2%, 2.5%, 3%, 3.5%, 4% (find % roi compared to trades 'Entry Price') 
        # --- you can see this sort of market data analyzing being done in Add_Best_Worst_Info()
        # --- df is the dataframe containing the trades, idx is the index of the df for this trade, trade is the row of the df (the current trade)
        #     ticker_df is a dataframe of market data just for this ticker, where 'Time' is the timestamp in seconds since midnight.
        # step 3) record all these values directly to the df. create the df columns in the normalized_trades variable. line 228

        # Get trade information
        entry_price = trade['Entry Price']
        direction = trade['Trade Type']
        
        entry_time_parts = trade['Entry Time'].split(':')
        entry_seconds = int(entry_time_parts[0]) * 3600 + int(entry_time_parts[1]) * 60 + int(entry_time_parts[2])
        
        # Find the starting index for the trade in ticker_df
        start_idx = None
        for ticker_idx, row in ticker_df.iterrows():
            if row['Time'] >= entry_seconds:
                start_idx = ticker_idx
                break
        
        # step 2) iterate over market data to find the price at each target point
        
        # Define target time points (in seconds from entry)
        time_targets = {
            'ROI_1_Hour': 3600,        # 1 hour
            'ROI_1_5_Hours': 5400,     # 1.5 hours
            'ROI_2_Hours': 7200,       # 2 hours
            'ROI_2_5_Hours': 9000,     # 2.5 hours
            'ROI_3_Hours': 10800       # 3 hours
        }
        
        # Define target ROI percentages
        roi_targets = {
            'percent_1': 1.0,
            'percent_2': 2.0,
            'percent_2_5': 2.5,
            'percent_3': 3.0,
            'percent_3_5': 3.5,
            'percent_4': 4.0
        }
        
        # Initialize results
        time_results = {}
        roi_results = {}
        
        # Track if we ever hit entry price AFTER reaching 0.6% (means trade went to breakeven/loss)
        hit_entry_price_after_06 = False
        reached_06_percent = False
        
        # Get starting position in the dataframe
        start_position = ticker_df.index.get_loc(start_idx)
        
        # Iterate through market data from entry time onwards
        for idx_pos in range(start_position, len(ticker_df)):
            row = ticker_df.iloc[idx_pos]
            current_time = row['Time']
            current_price = row['Price']
            
            # Calculate time elapsed since entry (in seconds)
            time_elapsed = current_time - entry_seconds
            
            # Calculate current ROI percentage
            if direction == 'buy':
                current_roi = ((current_price - entry_price) / entry_price) * 100
            else:  # direction == 'short'
                current_roi = ((entry_price - current_price) / entry_price) * 100
            
            # Check if we've reached 0.6% for the first time
            if not reached_06_percent and current_roi >= 0.6:
                reached_06_percent = True
            
            # Only check if price has reached entry price AFTER we've reached 0.6% AND after the actual trade end time
            if reached_06_percent and current_time >= exit_seconds:
                if direction == 'buy' and current_price <= entry_price:
                    hit_entry_price_after_06 = True
                elif direction == 'short' and current_price >= entry_price:
                    hit_entry_price_after_06 = True
            
            # Check time-based targets: "what is the roi if I held the trade for this long?"
            for time_key, target_seconds in time_targets.items():
                if time_key not in time_results and time_elapsed >= target_seconds:
                    if hit_entry_price_after_06:
                        time_results[time_key] = 0.0  # Hit entry price after reaching 0.6%
                    else:
                        time_results[time_key] = round(current_roi, 2)
            
            # Check ROI percentage-based targets: did it ever reach this roi percent?
            # Only check these targets AFTER we've reached 0.6%
            if reached_06_percent:
                for roi_key, target_percent in roi_targets.items():
                    if roi_key not in roi_results:
                        if hit_entry_price_after_06:
                            roi_results[roi_key] = 0.0  # Hit entry price after reaching 0.6%, so result is 0
                        elif current_roi >= target_percent:
                            roi_results[roi_key] = target_percent  # Reached target, so result is the target
        
        # Handle any remaining unfilled results using final market data
        if len(ticker_df) > 0:
            final_row = ticker_df.iloc[-1]
            final_price = final_row['Price']
            final_time = final_row['Time']
            
            # Calculate final ROI
            if direction == 'buy':
                final_roi = ((final_price - entry_price) / entry_price) * 100
            else:  # direction == 'short'
                final_roi = ((entry_price - final_price) / entry_price) * 100
            
            # Fill in any missing time targets with final values
            for time_key in time_targets.keys():
                if time_key not in time_results:
                    if hit_entry_price_after_06:
                        time_results[time_key] = 0.0
                    else:
                        time_results[time_key] = round(final_roi, 2)
            
            # Fill in any missing ROI targets with final ROI if not hit entry price after 0.6%
            # Only fill these if the trade reached 0.6% (was a holder trade)
            if reached_06_percent:
                for roi_key, target_percent in roi_targets.items():
                    if roi_key not in roi_results:
                        if hit_entry_price_after_06:
                            roi_results[roi_key] = 0.0
                        else:
                            roi_results[roi_key] = round(final_roi, 2)  # Final ROI if target not reached
        
        # step 3) record all these values directly to the df
        for time_key, roi_value in time_results.items():
            df.at[idx, time_key] = roi_value
        
        for roi_key, roi_value in roi_results.items():
            df.at[idx, roi_key] = roi_value

        return df

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


def Get_All_Trade_Log_Date():
    dates = []
    trade_log_dir = "Holder_Strat/Holder_Strat_Trade_Logs"

    for file in os.listdir(trade_log_dir):
        if file.endswith('.csv'):
            date = "-".join(file.split('-')[0:3])
            if ".csv" in date:
                date = date[:-4] # live days do this, on demand days don't
            dates.append(date)

    return dates


def Create_Bulk_Summaries(summary_dir):
    """
    Creates a bulk_summaries.csv file by combining all summary CSV files
    that do not start with "bulk" (case insensitive) from the summary directory.
    
    Args:
        summary_dir (str): Directory containing the summary CSV files
    
    Returns:
        str: Path to the created bulk CSV file, or None if no files were found
    """
    try:
        # Output file path
        output_file = os.path.join(summary_dir, "bulk_summaries.csv")
        
        # List to store all dataframes
        all_dataframes = []
        
        print("Looking for summary CSV files...")
        
        if os.path.exists(summary_dir):
            print(f"Checking directory: {summary_dir}")
            
            # Get all CSV files in the directory
            csv_files = glob.glob(os.path.join(summary_dir, "*.csv"))
            
            for csv_file in csv_files:
                filename = os.path.basename(csv_file).lower()
                
                # Skip files that start with "bulk" (case insensitive)
                if filename.startswith("bulk"):
                    print(f"Skipping bulk file: {csv_file}")
                    continue
                
                # Only process files that start with "summary"
                if filename.startswith("summary"):
                    print(f"Processing file: {csv_file}")
                    
                    try:
                        # Read the CSV file
                        df = pd.read_csv(csv_file)
                        
                        # Check if the dataframe is not empty
                        if not df.empty:
                            all_dataframes.append(df)
                            print(f"  Added {len(df)} rows from {os.path.basename(csv_file)}")
                        else:
                            print(f"  Skipped empty file: {os.path.basename(csv_file)}")
                    
                    except Exception as e:
                        print(f"  Error reading {csv_file}: {str(e)}")
        else:
            print(f"Directory does not exist: {summary_dir}")
            return None
        
        # Check if we found any files to combine
        if not all_dataframes:
            print("No summary CSV files found to combine!")
            return None
        
        print(f"\nCombining {len(all_dataframes)} CSV files...")
        
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Sort by Date and then by Entry Time for better organization
        try:
            # Convert Date column to datetime for proper sorting
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%m-%d-%y', errors='coerce')
            combined_df = combined_df.sort_values(['Date', 'Entry Time'], ascending=[True, True])
            
            # Convert Date back to original format
            combined_df['Date'] = combined_df['Date'].dt.strftime('%m-%d-%y')
        except Exception as e:
            print(f"Warning: Could not sort by date/time: {str(e)}")
            print("Proceeding without sorting...")
        
        # Save the combined dataframe to CSV
        combined_df.to_csv(output_file, index=False)
        
        print(f"\nBulk summary file created successfully!")
        print(f"Output file: {output_file}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Columns: {len(combined_df.columns)}")
        
        # Display some basic statistics
        if 'Ticker' in combined_df.columns:
            print(f"Unique tickers: {combined_df['Ticker'].nunique()}")
            print(f"Most common tickers:")
            print(combined_df['Ticker'].value_counts().head(5))
        
        return output_file, combined_df

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


def Controller(do_all_trade_logs):
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    source_log_dir = os.path.join(script_dir, 'Holder_Strat_Trade_Logs')   # where the trade log csv's are
    market_data_dir = "Holder_Strat/Approved_Checked_Market_Data"
    summary_dir = os.path.join(script_dir, 'Summary_Csvs')

    # step 1) check directories/files, remove everything from the summary dir
    # make the output csv, check the trade logs and market data dir exist
    result = Check_Dirs([source_log_dir, market_data_dir, summary_dir])
    if (result == False):
        print("bad check directories")
        return

    # step 2) find all the valid trade days with their market data days. we skip days w/o market days
    # [[trade file path, market data file path], ...]
    if (do_all_trade_logs == 'no'):
        do_set_days = ['06-24-2025'] # 'month-day-year'
    elif (do_all_trade_logs == 'yes'):
        do_set_days = Get_All_Trade_Log_Date()
    else:
        return
    file_pairs = Find_Valid_Files(market_data_dir, source_log_dir, do_set_days)
    if (file_pairs == []):
        print(f"no fail pairs found for date")
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
            print(f"ERROR**** just created summary df isn't a dataframe")
            return
        
        df = Add_Running_Percents(df) # running percent by ticker and all together

        df = Add_Market_Data(df, market_data_path)
        if not isinstance(df, pd.DataFrame):
            return
        
        month, day, year = (df.iloc[0]['Date']).split('-')
        if (len(month) == 1):
            month = '0' + month
        if (len(day) == 1):
            day = '0' + day
        if (len(year) == 2):
            year = '20' + year
        date = f"{month}-{day}-{year}"

        # df is ready to save as a csv, but I'll analyze it more to add final results to a supporting txt file
        result = Create_Summarized_Info_txt(df, date)
        if (result == False):
            return
        
        df.to_csv(f"{summary_dir}/Summary_{date}.csv", index=False)

    # Create bulk summaries CSV after all individual summaries are created
    print("\nCreating bulk summaries CSV...")
    bulk_df = Create_Bulk_Summaries(summary_dir)
    if isinstance(bulk_df, pd.DataFrame):
        print(f"Bulk summaries created successfully")
        Create_Overall_Summary_Info_Txt(bulk_df)
    else:
        print("Failed to create bulk summaries CSV")


Controller(do_all_trade_logs='yes')