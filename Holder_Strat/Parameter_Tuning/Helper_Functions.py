from datetime import datetime
import os
import pandas as pd
import json


def bulk_csv_date_converter(date):
    parts = date.split('-')

    if (len(parts[0]) == 1):
        parts[0] = f"0{parts[0]}"
    if (len(parts[1]) == 1):
        parts[1] = f"0{parts[1]}"
    if (len(parts[2]) == 2):
        parts[2] = f"20{parts[2]}"
    
    return '-'.join(parts)


def Check_We_Have_Data_For_Trade(market_df, entry_time):
    # Get the final timestamp from market data
    final_market_time = market_df.iloc[-1]['Time']
    
    # Convert times to datetime objects for comparison
    entry_time_obj = datetime.strptime(entry_time, '%H:%M:%S').time()
    final_market_time_obj = datetime.strptime(final_market_time, '%H:%M:%S').time()
    
    # Convert to seconds for easier comparison
    entry_seconds = entry_time_obj.hour * 3600 + entry_time_obj.minute * 60 + entry_time_obj.second
    final_market_seconds = final_market_time_obj.hour * 3600 + final_market_time_obj.minute * 60 + final_market_time_obj.second
    
    if entry_seconds > final_market_seconds:
        return False
    else:
        return True
    

# Check for time gaps larger than 6 seconds between consecutive rows in market data.
def Confirm_No_Market_Data_Time_Gaps(market_df, file_path):
    # Convert Time column to datetime objects for comparison
    time_objects = []
    for time_str in market_df['Time']:
        # Parse hour:minute:second format
        time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
        time_objects.append(time_obj)
    
    # Check for gaps between consecutive rows
    for i in range(1, len(time_objects)):
        prev_time = time_objects[i-1]
        curr_time = time_objects[i]
        
        # Calculate time difference in seconds
        prev_seconds = prev_time.hour * 3600 + prev_time.minute * 60 + prev_time.second
        curr_seconds = curr_time.hour * 3600 + curr_time.minute * 60 + curr_time.second
        
        time_diff = curr_seconds - prev_seconds
        
        # Since all data is from the same day, we only care about positive time gaps
        # Negative differences are just data collection timing variations (1-2 seconds backwards)
        # Only check for gaps larger than 6 seconds in the forward direction
        if time_diff > 6:
            prev_time_str = prev_time.strftime('%H:%M:%S')
            curr_time_str = curr_time.strftime('%H:%M:%S')
            raise ValueError(f"Time gap of {time_diff} seconds found between {prev_time_str} and {curr_time_str} (gap > 6 seconds) for file {file_path}")
    
    return True


def Load_Market_Data_Dictionary(bulk_df):
    """
    Load market data CSV files into a dictionary where key is date and value is dataframe.
    then drop unneeded columns
    then split it by ticker.
    
    Returns:
        {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    """
    market_data_dir = "Holder_Strat/Approved_Checked_Market_Data"
    market_data_dict = {}   # dictionary to store market data

    # Extract unique dates from the 'Date' column
    unique_dates = bulk_df['Date'].unique().tolist()
    # date needs to be mm-dd-yyyy format
    for i in range (len(unique_dates)):
        unique_dates[i] = bulk_csv_date_converter(unique_dates[i])

    print(f"Found {len(unique_dates)} unique dates in bulk_summaries.csv")
    
    # Step 1: Get list of market data files
    market_data_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]    
    
    # Step 2: Process each market data file
    for filename in market_data_files:
        date_from_filename = filename.split('_')[3]
        if ('.csv' in date_from_filename):
            date_from_filename = date_from_filename[0:-4]

        # Only load if date is in our unique dates list
        if date_from_filename in unique_dates:
            file_path = os.path.join(market_data_dir, filename)

            # Load the market data CSV
            market_df = pd.read_csv(file_path)
            if (Confirm_No_Market_Data_Time_Gaps(market_df, file_path) == True):
                # Keep only the specified columns
                required_columns = ['Ticker', 'Price', 'Volatility Percent', 'Time']
                market_df = market_df[required_columns]
                market_data_dict[date_from_filename] = market_df
                print(f"Loaded market data for {date_from_filename}")
            else:
                msg = f"time gap in {file_path}, crashing..."
                print(msg)
                raise ValueError(msg)

        else:
            print(f"Skipping {filename} - date {date_from_filename} not in bulk_summaries.csv")
    
    print(f"\nSuccessfully loaded {len(market_data_dict)} market data files")

    # step 3) go through the dictionary of df's and split by ticker
    nested_market_data_dict = {}
    
    for date, df in market_data_dict.items():
        # Get unique tickers for this date
        unique_tickers = df['Ticker'].unique()
        nested_market_data_dict[date] = {}
        
        # Split dataframe by ticker
        for ticker in unique_tickers:
            ticker_df = df[df['Ticker'] == ticker].copy()
            # Reset index to ensure sequential indexing starting from 0
            ticker_df = ticker_df.reset_index(drop=True)
            
            # Add 'Time Since Market Open' column (market opens at 6:30:00 AM)
            market_open_minutes = 6 * 60 + 30  # 6:30 AM in minutes
            time_since_market_open = []
            
            for time_str in ticker_df['Time']:
                time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
                current_minutes = time_obj.hour * 60 + time_obj.minute + time_obj.second / 60.0
                minutes_since_open = current_minutes - market_open_minutes
                # Round to nearest minute
                minutes_since_open = round(minutes_since_open)
                time_since_market_open.append(minutes_since_open)
            
            ticker_df['Time Since Market Open'] = time_since_market_open
            nested_market_data_dict[date][ticker] = ticker_df
            
        print(f"Split {date} data into {len(unique_tickers)} tickers: {list(unique_tickers)}")

    return nested_market_data_dict


def Add_Trade_Id(bulk_df):
    # Add a new 'Id' column that counts the trades starting from 1
    bulk_df['Id'] = range(1, len(bulk_df) + 1)

    return bulk_df


'''
-Make roi dictionary of lists for each trade -> {trade_id: [roi values], ...}
-iterate over the data. each trades stop loss is an roi of -0.4% until the trade reaches +0.6% roi, at which time the stop loss changes to 0% (entry price)
-For each trade, get its market data dataframe (market_data_dict_by_ticker[date][ticker]), start at the starting point for the trade (iterate over the 
data until we find the entry_time or the first timestamp after the entry_time (in case it's off by 1 second)). Then iterate over each row of market data;
for each market data row, add the trades roi to roi_list, note that trades direction can be "buy" (+roi is when the price is higher) or "short" (+roi is when the 
price is lower).

goal: create a list for each trade that has second by second roi
'''
def Create_Roi_Dictionary_For_Trades(bulk_df, market_data_dict_by_ticker, largest_sl_value):
    trade_start_indexes = {}
    trade_end_timestamps = {}
    roi_dictionary = {}
    skip_dates = []

    print("\nmaking roi dictionary...")

    for idx, row in bulk_df.iterrows():
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025
        if (date in skip_dates):
            continue

        entry_time = row['Entry Time']               # hour:minute:second
        trade_id = row['Id']
        entry_price = row['Entry Price']
        ticker = row['Ticker']
        direction = row['Trade Type']
        roi_list = []
        
        # Check if we have market data for this date and ticker
        if date not in market_data_dict_by_ticker:
            print(f"No market data found for date {date}")
            skip_dates.append(date)
            continue  
            
        if ticker not in market_data_dict_by_ticker[date]:
            msg = f"No market data found for ticker {ticker} on date {date}"
            print(msg)
            raise ValueError(msg)
            
        market_df = market_data_dict_by_ticker[date][ticker].copy()  # market data df for this ticker and date
        
        # Skip trade if entry time is after final market data time
        if (Check_We_Have_Data_For_Trade(market_df, entry_time) == False):
            print(f"Skipping trade {trade_id}: entry time {entry_time} is after final market data time")
            continue
        
        # Find the starting point in market data (entry time or first timestamp after)
        entry_found = False
        start_index = 0
        
        for i in range(len(market_df)):
            market_time = market_df.iloc[i]['Time']
            if market_time >= entry_time:
                start_index = i
                entry_found = True
                trade_start_indexes[trade_id] = i
                break
        
        if not entry_found:
            msg = f"Entry time {entry_time} not found in market data for trade {trade_id}"
            print(msg)
            raise ValueError(msg)
            
        # Track stop loss state
        stop_loss_triggered = False
        stop_loss_updated = False  # Track if stop loss changed from -0.4% to 0%
        
        # Iterate through market data starting from entry point
        for i in range(start_index, len(market_df)):
            current_price = market_df.iloc[i]['Price']
            current_time = market_df.iloc[i]['Time']
            
            # get roi
            if direction == 'buy':
                roi = round(((current_price - entry_price) / entry_price) * 100, 2)
            elif direction == 'short':
                roi = round(((entry_price - current_price) / entry_price) * 100, 2)
            else:
                msg = f"Unknown trade direction: {direction} for trade {trade_id}"
                print(msg)
                raise ValueError(msg)
            
            # Check stop loss conditions
            if stop_loss_triggered == False:
                if roi <= largest_sl_value:
                    stop_loss_triggered = True
                    roi_list.append(roi)
                    break

                elif roi >= 0.6 and stop_loss_updated == False:
                    stop_loss_updated = True
                    roi_list.append(roi)

                elif stop_loss_updated == True and roi <= 0:
                    stop_loss_triggered = True
                    roi_list.append(roi)
                    break

                else:
                    roi_list.append(roi)
            else:
                # Stop loss already triggered, no more data needed
                break
        
        trade_end_timestamps[trade_id] = current_time
        roi_dictionary[trade_id] = roi_list
    
    # Save ROI dictionary and related data to file
    roi_file_path = "Holder_Strat/Parameter_Tuning/model_files_and_data/roi_dictionary_saved.json"
    
    # Combine all trade data into one dictionary
    trade_data = {
        "roi_dictionary": roi_dictionary,
        "trade_end_timestamps": trade_end_timestamps,
        "trade_start_indexes": trade_start_indexes
    }
    
    with open(roi_file_path, 'w') as f:
        json.dump(trade_data, f, indent=2)
    print(f"Trade data (ROI dictionary, end timestamps, start indexes) saved to {roi_file_path}")
    
    print(f"roi dictionary done. Total trades processed: {len(roi_dictionary)}")

    return roi_dictionary, trade_end_timestamps, trade_start_indexes


# return (roi_dictionary, trade_end_timestamps, trade_start_indexes)
def Load_Roi_Dictionary_And_Values():
    roi_file_path = "Holder_Strat/Parameter_Tuning/model_files_and_data/roi_dictionary_saved.json"
    
    try:
        with open(roi_file_path, 'r') as f:
            trade_data = json.load(f)
        
        roi_dictionary = trade_data["roi_dictionary"]
        trade_end_timestamps = trade_data["trade_end_timestamps"]
        trade_start_indexes = trade_data["trade_start_indexes"]
        
        print(f"Successfully loaded trade data from {roi_file_path}")
        print(f"Loaded {len(roi_dictionary)} trades")
        
        return roi_dictionary, trade_end_timestamps, trade_start_indexes
        
    except FileNotFoundError:
        print(f"Error: File {roi_file_path} not found")
    except KeyError as e:
        print(f"Error: Missing key {e} in the saved data file")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {roi_file_path}: {e}")


