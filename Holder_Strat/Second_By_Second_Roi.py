import pandas as pd
from datetime import datetime
import os

'''
Use to assess if I should take profits when it reaches a certain percent
finds the second by second roi I had during the day by the sum of all tickers
'''

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

            market_df = pd.read_csv(file_path)
            # Keep only the specified columns
            required_columns = ['Ticker', 'Price', 'Volatility Percent', 'Time']
            market_df = market_df[required_columns]
            market_data_dict[date_from_filename] = market_df
            print(f"Loaded market data for {date_from_filename}")

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


# for each ticker, for each trade, make second by second roi's {ticker: {time: [trade_id, roi]}}
def Gather_Data(bulk_df, market_data_dict_by_ticker, tickers):
    holding_dict = {}
    for ticker in tickers:
        holding_dict[ticker] = {}
    
    roi_dictionary = {} # {date: ticker: {time: [trade_id, roi]}}
    skip_dates = []
    stop_loss = -0.4

    print("\nmaking roi dictionary...")

    for idx, row in bulk_df.iterrows():
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025
        if (date in skip_dates):
            continue
        if date not in list(roi_dictionary.keys()):
            roi_dictionary[date] = {}
            for ticker in tickers:
                roi_dictionary[date][ticker] = {}

        entry_time = row['Entry Time']               # hour:minute:second
        trade_id = row['Id']
        entry_price = row['Entry Price']
        ticker = row['Ticker']
        direction = row['Trade Type']
        
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
                break
        
        if not entry_found:
            msg = f"Entry time {entry_time} not found in market data for trade {trade_id}"
            print(msg)
            raise ValueError(msg)
            
        # Track stop loss state
        stop_loss_triggered = False
        stop_loss_updated = False    # Track if stop loss changed from -0.4% to 0%
        
        # Iterate through market data starting from entry point
        for i in range(start_index, len(market_df)):
            current_price = market_df.iloc[i]['Price']
            current_time_str = market_df.iloc[i]['Time']
            
            # Convert current time to seconds since midnight
            current_time_obj = datetime.strptime(current_time_str, '%H:%M:%S').time()
            curr_time_seconds = current_time_obj.hour * 3600 + current_time_obj.minute * 60 + current_time_obj.second
            
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
                if roi <= stop_loss:
                    stop_loss_triggered = True
                    roi_dictionary[date][ticker][curr_time_seconds] = [trade_id, roi]
                    break

                elif roi >= 0.6 and stop_loss_updated == False:
                    stop_loss_updated = True
                    roi_dictionary[date][ticker][curr_time_seconds] = [trade_id, roi]

                elif stop_loss_updated == True and roi <= 0:
                    stop_loss_triggered = True
                    roi_dictionary[date][ticker][curr_time_seconds] = [trade_id, roi]
                    break

                else:
                    roi_dictionary[date][ticker][curr_time_seconds] = [trade_id, roi]
            else:
                # Stop loss already triggered, no more data needed
                break
        
    
    print(f"roi dictionary done. Total trades processed: {len(roi_dictionary)}")

    return roi_dictionary

'''
-roi_dictionary: {date: {ticker: {time_in_seconds: [trade_id, roi]}, ticker: {...}, ...}, date: {...}}
-for each date, go second by second scanning each ticker for each second. if there's an roi value for that ticker
 then subtract the prev value from that ticker and add the new value unless it's a new trade
'''
def Find_Second_By_Second_Days_Roi_Sum(roi_dictionary, tickers):
    """
    Find the highest ROI sum achieved during each trading day.
    
    For each date, tracks the cumulative ROI second by second:
    - When a trade is active, updates the running sum with current ROI
    - When a trade ends (new trade_id), locks in the final ROI permanently
    - Tracks both current sum and highest sum reached during the day
    """
    
    # Market hours: 6:30 AM (23400 seconds) to 1:00 PM (46800 seconds)
    MARKET_OPEN = 23400  # 6:30 AM in seconds since midnight
    MARKET_CLOSE = 46800  # 1:00 PM in seconds since midnight
    
    results = {}
    
    for date in roi_dictionary.keys():
        print(f"\nProcessing date: {date}")
        
        current_sum = 0.0
        highest_sum = 0.0
        
        # Track previous ROI and trade_id for each ticker to handle transitions
        ticker_previous_roi = {ticker: 0.0 for ticker in tickers}
        ticker_previous_trade_id = {ticker: None for ticker in tickers}
        
        # Go second by second through the trading day
        for second in range(MARKET_OPEN, MARKET_CLOSE + 1):
            
            # Check each ticker for this second
            for ticker in tickers:
                if ticker in roi_dictionary[date]:
                    ticker_data = roi_dictionary[date][ticker].get(second, None)
                    
                    if ticker_data is not None:
                        current_trade_id, current_roi = ticker_data
                        
                        # Check if this is a new trade (trade_id changed)
                        if ticker_previous_trade_id[ticker] is not None and ticker_previous_trade_id[ticker] != current_trade_id:
                            # New trade detected - previous trade ended
                            # The previous ROI is already locked in, don't subtract it
                            # Just start tracking the new trade
                            current_sum += current_roi
                            #print(f"  {second}s: {ticker} new trade {current_trade_id}, ROI: {current_roi}%, sum: {current_sum}%")
                        
                        elif ticker_previous_trade_id[ticker] == current_trade_id:
                            # Same trade continuing - update ROI
                            # Subtract previous ROI and add new ROI
                            current_sum = current_sum - ticker_previous_roi[ticker] + current_roi
                            #print(f"  {second}s: {ticker} trade {current_trade_id} update, ROI: {current_roi}%, sum: {current_sum}%")
                        
                        else:
                            # First time seeing this ticker (ticker_previous_trade_id[ticker] is None)
                            current_sum += current_roi
                            #print(f"  {second}s: {ticker} first trade {current_trade_id}, ROI: {current_roi}%, sum: {current_sum}%")
                        
                        # Update tracking variables
                        ticker_previous_roi[ticker] = current_roi
                        ticker_previous_trade_id[ticker] = current_trade_id
                        
                        # Update highest sum if current sum is higher
                        if current_sum > highest_sum:
                            highest_sum = current_sum
                            print(f"    New highest sum: {highest_sum}%")
        
        results[date] = {
            'final_sum': round(current_sum, 3),
            'highest_sum': round(highest_sum, 3)
        }
        
        print(f"Date {date} - Final sum: {current_sum}%, Highest sum: {highest_sum}%")
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS - HIGHEST ROI SUMS BY DATE")
    print("="*50)
    
    for date, data in results.items():
        print(f"{date}: Highest Sum = {data['highest_sum']}% (Final Sum = {data['final_sum']}%)")
    
    return results







def Main():
    bulk_df = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[["Date", "Ticker", "Entry Time", "Time in Trade", "Entry Price", "Exit Price", "Trade Type", "Entry Volatility Percent", "Original Holding Reached", "Original Best Exit Percent", "Original Percent Change"]]
    
    market_data_dict_by_ticker = Load_Market_Data_Dictionary(bulk_df) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    tickers = bulk_df['Ticker'].unique()

    roi_dictionary = Gather_Data(bulk_df, market_data_dict_by_ticker, tickers) # {date: {ticker: {entry time in seconds: [trade_id, roi]}, ticker: {...}, ...}, date: {...}}

    Find_Second_By_Second_Days_Roi_Sum(roi_dictionary, tickers)


Main()