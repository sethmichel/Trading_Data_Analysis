import pandas as pd
from datetime import datetime
import os

'''
Use to assess if I should take profits when it reaches a certain percent
finds the second by second roi I had during the day by the sum of all tickers

status: 
-it has no way of knowing when I actually exited unless I change the algo to use exit time. so final sum is just the sum
 if I held everything to -0.4, 0, or eod. 

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
        trade_id = row['Trade Id']
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
                            #print(f"    New highest sum: {round(highest_sum, 2)}%")
        
        results[date] = {
            'final_sum': round(current_sum, 2),
            'highest_sum': round(highest_sum, 2)
        }
        
        #print(f"Date {date} - Final sum: {current_sum}%, Highest sum: {highest_sum}%")
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS - HIGHEST ROI SUMS BY DATE")
    print("="*50)
    
    for date, data in results.items():
        print(f"{date}: Highest Sum / 6 = {round(data['highest_sum'] / 6, 2)}% (Final Sum / 6 = {round(data['final_sum'] / 6, 2)}%)")

    # Calculate and print the sum of final_sum and highest_sum values for each date
    total_final_sum = sum(data['final_sum'] for data in results.values())
    total_highest_sum = sum(data['highest_sum'] for data in results.values())
    
    print(f"\nTotal Final Sum / 6 = {round(total_final_sum / 6, 2)}%")
    print(f"Total Highest Sum / 6 = {round(total_highest_sum / 6, 2)}%")

    return results


def Analyze_Optimal_Exit_Thresholds(roi_dictionary, tickers, thresholds=None):
    """
    Analyze what daily profit threshold would result in the highest total realized profits.
    
    For each threshold, simulate exiting all positions when daily ROI first reaches that level.
    If the threshold is never reached, use end-of-day ROI.
    
    Args:
        roi_dictionary: {date: {ticker: {time_in_seconds: [trade_id, roi]}}}
        tickers: List of ticker symbols
        thresholds: List of daily profit thresholds to test (default: 0.5% to 3.0% in 0.25% steps)
    
    Returns:
        Dictionary with results for each threshold
    """
    if thresholds is None:
        # Test thresholds from 1% to 3.0% in 0.25% increments
        thresholds = [round(x * 0.25 + 1, 2) for x in range(9)]  # [1.0, 1.25, 1.5, ..., 3.0]
    
    # Market hours: 6:30 AM (23400 seconds) to 1:00 PM (46800 seconds)
    MARKET_OPEN = 23400
    MARKET_CLOSE = 46800
    
    results = {}
    
    print(f"\nTesting {len(thresholds)} profit thresholds: {thresholds}")
    print("\nNOTE: All percentages shown are 'average per position' (raw sum ÷ 6)")
    print("This matches your original analysis where you divide by 6 to normalize across positions")
    print("="*75)
    
    for threshold in thresholds:
        print(f"\nAnalyzing threshold: {threshold}%")
        
        threshold_results = {
            'threshold': threshold,
            'daily_profits': [],
            'days_hit_threshold': 0,
            'days_missed_threshold': 0,
            'total_profit': 0.0,
            'hit_rate': 0.0,
            'avg_profit_hit_days': 0.0,
            'avg_profit_miss_days': 0.0
        }
        
        for date in roi_dictionary.keys():
            daily_profit = None  # Will store the profit for this day at this threshold
            threshold_hit = False
            
            current_sum = 0.0
            
            # Track previous ROI and trade_id for each ticker
            ticker_previous_roi = {ticker: 0.0 for ticker in tickers}
            ticker_previous_trade_id = {ticker: None for ticker in tickers}
            
            # Go second by second through the trading day
            for second in range(MARKET_OPEN, MARKET_CLOSE + 1):
                
                # If we already hit threshold, stop processing this day (simulate stopping trading)
                if threshold_hit:
                    break
                
                # Check each ticker for this second
                for ticker in tickers:
                    if ticker in roi_dictionary[date]:
                        ticker_data = roi_dictionary[date][ticker].get(second, None)
                        
                        if ticker_data is not None:
                            current_trade_id, current_roi = ticker_data
                            
                            # Handle trade transitions (same logic as Find_Second_By_Second_Days_Roi_Sum)
                            if ticker_previous_trade_id[ticker] is not None and ticker_previous_trade_id[ticker] != current_trade_id:
                                # New trade detected - previous trade ended
                                current_sum += current_roi
                            elif ticker_previous_trade_id[ticker] == current_trade_id:
                                # Same trade continuing - update ROI
                                current_sum = current_sum - ticker_previous_roi[ticker] + current_roi
                            else:
                                # First time seeing this ticker
                                current_sum += current_roi
                            
                            # Update tracking variables
                            ticker_previous_roi[ticker] = current_roi
                            ticker_previous_trade_id[ticker] = current_trade_id
                
                # Check if we've hit the threshold 
                # Note: current_sum is the raw sum of all position ROIs, divide by 6 to get average per position
                daily_roi_percentage = current_sum / 6
                if daily_roi_percentage >= threshold:
                    # Threshold reached! Exit all positions and stop trading for the day
                    daily_profit = daily_roi_percentage
                    threshold_hit = True
                    break  # Stop processing this day immediately
            
            # If threshold was never hit, use end-of-day profit
            if daily_profit is None:
                daily_profit = current_sum / 6
            
            # Record results for this day
            threshold_results['daily_profits'].append(daily_profit)
            threshold_results['total_profit'] += daily_profit
            
            if threshold_hit:
                threshold_results['days_hit_threshold'] += 1
            else:
                threshold_results['days_missed_threshold'] += 1
        
        # Calculate summary statistics
        total_days = len(threshold_results['daily_profits'])
        threshold_results['hit_rate'] = (threshold_results['days_hit_threshold'] / total_days) * 100
        
        # Calculate average profits for hit vs miss days
        hit_profits = []
        miss_profits = []
        
        for i, profit in enumerate(threshold_results['daily_profits']):
            if i < threshold_results['days_hit_threshold']:  # This is approximate - we'd need to track which specific days hit
                hit_profits.append(profit)
            else:
                miss_profits.append(profit)
        
        # More accurate way: re-identify hit vs miss days
        hit_profits = [p for p in threshold_results['daily_profits'] if p >= threshold]
        miss_profits = [p for p in threshold_results['daily_profits'] if p < threshold]
        
        threshold_results['avg_profit_hit_days'] = sum(hit_profits) / len(hit_profits) if hit_profits else 0
        threshold_results['avg_profit_miss_days'] = sum(miss_profits) / len(miss_profits) if miss_profits else 0
        
        results[threshold] = threshold_results
        
        print(f"  Total Profit (sum across all days): {round(threshold_results['total_profit'], 2)}%")
        print(f"  Hit Rate (% of days that reached this threshold): {round(threshold_results['hit_rate'], 2)}% ({threshold_results['days_hit_threshold']}/{total_days} days)")
        print(f"  Avg Profit on Hit Days: {round(threshold_results['avg_profit_hit_days'], 2)}%")
        print(f"  Avg Profit on Miss Days: {round(threshold_results['avg_profit_miss_days'], 2)}%")
        print(f"  Overall Avg Daily Profit: {round(threshold_results['total_profit'] / total_days, 2)}%")
    
    # Find optimal threshold
    best_threshold = max(results.keys(), key=lambda x: results[x]['total_profit'])
    best_result = results[best_threshold]
    
    # Get baseline (no threshold strategy) - use the lowest threshold's miss days as proxy for EOD
    baseline_profits = []
    for threshold in thresholds:
        baseline_profits.extend([p for p in results[threshold]['daily_profits'] if p < threshold])
    
    # Better baseline: calculate what happens with no threshold (end-of-day profits)
    # This would be the final sum from the original analysis
    total_days = len(results[best_threshold]['daily_profits'])
    
    print("\n" + "="*80)
    print("OPTIMAL EXIT THRESHOLD ANALYSIS - COMPREHENSIVE SUMMARY")
    print("="*80)
    print("📝 EXPLANATION:")
    print("   • All percentages are 'average per position' (total ROI / 6 positions)")
    print("   • Hit Rate = % of trading days that reached the threshold")
    print("   • Daily ROI = realized profits from closed trades + unrealized profits from open positions")
    print("   • When threshold is hit, you exit ALL open positions and stop trading that day")
    print("   • When threshold is missed, you get end-of-day profit")
    
    print(f"\n🎯 BEST THRESHOLD: {best_threshold}% (average per position)")
    print(f"   Total Profit Across All Days: {round(best_result['total_profit'], 2)}%")
    print(f"   Average Daily Profit: {round(best_result['total_profit'] / total_days, 2)}%")
    print(f"   Hit Rate: {round(best_result['hit_rate'], 2)}% ({best_result['days_hit_threshold']}/{total_days} days reached threshold)")
    print(f"   Average Profit on Hit Days: {round(best_result['avg_profit_hit_days'], 2)}%")
    print(f"   Average Profit on Miss Days: {round(best_result['avg_profit_miss_days'], 2)}%")
    
    print(f"\n📊 THRESHOLD COMPARISON TABLE:")
    print(f"{'Threshold':<10} {'Total Profit':<12} {'Hit Rate':<10} {'Avg Daily':<10} {'Hit Days':<8} {'Miss Days':<9}")
    print("-" * 70)
    
    for threshold in sorted(thresholds):
        result = results[threshold]
        total_profit = round(result['total_profit'], 2)
        hit_rate = round(result['hit_rate'], 2)
        avg_daily = round(result['total_profit'] / total_days, 2)
        hit_days = result['days_hit_threshold']
        miss_days = result['days_missed_threshold']
        
        marker = " 🏆" if threshold == best_threshold else ""
        print(f"{threshold}%{marker:<8} {total_profit}%{'':<8} {hit_rate}%{'':<6} {avg_daily}%{'':<6} {hit_days:<8} {miss_days:<9}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Best threshold is reached on {round(best_result['hit_rate'], 2)}% of trading days")
    print(f"   • On days that hit threshold: avg profit {round(best_result['avg_profit_hit_days'], 2)}% per position")
    print(f"   • On days that miss threshold: avg profit {round(best_result['avg_profit_miss_days'], 2)}% per position")
    
    # Find the threshold with highest hit rate for comparison
    highest_hit_rate_threshold = max(results.keys(), key=lambda x: results[x]['hit_rate'])
    if highest_hit_rate_threshold != best_threshold:
        hr_result = results[highest_hit_rate_threshold]
        print(f"   • Highest hit rate is {round(hr_result['hit_rate'], 2)}% at {highest_hit_rate_threshold}% threshold")
        print(f"     but total profit would be {round(hr_result['total_profit'], 2)}% vs {round(best_result['total_profit'], 2)}%")
    
    print(f"\n🎲 STRATEGY RECOMMENDATION:")
    print(f"   Set daily profit target at {best_threshold}% (per position average)")
    print(f"   This means: when your day's total ROI (realized + unrealized) reaches {best_threshold}%, exit all open positions & stop trading")
    print(f"   Expected results: {round(best_result['total_profit'] / total_days, 2)}% average daily profit")
    print(f"   You'll hit this target on {round(best_result['hit_rate'], 2)}% of trading days")
    
    return results



def Main():
    columns_to_keep = ["Date","Trade Id", "Ticker", "Entry Time", "Time in Trade", "Entry Price", "Exit Price", "Trade Type", 
                       "Entry Volatility Percent", "Original Holding Reached", "Original Best Exit Percent", "Original Percent Change"]
    bulk_df = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[columns_to_keep]
    
    market_data_dict_by_ticker = Load_Market_Data_Dictionary(bulk_df) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    tickers = bulk_df['Ticker'].unique()

    roi_dictionary = Gather_Data(bulk_df, market_data_dict_by_ticker, tickers) # {date: {ticker: {entry time in seconds: [trade_id, roi]}, ticker: {...}, ...}, date: {...}}

    # Original analysis - find highest daily ROI achieved (not that useful. it was 1.16% last I checked (split 6))
    #daily_results = Find_Second_By_Second_Days_Roi_Sum(roi_dictionary, tickers)
    
    # New analysis - find optimal exit threshold
    threshold_results = Analyze_Optimal_Exit_Thresholds(roi_dictionary, tickers)


Main()