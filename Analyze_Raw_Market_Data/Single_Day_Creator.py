import pandas as pd
import numpy as np
import re
from datetime import datetime


DUPLICATE_TIMESTAMP_THRESHOLD = 18   # for duplicate check threshold
CROSS_DURATION_THRESHOLD = 4         # for cross duration threshold (in seconds) (cross must be x min to be valid event)

file_dir = "Google_Sheets_Csvs"
#file_name = f"{file_dir}/Data_Test_Day-05-08-2025-TIME_FIX.csv"
file_name = f"{file_dir}/Data_06-09-2025.csv"
#file_name = f"2_Raw_Market_Data/Data_06-09-2025.csv"
#file_names = [f"{file_dir}/Data_Test_Day-05-06-2025-TIME_FIX.csv", f"{file_dir}/Data_Test_Day-05-07-2025-TIME_FIX.csv", 
#              f"{file_dir}/Data_Test_Day-05-08-2025-TIME_FIX.csv", f"{file_dir}/Data_06-09-2025.csv"]
trade_csv_name = f"Analyze_Raw_Market_Data/Single_Days_Cross_Data.csv"


def helper_parse_time(time_str):
    """
    Parse time string in HH:MM:SS format, handling cases where leading zeros might be missing.
    Returns a pandas datetime object.
    """
    try:
        # Split the time string into hours, minutes, seconds
        h, m, s = time_str.split(':')
        # Convert to integers and back to zero-padded strings
        h = f"{int(h):02d}"
        m = f"{int(m):02d}"
        s = f"{int(s):02d}"
        # Create properly formatted time string
        formatted_time = f"{h}:{m}:{s}"
        return pd.to_datetime(formatted_time, format='%H:%M:%S')
    except Exception as e:
        print(f"Warning: Could not parse time '{time_str}': {str(e)}")
        return None


def check_duplicate_timestamps(file_path):
    """
    Check for duplicate timestamps - timestamp appears more than DUPLICATE_TIMESTAMP_THRESHOLD times in a row
    prints the time, count of each occurance
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert Time column to datetime using our custom parser
    df['Time'] = df['Time'].apply(helper_parse_time)
    
    # Remove any rows where time parsing failed
    df = df.dropna(subset=['Time'])
    
    # Group by Time and count occurrences
    time_counts = df.groupby('Time').size()
    
    # Check for timestamps that appear more than threshold times
    duplicate_times = time_counts[time_counts > DUPLICATE_TIMESTAMP_THRESHOLD]
    
    if len(duplicate_times) > 0:
        print(f"\n1) duplicate timestamps: Found {len(duplicate_times)} timestamps that appear more than {DUPLICATE_TIMESTAMP_THRESHOLD} times:")
        for timestamp, count in duplicate_times.items():
            print(f"Time: {timestamp}, Count: {count}")
        return True
    else:
        print(f"\n1) duplicate timestamps: PASS - No timestamps found that appear more than {DUPLICATE_TIMESTAMP_THRESHOLD} times.")
        return False


def check_time_gaps(file_path):
    """
    Check for time gaps in the data:
    1. Gaps of 4 or more seconds forward
    2. Gaps of 3 or more seconds backward
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert Time column to datetime using our custom parser
    df['Time'] = df['Time'].apply(helper_parse_time)
    
    # Remove any rows where time parsing failed
    df = df.dropna(subset=['Time'])
    
    # Calculate time differences between consecutive rows
    df['Time_Diff'] = df['Time'].diff().dt.total_seconds()
    
    # Find forward gaps (4 or more seconds)
    forward_gaps = df[df['Time_Diff'] >= 4]
    
    # Find backward gaps (3 or more seconds)
    backward_gaps = df[df['Time_Diff'] <= -3]
    
    # Print results
    print("\n2) Time Gap Analysis: ")
    
    if len(forward_gaps) > 0:
        print(f"Found {len(forward_gaps)} forward gaps (≥4 seconds):")
        for idx, row in forward_gaps.iterrows():
            print(f"Line {idx+1}: {row['Time'].strftime('%H:%M:%S')} - Gap: {row['Time_Diff']:.1f} seconds")
    else:
        print("PASS - No forward gaps (≥4 seconds) found.")
    
    if len(backward_gaps) > 0:
        print(f"\nFound {len(backward_gaps)} backward gaps (≤-3 seconds):")
        for idx, row in backward_gaps.iterrows():
            print(f"Line {idx+1}: {row['Time'].strftime('%H:%M:%S')} - Gap: {row['Time_Diff']:.1f} seconds")
    else:
        print("PASS - No backward gaps (≤-3 seconds) found.")


def update_price_movement_tracking(state_dict, current_price, trade_direction):
    """
    Helper function to track unique 0.1% price movement levels crossed during a trade.
    
    Args:
        state_dict: Dictionary containing the price tracking variables including price_movement
        current_price: Current row's price
        trade_direction: 'buy' or 'short'
    """
    if state_dict['entry_price'] is None:
        return
    
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


def update_price_tracking(state_dict, current_price, current_time, trade_direction):
    """
    Helper function to update best/worst exit price tracking variables.
    
    Args:
        state_dict: Dictionary containing the price tracking variables
        current_price: Current row's price
        current_time: Current row's timestamp
        trade_direction: 'buy' or 'short'
    """
    if state_dict['entry_price'] is None:
        return
    
    # Calculate ROI for current price
    if trade_direction == 'buy':
        current_roi = (current_price - state_dict['entry_price']) / state_dict['entry_price'] * 100
    else:  # short
        current_roi = (state_dict['entry_price'] - current_price) / state_dict['entry_price'] * 100
    
    # Round to 2 decimal places
    current_roi = int(current_roi * 100) / 100  # Truncate to 2 decimal places # WARNING: DO NOT ROUND, if 1.8957 is rounded to 1.9 then 1.9 won't appear in price movement and you'll think it's a bug
    
    # Initialize best/worst if not set
    if state_dict['best_exit_percent'] is None:
        state_dict['best_exit_percent'] = current_roi
        state_dict['best_exit_timestamp'] = current_time
        state_dict['best_exit_price'] = current_price
        state_dict['worst_exit_percent'] = current_roi
        state_dict['worst_exit_timestamp'] = current_time
        state_dict['worst_exit_price'] = current_price
        return
    
    # Update best exit (higher ROI is better)
    if current_roi > state_dict['best_exit_percent']:
        state_dict['best_exit_percent'] = current_roi
        state_dict['best_exit_timestamp'] = current_time
        state_dict['best_exit_price'] = current_price
    
    # Update worst exit (lower ROI is worse)
    if current_roi < state_dict['worst_exit_percent']:
        state_dict['worst_exit_percent'] = current_roi
        state_dict['worst_exit_timestamp'] = current_time
        state_dict['worst_exit_price'] = current_price


def extract_date_from_filename(filename):
    """
    Extract date from filename in format MM-DD-YYYY.
    Handles both formats:
    - Data_Test_Day-MM-DD-YYYY.csv
    - Data_MM-DD-YYYY.csv
    """
    # Match date pattern MM-DD-YYYY
    match = re.search(r'(\d{2}-\d{2}-\d{4})', filename)
    if match:
        return match.group(1)
    return None


def Check_If_Date_In_Output_Csv(date):
    try:
        with open(trade_csv_name, 'r') as f:
            # Try to read header row
            try:
                next(f)
            except StopIteration:
                # File is empty, no need to check for date
                return
                
            # Try to read first data row
            first_line = next(f, None)
            if first_line:  # Only proceed if we have at least one data row
                first_ticker = first_line.split(',')[0]
                # Continue reading until ticker changes
                for line in f:
                    current_ticker = line.split(',')[0]
                    if current_ticker != first_ticker:
                        break
                    # Check date in current line
                    current_date = line.split(',')[1]
                    if current_date == date:
                        raise ValueError(f"Data for date {date} already exists in {trade_csv_name}")
                    
    except FileNotFoundError:
        # File doesn't exist yet, which is fine
        pass


def track_crosses(file_path):
    # Extract date from filename and check if it's in the output csv already
    date = extract_date_from_filename(file_path)
    if not date:
        print("Warning: Could not extract date from filename")
        raise ValueError("Warning: Could not extract date from filename")
    
    Check_If_Date_In_Output_Csv(date)
    
    # load input csv
    df = pd.read_csv(file_path)
    df['Time'] = df['Time'].apply(helper_parse_time)
    df = df.dropna(subset=['Time'])   # Remove any rows where time parsing failed
    df = df.sort_values('Time')   # Sort by Time to ensure chronological order
    curr_cross_states = {}   # Dictionary to store state for each ticker
    next_cross_best_worst_holder = {}  # when the next cross is detected and the 1 minute time tracking is going, this 
                                       # tracks the next trades data and moves it to the curr trade when the new cross is confirmed
    results = {}    

    # annoying first row edge case stuff
    first_row_direction = {}    # {[ticker]: "buy"}
    first_row_complete = {}     # {[ticker]: bool}  # checks if the first row was done for each ticker. can't use idx, so do this
    first_cross_confirmed_flag = {}

    for idx, row in df.iterrows():
        ticker = row['Ticker']
        if (ticker != "SOXL" and ticker != "MARA" and ticker != "HOOD" and ticker != "IONQ"): # TESTING, ONLY DEAL WITH SOXL FOR NOW
            continue
        row_time = row['Time']
        val = row['Val']
        avg = row['Avg']
        price = row['Price']
        atr14 = row['Atr14']
        atr28 = row['Atr28']
        rsi = row['Rsi']
        #if (row_time == pd.to_datetime('09:36:13', format='%H:%M:%S')):
        #    pass
        
        # Initialize state for new tickers
        if ticker not in curr_cross_states:
            first_row_direction[ticker] = None
            first_row_complete[ticker] = False
            first_cross_confirmed_flag[ticker] = False

            # again, this just tracks the data of the next cross until it's confirmed, then moves it to curr_cross_states
            next_cross_best_worst_holder[ticker] = {
                'best_exit_timestamp': None,
                'best_exit_percent': None,
                'best_exit_price': None,
                'worst_exit_timestamp': None,
                'worst_exit_percent': None,
                'worst_exit_price': None,
                'entry_price': None,
                'starting_atr14': None,
                'starting_atr28': None,
                'starting_rsi': None,
                'price_movement': None
            }

            curr_cross_states[ticker] = {
                'start_detected_time': None,  # Time of potential cross start
                'end_detected_time': None,    # Time of potential cross end
                'in_cross': False,        # Whether we're in a confirmed cross
                'direction': None,         # 'buy' or 'short'
                'entry_price': None,
                'starting_atr14': None,
                'starting_atr28': None,
                'starting_rsi': None,
                'best_exit_timestamp': None,
                'best_exit_percent': None,
                'best_exit_price': None,
                'worst_exit_timestamp': None,
                'worst_exit_percent': None,
                'worst_exit_price': None,
                'macd_exit_price': None,
                'price_movement': None
            }
            results[ticker] = []

        curr_state = curr_cross_states[ticker]
        next_cross_data = next_cross_best_worst_holder[ticker]

        # Check for cross
        if val > avg:
            if (first_row_direction[ticker] == None):
                first_row_direction[ticker] = "buy"
            direction = 'buy'
        elif val < avg:
            if (first_row_direction[ticker] == None):
                first_row_direction[ticker] = "short"
            direction = 'short'
        else: 
            continue

        # COIN,199.65,-0.1754,-0.2322,0.322,0.27,49.4,06:30:00
        # If we're in a cross
        if (first_cross_confirmed_flag[ticker] == True):
            if (curr_state['in_cross'] == True):
                if (direction == curr_state['direction']):
                    # we're still in the trade, so if we found a cross and it failed, reset vars
                    if (curr_state['end_detected_time'] != None):
                        curr_state['end_detected_time'] = None
                        next_cross_data['best_exit_timestamp'] = None
                        next_cross_data['best_exit_price'] = None
                        next_cross_data['best_exit_percent'] = None
                        next_cross_data['worst_exit_timestamp'] = None
                        next_cross_data['worst_exit_price'] = None
                        next_cross_data['worst_exit_percent'] = None
                        next_cross_data['entry_price'] = None
                        next_cross_data['starting_atr14'] = None
                        next_cross_data['starting_atr28'] = None
                        next_cross_data['starting_rsi'] = None
                        next_cross_data['price_movement'] = None
                        curr_state['macd_exit_price'] = None
                    
                    # Update price tracking in curr_state when not in exit confirmation
                    update_price_tracking(curr_state, price, row_time, curr_state['direction'])
                    update_price_movement_tracking(curr_state, price, curr_state['direction'])

                # we found a cross, have not recorded it yet
                elif (curr_state['end_detected_time'] == None):
                    curr_state['end_detected_time'] = row_time
                    curr_state['macd_exit_price'] = price
                    next_cross_data['entry_price'] = price
                    next_cross_data['starting_atr14'] = atr14
                    next_cross_data['starting_atr28'] = atr28
                    next_cross_data['starting_rsi'] = rsi
                    # last check for this cross
                    update_price_tracking(curr_state, price, row_time, curr_state['direction'])
                    update_price_movement_tracking(curr_state, price, curr_state['direction'])

                # check 1 minute trial period
                else:
                    # Update price tracking in next_cross_data during exit confirmation period
                    update_price_tracking(next_cross_data, price, row_time, direction)
                    update_price_movement_tracking(next_cross_data, price, direction)
                    
                    if ((row_time - curr_state['end_detected_time']).total_seconds() >= CROSS_DURATION_THRESHOLD):
                        # trade ends, record it with best/worst data
                        results[ticker].append([
                            curr_state['start_detected_time'], 
                            curr_state['end_detected_time'], 
                            curr_state['direction'],
                            curr_state['starting_atr14'],
                            curr_state['starting_atr28'],
                            curr_state['starting_rsi'],
                            curr_state['entry_price'],
                            curr_state['best_exit_timestamp'],
                            curr_state['best_exit_price'],
                            curr_state['best_exit_percent'],
                            curr_state['worst_exit_timestamp'],
                            curr_state['worst_exit_price'],
                            curr_state['worst_exit_percent'],
                            curr_state['macd_exit_price'],
                            # Calculate macd_exit_percent based on direction
                            ((curr_state['macd_exit_price'] - curr_state['entry_price']) / curr_state['entry_price'] * 100) if curr_state['direction'] == 'buy' else ((curr_state['entry_price'] - curr_state['macd_exit_price']) / curr_state['entry_price'] * 100),
                            curr_state['price_movement']   # this isn't part of the macd equations, it's the next index
                        ])

                        # Reset state (in_cross stays True)
                        curr_state['start_detected_time'] = curr_state['end_detected_time']
                        curr_state['end_detected_time'] = None
                        curr_state['direction'] = direction
                        
                        curr_state['best_exit_timestamp'] = next_cross_data['best_exit_timestamp']
                        curr_state['best_exit_price'] = next_cross_data['best_exit_price']
                        curr_state['best_exit_percent'] = next_cross_data['best_exit_percent']
                        curr_state['worst_exit_timestamp'] = next_cross_data['worst_exit_timestamp']
                        curr_state['worst_exit_price'] = next_cross_data['worst_exit_price']
                        curr_state['worst_exit_percent'] = next_cross_data['worst_exit_percent']
                        curr_state['entry_price'] = next_cross_data['entry_price']
                        curr_state['starting_atr14'] = next_cross_data['starting_atr14']
                        curr_state['starting_atr28'] = next_cross_data['starting_atr28']
                        curr_state['starting_rsi'] = next_cross_data['starting_rsi']
                        curr_state['price_movement'] = next_cross_data['price_movement']
                        curr_state['macd_exit_price'] = None

                        next_cross_data['best_exit_timestamp'] = None
                        next_cross_data['best_exit_price'] = None
                        next_cross_data['best_exit_percent'] = None
                        next_cross_data['worst_exit_timestamp'] = None
                        next_cross_data['worst_exit_price'] = None
                        next_cross_data['worst_exit_percent'] = None
                        next_cross_data['entry_price'] = None
                        next_cross_data['starting_atr14'] = None
                        next_cross_data['starting_atr28'] = None
                        next_cross_data['starting_rsi'] = None
                        next_cross_data['price_movement'] = None
            
        # working on the first cross (special case)
        else:
            # if it's the first row - don't check anything
            if (first_row_complete[ticker] == False):
                first_row_complete[ticker] = True
                continue

            # if cross detected and we're not in a cross
            if (direction != first_row_direction[ticker] and curr_state['start_detected_time'] == None):
                curr_state['direction'] = direction
                curr_state['start_detected_time'] = row_time
                curr_state['in_cross'] = True
                curr_state['entry_price'] = price
                curr_state['starting_atr14'] = atr14
                curr_state['starting_atr28'] = atr28
                curr_state['starting_rsi'] = rsi
            
            # in a cross, it failed
            elif (direction == first_row_direction[ticker]):
                curr_state['direction'] = None
                curr_state['start_detected_time'] = None
                curr_state['in_cross'] = False
                curr_state['entry_price'] = None
                curr_state['starting_atr14'] = None
                curr_state['starting_atr28'] = None
                curr_state['starting_rsi'] = None
                curr_state['macd_exit_price'] = None

                curr_state['best_exit_timestamp'] = None
                curr_state['best_exit_price'] = None
                curr_state['best_exit_percent'] = None
                curr_state['worst_exit_timestamp'] = None
                curr_state['worst_exit_price'] = None
                curr_state['worst_exit_percent'] = None
                curr_state['price_movement'] = None
            
            # check 1 minute trial period
            else:
                # Update price tracking in curr_state during first cross confirmation period
                update_price_tracking(curr_state, price, row_time, curr_state['direction'])
                update_price_movement_tracking(curr_state, price, curr_state['direction'])
                
                # if the cross is confirmed
                if ((row_time - curr_state['start_detected_time']).total_seconds() >= 60):
                    first_cross_confirmed_flag[ticker] = True
                    # trade is confirmed, go to normal process, edge case done

    # Write results to CSV
    # Check if file exists and has content
    file_exists = False
    try:
        with open(trade_csv_name, 'r') as f:
            first_line = f.readline().strip()
            file_exists = bool(first_line)
    except FileNotFoundError:
        file_exists = False

    # Open file in append mode
    with open(trade_csv_name, 'a') as f:
        # Write header only if file is new/empty
        if not file_exists:
            f.write("date,ticker,start_time,end_time,direction,best exit timestamp,worst exit timestamp,best exit price,worst exit price,entry_price,macd_exit_price,starting_atr14,starting_atr28,starting_rsi,best exit percent,worst exit percent,macd_exit_percent,price_movement\n")
        
        for ticker, crosses in results.items():
            if crosses:  # Only write if there are crosses for this ticker
                for cross_data in crosses:
                    start, end, direction, starting_atr14, starting_atr28, starting_rsi, entry_price, best_exit_time, best_exit_price, best_exit_percent, worst_exit_time, worst_exit_price, worst_exit_percent, macd_exit_price, macd_exit_percent, price_movement = cross_data
                    
                    # Format timestamps, prices, indicators, and percentages
                    best_exit_str = best_exit_time.strftime('%H:%M:%S') if best_exit_time else ''
                    worst_exit_str = worst_exit_time.strftime('%H:%M:%S') if worst_exit_time else ''
                    starting_atr14_str = str(starting_atr14) if starting_atr14 is not None else ''
                    starting_atr28_str = str(starting_atr28) if starting_atr28 is not None else ''
                    starting_rsi_str = str(starting_rsi) if starting_rsi is not None else ''
                    entry_price_str = str(entry_price) if entry_price is not None else ''
                    best_price_str = str(best_exit_price) if best_exit_price is not None else ''
                    worst_price_str = str(worst_exit_price) if worst_exit_price is not None else ''
                    best_percent_str = str(best_exit_percent) if best_exit_percent is not None else ''
                    worst_percent_str = str(worst_exit_percent) if worst_exit_percent is not None else ''
                    macd_exit_price_str = str(macd_exit_price) if macd_exit_price is not None else ''
                    macd_exit_percent_str = str(round(macd_exit_percent, 2)) if macd_exit_percent is not None else ''
                    
                    # Format price_movement as pipe-separated string
                    if price_movement is not None and len(price_movement) > 0:
                        price_movement_str = '|'.join(str(x) for x in price_movement)
                    else:
                        price_movement_str = ''
                    
                    f.write(f"{date},{ticker},{start.strftime('%H:%M:%S')},{end.strftime('%H:%M:%S') if end else ''},{direction},{best_exit_str},{worst_exit_str},{best_price_str},{worst_price_str},{entry_price_str},{macd_exit_price_str},{starting_atr14_str},{starting_atr28_str},{starting_rsi_str},{best_percent_str},{worst_percent_str},{macd_exit_percent_str},{price_movement_str}\n")



'''
TESTING - JUST TO GET A CSV WITH ONLY 1 TICKER DATA
'''
def extract_ticker_data(input_file, output_file="Analyze_Raw_Market_Data/justMaraData.csv"):
    """
    Extract all rows with Ticker 'MARA' from the input file and write them to the output file.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Filter for MARA ticker
    mara_data = df[df['Ticker'] == 'MARA']
    
    # Write to output file
    mara_data.to_csv(output_file, index=False)
    print(f"\nMARA data extraction complete. {len(mara_data)} rows written to {output_file}")


'''
TESTING - MAKE THE TIMESTAMPS CORRECT
'''
def Change_Timestamps(start_row, end_row, difference_in_start_time):
    """
    Adjust timestamps in the CSV file by subtracting difference_in_start_time from each timestamp.
    Creates a new file with "-TIME_FIX" appended to the original filename.
    Only processes rows between start_row and end_row (inclusive).
    """
    '''
    start time: 6:30:00, target time: 20:25:46, diff: 13:55:46. first line at new timestamps 96616
    '''
    negative_timestamp = False

    # Parse the time difference
    try:
        if (difference_in_start_time[0] != '-'):
            diff_time = pd.to_datetime(difference_in_start_time, format='%H:%M:%S')
        else:
            diff_time = pd.to_datetime(difference_in_start_time, format='-%H:%M:%S')
            negative_timestamp = True
    except Exception as e:
        print(f"Error: Invalid difference_in_start_time format: {difference_in_start_time}")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(file_name)
    except Exception as e:
        print(f"Error reading file {file_name}: {str(e)}")
        return

    # Create output filename
    output_file = f'{file_name}-TIME_FIX.csv'
    
    # Process only the specified range of lines
    for idx, row in df.iloc[start_row:end_row].iterrows():
        try:
            # Parse the current time
            current_time = helper_parse_time(row['Time'])
            if current_time is None:
                print(f"Error: Invalid time format at line {idx + 2}: {row['Time']}")
                continue

            if (negative_timestamp == False):
                # Subtract the time difference
                new_time = current_time - pd.Timedelta(hours=diff_time.hour, 
                                                    minutes=diff_time.minute, 
                                                    seconds=diff_time.second)
            else:
                # add the time difference
                new_time = current_time + pd.Timedelta(hours=diff_time.hour, 
                                                    minutes=diff_time.minute, 
                                                    seconds=diff_time.second)
            
            # Check if result would be negative
            if new_time < pd.to_datetime('00:00:00', format='%H:%M:%S'):
                raise ValueError(f"Subtraction would result in negative time at line {idx + 2}")
            
            # Update the time in the dataframe
            df.at[idx, 'Time'] = new_time.strftime('%H:%M:%S')
            
        except ValueError as e:
            print(f"Error: {str(e)}")
            return
        except Exception as e:
            print(f"Unexpected error at line {idx + 2}: {str(e)}")
            return

    # Save to new file (will overwrite if exists)
    try:
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully adjusted timestamps for rows {start_row} to {end_row}. File saved as: {output_file}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")


def find_time_difference_to_change_timestamps(start_time,changed_time):
    # Convert times to datetime objects
    start_dt = pd.to_datetime(start_time, format='%H:%M:%S')
    target_dt = pd.to_datetime(changed_time, format='%H:%M:%S')
    
    # Calculate time difference
    time_diff = target_dt - start_dt
    
    # Extract hours, minutes, seconds
    total_seconds = time_diff.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    # Print the result
    print(f"\nTo get from {changed_time} to {start_time}, you need to subtract:")
    print(f"{hours:02d}:{minutes:02d}:{seconds:02d}")


def Combine_Trade_Lines(start_line, middle_line_list, end_line):
    # the indexes of the df are 2 lower than the lines from the csv file
    start_line -= 2
    end_line -= 2
    for i in range(0, len(middle_line_list)):
        middle_line_list[i] -= 2

    source_file = "Analyze_Raw_Market_Data/Single_Days_Cross_Data.csv"
    output_file = "Analyze_Raw_Market_Data/Single_Days_Cross_Data-COMBINED_LINES.csv"     

    # Load the CSV file into dataframe
    df = pd.read_csv(source_file)
    
    # Get the start line data (this will be our base row to modify)
    start_row = df.iloc[start_line].copy()
    
    # Convert percentage columns to float for comparison
    start_best_percent = float(start_row['best exit percent']) if pd.notna(start_row['best exit percent']) and start_row['best exit percent'] != '' else None
    start_worst_percent = float(start_row['worst exit percent']) if pd.notna(start_row['worst exit percent']) and start_row['worst exit percent'] != '' else None
    
    print(f"Start line {start_line} data:")
    print(f"Best exit percent: {start_best_percent}")
    print(f"Worst exit percent: {start_worst_percent}")
    
    # Process middle lines
    for middle_line in middle_line_list:
        middle_row = df.iloc[middle_line]
        middle_best_percent = float(middle_row['best exit percent']) if pd.notna(middle_row['best exit percent']) and middle_row['best exit percent'] != '' else None
        middle_worst_percent = float(middle_row['worst exit percent']) if pd.notna(middle_row['worst exit percent']) and middle_row['worst exit percent'] != '' else None
        
        print(f"\nMiddle line {middle_line} data:")
        print(f"Best exit percent: {middle_best_percent}")
        print(f"Worst exit percent: {middle_worst_percent}")
        
        # Update best exit data if middle line has better (higher) best exit percent
        if middle_best_percent is not None and (start_best_percent is None or middle_best_percent > start_best_percent):
            start_row['best exit percent'] = middle_row['best exit percent']
            start_row['best exit timestamp'] = middle_row['best exit timestamp']
            start_row['best exit price'] = middle_row['best exit price']
            start_best_percent = middle_best_percent
            print(f"  Updated best exit data from middle line {middle_line}")
        
        # Update worst exit data if middle line has worse (lower) worst exit percent
        if middle_worst_percent is not None and (start_worst_percent is None or middle_worst_percent < start_worst_percent):
            start_row['worst exit percent'] = middle_row['worst exit percent']
            start_row['worst exit timestamp'] = middle_row['worst exit timestamp']
            start_row['worst exit price'] = middle_row['worst exit price']
            start_worst_percent = middle_worst_percent
            print(f"  Updated worst exit data from middle line {middle_line}")
    
    # Process end line
    end_row = df.iloc[end_line]
    end_best_percent = float(end_row['best exit percent']) if pd.notna(end_row['best exit percent']) and end_row['best exit percent'] != '' else None
    end_worst_percent = float(end_row['worst exit percent']) if pd.notna(end_row['worst exit percent']) and end_row['worst exit percent'] != '' else None
    
    print(f"\nEnd line {end_line} data:")
    print(f"Best exit percent: {end_best_percent}")
    print(f"Worst exit percent: {end_worst_percent}")
    
    # Always use end_time, macd_exit_price, macd_exit_percent from end line
    start_row['end_time'] = end_row['end_time']
    start_row['macd_exit_price'] = end_row['macd_exit_price']
    start_row['macd_exit_percent'] = end_row['macd_exit_percent']
    print(f"  Updated end_time, macd_exit_price, macd_exit_percent from end line {end_line}")
    
    # Update best exit data if end line has better (higher) best exit percent
    if end_best_percent is not None and (start_best_percent is None or end_best_percent > start_best_percent):
        start_row['best exit percent'] = end_row['best exit percent']
        start_row['best exit timestamp'] = end_row['best exit timestamp']
        start_row['best exit price'] = end_row['best exit price']
        print(f"  Updated best exit data from end line {end_line}")
    
    # Update worst exit data if end line has worse (lower) worst exit percent
    if end_worst_percent is not None and (start_worst_percent is None or end_worst_percent < start_worst_percent):
        start_row['worst exit percent'] = end_row['worst exit percent']
        start_row['worst exit timestamp'] = end_row['worst exit timestamp']
        start_row['worst exit price'] = end_row['worst exit price']
        print(f"  Updated worst exit data from end line {end_line}")
    
    # Recreate the price_movement value by combining all unique values
    # Start with start_line's price_movement
    start_price_movement = start_row['price_movement']
    if pd.notna(start_price_movement) and start_price_movement != '':
        combined_price_movement = start_price_movement.split('|')
    else:
        combined_price_movement = []
    
    print(f"\nStarting price_movement: {combined_price_movement}")
    
    # Process middle lines price_movement
    for middle_line in middle_line_list:
        middle_row = df.iloc[middle_line]
        middle_price_movement = middle_row['price_movement']
        if pd.notna(middle_price_movement) and middle_price_movement != '':
            middle_values = middle_price_movement.split('|')
            for value in middle_values:
                if value not in combined_price_movement:
                    combined_price_movement.append(value)
                    print(f"  Added {value} from middle line {middle_line}")
    
    # Process end line price_movement
    end_price_movement = end_row['price_movement']
    if pd.notna(end_price_movement) and end_price_movement != '':
        end_values = end_price_movement.split('|')
        for value in end_values:
            if value not in combined_price_movement:
                combined_price_movement.append(value)
                print(f"  Added {value} from end line {end_line}")
    
    # Combine back with '|' separator and update start_row
    if combined_price_movement:
        start_row['price_movement'] = '|'.join(combined_price_movement)
    else:
        start_row['price_movement'] = ''
    
    print(f"Final combined price_movement: {start_row['price_movement']}")
    
    # Update the start_line in the dataframe with the combined data
    df.iloc[start_line] = start_row
    
    # Delete the end line and middle lines (delete from highest index to lowest to avoid index shifting)
    lines_to_delete = sorted([end_line] + middle_line_list, reverse=True)
    for line_idx in lines_to_delete:
        df = df.drop(df.index[line_idx]).reset_index(drop=True)
        print(f"Deleted line {line_idx}")
    
    # Save the modified dataframe to output file
    df.to_csv(output_file, index=False)
    print(f"\nCombined trade line saved to {output_file}")
    print(f"Final combined row data:")
    print(start_row.to_string())

# 1: solo
# 2: lines 3-5
# 3: line 6 solo
# 4: lines 7-9
def auto_combine_trade_lines(df):
    df = df.copy()

    # Convert time columns to datetime (without date to simplify comparisons)
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time

    # Initialize columns
    df['tag'] = ""
    prev_direction = None
    current_group = []

    for i, row in df.iterrows():
        direction = row['direction']

        if prev_direction is None or direction != prev_direction:
            # Close previous group
            if current_group:
                if len(current_group) == 1:
                    df.at[current_group[0], 'tag'] = 'solo'
                else:
                    df.at[current_group[0], 'tag'] = 'start'
                    for mid in current_group[1:-1]:
                        df.at[mid, 'tag'] = 'middle'
                    df.at[current_group[-1], 'tag'] = 'end'

            # Start new group
            current_group = [i]
            prev_direction = direction
        else:
            current_group.append(i)

    # Final group
    if current_group:
        if len(current_group) == 1:
            df.at[current_group[0], 'tag'] = 'solo'
        else:
            df.at[current_group[0], 'tag'] = 'start'
            for mid in current_group[1:-1]:
                df.at[mid, 'tag'] = 'middle'
            df.at[current_group[-1], 'tag'] = 'end'

    return df




# Run the checks
#check_duplicate_timestamps(file_name)
#check_time_gaps(file_name)
#track_crosses(file_name)


result = auto_combine_trade_lines(pd.read_csv("Analyze_Raw_Market_Data/Single_Days_Cross_Data.csv"))
print(result.head(10))
pass
#for f_name in file_names:
#   track_crosses(f_name)

#find_time_difference_to_change_timestamps(start_time='07:53:02', changed_time='15:36:40')
#Change_Timestamps(start_row=44505, end_row=214741, difference_in_start_time="07:43:38")
# 96617 is first row with new timestamp
# 1st start_row=0, end_row=96615, difference_in_start_time="13:55:46"
# 2nd start_row=96615, end_row=111151, difference_in_start_time="-01:26:54" change name 
# fix: start_row=96615, end_row=111151, difference_in_start_time="00:53:48"
# start_row: must be 0 or be 2 rows behind where you want to start
# end_row: it'll change 1 row farther than this
# diff in start time: can be negative to add time


#extract_ticker_data(file_name)


#Combine_Trade_Lines(start_line = 3, middle_line_list = [4], end_line = 5)
                    