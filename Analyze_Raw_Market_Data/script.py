import pandas as pd
import numpy as np


DUPLICATE_TIMESTAMP_THRESHOLD = 18   # for duplicate check threshold
CROSS_DURATION_THRESHOLD = 1         # for cross duration threshold (in minutes) (cross must be x min to be valid event)

file_dir = "2MarketData"
file_name = f"{file_dir}/Data_Test_Day-05-07-2025-TIME_FIX.csv"
trade_csv_name = f"Analyze_Raw_Market_Data/Crosses.csv"


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


def track_crosses(file_path):
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
        row_time = row['Time']
        val = row['Val']
        avg = row['Avg']
        price = row['Price']

        # Initialize state for new tickers
        if ticker not in curr_cross_states:
            first_row_direction[ticker] = None
            first_row_complete[ticker] = False
            first_cross_confirmed_flag[ticker] = False

            # again, this just tracks the data of the next cross until it's confirmed, then moves it to curr_cross_states
            next_cross_best_worst_holder[ticker] = {
                'best_exit_timestamp': None,
                'best_exit_percent': None,
                'worst_exit_timestamp': None,
                'worst_exit_percent': None,
                'entry_price': None
            }

            curr_cross_states[ticker] = {
                'start_detected_time': None,  # Time of potential cross start
                'end_detected_time': None,    # Time of potential cross end
                'in_cross': False,        # Whether we're in a confirmed cross
                'direction': None,         # 'buy' or 'short'
                'entry_price': None,
                'best_exit_timestamp': None,
                'best_exit_percent': None,
                'worst_exit_timestamp': None,
                'worst_exit_percent': None
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
                    # we're still in the trade, so if we found a cross it failed, reset vars
                    if (curr_state['end_detected_time'] != None):
                        curr_state['end_detected_time'] = None
                        next_cross_data['best_exit_timestamp'] = None
                        next_cross_data['best_exit_percent'] = None
                        next_cross_data['worst_exit_timestamp'] = None
                        next_cross_data['worst_exit_percent'] = None
                        next_cross_data['entry_price'] = None

                # we found a cross, have not recorded it yet
                elif (curr_state['end_detected_time'] == None):
                    curr_state['end_detected_time'] = row_time
                    # NOTE: right here is an example of when you stop tracking the "price tracking variables" in curr_state, and start tracking them in next_cross_data. because a new cross is detected but not yet confirmed
                    # TODO: set next_cross_data['entry_price']

                # check 1 minute trial period
                else:
                    if ((row_time - curr_state['end_detected_time']).total_seconds() >= 60):
                        # trade ends, record it
                        results[ticker].append([curr_state['start_detected_time'], curr_state['end_detected_time'], curr_state['direction']])

                        # Reset state (in_cross stays True)
                        curr_state['start_detected_time'] = curr_state['end_detected_time']
                        curr_state['end_detected_time'] = None
                        curr_state['direction'] = direction
                        
                        curr_state['best_exit_timestamp'] = next_cross_data['best_exit_timestamp']
                        curr_state['best_exit_percent'] = next_cross_data['best_exit_percent']
                        curr_state['worst_exit_timestamp'] = next_cross_data['worst_exit_timestamp']
                        curr_state['worst_exit_percent'] = next_cross_data['worst_exit_percent']
                        curr_state['entry_price'] = next_cross_data['entry_price']

                        next_cross_data['best_exit_timestamp'] = None
                        next_cross_data['best_exit_percent'] = None
                        next_cross_data['worst_exit_timestamp'] = None
                        next_cross_data['worst_exit_percent'] = None
                        next_cross_data['entry_price'] = None
            
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
            
            # in a cross, it failed
            elif (direction == first_row_direction[ticker]):
                curr_state['direction'] = None
                curr_state['start_detected_time'] = None
                curr_state['in_cross'] = False
                curr_state['entry_price'] = None
            
            # check 1 minute trial period
            else:
                # if the cross is confirmed
                if ((row_time - curr_state['start_detected_time']).total_seconds() >= 60):
                    first_cross_confirmed_flag[ticker] = True
                    # trade is confirmed, go to normal process, edge case done

    # Write results to CSV
    with open(trade_csv_name, 'w') as f:
        for ticker, crosses in results.items():
            if crosses:  # Only write if there are crosses for this ticker
                f.write(f"{ticker}\n")
                f.write("start_time,end_time,direction\n")
                for start, end, direction in crosses:
                    f.write(f"{start.strftime('%H:%M:%S')},{end.strftime('%H:%M:%S') if end else ''},{direction}\n")
                f.write("\n")  # Add blank line between tickers


def Parameter_Testing():
    target = 0.2
    stop_loss = -0.5









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
def Change_Timestamps():
    """
    Adjust timestamps in the CSV file by subtracting difference_in_start_time from each timestamp.
    Creates a new file with "-TIME_FIX" appended to the original filename.
    Only processes rows between start_row and end_row (inclusive).
    """
    # Configuration variables
    start_row = 9653        # has to start at 0 or 2 rows behind where you want to start
    end_row = 156291        # it'll change 1 row farther than this 9653 
    difference_in_start_time = "03:26:19"  # subtract this much time from each row   "03:26:19" '15:12:30'

    # Parse the time difference
    try:
        diff_time = pd.to_datetime(difference_in_start_time, format='%H:%M:%S')
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
    output_file = file_name.replace('.csv', '-TIME_FIX.csv')
    
    # Process only the specified range of lines
    for idx, row in df.iloc[start_row:end_row].iterrows():
        try:
            # Parse the current time
            current_time = helper_parse_time(row['Time'])
            if current_time is None:
                print(f"Error: Invalid time format at line {idx + 2}: {row['Time']}")
                continue

            # Subtract the time difference
            new_time = current_time - pd.Timedelta(hours=diff_time.hour, 
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

    # Save to new file
    try:
        df.to_csv(output_file, index=False)
        print(f"\nSuccessfully adjusted timestamps for rows {start_row} to {end_row}. New file saved as: {output_file}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")


# Run the checks
#check_duplicate_timestamps(file_name)
#check_time_gaps(file_name)
#track_val_avg_crosses(file_name)
track_crosses(file_name)


#extract_ticker_data(file_name)
#Change_Timestamps()