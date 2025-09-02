import pandas as pd
import numpy as np
import os
import inspect
import sys
import Main_Globals
from multiprocessing import freeze_support
from numba import set_num_threads
import time
import Grid_Search_Helper_Create_Combos as Grid_Search_Helper_Create_Combos

# Set Numba to use all available cores
import multiprocessing
num_cores = min(multiprocessing.cpu_count(), 16)  # Ensure we don't exceed system limits
set_num_threads(num_cores)
print(f"Setting Numba to use {num_cores} threads")

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


# after analysis_results file is written to, this'll read the file and store the top result for each metric, 
#    then it'll write this in an easy to read way at the end
def Write_Easy_To_Read_Summary():
    try:
        file_path = "Analysis_Results.txt"
        top_result_lines = []
        
        with open(file_path, "r") as file:
            for line in file:
                # Check if line starts with "1)" 
                if line.strip().startswith("1)"):
                    top_result_lines.append(line.strip())
        
        message = f"\n\nSUMMARY\n"
        
        # edit the results
        for line in top_result_lines:
            line = line[6:]   # cut "1) id: "
            comma_splits = line.split(",")
            param_str = comma_splits[0]   # 6:30:00|0.0|0.0|-5|0.0|0.0|20.0|either|0.3|0.9|-0.5|-0.3
            param_list = param_str.split("|")

            # add parameters
            message += f"Time:{':'.join(param_list[0].split(':')[:2])}, "  # keep hours and minutes, cut off seconds
            if (param_list[1] != "0.0"):
                message += f"vol%: {param_list[1]}, "
            if (param_list[2] != "0.0"):
                message += f"ratio: {param_list[2]}, "
            if (param_list[3] != "-5.0"):
                message += f"avg_of_last_2_trades: {param_list[2]}, "
            if (param_list[4] != "0.0"):
                message += f"adx28: {param_list[4]}, "
            if (param_list[5] != "0.0"):
                message += f"adx14: {param_list[5]}, "
            if (param_list[6] != "0.0"):
                message += f"adx7: {param_list[6]}, "
            if (param_list[7] != 'either'):
                message += f"rsi: {param_list[7]}, "
            if (param_list[8] != 'either'):
                message += f"last_result_4_min_types: {param_list[7]}, "

            # add targets and stop losses
            message += "\n      "
            if len(param_list) == 11:  # 4 final parameters (t1, t2, sl1, sl2)
                message += f"t1: {param_list[7]}, t2: {param_list[8]}, sl1: {param_list[9]}, sl2: {param_list[10]}"
            elif len(param_list) == 13:  # 6 final parameters (t1, t2, t3, sl1, sl2, sl3)
                message += f"t1: {param_list[7]}, t2: {param_list[8]}, t3: {param_list[9]}, sl1: {param_list[10]}, sl2: {param_list[11]}, sl3: {param_list[12]}"

            # add metrics like sum and count
            message += "\n     "
            message += f"{','.join(comma_splits[1:])}\n\n"

            # just a sparator
            message += "-------------------------------------------------------------\n"
        
        # Write the summary to the end of the file
        with open(file_path, "a") as file:
            file.write(message)

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Round_Key_Values(key):
    """
    Rounds all numerical values in a pipe-delimited key string to 1 decimal place.
    Non-numerical values (like True/False/time strings) are kept as-is.
    
    Args:
        key (str): Pipe-delimited string containing parameter values
        
    Returns:
        str: Key string with numerical values rounded to 1 decimal place
    """
    try:
        key_parts = key.split('|')
        rounded_key_parts = []
        for part in key_parts:
            try:
                # Try to convert to float and round to 1 decimal place
                float_val = float(part)
                rounded_key_parts.append(str(round(float_val, 1)))
            except ValueError:
                # If it's not a number (like True/False/time), keep as is
                rounded_key_parts.append(part)
        
        return '|'.join(rounded_key_parts)
    except Exception as e:
        # If anything goes wrong, return the original key
        return key


def Write_Analysis(message):
    file_path = "Analysis_Results.txt"
    # a+ will create the file if it does not exist
    # and allow us to read from it
    with open(file_path, "a+") as file:
        # move the cursor to the beginning of the file
        # to read its content
        file.seek(0)
        # if the file is not empty, then
        if file.read():
            # add two new lines before appending
            # the new content
            file.write("\n\n")
        # Appends the message to the end of the file.
        # if the file was empty, it will append it
        # to the first line
        file.write(message)

# {entry_times_valid[i]}|{volatilities_valid[i]}|{ratios_valid[i]}|{avg_of_last_2_trades_valid[i]}|{adx28s_valid[i]}|{adx14s_valid[i]}|{adx7s_valid[i]}|{rsi_types_valid[i]}|{last_result_4_min_types_valid[i]}|{t1s_valid[i]}|{t2s_valid[i]}|{sl1s_valid[i]}|{sl2s_valid[i]}"

# if user mode = 3 all_sublists if blank, and time_top_sublists is populated. else otherway around
# time_top_sublists: {time list key: {sublist key: sublist}, ...}
def Write_Grid_Seach_Results(all_sublists, time_top_sublists, how_many_final_parameters, user_mode):
    try:
        if (user_mode == 1):
            # overall sum - current list is correct already
            message1 = (f"TEST 1: testing all combos of parameters\n"
                        f"User Mode: {user_mode} - Best Combos Overall:\n"
                        f"time, time key, volatility%, ratio, avg_of_last_2_trades, adx28, 14, 7, rsi_type, last_result_4_min_types")
            
            message2 = "\n"
            for i, key in enumerate(all_sublists.keys()):
                sublist = all_sublists[key]
                rounded_sum = round(sublist['sum'], 1)
                sublist['count'] = sublist['wins'] + sublist['losses'] + sublist['neither']
                sublist['avg'] = round(rounded_sum/sublist['count'], 2)
                
                # Round all numerical values in the key string for display
                rounded_key = Round_Key_Values(key)
                message2 += f"{i+1}) id: {rounded_key}, avg: {sublist['avg']}, sum: {rounded_sum:.1f}, count: {sublist['count']}, wins: {sublist['wins']}, losses: {sublist['losses']}, neither: {sublist['neither']}\n"

        elif (user_mode == 2):
            # Group sublists by volatility - current list is correct but it's all grouped together
            sorted_sublists_by_volatility = {}

            for sublist_key, sublist_data in all_sublists.items():
                # Parse string key format: "entry|volatility|ratio|adx28|adx14|adx7|rsi_type|..."
                key_parts = sublist_key.split('|')
                volatility = float(key_parts[1])
                if (volatility == 0.0):
                    upper_bound = 0.6
                else:
                    upper_bound = 3.0
                    
                if volatility not in sorted_sublists_by_volatility:
                    sorted_sublists_by_volatility[volatility] = []
                sorted_sublists_by_volatility[volatility].append((sublist_key, sublist_data))

            message1 = (f"TEST 2: testing all combos for volatility % ranges\n"
                        f"User Mode: {user_mode} - Best Combos for volatilty ranges. only using ratio:\n"
                        f"volatility%, ratio")
            
            # For each volatility level, get the top 10 results and add them to message2
            message2 = ""
            for vol in sorted_sublists_by_volatility.keys():
                message2 += f"\n--- Top 10 results for Volatility between {vol} and {upper_bound} (exclusive) ---\n "
                
                vol_list = sorted_sublists_by_volatility[vol]
                for i in range(min(10, len(vol_list))):
                    key, sublist = vol_list[i]
                    rounded_sum = round(sublist['sum'], 1)
                    sublist['count'] = sublist['wins'] + sublist['losses'] + sublist['neither']
                    sublist['avg'] = round(rounded_sum/sublist['count'], 2)
                    
                    # Round all numerical values in the key string for display
                    rounded_key = Round_Key_Values(key)
                    message2 += f"{i+1}) id: {rounded_key}, avg: {sublist['avg']}, sum: {rounded_sum:.1f}, count: {sublist['count']}, wins: {sublist['wins']}, losses: {sublist['losses']}, neither: {sublist['neither']}\n"

        elif (user_mode == 3):
            message = ""
            message1 = (f"TEST 3: testing all combos of parameters by time\n"
                        f"User Mode: {user_mode} - Best Combos for time ranges\n")

            for key, sublist_key_value in time_top_sublists.items():
                # Group by time - current list is correct but it's all grouped together. so oragnize it by intervauls
                sorted_sublists_by_time = {}

                for sublist_key, sublist_data in sublist_key_value.items():
                    # Parse string key format: "entry|volatility|ratio|adx28|adx14|adx7|rsi_type|..."
                    key_parts = sublist_key.split('|')
                    entry_time = key_parts[0]
                    if entry_time not in sorted_sublists_by_time:
                        sorted_sublists_by_time[entry_time] = []
                    sorted_sublists_by_time[entry_time].append((sublist_key, sublist_data))
                
                message1 += (f"\n-----Time List Key: {key}-----\n\n"
                            f"entry time, volatility%, ratio, avg_of_last_2_trades, adx28, 14, 7, rsi_type, last_result_4_min_types")
            
                if how_many_final_parameters == 4:
                    message1 += f", t1, t2, sl1, sl2"
                elif how_many_final_parameters == 6:
                    message1 += f", t1, t2, t3, sl1, sl2, sl3"

                # For each time, get the top 10 results and add them to message2
                message2 = ""
                for entry_time in sorted_sublists_by_time.keys():
                    message2 += f"\n--- Top 10 results for time between {entry_time} and whatever the interval is---\n"
                    time_list = sorted_sublists_by_time[entry_time]

                    for i in range(min(10, len(time_list))):
                        key, sublist = time_list[i]
                        rounded_sum = round(sublist['sum'], 1)
                        sublist['count'] = sublist['wins'] + sublist['losses'] + sublist['neither']
                        sublist['avg'] = round(rounded_sum/sublist['count'], 2)
                        
                        # Round all numerical values in the key string for display
                        rounded_key = Round_Key_Values(key)
                        message2 += f"{i+1}) id: {rounded_key}, avg: {sublist['avg']}, sum: {rounded_sum:.1f}, count: {sublist['count']}, wins: {sublist['wins']}, losses: {sublist['losses']}, neither: {sublist['neither']}\n"

                message = message + message1 + "\n" + message2
                message1 = ""

        # user mode 3 needs the same code but since it loops, I have to add it separately
        if (user_mode == 1 or user_mode == 2):
            if how_many_final_parameters == 4:
                message1 += f", t1, t2, sl1, sl2"
            elif how_many_final_parameters == 6:
                message1 += f", t1, t2, t3, sl1, sl2, sl3"

            message = message1 + "\n" + message2

        message = message.replace("'", '').replace("{", '').replace("}", '')

        Write_Analysis(message)

        Write_Easy_To_Read_Summary()

        print("\nCOMPLETE\n")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Convert_To_Numba_Arrays(data_holder, t1_indexes, t2_indexes, t3_indexes, sl1_indexes, sl2_indexes, sl3_indexes,
                            target_1s, target_2s, target_3s, stop_loss_1s, stop_loss_2s, stop_loss_3s):
    """
    CRITICAL PERFORMANCE FUNCTION: Converts Python dictionaries to Numba-compatible arrays
    
    This preprocessing step is essential for speed because:
    1. Numba cannot use Python dictionaries - needs contiguous numpy arrays
    2. Pre-computing all possible lookups avoids repeated searches in the hot loop
    3. Memory overhead is justified by 10-100x speed improvement in processing
    
    The resulting index arrays map [parameter_index][row_index] -> price_movement_index
    This transforms O(n) searches into O(1) array lookups during parallel processing.
    
    Memory usage: ~(num_params * num_rows * 4 bytes) but enables massive parallelization
    """
    try:
        data_rows = np.array([row[0] for row in data_holder], dtype=np.int32)
        data_values = np.array([row[1] for row in data_holder], dtype=np.float32)
        data_last_prices = np.array([row[2] for row in data_holder], dtype=np.float32)
        
        max_row_idx = np.max(data_rows) + 1 if len(data_rows) > 0 else 1

        # --- Convert parameter lists to numpy arrays for Numba compatibility ---
        # Just convert the original lists to numpy arrays for Numba
        target1s_np_array = np.array(target_1s, dtype=np.float32)
        stop_loss1s_np_array = np.array(stop_loss_1s, dtype=np.float32)
        target3s_np_array = np.array(target_3s, dtype=np.float32)
        stop_loss2s_np_array = np.array(stop_loss_2s, dtype=np.float32)
        target2s_np_array = np.array(target_2s, dtype=np.float32)
        stop_loss3s_np_array = np.array(stop_loss_3s, dtype=np.float32)

        # --- Create optimized index arrays for Numba compatibility ---
        # These arrays enable O(1) lookups in Numba but use significant memory
        # Each array maps [parameter_index][row_index] -> price_movement_index
        # The memory usage is justified because Numba cannot use Python dictionaries
        t1_idx_arrays = np.zeros((len(target1s_np_array), max_row_idx), dtype=np.int32)
        sl1_idx_arrays = np.zeros((len(stop_loss1s_np_array), max_row_idx), dtype=np.int32)
        t3_idx_arrays = np.zeros((len(target1s_np_array), len(target3s_np_array), max_row_idx), dtype=np.int32)
        sl2_idx_arrays = np.zeros((len(target1s_np_array), len(stop_loss2s_np_array), max_row_idx), dtype=np.int32)
        t2_idx_arrays = np.zeros((len(target1s_np_array), len(target2s_np_array), max_row_idx), dtype=np.int32)
        sl3_idx_arrays = np.zeros((len(target1s_np_array), len(stop_loss3s_np_array), max_row_idx), dtype=np.int32)

        # Convert t1 and sl1 index dictionaries to arrays (simple 2D structure)
        # these 2 are unique
        for i, t1 in enumerate(target1s_np_array):
            arr = np.full(max_row_idx, 50000, dtype=np.int32)  # 50000 = "not found" sentinel
            if t1 in t1_indexes:
                for idx, val in t1_indexes[t1].items():
                    if idx < max_row_idx: arr[idx] = val
            t1_idx_arrays[i] = arr

        for i, sl1 in enumerate(stop_loss1s_np_array):
            arr = np.full(max_row_idx, 50000, dtype=np.int32)
            if sl1 in sl1_indexes:
                for idx, val in sl1_indexes[sl1].items():
                    if idx < max_row_idx: arr[idx] = val
            sl1_idx_arrays[i] = arr

        # Convert hierarchical index dictionaries to 3D arrays (more complex structure)
        # These depend on t1 because t2/t3/sl2/sl3 are searched AFTER t1 is hit
        for i, t1 in enumerate(target1s_np_array):
            # t3 indices (target 3)
            for j, target3_val in enumerate(target3s_np_array):
                arr = np.full(max_row_idx, 50000, dtype=np.int32)
                if t1 in t3_indexes and target3_val in t3_indexes[t1]:
                    for idx, val in t3_indexes[t1][target3_val].items():
                        if idx < max_row_idx: arr[idx] = val
                t3_idx_arrays[i, j] = arr
                
            # sl2 indices (stop loss 2)
            for j, stop_loss2_val in enumerate(stop_loss2s_np_array):
                arr = np.full(max_row_idx, 50000, dtype=np.int32)
                if t1 in sl2_indexes and stop_loss2_val in sl2_indexes[t1]:
                    for idx, val in sl2_indexes[t1][stop_loss2_val].items():
                        if idx < max_row_idx: arr[idx] = val
                sl2_idx_arrays[i, j] = arr

            # t2 indices (target 2)
            for j, target2_val in enumerate(target2s_np_array):
                arr = np.full(max_row_idx, 50000, dtype=np.int32)
                if t1 in t2_indexes and target2_val in t2_indexes[t1]:
                    for idx, val in t2_indexes[t1][target2_val].items():
                        if idx < max_row_idx: arr[idx] = val
                t2_idx_arrays[i, j] = arr

            # sl3 indices (stop loss 3)
            for j, stop_loss3_val in enumerate(stop_loss3s_np_array):
                arr = np.full(max_row_idx, 50000, dtype=np.int32)
                if t1 in sl3_indexes and stop_loss3_val in sl3_indexes[t1]:
                    for idx, val in sl3_indexes[t1][stop_loss3_val].items():
                        if idx < max_row_idx: arr[idx] = val
                sl3_idx_arrays[i, j] = arr

        
        return (data_rows, data_values, data_last_prices, 
                target1s_np_array, target2s_np_array, target3s_np_array, stop_loss1s_np_array, stop_loss2s_np_array, 
                stop_loss3s_np_array, t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays)
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Create_2D_List_From_Df(df, target_1s, stop_loss_1s, target_3s, stop_loss_2s, target_2s, stop_loss_3s):
    try:
        # Pre-process all price movements to avoid repeated string operations
        df['price_movement_list'] = df['Price Movement'].apply(lambda x: [np.float32(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else [])

        # Keep only the columns specified in columns_to_keep
        columns_to_keep = ['price_movement_list','Entry Volatility Percent','Entry Volatility Ratio','avg_of_last_2_trades','Entry Adx28',
                           'Entry Adx14','Entry Adx7','Rsi Extreme Prev Cross', 'Entry Time', 'last_result_must_be_4_minutes']
        df = df[[col for col in columns_to_keep if col in df.columns]].copy()

        # Separate rows where 'price_movement_list' length <= 3. we'll deal with them at the end
        short_rows_df = df[df['price_movement_list'].apply(lambda x: len(x) <= 3)].copy()
        df = df[df['price_movement_list'].apply(lambda x: len(x) > 3)].copy()
        df.reset_index(drop=True, inplace=True)
        
        # find all indexes lists (they're different lengths)
        t1_indexes = {}     # {target: {df index: price movement index}, ...}
        t2_indexes = {}     # same structure as upper but t2 instead of ut
        t3_indexes = {}     # {normal target: {upper target: {df index: price movement index}, ...}, ...}
        sl1_indexes = {}  # {sl: {df index: price movement index}, ...}
        sl2_indexes = {}  # {normal sl: {upper target: {df index: price movement index}, ...}, ...}
        sl3_indexes = {}  # same structure as upper but sub usl instead of usl
        high_numb = 50000
        
        #t1
        for t1 in target_1s:
            t1_indexes[t1] = {}
            for idx, row in df.iterrows():
                for (i, value) in enumerate(row['price_movement_list']):
                    if (value == t1):
                        t1_indexes[t1][idx] = i
                        break
                else:
                    t1_indexes[t1][idx] = high_numb
        
        # sl1
        for sl in stop_loss_1s:
            sl1_indexes[sl] = {} 
            for idx, row in df.iterrows():
                for (i, value) in enumerate(row['price_movement_list']):
                    if (value == sl):
                        sl1_indexes[sl][idx] = i
                        break
                else:
                    sl1_indexes[sl][idx] = high_numb

        # t2
        for t1 in target_1s:
            t2_indexes[t1] = {}
            for t2 in target_2s:
                t2_indexes[t1][t2] = {}
                for idx, row in df.iterrows():
                    start = t1_indexes[t1][idx] +1
                    if start != high_numb:
                        for i, value in enumerate(row['price_movement_list'][start:]):
                            if value == t2:
                                t2_indexes[t1][t2][idx] = start + i
                                break
                        else:
                            t2_indexes[t1][t2][idx] = high_numb
                    else:
                        t2_indexes[t1][t2][idx] = high_numb

        # sl2
        for t1 in target_1s:
            sl2_indexes[t1] = {}
            for sl2 in stop_loss_2s:
                sl2_indexes[t1][sl2] = {}
                for idx, row in df.iterrows():
                    start = t1_indexes[t1][idx] +1
                    if start != high_numb:
                        for i, value in enumerate(row['price_movement_list'][start:]):
                            if value == sl2:
                                sl2_indexes[t1][sl2][idx] = start + i
                                break
                        else:
                            sl2_indexes[t1][sl2][idx] = high_numb
                    else:
                        sl2_indexes[t1][sl2][idx] = high_numb

        # t3
        for t1 in target_1s:
            t3_indexes[t1] = {}
            for t3 in target_3s:
                t3_indexes[t1][t3] = {}
                for idx, row in df.iterrows():
                    start = t1_indexes[t1][idx] +1
                    if start != high_numb:
                        for i, value in enumerate(row['price_movement_list'][start:]):
                            if value == t3:
                                t3_indexes[t1][t3][idx] = start + i
                                break
                        else:
                            t3_indexes[t1][t3][idx] = high_numb
                    else:
                        t3_indexes[t1][t3][idx] = high_numb

        # sl3
        for t1 in target_1s:
            sl3_indexes[t1] = {}
            for sl3 in stop_loss_3s:
                sl3_indexes[t1][sl3] = {}
                for idx, row in df.iterrows():
                    start = t1_indexes[t1][idx] +1
                    if start != high_numb:
                        for i, value in enumerate(row['price_movement_list'][start:]):
                            if value == sl3:
                                sl3_indexes[t1][sl3][idx] = start + i
                                break
                        else:
                            sl3_indexes[t1][sl3][idx] = high_numb
                    else:
                        sl3_indexes[t1][sl3][idx] = high_numb
            
        
        # Convert 'Entry Time' to seconds from midnight (easlier to deal with later)
        df['Entry Time'] = df['Entry Time'].apply(lambda t: sum(int(x) * 60 ** i for i, x in enumerate(reversed(str(t).split(':')))))
        
        # remove the old price_movement column
        columns_to_keep.pop(0)

        # now convert both df's into a minimized list so I can index really fast
        # [(index of original df, [list of values], last price)]
        data_holder = [] 
        for idx, row in df.iterrows():
            data_holder.append((idx, row[columns_to_keep].tolist(), row['price_movement_list'][-1]))

        short_rows_data_holder = []
        for idx, row in short_rows_df.iterrows():
            short_rows_data_holder.append((idx, row[columns_to_keep].tolist(), row['price_movement_list'][-1]))

        return data_holder, short_rows_data_holder, t1_indexes, t2_indexes, t3_indexes, sl1_indexes, sl2_indexes, sl3_indexes
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


'''
how to estimate valid combos: each list length * the other lengths * 0.786 (removes the invalid combos)
60 million used to take about 11 minutes

current implementation 7/31/25
-finding 2 million combos: 5-7 seconds
-processing: 13-22.7 seconds
    -finding filtered rows/filtered data: 12 seconds
    -Precompute_Parameter_Indices(): 2.5 seconds
    -removed a for loop and now its process takes 2 seconds
    -Parallel_Process_Helper(): 1 second
    -sublist creation for loop: 5.2 seconds
test data time group by time with only: ['6:30:00', '6:40:00', '6:50:00']: 30 seconds
real data: including time, if there's 291 million combos it takes 21 minutes on 250 data points
'''
def Grid_Search_Start(df):
    try:
        global how_many_final_parameters

        '''
        HOW TO USE
        1) pick which format of results you want: 1, 2, or 3 (these also might look at different params)
        2) choose how many final params you want: 4 or 6. this refers to stop loss/targets. when it reaches target 1 you move the target and stop loss. 4 means 2 sets of targets/stop losses, 6 means 3 sets
        3) if using time as a param, choose which tests you want to do
        '''

        # Ask user for mode selection
        print("\nSelect mode:")
        print("1) Overall Combinations")
        print("2) Volatility Groups With Ratio Combinations")
        print("3) Overall Time Group Combinations")
        #user_mode = int(input("Enter the number of the mode you want to use: "))
        user_mode = 3 # TESTING
        if (user_mode not in [1,2,3]):
            raise ValueError("Pick either 1, 2, or 3")

        how_many_final_parameters = 4
        
        time_dict = {'times_1_10m': np.array(['6:30:00', '6:40:00', '6:50:00', 
                                              '7:00:00', '7:10:00', '7:20:00', '7:30:00', '7:40:00', '7:50:00', 
                                              '8:00:00', '8:10:00', '8:20:00', '8:30:00', '8:40:00', '8:50:00', 
                                              '9:00:00'])}
                     #'times_2_15m': np.array(['6:30:00', '6:45:00', '7:00:00', '7:15:00', '7:30:00']), 
                     #'times_3_30m': np.array(['6:30:00', '7:00:00', '7:30:00']), 
                     #'times_4_custom_1': np.array(['6:30:00', '6:45:00', '7:00:00', '7:30:00'])}
        
        '''
        some stats I found. 
        if we test if a trade's previous trade on that ticker lasted at least 4 minutes: 74.3% it did
        if we test the average return of the last 2 trades in deciding if we should take the next trade:
            90% are over -0.5, 43% are over 0.0, 22.7% are over 0.3, 13.5% are over 0.6
        '''
        volatilities = np.array([0.0,0.3,0.4,0.5,0.6], dtype=np.float32)
        ratios = np.array([0.0,0.2,0.5,0.7, 0.8, 0.9, 1.0], dtype=np.float32) # 0 is base value to include all trades
        avg_of_last_2_trades = np.array([-5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4], dtype=np.float32) # -5 is base value to include all trades
        last_result_must_be_4_minutes = [True, False, "either"] # Keep as list for string handling
        adx28s = np.array([0,20, 30], dtype=np.float32)
        adx14s = np.array([0,20, 30], dtype=np.float32)
        adx7s = np.array([0,20, 30], dtype=np.float32)
        extreme_rsis = [True, False, "either"]  # Keep as list for string handling

        target_1s = np.array([0.2,0.3, 0.4, 0.5], dtype=np.float32)
        target_2s = np.array([0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
        target_3s = np.array([0.6, 0.7, 0.8, 0.9,1.0,1.1], dtype=np.float32)
        stop_loss_1s = np.array([-0.5,-0.4,-0.3], dtype=np.float32)
        stop_loss_2s = np.array([-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3], dtype=np.float32)
        stop_loss_3s = np.array([0.2,0.3,0.4,0.5,0.6], dtype=np.float32)
        '''
        # low compute testing lists
        time_dict = {'times_1_10m': np.array(['6:30:00', '6:40:00', '6:50:00']),
                'times_2_15m': np.array(['6:30:00', '6:45:00', '7:00:00']), 
                'times_3_30m': np.array(['6:30:00', '7:00:00', '7:30:00']), 
                'times_4_custom_1': np.array(['6:45:00', '7:00:00', '7:30:00'])}
        
        volatilities = [0.0,0.3,0.4,0.5,0.6]
        ratios = [0.0,0.2,0.5, 0.6, 0.7, 0.8]
        adx28s = [0,20]
        adx14s = [0,20]
        adx7s = [0,20]
        extreme_rsis = [True, False, "either"]  # Keep as list for string handling

        target_1s = [0.2,0.3, 0.4, 0.5,0.6]
        target_2s = [0.3,0.4, 0.5, 0.6,0.7,0.8]
        target_3s = [0.6, 0.7, 0.8, 0.9,1.0,1.1]
        stop_loss_1s = [-0.5,-0.4,-0.3]
        stop_loss_2s = [-0.1,0.0,0.1,0.2]
        stop_loss_3s = [0.2,0.3,0.4,0.5,0.6]
        '''
        (data_holder, short_rows_data_holder, 
         t1_indexes, t2_indexes, t3_indexes, sl1_indexes, sl2_indexes, sl3_indexes) = Create_2D_List_From_Df(
            df, target_1s, stop_loss_1s, target_3s, stop_loss_2s, target_2s, stop_loss_3s
        )

        # convert data_holder and the para lists into numba stuff
        # target3s_np_array = direct conversion of target_3s
        # t1_idx_arrays = not totally sure
        (data_rows, data_values, data_last_prices, 
         target1s_np_array, target2s_np_array, target3s_np_array, stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array,
         t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
         sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays) = Convert_To_Numba_Arrays(
            data_holder, t1_indexes, t2_indexes, t3_indexes,
            sl1_indexes, sl2_indexes, sl3_indexes,
            target_1s, target_2s, target_3s, 
            stop_loss_1s, stop_loss_2s, stop_loss_3s)
        
        print(f"Pre-processing data done. Now processing {len(data_holder)} rows of data (excluding really short time period trades)")
        message = "Running grid search "
        time_top_sublists = {}
        all_sublists = []
        start_time = time.time()

        # All 3 modes now use the unified Create_Entries function
        if (user_mode == 1):
            print(f"{message} overall sums...")
            # For user_mode=1, pass None for entry_times as it doesn't use time filtering
            all_sublists = Grid_Search_Helper_Create_Combos.Create_Entries(
                   None, volatilities, ratios, avg_of_last_2_trades, adx28s, adx14s, adx7s, extreme_rsis, last_result_must_be_4_minutes,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, target1s_np_array, target2s_np_array, target3s_np_array, 
                   stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode)
        
        elif (user_mode == 2):
            print(f"{message} grouped by volatility sums...")
            # For user_mode=2, pass None for entry_times as it doesn't use time filtering
            all_sublists = Grid_Search_Helper_Create_Combos.Create_Entries(
                   None, volatilities, ratios, avg_of_last_2_trades, adx28s, adx14s, adx7s, extreme_rsis, last_result_must_be_4_minutes,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, target1s_np_array, target2s_np_array, target3s_np_array, 
                   stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode)
            
        elif (user_mode == 3):
            time_top_sublists = {}
            print(f"{message} grouped by time sums (this will take a long time)...")
            for key, value in time_dict.items():
                # each call returns the top x for that time list. each time list is different intervaul times
                time_top_sublists[key] = Grid_Search_Helper_Create_Combos.Create_Entries(
                    value, volatilities, ratios, avg_of_last_2_trades, adx28s, adx14s, adx7s, extreme_rsis, last_result_must_be_4_minutes,
                    target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                    data_values, data_last_prices, target1s_np_array, target2s_np_array, target3s_np_array, 
                    stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array,
                    t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                    how_many_final_parameters, user_mode)

        time_diff_seconds = time.time() - start_time
        minutes = int(time_diff_seconds // 60)
        seconds = int(time_diff_seconds % 60)
        print(f"Total processing time: {minutes} minutes, {seconds} seconds")

        print("Writing results...")
        # Write results only once at the end
        Write_Grid_Seach_Results(all_sublists, time_top_sublists, how_many_final_parameters, user_mode)

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def main():
    data_dir = "Csv_Files/3_Final_Trade_Csvs"
    data_file = "Bulk_Combined-Small_Data_Set.csv"
    df = pd.read_csv(f"{data_dir}/{data_file}")

    Grid_Search_Start(df)

if __name__ == '__main__':
    freeze_support()
    main()