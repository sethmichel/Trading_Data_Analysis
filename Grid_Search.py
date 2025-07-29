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


def Write_Grid_Seach_Results(all_sublists, how_many_final_parameters, user_mode):
    try:
        if (user_mode == 1):
            # overall sum - current list is correct already
            message1 = (f"TEST 1: testing all combos of parameters\n"
                        f"User Mode: {user_mode} - Best Combos Overall:\n"
                        f"time, volatility%, ratio, adx28, 14, 7, abs zscore, rsi_type")
            
            message2 = "\n"
            for i, key in enumerate(all_sublists.keys()):
                sublist = all_sublists[key]
                sublist['sum'] = round(sublist['sum'], 2)
                sublist['count'] = sublist['wins'] + sublist['losses'] + sublist['neither']
                message2 += f"{i+1}) id: {key}, sum: {sublist['sum']}, count: {sublist['count']}, wins: {sublist['wins']}, losses: {sublist['losses']}, neither: {sublist['neither']}\n"

        elif (user_mode == 2):
            # Group sublists by volatility - current list is correct but it's all grouped together
            sorted_sublists_by_volatility = {}

            for sublist_key, sublist_data in all_sublists.items():
                volatility = sublist_key[1]
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
                message2 += f"\n--- Top 10 results for Volatility between {vol} and {upper_bound} (exclusive) ---"
                
                vol_list = sorted_sublists_by_volatility[vol]
                for i in range(min(10, len(vol_list))):
                    key, sublist = vol_list[i]
                    sublist['sum'] = round(sublist['sum'], 2)
                    sublist['count'] = sublist['wins'] + sublist['losses'] + sublist['neither']
                    message2 += f"{i+1}) id: {key}, sum: {sublist['sum']}, count: {sublist['count']}, wins: {sublist['wins']}, losses: {sublist['losses']}, neither: {sublist['neither']}\n"

        elif (user_mode == 3):
            # Group by time - current list is correct but it's all grouped together
            sorted_sublists_by_time = {}

            for sublist_key, sublist_data in all_sublists.items():
                entry_time = sublist_key[0]
                if entry_time not in sorted_sublists_by_time:
                    sorted_sublists_by_time[entry_time] = []
                sorted_sublists_by_time[entry_time].append((sublist_key, sublist_data))

            message1 = (f"TEST 3: testing all combos of parameters by time\n"
                        f"User Mode: {user_mode} - Best Combos for time ranges:\n"
                        f"entry time, volatility%, ratio, adx28, 14, 7, abs zscore, rsi_type")
            
            # For each time, get the top 10 results and add them to message2
            message2 = ""
            for entry_time in sorted_sublists_by_time.keys():
                message2 += f"\n--- Top 10 results for time between {entry_time} and 30 min after (exclusive) ---\n"
                time_list = sorted_sublists_by_time[entry_time]

                for i in range(min(10, len(time_list))):
                    key, sublist = time_list[i]
                    sublist['sum'] = round(sublist['sum'], 2)
                    sublist['count'] = sublist['wins'] + sublist['losses'] + sublist['neither']
                    message2 += f"{i+1}) id: {key}, sum: {sublist['sum']}, count: {sublist['count']}, wins: {sublist['wins']}, losses: {sublist['losses']}, neither: {sublist['neither']}\n"

        # all version need this text
        if how_many_final_parameters == 4:
            message1 += f", t1, t2, sl1, sl2"
        elif how_many_final_parameters == 6:
            message1 += f", t1, t2, t3, sl1, sl2, sl3"

        message = message1 + "\n" + message2
        message = message.replace("'", '').replace("{", '').replace("}", '')

        Write_Analysis(message)
        print("\nCOMPLETE\n")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Convert_To_Numba_Arrays(data_holder, t1_indexes, t2_indexes, t3_indexes, sl1_indexes, sl2_indexes, sl3_indexes,
                            target_1s, target_2s, target_3s, stop_loss_1s, stop_loss_2s, stop_loss_3s):
    try:
        data_rows = np.array([row[0] for row in data_holder], dtype=np.int32)
        data_values = np.array([row[1] for row in data_holder], dtype=np.float64)
        data_last_prices = np.array([row[2] for row in data_holder], dtype=np.float64)
        
        max_row_idx = np.max(data_rows) + 1 if len(data_rows) > 0 else 1

        # --- Create mapping arrays and tiered index arrays for Numba ---
        # these basically store the rows that each parameter point to, it's for faster lookup
        t1_map = np.array(target_1s, dtype=np.float64)
        sl1_map = np.array(stop_loss_1s, dtype=np.float64)
        t3_map = np.array(target_3s, dtype=np.float64)
        sl2_map = np.array(stop_loss_2s, dtype=np.float64)
        t2_map = np.array(target_2s, dtype=np.float64)
        sl3_map = np.array(stop_loss_3s, dtype=np.float64)

        # Create arrays to hold all the index arrays
        t1_idx_arrays = np.zeros((len(t1_map), max_row_idx), dtype=np.int32)
        sl1_idx_arrays = np.zeros((len(sl1_map), max_row_idx), dtype=np.int32)
        t3_idx_arrays = np.zeros((len(t1_map), len(t3_map), max_row_idx), dtype=np.int32)
        sl2_idx_arrays = np.zeros((len(t1_map), len(sl2_map), max_row_idx), dtype=np.int32)
        t2_idx_arrays = np.zeros((len(t1_map), len(t2_map), max_row_idx), dtype=np.int32)
        sl3_idx_arrays = np.zeros((len(t1_map), len(sl3_map), max_row_idx), dtype=np.int32)

        for i, nt in enumerate(t1_map):
            arr = np.full(max_row_idx, 50000, dtype=np.int32)
            if nt in t1_indexes:
                for idx, val in t1_indexes[nt].items():
                    if idx < max_row_idx: arr[idx] = val
            t1_idx_arrays[i] = arr

        for i, nsl in enumerate(sl1_map):
            arr = np.full(max_row_idx, 50000, dtype=np.int32)
            if nsl in sl1_indexes:
                for idx, val in sl1_indexes[nsl].items():
                    if idx < max_row_idx: arr[idx] = val
            sl1_idx_arrays[i] = arr

        for i, nt in enumerate(t1_map):
            # uppers
            for j, ut in enumerate(t3_map):
                arr = np.full(max_row_idx, 50000, dtype=np.int32)
                if nt in t3_indexes and ut in t3_indexes[nt]:
                    for idx, val in t3_indexes[nt][ut].items():
                        if idx < max_row_idx: arr[idx] = val
                t3_idx_arrays[i, j] = arr
                
            for j, usl in enumerate(sl2_map):
                arr = np.full(max_row_idx, 50000, dtype=np.int32)
                if nt in sl2_indexes and usl in sl2_indexes[nt]:
                    for idx, val in sl2_indexes[nt][usl].items():
                        if idx < max_row_idx: arr[idx] = val
                sl2_idx_arrays[i, j] = arr

            # subs
            for j, t2 in enumerate(t2_map):
                arr = np.full(max_row_idx, 50000, dtype=np.int32)
                if nt in t2_indexes and t2 in t2_indexes[nt]:
                    for idx, val in t2_indexes[nt][t2].items():
                        if idx < max_row_idx: arr[idx] = val
                t2_idx_arrays[i, j] = arr

            for j, sl3 in enumerate(sl3_map):
                arr = np.full(max_row_idx, 50000, dtype=np.int32)
                if nt in sl3_indexes and sl3 in sl3_indexes[nt]:
                    for idx, val in sl3_indexes[nt][sl3].items():
                        if idx < max_row_idx: arr[idx] = val
                sl3_idx_arrays[i, j] = arr

        
        return (data_rows, data_values, data_last_prices, 
                t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
                t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays)
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Create_2D_List_From_Df(df, target_1s, stop_loss_1s, target_3s, stop_loss_2s, target_2s, stop_loss_3s):
    try:
        # Pre-process all price movements to avoid repeated string operations
        df['price_movement_list'] = df['Price Movement'].apply(lambda x: [float(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else [])

        # Keep only the columns specified in columns_to_keep
        columns_to_keep = ['price_movement_list','Entry Volatility Percent','Entry Volatility Ratio','Entry Adx28',
                           'Entry Adx14','Entry Adx7','Entry Macd Z-Score','Rsi Extreme Prev Cross', 'Entry Time']
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
                    if start is not high_numb:
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
                    if start is not high_numb:
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
                    if start is not high_numb:
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
                    if start is not high_numb:
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


def Grid_Search_Start(df):
    try:
        global how_many_final_parameters

        # Ask user for mode selection
        print("\nSelect mode:")
        print("1) Overall Combinations")
        print("2) Volatility Groups With Ratio Combinations")
        print("3) Overall Time Group Combinations")
        user_mode = int(input("Enter the number of the mode you want to use: "))
        if (user_mode not in [1,2,3]):
            raise ValueError("Pick either 1, 2, or 3")

        how_many_final_parameters = 4
        
        
        # high compute real lists - 20 minutes on 147 rows of data (overall excluding time)
        entry_times = np.array(['6:30:00', '6:45:00', '7:00:00', '7:30:00']) 
        volatilities = np.array([0.0,0.3,0.4,0.5,0.6,0.7,0.8], dtype=np.float64)
        ratios = np.array([0.0,0.2,0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
        adx28s = np.array([0,20, 30, 40], dtype=np.float64)
        adx14s = np.array([0,20, 30, 40], dtype=np.float64)
        adx7s = np.array([0,20, 30, 40], dtype=np.float64)
        abs_macd_zScores = np.array([0,0.5, 1.0, 1.5, 2.0], dtype=np.float64)   # absolute value of z-score, not normal z-score
        extreme_rsis = [True, False, "either"]  # Keep as list for string handling

        target_1s = np.array([0.2,0.3, 0.4, 0.5], dtype=np.float64)
        target_2s = np.array([0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)      # must be higher than nt and less than ut
        target_3s = np.array([0], dtype=np.float64)
        stop_loss_1s = np.array([-0.5,-0.4,-0.3], dtype=np.float64)
        stop_loss_2s = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3], dtype=np.float64)
        stop_loss_3s = np.array([0], dtype=np.float64) # must be higher than nsl and HIGHER than usl
        '''
        # low compute testing lists
        entry_times = np.array(['6:30:00', '7:30:00']) 
        volatilities = np.array([0.0,0.2,0.6], dtype=np.float64)
        ratios = np.array([0.0,0.2,0.8], dtype=np.float64)
        adx28s = np.array([0,40], dtype=np.float64)
        adx14s = np.array([0,40], dtype=np.float64)
        adx7s = np.array([0,40], dtype=np.float64)
        abs_macd_zScores = np.array([0,1.5], dtype=np.float64)   # absolute value of z-score, not normal z-score
        extreme_rsis = [False, "either"]  # Keep as list for string handling

        target_1s = np.array([0.2,0.3, 0.4, 0.5], dtype=np.float64)
        target_2s = np.array([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)      # must be higher than nt and less than ut
        target_3s = np.array([5], dtype=np.float64) # must be higher than t1,t2, and all sl's
        stop_loss_1s = np.array([-0.5,-0.4,-0.3], dtype=np.float64)
        stop_loss_2s = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3], dtype=np.float64)
        stop_loss_3s = np.array([5], dtype=np.float64) # must be higher than sl2 and HIGHER than t2
        '''
        (data_holder, short_rows_data_holder, 
         t1_indexes, t2_indexes, t3_indexes, sl1_indexes, sl2_indexes, sl3_indexes) = Create_2D_List_From_Df(
            df, target_1s.tolist(), stop_loss_1s.tolist(), target_3s.tolist(), 
            stop_loss_2s.tolist(), target_2s.tolist(), stop_loss_3s.tolist()
        )

        # more preprocessing. all processing modes use this same data
        (data_rows, data_values, data_last_prices, 
         t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
         t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
         sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays) = Convert_To_Numba_Arrays(
            data_holder, t1_indexes, t2_indexes, t3_indexes,
            sl1_indexes, sl2_indexes, sl3_indexes,
            target_1s, target_2s, target_3s, 
            stop_loss_1s, stop_loss_2s, stop_loss_3s)
        
        print(f"Pre-processing data done. Now processing {len(data_holder)} rows of data (excluding really short time period trades)")
        message = "Running grid search "
        
        start_time = time.time()

        # all 3 modes filter rows different, it's better to just run it 3 times than to spend a day coding a 1 run solution
        if (user_mode == 1):
            print(f"{message} overall sums...")
            all_sublists = Grid_Search_Helper_Create_Combos.Create_Entries_Overall(
                   volatilities, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode)
        
        elif (user_mode == 2):
            print(f"{message} grouped by volatility sums...")
            all_sublists = Grid_Search_Helper_Create_Combos.Create_Entries_Group_By_Volatility(
                   volatilities, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode)
            
        elif (user_mode == 3):
            print(f"{message} grouped by time sums (this will take a long time)...")
            all_sublists = Grid_Search_Helper_Create_Combos.Create_Entries_Group_By_Time(
                   entry_times, volatilities, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode)

        time_diff_seconds = time.time() - start_time
        minutes = int(time_diff_seconds // 60)
        seconds = int(time_diff_seconds % 60)
        print(f"Total processing time: {minutes} minutes, {seconds} seconds")

        print("Writing results...")
        # Write results only once at the end
        Write_Grid_Seach_Results(all_sublists, how_many_final_parameters, user_mode)

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def main():
    data_dir = "Csv_Files/3_Final_Trade_Csvs"
    data_file = "Bulk_Combined.csv"
    df = pd.read_csv(f"{data_dir}/{data_file}")

    Grid_Search_Start(df)

if __name__ == '__main__':
    freeze_support()
    main()