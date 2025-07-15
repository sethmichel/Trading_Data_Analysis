import pandas as pd
import numpy as np
import os
import inspect
import sys
import shutil
import Main_Globals
from datetime import datetime
import concurrent.futures
import threading
from multiprocessing import freeze_support
from numba import jit, prange
import itertools

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


def Write_Analysis(message):
    # Write to Analysis_Results.txt - overwrite if exists, create if doesn't exist
    with open("Analysis_Results.txt", "a") as file:
        file.write(message)


# scans price movement for if price satisfied 0.3/-0.3. if neither are there then use the final value of the list
def Helper_Volatility_Percent_vs_Ratio_Target_Finder_1(df):
    # scan whole original df first, add a new column, then filter that df
    
    def find_target_result(price_movement_str):
        if price_movement_str == '':
            return None
            
        movements = [float(x.strip()) for x in price_movement_str.split('|')]
        for movement in movements:
            if movement == 0.3:
                return 0.3
            elif movement == -0.3:
                return -0.3
                
        # If neither 0.3 nor -0.3 found, return the final value
        return movements[-1]
    
    # Apply the function to create the new column
    df['target_Basic_0.3_-0.3_Result'] = df['Price Movement'].apply(find_target_result)
    
    return df

# scans price movement for if price satisfied target = 0.5, upper target = 0.9, upper stop loss = -0.1, stop loss = -0.5.
# if it doesn't then use the final value of the list
def Helper_Volatility_Percent_vs_Ratio_Target_Finder_2(df):
    target = 0.5
    upper_target = 0.9
    upper_stop_loss = -0.1
    stop_loss = -0.5
    
    def find_target_result(price_movement_str):
        if price_movement_str == '':
            return None
            
        movements = [float(x.strip()) for x in price_movement_str.split('|')]
        
        # Phase 1: Look for target or stop_loss
        for movement in movements:
            if movement == stop_loss:
                return stop_loss
            
            elif movement == target:
                # Switch to phase 2: look for upper_target or upper_stop_loss
                remaining_movements = movements[movements.index(movement) + 1:]
                for remaining_movement in remaining_movements:
                    if remaining_movement == upper_stop_loss:
                        return upper_stop_loss
                    
                    elif remaining_movement == upper_target:
                        return upper_target
                    
                # If we reach here, neither upper target nor upper stop loss found
                return movements[-1]
        
        # If neither target nor stop_loss found in phase 1, return final value
        return movements[-1]
    
    # Apply the function to create the new column
    df['Target_Complex_0.5_0.9_Result'] = df['Price Movement'].apply(find_target_result)
    
    return df


def Helper_Volailtity_Percent_vs_Ratio_Find_Top_5(result_data_basic, result_data_complex):
    best_basic_combos = []
    best_complex_combos = []
    
    # Round net change values to 2 decimal places
    for combo_id in result_data_basic:
        result_data_basic[combo_id]['net change'] = round(result_data_basic[combo_id]['net change'], 2)
    
    for combo_id in result_data_complex:
        result_data_complex[combo_id]['net change'] = round(result_data_complex[combo_id]['net change'], 2)
    
    # Find the top 5 combos in result_data_basic by 'net change'
    sorted_basic = sorted(
        result_data_basic.items(),
        key=lambda x: x[1]['net change'],
        reverse=True
    )
    best_basic_combos = []
    for combo_id, entry in sorted_basic[:5]:
        entry_with_id = entry.copy()
        # Move 'combo id' to be the first attribute
        entry_with_id = {'combo id': combo_id, **entry_with_id}
        best_basic_combos.append(entry_with_id)

    # Do the same for result_data_complex
    sorted_complex = sorted(
        result_data_complex.items(),
        key=lambda x: x[1]['net change'],
        reverse=True
    )
    best_complex_combos = []
    for combo_id, entry in sorted_complex[:5]:
        entry_with_id = entry.copy()
        # Move 'combo id' to be the first attribute
        entry_with_id = {'combo id': combo_id, **entry_with_id}
        best_complex_combos.append(entry_with_id)

    return best_basic_combos, best_complex_combos


'''
use all combos of volatility % and volatility ratio
do it on 0.3/-0.3 and on 
target = 0.5, upper target = 0.9, upper stop loss = -0.1, stop loss = -0.5
output: organized text saying what the best 5 combos are, their results, their counts
'''
def Volatility_Percent_vs_Ratio(df):
    global analysis_message

    percents = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    ratios = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
    result_data_basic = {}
    result_data_complex = {}
    
    # add 0.3/-0.3 target results and then the 0.5/0.9/-0.1/-0.5 one
    df = Helper_Volatility_Percent_vs_Ratio_Target_Finder_1(df)
    df = Helper_Volatility_Percent_vs_Ratio_Target_Finder_2(df)

    for percent in percents:
        for ratio in ratios:
            filtered_df = df[(df['Entry Volatility Percent'] >= percent) & (df['Entry Volatility Ratio'] >= ratio)]
            combo_id = f'{percent},{ratio}'
            result_data_basic[combo_id] = {"count": filtered_df.shape[0],  # total rows 
                                           "wins": 0,         # total wins
                                           'losses': 0,       # total losses
                                           'neither': 0,      # it didn't win or lose (exited at macd or something else)
                                           'net change': 0}   # % change
            
            result_data_complex[combo_id] = {"count": filtered_df.shape[0],
                                             "upper target wins": 0,
                                             'upper sl losses': 0,
                                             "lower sl losses": 0,
                                             'neither': 0,
                                             'net change': 0}

            for idx, row in filtered_df.iterrows():
                # 1) use 0.3 / -0.3. Get the count, win rate, percent change
                result_1 = row['target_Basic_0.3_-0.3_Result']
            
                if (result_1 == 0.3):
                    result_data_basic[combo_id]["wins"] += 1
                elif (result_1 == -0.3):
                    result_data_basic[combo_id]["losses"] += 1
                else:
                    result_data_basic[combo_id]["neither"] += 1

                result_data_basic[combo_id]['net change'] += result_1
    
                # 2) use the complex paramters. record the same things
                result_2 = row['Target_Complex_0.5_0.9_Result']

                if (result_2 == 0.9):
                    result_data_complex[combo_id]["upper target wins"] += 1
                elif (result_2 == -0.1):
                    result_data_complex[combo_id]["upper sl losses"] += 1
                elif (result_2 == -0.5):
                    result_data_complex[combo_id]["lower sl losses"] += 1
                else:
                    result_data_complex[combo_id]["neither"] += 1

                result_data_complex[combo_id]['net change'] += result_2

    # 3 find the best 5 combos for each subtest by net change. save the combo id as a new attribute
    best_basic_combos, best_complex_combos = Helper_Volailtity_Percent_vs_Ratio_Find_Top_5(result_data_basic, result_data_complex)

    # 4) write the results
    subtest_1_results = ""
    subtest_2_results = ""
    for i in range(0, len(best_basic_combos)):
        subtest_1_results += f"{i+1}) {best_basic_combos[i]}\n"

    for i in range(0, len(best_complex_combos)):
        subtest_2_results += f"{i+1}) {best_complex_combos[i]}\n"
    
    message = (f"TEST 1: testing all combos of volaility percent vs volatility ratio.\n"
               f"parameters: subtest 1: 0.3/-0.3, subtest 2: 0.5/0.9 lower/upper targets with -0.5/-0.1 lower/upper stop losses\n"
               f"subtest 1:\n"
               f"{subtest_1_results}\n"
               f"subtest 2:\n"
               f"{subtest_2_results}\n\n")
    message = message.replace("'", '').replace("{", '').replace("}", '')

    Write_Analysis(message)


def Write_Grid_Seach_Results(all_sublists):
    best_sublists_sum = {}
    best_sublists_winrate = {}
    
    # find top 10 sublists by sum
    sum_sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['sum'], reverse=True)
    for i in range(min(30, len(sum_sorted_items))):
        key, sublist = sum_sorted_items[i]
        sublist_rounded = sublist.copy()
        sublist_rounded['sum'] = round(sublist_rounded['sum'], 2)
        best_sublists_sum[key] = sublist_rounded

    # find top 10 sublists by winrate
    winrate_sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['winrate'], reverse=True)
    for i in range(min(30, len(winrate_sorted_items))):
        key, sublist = winrate_sorted_items[i]
        sublist_rounded = sublist.copy()
        sublist_rounded['winrate'] = sublist_rounded['winrate']
        best_sublists_winrate[key] = sublist_rounded

    message = (f"TEST 2: testing all combos of volatility percent vs volatility ratio vs parameters.\n"
        f"parameters: using an upper/lower target and upper/lower stop loss\n"
        f"Total combinations tested: {len(all_sublists)}\n"
        f"Results (Top 10 by sum):\n"
        f"volatility, ratio, adx28, 14, 7, ads zscore, rsi_type, normal_target, upper_target, normal_stop_loss, upper_stop_loss\n")

    for i, (key, sub_list) in enumerate(best_sublists_sum.items()):
        message += f"{i+1}) id: {sub_list['id']}, sum: {sub_list['sum']}, count: {sub_list['count']}, wins: {sub_list['wins']}, losses: {sub_list['losses']}, neither: {sub_list['neither']}\n"
        
    message += (f"\nResults (Top 10 by win rate):\n"
                f"volatility, ratio, adx28, 14, 7, ads zscore, rsi_type, normal_target, upper_target, normal_stop_loss, upper_stop_loss\n")

    for i, (key, sub_list) in enumerate(best_sublists_winrate.items()):
        message += f"{i+1}) id: {sub_list['id']}, sum: {sub_list['sum']}, count: {sub_list['count']}, wins: {sub_list['wins']}, losses: {sub_list['losses']}, neither: {sub_list['neither']}\n"

    message = message.replace("'", '').replace("{", '').replace("}", '')

    Write_Analysis(message)
    print("\nit's done\n")


def Grid_Search_Helper__Find_NT_Indexes(filtered_df, normal_target):
    try:
        # Vectorized approach - process all rows at once
        def find_first_target_index(price_movement_list):
            try:
                return next(i for i, val in enumerate(price_movement_list) if val == normal_target)
            except StopIteration:
                return None
        
        # Apply to entire column at once instead of manual loops
        # Using a list comprehension is more direct and avoids pandas converting int/None to float/NaN
        return [find_first_target_index(pm_list) for pm_list in filtered_df['price_movement_list']]
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Grid_Search_Helper__Find_NSL_Indexes(filtered_df, normal_sl, normal_target_index_list):
    try:
        # Vectorized approach with zip for parallel processing
        def find_sl_before_target(price_movement_list, target_idx):
            end_index = target_idx if target_idx is not None else len(price_movement_list)
            try:
                return next(i for i in range(end_index) if price_movement_list[i] == normal_sl)
            except StopIteration:
                return None
        
        # Process all rows at once using zip
        return [
            find_sl_before_target(pm_list, target_idx) 
            for pm_list, target_idx in zip(filtered_df['price_movement_list'], normal_target_index_list)
        ]

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# preformance note: it's much better to filter the df's as do the for loops
# adx28s, adx14s, adx7s parameters
def process_volatility_chunk(volatility, df, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis, normal_targets, upper_targets, upper_stop_losss, normal_stop_losss):
    try:
        local_sublists = {}
        df_vol = df[df['Entry Volatility Percent'] >= volatility]
        
        for ratio in ratios:
            df_ratio = df_vol[df_vol['Entry Volatility Ratio'] >= ratio]
            for adx28 in adx28s:
                df_adx28 = df_ratio[df_ratio['Entry Adx28'] >= adx28]
                for adx14 in adx14s:
                    df_adx14 = df_adx28[df_adx28['Entry Adx14'] >= adx14]
                    for adx7 in adx7s:
                        df_adx7 = df_adx14[df_adx14['Entry Adx7'] >= adx7]
                        for zscore in abs_macd_zScores:
                            df_zscore = df_adx7[abs(df_adx7['Entry Macd Z-Score']) >= zscore]

                            for rsi_type in extreme_rsis:
                                # Handle "either" case for RSI - include all entries regardless of RSI value
                                if rsi_type == "either":
                                    filtered_df = df_zscore
                                else:
                                    filtered_df = df_zscore[df_zscore['Rsi Extreme Prev Cross'] == rsi_type]

                                # if no results just skip it
                                if (len(filtered_df) == 0):
                                    continue

                                for normal_target in normal_targets:
                                    # 1) each iteration find all indexes of normal_target in price movement for each row of filtered_df (None if it's not there)
                                    normal_target_index_list = Grid_Search_Helper__Find_NT_Indexes(filtered_df, normal_target)
                                    
                                    for normal_stop_loss in normal_stop_losss:
                                        # 2) since we know normal_target index, find the index of each normal stop loss in price movement for each row of filtered df
                                        #    BUT, only check up to normal target index
                                        normal_sl_index_list = Grid_Search_Helper__Find_NSL_Indexes(filtered_df, normal_stop_loss, normal_target_index_list)                                
                                        
                                        for upper_target in upper_targets:
                                            if (upper_target <= normal_target):
                                                continue

                                            for upper_stop_loss in upper_stop_losss:
                                                if ((upper_stop_loss >= upper_target) or upper_stop_loss >= normal_target):
                                                    continue
                                
                                                # makeing the key a tuple is faster than a string
                                                sublist_key = (volatility, ratio, adx28, adx14, adx7, zscore, rsi_type, normal_target, upper_target, normal_stop_loss, upper_stop_loss)
                                                sublist = {'id': sublist_key, 'sum': 0, 'count': len(filtered_df), 'wins': 0, 'losses': 0, 'neither': 0, 'winrate': 0}

                                                # 3) we have indexes of normal target and normal stop loss. if normal stop loss appears first use that.
                                                #    otherwise start looking for upper target and upper stop loss 1 index after normal target
                                                for row_idx, (_, row) in enumerate(filtered_df.iterrows()):
                                                    price_movement_list = row['price_movement_list']
                                                    normal_target_idx = normal_target_index_list[row_idx]
                                                    normal_sl_idx = normal_sl_index_list[row_idx]
                                                    
                                                    # Case 1: Neither normal target nor normal stop loss appear
                                                    if (normal_sl_idx is None and normal_target_idx is None):
                                                        sublist['sum'] += price_movement_list[-1]
                                                        sublist['neither'] += 1

                                                    # Case 2: Normal stop loss appears before normal target
                                                    elif (normal_sl_idx is not None and 
                                                        (normal_target_idx is None or normal_sl_idx < normal_target_idx)):
                                                        sublist['sum'] += normal_stop_loss
                                                        sublist['losses'] += 1
                                                        
                                                    # Case 3: Normal target appears first, now look for upper targets/stops (we know normal target is not none)
                                                    else:
                                                        # Check if normal target is the last value
                                                        if normal_target_idx == len(price_movement_list) - 1:
                                                            sublist['sum'] += normal_target
                                                            sublist['wins'] += 1

                                                        else:
                                                            # Look for upper targets/stops after normal target
                                                            found_upper = False
                                                            for i in range(normal_target_idx + 1, len(price_movement_list)):
                                                                value = price_movement_list[i]

                                                                if value == upper_target:
                                                                    sublist['sum'] += upper_target
                                                                    sublist['wins'] += 1
                                                                    found_upper = True
                                                                    break

                                                                elif (value == upper_stop_loss):
                                                                    sublist['sum'] += upper_stop_loss
                                                                    sublist['losses'] += 1
                                                                    found_upper = True
                                                                    break
                                                            
                                                            # If no upper target/stop found, use final value
                                                            if found_upper == False:
                                                                sublist['sum'] += price_movement_list[-1]
                                                                sublist['neither'] += 1

                                                # this is after all trades have been looped - find winrate for this parameter combo
                                                if sublist['count'] > 0:
                                                    sublist['winrate'] = round(sublist['wins'] / sublist['count'], 2)
                                                else:
                                                    sublist['winrate'] = 0
                                                
                                                local_sublists[sublist_key] = sublist

        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# this multithreasd like mad, it's optimzied for my cpu: Intel(R) Core(TM) i5-14400F. check how many threads
#     each core of you cpu can handle
# 10 cores = 6 performance, 4 effiecient cores.
'''
my understanding is the 6p cores have 12 threads that support hyperthreading while the 4e cores have 4 slower threads
with no hyperthreading. it assignes cores based on availablilty so I need to give it to p cores. apparently you can't
tell cores apart in python without some tools, so I'll just deal with it
'''
def Grid_Search_Parameter_Optimization(df):
    try:
        volatilities = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        ratios = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
        adx28s = [0,10,20,30,40,50,60]
        adx14s = [0,10,20,30,40,50,60]
        adx7s = [0,10,20,30,40,50,60]
        abs_macd_zScores = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]   # absolute value of z-score, not normal z-score
        extreme_rsis = [True, False, "either"]
        normal_targets = [0.2,0.3,0.4,0.5,0.6]
        upper_targets = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        upper_stop_losss = [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
        normal_stop_losss = [-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
        
        # Pre-process price movements to avoid repeated string operations
        df['price_movement_list'] = df['Price Movement'].apply(
            lambda x: [float(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else []
        )
        
        all_sublists = {}
        
        num_workers = len(volatilities)
        print(f"Initializing a pool of {num_workers} worker processes.")
        
        # Use ProcessPoolExecutor to process each volatility value in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit one task per volatility value
            future_to_volatility = {
                executor.submit(
                    process_volatility_chunk, 
                    volatility, df, ratios, adx28s, adx14s, adx7s, abs_macd_zScores,
                    extreme_rsis, normal_targets, upper_targets, upper_stop_losss, normal_stop_losss
                ): volatility 
                for volatility in volatilities
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_volatility):
                volatility = future_to_volatility[future]
                try:
                    local_sublists = future.result()
                    # Merge local results into main dictionary
                    all_sublists.update(local_sublists)
                    print(f"Completed processing volatility {volatility}")
                except Exception as exc:
                    print(f"Thread processing volatility {volatility} generated an exception: {exc}")
                    # Stop everything and provide details
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise RuntimeError(f"Thread failed while processing volatility {volatility}: {exc}")
        
        # Write results only once at the end
        Write_Grid_Seach_Results(all_sublists)

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def main():
    # Ensure the text file "Analysis_Results.txt" exists (create if it doesn't)
    if not os.path.exists("Analysis_Results.txt"):
        with open("Analysis_Results.txt", "w") as f_create:
            pass
    # erase the text file if it exists
    else:
        with open("Analysis_Results.txt", "w") as f:
            pass

    data_dir = "Csv_Files/3_Final_Trade_Csvs"
    data_file = "Bulk_Combined.csv"
    df = pd.read_csv(f"{data_dir}/{data_file}")

    #Volatility_Percent_vs_Ratio(df)
    Grid_Search_Parameter_Optimization(df)

if __name__ == '__main__':
    freeze_support()
    main()
