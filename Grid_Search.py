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
import itertools
import heapq
from numba import jit, prange, typed, types
from numba.core import types as nb_types
from numba.typed import Dict, List

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


def Write_Analysis(message):
    # Write to Analysis_Results.txt - overwrite if exists, create if doesn't exist
    with open("Analysis_Results.txt", "a") as file:
        file.write(message)


def Write_Grid_Seach_Results(all_sublists):
    best_sublists_sum = {}
    #best_sublists_winrate = {}
    
    # find top x sublists by sum
    sum_sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['sum'], reverse=True)
    for i in range(min(50, len(sum_sorted_items))):
        key, sublist = sum_sorted_items[i]
        sublist_rounded = sublist.copy()
        sublist_rounded['sum'] = round(sublist_rounded['sum'], 2)
        sublist_rounded['count'] = sublist_rounded['wins'] + sublist_rounded['losses'] + sublist_rounded['neither']
        best_sublists_sum[key] = sublist_rounded

    # find top x sublists by winrate
    '''winrate_sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['winrate'], reverse=True)
    for i in range(min(50, len(winrate_sorted_items))):
        key, sublist = winrate_sorted_items[i]
        sublist_rounded = sublist.copy()
        sublist_rounded['winrate'] = sublist_rounded['winrate']
        best_sublists_winrate[key] = sublist_rounded
    '''

    message = (f"TEST 2: testing all combos of volatility percent vs volatility ratio vs parameters.\n"
        f"parameters: using an upper/lower target and upper/lower stop loss\n"
        f"Total combinations tested: {len(all_sublists)}\n"
        f"Results (Top 10 by sum):\n"
        f"volatility, ratio, adx28, 14, 7, ads zscore, rsi_type, normal_target, upper_target, normal_stop_loss, upper_stop_loss\n")

    for i, (key, sub_list) in enumerate(best_sublists_sum.items()):
        message += f"{i+1}) id: {key}, sum: {sub_list['sum']}, count: {sub_list['count']}, wins: {sub_list['wins']}, losses: {sub_list['losses']}, neither: {sub_list['neither']}\n"
        
    message += (f"\nResults (Top 10 by win rate):\n"
                f"volatility, ratio, adx28, 14, 7, ads zscore, rsi_type, normal_target, upper_target, normal_stop_loss, upper_stop_loss\n")

    #for i, (key, sub_list) in enumerate(best_sublists_winrate.items()):
    #    message += f"{i+1}) id: {sub_list['id']}, sum: {sub_list['sum']}, count: {sub_list['count']}, wins: {sub_list['wins']}, losses: {sub_list['losses']}, neither: {sub_list['neither']}\n"

    message = message.replace("'", '').replace("{", '').replace("}", '')

    Write_Analysis(message)
    print("\nit's done\n")


def prune_sublists(local_sublists, keep_count=50):
    """Prune local_sublists to keep only the top entries by sum"""
    try:
        if len(local_sublists) <= keep_count:
            return local_sublists
            
        # Use heapq.nlargest for faster top-N selection without full sorting
        top_items = heapq.nlargest(keep_count, local_sublists.items(), key=lambda x: x[1]['sum'])
        pruned_sublists = dict(top_items)
        
        return pruned_sublists
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return local_sublists


# filtered_rows = [(index of original df, [list of values], last price), ...]
def process_batch(batch_combinations, local_sublists, normal_target_indexes, normal_sl_indexes, 
                  upper_target_indexes, upper_sl_indexes):
    try:
        for combo in batch_combinations:
            filtered_rows = combo['filtered_rows']
            normal_target = combo['normal_target']
            normal_stop_loss = combo['normal_stop_loss']
            upper_target = combo['upper_target']
            upper_stop_loss = combo['upper_stop_loss']

            total_rows = len(filtered_rows)
            sixty_percent_mark = int(total_rows * 0.6)
            eighty_percent_mark = int(total_rows * 0.8)
            bad_combo_flag = False
            sublist = {'sum': 0, 'wins': 0, 'losses': 0, 'neither': 0}

            for i, row in enumerate(filtered_rows):
                row_idx = row[0]
                # EARLY EXIT: if we're x% through and sum is less than y
                if (i >= sixty_percent_mark and sublist['sum'] < 4):
                    bad_combo_flag = True
                    break
                elif (i >= eighty_percent_mark and sublist['sum'] < 7):
                    bad_combo_flag = True
                    break
                    
                normal_target_idx = normal_target_indexes[normal_target][row_idx]
                normal_target_sl_idx = normal_sl_indexes[normal_stop_loss][row_idx]
                
                # case 1: nsl is before nt
                if (normal_target_sl_idx < normal_target_idx):
                    sublist['sum'] += normal_stop_loss
                    sublist['losses'] += 1
                    continue
                
                # case 2: nt is before nsl
                elif (normal_target_idx < normal_target_sl_idx):
                    # continue to upper values
                    upper_target_idx = upper_target_indexes[normal_target][upper_target][row_idx]
                    upper_sl_idx = upper_sl_indexes[normal_target][upper_stop_loss][row_idx]

                    # case 2a: if ut is before usl
                    if (upper_target_idx < upper_sl_idx):
                        sublist['sum'] += upper_target
                        sublist['wins'] += 1
                        continue

                    # case 2b: if usl is before ut
                    elif (upper_sl_idx < upper_target_idx):
                        sublist['sum'] += upper_stop_loss
                        sublist['losses'] += 1
                        continue

                # case 3: either nt and nsl aren't there OR nt is there but ut and usl aren't there
                sublist['sum'] += row[2]
                sublist['neither'] += 1
                continue

            if (bad_combo_flag == False):
                # all checks completed
                sublist_key = (combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], combo['adx7'], combo['zscore'],
                            combo['rsi_type'], combo['normal_target'], combo['upper_target'], combo['normal_stop_loss'], 
                            combo['upper_stop_loss'])
                
                local_sublists[sublist_key] = sublist
            
                # Prune local_sublists when it reaches 200,000 entries
                if len(local_sublists) >= 400000:
                    local_sublists = prune_sublists(local_sublists, keep_count=50)
            else:
                bad_combo_flag = True
                
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# data_holder = [(index of original df, [list of values], last price), ...]
def Create_Entries(volatilities, data_holder, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis, normal_targets, 
                   upper_targets, upper_stop_losss, normal_stop_losss, normal_target_indexes, normal_sl_indexes,
                   upper_target_indexes, upper_sl_indexes):
    try:
        local_sublists = {}
        batch_combinations = []
        batch_size = 200000
        
        for volatility in volatilities:
            # data_holder is a 2d list
            # values are: [volatility, ratio, adx28, adx14, adx7, zscore, rsi_type]
            rows_vol = [row for row in data_holder if row[1][0] >= volatility]
            for ratio in ratios:
                rows_ratio = [row for row in rows_vol if row[1][1] >= ratio]
                for adx28 in adx28s:
                    rows_adx28 = [row for row in rows_ratio if row[1][2] >= adx28]
                    for adx14 in adx14s:
                        rows_adx14 = [row for row in rows_adx28 if row[1][3] >= adx14]
                        for adx7 in adx7s:
                            rows_adx7 = [row for row in rows_adx14 if row[1][4] >= adx7]
                            for zscore in abs_macd_zScores:
                                rows_zscore = [row for row in rows_adx7 if abs(row[1][5]) >= zscore]
                                for rsi_type in extreme_rsis:
                                    if rsi_type == "either":
                                        filtered_rows = rows_zscore
                                    else:
                                        filtered_rows = [row for row in rows_zscore if row[1][6] == rsi_type]

                                    if not filtered_rows:
                                        continue

                                    for normal_target in normal_targets:
                                        for normal_stop_loss in normal_stop_losss:
                                            for upper_target in upper_targets:
                                                if (upper_target <= normal_target):
                                                    continue

                                                for upper_stop_loss in upper_stop_losss:
                                                    if ((upper_stop_loss >= upper_target) or upper_stop_loss >= normal_target):
                                                        continue
                                
                                                    batch_combinations.append({
                                                        'filtered_rows': filtered_rows,
                                                        'volatility': volatility,
                                                        'ratio': ratio,
                                                        'adx28': adx28,
                                                        'adx14': adx14,
                                                        'adx7': adx7,
                                                        'zscore': zscore,
                                                        'rsi_type': rsi_type,
                                                        'normal_target': normal_target,
                                                        'normal_stop_loss': normal_stop_loss,
                                                        'upper_target': upper_target,
                                                        'upper_stop_loss': upper_stop_loss
                                                    })
                                                    
                                                    # Process batch when it reaches the specified size
                                                    if len(batch_combinations) >= batch_size:
                                                        local_sublists = process_batch(batch_combinations, local_sublists, normal_target_indexes, 
                                                                                       normal_sl_indexes, upper_target_indexes, upper_sl_indexes)
                                                        batch_combinations = []  # Reset batch
        
        # Process any remaining combinations in the final batch
        if batch_combinations:
            local_sublists = process_batch(batch_combinations, local_sublists, normal_target_indexes, 
                        normal_sl_indexes, upper_target_indexes, upper_sl_indexes)
            
            if (len(local_sublists) > 50):
                local_sublists = prune_sublists(local_sublists, keep_count=50)
                                                    
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Create_2D_List_From_Df(df, normal_targets, normal_stop_losss, upper_targets, upper_stop_losss):
    try:
        # Pre-process all price movements to avoid repeated string operations
        df['price_movement_list'] = df['Price Movement'].apply(lambda x: [float(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else [])

        # Keep only the columns specified in columns_to_keep
        columns_to_keep = ['price_movement_list','Entry Volatility Percent','Entry Volatility Ratio','Entry Adx28',
                        'Entry Adx14','Entry Adx7','Entry Macd Z-Score','Rsi Extreme Prev Cross']
        df = df[[col for col in columns_to_keep if col in df.columns]].copy()

        # Separate rows where 'price_movement_list' length <= 3. we'll deal with them at the end
        short_rows_df = df[df['price_movement_list'].apply(lambda x: len(x) <= 3)].copy()
        df = df[df['price_movement_list'].apply(lambda x: len(x) > 3)].copy()
        df.reset_index(drop=True, inplace=True)
        
        # find all indexes lists (they're different lengths)
        normal_targets_indexes = {}
        normal_sl_indexes = {}
        upper_target_indexes = {}
        upper_sl_indexes = {}
        high_numb = 50000
        
        for target in normal_targets:
            normal_targets_indexes[target] = {}
            for idx, row in df.iterrows():
                for (i, value) in enumerate(row['price_movement_list']):
                    if (value == target):
                        normal_targets_indexes[target][idx] = i
                        break
                else:
                    normal_targets_indexes[target][idx] = high_numb

        for sl in normal_stop_losss:
            normal_sl_indexes[sl] = {} 
            for idx, row in df.iterrows():
                for (i, value) in enumerate(row['price_movement_list']):
                    if (value == sl):
                        normal_sl_indexes[sl][idx] = i
                        break
                else:
                    normal_sl_indexes[sl][idx] = high_numb

        # uppers are different. they must start after normal target, but each normal target does each upper target/sl.
        #     so, you have to complicate the data structure. list[normal target][upper target][inx]
        for normal_target in normal_targets:
            upper_target_indexes[normal_target] = {}
            for upper_target in upper_targets:
                upper_target_indexes[normal_target][upper_target] = {}
                for idx, row in df.iterrows():
                    start = normal_targets_indexes[normal_target][idx] +1
                    if start is not high_numb:
                        for i, value in enumerate(row['price_movement_list'][start:]):
                            if value == upper_target:
                                upper_target_indexes[normal_target][upper_target][idx] = start + i
                                break
                        else:
                            upper_target_indexes[normal_target][upper_target][idx] = high_numb
                    else:
                        upper_target_indexes[normal_target][upper_target][idx] = high_numb

        for normal_target in normal_targets:
            upper_sl_indexes[normal_target] = {}
            for upper_sl in upper_stop_losss:
                upper_sl_indexes[normal_target][upper_sl] = {}
                for idx, row in df.iterrows():
                    start = normal_targets_indexes[normal_target][idx] +1
                    if start is not high_numb:
                        for i, value in enumerate(row['price_movement_list'][start:]):
                            if value == upper_sl:
                                upper_sl_indexes[normal_target][upper_sl][idx] = start + i
                                break
                        else:
                            upper_sl_indexes[normal_target][upper_sl][idx] = high_numb
                    else:
                        upper_sl_indexes[normal_target][upper_sl][idx] = high_numb
            
        # we don't need this anymore and it's hard to deal with later if we leave it in
        columns_to_keep.pop(0)

        # now convert both df's into a minimized list so I can index really fast
        # [(index of original df, [list of values], last price)]
        data_holder = [] 
        for idx, row in df.iterrows():
            data_holder.append((idx, row[columns_to_keep].tolist(), row['price_movement_list'][-1]))

        short_rows_data_holder = []
        for idx, row in short_rows_df.iterrows():
            short_rows_data_holder.append((idx, row[columns_to_keep].tolist(), row['price_movement_list'][-1]))

        return data_holder, short_rows_data_holder, normal_targets_indexes, normal_sl_indexes, upper_target_indexes, upper_sl_indexes
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Grid_Search_Parameter_Optimization(df):
    try:
        volatilities = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        ratios = [0.5,0.6,0.7,0.8,0.9,1.0,1.1]
        adx28s = [20,30,40,50,60]
        adx14s = [20,30,40,50,60]
        adx7s = [20,30,40,50,60]
        abs_macd_zScores = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]   # absolute value of z-score, not normal z-score
        extreme_rsis = [True, False, "either"]
        normal_targets = [0.2,0.3,0.4,0.5]
        upper_targets = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        upper_stop_losss = [0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3]
        normal_stop_losss = [-0.3,-0.4,-0.5]

        data_holder, short_rows_data_holder, normal_target_indexes, normal_sl_indexes, upper_target_indexes, upper_sl_indexes = Create_2D_List_From_Df(df, normal_targets, normal_stop_losss, upper_targets, upper_stop_losss)
        
        all_sublists = Create_Entries(
            volatilities, data_holder, ratios, adx28s, adx14s, adx7s, abs_macd_zScores,
            extreme_rsis, normal_targets, upper_targets, upper_stop_losss, normal_stop_losss,
            normal_target_indexes, normal_sl_indexes, upper_target_indexes, upper_sl_indexes
        )

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

    Grid_Search_Parameter_Optimization(df)

if __name__ == '__main__':
    freeze_support()
    main()