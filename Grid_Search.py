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


@jit(nopython=True, parallel=True)
def process_batch_numba(
    filtered_rows, filtered_prices,
    inner_combos,
    nt_map_arr, nt_idx_arrays,
    nsl_map_arr, nsl_idx_arrays,
    ut_map_arr, ut_idx_arrays,
    usl_map_arr, usl_idx_arrays
):
    """
    Processes a whole batch of inner-loop combinations in parallel using Numba.
    """
    n_combos = len(inner_combos)
    # Create arrays to hold the results for each combination in the batch
    sums = np.zeros(n_combos, dtype=np.float64)
    wins = np.zeros(n_combos, dtype=np.int32)
    losses = np.zeros(n_combos, dtype=np.int32)
    neithers = np.zeros(n_combos, dtype=np.int32)
    # A flag to mark which combinations are valid (not pruned)
    valid_mask = np.ones(n_combos, dtype=np.bool_)

    # Process all combinations in the batch in parallel
    for i in prange(n_combos):
        # Get the parameters for the current combination
        nt, nsl, ut, usl = inner_combos[i]

        # --- Map float parameters to integer indices ---
        nt_map_idx = np.where(nt_map_arr == nt)[0][0]
        nsl_map_idx = np.where(nsl_map_arr == nsl)[0][0]
        ut_map_idx = np.where(ut_map_arr == ut)[0][0]
        usl_map_idx = np.where(usl_map_arr == usl)[0][0]
        
        # --- Select the correct pre-computed index array ---
        nt_idx_array = nt_idx_arrays[nt_map_idx]
        nsl_idx_array = nsl_idx_arrays[nsl_map_idx]
        ut_idx_array = ut_idx_arrays[nt_map_idx][ut_map_idx]
        usl_idx_array = usl_idx_arrays[nt_map_idx][usl_map_idx]
        
        # --- Run the simulation for this single combination ---
        current_sum = 0.0
        current_wins = 0
        current_losses = 0
        current_neither = 0

        total_rows = len(filtered_rows)
        sixty_percent_mark = int(total_rows * 0.6)
        eighty_percent_mark = int(total_rows * 0.8)

        for j in range(total_rows):
            row_idx = filtered_rows[j]

            # Early exit pruning for this combination
            if j >= sixty_percent_mark and current_sum < 4:
                valid_mask[i] = False
                break
            if j >= eighty_percent_mark and current_sum < 7:
                valid_mask[i] = False
                break

            normal_target_idx = nt_idx_array[row_idx]
            normal_sl_idx = nsl_idx_array[row_idx]

            if normal_sl_idx < normal_target_idx:
                current_sum += nsl
                current_losses += 1
            elif normal_target_idx < normal_sl_idx:
                upper_target_idx = ut_idx_array[row_idx]
                upper_sl_idx = usl_idx_array[row_idx]
                if upper_target_idx < upper_sl_idx:
                    current_sum += ut
                    current_wins += 1
                elif upper_sl_idx < upper_target_idx:
                    current_sum += usl
                    current_losses += 1
                else: # Neither upper condition met
                    current_sum += filtered_prices[j]
                    current_neither += 1
            else: # Neither normal condition met
                current_sum += filtered_prices[j]
                current_neither += 1
        
        # Store results for this combination
        if valid_mask[i]:
            sums[i] = current_sum
            wins[i] = current_wins
            losses[i] = current_losses
            neithers[i] = current_neither

    return sums, wins, losses, neithers, valid_mask


def Convert_To_Numba_Arrays(data_holder, normal_target_indexes, normal_sl_indexes, upper_target_indexes, 
                            upper_sl_indexes, normal_targets, upper_targets, normal_stop_losss, upper_stop_losss):
    
    data_rows = np.array([row[0] for row in data_holder], dtype=np.int32)
    data_values = np.array([row[1] for row in data_holder], dtype=np.float64)
    data_last_prices = np.array([row[2] for row in data_holder], dtype=np.float64)
    
    max_row_idx = np.max(data_rows) + 1 if len(data_rows) > 0 else 1

    # --- Create mapping arrays and tiered index arrays for Numba ---
    # these basically store the rows that each parameter point to, it's for faster lookup
    nt_map = np.array(normal_targets, dtype=np.float64)
    nsl_map = np.array(normal_stop_losss, dtype=np.float64)
    ut_map = np.array(upper_targets, dtype=np.float64)
    usl_map = np.array(upper_stop_losss, dtype=np.float64)

    # Create arrays to hold all the index arrays
    nt_idx_arrays = np.zeros((len(nt_map), max_row_idx), dtype=np.int32)
    nsl_idx_arrays = np.zeros((len(nsl_map), max_row_idx), dtype=np.int32)
    ut_idx_arrays = np.zeros((len(nt_map), len(ut_map), max_row_idx), dtype=np.int32)
    usl_idx_arrays = np.zeros((len(nt_map), len(usl_map), max_row_idx), dtype=np.int32)

    for i, nt in enumerate(nt_map):
        arr = np.full(max_row_idx, 50000, dtype=np.int32)
        if nt in normal_target_indexes:
            for idx, val in normal_target_indexes[nt].items():
                if idx < max_row_idx: arr[idx] = val
        nt_idx_arrays[i] = arr

    for i, nsl in enumerate(nsl_map):
        arr = np.full(max_row_idx, 50000, dtype=np.int32)
        if nsl in normal_sl_indexes:
            for idx, val in normal_sl_indexes[nsl].items():
                if idx < max_row_idx: arr[idx] = val
        nsl_idx_arrays[i] = arr

    for i, nt in enumerate(nt_map):
        for j, ut in enumerate(ut_map):
            arr = np.full(max_row_idx, 50000, dtype=np.int32)
            if nt in upper_target_indexes and ut in upper_target_indexes[nt]:
                for idx, val in upper_target_indexes[nt][ut].items():
                    if idx < max_row_idx: arr[idx] = val
            ut_idx_arrays[i, j] = arr
            
        for j, usl in enumerate(usl_map):
            arr = np.full(max_row_idx, 50000, dtype=np.int32)
            if nt in upper_sl_indexes and usl in upper_sl_indexes[nt]:
                for idx, val in upper_sl_indexes[nt][usl].items():
                    if idx < max_row_idx: arr[idx] = val
            usl_idx_arrays[i, j] = arr

    return (data_rows, data_values, data_last_prices,
            nt_map, nt_idx_arrays, nsl_map, nsl_idx_arrays,
            ut_map, ut_idx_arrays, usl_map, usl_idx_arrays)


# data_holder = [(index of original df, [list of values], last price), ...]
def Create_Entries(volatilities, data_holder, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis, normal_targets, 
                   upper_targets, upper_stop_losss, normal_stop_losss, normal_target_indexes, normal_sl_indexes,
                   upper_target_indexes, upper_sl_indexes):
    try:
        (data_rows, data_values, 
         data_last_prices, nt_map, 
         nt_idx_arrays, nsl_map, 
         nsl_idx_arrays, ut_map, 
         ut_idx_arrays, usl_map, 
         usl_idx_arrays) = Convert_To_Numba_Arrays(
            data_holder, normal_target_indexes, 
            normal_sl_indexes, upper_target_indexes, 
            upper_sl_indexes, normal_targets,
            upper_targets, normal_stop_losss, 
            upper_stop_losss)
        
        local_sublists = {}
        total_combinations = 0
        batch_size = 200000
        
        # Collect all combinations into batches
        combination_batch = []
        
        for volatility in volatilities:
            vol_mask = data_values[:, 0] >= volatility
            for ratio in ratios:
                ratio_mask = data_values[:, 1] >= ratio
                for adx28 in adx28s:
                    adx28_mask = data_values[:, 2] >= adx28
                    for adx14 in adx14s:
                        adx14_mask = data_values[:, 3] >= adx14
                        for adx7 in adx7s:
                            adx7_mask = data_values[:, 4] >= adx7
                            for zscore in abs_macd_zScores:
                                zscore_mask = np.abs(data_values[:, 5]) >= zscore
                                for rsi_type in extreme_rsis:
                                    # Combine all filters
                                    combined_mask = (vol_mask & ratio_mask & adx28_mask & adx14_mask & adx7_mask & zscore_mask)
                                    
                                    if rsi_type != "either":
                                        rsi_val = 1.0 if rsi_type else 0.0
                                        rsi_mask = data_values[:, 6] == rsi_val
                                        combined_mask &= rsi_mask

                                    # Get the final filtered data for this batch
                                    filtered_indices = np.where(combined_mask)[0]
                                    if len(filtered_indices) == 0:
                                        continue

                                    filtered_rows = data_rows[filtered_indices]
                                    filtered_prices = data_last_prices[filtered_indices]
                                    
                                    # Add combinations for each inner parameter set
                                    for normal_target in normal_targets:
                                        for normal_stop_loss in normal_stop_losss:
                                            for upper_target in upper_targets:
                                                if (upper_target <= normal_target):
                                                    continue
                                                for upper_stop_loss in upper_stop_losss:
                                                    if (upper_stop_loss >= upper_target) or (upper_stop_loss >= normal_target):
                                                        continue
                                                    
                                                    # Store the complete combination
                                                    combination_batch.append({
                                                        'filtered_rows': filtered_rows,
                                                        'filtered_prices': filtered_prices,
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
                                                    
                                                    # Process batch when it reaches the target size
                                                    if len(combination_batch) >= batch_size:
                                                        local_sublists = process_combination_batch(
                                                            combination_batch, local_sublists,
                                                            nt_map, nt_idx_arrays, nsl_map, nsl_idx_arrays,
                                                            ut_map, ut_idx_arrays, usl_map, usl_idx_arrays
                                                        )
                                                        total_combinations += len(combination_batch)
                                                        combination_batch = []  # Reset for next batch
                                                        
                                                        # Prune when necessary to manage memory
                                                        if len(local_sublists) > 400000:
                                                            local_sublists = prune_sublists(local_sublists, keep_count=50)

        # Process any remaining combinations in the final batch
        if combination_batch:
            local_sublists = process_combination_batch(
                combination_batch, local_sublists,
                nt_map, nt_idx_arrays, nsl_map, nsl_idx_arrays,
                ut_map, ut_idx_arrays, usl_map, usl_idx_arrays
            )
            total_combinations += len(combination_batch)

        print(f"Processed {total_combinations} valid combinations")
        if len(local_sublists) > 50:
            local_sublists = prune_sublists(local_sublists, keep_count=50)
                                                    
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return {}


"""batch computation calls this, it preps the data and calls process_batch_numba_large"""
def process_combination_batch(combination_batch, local_sublists, nt_map, nt_idx_arrays, nsl_map, nsl_idx_arrays,
                             ut_map, ut_idx_arrays, usl_map, usl_idx_arrays):
    try:
        # Prepare arrays for numba processing
        batch_size = len(combination_batch)
        all_filtered_rows = []
        all_filtered_prices = []
        all_params = []
        all_metadata = []
        
        for i, combo in enumerate(combination_batch):
            all_filtered_rows.append(combo['filtered_rows'])     # all the rows need for each combination
            all_filtered_prices.append(combo['filtered_prices'])  # all final prices of each row for each combination
            all_params.append((combo['normal_target'], combo['normal_stop_loss'], 
                             combo['upper_target'], combo['upper_stop_loss']))
            all_metadata.append((combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], 
                               combo['adx7'], combo['zscore'], combo['rsi_type']))
        
        # Process all combinations with numba
        sums, wins, losses, neithers, valid_mask = process_batch_numba_large(
            all_filtered_rows, all_filtered_prices, all_params,
            nt_map, nt_idx_arrays, nsl_map, nsl_idx_arrays,
            ut_map, ut_idx_arrays, usl_map, usl_idx_arrays
        )
        
        # Add valid results to local_sublists
        for i in range(batch_size):
            if valid_mask[i]:
                combo = combination_batch[i]
                sublist_key = (
                    combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], 
                    combo['adx7'], combo['zscore'], combo['rsi_type'], combo['normal_target'], 
                    combo['upper_target'], combo['normal_stop_loss'], combo['upper_stop_loss']
                )
                local_sublists[sublist_key] = {
                    'sum': sums[i],
                    'wins': wins[i],
                    'losses': losses[i],
                    'neither': neithers[i]
                }
        
        return local_sublists
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return local_sublists


def process_batch_numba_large(all_filtered_rows, all_filtered_prices, all_params, nt_map_arr, nt_idx_arrays,
                              nsl_map_arr, nsl_idx_arrays, ut_map_arr, ut_idx_arrays, usl_map_arr, usl_idx_arrays):
    try:
        """
        Processes a large batch of parameter combinations using Numba (without @jit due to complex data structures).
        """
        n_combos = len(all_params)
        # Create arrays to hold the results for each combination in the batch
        sums = np.zeros(n_combos, dtype=np.float64)
        wins = np.zeros(n_combos, dtype=np.int32)
        losses = np.zeros(n_combos, dtype=np.int32)
        neithers = np.zeros(n_combos, dtype=np.int32)
        # A flag to mark which combinations are valid (not pruned)
        valid_mask = np.ones(n_combos, dtype=np.bool_)

        # Process each combination
        for i in range(n_combos):
            # Get the parameters for the current combination
            nt, nsl, ut, usl = all_params[i]
            filtered_rows = all_filtered_rows[i]
            filtered_prices = all_filtered_prices[i]

            # --- Map float parameters to integer indices ---
            nt_map_idx = np.where(nt_map_arr == nt)[0][0]
            nsl_map_idx = np.where(nsl_map_arr == nsl)[0][0]
            ut_map_idx = np.where(ut_map_arr == ut)[0][0]
            usl_map_idx = np.where(usl_map_arr == usl)[0][0]
            
            # --- Select the correct pre-computed index array ---
            nt_idx_array = nt_idx_arrays[nt_map_idx]
            nsl_idx_array = nsl_idx_arrays[nsl_map_idx]
            ut_idx_array = ut_idx_arrays[nt_map_idx][ut_map_idx]
            usl_idx_array = usl_idx_arrays[nt_map_idx][usl_map_idx]
            
            # --- Run for this single combination ---
            current_sum = 0.0
            current_wins = 0
            current_losses = 0
            current_neither = 0

            total_rows = len(filtered_rows)
            sixty_percent_mark = int(total_rows * 0.6)
            eighty_percent_mark = int(total_rows * 0.8)

            for j in range(total_rows):
                row_idx = filtered_rows[j]

                # Early exit pruning for this combination
                #if j >= sixty_percent_mark and current_sum < 4:
                #    valid_mask[i] = False
                #    break
                #if j >= eighty_percent_mark and current_sum < 7:
                #    valid_mask[i] = False
                #    break

                normal_target_idx = nt_idx_array[row_idx]
                normal_sl_idx = nsl_idx_array[row_idx]

                if normal_sl_idx < normal_target_idx:
                    current_sum += nsl
                    current_losses += 1
                elif normal_target_idx < normal_sl_idx:
                    upper_target_idx = ut_idx_array[row_idx]
                    upper_sl_idx = usl_idx_array[row_idx]
                    if upper_target_idx < upper_sl_idx:
                        current_sum += ut
                        current_wins += 1
                    elif upper_sl_idx < upper_target_idx:
                        current_sum += usl
                        current_losses += 1
                    else: # Neither upper condition met
                        current_sum += filtered_prices[j]
                        current_neither += 1
                else: # Neither normal condition met
                    current_sum += filtered_prices[j]
                    current_neither += 1
            
            # Store results for this combination
            if valid_mask[i]:
                sums[i] = current_sum
                wins[i] = current_wins
                losses[i] = current_losses
                neithers[i] = current_neither

        return sums, wins, losses, neithers, valid_mask
    
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
        volatilities = np.array([0.4,0.8,0.7,0.6,0.5,0.9,0.3], dtype=np.float64)
        ratios = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1], dtype=np.float64)
        adx28s = np.array([20, 30, 40, 50, 60], dtype=np.float64)
        adx14s = np.array([20, 30, 40, 50, 60], dtype=np.float64)
        adx7s = np.array([20, 30, 40, 50, 60], dtype=np.float64)
        abs_macd_zScores = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float64)   # absolute value of z-score, not normal z-score
        extreme_rsis = [True, False, "either"]  # Keep as list for string handling
        normal_targets = np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float64)
        upper_targets = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
        upper_stop_losss = np.array([0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3], dtype=np.float64)
        normal_stop_losss = np.array([-0.3, -0.4, -0.5], dtype=np.float64)
        '''
        volatilities = np.array([0.6], dtype=np.float64)
        ratios = np.array([1.1], dtype=np.float64)
        adx28s = np.array([20], dtype=np.float64)
        adx14s = np.array([20], dtype=np.float64)
        adx7s = np.array([20], dtype=np.float64)
        abs_macd_zScores = np.array([1.0], dtype=np.float64)   # absolute value of z-score, not normal z-score
        extreme_rsis = [True, False, "either"]  # Keep as list for string handling
        normal_targets = np.array([0.3, 0.4, 0.5], dtype=np.float64)
        upper_targets = np.array([0.6, 0.7, 0.8, 0.9], dtype=np.float64)
        upper_stop_losss = np.array([-0.1], dtype=np.float64)
        normal_stop_losss = np.array([-0.5], dtype=np.float64)
        '''

        data_holder, short_rows_data_holder, normal_target_indexes, normal_sl_indexes, upper_target_indexes, upper_sl_indexes = Create_2D_List_From_Df(
            df, normal_targets.tolist(), normal_stop_losss.tolist(), 
            upper_targets.tolist(), upper_stop_losss.tolist()
        )
        
        print(f"Processing {len(data_holder)} rows of data (excluding really short time period trades)")
        print("Running grid search...")
        
        all_sublists = Create_Entries(
            volatilities, data_holder, ratios, adx28s, adx14s, adx7s, abs_macd_zScores,
            extreme_rsis, normal_targets, upper_targets, 
            upper_stop_losss, normal_stop_losss,
            normal_target_indexes, normal_sl_indexes, upper_target_indexes, upper_sl_indexes
        )

        print("Writing results...")
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