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
from numba import jit, prange, typed, types, set_num_threads
from numba.core import types as nb_types
from numba.typed import Dict, List
import time
from optimized_grid_search import process_combination_batch_hybrid

# Set Numba to use all available cores
import multiprocessing
num_cores = min(multiprocessing.cpu_count(), 16)  # Ensure we don't exceed system limits
set_num_threads(num_cores)
print(f"Setting Numba to use {num_cores} threads")

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))
how_many_final_parameters = None

def Write_Analysis(message):
    # Write to Analysis_Results.txt - overwrite if exists, create if doesn't exist
    with open("Analysis_Results.txt", "a") as file:
        file.write(message)


def Write_Grid_Seach_Results(all_sublists):
    try:
        # Group sublists by volatility
        sorted_sublists_by_volatility = {}
        for sublist_key, sublist_data in all_sublists.items():
            volatility = sublist_key[0]
            if volatility not in sorted_sublists_by_volatility:
                sorted_sublists_by_volatility[volatility] = []
            sorted_sublists_by_volatility[volatility].append((sublist_key, sublist_data))

        # Prepare the message for the analysis file
        message = (
            f"TEST 1: testing all combos of parameters\n"
            f"Results (Top 10 for each volatility level):\n"
            f"volatility, ratio, adx28, 14, 7, abs zscore, rsi_type"
        )
        
        if how_many_final_parameters == 4:
            message += f", t1, t2, sl1, sl2\n"
        elif how_many_final_parameters == 6:
            message += f", t1, t2, t3, sl1, sl2, sl3\n"

        # For each volatility level, get the top 10 results and add them to the message
        for vol in sorted(sorted_sublists_by_volatility.keys()):
            if (vol == 0.0):
                upper_bound = 0.6
            elif (vol == 0.6):
                upper_bound = 3.0
            message += f"\n--- Top 10 results for Volatility between {vol} and {upper_bound} (exclusive) ---\n"
            
            # Get top 10 items for the current volatility, sorted by sum
            top_items = heapq.nlargest(10, sorted_sublists_by_volatility.get(vol, []), key=lambda x: x[1]['sum'])

            if not top_items:
                message += "No results found for this volatility level.\n"
                continue
            
            # Format and append each result
            for i, (key, sub_list) in enumerate(top_items, 1):
                count = sub_list['wins'] + sub_list['losses'] + sub_list['neither']
                message += f"{i}) id: {key}, sum: {round(sub_list['sum'], 1)}, count: {count}, wins: {sub_list['wins']}, losses: {sub_list['losses']}, neither: {sub_list['neither']}\n"
        
        message = message.replace("'", '').replace("{", '').replace("}", '')

        Write_Analysis(message)
        print("\nCOMPLETE\n")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def prune_sublists(local_sublists, keep_count_per_volatility=10):
    """Prune local_sublists to keep only the top entries by sum for each volatility level"""
    try:
        # Group sublists by volatility (first element of the key tuple)
        volatility_groups = {}
        for sublist_key, sublist_data in local_sublists.items():
            volatility = sublist_key[0]  # volatility is the first element in the tuple
            if volatility not in volatility_groups:
                volatility_groups[volatility] = []
            volatility_groups[volatility].append((sublist_key, sublist_data))
        
        # Keep top entries for each volatility level
        pruned_sublists = {}
        for volatility, sublists_for_vol in volatility_groups.items():
            # Use heapq.nlargest for faster top-N selection without full sorting
            top_items = heapq.nlargest(keep_count_per_volatility, sublists_for_vol, key=lambda x: x[1]['sum'])
            for sublist_key, sublist_data in top_items:
                pruned_sublists[sublist_key] = sublist_data
        
        return pruned_sublists
        
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


# data_holder = [(index of original df, [list of values], last price), ...]
def Create_Entries(volatilities, data_holder, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, t1_indexes, 
                   t2_indexes, t3_indexes, sl1_indexes, sl2_indexes, sl3_indexes):
    try:
        (data_rows, data_values, data_last_prices, 
         t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
         t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays) = Convert_To_Numba_Arrays(
            data_holder, t1_indexes, t2_indexes, t3_indexes,
            sl1_indexes, sl2_indexes, sl3_indexes,
            target_1s, target_2s, target_3s, 
            stop_loss_1s, stop_loss_2s, stop_loss_3s)
        
        # targeting 3 things like this creates a huge issue. I need 1 mask for each test because they all filter to different data rows
        local_sublists = {}

        total_combinations_tested = 0
        batch_size = 2000000  # Much larger batch size for better parallelization
        prune_size = 400000   # only allow _sublists to reach this length before pruning them
        batch_count = 0       # how many batches have been processed - used to update user with progress
        
        # Collect all combinations into batches
        combination_batch = []
        
        # time needs a for loop, but if that combo excluding time has already been tested 
        for volatility in volatilities:
            if (volatility < 0.6):
                lower_bound = 0.0
                upper_bound = 0.6
            else:
                lower_bound = 0.6
                upper_bound = 3.0

            vol_mask = (data_values[:, 0] >= lower_bound) & (data_values[:, 0] < upper_bound)
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
                                    for t1 in target_1s:
                                        for sl1 in stop_loss_1s:
                                            for t2 in target_2s:
                                                #if (t2 <= t1 or t2 >= t3): # 6
                                                if (t2 <= t1):
                                                    continue
                                            
                                                for sl2 in stop_loss_2s:
                                                    #if (sl2 >= t3) or (sl2 >= t1 - 0.1):  # 6
                                                    if (sl2 >= t1 - 0.1):
                                                        continue
                                                
                                                    ''' 6
                                                    for t3 in target_3s:
                                                        if (t3 <= t1):
                                                            continue

                                                        for sl3 in stop_loss_3s:
                                                            if (sl3 <= sl1 or sl3 <= sl2 or sl3 >= t2 - 0.1):
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
                                                                't1': t1,
                                                                'sl1': sl1,
                                                                't3': t3,
                                                                'sl2': sl2,
                                                                't2': t2,
                                                                'sl3': sl3
                                                            })
                                                        '''
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
                                                        't1': t1,
                                                        'sl1': sl1,
                                                        'sl2': sl2,
                                                        't2': t2,
                                                        't3': None,
                                                        'sl3': None
                                                    })
                                                    
                                                    # Process batch when it reaches the target size
                                                    if len(combination_batch) >= batch_size:
                                                        local_sublists = process_combination_batch(
                                                            combination_batch, local_sublists,
                                                            t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
                                                            t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                                                            sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                                                            how_many_final_parameters
                                                        )
                                                        total_combinations += len(combination_batch)
                                                        combination_batch = []  # Reset for next batch
                                                        batch_count += 1
                                                        
                                                        # Prune when necessary to manage memory
                                                        if len(local_sublists) >= prune_size:
                                                            local_sublists = prune_sublists(local_sublists, keep_count_per_volatility=10)
                                                        
                                                        if (batch_count % 5 == 0):
                                                            print(f"in progress, completed {batch_count} batches of {batch_size}...")

        # Process any remaining combinations in the final batch
        if combination_batch:
            local_sublists = process_combination_batch(
                combination_batch, local_sublists,
                t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
                t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                how_many_final_parameters
            )
            total_combinations += len(combination_batch)

        print(f"Processed {total_combinations} valid combinations")
        if len(local_sublists) > 90:  # 9 volatility levels * 10 per level
            local_sublists = prune_sublists(local_sublists, keep_count_per_volatility=10)
                                                    
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return {}


"""batch computation calls this, it preps the data and calls the optimized processing function"""
def process_combination_batch(combination_batch, local_sublists, t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
                              t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                              how_many_final_parameters):
    try:
        # Prepare arrays for optimized numba processing
        batch_size = len(combination_batch)
        all_filtered_rows = []
        all_filtered_prices = []
        all_params = []
        
        for i, combo in enumerate(combination_batch):
            all_filtered_rows.append(combo['filtered_rows'])     # all the rows need for each combination
            all_filtered_prices.append(combo['filtered_prices'])  # all final prices of each row for each combination
            all_params.append((combo['t1'], combo['sl1'], 
                               combo['t3'], combo['sl2'],
                               combo['t2'], combo['sl3']))
        
        # Process all combinations with optimized numba (O(1) parameter lookups)
        sums, wins, losses, neithers, valid_mask = process_batch_numba_large(
            all_filtered_rows, all_filtered_prices, all_params,
            t1_map, t1_idx_arrays, sl1_map, sl1_idx_arrays,
            t3_map, t3_idx_arrays, sl2_map, sl2_idx_arrays,
            t2_map, t2_idx_arrays, sl3_map, sl3_idx_arrays
        )
        
        # Add valid results to local_sublists
        for i in range(batch_size):
            if valid_mask[i]:
                combo = combination_batch[i]
                if (how_many_final_parameters == 4):
                    sublist_key = (
                        combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], 
                        combo['adx7'], combo['zscore'], combo['rsi_type'], combo['t1'], 
                        combo['t2'], combo['sl1'], combo['sl2']
                    )
                    
                elif (how_many_final_parameters == 6):
                    sublist_key = (
                        combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], 
                        combo['adx7'], combo['zscore'], combo['rsi_type'], combo['t1'], 
                        combo['t2'], combo['t3'], combo['sl1'], combo['sl2'], combo['sl3']
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


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  
def process_combinations_parallel_fixed(all_filtered_rows_flat, all_filtered_prices_flat, 
                                       combination_starts, combination_lengths,
                                       all_params, param_indices, t1_idx_arrays,
                                       sl1_idx_arrays, t3_idx_arrays, sl2_idx_arrays,
                                       t2_idx_arrays, sl3_idx_arrays, how_many_final_parameters):
    """
    Fixed Numba function with pre-computed parameter indices (O(1) lookups)
    """
    n_combos = len(all_params)
    sums = np.zeros(n_combos, dtype=np.float64)
    wins = np.zeros(n_combos, dtype=np.int32)
    losses = np.zeros(n_combos, dtype=np.int32)
    neithers = np.zeros(n_combos, dtype=np.int32)
    valid_mask = np.ones(n_combos, dtype=np.bool_)

    # Process combinations in parallel using prange
    for i in prange(n_combos):
        # Get the parameters for the current combination
        t1, sl1, t3, sl2, t2, sl3 = all_params[i]
        
        # Get the data for this combination
        start_idx = combination_starts[i]
        length = combination_lengths[i]
        
        # Use pre-computed parameter indices (O(1) instead of O(n))
        t1_map_idx = param_indices[i, 0]
        sl1_map_idx = param_indices[i, 1] 
        sl2_map_idx = param_indices[i, 3]
        t2_map_idx = param_indices[i, 4]
        
        # Skip if any index is invalid
        if (t1_map_idx == -1 or sl1_map_idx == -1 or sl2_map_idx == -1 or t2_map_idx == -1):
            valid_mask[i] = False
            continue
        
        # Select the correct pre-computed index arrays
        t1_idx_array = t1_idx_arrays[t1_map_idx]
        sl1_idx_array = sl1_idx_arrays[sl1_map_idx]
        sl2_idx_array = sl2_idx_arrays[t1_map_idx, sl2_map_idx]
        t2_idx_array = t2_idx_arrays[t1_map_idx, t2_map_idx]

        # do this separetly to save some compute when it's not 6
        if (how_many_final_parameters == 6):
            t3_map_idx = param_indices[i, 2]
            sl3_map_idx = param_indices[i, 5]
            if (t3_map_idx == -1 or sl3_map_idx == -1):
                valid_mask[i] = False
                continue
            t3_idx_array = t3_idx_arrays[t1_map_idx, t3_map_idx]
            sl3_idx_array = sl3_idx_arrays[t1_map_idx, sl3_map_idx]
                    
        # Initialize counters for this combination
        current_sum = 0.0
        current_wins = 0
        current_losses = 0
        current_neither = 0
        
        # Process each row for this combination
        for j in range(length):
            row_idx = all_filtered_rows_flat[start_idx + j]
            final_price = all_filtered_prices_flat[start_idx + j]
            
            t1_idx = t1_idx_array[row_idx]
            sl1_idx = sl1_idx_array[row_idx]
            
            if how_many_final_parameters == 4:
                # sl1 hit first
                if sl1_idx < t1_idx:
                    current_sum += sl1
                    current_losses += 1
                # t1 hit first
                elif t1_idx < sl1_idx:
                    # move to 2nd params
                    t2_idx = t2_idx_array[row_idx]
                    sl2_idx = sl2_idx_array[row_idx]
                    
                    # sl2 hit first
                    if sl2_idx < t2_idx:
                        current_sum += sl2
                        current_losses += 1
                    # t2 hit first
                    elif t2_idx < sl2_idx:
                        current_sum += t2
                        current_wins += 1
                    # neither hit
                    else:
                        current_sum += final_price
                        current_neither += 1
                # neither hit
                else:
                    current_sum += final_price
                    current_neither += 1
                    
            elif how_many_final_parameters == 6:
                # sl1 hit first
                if sl1_idx < t1_idx:
                    current_sum += sl1
                    current_losses += 1
                # t1 hit first
                elif t1_idx < sl1_idx:
                    # move to 2nd params
                    t2_idx = t2_idx_array[row_idx]
                    sl2_idx = sl2_idx_array[row_idx]
                    
                    # sl2 hit first
                    if sl2_idx < t2_idx:
                        current_sum += sl2
                        current_losses += 1
                    # t2 hit first
                    elif t2_idx < sl2_idx:
                        # move to 3rd params
                        t3_idx = t3_idx_array[row_idx]
                        sl3_idx = sl3_idx_array[row_idx]
                        
                        # sl3 hit first
                        if sl3_idx < t3_idx:
                            current_sum += sl3
                            current_wins += 1
                        # t3 hit first
                        elif t3_idx < sl3_idx:
                            current_sum += t3
                            current_wins += 1
                        # neither hit
                        else:
                            current_sum += final_price
                            current_neither += 1
                    # neither hit
                    else:
                        current_sum += final_price
                        current_neither += 1
                # neither hit
                else:
                    current_sum += final_price
                    current_neither += 1
        
        # Store results for this combination
        sums[i] = current_sum
        wins[i] = current_wins
        losses[i] = current_losses
        neithers[i] = current_neither
    
    return sums, wins, losses, neithers, valid_mask


def precompute_parameter_indices(all_params, t1_map, sl1_map, t3_map, sl2_map, t2_map, sl3_map):
    """Pre-compute parameter indices to eliminate O(n) searches"""
    n_combos = len(all_params)
    param_indices = np.full((n_combos, 6), -1, dtype=np.int32)
    
    # Create lookup dictionaries
    t1_lookup = {val: i for i, val in enumerate(t1_map)}
    sl1_lookup = {val: i for i, val in enumerate(sl1_map)}
    t3_lookup = {val: i for i, val in enumerate(t3_map)}
    sl2_lookup = {val: i for i, val in enumerate(sl2_map)}
    t2_lookup = {val: i for i, val in enumerate(t2_map)}
    sl3_lookup = {val: i for i, val in enumerate(sl3_map)}
    
    for i, (t1, sl1, t3, sl2, t2, sl3) in enumerate(all_params):
        param_indices[i, 0] = t1_lookup.get(t1, -1)
        param_indices[i, 1] = sl1_lookup.get(sl1, -1)
        param_indices[i, 2] = t3_lookup.get(t3, -1)
        param_indices[i, 3] = sl2_lookup.get(sl2, -1)
        param_indices[i, 4] = t2_lookup.get(t2, -1)
        param_indices[i, 5] = sl3_lookup.get(sl3, -1)
    
    return param_indices


def process_batch_numba_large(all_filtered_rows, all_filtered_prices, all_params, t1_map_arr, t1_idx_arrays,
                              sl1_map_arr, sl1_idx_arrays, t3_map_arr, t3_idx_arrays, sl2_map_arr, sl2_idx_arrays,
                              t2_map_arr, t2_idx_arrays, sl3_map_arr, sl3_idx_arrays):
    try:
        """
        Processes a large batch of parameter combinations using optimized Numba with parallel processing.
        """
        n_combos = len(all_params)
        
        # Pre-compute parameter indices to eliminate O(n) searches
        param_indices = precompute_parameter_indices(all_params, t1_map_arr, sl1_map_arr, 
                                                    t3_map_arr, sl2_map_arr, t2_map_arr, sl3_map_arr)
        
        # Flatten the data for Numba compatibility
        all_filtered_rows_flat = []
        all_filtered_prices_flat = []
        combination_starts = np.zeros(n_combos, dtype=np.int32)
        combination_lengths = np.zeros(n_combos, dtype=np.int32)
        
        current_start = 0
        for i in range(n_combos):
            filtered_rows = all_filtered_rows[i]
            filtered_prices = all_filtered_prices[i]
            
            combination_starts[i] = current_start
            combination_lengths[i] = len(filtered_rows)
            
            all_filtered_rows_flat.extend(filtered_rows)
            all_filtered_prices_flat.extend(filtered_prices)
            
            current_start += len(filtered_rows)
        
        # Convert to numpy arrays for Numba
        all_filtered_rows_flat = np.array(all_filtered_rows_flat, dtype=np.int32)
        all_filtered_prices_flat = np.array(all_filtered_prices_flat, dtype=np.float64)
        all_params_array = np.array(all_params, dtype=np.float64)
        
        sums, wins, losses, neithers, valid_mask = process_combinations_parallel_fixed(
            all_filtered_rows_flat, all_filtered_prices_flat,
            combination_starts, combination_lengths,
            all_params_array, param_indices, t1_idx_arrays,
            sl1_idx_arrays, t3_idx_arrays, sl2_idx_arrays,
            t2_idx_arrays, sl3_idx_arrays, how_many_final_parameters
        )
        
        return sums, wins, losses, neithers, valid_mask
    
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

        return data_holder, short_rows_data_holder, t1_indexes, t2_indexes, t3_indexes, sl1_indexes, sl2_indexes, sl3_indexes
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Grid_Search_Parameter_Optimization(df):
    try:
        global how_many_final_parameters

        how_many_final_parameters = 4
        
        # 20 minutes on 147 rows of data
        
        entry_times = np.array(['6:30:00', '7:00:00', '7:30:00'], dtype=np.float64) 
        volatilities = np.array([0.0,0.2,0.3,0.4,0.5,0.6,0.7,0.8], dtype=np.float64)
        ratios = np.array([0.0,0.2,0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
        adx28s = np.array([0,20, 30, 40], dtype=np.float64)
        adx14s = np.array([0,20, 30, 40], dtype=np.float64)
        adx7s = np.array([0,20, 30, 40], dtype=np.float64)
        abs_macd_zScores = np.array([0,0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)   # absolute value of z-score, not normal z-score
        extreme_rsis = [True, False, "either"]  # Keep as list for string handling

        target_1s = np.array([0.2,0.3, 0.4, 0.5], dtype=np.float64)
        target_2s = np.array([0.3,0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)      # must be higher than nt and less than ut
        target_3s = np.array([0], dtype=np.float64)
        stop_loss_1s = np.array([-0.5,-0.4,-0.3], dtype=np.float64)
        stop_loss_2s = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3], dtype=np.float64)
        stop_loss_3s = np.array([0], dtype=np.float64) # must be higher than nsl and HIGHER than usl
        '''
        volatilities = np.array([0.0,0.6], dtype=np.float64)
        ratios = np.array([0.0,0.2,0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float64)
        adx28s = np.array([0], dtype=np.float64)
        adx14s = np.array([0], dtype=np.float64)
        adx7s = np.array([0], dtype=np.float64)
        abs_macd_zScores = np.array([0], dtype=np.float64)   # absolute value of z-score, not normal z-score
        extreme_rsis = ["either"]  # Keep as list for string handling

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
        
        print(f"Pre-processing data done. Now processing {len(data_holder)} rows of data (excluding really short time period trades)")
        print("Running grid search...")
        
        start_time = time.time()
        all_sublists = Create_Entries(
            volatilities, data_holder, ratios, adx28s, adx14s, adx7s, abs_macd_zScores,
            extreme_rsis, target_1s, target_3s, stop_loss_2s, stop_loss_1s,
            target_2s, stop_loss_3s, t1_indexes, t2_indexes, t3_indexes, 
            sl1_indexes, sl2_indexes, sl3_indexes
        )

        time_diff_seconds = time.time() - start_time
        minutes = int(time_diff_seconds // 60)
        seconds = int(time_diff_seconds % 60)
        print(f"Total processing time: {minutes} minutes, {seconds} seconds")

        # Ensure the text file "Analysis_Results.txt" exists (create if it doesn't)
        if not os.path.exists("Analysis_Results.txt"):
            with open("Analysis_Results.txt", "w") as f_create:
                pass
        # erase the text file if it exists
        else:
            with open("Analysis_Results.txt", "w") as f:
                pass

        print("Writing results...")
        # Write results only once at the end
        Write_Grid_Seach_Results(all_sublists)

        

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def main():
    data_dir = "Csv_Files/3_Final_Trade_Csvs"
    data_file = "Bulk_Combined.csv"
    df = pd.read_csv(f"{data_dir}/{data_file}")

    Grid_Search_Parameter_Optimization(df)

if __name__ == '__main__':
    freeze_support()
    main()