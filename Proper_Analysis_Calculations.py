import pandas as pd
import numpy as np
import os
import inspect
import sys
import shutil
import Main_Globals
from datetime import datetime
from multiprocessing import freeze_support
from numba import jit, prange, types
from numba.typed import Dict
import itertools
import time

# NUMBA THREAD CONTROL: Uncomment to limit CPU usage
# os.environ['NUMBA_NUM_THREADS'] = '4'  # Use only 4 threads instead of all cores
# os.environ['OMP_NUM_THREADS'] = '4'    # Limit OpenMP threads too

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


'''
GRID SEARCH VECTORIZATION


FIRST: THIS IS SUPER INEFFIECIENT - HOWEREVER IT'S TOO HARD FOR ME TO CHANGE RIGHT NOW
part of a prompt: "Regarding the process in Grid_Search_Parameter_Optimization() which calls process_volatility_chunk():
I think there's a huge efficiency oversight. currently it processes every index of volailities by finding every 
combination of the other lists for that volaility and running those combinations of parameters over a data set. 
However, for each volatility it's looking at data in the data set where the volatility is greater than or equal to 
that volatility. this means for example that if it processes volatility = 0.6 then it processes volatility = 0.5, 
the 0.5 volatility will process the same combinations as the 0.6 volatility because it's looking for all volatilities 
greater than or equal to 0.5 (which includes 0.6). this is hugly inefficient
First, does that explaination make sense and do you think I'm correct?
..."
basically I can't do it because I prune the sublists in batches. those batches are huge but ultimatly I still delete data
that could be used to skip future loops. I'd have to redesign how combos are created so they calculate similar combos together
so they can use each others data. 


1) each process stores all combos in a list then finds their vector mask.
masks do df filtering. masks are normally bool lists, but it's doing numpy stuff. it ends up getting the an array of price movement data for each combo (filtered_data = get_filtered_data_vectorized(df_vol, combined_mask))

2) using vectors, finds index of all targets/stops for that combo. -1 means it's not in the list
	upper results contains upper target and upper sl data and flags for 	if they were found

3) it does all trades for that combo at once by comparing the result indicies. basically it does case 1: neither sl nor target appear, then sl appears, then normal target appears... in the end it makes a list of profits, wins (bool), losses (bool), neither (bool). this makes the local sublists which are the sublists I want. they have the tuple key and the releveant info (some not yet populated). however it's not pruned so volatility 0.9 for example is 4032 lists
error: it looks like all the ccaluations are wrong (sum, wins...)

4) profits, wins, loss, winrate all seem to be calcualted correct. correct rows are looked at

Ram savings: 
   sublists is calcualtes are pruned every x length into the top 50. 
   it only calculates then processes x combinations at a time, then does the next batch

numba: it's using parallel=True which means 1 worker (thread) maxes out my cpu. prange() makes multiple threads within that process
'''

# NUMBA JIT OPTIMIZATION: Core computation functions compiled for maximum speed
@jit(nopython=True, parallel=True, fastmath=True)
def find_target_indices_numba(padded_array, actual_lengths, target_value):
    """
    NUMBA OPTIMIZED: Ultra-fast compiled version using parallel execution.
    No Python overhead, direct memory access, parallel processing.
    """
    num_rows = padded_array.shape[0]
    result_indices = np.full(num_rows, -1, dtype=np.int32)
    
    for i in prange(num_rows):  # Parallel execution across CPU cores
        length = actual_lengths[i]
        if length > 0:
            for j in range(length):
                if padded_array[i, j] == target_value:
                    result_indices[i] = j
                    break
    
    return result_indices


@jit(nopython=True, parallel=True, fastmath=True)
def find_stop_loss_indices_numba(padded_array, actual_lengths, stop_loss_value, target_indices):
    """
    NUMBA OPTIMIZED: Ultra-fast compiled version.
    """
    num_rows = padded_array.shape[0]
    result_indices = np.full(num_rows, -1, dtype=np.int32)
    
    for i in prange(num_rows):  # Parallel execution
        length = actual_lengths[i]
        target_idx = target_indices[i]
        search_end = target_idx if target_idx != -1 else length
        
        if search_end > 0:
            for j in range(search_end):
                if padded_array[i, j] == stop_loss_value:
                    result_indices[i] = j
                    break
    
    return result_indices


@jit(nopython=True, parallel=True, fastmath=True)
def find_upper_target_stop_indices_numba(padded_array, actual_lengths, normal_target_indices, upper_target, upper_stop_loss):
    """
    NUMBA OPTIMIZED: Ultra-fast compiled version for upper target/stop finding.
    """
    num_rows = padded_array.shape[0]
    upper_target_indices = np.full(num_rows, -1, dtype=np.int32)
    upper_stop_loss_indices = np.full(num_rows, -1, dtype=np.int32)
    found_upper_flags = np.zeros(num_rows, dtype=np.bool_)
    
    for i in prange(num_rows):  # Parallel execution
        normal_target_idx = normal_target_indices[i]
        if normal_target_idx == -1:
            continue
            
        length = actual_lengths[i]
        search_start = normal_target_idx + 1
        
        if search_start < length:
            upper_target_found = False
            upper_stop_found = False
            upper_target_pos = -1
            upper_stop_pos = -1
            
            # Find both targets in one pass
            for j in range(search_start, length):
                if not upper_target_found and padded_array[i, j] == upper_target:
                    upper_target_found = True
                    upper_target_pos = j
                if not upper_stop_found and padded_array[i, j] == upper_stop_loss:
                    upper_stop_found = True
                    upper_stop_pos = j
                
                # Break early if both found
                if upper_target_found and upper_stop_found:
                    break
            
            # Set results based on which was found first
            if upper_target_found and upper_stop_found:
                if upper_target_pos <= upper_stop_pos:
                    upper_target_indices[i] = upper_target_pos
                    found_upper_flags[i] = True
                else:
                    upper_stop_loss_indices[i] = upper_stop_pos
                    found_upper_flags[i] = True
            elif upper_target_found:
                upper_target_indices[i] = upper_target_pos
                found_upper_flags[i] = True
            elif upper_stop_found:
                upper_stop_loss_indices[i] = upper_stop_pos
                found_upper_flags[i] = True
    
    return upper_target_indices, upper_stop_loss_indices, found_upper_flags


@jit(nopython=True, parallel=True, fastmath=True)
def calculate_trade_outcomes_numba(padded_array, actual_lengths, normal_target_indices, normal_sl_indices, 
                                 upper_target_indices, upper_stop_loss_indices, found_upper_flags,
                                 normal_target, normal_stop_loss, upper_target, upper_stop_loss):
    """
    NUMBA OPTIMIZED: Ultra-fast compiled version for trade outcome calculation.
    Eliminates all Python loops and overhead.
    """
    num_trades = padded_array.shape[0]
    profits = np.zeros(num_trades, dtype=np.float64)
    win_flags = np.zeros(num_trades, dtype=np.bool_)
    loss_flags = np.zeros(num_trades, dtype=np.bool_)
    neither_flags = np.zeros(num_trades, dtype=np.bool_)
    
    for i in prange(num_trades):  # Parallel execution
        normal_target_idx = normal_target_indices[i]
        normal_sl_idx = normal_sl_indices[i]
        length = actual_lengths[i]
        
        if length == 0:
            continue
        
        # Case 1: Neither normal target nor normal stop loss appear
        if normal_target_idx == -1 and normal_sl_idx == -1:
            profits[i] = padded_array[i, length - 1]
            neither_flags[i] = True
        
        # Case 2: Normal stop loss appears before normal target
        elif normal_sl_idx != -1 and (normal_target_idx == -1 or normal_sl_idx < normal_target_idx):
            profits[i] = normal_stop_loss
            loss_flags[i] = True
        
        # Case 3: Normal target appears first
        elif normal_target_idx != -1:
            # Sub-case 3a: Normal target is the last value
            if normal_target_idx == length - 1:
                profits[i] = normal_target
                win_flags[i] = True
            
            # Sub-case 3b: Look for upper targets/stops after normal target
            elif found_upper_flags[i]:
                upper_target_idx = upper_target_indices[i]
                upper_stop_idx = upper_stop_loss_indices[i]
                
                if upper_target_idx != -1:
                    profits[i] = upper_target
                    win_flags[i] = True
                elif upper_stop_idx != -1:
                    profits[i] = upper_stop_loss
                    loss_flags[i] = True
            
            # Sub-case 3c: No upper target/stop found, use final value
            else:
                profits[i] = padded_array[i, length - 1]
                neither_flags[i] = True
    
    return profits, win_flags, loss_flags, neither_flags


@jit(nopython=True)
def aggregate_results_numba(profits, win_flags, loss_flags, neither_flags):
    """
    NUMBA OPTIMIZED: Ultra-fast aggregation using compiled code.
    """
    total_profit = np.sum(profits)
    total_count = len(profits)
    win_count = np.sum(win_flags)
    loss_count = np.sum(loss_flags)
    neither_count = np.sum(neither_flags)
    
    winrate = win_count / total_count if total_count > 0 else 0.0
    
    return total_profit, total_count, win_count, loss_count, neither_count, winrate


def process_combination_batch_optimized(valid_combinations, batch_num, vol_ratio_vals, adx28_vals, 
                                      adx14_vals, adx7_vals, macd_zscore_vals, rsi_vals_numeric,
                                      vol_indices, padded_price_movements, actual_lengths,
                                      local_sublists, volatility):
    """
    OPTIMIZED BATCH PROCESSOR: Combines NUMBA speed with proper RAM management.
    Processes batches without using dictionaries in NUMBA functions.
    """
    try:
        if len(valid_combinations) == 0:
            return 0
        
        # this needs to be every 500 batches instead of every batch
        #print(f"Processing batch {batch_num} with {len(valid_combinations):,} combinations using NUMBA...")
        
        # Convert batch to arrays for NUMBA processing
        batch_arrays = list(zip(*valid_combinations))  # Transpose
        ratio_batch = np.array(batch_arrays[0], dtype=np.float64)
        adx28_batch = np.array(batch_arrays[1], dtype=np.float64)
        adx14_batch = np.array(batch_arrays[2], dtype=np.float64)
        adx7_batch = np.array(batch_arrays[3], dtype=np.float64)
        zscore_batch = np.array(batch_arrays[4], dtype=np.float64)
        rsi_batch = np.array(batch_arrays[5], dtype=np.int32)
        normal_target_batch = np.array(batch_arrays[6], dtype=np.float64)
        normal_stop_loss_batch = np.array(batch_arrays[7], dtype=np.float64)
        upper_target_batch = np.array(batch_arrays[8], dtype=np.float64)
        upper_stop_loss_batch = np.array(batch_arrays[9], dtype=np.float64)
        
        # Process entire batch with NUMBA (compiled code)
        batch_results = process_combination_batch_numba(
            vol_ratio_vals, adx28_vals, adx14_vals, adx7_vals, macd_zscore_vals, rsi_vals_numeric,
            vol_indices, padded_price_movements, actual_lengths,
            ratio_batch, adx28_batch, adx14_batch, adx7_batch, zscore_batch, rsi_batch,
            normal_target_batch, normal_stop_loss_batch, upper_target_batch, upper_stop_loss_batch
        )
        
        # Convert results back to dictionary format (this part uses Python, not NUMBA)
        results_added = 0
        for i, combo in enumerate(valid_combinations):
            if batch_results[i, 1] > 0:  # Only store if we have data (count > 0)
                # Convert RSI back to original format for the key
                rsi_orig = True if combo[5] == 1 else (False if combo[5] == 0 else "either")
                
                sublist_key = (volatility, combo[0], combo[1], combo[2], combo[3], 
                              combo[4], rsi_orig, combo[6], combo[8], combo[7], combo[9])
                
                local_sublists[sublist_key] = {
                    'id': sublist_key,
                    'sum': round(batch_results[i, 0], 2),
                    'count': int(batch_results[i, 1]),
                    'wins': int(batch_results[i, 2]),
                    'losses': int(batch_results[i, 3]),
                    'neither': int(batch_results[i, 4]),
                    'winrate': round(batch_results[i, 5], 2)
                }
                results_added += 1
        
        # this needs to be every 500 batches instead of every batch
        #print(f"Batch {batch_num} completed - {results_added} valid results found")
        return len(valid_combinations)
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return 0


# NUMBA-OPTIMIZED BATCH PROCESSING: Dictionary-free for maximum speed
@jit(nopython=True, parallel=True, fastmath=True)  
def process_combination_batch_numba(vol_ratio_vals, adx28_vals, adx14_vals, adx7_vals, macd_zscore_vals, rsi_vals,
                                   vol_indices, padded_price_movements, actual_lengths,
                                   ratio_batch, adx28_batch, adx14_batch, adx7_batch, zscore_batch, rsi_batch,
                                   normal_target_batch, normal_stop_loss_batch, upper_target_batch, upper_stop_loss_batch):
    """
    NUMBA OPTIMIZED: Process entire batches of combinations in compiled code.
    Returns numpy arrays instead of dictionaries for NUMBA compatibility.
    """
    batch_size = len(ratio_batch)
    num_data_points = len(vol_ratio_vals)
    
    # Pre-allocate results for entire batch
    batch_results = np.zeros((batch_size, 6), dtype=np.float64)  # [sum, count, wins, losses, neither, winrate]
    
    for batch_idx in prange(batch_size):  # Parallel batch processing
        ratio = ratio_batch[batch_idx]
        adx28 = adx28_batch[batch_idx]
        adx14 = adx14_batch[batch_idx]
        adx7 = adx7_batch[batch_idx]
        zscore = zscore_batch[batch_idx]
        rsi_type = rsi_batch[batch_idx]
        normal_target = normal_target_batch[batch_idx]
        normal_stop_loss = normal_stop_loss_batch[batch_idx]
        upper_target = upper_target_batch[batch_idx]
        upper_stop_loss = upper_stop_loss_batch[batch_idx]
        
        # Create combined mask efficiently using numba-compatible code
        mask = ((vol_ratio_vals >= ratio) & 
                (adx28_vals >= adx28) & 
                (adx14_vals >= adx14) & 
                (adx7_vals >= adx7) & 
                (np.abs(macd_zscore_vals) >= zscore) &
                ((rsi_type == 2) | (rsi_vals == rsi_type)))  # 2 = "either"
        
        # Count matches for pre-allocation
        num_matches = np.sum(mask)
        if num_matches == 0:
            continue
        
        # Extract matching indices efficiently
        filtered_indices_array = vol_indices[mask]
        
        # Extract price movements for this combination
        filtered_padded = padded_price_movements[filtered_indices_array]
        filtered_lengths = actual_lengths[filtered_indices_array]
        
        # Find indices using optimized functions
        normal_target_indices = find_target_indices_numba(filtered_padded, filtered_lengths, normal_target)
        normal_sl_indices = find_stop_loss_indices_numba(filtered_padded, filtered_lengths, normal_stop_loss, normal_target_indices)
        upper_target_indices, upper_stop_loss_indices, found_upper_flags = find_upper_target_stop_indices_numba(
            filtered_padded, filtered_lengths, normal_target_indices, upper_target, upper_stop_loss)
        
        # Calculate trade outcomes
        profits, win_flags, loss_flags, neither_flags = calculate_trade_outcomes_numba(
            filtered_padded, filtered_lengths, normal_target_indices, normal_sl_indices,
            upper_target_indices, upper_stop_loss_indices, found_upper_flags,
            normal_target, normal_stop_loss, upper_target, upper_stop_loss)
        
        # Aggregate results
        total_profit, total_count, win_count, loss_count, neither_count, winrate = aggregate_results_numba(
            profits, win_flags, loss_flags, neither_flags)
        
        # Store batch results
        batch_results[batch_idx, 0] = total_profit
        batch_results[batch_idx, 1] = total_count
        batch_results[batch_idx, 2] = win_count
        batch_results[batch_idx, 3] = loss_count
        batch_results[batch_idx, 4] = neither_count
        batch_results[batch_idx, 5] = winrate
    
    return batch_results


def Write_Analysis(message):
    # Write to Analysis_Results.txt - overwrite if exists, create if doesn't exist
    with open("Analysis_Results.txt", "a") as file:
        file.write(message)


def Write_Grid_Seach_Results(all_sublists):
    best_sublists_sum = {}
    best_sublists_winrate = {}
    num_to_write = 50
    
    # find top 10 sublists by sum
    sum_sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['sum'], reverse=True)
    for i in range(min(num_to_write, len(sum_sorted_items))):
        key, sublist = sum_sorted_items[i]
        sublist_rounded = sublist.copy()
        sublist_rounded['sum'] = round(sublist_rounded['sum'], 2)
        best_sublists_sum[key] = sublist_rounded

    # find top 10 sublists by winrate
    winrate_sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['winrate'], reverse=True)
    for i in range(min(num_to_write, len(winrate_sorted_items))):
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


# OLD FUNCTION REMOVED: create_combined_filter_mask()
# This has been replaced by pre-computed mask caching optimization
# The old approach recalculated the same masks millions of times


def get_filtered_data_vectorized(df, combined_mask):
    """
    Applies the combined filter mask to extract relevant data for calculations.
    Returns only the necessary columns as numpy arrays for maximum speed.
    
    Args:
        df: Original DataFrame
        combined_mask: Boolean mask from create_combined_filter_mask()
    
    Returns:
        dict containing filtered numpy arrays of the data we need
    """
    try:
        if not np.any(combined_mask):
            # No rows pass the filter - return empty arrays
            return {
                'price_movement_lists': np.array([], dtype=object),
                'count': 0,
                'indices': np.array([], dtype=int)
            }
        
        # Apply mask and extract only what we need for calculations
        filtered_price_movements = df.loc[combined_mask, 'price_movement_list'].values
        filtered_indices = np.where(combined_mask)[0]  # Store original indices for debugging
        
        return {
            'price_movement_lists': filtered_price_movements,
            'count': len(filtered_price_movements),
            'indices': filtered_indices
        }
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return {'price_movement_lists': np.array([], dtype=object), 'count': 0, 'indices': np.array([], dtype=int)}


# VECTORIZED OPTIMIZATION - CHUNK 3: Profit/Loss Calculation
# These functions replace the individual row processing with bulk numpy operations
# Processing all trades simultaneously using vectorized logic and boolean indexing
def calculate_trade_outcomes_vectorized(price_movement_lists, normal_target_indices, normal_sl_indices, 
                                      upper_results, normal_target, normal_stop_loss, upper_target, upper_stop_loss):
    """
    Vectorized calculation of trade outcomes for all price movements simultaneously.
    Replaces the individual row processing loop with bulk numpy operations.
    
    Args:
        price_movement_lists: numpy array of price movement lists
        normal_target_indices: numpy array of normal target indices (-1 if not found)
        normal_sl_indices: numpy array of normal stop loss indices (-1 if not found)
        upper_results: dict from find_upper_target_stop_indices()
        normal_target, normal_stop_loss, upper_target, upper_stop_loss: target/stop values
    
    Returns:
        dict with vectorized results: 'profits', 'win_flags', 'loss_flags', 'neither_flags'
    """
    try:
        if len(price_movement_lists) == 0:
            return {
                'profits': np.array([]),
                'win_flags': np.array([], dtype=bool),
                'loss_flags': np.array([], dtype=bool),
                'neither_flags': np.array([], dtype=bool)
            }
        
        num_trades = len(price_movement_lists)
        
        # Pre-allocate result arrays
        profits = np.zeros(num_trades, dtype=float)
        win_flags = np.zeros(num_trades, dtype=bool)
        loss_flags = np.zeros(num_trades, dtype=bool)
        neither_flags = np.zeros(num_trades, dtype=bool)
        
        # Extract upper target results
        upper_target_indices = upper_results['upper_target_indices']
        upper_stop_loss_indices = upper_results['upper_stop_loss_indices']
        found_upper_flags = upper_results['found_upper_flags']
        
        # Vectorized condition checking using boolean indexing
        
        # Case 1: Neither normal target nor normal stop loss appear
        # Both indices are -1
        case1_mask = (normal_target_indices == -1) & (normal_sl_indices == -1)
        case1_indices = np.where(case1_mask)[0]
        
        for i in case1_indices:
            pm_list = price_movement_lists[i]
            if len(pm_list) > 0:
                profits[i] = pm_list[-1]  # Use final value
                neither_flags[i] = True
        
        # Case 2: Normal stop loss appears before normal target (or target doesn't exist)
        # normal_sl_indices != -1 AND (normal_target_indices == -1 OR normal_sl_indices < normal_target_indices)
        case2_mask = (normal_sl_indices != -1) & ((normal_target_indices == -1) | (normal_sl_indices < normal_target_indices))
        case2_indices = np.where(case2_mask)[0]
        
        # Vectorized assignment for case 2
        profits[case2_indices] = normal_stop_loss
        loss_flags[case2_indices] = True
        
        # Case 3: Normal target appears first (normal_target_indices != -1 and case 2 doesn't apply)
        case3_mask = (normal_target_indices != -1) & ~case2_mask & ~case1_mask
        case3_indices = np.where(case3_mask)[0]
        
        for i in case3_indices:
            pm_list = price_movement_lists[i]
            normal_target_idx = normal_target_indices[i]
            
            # Sub-case 3a: Normal target is the last value
            if normal_target_idx == len(pm_list) - 1:
                profits[i] = normal_target
                win_flags[i] = True
            
            # Sub-case 3b: Look for upper targets/stops after normal target
            elif found_upper_flags[i]:
                # Check which upper event occurred first
                upper_target_idx = upper_target_indices[i]
                upper_stop_idx = upper_stop_loss_indices[i]
                
                if upper_target_idx != -1:
                    # Upper target found
                    profits[i] = upper_target
                    win_flags[i] = True
                elif upper_stop_idx != -1:
                    # Upper stop loss found
                    profits[i] = upper_stop_loss
                    loss_flags[i] = True
            
            # Sub-case 3c: No upper target/stop found, use final value
            else:
                profits[i] = pm_list[-1]
                neither_flags[i] = True
        
        if (num_trades > 1):
            pass
        return {
            'profits': profits,
            'win_flags': win_flags,
            'loss_flags': loss_flags,
            'neither_flags': neither_flags
        }
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return {
            'profits': np.array([]),
            'win_flags': np.array([], dtype=bool),
            'loss_flags': np.array([], dtype=bool),
            'neither_flags': np.array([], dtype=bool)
        }


def aggregate_results_vectorized(trade_outcomes, sublist_key):
    """
    Vectorized aggregation of trade results using numpy operations.
    Replaces manual counting loops with vectorized sum operations.
    
    Args:
        trade_outcomes: dict from calculate_trade_outcomes_vectorized()
        sublist_key: tuple key for this parameter combination
    
    Returns:
        dict with aggregated results ready for final output
    """
    try:
        profits = trade_outcomes['profits']
        win_flags = trade_outcomes['win_flags']
        loss_flags = trade_outcomes['loss_flags']
        neither_flags = trade_outcomes['neither_flags']
        
        # Vectorized aggregation using numpy operations
        total_profit = np.sum(profits)  # Much faster than manual loop
        total_count = len(profits)
        win_count = np.sum(win_flags)    # Boolean True counts as 1
        loss_count = np.sum(loss_flags)
        neither_count = np.sum(neither_flags)
        
        # Calculate win rate
        winrate = round(win_count / total_count, 2) if total_count > 0 else 0
        
        return {
            'id': sublist_key,
            'sum': round(total_profit, 2),
            'count': total_count,
            'wins': win_count,
            'losses': loss_count,
            'neither': neither_count,
            'winrate': winrate
        }
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return {
            'id': sublist_key,
            'sum': 0,
            'count': 0,
            'wins': 0,
            'losses': 0,
            'neither': 0,
            'winrate': 0
        }


# NUMBA OPTIMIZED VERSION: Ultra-fast processing using compiled code
def Process_Volatility_Chunk(volatility, df, padded_price_movements, actual_lengths, max_length, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis, normal_targets, upper_targets, upper_stop_losss, normal_stop_losss):
    """
    NUMBA OPTIMIZATION: Uses compiled code for 17.8M combinations.
    Expected 10-100x speed improvement over previous version.
    """
    try:
        local_sublists = {}
        
        # Pre-filter by volatility once
        vol_mask = df['Entry Volatility Percent'].values >= volatility
        df_vol = df[vol_mask]
        
        if len(df_vol) == 0:
            return local_sublists
        
        # Get the indices for the pre-converted padded array
        vol_indices = df_vol['padded_price_movements_index'].values
        
        # Extract column data once for NUMBA processing
        vol_ratio_vals = df_vol['Entry Volatility Ratio'].values.astype(np.float64)
        adx28_vals = df_vol['Entry Adx28'].values.astype(np.float64)
        adx14_vals = df_vol['Entry Adx14'].values.astype(np.float64)
        adx7_vals = df_vol['Entry Adx7'].values.astype(np.float64)
        macd_zscore_vals = df_vol['Entry Macd Z-Score'].values.astype(np.float64)
        
        # Convert RSI values to numeric for NUMBA (True=1, False=0, "either"=2)
        rsi_vals_numeric = np.zeros(len(df_vol), dtype=np.int32)
        rsi_vals_orig = df_vol['Rsi Extreme Prev Cross'].values
        for i, val in enumerate(rsi_vals_orig):
            if val is True:
                rsi_vals_numeric[i] = 1
            elif val is False:
                rsi_vals_numeric[i] = 0
            else:  # "either" case
                rsi_vals_numeric[i] = 2
        
        # RESTORED BATCH PROCESSING: Generate and process combinations in batches to prevent RAM overload
        print(f"Processing combinations in batches to prevent RAM overload...")
        
        batch_size = 300000  # Process x combinations at a time to save RAM
        valid_combinations = []
        total_processed = 0
        batch_num = 0
        
        for ratio in ratios:
            for adx28 in adx28s:
                for adx14 in adx14s:
                    for adx7 in adx7s:
                        for zscore in abs_macd_zScores:
                            for rsi_type in extreme_rsis:
                                # Convert RSI type to numeric for NUMBA
                                rsi_numeric = 1 if rsi_type is True else (0 if rsi_type is False else 2)
                                
                                for normal_target in normal_targets:
                                    for normal_stop_loss in normal_stop_losss:
                                        for upper_target in upper_targets:
                                            if upper_target <= normal_target:
                                                continue
                                            
                                            for upper_stop_loss in upper_stop_losss:
                                                if ((upper_stop_loss >= upper_target) or upper_stop_loss >= normal_target):
                                                    continue
                                                
                                                # Add valid combination to current batch
                                                valid_combinations.append((
                                                    ratio, adx28, adx14, adx7, zscore, rsi_numeric,
                                                    normal_target, normal_stop_loss, upper_target, upper_stop_loss
                                                ))
                                                
                                                # Process batch when it reaches batch_size to prevent RAM overload
                                                if len(valid_combinations) >= batch_size:
                                                    batch_num += 1
                                                    total_processed += process_combination_batch_optimized(
                                                        valid_combinations, batch_num, vol_ratio_vals, adx28_vals, 
                                                        adx14_vals, adx7_vals, macd_zscore_vals, rsi_vals_numeric,
                                                        vol_indices, padded_price_movements, actual_lengths,
                                                        local_sublists, volatility
                                                    )
                                                    # Clear the batch to free memory (critical for RAM management)
                                                    valid_combinations.clear()
                                                    
                                                    # Periodic pruning to prevent local_sublists from growing too large
                                                    if len(local_sublists) > 100000:
                                                        local_sublists = prune_sublists_efficient(local_sublists, keep_top_n=50)
                                                        #print(f"Pruned local_sublists to top 100 entries") # this should be much less frequent if at all
        
        # Process any remaining combinations in the final batch
        if valid_combinations:
            batch_num += 1
            total_processed += process_combination_batch_optimized(
                valid_combinations, batch_num, vol_ratio_vals, adx28_vals, 
                adx14_vals, adx7_vals, macd_zscore_vals, rsi_vals_numeric,
                vol_indices, padded_price_movements, actual_lengths,
                local_sublists, volatility
            )
            valid_combinations.clear()
        
        print(f"Total combinations processed: {total_processed:,}")
        
        # Final pruning
        local_sublists = prune_sublists_efficient(local_sublists, keep_top_n=50)
        
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return {}


# OLD FUNCTION REMOVED: Replaced by NUMBA optimized batch processing


# MEMORY OPTIMIZATION: Efficient sublist pruning to prevent RAM overload
def prune_sublists_efficient(local_sublists, keep_top_n=50):
    """
    Efficiently prunes local_sublists to keep only top N by sum and top N by winrate.
    Uses numpy argpartition for O(n) partial sorting instead of full O(n log n) sorting.
    
    Args:
        local_sublists: dict of sublists to prune
        keep_top_n: number of top entries to keep for each metric (default 50)
    
    Returns:
        pruned dict with at most 2*keep_top_n entries (may have duplicates)
    """
    try:
        if len(local_sublists) <= 2 * keep_top_n:
            return local_sublists  # No pruning needed
        
        # Extract keys and values for efficient numpy operations
        keys = list(local_sublists.keys())
        sums = np.array([local_sublists[key]['sum'] for key in keys])
        winrates = np.array([local_sublists[key]['winrate'] for key in keys])
        
        # Use argpartition for O(n) partial sorting - much faster than full sort
        # Get indices of top keep_top_n entries by sum (largest values)
        top_sum_indices = np.argpartition(sums, -keep_top_n)[-keep_top_n:]
        
        # Get indices of top keep_top_n entries by winrate (largest values)  
        top_winrate_indices = np.argpartition(winrates, -keep_top_n)[-keep_top_n:]
        
        # Combine indices (may have duplicates, which is fine per requirements)
        keep_indices = np.concatenate([top_sum_indices, top_winrate_indices])
        
        # Rebuild dictionary with only the top entries
        pruned_sublists = {}
        for idx in keep_indices:
            key = keys[idx]
            pruned_sublists[key] = local_sublists[key]
        
        return pruned_sublists
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return local_sublists  # Return original on error
    

# VECTORIZED OPTIMIZATION - INTEGRATION & TESTING
# New optimized grid search function and performance comparison utilities
def Grid_Search_Parameter_Optimization(df):
    """
    Uses numpy operations throughout for dramatic performance improvements.
    
    Key performance improvements:
    - Vectorized DataFrame filtering using boolean masks
    - Bulk numpy operations for index finding  
    - Vectorized profit/loss calculations
    - Reduced memory allocations and DataFrame copying
    """
    try:
        volatilities = [0.9,0.8,0.7,0.6,0.5,0.4,0.3] # KEEP IN DESCENDING ORDER
        ratios = [0.5,0.6,0.7,0.8,0.9,1.0,1.1]
        adx28s = [20,30,40,50,60]
        adx14s = [20,30,40,50,60]
        adx7s = [20,30,40,50,60]
        abs_macd_zScores = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]   
        extreme_rsis = [True, False, "either"]
        normal_targets = [0.2,0.3,0.4,0.5,0.6]
        upper_targets = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        upper_stop_losss = [0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5]
        normal_stop_losss = [-0.3,-0.4,-0.5,-0.6]
        
        # test lists
        '''
        volatilities = [0.5,0.6,0.7]
        ratios = [0.9,1.0,1.1]
        adx28s = [0,10,20,30]
        adx14s = [0,10,20,30]
        adx7s = [0,10,20,30]
        abs_macd_zScores = [1.5,2.0,2.5] 
        extreme_rsis = [True, False]
        normal_targets = [0.4,0.5,0.6]
        upper_targets = [0.5,0.6,0.7,0.8,0.9]
        upper_stop_losss = [-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
        normal_stop_losss = [-0.6,-0.7,-0.8,-0.9]
        '''
        
        # CRITICAL OPTIMIZATION: Pre-convert ALL price movements to 2D numpy array ONCE        
        # First convert to Python lists
        price_movement_lists = df['Price Movement'].apply(
            lambda x: [float(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else []
        ).tolist()
        
        # Find maximum length across all price movements
        max_length = max(len(pm_list) for pm_list in price_movement_lists if len(pm_list) > 0)
        print(f"Maximum price movement length: {max_length}")
        
        # Create single 2D padded array for ALL price movements (this happens only once!)
        padded_price_movements = np.full((len(price_movement_lists), max_length), np.nan, dtype=float)
        actual_lengths = np.zeros(len(price_movement_lists), dtype=int)
        
        # Fill the padded array once and store lengths
        for i, pm_list in enumerate(price_movement_lists):
            if len(pm_list) > 0:
                padded_price_movements[i, :len(pm_list)] = pm_list
                actual_lengths[i] = len(pm_list)
        
        # Store pre-converted arrays in DataFrame for easy filtering
        df['padded_price_movements_index'] = range(len(df))  # Index to map to padded array

        all_sublists = {}
        start_time = datetime.now()
        
        print(f"Starting NUMBA-optimized grid search for {len(volatilities)} volatility levels...")
        
        # Process each volatility level sequentially (NUMBA handles internal parallelization)
        for volatility in volatilities:
            print(f"Processing volatility {volatility}...")
            local_sublists = Process_Volatility_Chunk(
                volatility, df, padded_price_movements, actual_lengths, max_length,
                ratios, adx28s, adx14s, adx7s, abs_macd_zScores,
                extreme_rsis, normal_targets, upper_targets, upper_stop_losss, normal_stop_losss
            )
            # Merge local results into main dictionary
            all_sublists.update(local_sublists)
            elapsed_time = datetime.now() - start_time
            # Format elapsed_time as HH:MM:SS only
            elapsed_time_str = str(elapsed_time).split('.')[0]
            print(f"Completed NUMBA processing for volatility {volatility}. Time elapsed: {elapsed_time_str}\n")
        
        # Write results using existing function
        Write_Grid_Seach_Results(all_sublists)
        total_elapsed_time = datetime.now() - start_time
        total_elapsed_time_str = str(total_elapsed_time).split('.')[0]
        print(f"NUMBA-optimized grid search completed successfully! Total time: {total_elapsed_time_str}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# OLD VECTORIZED FUNCTIONS REMOVED: Replaced by NUMBA-optimized versions above


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

    # Run NUMBA-optimized implementation
    print("=== RUNNING NUMBA-OPTIMIZED GRID SEARCH ===")
    Grid_Search_Parameter_Optimization(df)

if __name__ == '__main__':
    freeze_support()
    main()
