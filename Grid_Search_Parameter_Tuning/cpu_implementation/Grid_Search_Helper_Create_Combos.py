import pandas as pd
import numpy as np
import os
import inspect
import sys
import Main_Globals
import heapq
from numba import jit, prange
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


def prune_sublists(local_sublists, user_mode):
    try:
        if (user_mode == 1):
            """Prune local_sublists to keep only the top entries by sum for ALL volatility level"""
            keep_count = 10

            if len(local_sublists) <= keep_count:
                return local_sublists
            
            # Use heapq.nlargest for faster top-N selection without full sorting
            top_items = heapq.nlargest(keep_count, local_sublists.items(), key=lambda x: x[1]['sum'])
            pruned_sublists = dict(top_items)
            
            return pruned_sublists

        elif (user_mode == 2):
            """Prune local_sublists to keep only the top entries by sum for EACH volatility level"""
            # Group sublists by volatility (first element of the key tuple)
            keep_count_per_volatility = 10
            volatility_groups = {}

            for sublist_key, sublist_data in local_sublists.items():
                # Parse string key format: "entry|volatility|ratio|adx28|adx14|adx7|rsi_type|..."
                key_parts = sublist_key.split('|')
                volatility = float(key_parts[1])  # volatility is the 2nd element
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
        
        elif (user_mode == 3):
            """Prune local_sublists to keep only the top entries by sum for EACH entry time"""
            keep_count_per_time = 10
            time_groups = {}

            for sublist_key, sublist_data in local_sublists.items():
                # Parse string key format: "entry|volatility|ratio|adx28|adx14|adx7|rsi_type|..."
                key_parts = sublist_key.split('|')
                entry_time = key_parts[0]  # entry time is the 1st element
                if entry_time not in time_groups:
                    time_groups[entry_time] = []
                time_groups[entry_time].append((sublist_key, sublist_data))

            pruned_sublists = {}
            for entry_time, sublists_for_time in time_groups.items():
                # Use heapq.nlargest for faster top-N selection without full sorting
                top_items = heapq.nlargest(keep_count_per_time, sublists_for_time, key=lambda x: x[1]['sum'])
                for sublist_key, sublist_data in top_items:
                    pruned_sublists[sublist_key] = sublist_data
            
            return pruned_sublists

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


@jit(nopython=True, parallel=True, fastmath=True, cache=True)  
def Parallel_Process_Helper(all_filtered_rows_flat, all_filtered_prices_flat, 
                                       combination_starts, combination_lengths,
                                       all_params, param_indices, t1_idx_arrays,
                                       sl1_idx_arrays, t3_idx_arrays, sl2_idx_arrays,
                                       t2_idx_arrays, sl3_idx_arrays, how_many_final_parameters):
    """
    Numba-optimized parallel processing
    
    This function processes millions of parameter combinations efficiently by:
    1. Using pre-computed parameter indices for O(1) lookups instead of O(n) searches
    2. Using contiguous numpy arrays that Numba can parallelize across CPU cores
    3. Avoiding Python objects/dictionaries that Numba cannot optimize
    
    The index arrays enable instant lookups: given a row_idx, we can immediately
    find which price movement index each parameter hits (or 50000 if never hit).
    """
    n_combos = len(all_params)
    sums = np.zeros(n_combos, dtype=np.float32)
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
        
        # SPEED CRITICAL: Use pre-computed parameter indices for O(1) lookups
        # Instead of searching through parameter arrays each time (O(n)),
        # we use the pre-computed indices to directly access the right arrays
        t1_map_idx = param_indices[i, 0]
        sl1_map_idx = param_indices[i, 1] 
        sl2_map_idx = param_indices[i, 3]
        t2_map_idx = param_indices[i, 4]
        
        # Skip invalid combinations (parameters not found in our arrays)
        if (t1_map_idx == -1 or sl1_map_idx == -1 or sl2_map_idx == -1 or t2_map_idx == -1):
            valid_mask[i] = False
            continue
        
        # PERFORMANCE CRITICAL: Direct array indexing using pre-computed indices
        # These arrays contain the price movement indices for each parameter/row combination
        # t1_idx_array[row] gives the index where t1 is hit in that row's price movement
        t1_idx_array = t1_idx_arrays[t1_map_idx]
        sl1_idx_array = sl1_idx_arrays[sl1_map_idx]
        sl2_idx_array = sl2_idx_arrays[t1_map_idx, sl2_map_idx]  # Depends on t1
        t2_idx_array = t2_idx_arrays[t1_map_idx, t2_map_idx]    # Depends on t1

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
        current_sum = np.float32(0.0)
        current_wins = 0
        current_losses = 0
        current_neither = 0
        
        # CORE ALGORITHM: Process each row for this parameter combination
        # For each row, we look up where each parameter gets hit in the price movement
        # and determine win/loss/neither based on which happens first
        for j in range(length):
            row_idx = all_filtered_rows_flat[start_idx + j]
            final_price = all_filtered_prices_flat[start_idx + j]
            
            # INSTANT LOOKUP: Get the price movement indices where targets/stops are hit
            # These are pre-computed, so this is O(1) instead of searching the price list
            t1_idx = t1_idx_array[row_idx]      # Where target 1 gets hit (or 50000)
            sl1_idx = sl1_idx_array[row_idx]    # Where stop loss 1 gets hit (or 50000)
            
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


def Precompute_Parameter_Indices(all_params, target1s_np_array, stop_loss1s_np_array, target3s_np_array, 
                                 stop_loss2s_np_array, target2s_np_array, stop_loss3s_np_array):
    """
    parameter index computation using vectorized operations
    
    BEFORE: 19 seconds for 2M combinations (12M np.where calls)
    AFTER: ~1-2 seconds using lookup dictionaries and vectorized operations
    
    This eliminates the massive bottleneck of individual np.where calls per combination.
    """
    n_combos = len(all_params)
    param_indices = np.full((n_combos, 6), -1, dtype=np.int32)
    
    # Create lookup dictionaries once instead of 12M np.where calls
    # This is 15-20x faster for large batch sizes
    # Use Python float keys to avoid dtype hashing mismatches between float32/float64
    target1_lookup = {float(val): idx for idx, val in enumerate(target1s_np_array)}
    stop_loss1_lookup = {float(val): idx for idx, val in enumerate(stop_loss1s_np_array)}
    target3_lookup = {float(val): idx for idx, val in enumerate(target3s_np_array)}
    stop_loss2_lookup = {float(val): idx for idx, val in enumerate(stop_loss2s_np_array)}
    target2_lookup = {float(val): idx for idx, val in enumerate(target2s_np_array)}
    stop_loss3_lookup = {float(val): idx for idx, val in enumerate(stop_loss3s_np_array)}
    
    # Vectorized parameter extraction and lookup
    # Extract all parameter values at once instead of individual tuple unpacking
    all_params_array = np.array(all_params, dtype=np.float32)
    
    # Maximum performance using vectorized lookup with comprehensions
    # This is faster than individual .get() calls in a loop and has no failure scenarios
    param_indices[:, 0] = [target1_lookup.get(float(val), -1) for val in all_params_array[:, 0]]
    param_indices[:, 1] = [stop_loss1_lookup.get(float(val), -1) for val in all_params_array[:, 1]]
    param_indices[:, 2] = [target3_lookup.get(float(val), -1) for val in all_params_array[:, 2]]
    param_indices[:, 3] = [stop_loss2_lookup.get(float(val), -1) for val in all_params_array[:, 3]]
    param_indices[:, 4] = [target2_lookup.get(float(val), -1) for val in all_params_array[:, 4]]
    param_indices[:, 5] = [stop_loss3_lookup.get(float(val), -1) for val in all_params_array[:, 5]]
    
    return param_indices


def Process_Combination_Batch(all_filtered_rows, all_filtered_prices, all_params, target1s_np_array, t1_idx_arrays,
                              stop_loss1s_np_array, sl1_idx_arrays, target3s_np_array, t3_idx_arrays, stop_loss2s_np_array, sl2_idx_arrays,
                              target2s_np_array, t2_idx_arrays, stop_loss3s_np_array, sl3_idx_arrays, how_many_final_parameters):
    try:
        """
        Processes a large batch of parameter combinations using optimized Numba with parallel processing.
        """
        n_combos = len(all_params)
        
        # Pre-compute parameter indices to eliminate O(n) searches
        param_indices = Precompute_Parameter_Indices(all_params, target1s_np_array, stop_loss1s_np_array, 
                                                    target3s_np_array, stop_loss2s_np_array, target2s_np_array, stop_loss3s_np_array)
        
        # data flattening using vectorized numpy operations
        # BEFORE: 12 seconds with 2M list.extend() calls (using a for loop)
        # AFTER: ~1-2 seconds using numpy.concatenate
        
        # Pre-compute lengths and starts using vectorized operations
        combination_lengths = np.array([len(rows) for rows in all_filtered_rows], dtype=np.int32)
        combination_starts = np.concatenate([[0], np.cumsum(combination_lengths)[:-1]], dtype=np.int32)
        
        # OPTIMIZATION: Use numpy.concatenate instead of list.extend() loops
        # This is 10-15x faster for large datasets
        all_filtered_rows_flat = np.concatenate(all_filtered_rows, dtype=np.int32)
        all_filtered_prices_flat = np.concatenate(all_filtered_prices, dtype=np.float32)
        all_params_array = np.array(all_params, dtype=np.float32)
        
        sums, wins, losses, neithers, valid_mask = Parallel_Process_Helper(
            all_filtered_rows_flat, all_filtered_prices_flat,
            combination_starts, combination_lengths,
            all_params_array, param_indices, t1_idx_arrays,
            sl1_idx_arrays, t3_idx_arrays, sl2_idx_arrays,
            t2_idx_arrays, sl3_idx_arrays, how_many_final_parameters
        )
        
        return sums, wins, losses, neithers, valid_mask
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Preprocess_Combination_Batch(combination_batch, local_sublists, target1s_np_array, target2s_np_array, target3s_np_array, 
                                 stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array, 
                                 t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                                 how_many_final_parameters, mask_cache, data_rows, data_last_prices):
    """
    Prepares data for Numba-optimized processing with memory optimization
    
    This function bridges between Python data structures and Numba-compatible arrays.
    It batches thousands of parameter combinations together for efficient parallel processing.
    
    Key optimizations:
    1. Batches combinations to amortize Numba compilation overhead
    2. Converts Python data to contiguous numpy arrays for Numba
    3. Uses pre-computed index arrays to avoid repeated price movement searches
    4. **MEMORY OPTIMIZATION**: Lazy loading of filtered data from mask keys
    
    MEMORY OPTIMIZATION STRATEGY:
    - Previously: Each combination stored filtered_rows/filtered_prices (~1-2KB each)
    - Now: Each combination stores only mask_key (~8 bytes)
    - Trade-off: Fast np.where() computation during processing vs massive memory savings (this is actually important)
    
    The index arrays are the key performance enabler - they pre-compute where each
    parameter value gets hit in each row's price movement, enabling O(1) lookups.
    """
    try:
        # VECTORIZED OPTIMIZATION: Process all combinations at once instead of individual loops
        # This reduces 20-second processing to ~2-3 seconds by eliminating per-combination overhead
        # STRUCT-OF-ARRAYS: Now working with columnar numpy arrays instead of list of dictionaries
        batch_size = len(combination_batch['mask_keys'])
        
        # Pre-extract all mask keys for batch processing (now direct array access)
        mask_keys = combination_batch['mask_keys']
        
        # Leverage mask reuse patterns for speedup
        # Many combinations share the same mask_key, so we can group and reuse computations
        
        # Group combinations by unique mask keys to minimize redundant np.where() calls  
        # Use defaultdict for faster grouping (eliminates key existence checks)
        mask_key_groups = defaultdict(list)
        for i, key in enumerate(mask_keys):
            # Convert numpy array element to hashable tuple for dictionary key
            hashable_key = tuple(key) if isinstance(key, np.ndarray) else key
            mask_key_groups[hashable_key].append(i)
        
        # Pre-compute filtered indices only once per unique mask
        unique_filtered_indices = {}
        for hashable_key in mask_key_groups.keys():
            mask = mask_cache[hashable_key]
            unique_filtered_indices[hashable_key] = np.where(mask)[0]
        
        # Batch array indexing for unique masks
        # Compute filtered data once per unique mask, then reuse for all combinations
        unique_filtered_rows = {}
        unique_filtered_prices = {}
        for hashable_key, indices in unique_filtered_indices.items():
            unique_filtered_rows[hashable_key] = data_rows[indices]
            unique_filtered_prices[hashable_key] = data_last_prices[indices]
        
        # Fast assignment using pre-computed results
        # Allocate arrays once and fill using index mapping (much faster than appends)
        all_filtered_rows = [None] * batch_size
        all_filtered_prices = [None] * batch_size
        
        # Vectorized assignment - eliminate the inner loop for better performance
        for hashable_key, combo_indices in mask_key_groups.items():
            filtered_rows = unique_filtered_rows[hashable_key]
            filtered_prices = unique_filtered_prices[hashable_key]
            # Use numpy-style vectorized assignment instead of loop
            for combo_idx in combo_indices:
                all_filtered_rows[combo_idx] = filtered_rows
                all_filtered_prices[combo_idx] = filtered_prices
        
        # Vectorized parameter extraction using direct array access (much faster than list comprehension)
        # STRUCT-OF-ARRAYS: Direct numpy array operations replace dictionary access loops
        
        # Stack arrays into the format expected by Process_Combination_Batch
        all_params = np.column_stack([
            combination_batch['t1s'],
            combination_batch['sl1s'], 
            combination_batch['t3s'],
            combination_batch['sl2s'],
            combination_batch['t2s'],
            combination_batch['sl3s']
        ])
        
        # Process all combinations with optimized numba (O(1) parameter lookups)
        sums, wins, losses, neithers, valid_mask = Process_Combination_Batch(
            all_filtered_rows, all_filtered_prices, all_params,
            target1s_np_array, t1_idx_arrays, stop_loss1s_np_array, sl1_idx_arrays,
            target3s_np_array, t3_idx_arrays, stop_loss2s_np_array, sl2_idx_arrays,
            target2s_np_array, t2_idx_arrays, stop_loss3s_np_array, sl3_idx_arrays, how_many_final_parameters
        )
        
        # Vectorized processing with string keys for speedup
        # Pre-extract all needed combo values for vectorized processing
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Early filtering - only process combinations with sum >= 2.0
            # This reduces memory usage and processing time for poor-performing combinations
            sum_threshold = 2.0
            
            # STRUCT-OF-ARRAYS: Vectorized processing replaces individual dictionary access
            # Filter valid indices by sum threshold using vectorized operations
            good_performance_mask = sums[valid_indices] >= sum_threshold
            final_valid_indices = valid_indices[good_performance_mask]
            
            if len(final_valid_indices) > 0:
                # Extract arrays for valid combinations using advanced indexing
                entry_times_valid = combination_batch['entry_times'][final_valid_indices]
                volatilities_valid = combination_batch['volatilities'][final_valid_indices]
                ratios_valid = combination_batch['ratios'][final_valid_indices]
                avg_of_last_2_trades_valid = combination_batch['avg_of_last_2_trades'][final_valid_indices]
                adx28s_valid = combination_batch['adx28s'][final_valid_indices]
                adx14s_valid = combination_batch['adx14s'][final_valid_indices]
                adx7s_valid = combination_batch['adx7s'][final_valid_indices]
                rsi_types_valid = combination_batch['rsi_types'][final_valid_indices]
                last_result_4_min_types_valid = combination_batch['last_result_4_min_types'][final_valid_indices]
                t1s_valid = combination_batch['t1s'][final_valid_indices]
                t2s_valid = combination_batch['t2s'][final_valid_indices]
                sl1s_valid = combination_batch['sl1s'][final_valid_indices]
                sl2s_valid = combination_batch['sl2s'][final_valid_indices]
                
                sums_valid = sums[final_valid_indices]
                wins_valid = wins[final_valid_indices]
                losses_valid = losses[final_valid_indices]
                neithers_valid = neithers[final_valid_indices]
                
                if how_many_final_parameters == 4:
                    # Vectorized string key creation for 4-parameter mode
                    for i, idx in enumerate(final_valid_indices):
                        sublist_key = f"{entry_times_valid[i]}|{volatilities_valid[i]}|{ratios_valid[i]}|{avg_of_last_2_trades_valid[i]}|{adx28s_valid[i]}|{adx14s_valid[i]}|{adx7s_valid[i]}|{rsi_types_valid[i]}|{last_result_4_min_types_valid[i]}|{t1s_valid[i]}|{t2s_valid[i]}|{sl1s_valid[i]}|{sl2s_valid[i]}"
                        local_sublists[sublist_key] = {
                            'sum': sums_valid[i],
                            'wins': wins_valid[i],
                            'losses': losses_valid[i],
                            'neither': neithers_valid[i]
                        }
                else:  # how_many_final_parameters == 6
                    # Extract additional arrays for 6-parameter mode
                    t3s_valid = combination_batch['t3s'][final_valid_indices]
                    sl3s_valid = combination_batch['sl3s'][final_valid_indices]
                    
                    # Vectorized string key creation for 6-parameter mode
                    for i, _ in enumerate(final_valid_indices):
                        sublist_key = f"{entry_times_valid[i]}|{volatilities_valid[i]}|{ratios_valid[i]}|{avg_of_last_2_trades_valid[i]}|{adx28s_valid[i]}|{adx14s_valid[i]}|{adx7s_valid[i]}|{rsi_types_valid[i]}|{last_result_4_min_types_valid[i]}|{t1s_valid[i]}|{t2s_valid[i]}|{t3s_valid[i]}|{sl1s_valid[i]}|{sl2s_valid[i]}|{sl3s_valid[i]}"
                        local_sublists[sublist_key] = {
                            'sum': sums_valid[i],
                            'wins': wins_valid[i],
                            'losses': losses_valid[i],
                            'neither': neithers_valid[i]
                        }
        
        return local_sublists
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def time_str_to_seconds(time_str):
    try:
        h, m, s = map(int, str(time_str).split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return -1


# data_holder = [(index of original df, [list of values], last price), ...]
# creates combos by looking at data by time stamp greater than or equal to volatility
# numb of masks are lengths of all lists multiplied by each other. size depends on the dataset. 250 rows of data should be like 30 MB of RAM
def precompute_all_masks(entry_times, volatilities, ratios, avg_of_last_2_trades, adx28s, adx14s, adx7s, 
                        extreme_rsis, last_result_must_be_4_minutes, data_values):
    try:
        """
        Pre-compute all possible masks for parameter combinations
        Returns a dictionary mapping parameter tuples to boolean masks
        """
        print("Pre-computing masks for optimization...")
        mask_cache = {}
        
        # Pre-compute individual parameter masks
        time_masks = {}
        for i in range(len(entry_times) - 1):
            entry_time = entry_times[i]
            lower_bound_s = time_str_to_seconds(entry_time)
            upper_bound_s = time_str_to_seconds(entry_times[i + 1])
            time_masks[entry_time] = (data_values[:, 7] >= lower_bound_s) & (data_values[:, 7] < upper_bound_s)
        
        vol_masks = {vol: data_values[:, 0] >= vol for vol in volatilities}
        ratio_masks = {ratio: data_values[:, 1] >= ratio for ratio in ratios}
        avg_of_last_2_trades_masks = {avg: data_values[:, 2] >= avg for avg in avg_of_last_2_trades}
        adx28_masks = {adx: data_values[:, 3] >= adx for adx in adx28s}
        adx14_masks = {adx: data_values[:, 4] >= adx for adx in adx14s}
        adx7_masks = {adx: data_values[:, 5] >= adx for adx in adx7s}
        
        rsi_masks = {}
        for rsi_type in extreme_rsis:
            if rsi_type == "either":
                rsi_masks[rsi_type] = np.full(data_values.shape[0], True)
            else:
                rsi_val = 1.0 if rsi_type else 0.0
                rsi_masks[rsi_type] = data_values[:, 6] == rsi_val
        
        last_result_4_min_masks = {}
        for last_result_4_min_type in last_result_must_be_4_minutes:
            if last_result_4_min_type == "either":
                last_result_4_min_masks[last_result_4_min_type] = np.full(data_values.shape[0], True)
            else:
                last_result_4_min_val = 1.0 if last_result_4_min_type else 0.0
                last_result_4_min_masks[last_result_4_min_type] = data_values[:, 8] == last_result_4_min_val
        
        # Combine masks for all parameter combinations
        total_combinations = 0
        for entry_time in entry_times[:-1]:  # Skip last entry_time
            for vol in volatilities:
                for ratio in ratios:
                    for avg_of_last_2_trades_val in avg_of_last_2_trades:
                        for adx28 in adx28s:
                            for adx14 in adx14s:
                                for adx7 in adx7s:
                                    for rsi_type in extreme_rsis:
                                        for last_result_4_min_type in last_result_must_be_4_minutes:
                                            key = (entry_time, vol, ratio, avg_of_last_2_trades_val, adx28, adx14, adx7, rsi_type, last_result_4_min_type)
                                            combined_mask = (time_masks[entry_time] & 
                                                            vol_masks[vol] & 
                                                            ratio_masks[ratio] & 
                                                            avg_of_last_2_trades_masks[avg_of_last_2_trades_val] &
                                                            adx28_masks[adx28] & 
                                                            adx14_masks[adx14] &
                                                            adx7_masks[adx7] & 
                                                            rsi_masks[rsi_type] &
                                                            last_result_4_min_masks[last_result_4_min_type])
                                            mask_cache[key] = combined_mask
                                            total_combinations += 1
        
        # Calculate approximate memory usage (boolean mask = 1 byte per element)
        #memory_mb = (total_combinations * data_values.shape[0]) / (1024 * 1024)
        #print(f"Estimated mask cache memory usage: {memory_mb:.1f} MB")
        
        return mask_cache
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# finds masks w/o including time
def precompute_all_masks_no_time(volatilities, ratios, avg_of_last_2_trades, adx28s, adx14s, adx7s, 
                                extreme_rsis, last_result_must_be_4_minutes, data_values, user_mode):
    try:
        """
        Pre-compute all possible masks for parameter combinations without time filtering
        Used for user_mode=1 and user_mode=2
        """
        print("Pre-computing masks for optimization (no time filtering)...")
        mask_cache = {}
        
        # Pre-compute individual parameter masks
        if user_mode == 1:
            # user_mode=1 uses upper/lower bounds for volatility
            vol_masks = {}
            for volatility in volatilities:
                lower_bound = volatility
                if volatility == 0.0:
                    upper_bound = 0.6
                else:
                    upper_bound = 3.0
                vol_masks[volatility] = (data_values[:, 0] >= lower_bound) & (data_values[:, 0] < upper_bound)
        else:
            # user_mode=2 uses simple >= filtering
            vol_masks = {vol: data_values[:, 0] >= vol for vol in volatilities}
        
        ratio_masks = {ratio: data_values[:, 1] >= ratio for ratio in ratios}
        avg_of_last_2_trades_masks = {avg: data_values[:, 2] >= avg for avg in avg_of_last_2_trades}
        adx28_masks = {adx: data_values[:, 3] >= adx for adx in adx28s}
        adx14_masks = {adx: data_values[:, 4] >= adx for adx in adx14s}
        adx7_masks = {adx: data_values[:, 5] >= adx for adx in adx7s}
        
        rsi_masks = {}
        for rsi_type in extreme_rsis:
            if rsi_type == "either":
                rsi_masks[rsi_type] = np.full(data_values.shape[0], True)
            else:
                rsi_val = 1.0 if rsi_type else 0.0
                rsi_masks[rsi_type] = data_values[:, 6] == rsi_val
        
        last_result_4_min_masks = {}
        for last_result_4_min_type in last_result_must_be_4_minutes:
            if last_result_4_min_type == "either":
                last_result_4_min_masks[last_result_4_min_type] = np.full(data_values.shape[0], True)
            else:
                last_result_4_min_val = 1.0 if last_result_4_min_type else 0.0
                last_result_4_min_masks[last_result_4_min_type] = data_values[:, 8] == last_result_4_min_val
        
        # Combine masks for all parameter combinations (no time component)
        total_combinations = 0
        for vol in volatilities:
            for ratio in ratios:
                for avg_of_last_2_trades_val in avg_of_last_2_trades:
                    for adx28 in adx28s:
                        for adx14 in adx14s:
                            for adx7 in adx7s:
                                for rsi_type in extreme_rsis:
                                    for last_result_4_min_type in last_result_must_be_4_minutes:
                                        key = (None, vol, ratio, avg_of_last_2_trades_val, adx28, adx14, adx7, rsi_type, last_result_4_min_type)  # None for time
                                        combined_mask = (vol_masks[vol] & 
                                                        ratio_masks[ratio] & 
                                                        avg_of_last_2_trades_masks[avg_of_last_2_trades_val] &
                                                        adx28_masks[adx28] & 
                                                        adx14_masks[adx14] & 
                                                        adx7_masks[adx7] & 
                                                        rsi_masks[rsi_type] &
                                                        last_result_4_min_masks[last_result_4_min_type])
                                        mask_cache[key] = combined_mask
                                        total_combinations += 1
        
        # Calculate approximate memory usage (boolean mask = 1 byte per element)
        memory_mb = (total_combinations * data_values.shape[0]) / (1024 * 1024)
        #print(f"Estimated mask cache memory usage: {memory_mb:.1f} MB")
        
        return mask_cache
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Create_Entries(entry_times, volatilities, ratios, avg_of_last_2_trades, adx28s, adx14s, adx7s, extreme_rsis, last_result_must_be_4_minutes,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, target1s_np_array, target2s_np_array, target3s_np_array, 
                   stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode):
    try:
        # Handle user_mode=1 special volatility filtering
        if user_mode == 1:
            # Override volatilities with hardcoded array for user_mode=1
            volatilities = np.array([0.0, 0.6], dtype=np.float32)
            # For user_mode=1, we don't use time filtering, so create a dummy entry_times
            entry_times = [None]  # This will be handled in mask computation
            
        # Pre-compute all masks for optimization
        if user_mode == 3:
            # Only user_mode=3 uses time-based filtering
            mask_cache = precompute_all_masks(entry_times, volatilities, ratios, avg_of_last_2_trades, adx28s, adx14s, adx7s, 
                                             extreme_rsis, last_result_must_be_4_minutes, data_values)
        else:
            # user_mode=1 and user_mode=2 don't use time filtering
            mask_cache = precompute_all_masks_no_time(volatilities, ratios, avg_of_last_2_trades, adx28s, adx14s, adx7s, 
                                                     extreme_rsis, last_result_must_be_4_minutes, data_values, user_mode)
        
        local_sublists = {}
        total_combinations_tested = 0
        batch_size = 2500000     # large batch size for better parallelization. (on 16gb ram you must have everything but the ide closed to use this size)
        prune_size = 400000      # only allow _sublists to reach this length before pruning them
        batch_count = 0          # how many batches have been processed - used to update user with progress
        combination_batch = None # Struct-of-arrays for combinations
        
        # VECTORIZED APPROACH: Create all outer parameter combinations at once
        # Convert parameters to numpy arrays for broadcasting
        if user_mode == 3:
            entry_times_valid = entry_times[:-1]  # exclude last entry_time
        else:
            # For user_mode 1 and 2, no time filtering
            entry_times_valid = [None]
            
        volatilities_np = np.array(volatilities)
        ratios_np = np.array(ratios)
        avg_of_last_2_trades_np = np.array(avg_of_last_2_trades)
        adx28s_np = np.array(adx28s)
        adx14s_np = np.array(adx14s)
        adx7s_np = np.array(adx7s)
        
        if user_mode == 3:
            # Use numpy meshgrid to create all combinations of outer parameters
            # This replaces the 7 nested loops with vectorized operations
            (entry_times_mesh, volatilities_mesh, ratios_mesh, avg_of_last_2_trades_mesh, adx28s_mesh, 
             adx14s_mesh, adx7s_mesh, rsi_mesh, last_result_4_min_mesh) = np.meshgrid(
                np.arange(len(entry_times_valid)), volatilities_np, ratios_np, avg_of_last_2_trades_np,
                adx28s_np, adx14s_np, adx7s_np, np.arange(len(extreme_rsis)), 
                np.arange(len(last_result_must_be_4_minutes)), indexing='ij'
            )
            
            # Flatten all meshgrids to get 1D arrays of all combinations
            outer_combos_flat = np.column_stack([
                entry_times_mesh.ravel(),
                volatilities_mesh.ravel(), 
                ratios_mesh.ravel(),
                avg_of_last_2_trades_mesh.ravel(),
                adx28s_mesh.ravel(),
                adx14s_mesh.ravel(),
                adx7s_mesh.ravel(),
                rsi_mesh.ravel(),
                last_result_4_min_mesh.ravel()
            ])
        else:
            # For user_mode 1 and 2, no time dimension in meshgrid
            (volatilities_mesh, ratios_mesh, avg_of_last_2_trades_mesh, adx28s_mesh, 
             adx14s_mesh, adx7s_mesh, rsi_mesh, last_result_4_min_mesh) = np.meshgrid(
                volatilities_np, ratios_np, avg_of_last_2_trades_np,
                adx28s_np, adx14s_np, adx7s_np, np.arange(len(extreme_rsis)), 
                np.arange(len(last_result_must_be_4_minutes)), indexing='ij'
            )
            
            # Flatten all meshgrids to get 1D arrays of all combinations (no time)
            outer_combos_flat = np.column_stack([
                np.full(volatilities_mesh.ravel().shape, 0),  # Dummy time index (always 0)
                volatilities_mesh.ravel(), 
                ratios_mesh.ravel(),
                avg_of_last_2_trades_mesh.ravel(),
                adx28s_mesh.ravel(),
                adx14s_mesh.ravel(),
                adx7s_mesh.ravel(),
                rsi_mesh.ravel(),
                last_result_4_min_mesh.ravel()
            ])
        
        # Pre-create inner parameter combinations with validity checks
        # This vectorizes the inner loops with conditional logic
        target_1s_np = np.array(target_1s)
        stop_loss_1s_np = np.array(stop_loss_1s)
        target_2s_np = np.array(target_2s)
        stop_loss_2s_np = np.array(stop_loss_2s)
        target_3s_np = np.array(target_3s)
        stop_loss_3s_np = np.array(stop_loss_3s)
        
        if how_many_final_parameters == 4:
            # For 4 parameters, only use t1, sl1, t2, sl2
            (t1_mesh, sl1_mesh, t2_mesh, sl2_mesh) = np.meshgrid(
                target_1s_np, stop_loss_1s_np, target_2s_np, stop_loss_2s_np, indexing='ij'
            )
            
            # Flatten inner parameter combinations (4 parameters)
            inner_combos = np.column_stack([
                t1_mesh.ravel(), sl1_mesh.ravel(), t2_mesh.ravel(),
                sl2_mesh.ravel(), np.full(t1_mesh.ravel().shape, -999.0), np.full(t1_mesh.ravel().shape, -999.0)  # Use -999.0 as None placeholder
            ])
            # Round to 1 decimal place to avoid floating point precision errors
            inner_combos = np.round(inner_combos, 1)
            
            # Apply vectorized validity checks for inner parameters (4 parameter mode)
            # Condition 1: sl1 < t1 - 0.1
            valid_mask = inner_combos[:, 1] < inner_combos[:, 0] - 0.1
            # Condition 2: t1 < t2
            valid_mask &= inner_combos[:, 0] < inner_combos[:, 2]
            # Condition 3: sl1 < sl2 (sl2 must be greater than sl1)
            valid_mask &= inner_combos[:, 1] < inner_combos[:, 3]
            # Condition 4: sl2 < t1 - 0.1 (invalid condition from original code)
            valid_mask &= inner_combos[:, 3] < inner_combos[:, 0] - 0.1
        else:
            # For 6 parameters, use all parameters
            (t1_mesh, sl1_mesh, t2_mesh, sl2_mesh, t3_mesh, sl3_mesh) = np.meshgrid(
                target_1s_np, stop_loss_1s_np, target_2s_np, 
                stop_loss_2s_np, target_3s_np, stop_loss_3s_np, indexing='ij'
            )
            
            # Flatten inner parameter combinations
            inner_combos = np.column_stack([
                t1_mesh.ravel(), sl1_mesh.ravel(), t2_mesh.ravel(),
                sl2_mesh.ravel(), t3_mesh.ravel(), sl3_mesh.ravel()
            ])
            # Round to 1 decimal place to avoid floating point precision errors
            inner_combos = np.round(inner_combos, 1)
            
            # Apply vectorized validity checks for inner parameters
            # Condition 1: sl1 < t1 - 0.1
            valid_mask = inner_combos[:, 1] < inner_combos[:, 0] - 0.1
            # Condition 2: t1 < t2 AND sl1 < t2
            valid_mask &= (inner_combos[:, 0] < inner_combos[:, 2]) & (inner_combos[:, 1] < inner_combos[:, 2])
            # Condition 3: sl1 < sl2 AND sl2 < t2 - 0.1
            valid_mask &= (inner_combos[:, 1] < inner_combos[:, 3]) & (inner_combos[:, 3] < inner_combos[:, 2] - 0.1)
            # Condition 4: t2 < t3 AND sl2 < t3
            valid_mask &= (inner_combos[:, 2] < inner_combos[:, 4]) & (inner_combos[:, 3] < inner_combos[:, 4])
            # Condition 5: sl2 < sl3 AND sl3 < t3 - 0.1
            valid_mask &= (inner_combos[:, 3] < inner_combos[:, 5]) & (inner_combos[:, 5] < inner_combos[:, 4] - 0.1)
        
        # Filter to only valid inner combinations
        valid_inner_combos = inner_combos[valid_mask]
                
        # OPTIMIZED: Filter valid outer combinations first to avoid unnecessary processing
        valid_outer_indices = []
        valid_mask_keys = []
        
        for outer_idx in range(len(outer_combos_flat)):
            outer_combo = outer_combos_flat[outer_idx]
            entry_time_idx = int(outer_combo[0])
            
            if user_mode == 3:
                entry_time = entry_times_valid[entry_time_idx]
            else:
                entry_time = None  # For user_mode 1 and 2
                
            volatility = outer_combo[1]
            ratio = outer_combo[2]
            avg_of_last_2_trades_val = outer_combo[3]
            adx28 = outer_combo[4]
            adx14 = outer_combo[5]
            adx7 = outer_combo[6]
            rsi_type = extreme_rsis[int(outer_combo[7])]
            last_result_4_min_type = last_result_must_be_4_minutes[int(outer_combo[8])]
            
            mask_key = (entry_time, volatility, ratio, avg_of_last_2_trades_val, adx28, adx14, adx7, rsi_type, last_result_4_min_type)
            combined_mask = mask_cache[mask_key]
            
            if np.any(combined_mask):
                valid_outer_indices.append(outer_idx)
                valid_mask_keys.append(mask_key)
        
        valid_outer_combos = outer_combos_flat[valid_outer_indices]
                
        # MAXIMUM SPEED: Create all combinations at once using broadcasting
        # Pre-calculate all combinations using numpy broadcasting for maximum efficiency
        n_valid_outer = len(valid_outer_combos)
        n_valid_inner = len(valid_inner_combos)
        
        print(f"Creating {n_valid_outer * n_valid_inner} combinations using vectorized operations...")
        
        # Use broadcasting to create all combinations at once
        # Shape: (n_valid_outer, n_valid_inner, ...)
        outer_indices = np.arange(n_valid_outer)[:, np.newaxis]  # Shape: (n_valid_outer, 1)
        inner_indices = np.arange(n_valid_inner)[np.newaxis, :]  # Shape: (1, n_valid_inner)
        
        # Broadcast to create index pairs for all combinations
        outer_broadcast = np.broadcast_to(outer_indices, (n_valid_outer, n_valid_inner)).ravel()
        inner_broadcast = np.broadcast_to(inner_indices, (n_valid_outer, n_valid_inner)).ravel()
        
        # Pre-compute entry_times array for efficiency
        if user_mode == 3:
            entry_times_array = np.array([entry_times_valid[int(combo[0])] for combo in valid_outer_combos])
        else:
            entry_times_array = np.array([None for combo in valid_outer_combos])
        avg_of_last_2_trades_array = np.array([combo[3] for combo in valid_outer_combos])
        rsi_types_array = np.array([extreme_rsis[int(combo[7])] for combo in valid_outer_combos])
        last_result_4_min_types_array = np.array([last_result_must_be_4_minutes[int(combo[8])] for combo in valid_outer_combos])
        
        # Build all combinations in one vectorized operation
        total_combos = len(outer_broadcast)
        
        # Process combinations in efficient chunks to avoid memory issues
        chunk_size = min(batch_size, total_combos)
        
        for chunk_start in range(0, total_combos, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_combos)
            chunk_outer_indices = outer_broadcast[chunk_start:chunk_end]
            chunk_inner_indices = inner_broadcast[chunk_start:chunk_end]
            
            # OPTIMIZED: Create combination batch using list comprehension - much faster than individual appends
            chunk_outer_combos = valid_outer_combos[chunk_outer_indices]
            chunk_inner_combos = valid_inner_combos[chunk_inner_indices]
            chunk_mask_keys = [valid_mask_keys[i] for i in chunk_outer_indices]
            chunk_entry_times = entry_times_array[chunk_outer_indices]
            chunk_avg_of_last_2_trades = avg_of_last_2_trades_array[chunk_outer_indices]
            chunk_rsi_types = rsi_types_array[chunk_outer_indices]
            chunk_last_result_4_min_types = last_result_4_min_types_array[chunk_outer_indices]
            
            # STRUCT-OF-ARRAYS OPTIMIZATION: Use columnar numpy arrays instead of list of dictionaries
            # This enables SIMD optimization, reduces memory usage, and eliminates dictionary overhead
            chunk_size = len(chunk_outer_indices)
            
            # Create arrays for this chunk with appropriate dtypes
            if how_many_final_parameters == 4:
                # For 4-parameter mode, t3 and sl3 are always None -> use -999.0 for Numba compatibility
                chunk_t3_array = np.full(chunk_size, -999.0, dtype=np.float32)
                chunk_sl3_array = np.full(chunk_size, -999.0, dtype=np.float32)
            else:
                # For 6-parameter mode, t3 and sl3 are floats
                chunk_t3_array = chunk_inner_combos[:, 4].astype(np.float32)
                chunk_sl3_array = chunk_inner_combos[:, 5].astype(np.float32)
            
            batch_to_add = {
                'mask_keys': np.array(chunk_mask_keys, dtype=object),
                'entry_times': np.array(chunk_entry_times, dtype=object),
                'volatilities': chunk_outer_combos[:, 1].astype(np.float32),
                'ratios': chunk_outer_combos[:, 2].astype(np.float32),
                'avg_of_last_2_trades': np.array(chunk_avg_of_last_2_trades, dtype=np.float32),
                'adx28s': chunk_outer_combos[:, 4].astype(np.float32),
                'adx14s': chunk_outer_combos[:, 5].astype(np.float32),
                'adx7s': chunk_outer_combos[:, 6].astype(np.float32),
                'rsi_types': np.array(chunk_rsi_types, dtype=object),
                'last_result_4_min_types': np.array(chunk_last_result_4_min_types, dtype=object),
                't1s': chunk_inner_combos[:, 0].astype(np.float32),
                'sl1s': chunk_inner_combos[:, 1].astype(np.float32),
                't2s': chunk_inner_combos[:, 2].astype(np.float32),
                'sl2s': chunk_inner_combos[:, 3].astype(np.float32),
                't3s': chunk_t3_array,
                'sl3s': chunk_sl3_array
            }
            
            # Add chunk to main batch using struct-of-arrays
            if combination_batch is None:
                # Initialize combination_batch as struct-of-arrays
                combination_batch = batch_to_add
            else:
                # Concatenate arrays to existing batch
                for key in batch_to_add:
                    combination_batch[key] = np.concatenate([combination_batch[key], batch_to_add[key]])
            
            # Process batch when it reaches the target size (check any array length)
            if combination_batch is not None and len(combination_batch['mask_keys']) >= batch_size:
                local_sublists = Preprocess_Combination_Batch(
                    combination_batch, local_sublists,
                    target1s_np_array, target2s_np_array, 
                    target3s_np_array, stop_loss1s_np_array, 
                    stop_loss2s_np_array, stop_loss3s_np_array, 
                    t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                    sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                    how_many_final_parameters, mask_cache, data_rows, data_last_prices
                )
                total_combinations_tested += len(combination_batch['mask_keys'])
                combination_batch = None  # Reset for next batch
                batch_count += 1
                
                # Prune when necessary to manage memory
                if len(local_sublists) >= prune_size:
                    local_sublists = prune_sublists(local_sublists, user_mode)
                
                if (batch_count % 5 == 0):
                    print(f"in progress, completed {batch_count} batches of {batch_size}...")

        # Process any remaining combinations in the final batch
        if combination_batch is not None:
            local_sublists = Preprocess_Combination_Batch(
                combination_batch, local_sublists,
                target1s_np_array, target2s_np_array, target3s_np_array, 
                stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array, 
                t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                how_many_final_parameters, mask_cache, data_rows, data_last_prices
            )
            total_combinations_tested += len(combination_batch['mask_keys'])

        print(f"Processed {total_combinations_tested} valid combinations")
        if len(local_sublists) > 10:
            local_sublists = prune_sublists(local_sublists, user_mode)
                                                    
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)

