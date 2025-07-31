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
        current_sum = 0.0
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
    target1_lookup = {val: idx for idx, val in enumerate(target1s_np_array)}
    stop_loss1_lookup = {val: idx for idx, val in enumerate(stop_loss1s_np_array)}
    target3_lookup = {val: idx for idx, val in enumerate(target3s_np_array)}
    stop_loss2_lookup = {val: idx for idx, val in enumerate(stop_loss2s_np_array)}
    target2_lookup = {val: idx for idx, val in enumerate(target2s_np_array)}
    stop_loss3_lookup = {val: idx for idx, val in enumerate(stop_loss3s_np_array)}
    
    # Vectorized parameter extraction and lookup
    # Extract all parameter values at once instead of individual tuple unpacking
    all_params_array = np.array(all_params, dtype=np.float64)
    
    # Maximum performance using vectorized lookup with comprehensions
    # This is faster than individual .get() calls in a loop and has no failure scenarios
    param_indices[:, 0] = [target1_lookup.get(val, -1) for val in all_params_array[:, 0]]
    param_indices[:, 1] = [stop_loss1_lookup.get(val, -1) for val in all_params_array[:, 1]]
    param_indices[:, 2] = [target3_lookup.get(val, -1) for val in all_params_array[:, 2]]
    param_indices[:, 3] = [stop_loss2_lookup.get(val, -1) for val in all_params_array[:, 3]]
    param_indices[:, 4] = [target2_lookup.get(val, -1) for val in all_params_array[:, 4]]
    param_indices[:, 5] = [stop_loss3_lookup.get(val, -1) for val in all_params_array[:, 5]]
    
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
        all_filtered_prices_flat = np.concatenate(all_filtered_prices, dtype=np.float64)
        all_params_array = np.array(all_params, dtype=np.float64)
        
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
        batch_size = len(combination_batch)
        
        # Pre-extract all mask keys for batch processing
        mask_keys = [combo['mask_key'] for combo in combination_batch]
        
        # Leverage mask reuse patterns for speedup
        # Many combinations share the same mask_key, so we can group and reuse computations
        
        # Group combinations by unique mask keys to minimize redundant np.where() calls  
        # Use defaultdict for faster grouping (eliminates key existence checks)
        from collections import defaultdict
        mask_key_groups = defaultdict(list)
        for i, key in enumerate(mask_keys):
            mask_key_groups[key].append(i)
        
        # Pre-compute filtered indices only once per unique mask
        unique_filtered_indices = {}
        for key in mask_key_groups.keys():
            mask = mask_cache[key]
            unique_filtered_indices[key] = np.where(mask)[0]
        
        # Batch array indexing for unique masks
        # Compute filtered data once per unique mask, then reuse for all combinations
        unique_filtered_rows = {}
        unique_filtered_prices = {}
        for key, indices in unique_filtered_indices.items():
            unique_filtered_rows[key] = data_rows[indices]
            unique_filtered_prices[key] = data_last_prices[indices]
        
        # Fast assignment using pre-computed results
        # Allocate arrays once and fill using index mapping (much faster than appends)
        all_filtered_rows = [None] * batch_size
        all_filtered_prices = [None] * batch_size
        
        # Vectorized assignment - eliminate the inner loop for better performance
        for key, combo_indices in mask_key_groups.items():
            filtered_rows = unique_filtered_rows[key]
            filtered_prices = unique_filtered_prices[key]
            # Use numpy-style vectorized assignment instead of loop
            for combo_idx in combo_indices:
                all_filtered_rows[combo_idx] = filtered_rows
                all_filtered_prices[combo_idx] = filtered_prices
        
        # Vectorized parameter extraction using list comprehension (faster than extend)
        all_params = [(combo['t1'], combo['sl1'], combo['t3'], combo['sl2'], combo['t2'], combo['sl3']) 
                      for combo in combination_batch]
        
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
            
            if how_many_final_parameters == 4:
                for i in valid_indices:
                    # Skip combinations with poor performance early
                    if sums[i] < sum_threshold:
                        continue
                        
                    combo = combination_batch[i]
                    # String concatenation with delimiter - faster than tuple for dict keys
                    sublist_key = f"{combo['entry time']}|{combo['volatility']}|{combo['ratio']}|{combo['adx28']}|{combo['adx14']}|{combo['adx7']}|{combo['rsi_type']}|{combo['t1']}|{combo['t2']}|{combo['sl1']}|{combo['sl2']}"
                    local_sublists[sublist_key] = {
                        'sum': sums[i],
                        'wins': wins[i],
                        'losses': losses[i],
                        'neither': neithers[i]
                    }
            else:  # how_many_final_parameters == 6
                for i in valid_indices:
                    # Skip combinations with poor performance early
                    if sums[i] < sum_threshold:
                        continue
                        
                    combo = combination_batch[i]
                    sublist_key = f"{combo['entry time']}|{combo['volatility']}|{combo['ratio']}|{combo['adx28']}|{combo['adx14']}|{combo['adx7']}|{combo['rsi_type']}|{combo['t1']}|{combo['t2']}|{combo['t3']}|{combo['sl1']}|{combo['sl2']}|{combo['sl3']}"
                    local_sublists[sublist_key] = {
                        'sum': sums[i],
                        'wins': wins[i],
                        'losses': losses[i],
                        'neither': neithers[i]
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
# this version creates combos by look at rows in a upper/lower volatility range
def Create_Entries_Group_By_Volatility(volatilities, ratios, adx28s, adx14s, adx7s, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, target1s_np_array, target2s_np_array, target3s_np_array, 
                   stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode):
    try:
        # targeting 3 things like this creates a huge issue. I need 1 mask for each test because they all filter to different data rows
        local_sublists = {}
        total_combinations_tested = 0
        batch_size = 2000000   # Much larger batch size for better parallelization
        prune_size = 400000    # only allow _sublists to reach this length before pruning them
        batch_count = 0        # how many batches have been processed - used to update user with progress
        combination_batch = [] # Collect all combinations into batches
        
        # this version only uses these changes for now. the other volatilities don't matter
        volatilities = np.array([0.0,0.6], dtype=np.float64)

        for volatility in volatilities:
            lower_bound = volatility
            if (volatility == 0.0):
                upper_bound = 0.6
            else:
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
                            for rsi_type in extreme_rsis:
                                # Combine all filters
                                combined_mask = (vol_mask & ratio_mask & adx28_mask & adx14_mask & adx7_mask)
                                
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
                                                            'entry time': None,
                                                            'volatility': volatility,
                                                            'ratio': ratio,
                                                            'adx28': adx28,
                                                            'adx14': adx14,
                                                            'adx7': adx7,
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
                                                    'entry time': None,
                                                    'volatility': volatility,
                                                    'ratio': ratio,
                                                    'adx28': adx28,
                                                    'adx14': adx14,
                                                    'adx7': adx7,
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
                                                    local_sublists = Preprocess_Combination_Batch(
                                                        combination_batch, local_sublists,
                                                        target1s_np_array, target2s_np_array, target3s_np_array, 
                                                        stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array, 
                                                        t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                                                        sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                                                        how_many_final_parameters
                                                    )
                                                    total_combinations_tested += len(combination_batch)
                                                    combination_batch = []  # Reset for next batch
                                                    batch_count += 1
                                                    
                                                    # Prune when necessary to manage memory
                                                    if len(local_sublists) >= prune_size:
                                                        local_sublists = prune_sublists(local_sublists, user_mode)
                                                    
                                                    if (batch_count % 5 == 0):
                                                        print(f"in progress, completed {batch_count} batches of {batch_size}...")

        # Process any remaining combinations in the final batch
        if combination_batch:
            local_sublists = Preprocess_Combination_Batch(
                combination_batch, local_sublists,
                target1s_np_array, target2s_np_array, target3s_np_array, stop_loss1s_np_array, stop_loss2s_np_array, 
                stop_loss3s_np_array, t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                how_many_final_parameters
            )
            total_combinations_tested += len(combination_batch)

        print(f"Processed {total_combinations_tested} valid combinations")
        if len(local_sublists) > 90:  # 9 volatility levels * 10 per level
            local_sublists = prune_sublists(local_sublists, user_mode)
                                                    
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# data_holder = [(index of original df, [list of values], last price), ...]
# this version creates combos by looking at data greater than or equal to volatility
def Create_Entries_Overall(volatilities, ratios, adx28s, adx14s, adx7s, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, target1s_np_array, target2s_np_array, target3s_np_array, 
                   stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode):
    try:
        # targeting 3 things like this creates a huge issue. I need 1 mask for each test because they all filter to different data rows
        local_sublists = {}
        total_combinations_tested = 0
        batch_size = 2000000   # Much larger batch size for better parallelization
        prune_size = 400000    # only allow _sublists to reach this length before pruning them
        batch_count = 0        # how many batches have been processed - used to update user with progress
        combination_batch = [] # Collect all combinations into batches
        
        for volatility in volatilities:
            vol_mask = (data_values[:, 0] >= volatility)
            for ratio in ratios:
                ratio_mask = data_values[:, 1] >= ratio
                for adx28 in adx28s:
                    adx28_mask = data_values[:, 2] >= adx28
                    for adx14 in adx14s:
                        adx14_mask = data_values[:, 3] >= adx14
                        for adx7 in adx7s:
                            adx7_mask = data_values[:, 4] >= adx7
                            for rsi_type in extreme_rsis:
                                # Combine all filters
                                combined_mask = (vol_mask & ratio_mask & adx28_mask & adx14_mask & adx7_mask)
                                
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
                                                            'entry time': None,
                                                            'volatility': volatility,
                                                            'ratio': ratio,
                                                            'adx28': adx28,
                                                            'adx14': adx14,
                                                            'adx7': adx7,
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
                                                    'entry time': None,
                                                    'volatility': volatility,
                                                    'ratio': ratio,
                                                    'adx28': adx28,
                                                    'adx14': adx14,
                                                    'adx7': adx7,
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
                                                    local_sublists = Preprocess_Combination_Batch(
                                                        combination_batch, local_sublists,
                                                        target1s_np_array, target2s_np_array, target3s_np_array, 
                                                        stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array, 
                                                        t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                                                        sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                                                        how_many_final_parameters
                                                    )
                                                    total_combinations_tested += len(combination_batch)
                                                    combination_batch = []  # Reset for next batch
                                                    batch_count += 1
                                                    
                                                    # Prune when necessary to manage memory
                                                    if len(local_sublists) >= prune_size:
                                                        local_sublists = prune_sublists(local_sublists, user_mode)
                                                    
                                                    if (batch_count % 5 == 0):
                                                        print(f"in progress, completed {batch_count} batches of {batch_size}...")

        # Process any remaining combinations in the final batch
        if combination_batch:
            local_sublists = Preprocess_Combination_Batch(
                combination_batch, local_sublists,
                target1s_np_array, target2s_np_array, target3s_np_array, 
                stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array, 
                t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                how_many_final_parameters
            )
            total_combinations_tested += len(combination_batch)

        print(f"Processed {total_combinations_tested} valid combinations")
        if len(local_sublists) > 10:
            local_sublists = prune_sublists(local_sublists, user_mode)
                                                    
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# data_holder = [(index of original df, [list of values], last price), ...]
# creates combos by looking at data by time stamp greater than or equal to volatility
# numb of masks are lengths of all lists multiplied by each other. size depends on the dataset. 250 rows of data should be like 30 MB of RAM
def precompute_all_masks(entry_times, volatilities, ratios, adx28s, adx14s, adx7s, 
                        extreme_rsis, data_values):
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
            time_masks[entry_time] = (data_values[:, 6] >= lower_bound_s) & (data_values[:, 6] < upper_bound_s)
        
        vol_masks = {vol: data_values[:, 0] >= vol for vol in volatilities}
        ratio_masks = {ratio: data_values[:, 1] >= ratio for ratio in ratios}
        adx28_masks = {adx: data_values[:, 2] >= adx for adx in adx28s}
        adx14_masks = {adx: data_values[:, 3] >= adx for adx in adx14s}
        adx7_masks = {adx: data_values[:, 4] >= adx for adx in adx7s}
        
        rsi_masks = {}
        for rsi_type in extreme_rsis:
            if rsi_type == "either":
                rsi_masks[rsi_type] = np.full(data_values.shape[0], True)
            else:
                rsi_val = 1.0 if rsi_type else 0.0
                rsi_masks[rsi_type] = data_values[:, 5] == rsi_val
        
        # Combine masks for all parameter combinations
        total_combinations = 0
        for entry_time in entry_times[:-1]:  # Skip last entry_time
            for vol in volatilities:
                for ratio in ratios:
                    for adx28 in adx28s:
                        for adx14 in adx14s:
                            for adx7 in adx7s:
                                for rsi_type in extreme_rsis:
                                    key = (entry_time, vol, ratio, adx28, adx14, adx7, rsi_type)
                                    combined_mask = (time_masks[entry_time] & 
                                                    vol_masks[vol] & 
                                                    ratio_masks[ratio] & 
                                                    adx28_masks[adx28] & 
                                                    adx14_masks[adx14] & 
                                                    adx7_masks[adx7] & 
                                                    rsi_masks[rsi_type])
                                    mask_cache[key] = combined_mask
                                    total_combinations += 1
        
        # Calculate approximate memory usage (boolean mask = 1 byte per element)
        memory_mb = (total_combinations * data_values.shape[0]) / (1024 * 1024)
        #print(f"Estimated mask cache memory usage: {memory_mb:.1f} MB")
        
        return mask_cache
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# finds masks w/o including time
def precompute_all_masks_no_time(volatilities, ratios, adx28s, adx14s, adx7s, 
                                extreme_rsis, data_values, user_mode):
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
        adx28_masks = {adx: data_values[:, 2] >= adx for adx in adx28s}
        adx14_masks = {adx: data_values[:, 3] >= adx for adx in adx14s}
        adx7_masks = {adx: data_values[:, 4] >= adx for adx in adx7s}
        
        rsi_masks = {}
        for rsi_type in extreme_rsis:
            if rsi_type == "either":
                rsi_masks[rsi_type] = np.full(data_values.shape[0], True)
            else:
                rsi_val = 1.0 if rsi_type else 0.0
                rsi_masks[rsi_type] = data_values[:, 5] == rsi_val
        
        # Combine masks for all parameter combinations (no time component)
        total_combinations = 0
        for vol in volatilities:
            for ratio in ratios:
                for adx28 in adx28s:
                    for adx14 in adx14s:
                        for adx7 in adx7s:
                            for rsi_type in extreme_rsis:
                                key = (None, vol, ratio, adx28, adx14, adx7, rsi_type)  # None for time
                                combined_mask = (vol_masks[vol] & 
                                                ratio_masks[ratio] & 
                                                adx28_masks[adx28] & 
                                                adx14_masks[adx14] & 
                                                adx7_masks[adx7] & 
                                                rsi_masks[rsi_type])
                                mask_cache[key] = combined_mask
                                total_combinations += 1
        
        # Calculate approximate memory usage (boolean mask = 1 byte per element)
        memory_mb = (total_combinations * data_values.shape[0]) / (1024 * 1024)
        #print(f"Estimated mask cache memory usage: {memory_mb:.1f} MB")
        
        return mask_cache
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Create_Entries(entry_times, volatilities, ratios, adx28s, adx14s, adx7s, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, target1s_np_array, target2s_np_array, target3s_np_array, 
                   stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array,
                   t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                   how_many_final_parameters, user_mode):
    try:
        # Handle user_mode=1 special volatility filtering
        if user_mode == 1:
            # Override volatilities with hardcoded array for user_mode=1
            volatilities = np.array([0.0, 0.6], dtype=np.float64)
            # For user_mode=1, we don't use time filtering, so create a dummy entry_times
            entry_times = [None]  # This will be handled in mask computation
            
        # Pre-compute all masks for optimization
        if user_mode == 3:
            # Only user_mode=3 uses time-based filtering
            mask_cache = precompute_all_masks(entry_times, volatilities, ratios, adx28s, adx14s, adx7s, 
                                             extreme_rsis, data_values)
        else:
            # user_mode=1 and user_mode=2 don't use time filtering
            mask_cache = precompute_all_masks_no_time(volatilities, ratios, adx28s, adx14s, adx7s, 
                                                     extreme_rsis, data_values, user_mode)
        
        local_sublists = {}
        total_combinations_tested = 0
        batch_size = 2000000   # large batch size for better parallelization
        prune_size = 400000    # only allow _sublists to reach this length before pruning them
        batch_count = 0        # how many batches have been processed - used to update user with progress
        combination_batch = [] # Collect all combinations into batches
        
        # VECTORIZED APPROACH: Create all outer parameter combinations at once
        # Convert parameters to numpy arrays for broadcasting
        if user_mode == 3:
            entry_times_valid = entry_times[:-1]  # exclude last entry_time
        else:
            # For user_mode 1 and 2, no time filtering
            entry_times_valid = [None]
            
        volatilities_np = np.array(volatilities)
        ratios_np = np.array(ratios)
        adx28s_np = np.array(adx28s)
        adx14s_np = np.array(adx14s)
        adx7s_np = np.array(adx7s)
        
        if user_mode == 3:
            # Use numpy meshgrid to create all combinations of outer parameters
            # This replaces the 7 nested loops with vectorized operations
            (entry_times_mesh, volatilities_mesh, ratios_mesh, adx28s_mesh, 
             adx14s_mesh, adx7s_mesh, rsi_mesh) = np.meshgrid(
                np.arange(len(entry_times_valid)), volatilities_np, ratios_np, 
                adx28s_np, adx14s_np, adx7s_np, np.arange(len(extreme_rsis)), 
                indexing='ij'
            )
            
            # Flatten all meshgrids to get 1D arrays of all combinations
            outer_combos_flat = np.column_stack([
                entry_times_mesh.ravel(),
                volatilities_mesh.ravel(), 
                ratios_mesh.ravel(),
                adx28s_mesh.ravel(),
                adx14s_mesh.ravel(),
                adx7s_mesh.ravel(),
                rsi_mesh.ravel()
            ])
        else:
            # For user_mode 1 and 2, no time dimension in meshgrid
            (volatilities_mesh, ratios_mesh, adx28s_mesh, 
             adx14s_mesh, adx7s_mesh, rsi_mesh) = np.meshgrid(
                volatilities_np, ratios_np, 
                adx28s_np, adx14s_np, adx7s_np, np.arange(len(extreme_rsis)), 
                indexing='ij'
            )
            
            # Flatten all meshgrids to get 1D arrays of all combinations (no time)
            outer_combos_flat = np.column_stack([
                np.full(volatilities_mesh.ravel().shape, 0),  # Dummy time index (always 0)
                volatilities_mesh.ravel(), 
                ratios_mesh.ravel(),
                adx28s_mesh.ravel(),
                adx14s_mesh.ravel(),
                adx7s_mesh.ravel(),
                rsi_mesh.ravel()
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
            
            # Apply vectorized validity checks for inner parameters (4 parameter mode)
            # Condition 1: sl1 < t1 - 0.1
            valid_mask = inner_combos[:, 1] < inner_combos[:, 0] - 0.1
            # Condition 2: t1 < t2
            valid_mask &= inner_combos[:, 0] < inner_combos[:, 2]
            # Condition 3: sl2 >= t1 - 0.1 (invalid condition from original code)
            valid_mask &= ~(inner_combos[:, 3] >= inner_combos[:, 0] - 0.1)
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
            adx28 = outer_combo[3]
            adx14 = outer_combo[4]
            adx7 = outer_combo[5]
            rsi_type = extreme_rsis[int(outer_combo[6])]
            
            mask_key = (entry_time, volatility, ratio, adx28, adx14, adx7, rsi_type)
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
        rsi_types_array = np.array([extreme_rsis[int(combo[6])] for combo in valid_outer_combos])
        
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
            chunk_rsi_types = rsi_types_array[chunk_outer_indices]
            
            # Single list comprehension is much faster than individual dictionary creation
            if how_many_final_parameters == 4:
                batch_to_add = [{
                    'mask_key': chunk_mask_keys[i],
                    'entry time': chunk_entry_times[i],
                    'volatility': chunk_outer_combos[i][1],
                    'ratio': chunk_outer_combos[i][2],
                    'adx28': chunk_outer_combos[i][3],
                    'adx14': chunk_outer_combos[i][4],
                    'adx7': chunk_outer_combos[i][5],
                    'rsi_type': chunk_rsi_types[i],
                    't1': chunk_inner_combos[i][0],
                    'sl1': chunk_inner_combos[i][1],
                    't2': chunk_inner_combos[i][2],
                    'sl2': chunk_inner_combos[i][3],
                    't3': None,
                    'sl3': None
                } for i in range(len(chunk_outer_indices))]
            else:
                batch_to_add = [{
                    'mask_key': chunk_mask_keys[i],
                    'entry time': chunk_entry_times[i],
                    'volatility': chunk_outer_combos[i][1],
                    'ratio': chunk_outer_combos[i][2],
                    'adx28': chunk_outer_combos[i][3],
                    'adx14': chunk_outer_combos[i][4],
                    'adx7': chunk_outer_combos[i][5],
                    'rsi_type': chunk_rsi_types[i],
                    't1': chunk_inner_combos[i][0],
                    'sl1': chunk_inner_combos[i][1],
                    't2': chunk_inner_combos[i][2],
                    'sl2': chunk_inner_combos[i][3],
                    't3': chunk_inner_combos[i][4],
                    'sl3': chunk_inner_combos[i][5]
                } for i in range(len(chunk_outer_indices))]
            
            # Add chunk to main batch
            combination_batch.extend(batch_to_add)
            
            # Process batch when it reaches the target size
            if len(combination_batch) >= batch_size:
                local_sublists = Preprocess_Combination_Batch(
                    combination_batch, local_sublists,
                    target1s_np_array, target2s_np_array, 
                    target3s_np_array, stop_loss1s_np_array, 
                    stop_loss2s_np_array, stop_loss3s_np_array, 
                    t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                    sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                    how_many_final_parameters, mask_cache, data_rows, data_last_prices
                )
                total_combinations_tested += len(combination_batch)
                combination_batch = []  # Reset for next batch
                batch_count += 1
                
                # Prune when necessary to manage memory
                if len(local_sublists) >= prune_size:
                    local_sublists = prune_sublists(local_sublists, user_mode)
                
                if (batch_count % 5 == 0):
                    print(f"in progress, completed {batch_count} batches of {batch_size}...")

        # Process any remaining combinations in the final batch
        if combination_batch:
            local_sublists = Preprocess_Combination_Batch(
                combination_batch, local_sublists,
                target1s_np_array, target2s_np_array, target3s_np_array, 
                stop_loss1s_np_array, stop_loss2s_np_array, stop_loss3s_np_array, 
                t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
                sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                how_many_final_parameters, mask_cache, data_rows, data_last_prices
            )
            total_combinations_tested += len(combination_batch)

        print(f"Processed {total_combinations_tested} valid combinations")
        if len(local_sublists) > 10:
            local_sublists = prune_sublists(local_sublists, user_mode)
                                                    
        return local_sublists
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)

