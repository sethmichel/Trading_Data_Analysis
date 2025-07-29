import pandas as pd
import numpy as np
import os
import inspect
import sys
import Main_Globals
import heapq
from numba import jit, prange

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
                volatility = sublist_key[1]  # volatility is the 2nd element in the tuple
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
                entry_time = sublist_key[0]  # entry time is the 1st element in the tuple
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


def Precompute_Parameter_Indices(all_params, t1_map, sl1_map, t3_map, sl2_map, t2_map, sl3_map):
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


def Process_Combination_Batch(all_filtered_rows, all_filtered_prices, all_params, t1_map_arr, t1_idx_arrays,
                              sl1_map_arr, sl1_idx_arrays, t3_map_arr, t3_idx_arrays, sl2_map_arr, sl2_idx_arrays,
                              t2_map_arr, t2_idx_arrays, sl3_map_arr, sl3_idx_arrays, how_many_final_parameters):
    try:
        """
        Processes a large batch of parameter combinations using optimized Numba with parallel processing.
        """
        n_combos = len(all_params)
        
        # Pre-compute parameter indices to eliminate O(n) searches
        param_indices = Precompute_Parameter_Indices(all_params, t1_map_arr, sl1_map_arr, 
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


"""batch computation calls this, it preps the data and calls the optimized processing function"""
def Preprocess_Combination_Batch(combination_batch, local_sublists, t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
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
        sums, wins, losses, neithers, valid_mask = Process_Combination_Batch(
            all_filtered_rows, all_filtered_prices, all_params,
            t1_map, t1_idx_arrays, sl1_map, sl1_idx_arrays,
            t3_map, t3_idx_arrays, sl2_map, sl2_idx_arrays,
            t2_map, t2_idx_arrays, sl3_map, sl3_idx_arrays, how_many_final_parameters
        )
        
        # Add valid results to local_sublists
        for i in range(batch_size):
            if valid_mask[i]:
                combo = combination_batch[i]
                if (how_many_final_parameters == 4):
                    sublist_key = (
                        combo['entry time'], combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], 
                        combo['adx7'], combo['zscore'], combo['rsi_type'], combo['t1'], 
                        combo['t2'], combo['sl1'], combo['sl2']
                    )
                    
                elif (how_many_final_parameters == 6):
                    sublist_key = (
                        combo['entry time'], combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], 
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


def time_str_to_seconds(time_str):
    try:
        h, m, s = map(int, str(time_str).split(':'))
        return h * 3600 + m * 60 + s
    except (ValueError, AttributeError):
        return -1


# data_holder = [(index of original df, [list of values], last price), ...]
# this version creates combos by look at rows in a upper/lower volatility range
def Create_Entries_Group_By_Volatility(volatilities, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
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
                                                                'entry time': None,
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
                                                        'entry time': None,
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
                                                        local_sublists = Preprocess_Combination_Batch(
                                                            combination_batch, local_sublists,
                                                            t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
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
                t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
                t1_idx_arrays, t2_idx_arrays, t3_idx_arrays, 
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
def Create_Entries_Overall(volatilities, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
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
                                                                'entry time': None,
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
                                                        'entry time': None,
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
                                                        local_sublists = Preprocess_Combination_Batch(
                                                            combination_batch, local_sublists,
                                                            t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
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
                t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
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
# this version creates combos by looking at data by time stamp greater than or equal to volatility
def Create_Entries_Group_By_Time(entry_times, volatilities, ratios, adx28s, adx14s, adx7s, abs_macd_zScores, extreme_rsis,
                   target_1s, target_3s, stop_loss_2s, stop_loss_1s, target_2s, stop_loss_3s, data_rows, 
                   data_values, data_last_prices, t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map,
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
        
        for entry_time in entry_times:
            lower_bound_s = time_str_to_seconds(entry_time)

            if (entry_time == '6:30:00'):
                upper_bound_s = time_str_to_seconds('6:45:00')
            elif (entry_time == '6:45:00'):
                upper_bound_s = time_str_to_seconds('7:00:00')
            elif (entry_time == '7:00:00'):
                upper_bound_s = time_str_to_seconds('7:30:00')

            time_mask = (data_values[:, 7] >= lower_bound_s) & (data_values[:, 7] < upper_bound_s)

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
                                    if zscore == 0:
                                        # For a zscore of 0, include all data, including NaNs.
                                        zscore_mask = np.full(data_values.shape[0], True)
                                    else:
                                        # Otherwise, use the absolute value for filtering, which will exclude NaNs.
                                        zscore_mask = np.abs(data_values[:, 5]) >= zscore
                                    
                                    for rsi_type in extreme_rsis:
                                        # Combine all filters
                                        combined_mask = (time_mask & vol_mask & ratio_mask & adx28_mask & adx14_mask & adx7_mask & zscore_mask)
                                        
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
                                                                    'entry time': entry_time,
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
                                                            'entry time': entry_time,
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
                                                            local_sublists = Preprocess_Combination_Batch(
                                                                combination_batch, local_sublists,
                                                                t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
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
                t1_map, t2_map, t3_map, sl1_map, sl2_map, sl3_map, 
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