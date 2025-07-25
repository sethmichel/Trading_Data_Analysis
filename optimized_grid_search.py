import pandas as pd
import numpy as np
import multiprocessing as mp
from numba import jit, prange, types
from numba.typed import Dict
import time

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def process_combinations_optimized(all_filtered_rows_flat, all_filtered_prices_flat, 
                                 combination_starts, combination_lengths,
                                 all_params, t1_idx_dict, sl1_idx_dict, t3_idx_dict,
                                 sl2_idx_dict, t2_idx_dict, sl3_idx_dict,
                                 t1_idx_arrays, sl1_idx_arrays, t3_idx_arrays,
                                 sl2_idx_arrays, t2_idx_arrays, sl3_idx_arrays,
                                 how_many_final_parameters):
    """
    Ultra-optimized Numba function with O(1) parameter lookups
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
        
        # O(1) hash lookups instead of O(n) linear searches
        if t1 not in t1_idx_dict or sl1 not in sl1_idx_dict or t3 not in t3_idx_dict:
            valid_mask[i] = False
            continue
        if sl2 not in sl2_idx_dict or t2 not in t2_idx_dict or sl3 not in sl3_idx_dict:
            valid_mask[i] = False
            continue
            
        t1_map_idx = t1_idx_dict[t1]
        sl1_map_idx = sl1_idx_dict[sl1]
        sl2_map_idx = sl2_idx_dict[sl2]
        t2_map_idx = t2_idx_dict[t2]
        
        # Select the correct pre-computed index arrays
        t1_idx_array = t1_idx_arrays[t1_map_idx]
        t2_idx_array = t2_idx_arrays[t1_map_idx, t2_map_idx]
        sl1_idx_array = sl1_idx_arrays[sl1_map_idx]
        sl2_idx_array = sl2_idx_arrays[t1_map_idx, sl2_map_idx]
        
        # only do these if we're doing the extended calculation
        if (how_many_final_parameters == 6):
            t3_map_idx = t3_idx_dict[t3]
            sl3_map_idx = sl3_idx_dict[sl3]
            t3_idx_array = t3_idx_arrays[t1_map_idx, t3_map_idx]
            sl3_idx_array = sl3_idx_arrays[t1_map_idx, sl3_map_idx]
            
        # Initialize counters for this combination
        current_sum = 0.0
        current_wins = 0
        current_losses = 0
        current_neither = 0
        
        # Vectorized inner loop processing
        for j in range(length):
            row_idx = all_filtered_rows_flat[start_idx + j]
            final_price = all_filtered_prices_flat[start_idx + j]
            
            t1_idx = t1_idx_array[row_idx]
            sl1_idx = sl1_idx_array[row_idx]
            
            if how_many_final_parameters == 4:
                if sl1_idx < t1_idx:
                    current_sum += sl1
                    current_losses += 1
                elif t1_idx < sl1_idx:
                    t2_idx = t2_idx_array[row_idx]
                    sl2_idx = sl2_idx_array[row_idx]
                    
                    if sl2_idx < t2_idx:
                        current_sum += sl2
                        current_losses += 1
                    elif t2_idx < sl2_idx:
                        current_sum += t2
                        current_wins += 1
                    else:
                        current_sum += final_price
                        current_neither += 1
                else:
                    current_sum += final_price
                    current_neither += 1
                    
            elif how_many_final_parameters == 6:
                if sl1_idx < t1_idx:
                    current_sum += sl1
                    current_losses += 1
                elif t1_idx < sl1_idx:
                    t2_idx = t2_idx_array[row_idx]
                    sl2_idx = sl2_idx_array[row_idx]
                    
                    if sl2_idx < t2_idx:
                        current_sum += sl2
                        current_losses += 1
                    elif t2_idx < sl2_idx:
                        t3_idx = t3_idx_array[row_idx]
                        sl3_idx = sl3_idx_array[row_idx]
                        
                        if sl3_idx < t3_idx:
                            current_sum += sl3
                            current_wins += 1
                        elif t3_idx < sl3_idx:
                            current_sum += t3
                            current_wins += 1
                        else:
                            current_sum += final_price
                            current_neither += 1
                    else:
                        current_sum += final_price
                        current_neither += 1
                else:
                    current_sum += final_price
                    current_neither += 1
        
        # Store results for this combination
        sums[i] = current_sum
        wins[i] = current_wins
        losses[i] = current_losses
        neithers[i] = current_neither
    
    return sums, wins, losses, neithers, valid_mask


def create_parameter_hash_dicts(t1_map, sl1_map, t3_map, sl2_map, t2_map, sl3_map):
    """Create Numba-compatible hash dictionaries for O(1) parameter lookups"""
    
    # Create Numba typed dictionaries for fast lookups
    t1_idx_dict = Dict.empty(key_type=types.float64, value_type=types.int32)
    sl1_idx_dict = Dict.empty(key_type=types.float64, value_type=types.int32)
    t3_idx_dict = Dict.empty(key_type=types.float64, value_type=types.int32)
    sl2_idx_dict = Dict.empty(key_type=types.float64, value_type=types.int32)
    t2_idx_dict = Dict.empty(key_type=types.float64, value_type=types.int32)
    sl3_idx_dict = Dict.empty(key_type=types.float64, value_type=types.int32)
    
    for i, val in enumerate(t1_map):
        t1_idx_dict[val] = i
    for i, val in enumerate(sl1_map):
        sl1_idx_dict[val] = i
    for i, val in enumerate(t3_map):
        t3_idx_dict[val] = i
    for i, val in enumerate(sl2_map):
        sl2_idx_dict[val] = i
    for i, val in enumerate(t2_map):
        t2_idx_dict[val] = i
    for i, val in enumerate(sl3_map):
        sl3_idx_dict[val] = i
        
    return t1_idx_dict, sl1_idx_dict, t3_idx_dict, sl2_idx_dict, t2_idx_dict, sl3_idx_dict


def process_batch_chunk_multiprocess(args):
    """Worker function for multiprocessing - processes a chunk of combinations"""
    (chunk_data, t1_map, sl1_map, t3_map, sl2_map, t2_map, sl3_map,
     t1_idx_arrays, sl1_idx_arrays, t3_idx_arrays, sl2_idx_arrays, 
     t2_idx_arrays, sl3_idx_arrays, how_many_final_parameters) = args
    
    # Unpack chunk data
    all_filtered_rows, all_filtered_prices, all_params = chunk_data
    n_combos = len(all_params)
    
    if n_combos == 0:
        return [], [], [], [], []
    
    # Create hash dictionaries for fast parameter lookups
    t1_idx_dict, sl1_idx_dict, t3_idx_dict, sl2_idx_dict, t2_idx_dict, sl3_idx_dict = create_parameter_hash_dicts(
        t1_map, sl1_map, t3_map, sl2_map, t2_map, sl3_map)
    
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
    
    # Process with optimized Numba function
    sums, wins, losses, neithers, valid_mask = process_combinations_optimized(
        all_filtered_rows_flat, all_filtered_prices_flat,
        combination_starts, combination_lengths,
        all_params_array, t1_idx_dict, sl1_idx_dict, t3_idx_dict,
        sl2_idx_dict, t2_idx_dict, sl3_idx_dict,
        t1_idx_arrays, sl1_idx_arrays, t3_idx_arrays,
        sl2_idx_arrays, t2_idx_arrays, sl3_idx_arrays,
        how_many_final_parameters
    )
    
    return sums, wins, losses, neithers, valid_mask


def split_batch_for_multiprocessing(combination_batch, num_processes=None):
    """Split a large batch into smaller chunks for multiprocessing"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    chunk_size = max(1, len(combination_batch) // num_processes)
    chunks = []
    
    for i in range(0, len(combination_batch), chunk_size):
        chunk = combination_batch[i:i + chunk_size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks


def process_combination_batch_hybrid(combination_batch, local_sublists, t1_map, t2_map, t3_map, 
                                   sl1_map, sl2_map, sl3_map, t1_idx_arrays, t2_idx_arrays, 
                                   t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                                   how_many_final_parameters):
    """
    Hybrid approach: Numba + Hash lookups + Multiprocessing
    """
    try:
        batch_size = len(combination_batch)
        
        # If batch is small, use single-process Numba
        if batch_size < 100000:
            return process_single_batch_numba(combination_batch, local_sublists, t1_map, t2_map, t3_map,
                                            sl1_map, sl2_map, sl3_map, t1_idx_arrays, t2_idx_arrays,
                                            t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                                            how_many_final_parameters)
        
        # For large batches, use multiprocessing with Numba
        num_processes = mp.cpu_count()
        chunks = split_batch_for_multiprocessing(combination_batch, num_processes)
        
        # Prepare data for each chunk
        chunk_data_list = []
        for chunk in chunks:
            all_filtered_rows = []
            all_filtered_prices = []
            all_params = []
            
            for combo in chunk:
                all_filtered_rows.append(combo['filtered_rows'])
                all_filtered_prices.append(combo['filtered_prices'])
                all_params.append((combo['t1'], combo['sl1'], combo['t3'], 
                                 combo['sl2'], combo['t2'], combo['sl3']))
            
            chunk_data = (all_filtered_rows, all_filtered_prices, all_params)
            chunk_args = (chunk_data, t1_map, sl1_map, t3_map, sl2_map, t2_map, sl3_map,
                         t1_idx_arrays, sl1_idx_arrays, t3_idx_arrays, sl2_idx_arrays,
                         t2_idx_arrays, sl3_idx_arrays, how_many_final_parameters)
            chunk_data_list.append(chunk_args)
        
        # Process chunks in parallel
        print(f"Processing {batch_size} combinations using {len(chunks)} processes...")
        start_time = time.time()
        
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(process_batch_chunk_multiprocess, chunk_data_list)
        
        process_time = time.time() - start_time
        print(f"Parallel processing completed in {process_time:.2f}s")
        
        # Combine results from all chunks
        chunk_idx = 0
        for chunk, (sums, wins, losses, neithers, valid_mask) in zip(chunks, results):
            for i, combo in enumerate(chunk):
                if valid_mask[i]:
                    if how_many_final_parameters == 4:
                        sublist_key = (
                            combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], 
                            combo['adx7'], combo['zscore'], combo['rsi_type'], combo['t1'], 
                            combo['t2'], combo['sl1'], combo['sl2']
                        )
                    elif how_many_final_parameters == 6:
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
            chunk_idx += 1
        
        return local_sublists
        
    except Exception as e:
        print(f"Error in hybrid processing: {e}")
        return local_sublists


def process_single_batch_numba(combination_batch, local_sublists, t1_map, t2_map, t3_map,
                              sl1_map, sl2_map, sl3_map, t1_idx_arrays, t2_idx_arrays,
                              t3_idx_arrays, sl1_idx_arrays, sl2_idx_arrays, sl3_idx_arrays,
                              how_many_final_parameters):
    """Single-process version with hash lookup optimization"""
    
    # Create hash dictionaries for fast parameter lookups
    t1_idx_dict, sl1_idx_dict, t3_idx_dict, sl2_idx_dict, t2_idx_dict, sl3_idx_dict = create_parameter_hash_dicts(
        t1_map, sl1_map, t3_map, sl2_map, t2_map, sl3_map)
    
    # Prepare data
    all_filtered_rows = []
    all_filtered_prices = []
    all_params = []
    
    for combo in combination_batch:
        all_filtered_rows.append(combo['filtered_rows'])
        all_filtered_prices.append(combo['filtered_prices'])
        all_params.append((combo['t1'], combo['sl1'], combo['t3'], 
                          combo['sl2'], combo['t2'], combo['sl3']))
    
    n_combos = len(all_params)
    
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
    
    # Process with optimized Numba function
    sums, wins, losses, neithers, valid_mask = process_combinations_optimized(
        all_filtered_rows_flat, all_filtered_prices_flat,
        combination_starts, combination_lengths,
        all_params_array, t1_idx_dict, sl1_idx_dict, t3_idx_dict,
        sl2_idx_dict, t2_idx_dict, sl3_idx_dict,
        t1_idx_arrays, sl1_idx_arrays, t3_idx_arrays,
        sl2_idx_arrays, t2_idx_arrays, sl3_idx_arrays,
        how_many_final_parameters
    )
    
    # Add results to local_sublists
    for i, combo in enumerate(combination_batch):
        if valid_mask[i]:
            if how_many_final_parameters == 4:
                sublist_key = (
                    combo['volatility'], combo['ratio'], combo['adx28'], combo['adx14'], 
                    combo['adx7'], combo['zscore'], combo['rsi_type'], combo['t1'], 
                    combo['t2'], combo['sl1'], combo['sl2']
                )
            elif how_many_final_parameters == 6:
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


print("Optimized Grid Search module loaded - Numba + Hash Lookups + Multiprocessing") 