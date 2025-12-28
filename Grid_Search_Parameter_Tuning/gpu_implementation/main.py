import time
import cupy as cp
import pandas as pd
import numpy as np
from numba import cuda

import config
import memory
import data_loader
import strategy_generator
import kernels

def main():
    print("Starting GPU Grid Search...")
    start_time = time.time()
    
    # 1. Initialize Memory Manager
    memory.initialize_memory()
    
    # 2. Load Data
    try:
        market_prices, trades = data_loader.load_data()
    except FileNotFoundError as e:
        print(e)
        return

    # 3. Setup Processing Loop
    top_results = [] # List to store (score, params_dict)
    
    batch_gen = strategy_generator.generate_strategy_batches()
    
    # Kernel Launch Configuration
    threads_per_block = config.THREADS_PER_BLOCK
    
    print("\n--- Starting Batch Processing ---")
    
    total_strategies_processed = 0
    
    for batch_params_gpu, start_global_idx, end_global_idx in batch_gen:
        batch_size = batch_params_gpu.shape[0]
        blocks = (batch_size + threads_per_block - 1) // threads_per_block
        
        # Allocate results array for this batch
        results_gpu = cp.zeros(batch_size, dtype=cp.float32)
        
        # Launch Kernel
        kernels.simulate_strategies_kernel[blocks, threads_per_block](
            market_prices,
            trades['start_index'],
            trades['end_index'],
            trades['entry_price'],
            trades['volatility'],
            trades['ratio'],
            trades['adx28'],
            trades['adx14'],
            trades['adx7'],
            batch_params_gpu,
            results_gpu
        )
        
        # Synchronize to ensure kernel finishes before pruning
        cuda.synchronize()
        
        # Pruning: Find Top 10 in this batch
        # If batch is smaller than 10, take all
        k = min(10, batch_size)
        
        # argpartition moves the k largest elements to the end (unsorted)
        # We use -k to get the indices of the k largest elements
        top_k_indices_gpu = cp.argpartition(results_gpu, -k)[-k:]
        
        # Extract values
        top_k_scores_gpu = results_gpu[top_k_indices_gpu]
        top_k_params_gpu = batch_params_gpu[top_k_indices_gpu]
        
        # Transfer to CPU
        top_k_scores = top_k_scores_gpu.get() # numpy array
        top_k_params = top_k_params_gpu.get() # numpy array
        
        # Store in list
        for score, params in zip(top_k_scores, top_k_params):
            # Convert params array to dictionary for readability
            p_dict = {
                'volatility': params[0],
                'ratio': params[1],
                'adx28': params[2],
                'adx14': params[3],
                'adx7': params[4],
                'target': params[5],
                'stop_loss': params[6]
            }
            top_results.append({'score': float(score), 'params': p_dict})
            
        total_strategies_processed += batch_size
        print(f"Processed batch {start_global_idx}-{end_global_idx}. Total: {total_strategies_processed}")
        
        # Explicitly free memory (optional given RMM, but good practice in loop)
        del results_gpu
        del top_k_indices_gpu
        del top_k_scores_gpu
        del top_k_params_gpu
        del batch_params_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # 4. Final Aggregation
    print("\n--- Aggregating Results ---")
    
    # Sort all collected top candidates by score descending
    top_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Keep absolute Top 10
    final_top_10 = top_results[:10]
    
    # 5. Output
    print(f"\nTop 10 Strategies found in {time.time() - start_time:.2f} seconds:")
    
    output_rows = []
    for rank, res in enumerate(final_top_10, 1):
        row = res['params']
        row['rank'] = rank
        row['total_roi'] = res['score']
        output_rows.append(row)
        print(f"Rank {rank}: ROI={res['score']:.4f}, Params={res['params']}")
        
    df_out = pd.DataFrame(output_rows)
    # Reorder columns
    cols = ['rank', 'total_roi'] + config.PARAM_NAMES
    df_out = df_out[cols]
    
    df_out.to_csv(config.OUTPUT_FILE, index=False)
    print(f"\nSaved results to {config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()

