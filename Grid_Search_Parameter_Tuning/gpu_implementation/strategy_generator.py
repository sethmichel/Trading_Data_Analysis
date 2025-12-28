import itertools
import cupy as cp
import numpy as np
import config

def generate_strategy_batches():
    """
    Generator that yields batches of strategy parameters.
    Returns:
        tuple: (batch_params_gpu, start_index, end_index)
        - batch_params_gpu: CuPy array of shape (batch_size, 7)
        - start_index: Global index of the first strategy in this batch
        - end_index: Global index of the last strategy in this batch
    """
    
    # Extract parameter ranges from config
    ranges = [config.PARAM_RANGES[key] for key in [
        'volatilities', 'ratios', 'adx28s', 'adx14s', 'adx7s', 'targets', 'stop_losses'
    ]]
    
    # Create the full product iterator
    # param_names = ['volatility', 'ratio', 'adx28', 'adx14', 'adx7', 'target', 'stop_loss']
    strategy_iterator = itertools.product(*ranges)
    
    batch_size = config.BATCH_SIZE
    batch_buffer = []
    
    global_idx = 0
    
    for strategy in strategy_iterator:
        batch_buffer.append(strategy)
        
        if len(batch_buffer) >= batch_size:
            # Convert to numpy then cupy (efficient bulk transfer)
            batch_np = np.array(batch_buffer, dtype=np.float32)
            batch_gpu = cp.array(batch_np)
            
            yield batch_gpu, global_idx, global_idx + len(batch_buffer)
            
            global_idx += len(batch_buffer)
            batch_buffer = []
            
    # Yield remaining
    if batch_buffer:
        batch_np = np.array(batch_buffer, dtype=np.float32)
        batch_gpu = cp.array(batch_np)
        yield batch_gpu, global_idx, global_idx + len(batch_buffer)
        global_idx += len(batch_buffer)

def get_total_combinations():
    ranges = [config.PARAM_RANGES[key] for key in [
        'volatilities', 'ratios', 'adx28s', 'adx14s', 'adx7s', 'targets', 'stop_losses'
    ]]
    total = 1
    for r in ranges:
        total *= len(r)
    return total

if __name__ == "__main__":
    print(f"Total combinations: {get_total_combinations()}")
    for batch, start, end in generate_strategy_batches():
        print(f"Batch shape: {batch.shape}, Range: {start}-{end}")
        print("First strategy in batch:", batch[0])
        break

