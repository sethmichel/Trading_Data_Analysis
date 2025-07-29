import multiprocessing as mp
import numpy as np
import time
import os

def simple_computation_worker(data_chunk):
    """Simple worker function that does intensive computation"""
    result = 0.0
    for i in range(len(data_chunk)):
        # Simulate your parameter lookup and computation
        for j in range(1000):  # Simulate complex nested loops
            result += data_chunk[i] * j * 0.1
    return result

def test_multiprocessing_performance():
    print(f"CPU cores available: {os.cpu_count()}")
    
    # Create test data
    data_size = 100000
    test_data = np.random.random(data_size).astype(np.float64)
    
    # Split data into chunks for each core
    num_cores = os.cpu_count()
    chunk_size = len(test_data) // num_cores
    chunks = [test_data[i:i+chunk_size] for i in range(0, len(test_data), chunk_size)]
    
    # Test multiprocessing approach
    start_time = time.time()
    
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(simple_computation_worker, chunks)
    
    multiprocess_time = time.time() - start_time
    total_result = sum(results)
    
    print(f"Multiprocessing result: {total_result}")
    print(f"Multiprocessing time: {multiprocess_time:.4f}s")
    print(f"Should use 100% CPU during computation")
    
    return total_result

if __name__ == '__main__':
    test_multiprocessing_performance() 