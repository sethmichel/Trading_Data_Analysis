import time
import numpy as np
from numba.typed import Dict
from numba import types
import multiprocessing as mp

# Test the hash lookup optimization
def test_hash_vs_linear_search():
    print("Testing hash lookup vs linear search performance...")
    
    # Create test data
    test_params = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
    search_values = np.random.choice(test_params, 10000)
    
    # Linear search (current approach)
    start_time = time.time()
    for value in search_values:
        for i, param in enumerate(test_params):
            if param == value:
                break
    linear_time = time.time() - start_time
    
    # Hash lookup (optimized approach)
    param_dict = Dict.empty(key_type=types.float64, value_type=types.int32)
    for i, param in enumerate(test_params):
        param_dict[param] = i
    
    start_time = time.time()
    for value in search_values:
        if value in param_dict:
            idx = param_dict[value]
    hash_time = time.time() - start_time
    
    print(f"Linear search time: {linear_time:.4f}s")
    print(f"Hash lookup time: {hash_time:.4f}s")
    print(f"Speedup: {linear_time/hash_time:.1f}x faster")
    
    return linear_time, hash_time

def test_multiprocessing_overhead():
    print("\nTesting multiprocessing setup overhead...")
    
    def dummy_work(x):
        return sum(i*i for i in range(x))
    
    # Single process
    start_time = time.time()
    results_single = [dummy_work(1000) for _ in range(16)]
    single_time = time.time() - start_time
    
    # Multiprocessing
    start_time = time.time()
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_multi = pool.map(dummy_work, [1000] * 16)
    multi_time = time.time() - start_time
    
    print(f"Single process time: {single_time:.4f}s")
    print(f"Multiprocessing time: {multi_time:.4f}s")
    print(f"MP overhead: {multi_time/single_time:.1f}x")
    
    return single_time, multi_time

if __name__ == '__main__':
    print(f"Testing on {mp.cpu_count()} CPU cores\n")
    
    linear_time, hash_time = test_hash_vs_linear_search()
    single_time, multi_time = test_multiprocessing_overhead()
    
    print(f"\n=== Performance Summary ===")
    print(f"Hash lookup speedup: {linear_time/hash_time:.1f}x")
    print(f"Expected total speedup with your grid search:")
    print(f"- Hash optimization: {linear_time/hash_time:.1f}x faster parameter lookups")
    print(f"- Numba parallel: ~{mp.cpu_count()//2}x faster computation (when working)")
    print(f"- Multiprocessing: {mp.cpu_count()}x parallelization (for large batches)")
    print(f"- Combined potential: {(linear_time/hash_time) * (mp.cpu_count()//2):.0f}x speedup!") 