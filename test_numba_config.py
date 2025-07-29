import os
import numpy as np
from numba import jit, prange, set_num_threads, get_num_threads, config
import time

print(f"CPU cores available: {os.cpu_count()}")
print(f"Current Numba threads: {get_num_threads()}")

# Set to use all cores
import multiprocessing
num_cores = min(multiprocessing.cpu_count(), 16)  # Ensure we don't exceed system limits
set_num_threads(num_cores)
print(f"Numba threads after set_num_threads({num_cores}): {get_num_threads()}")

# Test simple parallel function
@jit(nopython=True, parallel=True)
def test_parallel_sum(arr):
    result = 0.0
    for i in prange(len(arr)):
        result += arr[i] * arr[i]
    return result

# Create test data
test_data = np.random.random(10000000).astype(np.float64)

# Time sequential vs parallel
start_time = time.time()
result = test_parallel_sum(test_data)
parallel_time = time.time() - start_time

print(f"Parallel computation result: {result}")
print(f"Parallel computation time: {parallel_time:.4f}s")
print(f"Expected high CPU usage during computation") 