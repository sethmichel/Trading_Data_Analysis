config
- THREADS_PER_BLOCK = 256  # Standard CUDA block size
- BATCH_SIZE = 1_000_000 

memory.py
- pool_allocator=True,
  initial_pool_size=None, # Use default (usually 1/2 GPU memory) or specify bytes
  maximum_pool_size=None

data loader.py
- we're not loading in 'Worst Exit Percent', we're checking market data instead