import rmm
import cudf

def initialize_memory():
    """
    Initialize RAPIDS Memory Manager (RMM) to create a shared memory pool.
    This prevents memory contention between Numba and CuPy/cuDF.
    """
    try:
        # Initialize RMM with a pool allocator.
        # This allocates a large chunk of VRAM upfront and manages it.
        # It is generally faster and prevents fragmentation.
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=None, # Use default (usually 1/2 GPU memory) or specify bytes
            maximum_pool_size=None
        )

        print("RMM Initialized successfully.")
        
    except Exception as e:
        print(f"Failed to initialize RMM: {e}")
        raise

