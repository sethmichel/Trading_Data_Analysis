---
name: GPU Grid Search Implementation
overview: Implement a GPU-accelerated grid search system using CuPy, Numba, and RAPIDS. The system will load pre-processed static market/trade data into VRAM and iterate through strategy parameter batches to find the top 10 performing combinations.
todos:
  - id: impl-config
    content: Create config.py with parameter constants and paths
    status: completed
  - id: impl-memory
    content: Create memory.py for RMM initialization
    status: completed
  - id: impl-loader
    content: Implement data_loader.py with DLPack/CuPy conversion
    status: completed
    dependencies:
      - impl-config
      - impl-memory
  - id: impl-gen
    content: Implement strategy_generator.py using itertools and batching
    status: completed
    dependencies:
      - impl-config
  - id: impl-kernels
    content: Implement kernels.py with Numba CUDA logic
    status: completed
    dependencies:
      - impl-config
  - id: impl-main
    content: Implement main.py orchestration logic
    status: completed
    dependencies:
      - impl-loader
      - impl-gen
      - impl-kernels
---

# GPU Grid Search Implementation Plan

This plan details the construction of the GPU-accelerated grid search engine in `Grid_Search_Parameter_Tuning/gpu_implementation/`.

## Architecture

1.  **Memory Management (`memory.py`)**: Centralized RMM (RAPIDS Memory Manager) pool initialization to prevent memory fighting between Numba and CuPy.
2.  **Configuration (`config.py`)**: Stores file paths, GPU kernel settings (thread block sizes), and parameter definitions.
3.  **Data Loading (`data_loader.py`)**:

    -   Loads `market_data.parquet` and `trades.parquet` using `cudf`.
    -   Uses DLPack to zero-copy transfer data into `cupy` arrays.
    -   Returns these arrays which act as static "read-only" textures in VRAM.

4.  **Strategy Generator (`strategy_generator.py`)**:

    -   Defines the specific parameter lists provided (volatility, ratios, adx, targets, stops).
    -   Uses `itertools.product` to generate combinations.
    -   Yields "batches" of combinations (e.g., 50M at a time) to avoid timeouts and 32-bit indexing limits.

5.  **Kernels (`kernels.py`)**:

    -   A pure Numba `@cuda.jit` kernel.
    -   **Inputs**: Static market arrays, static trade arrays, and a batch of strategy parameters.
    -   **Logic**: For each strategy (thread), iterate through filtered trades, simulate entry/exit based on market data lookup, and compute total ROI.

6.  **Orchestrator (`main.py`)**:

    -   Initializes memory.
    -   Loads data once.
    -   Loops through strategy batches.
    -   Runs kernel -> `cupy.argpartition` (parallel sort) -> Saves top 10.
    -   Aggregates final results.

## Implementation Details

### Parameters

The grid search will iterate over these discrete values:

-   `volatilities`: `[0.0, 0.3, 0.4, 0.5, 0.6]`
-   `ratios`: `[0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0]`
-   `adx28s`: `[0, 20, 30]`
-   `adx14s`: `[0, 20, 30]`
-   `adx7s`: `[0, 20, 30]`
-   `targets`: `[0.2, 0.3, 0.4, 0.5]` (formatted as percentages in logic)
-   `stop_losses`: `[-0.5, -0.4, -0.3]`

### Data Flow

`Parquet Files` -> `cuDF DataFrame` -> `DLPack` -> `CuPy Array` -> `Numba Kernel`

## File Structure & Steps

### Step 1: Core Setup

-   **`config.py`**: Define constants, batch size, and the parameter lists.
-   **`memory.py`**: Create `initialize_memory()` function using `rmm.reinitialize`.

### Step 2: Data Loading

-   **`data_loader.py`**: Implement `load_data()` which returns a dictionary of CuPy arrays for `prices`, `trade_entry_idxs`, `trade_metrics` (volatility, adx, etc.).

### Step 3: Strategy Generation

-   **`strategy_generator.py`**: Implement a generator class that yields `(batch_size, 7)` arrays containing the parameter sets for the current batch.

### Step 4: Kernel Implementation

-   **`kernels.py`**: Implement `simulate_strategies_kernel`.
    -   Use `cuda.grid(1)` to map thread ID to strategy index.
    -   Filter trades: `if trade_vol < strat_vol: continue`.
    -   Simulation loop: Check `price[k] `against `entry_price * (1 + target/100)` and `entry_price * (1 + stop/100)`.

### Step 5: Main Loop

-   **`main.py`**:
    -   Call `memory.initialize_memory()`.
    -   `data = data_loader.load_data()`.
    -   Loop `batch` in `generator`:
        -   `results = cupy.zeros(batch_size)`
        -   `kernels.simulate_strategies_kernel[blocks, threads](..., results)`
        -   `top_10_idx = cupy.argpartition(results, -10)[-10:]`