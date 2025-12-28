data and use case
I have tons of trade logs (data on live trades, like entry time, metrics upon entry, ...), and second by second market data. I have a few parameters and want to know what combo of those parameters would result in the highest roi on this data. So, we'll use these parameters to filter the trade logs (example: volatility = 0.5 means we exclude all trades that had volatility < 0.5 at entry). So we need the roi over the trade history for each combination, and I want the top 10 combinations.

Additional parameters used are roi targets (exit at this roi no matter what) and roi stop losses.

It's currently written to run on a cpu in python, numpy, numba. it can do about 1.2 billion combination checks in 8 minutes. I want to move to this to a gpu to compute more combinations faster

new design
tech stack
- python + cupy
    - python is used because all the libraries it has access to
    - cupy remplaces numpy. we'll use it for the meshgrids
- rapids cuDF 
    - for zero copy to load a csv directly into my gpu
- rmm (rapids memory manager) 
    - to make 1 memeory pool that numba and cupy share instead of fight over
- dlpack
    - let's me pass cuDF to numba kernal w/o copying the data
- nvidia 4060 gpu (8gb vram, large L2 cache)
- modern intel cpu (2023 model), meaning it's capable of hard tasks

walk through
1. Setup: The Manager (RMM)
- Your thought: "We use rapids memory manager to operate the memory pool."
- Verdict: Correct.
- Why: Without this, when you hand data from cuDF to Numba, Python might try to copy it, double RAM usage, and crash. RMM creates one big pool that everyone shares.

2. Data Ingest: The "Static" Load (Crucial Change)
- Your thought: "Upload this csv as a batch... only as much as our vram can handle."
- The Correction: Do NOT batch the trade data. Load ALL of it once.
- Why:
    - Market data is smaller than you think. 5 years of minute data is only ~1M rows (~100MB). Even tick data fits in 8GB.
    - The Bottleneck is the Strategies (10 Million combinations), not the Data.
    - Architecture: Load the entire history into VRAM as a static, read-only "Texture". It never moves.

3. The Bridge: DLPack
- Your thought: "cuDF batch goes to dlPack which sends it to custom numba kernel. this kernel keeps data in optimal arrays that are in continuous memory. so each index of array 1 links up with the same index of array 2, and so on. it does this efficiently without copying the data."
- Verdict: Spot on.
- The Tech: You perform entry_array = numba.from_dlpack(cudf_df['entry'].to_dlpack()).
- Result: Numba now has a pointer to the exact memory address where RAPIDS loaded the CSV. Zero copying occurred.

4. The Kernel: Parallelism & The "Running Total"
- Your thought: "Each thread will test the combination over the whole data sample space, only updating 1 - running_total per combination. at the end it'll write this final running_total to vram as part of a cuPy array. so each thread writes this 1 number representing 1 combination
- Verdict: Perfect.
- The Visual:
    - Grid: 1 Block of threads = 1 Batch of Strategies (e.g., 250,000 strategies).
    - Thread 0: Takes Strategy[0] (Stop: 1%, Target: 2%). Loops over all 1M rows of trade data. Calculates final ROI. Writes 1.05.
    - Thread 1: Takes Strategy[1] (Stop: 1.1%, Target: 2%). Loops over all 1M rows. Writes 0.98.

5. Pruning: The "Parallel Sort" (Correction)
- Your thought: "We'll likely need 1 thread to prune the results so we keep just the top 10."
- The Correction: Never use "1 thread" on a GPU.
- The Fix: Use cupy.argpartition or cupy.argsort.
- Why: Sorting a list of 250,000 results takes a long time for 1 thread. cupy.argsort uses thousands of threads to sort the list instantly in parallel.
- The Flow:
    - Kernel finishes. You have an array Results of size 250,000.
    - Run top_indices = cupy.argpartition(Results, -10)[-10:].
    - Save those 10 indices and scores to a small CPU list.
    - Discard the rest.

6. The Loop: Streaming Strategies
- Your thought: "When that data batch is done, we send the next data batch."
- The Correction: When that Strategy Batch is done, we send the next Strategy Batch.
- The Loop:
    - Batch 1: Strategies 0 to 250k. (Run -> Prune -> Save Top 10).
    - Batch 2: Strategies 250k to 500k. (Run -> Prune -> Save Top 10).
    - ... Repeat until 10M strategies are done.
    - Final Step: Compare the "Top 10" from all batches to find the absolute winner.


Notes
- my gpu has a good sized L2 cache, this means it's likely that hot parts of my trade data can exist in the l2 cache. This means all threads hit the cache instead of vram (faster).
- use float32 instead of float64 for trade data

possible major issues: 
- issue: going over the 32-bit integer limit. this is likely to happen if we try to allocate a single array with more than 2 billion elements, or if the grid dimensions get too large.
    - solution: don't launch 1 kernel for x billion combinations; on windows, if 1 kernel takes longer than 2 seconds the os will kill the gpu driver thinking it froze. So, let's break the x billion combinations into 50 million combo kernel batches.
        - Python Loop: Iterate through your strategy parameter ranges.
        - Launch: Run kernel for batch N.
        - Result: Kernel returns results array for that batch.
        - Prune: cupy.argpartition the batch results, save top 10, discard the rest.
        - Repeat.This completely bypasses the 32-bit limit because no single grid or array ever exceeds manageable sizes.

- issue: in the old approach, we used masks to avoid if statements. but on a gpu, if statements are dangerous due to warp divergence. gpu threads run in groups of 32 warps, if thread A takes the if branch and thread b takes the else branch, both threads execute both branches - hurting performance. for example, we have a conditional "exit at roi target"; so this will happen.
    - solution: hard to avoid this. but we can group combinations with similar parameters together in teh same batch. they're likely to exit at similar times, thus minimizing divergence wait times.

file structure
├── config.py               # Constants, batch size, gpu settings (thread block dimensions)file paths, and hyperparameters
├── memory.py               # safety net so rmm is isolated. import rmm first in main.py. RMM setup and memory pool management
├── data_loader.py          # Loads static data (cuDF) & converts to Arrays (DLPack). this prepares data for the gpu. should also return pointers to the static read only data that stays in vram
├── strategy_generator.py   # should solve the 32-bit integer problem. Yields batches of similar parameter combinations. should use itertools or simple loops, and a generator pattern
├── kernels.py              # The raw Numba CUDA kernels (@cuda.jit). likely only @cuda.jit functions. Writes the resulting ROI to a pre-allocated results array. should be pure numba
└── main.py                 # The orchestrator (loops batches, prunes results). import memory.py first, call data_loader to load trade histroy into vram, loop through strategy_generator, after the loop sort the top 10 results to find the best combinations

Debugging: If the math is wrong, you check kernels.py. If you run out of memory, you check memory.py or config.py.

data structure
only use data for 
- for simplicity, only use trade logs and market data for 'SOXL', 'IONQ', 'MARA' rather than all ticker symbols
    - all_trade_logs.csv, contains all trade log history
    - cleaned_and_verified_market_data directory, contains various csv files of market data by date. each file has second by second market for 7 tickers. timestamps are in hour:minute:second format. all data is sorted by timestamp in ascending order (earliest part of the day is first, latest part of the day is last)
    - market data csv header: Ticker,Price,Val,Avg,Atr14,Atr28,Rsi,Volume,Adx28,Adx14,Adx7,Volatility Percent,Volatility Ratio,Time
        - remove these columns, we won't use them: Val,Avg,Atr14,Atr28,Rsi,Volume
    - usable trade log csv header (only use these): Date,Trade Id,Ticker,Entry Price,Exit Price,Entry Time,Exit Time,Worst Exit Percent,Entry Volatility Percent,Entry Volatility Ratio,Entry Adx28,Entry Adx14,Entry Adx7,Trade Best Exit Percent,Trade Percent Change
    - the market data for each date likely needs to split into 1 dataframe for each ticker. let's also reduce the tickers to just 3 'SOXL', 'IONQ','MARA'. 'Trade Percent Change' is the percent roi of the trade, 'Trade Best Exit Percent' is the best possible roi if we had exited at the best possible time. 'Trade Worst Exit Percent' is the largest loss we had to endure over the course of the trade. In the grid search, we can use the parameters to filter those trades, then use the 'Trade Percent Change' metric to find the roi of each combination

- data pre-processing strategy
    - Market Data: 
        - Will become one giant, continuous array of Prices (float32) sorted by Ticker -> Time.
        - Load & Filter: Loop through all CSV files. Filter for Ticker in ['SOXL', 'IONQ', 'MARA'].
        - Inject Date: Your CSV has Time (HH:MM:SS) but not Date. You must parse the filename to get the Date and combine it with Time to create a full Datetime object. all file names are in this exact format: 'Raw_Market_data_MM-DD-YYYY.csv'
        - Select Columns: remove Val,Avg,Atr14,Atr28,Rsi,Volume columns to save vram
        - Sort: Sort by Ticker (primary) and Datetime (secondary).
        - Save: Save as a single compressed file (Parquet is best)

    - Trade Log: Will gain a new column: MarketDataStartIndex. This tells the GPU: "For Trade #105, start reading prices at Index 5,002,100."
        - Filter: Keep only the "usable" columns I listed. Filter for the 3 tickers.
        - Clean Times: convert any timestamps (Entry Time, Exit Time) to full Datetime objects from hour:minute:second format.
        - The "Pointer" Logic (Crucial):
            - Load the sorted market_data_clean.
            - Use np.searchsorted (or pandas.merge_asof) to find the index in the market data where the Entry Time occurs for that specific ticker.
            - Store this as MarketDataStartIndex.
            - Calculate MarketDataEndIndex (based on Exit Time) or MaxDurationSeconds so the GPU knows when to stop looking if no stop/target is hit.
        - Map Tickers: Convert 'SOXL', 'IONQ', 'MARA' to integers 0, 1, 2.

    - Handling The Metrics (Conflict Resolution)
        - issue: If you set a target roi of +5% and a stop loss of -2%, and the trade logs say Best was +10% and Worst was -5%, did you win or lose? You don't know which happened first
        - solution: use the market data
            - The GPU kernel starts at MarketDataStartIndex.
            - It loops: price[i], price[i+1], ...
            - If price hits Stop -> Result = Stop Loss.
            - If price hits Target -> Result = Target.
            - If price reaches MarketDataStartIndex + Length (Original Exit Time) without hitting either -> Result = Trade Percent Change.

    - summary
        - Preprocessing Script: Write a python script to generate market_data.parquet and trades.parquet using the logic above. Do this once on CPU.
        - Market Data File: Should just be a massive array of floats (Price). You might not even need the Ticker/Time columns on the GPU if your pointers are correct! (Just one array of 100M prices).
        - Trade File: Contains your filter parameters (Entry Volatility, etc.) and the StartIndex pointer.







option 3
===
smart optimization instead of grid search
- grid is O(n^d). so 10 paras basically kills me. this finds promising areas instead
sub-option 1) bayesian optimization. probability model that guesses where the profit is likely to be
    - botorch
    - weak if lots of fake local max's

sub-option 2) genetic algorithms: spawn 1000 random strats, kill the bottom 50%, mix the top 50%
    - DEAP (cupy/pytorch hooks) or EvoTorch
    - evo is made for this exact thing on pytorch tensors
    - best if opimization space is jagged or non convex (lots of fake local max's that trick bayesian)

sub-option 3) TPE (tree structured parzen estimator)
    - same as optuna. paras are indepdendent vars and models probability of parameters given a score
    - optuna (controller) + dask (scheduler)
    - Optuna is the industry standard for usability. It’s not "native GPU" for the search logic itself, but its Pruning (Asynchronous Successive Halving) is incredible. It will start 100 trials and kill 90 of them halfway through if they look bad.


Operating notes
- I should mix days together instead of sequential. a combo could suck 1 month but be good another month. if I check 1 month I'm getting back data. so mix the days up. this also lets me drop bad combos early


How to detect "Fake Local Maximums"
- actually I think I'll just skip baysian, it assumes smoothness and I don't like it. trading data is usually jagged apparently






Parameters
volatilities = [0.0,0.3,0.4,0.5,0.6]       # Entry Volatility Percent. this is a custom metric not present in market data (round((atr14 / row['price']) * 100, 2))
ratios = [0.0,0.2,0.5,0.7, 0.8, 0.9, 1.0]  # Entry Volatility Ratio. this is a custom metric not present in market data (round(atr14 / row['atr28'], 2))
adx28s = [0,20, 30]
adx14s = [0,20, 30]
adx7s = [0,20, 30]

target_1 = [0.2,0.3, 0.4, 0.5]             # this is the target roi percent (0.5 = 0.5%). if we reach this then force the trade to exit
stop_loss_1 = [-0.5,-0.4,-0.3]             # this is the max stop loss percent (-0.3=-0.3%). if we reach this then force the trade to exit