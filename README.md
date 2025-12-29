# Trade Data Analysis Pipeline

**This is the backend of the quant platform. it trains the models, validates data, and processes strategy simulations**

This repo pairs with the manual trading system, the auto trading system, and the data gathering system to form a miniture quant platform. The platform is meant to gather data locally and remotly, feed it into trading bots/or manual traders, and then give it to this backend to do longer processing.

**System Requirements:**
- 16 GB RAM, 
- Consumer grade gpu (8gb of gpu VRAM, decent size L2-cache)

## Summary

The main use of this program is ingesting new market data and user trade data daily into a processing pipeline

**Pipeline Functions:**
- Validate/clean the data
- Generate detailed summaries of the trades
- Retrain the statistics models and run diagnostics on them
- Optionally retrain the XG feature importance model
- Optionally rerun the grid search
- Save metadata of everything

**Example Pipeline Useage**
1) User will have a new day of market data and trade data in the database, and also in local csv files. The user starts this program.
2) It will validate/clean the csv files and database data, then update the metadata tables which control what files can and can't be used in processing. Then it makes a bulk summary file and uploades this to the database
3) Next, it retrains the statistics models, and saves their diagnostic results under an updated version id in the database
4) Optionally, the user can run XG boost feature model and review feature importances visuals
5) Optionally, the user can run grid search on the data

**Key Features**
- Grid search: Using market data and trade history, simulate billions of combinations of parameters to see what their ROI would have been over my trade history. The stress test shows this operates at about 5 trillion operations per second (5 TFLOPS). I estimate I can get 50% more performance out of this but I have no need to do so at the moment.

- XG Boost Feature Importance: Ml model that outputs charts, graphs, txt files detailing relationships between data features and their impact on ROI

- Statistic Models: See the models section, but these are models trained specifically for predicting ROI, stop loss, and success probability based on market conditions. They all run live with the trading system every 2 minutes.

## Validation/Cleaning

### Expected Data Format

**Market Data:**
- Data gathering computers record market data every **1 second** for the whole market open duration for each ticker
- Approximately **23,400 lines per ticker**
- Data is recorded to the remote database and also in CSV form locally (as a backup)
- Currently records 12 data points per ticker

**Trade Data:**
- Auto bot trade logs OR manual trade logs from the ThinkOrSwim platform

### Database Structure

The database maintains metadata tables:
- Market Data & Trade Data Validation Tables: Tracks validation status, errors, etc. for each date
- Partitioned Market Data & Raw Trade Logs Tables: storage for the raw data
- Trade Summary Table: Combined trade data + market data + algorithms summary for each trade

**Features:**
- Choose whether to revalidate erroneous info after fixing the data
- If data has errors, the rest of the system won't use that date's data
- Data remains in the database mixed with clean data, but the metadata tables stop is from being used anywhere until fixed

### Validation Checks

Examples of things checked:
- File dates and contents are normalized to remove useless stuff (like canceled trades in trade logs)
- Market data undergoes extensive tests:
  - Time gaps in the data
  - Missing values
  - Inconsistencies in the frequency of data between tickers
  - And more...
- Some data is corrected as it's validated (like the filename) but usually the user will have to fix it

## Generating Trade Summaries

After validation, we combine the market data with the trade logs to generate an **extremely detailed summary** of each trade.

Beacuse we have the second by second market data:
- See entry conditions or conditions at any point in the day after that trade (even after the trade closed)
- Record ROI values for different small strategy tweaks to compare side by side
- Simulate strategy changes: See what the trade would have done with minor tweaks (lower stop loss, later exit point, etc.)
- Record ROI in intervals: Track ROI percent in 0.01 intervals as it goes up and down (record ROI changes in a list)
- Can have hundreds of relevant columns using the market data

## The Models

### Model 1: Optimal Exit Point Model
> Model: LinearGAM and GammaGAM (overhauled to handle heteroscedasticity bias)

Model assumes all trades are stable and gives ROI prediction assuming the trade succeeds

Specifics
- X features: Minutes since market open, volatility percent (custom metric)
- Y feature: Max ROI of the trade until the trade falls to the exit point
- Note: Does NOT go until the user exited the trade, but rather until an exit point we define
- Sampling: Take a data sample every 10 minutes of a trade (with many adjustments to fix bias)
- Filtering: Only using 'good' trades (ones that fail quickly are removed)

---

### Model 2: Optimal Stop Loss Model (Emergency Exit Model)
> Model: CatBoostRegressor

This model seeks to predict the best stop loss to use based on current market conditions, such that the trade has the highest chance possible to reach +0.6% ROI.

Specifics
- Data Gathering: Data is gathered from a grid search of various possible stop losses
- Simulation: Stop losses are simulated over the trade history using market data and recorded as 1 or 0
- Goal: Find the least risky stop loss to use for each trade
- Input Values: Minutes since market open, and volatility percent (scaled relative to each other, then fed into StandardScaler)
- Y Feature: Not scaled
- Status: Has the least amount of work put into it because it actually predicts pretty much what my normal stop loss values are. I use 2 values for different situations and this model also predicts those values. Focused on other models until I had more data.

---

### Model 3: Success Probability Prediction Model
> Model: LogisticGAM

Predict, based just on the current market conditions, the success probability of a trade. Success is defined as reaching an ROI of +0.6%. You can run this every 1 minute during a trade if you want, and you can pair it with a "survival model" for better results.

Specifics

1. **Preprocessing:**
   - Scale input features so they're relative to each other
   - Use StandardScaler or RobustScaler on them

2. **Model Configuration:**
   - Use LogisticGAM model with low spline count to reduce overfitting
   - Tune lambda really harsh for this dataset
   - Most y data is a fail, so we have to weigh the classes in each fold

3. **Cross-Validation:**
   - Use StratifiedKFold
   - In each k-fold, grid search a lambda range
   - Pick the version with the best:
     - Brier score
     - AUC value
     - Binomial deviance

4. **Goals:**
   - Minimize Brier score
   - Maximize mean ROC and AUC

---

### Diagnostics We Run on the 3 Models

Main diagnostics are saved in the database. It does different things for each model.

**Example (Optimal Exit Model):**
- Minimize the standard deviation of residuals
- Make residuals group around 0 in a random cluster with no pattern
- Get low correlation between fitted values and residuals
- Get low variance ratio

**Outputs:** Graphs, charts, and txt files are generated.

---

### How These Models Are Used

Both the **auto trading bot** and the **manual trading system** use these models.

**Usage:**
- **Success Probability Model:** Runs each minute and is used to decide if we should enter a trade when our strategy lines up
- **Stop Loss Model:** Tells you the optimal emergency exit point (stop loss) to use
- **Optimal Exit Model:** Runs every 2 minutes for a entrered trade to tells you the exit point based on current market conditions

**Frequency:** All these models can run once or run each X minutes while in a trade. You ideally want them continuously running to handle changing market conditions.

## XG Boost Feature Importance

### Purpose

To show correlations between data points in as many ways as possible. Its output isn't text metrics but rather many graphs and charts.

### Output Categories

Results cover these categories
1. Overview
2. Interactions and Ranges
3. Correlations
4. Datapoint Analysis
5. Classification Report

### Current Output

- **21 visuals** and **4 txt files**
Currently it outputs 21 visuals and 4 txt files. Changing what features are used for what visuals is easy, and you could generate over 100 visuals if you wanted to. For example, the waterfall visual is extremely information dense, and you can generate additional smaller shap waterfall plots for individual samples that seem interesting.

## GPU Grid Search

### Summary

Wrote initially for the CPU, later rewrote optimally on the GPU

**Search Modes:**
1. Using each parameter as a lower bound
2. Using parameters as lower and upper bounds
3. **(Most Important)** Doing 1 grid search per time range so you get different values for different times of the day

Data
- Trade logs: real logs for actual tests, machine generated logs for stress test
- Market data: second-by-second, cleaned

Tech stack
- Python + cuPy
    - python is used because of the data science tech stack it has access to
    - cupy remplaces numpy. we'll use it for the meshgrids
- Rapids suit
   - cuDF
      - for zero-copy to load a csv directly into my gpu
   - rmm (rapids memory manager) 
      - to make 1 memeory pool that numba and cupy share instead of fight over
- dlpack
    - let's me pass cuDF to numba kernal w/o copying the data (zero-copy)

Hardware
- RTX 4060 gpu (8gb vram, decent size L2-cache, ~3000 threads)
- intel 14th gen cpu (8 cores, so 16 possible threads)
- 16gb ram

Basic Strategy
- RMM optimzes the memory pool and stops tools from competing over it 
- cuDF and dlPack make it so python won't re-copy data to vram
- Market data can be optimized and fit inside the L2-cache
- Keep data in optimized arrays with their array indexes lined up with each other (continuous memory)
- Custom kernel written in Numba makes full use of these design choices, escpecially the L2-cache

Technical Walk Through
1. Setup: The Manager (RMM)
   - We use rapids memory manager to operate the memory pool. Without this, when you hand data from cuDF to Numba, Python might try to copy it, double RAM usage, and crash. RMM creates one big pool that everyone shares.

2. Data Ingest: The "Static" Load (Crucial Change)
   - We don't batch trade data, we upload the csv at once.
      - Market data is actually small. our 8gb vram is plenty, and the l2 cache is probably enough
      - Load the entire history into the L2 cache as read-only. It never moves.

3. The Bridge: DLPack
   - cuDF batch goes to dlPack which sends it to custom numba kernel. this kernel keeps data in optimal arrays that are in continuous memory. so each index of array 1 links up with the same index of array 2, and so on. 
   - Numba will have a pointer to the exact memory address where RAPIDS loaded the CSV. Zero copying occurred.

4. The Kernel: Parallelism & The 1 Number Strategy Result
   - Each thread will test the combination over the whole data sample space, only updating a single running_total per combination. At the end it'll write this final running_total to vram as part of a cuPy array. so each thread writes this 1 number representing 1 combination

5. Pruning: The Parallel Sorting
   - Use cupy.argpartition to use all threads to prune the results in parallel. So when we have 250,000 results we prune them in parallel until we have 10 results. this saves vram.

6. The Loop: Streaming Strategies
   - When that data batch is done, we send the next data batch
      - Batch 1: Strategies 0 to 250k. (Run -> Prune -> Save Top 10).
      - Batch 2: Strategies 250k to 500k. (Run -> Prune -> Save Top 10).
      - ... Repeat until 10M strategies are done.
      - Final Step: Compare the "Top 10" from all batches to find the absolute winner.

### Depreceated CPU Version - What It Does (Basic)

This algorithm is nearly impossible to speed up now without a total rewrite in an extremely customized design. It is **hyper optimized** for speed and **hyper optimized** for space.
It has special preprocessing to covert a ton of the code from O(n) time to O(1). 

**Basic Approach:**
- Written in Python (python is a slow language)
- Very particular pre-processing before algorithm starts
- Convert everything into numpy arrays (which are C)
- Use numpy broadcasting/vectorization to compute things astronomically faster than Python
- For extremely heavy sections, convert from C code (numpy) into machine code (numba Python library)
- Go through each section of the algorithm and get their times down as low as possible through things like SIMD optimizations

1. **User Chooses Promising Features:**
   - Select features from market data
   - Make a list of common ranges that data goes in
   - **Example:** Volatility commonly has these values: `[0.0, 0.3, 0.4, 0.5, 0.6]`
   - This is computation explosion if you pick too many values

2. **Simulation:**
   - Simulate what would've happened over your trading history if you had used **every possible combination** of values in all your lists
   - Mode 1: Simulate using lower bound limits
   - Mode 2: Scan ranges (e.g., volatility 0.2-0.5)

3. **Stop Loss and Targets:**
   - Built to include various stop loss and targets combinations
   - As a trade progresses:
     - If it hits target 1 → move stop loss to stop loss position 2
     - If it hits target 2 → move stop loss to stop loss position 3
   - You can turn this off

4. **Output:**
   - Writes detailed reports on its **top 10 findings**
   - These are the 10 combinations of parameters you could have used to get the highest possible net ROI over your trade history

5. **Mode 3 - Time Ranges:**
   - Takes **MUCH longer** but does 1 grid search per time range
   - **Critical:** You can find the optimal parameter combination for different periods of the day, which is much more useful


