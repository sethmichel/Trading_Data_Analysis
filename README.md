# Trade Data Analysis Pipeline

**This is the backend of the quant platform. it trains the models and validates the data.**

This repo pairs with the manual trading system, the auto trading system, and the data gathering system to form a miniture quant platform. The platform is meant to gather data locally and remotly, feed it into trading bots/or manual traders, and then give it to this backend to do longer processing.

## ⚠️ WARNING

The Grid search simulates **billions** of parameters in approximately **7 minutes** per billion, and has to organize it in complex string keys while doing extensive compute. It will max your CPU and your RAM. You **must close everything else** on your computer to give this as many resources as possible. The more you search, the more time your computer is maxed out and heating up.

It did **1.3 billion combinations** of parameters in approximately **8 minutes** (last time I ran it). 

**System Requirements:**
- Tuned for: 16 GB RAM, Intel i5 14th gen desktop CPU (the more cores the better)
- To re-tune: Change the batch size, pruning frequency, or slow it down if your CPU can't handle it
- Alternative: Run it on a GPU (not yet implemented)
- Note: It you tune it for a lot more ram than you have it can crash your computer (or you can code a check to stop this).

---

## Summary

You run this when you have new data to validate. The pipeline will move data around in CSV form, read/write to the database and record the status of the data.

**Pipeline Functions:**
- Validate/clean the data
- Generate detailed summaries of the trades
- Retrain the 3 statistics models and run diagnostics on them
- Optionally retrain the XG feature importance model
- Optionally rerun the grid search


**Example Useage**
1) User will have a new day of market data and trade data in the database, and also in local csv files. The user starts this program.
2) It will validate/clean the csv files and database data, then update the metadata tables which control what files can and can't be used in processing. Then it makes a builk summary file and uploades this to the database
3) Next, it retrains the statistics models, and saves their diagnostic results under an updated version id in the database
4) Optionally, the user can run XG boost feature model and review feature importances visuals
5) Optionally, the user can run grid search on the data

---

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

---

## Generating Trade Summaries

After validation, we combine the market data with the trade logs to generate an **extremely detailed summary** of each trade.

Beacuse we have the second by second market data:
- See entry conditions or conditions at any point in the day after that trade (even after the trade closed)
- Record ROI values for different small strategy tweaks to compare side by side
- Simulate strategy changes: See what the trade would have done with minor tweaks (lower stop loss, later exit point, etc.)
- Record ROI in intervals: Track ROI percent in 0.01 intervals as it goes up and down (record ROI changes in a list)
- Can have hundreds of relevant columns using the market data

---

## The Statistics Models

**Notes:** 
1. I go into so much detail for these models because I don't want to try to remember them in case I edit them again. Also, most of the logic used in the optimal exit point model can also be applied to the other models; You don't have to read all of them

2. Because we have so much second-by-second market data, we can use whatever input features we want and can calculate things that the trade logs would never tell us.

3. All these models use input features **volatility percent** and **minutes since market open**. These 2 variables are dependent on each other, so ideally use joint relationships for the model.

---

### Model 1: Optimal Exit Point Model
> Model: LinearGAM/GammaGAM (overhauled to handle heteroscedasticity bias)

#### How It Works

Specifics
   - X features: Minutes since market open, volatility percent (custom metric)
   - Y feature: Max ROI of the trade until the trade falls to the exit point
   - Note: Does NOT go until the user exited the trade, but rather until an exit point we define
   - Sampling: Take a data sample every 10 minutes of a trade (with many adjustments to fix bias)
   - Filtering: Only using 'good' trades (ones that fail quickly are removed)
   - Purpose: Model assumes all trades are stable and gives ROI prediction assuming the trade succeeds

1. x features are minutes since market open, and volatility percent (volatility percent is custom metric I use), y feature  is the max roi of the trade until the trade falls to the exit point. so it does NOT go until the user exited the trade, but rather until an exit point we define. We take a data sample every 10 minutes of a trade (we do a lot of adjustements to fix bias from this). Also, it's only using 'good' trades, so ones that fail quickly are removed. the model is meant to assume all trades are stable and give the roi prediction assuming the trade succeeds, including failed trades would lower the estimate.

2. The raw data is extracted, both x features are scaled so they're the same approximate range before having StandardScaler applied to them (minutes can be 300 but volatility will never be over 2). Then the y data has a sevear skew bias we can fix with either a GammaGAM model instead of a linearGAM model, or a linearGAM model which scales the data with np.log1p. Gamma handles this interally, but np.log1p introduces a strong back transformation bias which we deal with a little later via duan smearing.

2. Next we weigh the data to prevent longer trades (more data samples per trade) from dominating. The data is 'group    
weighted'. (solves issue of 1 hour long trade vs 2 minute long trade)
 
3. (lam means lambda, and it's not a 1 it's a L) Next, because the y distribution of the data is right skewed and we ideally want all samples of the same trade grouped together we use StratifiedGroupKFold grouped by trade_id. This is used to select the smoothing parameters (lam) which is something like this for example: np.logspace(-3, 3, num=8). Original unscaled y values are separated into folds and we find the best MAE and RMSE values by comparing each fold against the validation set. We pick the lam that minimizes the MAE. The process of finding this lam is the raw x values of each fold are scaled and the scaled y values for this fold are obtained, then a model is trained on this folds scaled x, scaled y data, then (only if we used np.log1p as I mentioned earlier) we find the duan smearing factor to correct the back transformation bias we get from the scaled y values, we can use this smear variable to get non-back-transformation bias predictions from the model (prediction is on log scale, but smearing with exp(prediction) brings it to the normal scale). We then get the MAE and RMSE values by converting the prediction value back to the original scale (so we calculate these on the original scale, not log scale). After we do this for each fold we find the average MAE and RMSE across all folds. This results in the best lam and the best cv report
	
4. With the best lam, we can train the actual model and fit it using the weights

5. Only if we used np.log1p: Now, with the trained model we finally compute the actual duan smearing factor for the full training set. We use this smearing factor to correct predictions when we use the model on live data

6. save the scaler, smearing factor (if we used np.log1p), model, any relevant tradining data in json files.

7. Run (multiply prediction by smear if we used np.log1p):
		 x_scaled = scaler.transform([[scaled_minutes_since_open, scaled_volatility_percent]])
         actual_prediction = np.exp(gam.predict(x_scaled))[0] - 1.0
---

### Model 2: Optimal Stop Loss Model (Emergency Exit Model)
> Model: CatBoostRegressor

#### Overview

This model seeks to predict the best stop loss to use based on current market conditions, such that the trade has the highest chance possible to reach +0.6% ROI.

#### How It Works

- Data Gathering: Data is gathered from a grid search of various possible stop losses
- Simulation: Stop losses are simulated over the trade history using market data and recorded as 1 or 0
- Goal: Find the least risky stop loss to use for each trade
- Input Values: Minutes since market open, and volatility percent (scaled relative to each other, then fed into StandardScaler)
- Y Feature: Not scaled

- Status: Has the least amount of work put into it because it actually predicts pretty much what my normal stop loss values are. I use 2 values for different situations and this model also predicts those values. Focused on other models until I had more data.

---

### Model 3: Success Probability Prediction Model
> Model: LogisticGAM

#### Overview

Predict, based just on the current market conditions, the success probability of a trade. Success is defined as reaching an ROI of +0.6%.

**Usage:** You can run this every 1 minute during a trade if you want, and you can pair it with a "survival model" for better results.

#### How It Works

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

### Main Bottleneck of These Models

**Issue:** Trade data, not market data. I simply don't have enough raw trades to train these models better.

**Possible Solution:** Use the auto trading bot to backtest trading on 100% of my market data from market open to market close. This would likely 3x my dataset at least and likely be silver quality data (as opposed to gold quality that human made trades are)

---

### How These Models Are Used

Both the **auto trading bot** and the **manual trading system** use these models.

**Usage:**
- **Success Probability Model:** Runs each minute and is used to decide if we should enter a trade when our strategy lines up
- **Stop Loss Model:** Tells you the optimal emergency exit point (stop loss) to use
- **Optimal Exit Model:** Runs every 2 minutes for a entrered trade to tells you the exit point based on current market conditions

**Frequency:** All these models can run once or run each X minutes while in a trade. You ideally want them continuously running to handle changing market conditions.

---

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
---

## Grid Search

### Summary

**Search Modes:**
1. Using each parameter as a lower bound
2. Using parameters as lower and upper bounds
3. **(Most Important)** Doing 1 grid search per time range so you get different values for different times of the day

### Performance

This algorithm is nearly impossible to speed up now without a total rewrite in an extremely customized design. It is **hyper optimized** for speed and **hyper optimized** for space.

**Performance Comparison (1.3 billion checks):**
- Base Python: **Well over 1 day**
- Base C code: **Likely over 8 hours**
- My Python → C → Machine code design (without hyper optimizations): **Over an hour**
- **My optimized version: ~8.5 minutes**

**Last Run:** Took about 8.5 minutes and searched 1.3 billion parameter combinations.

### How It Works

**Basic Approach:**
1. Written in Python (python is a slow language)
2. Very particular pre-processing before algorithm starts
3. Convert everything into numpy arrays (which are C)
4. Use numpy broadcasting/vectorization to compute things astronomically faster than Python
5. For extremely heavy sections, convert from C code (numpy) into machine code (numba Python library)
6. Go through each section of the algorithm and get their times down as low as possible through things like SIMD optimizations

### What It Does (Basic)

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

### What It Does (Complex)

#### 1. Data Preprocessing

- Preprocess data to make a complex dictionary which breaks apart into many numpy arrays
- These arrays contain indexes for all trades that each target and each stop loss value hits
- **This converts TONS of compute areas from O(n) time to O(1) time**

#### 2. Masks / MeshGrid

- **Pre-process boolean masks** for all valid parameter combinations
   - Lets me sometimes convert comparisons from O(n) to O(1) time, and lets me utilize numba for parallel processing, however is a bit slower in other places but worth it for memory savings
   - Store and pass around boolean masks instead of lists (~2 KB instead of 8 Bytes). At this scale, it matters a lot
   - Overall: Largest memory savings technique used and Simplifies things a lot

#### 3. Parallelization - Will Max Out Your CPU

- **Numba Configuration:**
   - Set to use all available CPU cores
   - @jit sets up compiler to handle parallel processing

- **SIMD Usage:**
   - Data is morphed such that I can take advantage of SIMD in my CPU
   - If data is in continuous memory (like numpy arrays but not Python lists), it can do **8 operations at once** instead of 1

- **NumPy Vectorization / Broadcasting:**
   - So efficient that it's basically parallel. It processes the whole array "at once"

#### 4. Batch Processing & Pruning - Will Max Out Your RAM

- **Batch Processing & Pruning:**
   - Only processes X many combinations per batch so we don't run out of RAM (2.5 million in my case)
   - Each batch is processed and the top results are saved
   - Keys for processed batches and data are still very large so we prune out the list every so often (when we reach 400,000 in my case)

#### 5. Uses Numpy and Machine Code (Numba) Instead of Python

- Numpy can do vectorization/broadcasting to do many computes at once (like a for loop but way faster than O(n) time)
- Numba is made for parallelization and is in machine code

#### 6. Misc Optimizations

- **Avoid Slow Structures:**
   - No slow structures like dictionaries or `np.where()` are ever used unless unavoidable
   - This is why there are so many numpy and numba maps/arrays everywhere
   - Small time saves in many places add up quickly
   - **Example:** Removing `np.where()` in 1 spot can be 2 million fewer calls per batch and turn 19 second areas into 3 second areas. and there's 650 batches. That's 2.8 hours.
   - At this scale, even function calls are time consuming
   - Those are minimized

- **Logical Rules:**
   - Basic rules to skip logically invalid combinations

