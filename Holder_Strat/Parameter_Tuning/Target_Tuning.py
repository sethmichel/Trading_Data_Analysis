import pandas as pd
from pygam import LinearGAM, te
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import Helper_Functions
import json

version = 'v0.2'
'''
Exactly how v0.2 works
	1. Data gathering hasn't changed much, it's still 1 sample for every 10 minutes of a trade (including the trade entry)
	2. The raw data is extracted, the x data is scaled, then the y data is scaled with np.log1p. This introduces a strong 
    back transformation bias which we deal with a little later via duan smearing.
	3. Next we weigh the data to prevent longer trades (more data samples per trade) from dominating. The data is 'group 
    weighted'. We could've done 'per trade' weights and I have no idea which is better.
	4. (lam means lambda, and it's not a 1 it's L) Next, because the y distribution of the data is right skewed and we 
    ideally want all samples of the same trade grouped together we use StratifiedGroupKFold grouped by trade_id. This is 
    used to select the smoothing parameters (lam) which is something like this for example: np.logspace(-3, 3, num=8). 
		a. Original unscaled y values are separated into folds and we find the best mae and rmse values by comparing each 
        fold against the validation set. We pick the lam that minimizes the mae. The process of finding this lam is the 
        raw x values of each fold are scaled and the scaled y values for this fold are obtained, then a model is trained on 
        this folds scaled x, scaled y data, then we find the duan smearing factor to correct the back transformation bias we 
        get from the scaled y values, we can use this smear variable to get non-back-transformation bias predictions from the 
        model (prediction is on log scale, but smearing with exp(prediction) brings it to the normal scale). We then get the 
        mae and rmse values by converting the prediction value back to the original scale (so we calculate these on the 
        original scale, not log scale). After we do this for each fold we find the average mae and rmse across all folds. 
        This results in the best lam and the cv report
	5. With the best lam, we can train the actual linearGAM model and fit it using the weights
	6. Now, with the trained model we finally compute the actual duan smearing factor for the full training set. We use this 
    smearing factor to correct predictions when we use the model on live data
	7. Run
		x_scaled = scaler.transform([[minutes_since_open, volatility_percent]])
        actual_prediction = smear * np.exp(gam.predict(x_scaled))[0] - 1.0
'''




'''
- For each trade record for every x minutes (including the start of the trade) record x variables with the trade id
- Continue each trade until the stop loss is reached or we run out of market data
- x = [{trade_id: [time since market open, volatility %]}, ...], y = [{trade_id: roi, trade_id: roi, ...]
'''
def Collect_Data(bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes):
    skip_dates = []
    all_data_samples_x = []   # [{trade_id: [time since market open, volatility %]}, ...]
    all_roi_samples_y = []    # [{trade_id: roi, trade_id: roi, ...]

    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        date = Helper_Functions.bulk_csv_date_converter(row['Date'])  # 08-09-2025
        if (date in skip_dates):
            continue
        
        # Check if we have market data for this date and ticker
        if date not in market_data_dict_by_ticker:
            print(f"No market data found for date {date}")
            skip_dates.append(date)
            continue  
            
        if ticker not in market_data_dict_by_ticker[date]:
            msg = f"No market data found for ticker {ticker} on date {date}"
            print(msg)
            raise ValueError(msg)

        entry_time = row['Entry Time']               # hour:minute:second
        trade_id = row['Trade Id']
        roi_list = roi_dictionary[trade_id]
        market_df = market_data_dict_by_ticker[date][ticker].copy()  # market data df for this ticker and date
        start_index = trade_start_indexes[trade_id]
        
        # Skip trade if entry time is after final market data time
        if (Helper_Functions.Check_We_Have_Data_For_Trade(market_df, entry_time) == False):
            print(f"Skipping trade {trade_id}: entry time {entry_time} is after final market data time")
            continue
        
        # Track stop loss state
        counter = -1                # this makes it easier to look up roi from roi_list
        next_sample_time = None     # Track when to take next sample
        
        # Iterate through market data starting from entry point
        for i in range(start_index, len(market_df)):
            counter += 1
            
            current_time = market_df.iloc[i]['Time']
            current_volatility = market_df.iloc[i]['Volatility Percent']
            time_since_market_open = market_df.iloc[i]['Time Since Market Open']
            
            # Take sample at start of trade (first iteration) or every 10 minutes
            if counter == 0:  # First sample at trade start
                next_sample_time = time_since_market_open + 10  # Next sample in 10 minutes
                
                # Calculate best future ROI from this point
                if counter < len(roi_list):
                    best_future_roi = max(roi_list[counter:])
                else:
                    best_future_roi = roi_list[-1] if roi_list else 0
                
                # Add sample
                all_data_samples_x.append({trade_id: [time_since_market_open, current_volatility]})
                all_roi_samples_y.append({trade_id: best_future_roi})
                
            elif time_since_market_open >= next_sample_time:  # Time for next 10-minute sample
                next_sample_time += 10  # Set next sample time
                
                # Calculate best future ROI from this point
                if counter < len(roi_list):
                    best_future_roi = max(roi_list[counter:])
                else:
                    best_future_roi = roi_list[-1] if roi_list else 0
                
                # Add sample
                all_data_samples_x.append({trade_id: [time_since_market_open, current_volatility]})
                all_roi_samples_y.append({trade_id: best_future_roi})
            
            # Check if trade is done
            if current_time == trade_end_timestamps[trade_id]:
                # Trade is complete
                break
            
    return all_data_samples_x, all_roi_samples_y


'''
- scaling x prevents the variable with larger numbers from dominating & we also save the scaler to use for predicting
- x = [{trade_id: [time since market open, volatility %]}, ...], y = [{trade_id: roi, trade_id: roi, ...]
- techniques: 
    1) data is weighted so longer trades with more samples don't dominate results
    2) StratifiedGroupKFold is used since y distribution is skewed and it keeps samples of the same trade id together
    3) y is scaled with log(1+y) to better normalize results
'''
def Train_Model(all_data_samples_x, all_roi_samples_y):
    raw_x = []
    raw_y = []
    groups = []  # trade_id per sample for group-aware CV/weights
    
    # 1) Extract aligned features, targets, and groups
    for i, (x_dict, y_dict) in enumerate(zip(all_data_samples_x, all_roi_samples_y)):
        trade_id_x, features = next(iter(x_dict.items()))
        trade_id_y, roi_value = next(iter(y_dict.items()))
        if trade_id_x != trade_id_y:
            raise ValueError(f"Trade ID mismatch at index {i}: {trade_id_x} vs {trade_id_y}")
        raw_x.append(features)
        raw_y.append(roi_value)
        groups.append(trade_id_x)

    # 2) Prepare X and y values
    # Note: We will fit a global scaler for the final model, but inside CV we will fit
    # a scaler per-fold to avoid data leakage.
    X_raw = np.asarray(raw_x, dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)  # used for final fit only
    # scale y values. log-transformed target to stabilize variance and reduce outlier influence
    y_arr = np.asarray(raw_y, dtype=float)
    y_trans = np.log1p(y_arr)

    # 4) weigh samples: per-sample weights so each trade contributes ~equally (group-weighted)
    # weight_i = 1 / (#samples in same trade)
    unique, counts = np.unique(np.asarray(groups), return_counts=True)
    group_to_count = {g: c for g, c in zip(unique, counts)}
    sample_weights = np.asarray([1.0 / group_to_count[g] for g in groups], dtype=float)

    # 5) Use StratifiedGroupKFold to select smoothing parameter with grouped, stratified CV
    # IMPORTANT: pass raw X; scaling will be learned per-fold to prevent leakage.
    best_lam, cv_report = Select_Best_Lambda_StratifGroupCV(X_raw, y_trans, y_arr, np.asarray(groups), sample_weights)

    # 6) Fit final model on all scaled data with cluster weights
    model = LinearGAM(te(0,1), lam=best_lam)
    model.fit(X_scaled, y_trans, weights=sample_weights)

    # 6b) Compute Duan smearing factor on full training data (unbiased back-transform)
    fitted_trans = model.predict(X_scaled)
    smearing_factor = Compute_Smearing_Factor(y_trans, fitted_trans, sample_weights)

    # 7) Print CV summary (cross validation summary)
    print("\nStratifiedGroupKFold CV Summary (original ROI scale):")
    print(f"Splits: {cv_report['n_splits']}, bins: {cv_report['n_bins']}")
    print(f"Best lam: {best_lam:.6f}")
    print(f"Mean MAE: {cv_report['mean_mae']:.6f}, Mean RMSE: {cv_report['mean_rmse']:.6f}")

    return model, X_scaled, scaler, smearing_factor


'''
- Create stratification labels for regression by binning y into quantiles.
- basically it's taking y and turning it into bins (classification) based on quantiles
- Reduces bins adaptively if too few unique edges.
- Stratification: process of dividing a population into distinct, non-overlapping subgroups, or "strata," before sampling
'''
def Make_Strat_Labels(y_orig: np.ndarray, n_bins: int = 10) -> np.ndarray:
    y = np.asarray(y_orig, dtype=float)

    # Ensure enough unique values for binning. if I have more than 10 unique y's it'll return 10
    unique_vals = np.unique(y)
    max_bins = min(n_bins, max(2, unique_vals.size))
    if (max_bins < 10 or max_bins > 15):
        raise ValueError(f"idk what's going on with the bins. Make_Strat_Labels(), max_bins = {max_bins}")

    # Compute quantile edges
    quantiles = np.linspace(0, 1, num=max_bins + 1)
    edges = np.quantile(y, quantiles)

    # Make edges strictly increasing to avoid digitize issues
    edges = np.unique(edges)
    if edges.size < 3:
        # Fallback to two bins
        edges = np.array([y.min() - 1e-9, np.median(y), y.max() + 1e-9])

    # Digitize to get bin indices in [0, len(edges)-2]
    labels = np.digitize(y, edges[1:-1], right=False)

    return labels.astype(int)


'''
- Select smoothing parameter using StratifiedGroupKFold with per-trade weights.
- Scores are computed on the original ROI scale using MAE and RMSE.
- Note: lam uses letter 'L' (lambda), not the digit '1'
- 10 bins is pretty ideal. not worth messing with this
- this for example) np.logspace(-3, 3, num=8) is the lambda soothing penalty. it's how smooth the curve is
- Stratification: process of dividing a population into distinct, non-overlapping subgroups, or "strata," before sampling
- for each fold, we're back-transforming y (log) to their original values, computing mae and rmse against the validation set
  averaging them across folds (mean mae, mean rmse), and choosing the lam that minimizes the mean mae
'''
def Select_Best_Lambda_StratifGroupCV(X_raw: np.ndarray,
                                      y_trans: np.ndarray,
                                      y_orig: np.ndarray,
                                      groups: np.ndarray,
                                      sample_weights: np.ndarray,
                                      n_splits: int = 5,
                                      lam_grid: np.ndarray = None):

    if lam_grid is None:
        lam_grid = np.logspace(-3, 3, num=8) # this is the lambda soothing penalty

    # Prepare stratification labels on ORIGINAL y to respect skewness (turn y into bins)
    strat_labels = Make_Strat_Labels(y_orig, n_bins=10)

    # Ensure split count does not exceed unique group count
    n_unique_groups = np.unique(groups).size
    n_splits_eff = int(min(n_splits, max(2, n_unique_groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits_eff, shuffle=True, random_state=42)

    best_lam = None
    best_mae = np.inf
    mae_per_lam = []
    rmse_per_lam = []

    for lam in lam_grid:
        fold_mae = []
        fold_rmse = []

        for train_idx, valid_idx in cv.split(X_raw, strat_labels, groups):
            # Per-fold scaling to prevent leakage
            scaler_fold = StandardScaler()
            X_tr_raw, X_va_raw = X_raw[train_idx], X_raw[valid_idx]
            X_tr = scaler_fold.fit_transform(X_tr_raw)
            X_va = scaler_fold.transform(X_va_raw)

            y_tr, y_va_trans = y_trans[train_idx], y_trans[valid_idx]
            w_tr = sample_weights[train_idx]
            w_va = sample_weights[valid_idx]

            # Fit model on train fold with weights
            gam = LinearGAM(te(0,1), lam=lam)
            gam.fit(X_tr, y_tr, weights=w_tr)

            # Duan smearing factor computed on training residuals (log1p scale)
            fitted_tr = gam.predict(X_tr)
            smear = Compute_Smearing_Factor(y_tr, fitted_tr, w_tr)

            # Predict on validation; unbiased back-transform to original ROI scale
            y_pred_va = smear * np.exp(gam.predict(X_va)) - 1.0
            y_true_va = np.expm1(y_va_trans)  # original scale for metric

            # Group-weighted metrics: weights sum to ~1 per trade so trades contribute equally
            mae = mean_absolute_error(y_true_va, y_pred_va, sample_weight=w_va)
            rmse = np.sqrt(mean_squared_error(y_true_va, y_pred_va, sample_weight=w_va))
            fold_mae.append(mae)
            fold_rmse.append(rmse)

        mean_mae = float(np.mean(fold_mae))
        mean_rmse = float(np.mean(fold_rmse))
        mae_per_lam.append(mean_mae)
        rmse_per_lam.append(mean_rmse)

        if mean_mae < best_mae:
            best_mae = mean_mae
            best_lam = lam

    report = {
        'n_splits': n_splits_eff,
        'n_bins': int(np.unique(strat_labels).size),
        'lam_grid': lam_grid.tolist(),
        'mae_per_lam': mae_per_lam,
        'rmse_per_lam': rmse_per_lam,
        'mean_mae': float(best_mae),
        'mean_rmse': float(rmse_per_lam[np.argmin(mae_per_lam)])
    }

    return float(best_lam), report


def Compute_Smearing_Factor(y_trans: np.ndarray,
                            fitted_trans: np.ndarray,
                            weights: np.ndarray = None) -> float:
    """
    Duan's smearing estimate for unbiased back-transform after log1p.
    s = E[exp(residual)], where residual = y_trans - fitted_trans.
    If weights are provided, compute a weighted average so each trade's samples
    do not dominate the correction.
    """
    residuals = y_trans - fitted_trans
    exp_resid = np.exp(residuals)
    if weights is not None:
        return float(np.average(exp_resid, weights=weights))
    return float(np.mean(exp_resid))


def Save_Model(model, scaler, smearing_factor: float):
    model_data = {
        'model': model,
        'scaler': scaler,
        'smearing_factor': float(smearing_factor)
    }

    with open(f'Holder_Strat/Parameter_Tuning/model_files_and_data/trained_roi_predictor_model_{version}.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Saved model, scaler, and smearing_factor to trained_roi_predictor_model_{version}.pkl")


def Load_Model():
    try:
        with open(f'Holder_Strat/Parameter_Tuning/model_files_and_data/trained_roi_predictor_model_{version}.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        smearing_factor = model_data.get('smearing_factor', 1.0)
        
        print("Successfully loaded model, scaler, and smearing_factor")
        return model, scaler, smearing_factor
        
    except FileNotFoundError:
        print(f"trained_roi_predictor_model_{version}.pkl not found. Please train and save a model first.")
        return None, None, None

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


'''
- Scale inputs with the saved scaler, run model.predict (which outputs log1p(y)), and invert back to the 
  original ROI scale with expm1.
ex): pred = Predict_Max_ROI(model, scaler, minutes_since_open=30, volatility_percent=0.8)
'''
def Predict_Max_ROI(model, scaler, minutes_since_open: float, volatility_percent: float, smearing_factor: float) -> float:
    """
    Scale inputs with the saved scaler, run model.predict (which outputs log1p(y)), and
    invert back to the original ROI scale with Duan smearing for unbiased estimates:
    y_hat = smear * exp(pred) - 1
    """
    data = scaler.transform([[minutes_since_open, volatility_percent]])
    return float(smearing_factor * np.exp(model.predict(data))[0] - 1.0)


# Save all_data_samples_x and all_roi_samples_y to a local txt file
def Save_Test_Values(all_data_samples_x, all_roi_samples_y):
    # Convert pandas/numpy types to Python native types for JSON serialization
    # Convert all_data_samples_x (list of dicts with trade_id: [time, volatility])
    converted_x = []
    for sample_dict in all_data_samples_x:
        converted_sample = {}
        for trade_id, values in sample_dict.items():
            converted_sample[trade_id] = [float(val) for val in values]
        converted_x.append(converted_sample)
    
    # Convert all_roi_samples_y (list of dicts with trade_id: roi)
    converted_y = []
    for roi_dict in all_roi_samples_y:
        converted_roi = {}
        for trade_id, roi in roi_dict.items():
            converted_roi[trade_id] = float(roi)
        converted_y.append(converted_roi)
    
    # Create the data structure to save
    data_to_save = {
        'all_data_samples_x': converted_x,
        'all_roi_samples_y': converted_y
    }
    
    # Save to txt file
    with open(f'Holder_Strat/Parameter_Tuning/model_files_and_data/target_runing_regression_numbers_{version}.txt', 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Saved {len(all_data_samples_x)} data samples to target_runing_regression_numbers_{version}.txt")


'''
- Load all_data_samples_x and all_roi_samples_y from the local txt file.
- Returns data in the new format: [{trade_id: [time, volatility]}, ...] and [{trade_id: roi}, ...]
'''
def Load_Test_Values():

    try:
        with open(f'Holder_Strat/Parameter_Tuning/model_files_and_data/target_runing_regression_numbers_{version}.txt', 'r') as f:
            data = json.load(f)
        
        all_data_samples_x = data['all_data_samples_x']
        all_roi_samples_y = data['all_roi_samples_y']
        
        print(f"Loaded {len(all_data_samples_x)} data samples from target_runing_regression_numbers_{version}.txt")
        return all_data_samples_x, all_roi_samples_y
        
    except FileNotFoundError:
        print(f"target_runing_regression_numbers_{version}.txt not found. Please run Save_Test_Values() first.")
        return None, None
    except Exception as e:
        print(f"Error loading target_runing_regression_numbers_{version}.txt: {e}")
        return None, None


'''
- basically do the same as when I found the roi predictions, but don't look ahead, instead just spit out the prediction
- I need it to be grouped by ticker: ticker, timestamp, prediction
- roi dictionary: {trade_id: [roi values], ...}
'''
def Full_Test_Model_Over_Trade_Data(model, scaler, smearing_factor, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes):
    skip_dates = []
    results = [] # [[ticker, timestamp, prediction], ...] this is so I can check any value on what on the actual charts
    sums = {}

    for _, row in bulk_df.iterrows():
        date = Helper_Functions.bulk_csv_date_converter(row['Date'])  # 08-09-2025
        ticker = row['Ticker']

        if (date in skip_dates):
            continue

        # Check if we have market data for this date and ticker
        if date not in market_data_dict_by_ticker:
            print(f"No market data found for date {date}")
            skip_dates.append(date)
            continue  
            
        if ticker not in market_data_dict_by_ticker[date]:
            msg = f"No market data found for ticker {ticker} on date {date}"
            print(msg)
            raise ValueError(msg)

        entry_time = row['Entry Time']  # hour:minute:second
        trade_id = row['Trade Id']
        market_df = market_data_dict_by_ticker[date][ticker].copy()  # market data df for this ticker and date
        
        for i in range(len(market_df)):
            market_time = market_df.iloc[i]['Time']
            if market_time >= entry_time:
                start_index = i
                trade_start_indexes[trade_id] = i
                break
        
        next_sample_time = -99      # Track when to take next sample
        counter = -1
        
        # Iterate through market data starting from entry point
        for i in range(start_index, len(market_df)):
            counter += 1
            current_time = market_df.iloc[i]['Time']
            current_price = market_df.iloc[i]['Price']
            current_volatility = market_df.iloc[i]['Volatility Percent']
            time_since_market_open = market_df.iloc[i]['Time Since Market Open']
            curr_roi = roi_dictionary[trade_id][counter]

            # Take sample every 10 minutes
            if time_since_market_open >= next_sample_time:
                next_sample_time += 10
                
                # Unbiased back-transform with Duan smearing
                roi_prediction = Predict_Max_ROI(model, scaler, time_since_market_open, current_volatility, smearing_factor)
                results.append([ticker, current_time, roi_prediction])
            
            # check if we're at our roi prediction target
            if (curr_roi >= roi_prediction):
                # end the trade
                if (sums.get(ticker) == None):
                    sums[ticker] = curr_roi
                else:
                    sums[ticker] += curr_roi
                break

            # Check if trade is done
            if current_time == trade_end_timestamps[trade_id]:
                # Trade is complete
                if (sums.get(ticker) == None):
                    sums[ticker] = curr_roi
                else:
                    sums[ticker] += curr_roi
                break

    # Write results to text file
    with open(f'Holder_Strat/Parameter_Tuning/model_files_and_data/roi_prediction_model_trade_results_{version}.txt', 'w') as f:
        f.write('sums by ticker\n')
        for ticker in sums:
            f.write(f'{ticker} = {round(sums[ticker], 2)}\n')
        
        overall = sum(sums.values())
        f.write(f'overall = {round(overall, 2)}\n\n')
        
        f.write('Specific Results\n')
        results.sort(key=lambda x: x[0]) # sort by ticker
        for result in results:
            f.write(f'{result[0]}, {result[1]}, {round(result[2], 2)}\n')

    return results


# keep: this is useful for assessing what optimizations we can do next. it's the distribution of y values
def Summarize_Response_Distribution(all_roi_samples_y, save_hist: bool = True, show_plot: bool = False):
    """
    Summarize and visualize the distribution of the response variable (Y):
    best future ROI values used to train the model.

    Saves a concise text report and a histogram plot for quick inspection.
    """
    # Extract response values preserving original scale (percent ROI)
    y_values = []
    for roi_dict in all_roi_samples_y:
        _, roi_value = next(iter(roi_dict.items()))
        try:
            y_values.append(float(roi_value))
        except Exception:
            continue

    if len(y_values) == 0:
        print("No response values found; skipping response distribution summary.")
        return

    y = np.array(y_values, dtype=float)

    # Basic statistics
    count = y.size
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_mean = float(np.mean(y))
    y_median = float(np.median(y))
    y_std = float(np.std(y))

    # Percentiles
    percentiles_list = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(y, percentiles_list).tolist()

    # Skewness and kurtosis (excess) without SciPy
    if y_std > 0:
        centered = y - y_mean
        skewness = float(np.mean(centered ** 3) / (y_std ** 3))
        kurtosis_excess = float(np.mean(centered ** 4) / (y_std ** 4) - 3.0)
    else:
        skewness = 0.0
        kurtosis_excess = 0.0

    # Useful thresholds for trading context
    frac_lt_0 = float(np.mean(y < 0))
    frac_ge_target = float(np.mean(y >= 0.6))  # >= original target
    frac_between_0_0p1 = float(np.mean((y >= 0) & (y < 0.1)))

    # Write summary to file
    summary_path = f'Holder_Strat/Parameter_Tuning/model_files_and_data/Response_Distribution_Summary_{version}.txt'
    with open(summary_path, 'w') as f:
        f.write('Response (Y) Distribution Summary - Best Future ROI\n')
        f.write('===============================================\n')
        f.write(f'Count: {count}\n')
        f.write(f'Min: {y_min:.4f}%\n')
        f.write(f'P1/P5/P10: {pct_vals[0]:.4f}% / {pct_vals[1]:.4f}% / {pct_vals[2]:.4f}%\n')
        f.write(f'P25/P50/P75: {pct_vals[3]:.4f}% / {pct_vals[4]:.4f}% / {pct_vals[5]:.4f}%\n')
        f.write(f'P90/P95/P99: {pct_vals[6]:.4f}% / {pct_vals[7]:.4f}% / {pct_vals[8]:.4f}%\n')
        f.write(f'Max: {y_max:.4f}%\n')
        f.write(f'Mean: {y_mean:.4f}%\n')
        f.write(f'Median: {y_median:.4f}%\n')
        f.write(f'Std Dev: {y_std:.4f}\n')
        f.write(f'Skewness: {skewness:.4f}\n')
        f.write(f'Excess Kurtosis: {kurtosis_excess:.4f}\n')
        f.write('\nShares of interest:\n')
        f.write(f'  < 0%: {frac_lt_0:>.2%}\n')
        f.write(f'  >= 0.6% (target): {frac_ge_target:>.2%}\n')
        f.write(f'  [0%, 0.1%): {frac_between_0_0p1:>.2%}\n')

    print("\nResponse distribution summary written to:")
    print(summary_path)

    # Save histogram plot
    if save_hist:
        plt.figure(figsize=(10, 6))
        # Use a simple rule for bin count for readability
        num_bins = max(20, int(np.ceil(np.sqrt(count))))
        plt.hist(y, bins=num_bins, color='steelblue', edgecolor='black', alpha=0.75)
        plt.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7)
        plt.axvline(0.6, color='green', linestyle='--', linewidth=1, alpha=0.7)
        plt.title('Response (Y) Distribution: Best Future ROI')
        plt.xlabel('ROI (%)')
        plt.ylabel('Count')
        plt.grid(True, axis='y', alpha=0.25)
        plt.tight_layout()
        plot_path = f'Holder_Strat/Parameter_Tuning/model_files_and_data/Response_Distribution_Hist_{version}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        print('Histogram saved to:')
        print(plot_path)

# all_data_samples_x: [{trade_id: [minutes_since_open, volatility_percent]} ...]
# all_roi_samples_y: [{trade_id: roi}, ...]
# scaler: used to scale x values for the model input
def Run_Model_Diagnostics(model, scaler, smearing_factor, all_data_samples_x, all_roi_samples_y):
    """
    Director function for running various diagnostic tests on the model.
    """
    print("\nRunning model diagnostics...")
    # 0) Response variable distribution (Y)
    Summarize_Response_Distribution(all_roi_samples_y, save_hist=True, show_plot=False)
    
    # Run residual plot analysis
    Plot_Residuals_Vs_Fitted(model, scaler, smearing_factor, all_data_samples_x, all_roi_samples_y)
    
    print("Diagnostics complete!")


def Plot_Residuals_Vs_Fitted(model, scaler, smearing_factor, all_data_samples_x, all_roi_samples_y):
    """
    Create a residual plot vs fitted values with color coding by trade ID.
    
    -Residuals should scatter randomly around 0 with no patterns if the model is good.
    Patterns indicate missing nonlinearity or heteroscedasticity.
    -because this is many samples per trade, samples should be id'd somehow by trade
    (it's possible samples from the same trade violate expectations but are correct)
    """
    print("Creating residual plot vs fitted values...")
    
    # Extract data for plotting
    raw_x = []
    actual_y = []
    trade_ids = []
    
    # Extract features and actual values
    for i, (x_dict, y_dict) in enumerate(zip(all_data_samples_x, all_roi_samples_y)):
        # Get trade_id and features from x_dict
        trade_id_x, features = next(iter(x_dict.items()))
        raw_x.append(features)
        
        # Get trade_id and roi from y_dict
        trade_id_y, roi_value = next(iter(y_dict.items()))
        actual_y.append(roi_value)
        
        # Verify trade IDs match (they should)
        if trade_id_x != trade_id_y:
            print(f"Warning: Trade ID mismatch at index {i}: {trade_id_x} vs {trade_id_y}")
            return
        
        trade_ids.append(trade_id_x)
    
    # Scale the features and get predictions
    x_scaled = scaler.transform(raw_x)
    # Model outputs log1p(y); use unbiased back-transform for diagnostics
    predictions = smearing_factor * np.exp(model.predict(x_scaled)) - 1.0
    
    # Calculate residuals (actual - predicted) on original scale
    residuals = np.array(actual_y) - predictions
    
    # Create color map for trade IDs
    unique_trade_ids = list(set(trade_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_trade_ids)))
    color_map = {trade_id: colors[i % len(colors)] for i, trade_id in enumerate(unique_trade_ids)}
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each trade ID with a different color
    for trade_id in unique_trade_ids:
        # Get indices for this trade ID
        trade_indices = [i for i, tid in enumerate(trade_ids) if tid == trade_id]
        
        if len(trade_indices) > 0:
            trade_predictions = [predictions[i] for i in trade_indices]
            trade_residuals = [residuals[i] for i in trade_indices]
            
            plt.scatter(trade_predictions, trade_residuals, 
                       c=[color_map[trade_id]], 
                       alpha=0.6, 
                       s=30,
                       label=f'Trade {trade_id}' if len(unique_trade_ids) <= 20 else None)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Formatting
    plt.xlabel('Fitted Values (Predicted ROI)', fontsize=12)
    plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot vs Fitted Values\n(Color-coded by Trade ID)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add legend only if we have a reasonable number of trade IDs
    if len(unique_trade_ids) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
    else:
        plt.tight_layout()
        print(f"Note: {len(unique_trade_ids)} unique trade IDs - legend omitted for clarity")
    
    # Save the plot
    plt.savefig(f'Holder_Strat/Parameter_Tuning/model_files_and_data/Residual_Plot_All_Samples_{version}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print diagnostic statistics
    print(f"\nResidual Analysis Summary (All Samples):")
    print(f"Number of data points: {len(residuals)}")
    print(f"Number of unique trades: {len(unique_trade_ids)}")
    print(f"Mean residual: {np.mean(residuals):.4f}")
    print(f"Std deviation of residuals: {np.std(residuals):.4f}")
    print(f"Min residual: {np.min(residuals):.4f}")
    print(f"Max residual: {np.max(residuals):.4f}")
    
    # Check for patterns (basic statistical tests)
    print(f"\nPattern Detection (All Samples):")
    
    # Calculate correlation between fitted values and residuals
    correlation = np.corrcoef(predictions, residuals)[0, 1]
    print(f"Correlation between fitted values and residuals: {correlation:.4f}")
    if abs(correlation) > 0.1:
        print("  ⚠️  Warning: Moderate correlation detected - may indicate model issues")
    elif abs(correlation) > 0.05:
        print("  ⚠️  Slight correlation detected - monitor for patterns")
    else:
        print("  ✓ Low correlation - good sign")
    
    # Check for heteroscedasticity (variance changes with fitted values)
    # Split data into low and high fitted value groups
    median_fitted = np.median(predictions)
    low_fitted_residuals = [residuals[i] for i, pred in enumerate(predictions) if pred <= median_fitted]
    high_fitted_residuals = [residuals[i] for i, pred in enumerate(predictions) if pred > median_fitted]
    
    if len(low_fitted_residuals) > 0 and len(high_fitted_residuals) > 0:
        low_var = np.var(low_fitted_residuals)
        high_var = np.var(high_fitted_residuals)
        var_ratio = max(low_var, high_var) / min(low_var, high_var)
        
        print(f"Variance ratio (high/low fitted values): {var_ratio:.2f}")
        if var_ratio > 2.0:
            print("  ⚠️  Warning: Potential heteroscedasticity detected")
        else:
            print("  ✓ Variance appears relatively constant")
    
    print(f"\nPlot saved to: Holder_Strat/Parameter_Tuning/model_files_and_data/Residual_Plot_All_Samples_{version}.png")
    
    # Now create the second plot with averaged data per trade
    Plot_Residuals_Vs_Fitted_Mean_Per_Trade(predictions, residuals, actual_y, trade_ids, unique_trade_ids)


def Plot_Residuals_Vs_Fitted_Mean_Per_Trade(predictions, residuals, actual_y, trade_ids, unique_trade_ids):
    """
    Create a residual plot using the average residual and fitted value for each trade ID.
    This reduces each trade to a single data point.
    """
    print("\nCreating residual plot with mean values per trade...")
    
    # Calculate mean fitted values and residuals for each trade
    mean_fitted_per_trade = []
    mean_residuals_per_trade = []
    
    for trade_id in unique_trade_ids:
        # Get indices for this trade ID
        trade_indices = [i for i, tid in enumerate(trade_ids) if tid == trade_id]
        
        if len(trade_indices) > 0:
            # Calculate mean fitted value and residual for this trade
            trade_fitted_values = [predictions[i] for i in trade_indices]
            trade_residual_values = [residuals[i] for i in trade_indices]
            
            mean_fitted = np.mean(trade_fitted_values)
            mean_residual = np.mean(trade_residual_values)
            
            mean_fitted_per_trade.append(mean_fitted)
            mean_residuals_per_trade.append(mean_residual)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot all points without color coding
    plt.scatter(mean_fitted_per_trade, mean_residuals_per_trade, 
               alpha=0.7, s=50, color='blue')
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Formatting
    plt.xlabel('Mean Fitted Values per Trade (Predicted ROI)', fontsize=12)
    plt.ylabel('Mean Residuals per Trade (Actual - Predicted)', fontsize=12)
    plt.title('Residual Plot vs Fitted Values\n(Mean Values per Trade)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'Holder_Strat/Parameter_Tuning/model_files_and_data/Residual_Plot_Mean_Sample_{version}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print diagnostic statistics for mean values
    mean_fitted_array = np.array(mean_fitted_per_trade)
    mean_residuals_array = np.array(mean_residuals_per_trade)
    
    print(f"\nResidual Analysis Summary (Mean per Trade):")
    print(f"Number of trades: {len(mean_residuals_per_trade)}")
    print(f"Mean residual: {np.mean(mean_residuals_array):.4f}")
    print(f"Std deviation of residuals: {np.std(mean_residuals_array):.4f}")
    print(f"Min residual: {np.min(mean_residuals_array):.4f}")
    print(f"Max residual: {np.max(mean_residuals_array):.4f}")
    
    # Pattern detection for mean values
    print(f"\nPattern Detection (Mean per Trade):")
    
    # Calculate correlation between mean fitted values and mean residuals
    if len(mean_fitted_per_trade) > 1:
        correlation = np.corrcoef(mean_fitted_array, mean_residuals_array)[0, 1]
        print(f"Correlation between fitted values and residuals: {correlation:.4f}")
        if abs(correlation) > 0.1:
            print("  ⚠️  Warning: Moderate correlation detected - may indicate model issues")
        elif abs(correlation) > 0.05:
            print("  ⚠️  Slight correlation detected - monitor for patterns")
        else:
            print("  ✓ Low correlation - good sign")
        
        # Check for heteroscedasticity with mean values
        median_fitted = np.median(mean_fitted_array)
        low_fitted_residuals = [mean_residuals_array[i] for i, pred in enumerate(mean_fitted_array) if pred <= median_fitted]
        high_fitted_residuals = [mean_residuals_array[i] for i, pred in enumerate(mean_fitted_array) if pred > median_fitted]
        
        if len(low_fitted_residuals) > 0 and len(high_fitted_residuals) > 0:
            low_var = np.var(low_fitted_residuals)
            high_var = np.var(high_fitted_residuals)
            var_ratio = max(low_var, high_var) / min(low_var, high_var) if min(low_var, high_var) > 0 else float('inf')
            
            print(f"Variance ratio (high/low fitted values): {var_ratio:.2f}")
            if var_ratio > 2.0:
                print("  ⚠️  Warning: Potential heteroscedasticity detected")
            else:
                print("  ✓ Variance appears relatively constant")
    
    print(f"\nPlot saved to: Holder_Strat/Parameter_Tuning/model_files_and_data/Residual_Plot_Mean_Sample_{version}.png")


def Main():
    columns_to_keep = ["Date", "Trade Id", "Ticker", "Entry Time", "Time in Trade", "Entry Price", "Exit Price", "Trade Type", "Exit Price", "Entry Volatility Percent", "Original Holding Reached", "Original Best Exit Percent", "Original Percent Change"]
    bulk_df = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[columns_to_keep]
    market_data_dict_by_ticker = Helper_Functions.Load_Market_Data_Dictionary(bulk_df) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    
    roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Create_Roi_Dictionary_For_Trades(bulk_df, market_data_dict_by_ticker, -0.4)
    
    all_data_samples_x, all_roi_samples_y = Collect_Data(bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes)
    Save_Test_Values(all_data_samples_x, all_roi_samples_y)
    
    model, x_scaled, scaler, smearing_factor = Train_Model(all_data_samples_x, all_roi_samples_y)
    Save_Model(model, scaler, smearing_factor)
    
    #roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Load_Roi_Dictionary_And_Values()
    #all_data_samples_x, all_roi_samples_y = Load_Test_Values()
    #model, scaler, smearing_factor = Load_Model()

    # Print the number of negative values in all_roi_samples_y
    negative_count = sum(1 for roi_dict in all_roi_samples_y for value in roi_dict.values() if value < 0)
    print(f"Number of negative values in all_roi_samples_y: {negative_count}") # 14

    Run_Model_Diagnostics(model, scaler, smearing_factor, all_data_samples_x, all_roi_samples_y)

    #Full_Test_Model_Over_Trade_Data(model, scaler, smearing_factor, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes)
    


if __name__ == "__main__":
    Main()


'''
considerations
1) market data should be rechecked for errors like time gaps and only use live market data to avoid data errors/outlier wrong 
   data points
2) trades that start after market data ends are skipped
3) some trades end after market data ends. we can't skip these because we're testing what happens if I don't use a max roi 
   target, but if we only have like 10 minutes of data for a trade it's basically wrong data. so we need at least 30 minutes
   of data after a trade entry time, but also if it's not 30 minutes after is after 12:30 we do count it (since we can't have 
   30 minutes). this is checked after roi is found. the roi list will show it regardless
'''