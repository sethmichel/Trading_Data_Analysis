import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pygam import LinearGAM, te
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

version = None
target_dir = "Holder_Strat/Parameter_Tuning/Target_Model"
def Set_Version(passed_version):
    global version
    version = passed_version

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

def Save_Model_Data(model, scaler, smearing_factor):
    model_data = {
        'model': model,
        'scaler': scaler,
        'smearing_factor': float(smearing_factor)
    }

    file_path = f'{target_dir}/trained_roi_predictor_model_{version}.pkl'
    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Saved model, scaler, and smearing_factor to {file_path}")


def Load_Model_Data():
    file_path = f'{target_dir}/trained_roi_predictor_model_{version}.pkl'
    
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    smearing_factor = model_data.get('smearing_factor', 1.0)
    
    print("Successfully loaded model, scaler, and smearing_factor")

    return model, scaler, smearing_factor


# Save all_data_samples_x and all_roi_samples_y to a local txt file
def Save_Training_Data(all_data_samples_x, all_roi_samples_y):
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
    file_path = f"{target_dir}/Data/target_runing_regression_numbers.txt"

    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Saved {len(all_data_samples_x)} data samples to {file_path}")


'''
- Load all_data_samples_x and all_roi_samples_y from the local txt file.
- Returns data in the new format: [{trade_id: [time, volatility]}, ...] and [{trade_id: roi}, ...]
'''
def Load_Test_Data():
    file_path = f'{target_dir}/Data/target_runing_regression_numbers.txt'
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    all_data_samples_x = data['all_data_samples_x']
    all_roi_samples_y = data['all_roi_samples_y']
    
    print(f"Loaded {len(all_data_samples_x)} data samples from {file_path}")
    
    return all_data_samples_x, all_roi_samples_y


def bulk_csv_date_converter(date):
    parts = date.split('-')

    if (len(parts[0]) == 1):
        parts[0] = f"0{parts[0]}"
    if (len(parts[1]) == 1):
        parts[1] = f"0{parts[1]}"
    if (len(parts[2]) == 2):
        parts[2] = f"20{parts[2]}"
    
    return '-'.join(parts)


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
        entry_time = row['Entry Time']               # hour:minute:second
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025
        market_df = market_data_dict_by_ticker[date][ticker]  # market data df for this ticker and date
        final_timestamp = datetime.strptime(market_df.at[-1, 'Time'], '%H:%M:%S').time()
        final_seconds = final_timestamp.hour * 3600 + final_timestamp.minute * 60 + final_timestamp.second

        # if we don't have data for this date
        if (date in skip_dates):
            continue
        # if we have market data for this date and ticker
        if date not in market_data_dict_by_ticker:
            print(f"No market data found for date {date}")
            skip_dates.append(date)
            continue
        # if we dont have ticker data for this date
        if ticker not in market_data_dict_by_ticker[date]:
            msg = f"No market data found for ticker {ticker} on date {date}"
            print(msg)
            raise ValueError(msg)
        # Skip trade if entry time is after final market data time
        entry_time_obj = datetime.strptime(entry_time, '%H:%M:%S').time()
        entry_seconds = entry_time_obj.hour * 3600 + entry_time_obj.minute * 60 + entry_time_obj.second
        if (entry_seconds > final_seconds):
            msg = f"entry time is after the end of market data. skipping trade"
            print(msg)
            continue
        
        trade_id = row['Trade Id']
        roi_list = roi_dictionary[trade_id]
        start_index = trade_start_indexes[trade_id]
        counter = -1                # this makes it easier to look up roi from roi_list
        next_sample_time = None     # Track when to take next sample
        
        # Iterate through market data starting from entry time
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
y_orig = array, n_bins = int
return array
'''
def Make_Strat_Labels(y_orig, n_bins=10):
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


'''
Duan's smearing estimate for unbiased back-transform after log1p.
s = E[exp(residual)], where residual = y_trans - fitted_trans.
If weights are provided, compute a weighted average so each trade's samples
do not dominate the correction.
y_trans = y values transformed (array). fitted_trans=idk (array), weights = array, returns float
'''
def Compute_Smearing_Factor(y_trans, fitted_trans, weights=None):
    residuals = y_trans - fitted_trans
    exp_resid = np.exp(residuals)
    if weights is not None:
        return float(np.average(exp_resid, weights=weights))
    return float(np.mean(exp_resid))












