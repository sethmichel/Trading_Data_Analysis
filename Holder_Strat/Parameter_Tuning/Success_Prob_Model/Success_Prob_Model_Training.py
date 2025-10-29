import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pygam import LogisticGAM, s, te
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import matplotlib.pyplot as plt

#np.seterr(all='raise') # NOTE: WARNING, this raises warnings to errors so that I can find the line causing problems

version = None
success_prob_dir = "Holder_Strat/Parameter_Tuning/Success_Prob_Model"
def Set_Version(passed_version):
    global version
    version = passed_version


def Save_Model_Data(model, scaler, x_scaled, holding_value, holding_sl_value, largest_sl_value):
    file_path = f'{success_prob_dir}/Data/Success_Probability_model_{version}_holding_value_{holding_value}_holding_sl_value_{holding_sl_value}_largest_sl_value_{largest_sl_value}.pkl'

    model_data = {
        'model': model,
        'scaler': scaler,
        'x_scaled': x_scaled
    }

    with open(file_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"Saved model, scaler to {file_path}")


def Load_Model_Data(holding_value, holding_sl_value, largest_sl_value):
    file_path = f'{success_prob_dir}/Data/Success_Probability_model_{version}_holding_value_{holding_value}_holding_sl_value_{holding_sl_value}_largest_sl_value_{largest_sl_value}.pkl'
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    scaler = data['scaler']
    x_scaled = data['x_scaled']
    
    print("Successfully loaded model, and scaler")

    return model, scaler, x_scaled


# save training data so I can quickly get it for tests
def Save_Training_Data(results_df, neither_count, trade_count):
    file_path = f"{success_prob_dir}/Data/Training_Data.txt"
    
    data_to_save = {
        'results_df': results_df.to_dict(orient='records'), # df's can't be serialized. this deconstructs it into a list of dicts
        'neither_count': neither_count,
        'trade_count': trade_count
    }
    
    # Save to txt file
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Saved results_df, neither trade count, and trade count training data to {file_path}")


# load training data so I can quickly get it for tests
def Load_Training_Data():
    file_path = f'{success_prob_dir}/Data/Training_Data.txt'
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results_df = pd.DataFrame(data['results_df'])
    neither_count = data['neither_count']
    trade_count = data['trade_count']

    print(f"Loaded results_df, neither trade count, and trade count from {file_path}")
    
    return results_df, neither_count, trade_count


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
how this works
-for each trade, record its starting volatility percent and minutes since market open (x features), and the trade id and ticker
-then, look though its roi values until it hits the stop loss or the target (y value). target = 1, stop loss/failure = 0. if we run out of data
   then drop the trade since that's so rare. record the number of dropped trades though
-return results in a easily to read dataframe, the count of skipped trades, and the number of processed trades

parameters
-roi dictionary: each trades second by second roi for the trade duration and beyond
-market_data_dict_by_ticker: market data for each ticker for each date in dataframes. {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
-bulk_df: dataframe of all trades

target variables
x features: minutes since market open, volatility percent
y features: success (1) or failure (0)
ending columns: [trade_id, ticker, minutes since open, volatility percent, result] (where result is y (0 or 1))
'''
def Collect_Training_Data(bulk_df, roi_dictionary, holding_value, stop_loss_value):
    all_trade_data = [] # growing the results dataframe in a loop is super inefficient. just update this dictionary then add it at the end
    skip_dates = []
    trade_count = 0
    neither_count = 0

    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        trade_id = row['Trade Id']
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025

        # if we don't have data for this date
        if (date in skip_dates):
            continue

        # trade is valid if it reaches here

        # 2) process the trade
        roi_list = roi_dictionary[trade_id]
        trade_count += 1
        result = None

        # Iterate through market data starting from entry time
        for roi in roi_list:
            if (roi >= holding_value):
                result = 1
                break
            
            elif (roi <= stop_loss_value):
                result = 0
                break

        else: # for else, this is hit if we don't hit a break in the for loop
            # if here, the trade doesn't reach holding or stop loss or we ran out of market data. so it's NEITHER. skip it
            neither_count += 1
            continue

        entry_time_obj = datetime.strptime(row['Entry Time'] , '%H:%M:%S').time()
        minutes_since_open = (entry_time_obj.hour * 60 + entry_time_obj.minute) - (6 * 60 + 30)  # current minutes - market open minutes

        all_trade_data.append({
            'trade id': trade_id,
            'ticker': ticker,
            'minutes since market open': minutes_since_open, 
            'volatility percent': row['Entry Volatility Percent'],
            'result': result
            })
    
    results_df = pd.DataFrame(all_trade_data)

    return results_df, neither_count, trade_count


# validate training data before training model
def Helper_Train_Model_Validate_Input(results_df):
    # 1) check all columns are there
    required_cols = ['trade id', 'ticker', 'minutes since market open', 'volatility percent', 'result']
    missing_cols = [c for c in required_cols if c not in results_df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Train_Model missing required columns: {missing_cols}")

    # 2) crash if there's any nan values
    required_cols = ['minutes since market open', 'volatility percent', 'result']
    if results_df[required_cols].isnull().values.any():
        missing = results_df[required_cols].isnull().sum()
        raise ValueError(f"Missing values in required columns: {missing.to_dict()}")


# test each lam value by running it through each kfold to find deviance. pick best lam
def Helper_Train_Model_Find_Optimal_Lam(lam_grid, kfold, x_pre, y):
    best_lam = None
    best_cv_dev = np.inf
    best_cv_brier = np.inf
    best_cv_auc = np.nan

    for lam_value in lam_grid:
        fold_deviances = []
        fold_briers = []
        fold_aucs = []

        for train_idx, val_idx in kfold.split(x_pre, y):
            X_train_pre, X_val_pre = x_pre[train_idx], x_pre[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Fit scaler on train only to avoid leakage
            scaler = RobustScaler()
            X_train = scaler.fit_transform(X_train_pre)
            X_val = scaler.transform(X_val_pre)

            # Compute balanced sample weights on train fold
            classes = np.array([0, 1])
            class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            weight_map = {cls: w for cls, w in zip(classes, class_weights)}
            sample_weights = np.array([weight_map[cls] for cls in y_train])
            #sample_weights = sample_weights / sample_weights.mean()  # added to help with issue #1

            # Define and fit LogisticGAM with specified smooth terms
            gam = LogisticGAM(
                te(0, 1, n_splines=(4, 4)),
                lam=lam_value,
                max_iter=2000,
                verbose=False,
            )

            gam.fit(X_train, y_train, weights=sample_weights)

            # Probabilities for metrics
            y_prob = gam.predict_mu(X_val).reshape(-1)

            # Binomial deviance = 2 * log loss. basically equalivalent to gam.deviance() which is unsupported
            fold_dev = 2.0 * log_loss(y_val, y_prob, labels=[0, 1])
            fold_deviances.append(fold_dev)
            fold_briers.append(brier_score_loss(y_val, y_prob))

            # AUC requires both classes present in y_val
            if len(np.unique(y_val)) > 1:
                try:
                    auc = roc_auc_score(y_val, y_prob)
                except Exception:
                    auc = np.nan
            else:
                auc = np.nan
            fold_aucs.append(auc)

        mean_dev = float(np.mean(fold_deviances)) if len(fold_deviances) > 0 else np.inf
        mean_brier = float(np.mean(fold_briers)) if len(fold_briers) > 0 else np.inf
        mean_auc = float(np.nanmean(fold_aucs)) if len(fold_aucs) > 0 else np.nan

        # Primary criterion: minimize Brier score
        if mean_brier < best_cv_brier:
            best_cv_brier = mean_brier
            best_cv_dev = mean_dev
            best_cv_auc = mean_auc
            best_lam = lam_value

    if best_lam is None:
        # Fallback in extremely degenerate cases
        best_lam = 1.0

    return best_lam, best_cv_dev, best_cv_brier, best_cv_auc


''' How this works
-minutes since open (minutes) and volatility percent (vol) are vastly different numbers. we must scale them and THEN apply 
 standard scaler to them
-next, we do stratifiedkFold (5 folds) (this is out cross validation (cv))
-next, for each fold
    find lam (smoothing penality) per fold: we test different lam's over a gridsearch
    scale the features (avoids leakage by doing it in each fold) and apply weights to the classes (balanced)
    train the model: LogisticGAM(s(0) + s(1) + te(0, 1)) for this fold. that equation is optimal for limited data. 
        use low amount of splines also due to low data
    find the folds deviance, brier score, and roc auc
    for each lam test (outside of the folds) we find the mean deviance per lam value. we pick the best lam value
-finally, we scale the values for real, train model with weights

goals:
-lam: brier score and validation deviance, lower is better. both measure probability calibration quality
-lam: find ROC AUC on the folds. higher auc is better (up to 1). tells how well model discriminates between 1 and 0
      roc on a graph would show false positives vs true positives

variables
-result_df=dataframe of columns 'trade id', 'ticker', 'minutes since market open', 'volatility percent', 'result'. where result is 1 or 0. if a trade was neither of those it was previsouly dropped
-return: model, scaler, scaled x values, scaled y values
'''
def Train_Model(results_df):
    Helper_Train_Model_Validate_Input(results_df)

    # Extract features and target
    minutes_since_open = results_df['minutes since market open'].astype(float).to_numpy()
    volatility = results_df['volatility percent'].astype(float).to_numpy()
    y = results_df['result'].astype(int).to_numpy()

    # features are much different ranges from each other. so we must scale them before using something like standard scaler
    # minutes scaled to hours; volatility log1p to tame skew
    minutes_feat = minutes_since_open / 60.0
    volatility_feat = np.log1p(volatility)  # log1p won't create a back-transformation bias since we never revert it back to normal 
    x_pre = np.column_stack([minutes_feat, volatility_feat])

    # Cross-validation setup
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    lam_grid = 10 ** np.linspace(-4, 4, 13) #np.linspace(-2, 5, 8) #np.linspace(-4, 4, 13)

    # Select lambda by minimizing mean CV Brier score; also compute deviance and AUC
    best_lam, best_cv_dev, best_cv_brier, best_cv_auc = Helper_Train_Model_Find_Optimal_Lam(lam_grid, kfold, x_pre, y)

    # Fit final scaler and model on the full dataset using best lambda
    final_scaler = RobustScaler()
    X_scaled_full = final_scaler.fit_transform(x_pre)

    classes_full = np.array([0, 1])
    class_weights_full = compute_class_weight(class_weight='balanced', classes=classes_full, y=y)
    weight_map_full = {cls: w for cls, w in zip(classes_full, class_weights_full)}
    sample_weights_full = np.array([weight_map_full[cls] for cls in y])

    final_model = LogisticGAM(
        te(0, 1, n_splines=(4, 4)),
        lam=best_lam,
        max_iter=2000,
        verbose=False,
    )
    final_model.fit(X_scaled_full, y, weights=sample_weights_full)

    msg = (
        f"Selected lambda (lam): {best_lam} | "
        f"CV mean validation deviance: {best_cv_dev:.4f} | "
        f"CV mean Brier score: {best_cv_brier:.4f} | "
        f"CV mean ROC AUC: {best_cv_auc if not np.isnan(best_cv_auc) else 'nan'}"
    )
    print(msg)

    return final_model, final_scaler, X_scaled_full



'''
issue #1. 4 different runtime warnings during training. line: gam.fit(X_train, y_train, weights=sample_weights).
come from LogisticGAM’s IRLS steps driving the linear predictor to extreme values so that mu ≈ 0 or 1. The logit 
link’s derivatives blow up there: link.gradient(mu) = 1/(mu(1−mu)) → divide-by-zero, exp(eta) overflows, and 
subsequent multiplies/squares yield invalid/overflow. y has exact 0/1 (as expected), and during fitting some IRLS 
iterations produce eta with very large magnitude → mu numerically reaches 0 or 1. model is fairly flexible 
relative to 240 samples: s(0, 8) + s(1, 8) + te(0,1,(6,6)). With a small lam on some folds (your grid goes down 
to 1e-4), the fit can become nearly separable, which causes the logit coefficients to explode

changes: 
-lowered splines from 8 to 6 and (6,6) to (4,4)
-changed lam space to 10 ** np.linspace(-2, 5, 8) from 10**np.linspace(-4, 4, 13)
-changed sample weights to sample_weights = sample_weights / sample_weights.mean()
-changed Standardscaler to robust scaler (outliers)

'''