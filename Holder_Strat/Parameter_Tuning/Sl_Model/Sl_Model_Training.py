import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

version = None
sl_dir = "Holder_Strat/Parameter_Tuning/Sl_Model"

def Set_Version(passed_version):
    global version
    version = passed_version


def Save_Training_Data(trades_processed, trades_dropped_neither, neither_trades_detected):
    file_path = f"{sl_dir}/Data/Training_Data.txt"
    with open(file_path, 'wb') as f:
        pickle.dump({'trades_processed': trades_processed, 'trades_dropped_neither': trades_dropped_neither, 
                     'neither_trades_detected': neither_trades_detected}, f)

    print(f"Training data saved to: {file_path}")


def Load_Training_data():
    file_path = f"{sl_dir}/Data/Training_Data.txt"

    with open(file_path, 'f') as f:
        data = json.load(f)

    print(f"Training data loaded from: {file_path}")

    return data['results_df'], data['trades_processed'], data['trades_dropped_neither'], data['neither_trades_detected']


def Save_Model_Data(model, scaler, result_df):
    file_path = f"{sl_dir}/Data/trained_sl_predictor_model.pkl"
    
    with open(file_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'result_df': result_df}, f)

    print(f"Model and scaler saved to: {file_path}")


def Load_Model_data():
    file_path = f"{sl_dir}/Data/trained_sl_predictor_model.pkl"
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Model and scaler loaded from: {file_path}")

    return data['model'], data['scaler'], data['result_df']


def bulk_csv_date_converter(date):
    parts = date.split('-')

    if (len(parts[0]) == 1):
        parts[0] = f"0{parts[0]}"
    if (len(parts[1]) == 1):
        parts[1] = f"0{parts[1]}"
    if (len(parts[2]) == 2):
        parts[2] = f"20{parts[2]}"
    
    return '-'.join(parts)


# called by train_model
def Format_Optimal_SL_Data(results_df):
    """
    Data validation and cleaning for optimal stop loss regression.
    """
    print(f"Input data shape: {results_df.shape}")
    print(f"Columns: {results_df.columns.tolist()}")
    
    # Check for missing values
    missing_counts = results_df.isnull().sum()
    if missing_counts.any():
        print("Missing values found:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"  {col}: {count} missing")
        
        # Drop rows with missing optimal_stop_loss
        results_df = results_df.dropna(subset=['optimal_stop_loss'])
        print(f"After dropping missing values: {results_df.shape}")
    
    # Validate optimal_stop_loss values are reasonable
    sl_values = results_df['optimal_stop_loss'].values
    print(f"Optimal stop loss range: {sl_values.min():.3f} to {sl_values.max():.3f}")
    
    # Check if values are within expected range (should be negative percentages)
    if sl_values.max() > 0:
        print("WARNING: Found positive stop loss values - this might be incorrect")
    if sl_values.min() < -2.0:
        print("WARNING: Found very large negative stop loss values - this might be incorrect")
    
    print(f"Final data shape: {results_df.shape}")
    return results_df


def Collect_data(sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes, drop_neither_trades=True):
    """
    For each trade, find the optimal stop loss that maximizes success probability.
    
    Args:
        drop_neither_trades: If True, exclude trades that reach neither stop loss nor target.
                           If False, treat them as failures (original behavior).
    
    Returns:
        tuple: (result_df, trades_processed, trades_dropped_neither, neither_trades_detected)
        - result_df: DataFrame with columns: ['trade id', 'minutes since market open', 'volatility percent', 'optimal_stop_loss']
        - trades_processed: Number of trades successfully processed
        - trades_dropped_neither: Number of trades dropped due to 'neither' outcomes
        - neither_trades_detected: Total number of trades that had 'neither' outcomes (regardless of drop setting)
    """
    skip_dates = []
    result_df = pd.DataFrame(columns=['trade id', 'minutes since market open', 'volatility percent', 'optimal_stop_loss'])
    holding_percent = 0.6
    
    trades_processed = 0
    trades_dropped_neither = 0
    neither_trades_detected = 0  # Track total 'neither' trades found
    
    print(f"Finding optimal stop loss for each trade using SL candidates: {sl_list}")
    print(f"Drop 'neither' trades: {drop_neither_trades}")
    
    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        entry_time = row['Entry Time']               # hour:minute:second
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025
        market_df = market_data_dict_by_ticker[date][ticker]
        trade_id = row['Trade Id']
        final_timestamp = datetime.strptime(market_df.at[-1, 'Time'], '%H:%M:%S').time()
        final_seconds = final_timestamp.hour * 3600 + final_timestamp.minute * 60 + final_timestamp.second
        
        # if we don't have market data for this date
        if (date in skip_dates):
            continue
        # if we don't have market data for this date
        if date not in market_data_dict_by_ticker:
            print(f"No market data found for date {date}")
            skip_dates.append(date)
            continue  
        # if we don't have ticker data for this date
        if ticker not in market_data_dict_by_ticker[date]:
            msg = f"No market data found for ticker {ticker} on date {date}"
            print(msg)
            raise ValueError(msg)
        # if we don't have market data for this trade
        entry_time_obj = datetime.strptime(entry_time, '%H:%M:%S').time()
        entry_seconds = entry_time_obj.hour * 3600 + entry_time_obj.minute * 60 + entry_time_obj.second
        if (entry_seconds > final_seconds):
            print(f"Skipping trade {trade_id}: entry time {entry_time} is after final market data time")
            continue
        
        roi_list = roi_dictionary[trade_id]
        start_index = trade_start_indexes[trade_id]

        # Test each stop loss and find the optimal one for this trade
        sl_results = {}  # {stop_loss: success(1) or failure(0)}
        has_neither_outcome = False
        
        for sl in sl_list:
            counter = -1
            
            # Simulate this stop loss on the trade
            for i in range(start_index, len(market_df)):
                counter += 1
                if counter >= len(roi_list):
                    # Ran out of ROI data - this is a "neither" case
                    if drop_neither_trades:
                        has_neither_outcome = True
                        break
                    else:
                        sl_results[sl] = 0  # Consider as failure (original behavior)
                        break
                    
                curr_roi = roi_list[counter]

                if (curr_roi <= sl):
                    # Stop loss hit - failure
                    sl_results[sl] = 0
                    break
                elif (curr_roi >= holding_percent):
                    # Target reached - success
                    sl_results[sl] = 1
                    break
            else:
                # Neither stop loss nor target hit - this is a "neither" case
                if drop_neither_trades:
                    has_neither_outcome = True
                else:
                    sl_results[sl] = 0  # Consider as failure (original behavior)
            
            # If we found a "neither" case and we're dropping them, skip this entire trade
            if has_neither_outcome and drop_neither_trades:
                break

        # Track 'neither' trades detected (regardless of drop setting)
        if has_neither_outcome:
            neither_trades_detected += 1
            
        # Skip this trade if it has "neither" outcomes and we're dropping them
        if has_neither_outcome and drop_neither_trades:
            trades_dropped_neither += 1
            continue
            
        trades_processed += 1

        # Find the optimal stop loss: the loosest SL that still allows success
        # Priority: 1) Must reach target, 2) Prefer looser stop loss for better risk/reward
        successful_sls = [sl for sl, success in sl_results.items() if success == 1]
        
        if successful_sls:
            # Choose the loosest (least negative) stop loss that still succeeds
            optimal_sl = max(successful_sls)  # max of [-0.3, -0.4, -0.5] = -0.3 (loosest)
        else:
            # No stop loss worked, choose the loosest one anyway (best risk/reward)
            optimal_sl = max(sl_list)  # Most conservative choice
        
        # Add one row per trade with its optimal stop loss
        new_row = {
            'trade id': trade_id,
            'minutes since market open': market_df.iloc[start_index]['Time Since Market Open'],
            'volatility percent': market_df.iloc[start_index]['Volatility Percent'],
            'optimal_stop_loss': optimal_sl
        }
        result_df = result_df._append(new_row, ignore_index=True)

    print(f"\nTraining Data Creation Results:")
    print(f"  Trades processed: {trades_processed}")
    print(f"  'Neither' trades detected: {neither_trades_detected}")
    if drop_neither_trades:
        total_trades = trades_processed + trades_dropped_neither
        drop_rate = (trades_dropped_neither / total_trades) * 100 if total_trades > 0 else 0
        print(f"  Trades dropped ('neither'): {trades_dropped_neither}")
        print(f"  Drop rate: {drop_rate:.1f}%")
    else:
        print(f"  'Neither' trades treated as failures: {neither_trades_detected}")
    
    # Show distribution of optimal stop losses
    if len(result_df) > 0:
        sl_counts = result_df['optimal_stop_loss'].value_counts().sort_index()
        print(f"\nOptimal Stop Loss Distribution ({len(result_df)} trades):")
        for sl, count in sl_counts.items():
            pct = count / len(result_df) * 100
            print(f"  {sl:>6.1f}%: {count:>4} trades ({pct:>5.1f}%)")
    
    return result_df, trades_processed, trades_dropped_neither, neither_trades_detected


def Train_Model(results_df):
    """
    Train CatBoost Regressor to predict optimal stop loss.
    
    Features: ['minutes since market open', 'volatility percent'] 
    Target: 'optimal_stop_loss' (continuous values like -0.3, -0.4, etc.)
    """
    x_cols = ['minutes since market open', 'volatility percent']  # Only entry conditions, no stop loss
    
    results_df = Format_Optimal_SL_Data(results_df)

    # Prepare feature matrix
    x_raw = results_df[x_cols].values            # Convert to numpy array
    y = results_df['optimal_stop_loss'].values   # Target is now optimal stop loss
    
    # Scale features (but not target - we want to predict actual stop loss values)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    
    print(f"Feature matrix shape: {x_scaled.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target range: {y.min():.3f} to {y.max():.3f}")
    print(f"Target distribution:")
    unique_vals, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique_vals, counts):
        pct = count / len(y) * 100
        print(f"  {val:>6.1f}%: {count:>4} samples ({pct:>5.1f}%)")
    
    # Train/validation split (no stratification needed for regression)
    x_train, x_val, y_train, y_val = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {x_train.shape}, Validation set: {x_val.shape}")

    # Convert to CatBoost Pool
    train_pool = Pool(x_train, y_train, feature_names=x_cols)
    val_pool = Pool(x_val, y_val, feature_names=x_cols)

    # Train CatBoost Regressor (not Classifier!)
    model = CatBoostRegressor(
        iterations=600,              # total trees
        depth=4,                     # shallower for regression
        learning_rate=0.05,          # slightly higher for regression
        loss_function='RMSE',        # Root Mean Square Error for regression
        eval_metric='RMSE',
        l2_leaf_reg=3,               # L2 regularization
        random_seed=42,              # for reproducibility
        early_stopping_rounds=50,    # stop if validation not improving
        verbose=100,                 # print every 100 iterations
    )

    print("Training model...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print("Done training")
    
    # Model evaluation for regression
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    
    # Regression metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    print(f"\nModel Performance (Regression Metrics):")
    print(f"Training RMSE:   {train_rmse:.4f}")
    print(f"Validation RMSE: {val_rmse:.4f}")
    print(f"Training MAE:    {train_mae:.4f}")
    print(f"Validation MAE:  {val_mae:.4f}")
    print(f"Training R²:     {train_r2:.4f}")
    print(f"Validation R²:   {val_r2:.4f}")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    print(f"\nFeature Importance:")
    for i, (feature, importance) in enumerate(zip(x_cols, feature_importance)):
        print(f"  {feature}: {importance:.3f}")
    
    # Show prediction examples
    print(f"\nSample Predictions vs Actual:")
    for i in range(min(10, len(val_pred))):
        print(f"  Predicted: {val_pred[i]:>6.3f}, Actual: {y_val[i]:>6.3f}, Diff: {abs(val_pred[i] - y_val[i]):>6.3f}")
    
    return model, results_df, scaler
