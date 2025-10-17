
import os
from re import X
import pandas as pd
import inspect
import sys
import glob
from pandas._config import dates
import datetime
from pygam import LogisticGAM, s, te
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import Helper_Functions


trade_start_indexes = {}
trade_end_timestamps = {}


def Add_Trade_Id(bulk_df):
    # Add a new 'Id' column that counts the trades starting from 1
    bulk_df['Id'] = range(1, len(bulk_df) + 1)

    return bulk_df


# Check for time gaps larger than 6 seconds between consecutive rows in market data.
def Confirm_No_Market_Data_Time_Gaps(market_df, file_path):
    # Convert Time column to datetime objects for comparison
    time_objects = []
    for time_str in market_df['Time']:
        # Parse hour:minute:second format
        time_obj = datetime.datetime.strptime(time_str, '%H:%M:%S').time()
        time_objects.append(time_obj)
    
    # Check for gaps between consecutive rows
    for i in range(1, len(time_objects)):
        prev_time = time_objects[i-1]
        curr_time = time_objects[i]
        
        # Calculate time difference in seconds
        prev_seconds = prev_time.hour * 3600 + prev_time.minute * 60 + prev_time.second
        curr_seconds = curr_time.hour * 3600 + curr_time.minute * 60 + curr_time.second
        
        time_diff = curr_seconds - prev_seconds
        
        # Since all data is from the same day, we only care about positive time gaps
        # Negative differences are just data collection timing variations (1-2 seconds backwards)
        # Only check for gaps larger than 6 seconds in the forward direction
        if time_diff > 6:
            prev_time_str = prev_time.strftime('%H:%M:%S')
            curr_time_str = curr_time.strftime('%H:%M:%S')
            raise ValueError(f"Time gap of {time_diff} seconds found between {prev_time_str} and {curr_time_str} (gap > 6 seconds) for file {file_path}")
    
    return True


def Load_Market_Data_Dictionary(bulk_df):
    """
    Load market data CSV files into a dictionary where key is date and value is dataframe.
    then drop unneeded columns
    then split it by ticker.
    
    Returns:
        {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    """
    market_data_dir = "Holder_Strat/Approved_Checked_Market_Data"
    market_data_dict = {}   # dictionary to store market data

    # Extract unique dates from the 'Date' column
    unique_dates = bulk_df['Date'].unique().tolist()
    # date needs to be mm-dd-yyyy format
    for i in range (len(unique_dates)):
        unique_dates[i] = Helper_Functions.bulk_csv_date_converter(unique_dates[i])

    print(f"Found {len(unique_dates)} unique dates in bulk_summaries.csv")
    
    # Step 1: Get list of market data files
    market_data_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]    
    
    # Step 2: Process each market data file
    for filename in market_data_files:
        date_from_filename = filename.split('_')[3]
        if ('.csv' in date_from_filename):
            date_from_filename = date_from_filename[0:-4]

        # Only load if date is in our unique dates list
        if date_from_filename in unique_dates:
            file_path = os.path.join(market_data_dir, filename)

            # Load the market data CSV
            market_df = pd.read_csv(file_path)
            if (Confirm_No_Market_Data_Time_Gaps(market_df, file_path) == True):
                # Keep only the specified columns
                required_columns = ['Ticker', 'Price', 'Volatility Percent', 'Time']
                market_df = market_df[required_columns]
                market_data_dict[date_from_filename] = market_df
                print(f"Loaded market data for {date_from_filename}")
            else:
                msg = f"time gap in {file_path}, crashing..."
                print(msg)
                raise ValueError(msg)

        else:
            print(f"Skipping {filename} - date {date_from_filename} not in bulk_summaries.csv")
    
    print(f"\nSuccessfully loaded {len(market_data_dict)} market data files")

    # step 3) go through the dictionary of df's and split by ticker
    nested_market_data_dict = {}
    
    for date, df in market_data_dict.items():
        # Get unique tickers for this date
        unique_tickers = df['Ticker'].unique()
        nested_market_data_dict[date] = {}
        
        # Split dataframe by ticker
        for ticker in unique_tickers:
            ticker_df = df[df['Ticker'] == ticker].copy()
            # Reset index to ensure sequential indexing starting from 0
            ticker_df = ticker_df.reset_index(drop=True)
            
            # Add 'Time Since Market Open' column (market opens at 6:30:00 AM)
            market_open_minutes = 6 * 60 + 30  # 6:30 AM in minutes
            time_since_market_open = []
            
            for time_str in ticker_df['Time']:
                time_obj = datetime.datetime.strptime(time_str, '%H:%M:%S').time()
                current_minutes = time_obj.hour * 60 + time_obj.minute + time_obj.second / 60.0
                minutes_since_open = current_minutes - market_open_minutes
                # Round to nearest minute
                minutes_since_open = round(minutes_since_open)
                time_since_market_open.append(minutes_since_open)
            
            ticker_df['Time Since Market Open'] = time_since_market_open
            nested_market_data_dict[date][ticker] = ticker_df
            
        print(f"Split {date} data into {len(unique_tickers)} tickers: {list(unique_tickers)}")

    return nested_market_data_dict


'''
-Make roi dictionary of lists for each trade -> {trade_id: [roi values], ...}
-iterate over the data. each trades stop loss is an roi of -0.4% until the trade reaches +0.6% roi, at which time the stop loss changes to 0% (entry price)
-For each trade, get its market data dataframe (market_data_dict_by_ticker[date][ticker]), start at the starting point for the trade (iterate over the 
data until we find the entry_time or the first timestamp after the entry_time (in case it's off by 1 second)). Then iterate over each row of market data;
for each market data row, add the trades roi to roi_list, note that trades direction can be "buy" (+roi is when the price is higher) or "short" (+roi is when the 
price is lower).

goal: create a list for each trade that has second by second roi
'''
def Create_Roi_Dictionary_For_Trades(bulk_df, market_data_dict_by_ticker, largest_sl_value):
    global trade_start_indexes, trade_end_timestamps

    roi_dictionary = {}
    skip_dates = []

    for idx, row in bulk_df.iterrows():
        date = Helper_Functions.bulk_csv_date_converter(row['Date'])  # 08-09-2025
        if (date in skip_dates):
            continue

        entry_time = row['Entry Time']               # hour:minute:second
        trade_id = row['Id']
        entry_price = row['Entry Price']
        ticker = row['Ticker']
        direction = row['Trade Type']
        roi_list = []
        
        # Check if we have market data for this date and ticker
        if date not in market_data_dict_by_ticker:
            print(f"No market data found for date {date}")
            skip_dates.append(date)
            continue  
            
        if ticker not in market_data_dict_by_ticker[date]:
            msg = f"No market data found for ticker {ticker} on date {date}"
            print(msg)
            raise ValueError(msg)
            
        market_df = market_data_dict_by_ticker[date][ticker].copy()  # market data df for this ticker and date
        
        # Skip trade if entry time is after final market data time
        if (Helper_Functions.Check_We_Have_Data_For_Trade(market_df, entry_time) == False):
            print(f"Skipping trade {trade_id}: entry time {entry_time} is after final market data time")
            continue
        
        # Find the starting point in market data (entry time or first timestamp after)
        entry_found = False
        start_index = 0
        
        for i in range(len(market_df)):
            market_time = market_df.iloc[i]['Time']
            if market_time >= entry_time:
                start_index = i
                entry_found = True
                trade_start_indexes[trade_id] = i
                break
        
        if not entry_found:
            msg = f"Entry time {entry_time} not found in market data for trade {trade_id}"
            print(msg)
            raise ValueError(msg)
            
        # Track stop loss state
        stop_loss_triggered = False
        stop_loss_updated = False  # Track if stop loss changed from -0.4% to 0%
        
        # Iterate through market data starting from entry point
        for i in range(start_index, len(market_df)):
            current_price = market_df.iloc[i]['Price']
            current_time = market_df.iloc[i]['Time']
            
            # get roi
            if direction == 'buy':
                roi = round(((current_price - entry_price) / entry_price) * 100, 2)
            elif direction == 'short':
                roi = round(((entry_price - current_price) / entry_price) * 100, 2)
            else:
                msg = f"Unknown trade direction: {direction} for trade {trade_id}"
                print(msg)
                raise ValueError(msg)
            
            # Check stop loss conditions
            if stop_loss_triggered == False:
                if roi <= largest_sl_value:
                    stop_loss_triggered = True
                    roi_list.append(roi)
                    break

                elif roi >= 0.6 and stop_loss_updated == False:
                    stop_loss_updated = True
                    roi_list.append(roi)

                elif stop_loss_updated == True and roi <= 0:
                    stop_loss_triggered = True
                    roi_list.append(roi)
                    break

                else:
                    roi_list.append(roi)
            else:
                # Stop loss already triggered, no more data needed
                break
        
        trade_end_timestamps[trade_id] = current_time
        roi_dictionary[trade_id] = roi_list
    
    return roi_dictionary


def Grid_Search(sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary):
    skip_dates = []
    result_df = pd.DataFrame(columns=['trade id', 'minutes since market open', 'volatility percent', 'stop loss', 'result'])
    holding_percent = 0.6
    
    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        date = Helper_Functions.bulk_csv_date_converter(row['Date'])  # 08-09-2025
        trade_id = row['Id']  # Move this up to avoid undefined variable error
        
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
        market_df = market_data_dict_by_ticker[date][ticker].copy()

        # Skip trade if entry time is after final market data time
        if (Helper_Functions.Check_We_Have_Data_For_Trade(market_df, entry_time) == False):
            print(f"Skipping trade {trade_id}: entry time {entry_time} is after final market data time")
            continue
        
        trade_id = row['Id']
        roi_list = roi_dictionary[trade_id]
        start_index = trade_start_indexes[trade_id]

        # for each sl_list value, record 1 for ones that allow the trade to reach holding, or 0 if they do not reach holding
        for sl in sl_list:
            counter = -1                # this makes it easier to look up roi from roi_list
            new_row = {
                'trade id': trade_id,
                'minutes since market open': market_df.iloc[start_index]['Time Since Market Open'],
                'volatility percent': market_df.iloc[start_index]['Volatility Percent'],
                'stop loss': sl,
                'result': None
            }
            result_df = result_df._append(new_row, ignore_index=True)
            new_index = len(result_df) - 1

            # Iterate through market data starting from entry point
            for i in range(start_index, len(market_df)):
                counter += 1
                curr_roi = roi_list[counter]

                if (curr_roi <= sl):
                    # trade over
                    result_df.at[new_index, 'result'] = int(0)
                    break

                elif (curr_roi >= holding_percent):
                    result_df.at[new_index, 'result'] = int(1)
                    break

            else: # triggers if a break didn't trigger in the for loop
                result_df.at[new_index, 'result'] = 'NaN'

    return result_df

'''
def Format_Data(result_df):
    """
    DEPRECATED: This function was designed for the old data structure.
    The new CatBoost implementation works directly with the DataFrame.
    
    This function converts the DataFrame format to the old list format
    for compatibility with the GAM implementation.
    """
    # Flatten data from DataFrame into master lists
    x_data = []     # [minutes_since_open, volatility_percent, stop_loss]
    y_data = []     # [result]
    trade_ids = []  # Keep track of which trade each sample came from
        
    for trade_id, trade_data in results.items():
        x_values = trade_data['x']  # List of [minutes_since_open, volatility_percent]
        y_values = trade_data['y']  # List of results (0, 1, or 'NaN')
        
        # Each index corresponds to the same stop loss in sl_list
        for i in range(len(x_values)):
            # Skip NaN values
            if y_values[i] == 'NaN':
                continue
                
            # Extract features
            minutes_since_open = x_values[i][0]
            volatility_percent = x_values[i][1]
            stop_loss = sl_list[i]
            
            # Add to master lists
            x_data.append([minutes_since_open, volatility_percent, stop_loss])
            y_data.append(y_values[i])
            trade_ids.append(trade_id)
    
    return x_data, y_data, trade_ids
'''

# NOTE: x is scaled data
def Model_Diagnostics(model, results_df, scaler):
    """
    Comprehensive model evaluation for CatBoost trading classifier.
    
    Evaluates:
    - Classification metrics (accuracy, precision, recall, F1)
    - Confusion matrix
    - Feature importance
    - Prediction distribution
    - Class-specific performance
    - Feature partial dependence (if possible)
    """
    print("="*60)
    print("MODEL DIAGNOSTICS - CatBoost Trading Classifier")
    print("="*60)
    
    # Prepare data
    x_cols = ['minutes since market open', 'volatility percent', 'stop loss']
    results_df = results_df.copy()
    # Handle missing values same as training - drop NaN rows
    results_df['result'] = results_df['result'].replace('NaN', pd.NA)
    results_df = results_df.dropna(subset=['result'])
    
    # Handle float to int conversion (0.0 -> 0, 1.0 -> 1) same as training
    valid_numeric_values = {0, 1, 0.0, 1.0}
    actual_values = set(results_df['result'].unique())
    if not actual_values.issubset(valid_numeric_values):
        results_df = results_df[results_df['result'].isin([0, 1, 0.0, 1.0])]
    results_df['result'] = results_df['result'].astype(int)
    
    X_raw = results_df[x_cols].values
    y = results_df['result'].values
    X_scaled = scaler.transform(X_raw)  # Use transform, not fit_transform
    
    # Check class distribution for stratification
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_class_count = min(class_counts)
    
    # Split data (same as training)
    if min_class_count < 2:
        # Use regular split without stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    else:
        # Use stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
    
    # 1. BASIC MODEL PERFORMANCE
    print("\n1. BASIC MODEL PERFORMANCE")
    print("-" * 30)
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    
    print(f"Training Accuracy: {train_acc:.3f}")
    print(f"Validation Accuracy: {val_acc:.3f}")
    print(f"Overfitting Check: {train_acc - val_acc:.3f} (< 0.05 is good)")
    
    # 2. DETAILED CLASSIFICATION REPORT
    print("\n2. DETAILED CLASSIFICATION REPORT")
    print("-" * 40)
    class_names = ['Stopped Out (0)', 'Reached Target (1)']
    print("Validation Set Performance:")
    print(classification_report(y_val, val_pred, target_names=class_names))
    
    # 3. CONFUSION MATRIX
    print("\n3. CONFUSION MATRIX")
    print("-" * 20)
    cm = confusion_matrix(y_val, val_pred)
    print("Validation Confusion Matrix:")
    print("Rows = Actual, Columns = Predicted")
    print(f"{'':>12} {'Stop(0)':>8} {'Target(1)':>9}")
    for i, class_name in enumerate(['Stop(0)', 'Target(1)']):
        print(f"{class_name:>12} {cm[i,0]:>8} {cm[i,1]:>9}")
    
    # 4. FEATURE IMPORTANCE
    print("\n4. FEATURE IMPORTANCE")
    print("-" * 25)
    feature_importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': x_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']:.<35} {row['importance']:>6.1f}")
    
    # 5. CLASS DISTRIBUTION ANALYSIS
    print("\n5. CLASS DISTRIBUTION ANALYSIS")
    print("-" * 35)
    
    # Original distribution
    orig_dist = pd.Series(y).value_counts().sort_index()
    print("Original Data Distribution:")
    for class_val, count in orig_dist.items():
        pct = count / len(y) * 100
        print(f"  Class {class_val} ({class_names[class_val]}): {count:>5} ({pct:>5.1f}%)")
    
    # Prediction distribution
    pred_dist = pd.Series(val_pred).value_counts().sort_index()
    print("\nValidation Predictions Distribution:")
    for class_val, count in pred_dist.items():
        pct = count / len(val_pred) * 100
        print(f"  Class {class_val} ({class_names[class_val]}): {count:>5} ({pct:>5.1f}%)")
    
    # 6. PREDICTION CONFIDENCE ANALYSIS
    print("\n6. PREDICTION CONFIDENCE ANALYSIS")
    print("-" * 40)
    val_proba = model.predict_proba(X_val)
    max_proba = np.max(val_proba, axis=1)
    
    print(f"Average Prediction Confidence: {np.mean(max_proba):.3f}")
    print(f"Min Confidence: {np.min(max_proba):.3f}")
    print(f"Max Confidence: {np.max(max_proba):.3f}")
    
    # Confidence by class
    for class_val in range(2):
        class_mask = val_pred == class_val
        if np.any(class_mask):
            class_conf = max_proba[class_mask]
            print(f"Avg confidence for {class_names[class_val]}: {np.mean(class_conf):.3f}")
    
    # 7. FEATURE STATISTICS
    print("\n7. FEATURE STATISTICS")
    print("-" * 25)
    print("Feature ranges in training data:")
    for i, feature in enumerate(x_cols):
        feature_values = X_raw[:, i]
        print(f"  {feature:.<35} {np.min(feature_values):>8.2f} to {np.max(feature_values):>8.2f}")
    
    # 8. TRADING-SPECIFIC METRICS
    print("\n8. TRADING-SPECIFIC ANALYSIS")
    print("-" * 35)
    
    # Success rate by stop loss value
    print("Success Rate by Stop Loss Value:")
    for sl_val in sorted(results_df['stop loss'].unique()):
        sl_mask = results_df['stop loss'] == sl_val
        sl_results = results_df[sl_mask]['result']
        success_rate = (sl_results == 1).mean()
        total_trades = len(sl_results)
        print(f"  SL {sl_val:>5.1f}%: {success_rate:>6.1%} success ({total_trades:>4} trades)")
    
    # Success rate by time of day
    print("\nSuccess Rate by Time Since Market Open:")
    time_bins = [0, 60, 120, 180, 240, 300, 999]  # 0-1hr, 1-2hr, etc.
    time_labels = ['0-1h', '1-2h', '2-3h', '3-4h', '4-5h', '5h+']
    
    results_df['time_bin'] = pd.cut(results_df['minutes since market open'], 
                                   bins=time_bins, labels=time_labels, include_lowest=True)
    
    for time_label in time_labels:
        time_mask = results_df['time_bin'] == time_label
        if time_mask.sum() > 0:
            time_results = results_df[time_mask]['result']
            success_rate = (time_results == 1).mean()
            total_trades = len(time_results)
            print(f"  {time_label:>4}: {success_rate:>6.1%} success ({total_trades:>4} trades)")
    
    # 9. CREATE VISUALIZATIONS
    print("\n9. GENERATING VISUALIZATIONS...")
    print("-" * 35)
    
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Feature Importance
    plt.subplot(3, 3, 1)
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    
    # Plot 2: Confusion Matrix Heatmap
    plt.subplot(3, 3, 2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Stop', 'Target'],
                yticklabels=['Stop', 'Target'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    # Plot 3: Prediction Confidence Distribution
    plt.subplot(3, 3, 3)
    plt.hist(max_proba, bins=30, alpha=0.7, edgecolor='black')
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Max Probability')
    plt.ylabel('Count')
    
    # Plot 4: Success Rate by Stop Loss
    plt.subplot(3, 3, 4)
    sl_success = []
    sl_values = sorted(results_df['stop loss'].unique())
    for sl_val in sl_values:
        sl_mask = results_df['stop loss'] == sl_val
        success_rate = (results_df[sl_mask]['result'] == 1).mean()
        sl_success.append(success_rate)
    
    plt.plot(sl_values, sl_success, 'bo-')
    plt.title('Success Rate by Stop Loss')
    plt.xlabel('Stop Loss (%)')
    plt.ylabel('Success Rate')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Success Rate by Time of Day
    plt.subplot(3, 3, 5)
    time_success = []
    time_counts = []
    for time_label in time_labels:
        time_mask = results_df['time_bin'] == time_label
        if time_mask.sum() > 0:
            success_rate = (results_df[time_mask]['result'] == 1).mean()
            time_success.append(success_rate)
            time_counts.append(time_mask.sum())
        else:
            time_success.append(0)
            time_counts.append(0)
    
    plt.bar(time_labels, time_success, alpha=0.7)
    plt.title('Success Rate by Time Since Open')
    plt.xlabel('Time Period')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=45)
    
    # Plot 6: Volatility vs Success Rate
    plt.subplot(3, 3, 6)
    vol_bins = np.percentile(results_df['volatility percent'], [0, 25, 50, 75, 100])
    results_df['vol_quartile'] = pd.cut(results_df['volatility percent'], bins=vol_bins, 
                                       labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
    
    vol_success = []
    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        vol_mask = results_df['vol_quartile'] == quartile
        if vol_mask.sum() > 0:
            success_rate = (results_df[vol_mask]['result'] == 1).mean()
            vol_success.append(success_rate)
        else:
            vol_success.append(0)
    
    plt.bar(['Q1', 'Q2', 'Q3', 'Q4'], vol_success, alpha=0.7)
    plt.title('Success Rate by Volatility Quartile')
    plt.xlabel('Volatility Quartile')
    plt.ylabel('Success Rate')
    
    # Plot 7: Class Distribution Comparison
    plt.subplot(3, 3, 7)
    x_pos = np.arange(2)
    actual_pcts = [orig_dist[i] / len(y) * 100 for i in range(2)]
    pred_pcts = [pred_dist.get(i, 0) / len(val_pred) * 100 for i in range(2)]
    
    width = 0.35
    plt.bar(x_pos - width/2, actual_pcts, width, label='Actual', alpha=0.7)
    plt.bar(x_pos + width/2, pred_pcts, width, label='Predicted', alpha=0.7)
    plt.title('Class Distribution: Actual vs Predicted')
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.xticks(x_pos, ['Stop', 'Target'])
    plt.legend()
    
    # Plot 8: Feature Correlation with Success
    plt.subplot(3, 3, 8)
    success_mask = results_df['result'] == 1
    fail_mask = results_df['result'] == 0
    
    feature_idx = 1  # Volatility percent
    success_vals = results_df[success_mask]['volatility percent']
    fail_vals = results_df[fail_mask]['volatility percent']
    
    plt.hist(success_vals, bins=20, alpha=0.5, label='Success', density=True)
    plt.hist(fail_vals, bins=20, alpha=0.5, label='Stopped Out', density=True)
    plt.title('Volatility Distribution by Outcome')
    plt.xlabel('Volatility Percent')
    plt.ylabel('Density')
    plt.legend()
    
    # Plot 9: Model Performance Summary
    plt.subplot(3, 3, 9)
    metrics = ['Train Acc', 'Val Acc', 'Precision', 'Recall', 'F1-Score']
    
    # Calculate weighted averages for multi-class metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_val, val_pred, average='weighted')
    recall = recall_score(y_val, val_pred, average='weighted')
    f1 = f1_score(y_val, val_pred, average='weighted')
    
    values = [train_acc, val_acc, precision, recall, f1]
    colors = ['green' if v > 0.6 else 'orange' if v > 0.5 else 'red' for v in values]
    
    plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Model Performance Summary')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = "Holder_Strat/Parameter_Tuning/model_files_and_data/catboost_model_diagnostics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Diagnostic plots saved to: {plot_path}")
    plt.show()
    
    print("\n" + "="*60)
    print("MODEL DIAGNOSTICS COMPLETE")
    print("="*60)


def Format_CatBoost_Data(results_df):
    # Data validation and cleaning
    print(f"Input data shape: {results_df.shape}")
    print(f"Columns: {results_df.columns.tolist()}")
    
    # Replace string 'NaN' with actual pandas NaN, then drop rows with NaN results
    results_df['result'] = results_df['result'].replace('NaN', pd.NA)
    results_df = results_df.dropna(subset=['result'])  # Drop rows where 'result' is NaN
    
    # Validation and conversion - handle float values like 0.0, 1.0
    print(f"Before conversion - Unique values: {sorted(results_df['result'].unique())}")
    print(f"Data types in result column: {results_df['result'].dtype}")

    # First ensure all values are numeric and either 0 or 1 (allowing for 0.0, 1.0)
    valid_numeric_values = {0, 1, 0.0, 1.0}
    actual_values = set(results_df['result'].unique())
    
    if not actual_values.issubset(valid_numeric_values):
        invalid_values = actual_values - valid_numeric_values
        msg = f"ERROR: Found truly invalid values in result column: {invalid_values}"
        print(msg)
        raise ValueError(msg)
    
    # Convert all values to integers (this handles 0.0 -> 0, 1.0 -> 1). the floats are likely an annoying thing due to using pandas
    results_df['result'] = results_df['result'].astype(int)
    
    print(f"After cleaning - Data shape: {results_df.shape}")
    print(f"Unique values in result column: {sorted(results_df['result'].unique())}")

    # Check class distribution - if a class has too few samples we should drop it
    y = results_df['result'].values
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"Class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")
    
    min_class_count = min(class_counts)
    if min_class_count < 20:
        msg = (f"WARNING: Minimum class count is {min_class_count}, which is too few for stratified splitting. basically there's like no samples "
               f"of one of your y values. this model fails if you have fewer than 2 samples")
        print(msg)
        raise ValueError(msg)

    return results_df


def Train_CatBoost_Model(results_df):
    x_cols = ['minutes since market open', 'volatility percent', 'stop loss']
    
    results_df = Format_CatBoost_Data(results_df)

    # Prepare feature matrix - ALL features need to be scaled together
    x_raw = results_df[x_cols].values  # Convert to numpy array
    y = results_df['result'].values
    
    # Scale all features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    
    print(f"Feature matrix shape: {x_scaled.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train/validation split with stratification
    x_train, x_val, y_train, y_val = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {x_train.shape}, Validation set: {x_val.shape}")

    # Convert to CatBoost Pool
    train_pool = Pool(x_train, y_train, feature_names=x_cols)
    val_pool = Pool(x_val, y_val, feature_names=x_cols)

    # Train CatBoost model
    model = CatBoostClassifier(
        iterations=600,              # total trees
        depth=5,                     # shallower = less overfit
        learning_rate=0.03,          # slower = smoother learning
        loss_function='Logloss',     # MultiClass for 3-class problem (0, 1, 2), binary for 2
        eval_metric='Logloss',
        l2_leaf_reg=4,               # L2 regularization
        random_seed=42,              # for reproducibility
        early_stopping_rounds=50,    # stop if validation not improving
        verbose=100,                 # print every 100 iterations
        # class_weights not needed for binary classification (you can change them like: class_weights=[1, 0.7])
    )

    print("Training CatBoost model...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print("Done training")
    
    # Model evaluation
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    
    train_acc = np.mean(train_pred == y_train)
    val_acc = np.mean(val_pred == y_val)
    
    print(f"\nModel Performance:")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")
    
    # Feature importance
    feature_importance = model.get_feature_importance()
    print(f"\nFeature Importance:")
    for i, (feature, importance) in enumerate(zip(x_cols, feature_importance)):
        print(f"  {feature}: {importance:.3f}")
    
    ''' Example usage after training:
    # For new prediction, make sure to scale the input
    example_raw = np.array([[45, 0.22, -0.5]])  # minutes, volatility, stop loss
    example_scaled = scaler.transform(example_raw)
    pred_proba = model.predict_proba(example_scaled)
    pred_class = model.predict(example_scaled)
    print(f"Prediction probabilities: {pred_proba}")
    print(f"Predicted class: {pred_class}")
    '''

    return model, results_df, scaler


'''
# P(success = 1 | minutes_since_open, vol%, stop_value)
# results[trade_id] has ['x'] and ['y']. each are a list where the indexes match sl_list indexes
# 1 entry for each stop loss: ['x'] = [minutes since market open, volatility %], ['y'] = result (1, 0, or 'NaN')
def OLD_Train_GAM(result_df):
    """
    Train a GAM classifier to predict trade success probability.
    
    Features:
    - minutes_since_open: Time since market open when trade was entered
    - volatility_percent: Market volatility at trade entry
    - stop_loss: Stop loss value being tested
    
    Target:
    - result: 1 if trade reached holding target, 0 if stopped out, NaN if neither (filtered out)
    """

    x_data, y_data, trade_ids = Format_Data(result_df)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)

    # Convert to numpy arrays
    x = np.array(x_scaled)
    y = np.array(y_data)
    
    print(f"Total samples after filtering NaN: {len(x)}")
    print(f"Success rate: {np.mean(y):.3f}")
    print(f"Feature ranges:")
    print(f"  Minutes since open: {x[:, 0].min():.1f} to {x[:, 0].max():.1f}")
    print(f"  Volatility percent: {x[:, 1].min():.3f} to {x[:, 1].max():.3f}")
    print(f"  Stop loss: {x[:, 2].min():.1f} to {x[:, 2].max():.1f}")
    
    # Train GAM classifier with binomial (logistic) link
    # Use smooth functions for continuous variables
    gam = LogisticGAM(
        s(0, n_splines=10) +  # Smooth function for minutes_since_open
        s(1, n_splines=10) +  # Smooth function for volatility_percent
        s(2, n_splines=6)     # Smooth function for stop_loss (fewer splines since fewer unique values)
    )
    
    print("Training GAM classifier...")
    gam.fit(x, y)
    print("Training finished")
    
    return gam, x, y, scaler
'''

def Save_Regression_Model_And_Data(model, result_df, scaler):
    model_path = "Holder_Strat/Parameter_Tuning/model_files_and_data/trained_sl_predictor_model.pkl"
    values_path = "Holder_Strat/Parameter_Tuning/model_files_and_data/sl_model_training_data.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump({'gam': model, 'scaler': scaler}, f)

    with open(values_path, 'wb') as f:
        pickle.dump({'result_df': result_df}, f)

    print(f"Model and scaler saved to: {model_path}")
    print(f"Model data saved to: {values_path}")


def Load_Regression_Model():
    try:
        model_path = "Holder_Strat/Parameter_Tuning/model_files_and_data/trained_sl_predictor_model.pkl"
        values_path = "Holder_Strat/Parameter_Tuning/model_files_and_data/sl_model_training_data.pkl"

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['gam']
            scaler = model_data['scaler']
        
        with open(values_path, 'rb') as f:
            result_df = pickle.load(f)['result_df']
        
        print("Successfully loaded model, scaler, and training data")

        return model, result_df, scaler
        
    except FileNotFoundError:
        print(f"{model_path} or {values_path} not found. Please train and save a model first.")
        return None, None, None

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def Give_Model_Test_Input(model, scaler):
    ex1 = [[45, 0.22, -0.4], [45, 0.22, -0.5], [45, 0.22, -0.6], [45, 0.22, -0.7], [45, 0.22, -0.8]]
    for input in ex1:
        input = np.array(input).reshape(1, -1)  # Reshape to 2D array (1 sample, n features)
        input_scaled = scaler.transform(input)  # Use transform, not fit_transform (fit transform recalculates scaling)
        probability = model.predict_proba(input_scaled)
        pred_class = model.predict(input_scaled)
        print(f"Input: {input.flatten()}")
        print(f"Prediction probabilities: {probability}") # [[class 1 prob, class 2 prob]]
        print(f"Predicted class: {pred_class}")
        print("---")


def main():
    sl_list = [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    '''
    bulk_df = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[["Date", "Ticker", "Entry Time", "Time in Trade", "Entry Price", "Exit Price", "Trade Type", "Exit Price", "Entry Volatility Percent", "Original Holding Reached", "Original Best Exit Percent", "Original Percent Change"]]
    bulk_df = Add_Trade_Id(bulk_df)

    market_data_dict_by_ticker = Load_Market_Data_Dictionary(bulk_df) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    print("\nmaking roi dictionary...")
    roi_dictionary = Create_Roi_Dictionary_For_Trades(bulk_df, market_data_dict_by_ticker, sl_list[-1])
    print(f"roi dictionary done. Total trades processed: {len(roi_dictionary)}\n")
    
    # collect data results = [trade_id]['x'] and [trade_id]['y']
    results_df = Grid_Search(sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary)
    
    model, result_df, scaler = Train_CatBoost_Model(results_df)
    Save_Regression_Model_And_Data(model, result_df, scaler)
    '''
    model, result_df, scaler = Load_Regression_Model()

    Give_Model_Test_Input(model, scaler)

    #Model_Diagnostics(model, result_df, scaler)




main()