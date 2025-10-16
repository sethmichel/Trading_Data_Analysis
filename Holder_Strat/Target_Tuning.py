import os
from re import X
import pandas as pd
import inspect
import sys
import glob
from pandas._config import dates
import datetime
from pygam import LinearGAM, s, te
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

trade_start_indexes = {}
trade_end_timestamps = {}


def bulk_csv_date_converter(date):
    parts = date.split('-')

    if (len(parts[0]) == 1):
        parts[0] = f"0{parts[0]}"
    if (len(parts[1]) == 1):
        parts[1] = f"0{parts[1]}"
    if (len(parts[2]) == 2):
        parts[2] = f"20{parts[2]}"
    
    return '-'.join(parts)


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
        unique_dates[i] = bulk_csv_date_converter(unique_dates[i])

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


def Add_Trade_Id(bulk_df):
    # Add a new 'Id' column that counts the trades starting from 1
    bulk_df['Id'] = range(1, len(bulk_df) + 1)

    return bulk_df


def Check_We_Have_Data_For_Trade(market_df, entry_time):
    # Get the final timestamp from market data
    final_market_time = market_df.iloc[-1]['Time']
    
    # Convert times to datetime objects for comparison
    entry_time_obj = datetime.datetime.strptime(entry_time, '%H:%M:%S').time()
    final_market_time_obj = datetime.datetime.strptime(final_market_time, '%H:%M:%S').time()
    
    # Convert to seconds for easier comparison
    entry_seconds = entry_time_obj.hour * 3600 + entry_time_obj.minute * 60 + entry_time_obj.second
    final_market_seconds = final_market_time_obj.hour * 3600 + final_market_time_obj.minute * 60 + final_market_time_obj.second
    
    if entry_seconds > final_market_seconds:
        return False
    else:
        return True
    

'''
-Make roi dictionary of lists for each trade -> {trade_id: [roi values], ...}
-iterate over the data. each trades stop loss is an roi of -0.4% until the trade reaches +0.6% roi, at which time the stop loss changes to 0% (entry price)
-For each trade, get its market data dataframe (market_data_dict_by_ticker[date][ticker]), start at the starting point for the trade (iterate over the 
data until we find the entry_time or the first timestamp after the entry_time (in case it's off by 1 second)). Then iterate over each row of market data;
for each market data row, add the trades roi to roi_list, note that trades direction can be "buy" (+roi is when the price is higher) or "short" (+roi is when the 
price is lower).

goal: create a list for each trade that has second by second roi
'''
def Create_Roi_Dictionary_For_Trades(bulk_df, market_data_dict_by_ticker):
    global trade_start_indexes, trade_end_timestamps

    roi_dictionary = {}
    skip_dates = []

    for idx, row in bulk_df.iterrows():
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025
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
        if (Check_We_Have_Data_For_Trade(market_df, entry_time) == False):
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
                if roi <= -0.4:
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


def Take_Samples(bulk_df, market_data_dict_by_ticker, roi_dictionary):
    skip_dates = []
    all_data_samples_x = []   # 2d list [[time since market open, volatility %], ...]
    all_roi_samples_y = []    # [roi, roi, roi, ...]

    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025
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
        entry_price = row['Entry Price']
        direction = row['Trade Type']
        trade_id = row['Id']
        roi_list = roi_dictionary[trade_id]
        market_df = market_data_dict_by_ticker[date][ticker].copy()  # market data df for this ticker and date
        start_index = trade_start_indexes[trade_id]
        
        # Skip trade if entry time is after final market data time
        if (Check_We_Have_Data_For_Trade(market_df, entry_time) == False):
            print(f"Skipping trade {trade_id}: entry time {entry_time} is after final market data time")
            continue
        
        # Track stop loss state
        stop_loss_triggered = False
        stop_loss_updated = False  # Track if stop loss changed from -0.4% to 0%
        data_sample_x = []
        roi_sample_y = []
        counter = -1                # this makes it easier to look up roi from roi_list
        next_sample_time = None     # Track when to take next sample
        
        # Iterate through market data starting from entry point
        for i in range(start_index, len(market_df)):
            counter += 1
            
            '''
            -for loop starts at the stock trade entry time, each row of market_df is market data for that this ticker. roi_list
            is a list of the roi for each line (like another column of market_df), so we can skip finding each lines roi. the counter
            variable is useful because it starts at 0 and iterates up by 1, so it's easier for a human to read since it's used 
            to get the currect index of roi_list for each row of market_df.
            -including the start of the trade (so take this data right away) we want to take a data sample every 10 minutes (track time
            by comparing start_time to market_df's 'Time' column, both are in hour:minute:second format). the data is the time since the
            market opened in minutes (the market opens here at 6:30:00 am, so compare the timestamp to that to find out how long its been.
            note that this data is isolated to the same day so no day turnover logic is needed), the volatility percent (market_df's 
            'Volatility Percent' column), and the best future roi from this point until the end of the trade.
            -do data_sample_x.append([time since market open, Volatility Percent])
            -for best future roi, we'll need to use roi_list. roi_list is the current roi for each second of the trade. so the roi at any
            given time is roi_list[counter]. we need to find the highest roi value in roi_list[counter:] (inclusive of counter's index and
            including the final roi value). save that highest value to roi_sample_y
            -we know when a trade is done by seeing if the current row of market_df matches trade_end_timestamps[trade_id]. both are
            timestamps in hour:minute:second format. once a trade is done append data_sample_x to all_data_samples_x and roi_sample_y 
            to all_roi_samples_y
            '''
            
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
                data_sample_x.append([time_since_market_open, current_volatility])
                roi_sample_y.append(best_future_roi)
                
            elif time_since_market_open >= next_sample_time:  # Time for next 10-minute sample
                next_sample_time += 10  # Set next sample time
                
                # Calculate best future ROI from this point
                if counter < len(roi_list):
                    best_future_roi = max(roi_list[counter:])
                else:
                    best_future_roi = roi_list[-1] if roi_list else 0
                
                # Add sample
                data_sample_x.append([time_since_market_open, current_volatility])
                roi_sample_y.append(best_future_roi)
            
            # Check if trade is done
            if current_time == trade_end_timestamps[trade_id]:
                # Trade is complete, add samples to master lists
                all_data_samples_x.extend(data_sample_x)
                all_roi_samples_y.extend(roi_sample_y)
                break
            
    return all_data_samples_x, all_roi_samples_y


# scaling here forces both features to have mean 0 and std 1, making the smoothing penalties comparable and preventing the “minutes” spline from dominating
def Regression_System(all_data_samples_x, all_roi_samples_y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(all_data_samples_x)

    # joint is clearly much better than additive
    gam = LinearGAM(te(0,1)).gridsearch(X_scaled, all_roi_samples_y)

    # test - do I need scaling?
    # test resulits
    # 1) show 400x scale difference
    # 2) Typical values are far apart in numeric space.
    # 3) Same story: 100× difference in spread.
    # 4) After scaling, both features are now roughly on comparable ranges. Perfect.
    # check raw ranges
    print("1) raw X min, max:", np.min(all_data_samples_x, axis=0), np.max(all_data_samples_x, axis=0))

    # check scaler that should be used
    print("2) scaler mean:", scaler.mean_)   # show the centers for minutes and volatility
    print("3) scaler scale (std dev):", scaler.scale_)    # should show a big difference (minutes >> volatility). After scaling, both will be O(1) and comparable
    # If X_scaled.min(axis=0) == X_scaled.max(axis=0) then you accidentally passed only one row to .min()/.max() — run it on the entire training array
    print("4) scaled X min, max:", X_scaled.min(axis=0), X_scaled.max(axis=0))
    print("\n")

    return gam, X_scaled, scaler


def Save_Regression_Model(gam, scaler):
    model_data = {
        'gam_model': gam,
        'scaler': scaler
    }

    with open('Holder_Strat/trained_roi_predictor_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Saved GAM model and scaler to trained_roi_predictor_model.pkl")


def Load_Regression_Model():
    try:
        with open('Holder_Strat/trained_roi_predictor_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        gam_model = model_data['gam_model']
        scaler = model_data['scaler']
        
        print("Successfully loaded GAM model and scaler")
        return gam_model, scaler
        
    except FileNotFoundError:
        print("trained_roi_predictor_model.pkl not found. Please train and save a model first.")
        return None, None

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def Save_Test_Values(all_data_samples_x, all_roi_samples_y):
    """
    Save all_data_samples_x and all_roi_samples_y to a local txt file.
    """
    import json
    
    # Convert pandas/numpy types to Python native types for JSON serialization
    # Convert all_data_samples_x (list of lists with 2 float values each)
    converted_x = []
    for sample in all_data_samples_x:
        converted_x.append([float(val) for val in sample])
    
    # Convert all_roi_samples_y (list of floats)
    converted_y = [float(val) for val in all_roi_samples_y]
    
    # Create the data structure to save
    data_to_save = {
        'all_data_samples_x': converted_x,
        'all_roi_samples_y': converted_y
    }
    
    # Save to txt file
    with open('Holder_Strat/target_runing_regression_numbers.txt', 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Saved {len(all_data_samples_x)} data samples to target_runing_regression_numbers.txt")


def Load_Test_Values():
    """
    Load all_data_samples_x and all_roi_samples_y from the local txt file.
    """
    import json
    
    try:
        with open('Holder_Strat/target_runing_regression_numbers.txt', 'r') as f:
            data = json.load(f)
        
        all_data_samples_x = data['all_data_samples_x']
        all_roi_samples_y = data['all_roi_samples_y']
        
        print(f"Loaded {len(all_data_samples_x)} data samples from target_runing_regression_numbers.txt")
        return all_data_samples_x, all_roi_samples_y
        
    except FileNotFoundError:
        print("target_runing_regression_numbers.txt not found. Please run Save_Test_Values() first.")
        return None, None
    except Exception as e:
        print(f"Error loading target_runing_regression_numbers.txt: {e}")
        return None, None


def Joint_Vs_Additive_Model_Testing(gam_additive, gam_joint, all_data_samples_x, all_data_samples_y, X_scaled):
    print("=== MODEL COMPARISON ANALYSIS ===")
    
    # Model statistics
    print(f"Additive Model - Samples: {gam_additive.statistics_['n_samples']}")
    print(f"Joint Model - Samples: {gam_joint.statistics_['n_samples']}")
    # Note: Pseudo R² statistics are complex objects, will calculate manually below
    
    # Data range check
    X = np.array(X_scaled)
    print(f"Scaled data range: {X.min(axis=0)} to {X.max(axis=0)}")
    
    # Test on actual data points
    print("\n=== PREDICTIONS ON ACTUAL DATA POINTS ===")
    print(f"test 1 additive) {gam_additive.predict(X_scaled[0].reshape(1, -1))[0]:.3f}")
    print(f"test 1 joint) {gam_joint.predict(X_scaled[0].reshape(1, -1))[0]:.3f}")
    print(f"test 2 additive) {gam_additive.predict(X_scaled[1].reshape(1, -1))[0]:.3f}")
    print(f"test 2 joint) {gam_joint.predict(X_scaled[1].reshape(1, -1))[0]:.3f}")
    print(f"test 3 additive) {gam_additive.predict(X_scaled[2].reshape(1, -1))[0]:.3f}")
    print(f"test 3 joint) {gam_joint.predict(X_scaled[2].reshape(1, -1))[0]:.3f}")
    
    # Test on edge cases and typical values
    print("\n=== PREDICTIONS ON EDGE CASES ===")
    # Create test points in scaled space
    test_points_scaled = np.array([
        [-1.18, -1.56],  # Min values (early market, low volatility)
        [2.68, 3.88],    # Max values (late market, high volatility)
        [0, 0],          # Mean values (mid market, average volatility)
        [-1, 0],         # Early market, average volatility
        [1, 1],          # Late market, high volatility
    ])
    
    for i, point in enumerate(test_points_scaled):
        add_pred = gam_additive.predict(point.reshape(1, -1))[0]
        joint_pred = gam_joint.predict(point.reshape(1, -1))[0]
        print(f"Test point {i+1}: Additive={add_pred:.3f}, Joint={joint_pred:.3f}, Diff={abs(add_pred-joint_pred):.3f}")
    
    # Calculate prediction errors on training data
    print("\n=== TRAINING DATA PERFORMANCE ===")
    y_true = np.array(all_data_samples_y)
    add_preds = gam_additive.predict(X_scaled)
    joint_preds = gam_joint.predict(X_scaled)
    
    add_mse = np.mean((y_true - add_preds)**2)
    joint_mse = np.mean((y_true - joint_preds)**2)
    add_mae = np.mean(np.abs(y_true - add_preds))
    joint_mae = np.mean(np.abs(y_true - joint_preds))
    
    print(f"Additive Model - MSE: {add_mse:.4f}, MAE: {add_mae:.4f}")
    print(f"Joint Model - MSE: {joint_mse:.4f}, MAE: {joint_mae:.4f}")
    
    # Check for unrealistic predictions
    print(f"\nAdditive predictions range: {add_preds.min():.3f} to {add_preds.max():.3f}")
    print(f"Joint predictions range: {joint_preds.min():.3f} to {joint_preds.max():.3f}")
    print(f"Actual ROI range: {y_true.min():.3f} to {y_true.max():.3f}")
    
    # Count extreme predictions
    extreme_add = np.sum((add_preds < -2) | (add_preds > 15))
    extreme_joint = np.sum((joint_preds < -2) | (joint_preds > 15))
    print(f"Extreme predictions (< -2 or > 15): Additive={extreme_add}, Joint={extreme_joint}")

    # Visualization
    plt.ion()  # turn on interactive mode
    
    # Plot 1: Model predictions across time dimension
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    XX = np.linspace(-1.2, 2.7, 200)  # Scaled time range
    volatility_fixed = 0.0  # Average volatility (scaled)
    add_line = gam_additive.predict(np.column_stack([XX, np.repeat(volatility_fixed, 200)]))
    joint_line = gam_joint.predict(np.column_stack([XX, np.repeat(volatility_fixed, 200)]))
    plt.plot(XX, add_line, label='Additive', color='red')
    plt.plot(XX, joint_line, label='Joint', color='blue')
    plt.xlabel('Time Since Market Open (scaled)')
    plt.ylabel('Predicted ROI')
    plt.title('Predictions vs Time (Fixed Volatility)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Model predictions across volatility dimension
    plt.subplot(2, 2, 2)
    VV = np.linspace(-1.6, 3.9, 200)  # Scaled volatility range
    time_fixed = 0.0  # Average time (scaled)
    add_line_vol = gam_additive.predict(np.column_stack([np.repeat(time_fixed, 200), VV]))
    joint_line_vol = gam_joint.predict(np.column_stack([np.repeat(time_fixed, 200), VV]))
    plt.plot(VV, add_line_vol, label='Additive', color='red')
    plt.plot(VV, joint_line_vol, label='Joint', color='blue')
    plt.xlabel('Volatility (scaled)')
    plt.ylabel('Predicted ROI')
    plt.title('Predictions vs Volatility (Fixed Time)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Residuals comparison
    plt.subplot(2, 2, 3)
    add_residuals = y_true - add_preds
    joint_residuals = y_true - joint_preds
    plt.scatter(add_preds, add_residuals, alpha=0.5, label='Additive', color='red', s=10)
    plt.scatter(joint_preds, joint_residuals, alpha=0.5, label='Joint', color='blue', s=10)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted ROI')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predictions')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Prediction comparison
    plt.subplot(2, 2, 4)
    plt.scatter(add_preds, joint_preds, alpha=0.5, s=10)
    min_pred = min(add_preds.min(), joint_preds.min())
    max_pred = max(add_preds.max(), joint_preds.max())
    plt.plot([min_pred, max_pred], [min_pred, max_pred], 'r--', label='Perfect Agreement')
    plt.xlabel('Additive Predictions')
    plt.ylabel('Joint Predictions')
    plt.title('Additive vs Joint Predictions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return add_mse, joint_mse, add_mae, joint_mae


def Test_Model_On_Trade_Data(gam, X_scaled, all_data_samples_x, all_roi_samples_y):
    print("TESTING model on various data points")

    indexes = [1,15,16,18,21,53,35,94,105,87,119,205,206,207,208,209,340,350,400,18,500]
    for i in range(0, len(indexes)):
        print(f"test {i}) prediction: {gam.predict(X_scaled[indexes[i]].reshape(1, -1))[0]:.3f}, training: {all_roi_samples_y[indexes[i]]}")
    pass


# roi dictionary: {trade_id: [roi values], ...}
def Full_Test_Model_Over_Trade_Data(gam, scaler, bulk_df, market_data_dict_by_ticker, roi_dictionary):
    '''
    basically do the same as when I found teh roi predictions, but don't look ahead, instead just spit out the prediction
    I need it to be grouped by ticke: ticker, timestamp, prediction
    '''
    skip_dates = []
    results = [] # [[ticker, timestamp, prediction], ...] this is so I can check any value on what on the actual charts
    sums = {}

    for _, row in bulk_df.iterrows():
        date = bulk_csv_date_converter(row['Date'])  # 08-09-2025
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
        trade_id = row['Id']
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
                
                data = scaler.transform([[time_since_market_open, current_volatility]])
                roi_prediction = gam.predict(data)
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
    with open('Holder_Strat/roi_prediction_model_trade_results.txt', 'w') as f:
        f.write('sums by ticker\n')
        for ticker in sums:
            f.write(f'{ticker} = {round(sums[ticker], 2)}\n')
        
        overall = sum(sums.values())
        f.write(f'overall = {round(overall, 2)}\n\n')
        
        f.write('Specific Results\n')
        for result in results:
            f.write(f'{result}\n')

    return results

    
def Main():
    # this can fun the sample collection if we get new data, otherwise just load the samples from the txt file
    # bulk_df and roi_dictionary are needed for actual trade estimates
    bulk_df = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[["Date", "Ticker", "Entry Time", "Time in Trade", "Entry Price", "Exit Price", "Trade Type", "Exit Price", "Entry Volatility Percent", "Original Holding Reached", "Original Best Exit Percent", "Original Percent Change"]]
    bulk_df = Add_Trade_Id(bulk_df)
    market_data_dict_by_ticker = Load_Market_Data_Dictionary(bulk_df) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    print("\nmaking roi dictionary...")
    roi_dictionary = Create_Roi_Dictionary_For_Trades(bulk_df, market_data_dict_by_ticker)
    print(f"roi dictionary done. Total trades processed: {len(roi_dictionary)}")
    
    '''
    all_data_samples_x, all_roi_samples_y = Take_Samples(bulk_df, market_data_dict_by_ticker, roi_dictionary)
    Save_Test_Values(all_data_samples_x, all_roi_samples_y)
    
    '''
    all_data_samples_x, all_roi_samples_y = Load_Test_Values()

    gam, X_scaled, scaler = Regression_System(all_data_samples_x, all_roi_samples_y)

    #Joint_Vs_Additive_Model_Testing(gam_additive, gam_joint, all_data_samples_x, all_roi_samples_y, X_scaled)
    #Test_Model_On_Trade_Data(gam, X_scaled, all_data_samples_x, all_roi_samples_y)

    Save_Regression_Model(gam, scaler)

    Full_Test_Model_Over_Trade_Data(gam, scaler, bulk_df, market_data_dict_by_ticker, roi_dictionary)
    


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