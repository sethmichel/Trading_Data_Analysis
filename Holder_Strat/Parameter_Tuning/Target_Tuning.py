import pandas as pd
from pygam import LinearGAM, te
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import Helper_Functions
import json


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
        trade_id = row['Id']
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


# scaling prevents the variable with larger numbers from dominating
# we must save the scaler along with the model and use both for predicting
# x and y are lists of dicts, trade id is the key
def Train_Model(all_data_samples_x, all_roi_samples_y):
    raw_x = []
    raw_y = []
    for data_dict in all_data_samples_x:
        trade_id, features = next(iter(data_dict.items()))
        raw_x.append(features)
    
    for roi_dict in all_roi_samples_y:
        trade_id, roi_value = next(iter(roi_dict.items()))
        raw_y.append(roi_value)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_x)

    model = LinearGAM(te(0,1)).gridsearch(X_scaled, raw_y)

    return model, X_scaled, scaler


def Save_Model(model, scaler):
    model_data = {
        'model': model,
        'scaler': scaler
    }

    with open('Holder_Strat/Parameter_Tuning/model_files_and_data/trained_roi_predictor_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Saved model and scaler to trained_roi_predictor_model.pkl")


def Load_Model():
    try:
        with open('Holder_Strat/Parameter_Tuning/model_files_and_data/trained_roi_predictor_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        
        print("Successfully loaded model and scaler")
        return model, scaler
        
    except FileNotFoundError:
        print("trained_roi_predictor_model.pkl not found. Please train and save a model first.")
        return None, None

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


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
    with open('Holder_Strat/Parameter_Tuning/model_files_and_data/target_runing_regression_numbers.txt', 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"Saved {len(all_data_samples_x)} data samples to target_runing_regression_numbers.txt")


def Load_Test_Values():
    """
    Load all_data_samples_x and all_roi_samples_y from the local txt file.
    Returns data in the new format: [{trade_id: [time, volatility]}, ...] and [{trade_id: roi}, ...]
    """

    try:
        with open('Holder_Strat/Parameter_Tuning/model_files_and_data/target_runing_regression_numbers.txt', 'r') as f:
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


# roi dictionary: {trade_id: [roi values], ...}
def Full_Test_Model_Over_Trade_Data(model, scaler, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes):
    '''
    basically do the same as when I found teh roi predictions, but don't look ahead, instead just spit out the prediction
    I need it to be grouped by ticke: ticker, timestamp, prediction
    '''
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
                roi_prediction = model.predict(data)
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
    with open('Holder_Strat/Parameter_Tuning/model_files_and_data/roi_prediction_model_trade_results.txt', 'w') as f:
        f.write('sums by ticker\n')
        for ticker in sums:
            f.write(f'{ticker} = {round(sums[ticker], 2)}\n')
        
        overall = sum(sums.values())
        f.write(f'overall = {round(overall, 2)}\n\n')
        
        f.write('Specific Results\n')
        results.sort(key=lambda x: x[0]) # sort by ticker
        for result in results:
            f.write(f'{result[0]}, {result[1]}, {round(result[2][0], 2)}\n')

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
    summary_path = 'Holder_Strat/Parameter_Tuning/model_files_and_data/Response_Distribution_Summary.txt'
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
        plot_path = 'Holder_Strat/Parameter_Tuning/model_files_and_data/Response_Distribution_Hist.png'
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
def Run_Model_Diagnostics(model, scaler, all_data_samples_x, all_roi_samples_y):
    """
    Director function for running various diagnostic tests on the model.
    """
    print("\nRunning model diagnostics...")
    # 0) Response variable distribution (Y)
    Summarize_Response_Distribution(all_roi_samples_y, save_hist=True, show_plot=False)
    
    # Run residual plot analysis
    Plot_Residuals_Vs_Fitted(model, scaler, all_data_samples_x, all_roi_samples_y)
    
    print("Diagnostics complete!")


def Plot_Residuals_Vs_Fitted(model, scaler, all_data_samples_x, all_roi_samples_y):
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
    predictions = model.predict(x_scaled)
    
    # Calculate residuals (actual - predicted)
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
    plt.savefig('Holder_Strat/Parameter_Tuning/model_files_and_data/Residual_Plot_All_Samples.png', 
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
    
    print(f"\nPlot saved to: Holder_Strat/Parameter_Tuning/model_files_and_data/Residual_Plot_All_Samples.png")
    
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
    plt.savefig('Holder_Strat/Parameter_Tuning/model_files_and_data/Residual_Plot_Mean_Sample.png', 
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
    
    print(f"\nPlot saved to: Holder_Strat/Parameter_Tuning/model_files_and_data/Residual_Plot_Mean_Sample.png")


def Main():
    '''
    bulk_df = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[["Date", "Ticker", "Entry Time", "Time in Trade", "Entry Price", "Exit Price", "Trade Type", "Exit Price", "Entry Volatility Percent", "Original Holding Reached", "Original Best Exit Percent", "Original Percent Change"]]
    market_data_dict_by_ticker = Helper_Functions.Load_Market_Data_Dictionary(bulk_df) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    
    roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Create_Roi_Dictionary_For_Trades(bulk_df, market_data_dict_by_ticker, -0.4)
    
    all_data_samples_x, all_roi_samples_y = Collect_Data(bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes)
    Save_Test_Values(all_data_samples_x, all_roi_samples_y)
    
    model, x_scaled, scaler = Train_Model(all_data_samples_x, all_roi_samples_y)
    Save_Model(model, scaler)
    '''
    roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Load_Roi_Dictionary_And_Values()
    all_data_samples_x, all_roi_samples_y = Load_Test_Values()
    model, scaler = Load_Model()

    # Print the number of negative values in all_roi_samples_y
    negative_count = sum(1 for roi_dict in all_roi_samples_y for value in roi_dict.values() if value < 0)
    print(f"Number of negative values in all_roi_samples_y: {negative_count}") # 14

    Run_Model_Diagnostics(model, scaler, all_data_samples_x, all_roi_samples_y)

    #Full_Test_Model_Over_Trade_Data(model, scaler, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes)
    


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