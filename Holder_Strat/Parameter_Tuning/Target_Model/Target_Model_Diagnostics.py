import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

version = None
target_dir = "Holder_Strat/Parameter_Tuning/Target_Model"
def Set_Version(passed_version):
    global version
    version = passed_version


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



# purpose: give the model a bunch of test inputs and write out the y values. compare visually to see if it sucks
def Get_Model_Test_Values(model, scaler, smearing_factor):
    test_vol_percents = [0.19,0.31,0.3,0.36,0.63,0.52]
    test_minutes_since_entry = [1,8,8,9,51,61]

    if (len(test_vol_percents) != len(test_minutes_since_entry)):
        raise ValueError("lengths of lists aren't the same")

    print("\n")
    for i in range(0, len(test_vol_percents)):
        roi_target = Predict_Max_ROI(model, scaler, test_minutes_since_entry[i], test_vol_percents[i], smearing_factor)
        print(f"vol%: {test_vol_percents[i]}, minutes: {test_minutes_since_entry[i]}, y = {round(roi_target, 2)}")
    print("\n")


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
    plot_path = f'{target_dir}/Diagnostics/Residual_Plot_All_Samples_{version}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
    
    print(f"\nPlot saved to: {plot_path}")
    
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
    plot_path = f'{target_dir}/Diagnostics/Residual_Plot_Mean_Sample_{version}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
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
    
    print(f"\nPlot saved to: {plot_path}")


'''
- basically do the same as when I found the roi predictions, but don't look ahead, instead just spit out the prediction
- I need it to be grouped by ticker: ticker, timestamp, prediction
- roi dictionary: {trade_id: [roi values], ...}
- mode: string for what the roi target is. either the model prediction or set 'max' values. used for comparisons
'''
def Run_Model_Performance_Over_Trade_History(model, scaler, smearing_factor, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes, mode):
    max_values = {'HOOD': 3.0, 'IONQ': 3.5, 'MARA': 3.0, 'RDDT': 2.7, 'SMCI': 2.5, 'SOXL': 3.0, 'TSLA': 1.8} # NOTE: only used if mode = 'max values
    skip_dates = []
    results = [] # [[ticker, timestamp, prediction], ...] this is so I can check any value on what on the actual charts
    trades = 0   # needed because some trades are skipped for various reasons

    # these are restictions on trades. They say "only test trades meeting these restrictions"
    test_restrictions = False # enables testing things like vol%, ratio, and time
    vol_percent_restriction = 0.5 # greater than or equal to this
    include_all_trades_before_this_time = '06:40:00'
    
    for idx, row in bulk_df.iterrows():
        if (test_restrictions == True):
            #if (row['Entry Time'] < include_all_trades_before_this_time):
            #    pass
            if row['Entry Volatility Percent'] < vol_percent_restriction:
                continue

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

        trade_id = row['Trade Id']
        market_df = market_data_dict_by_ticker[date][ticker].copy()  # market data df for this ticker and date
        start_index = trade_start_indexes[trade_id]
        
        trade_start_minutes_since_open = market_df.iloc[start_index]['Time Since Market Open']
        next_sample_time = 0
        counter = -1
        trades += 1
        
        # Iterate through market data starting from entry point
        for i in range(start_index, len(market_df)):
            counter += 1
            curr_time = market_df.iloc[i]['Time']
            curr_volatility = market_df.iloc[i]['Volatility Percent']
            time_since_market_open = market_df.iloc[i]['Time Since Market Open']
            curr_roi = roi_dictionary[trade_id][counter]
            trade_duration = time_since_market_open - trade_start_minutes_since_open # starts at 0 and iterates each minute

            # Take sample every x minutes
            if (mode == 'model values'):
                if trade_duration % 5 == 0 and next_sample_time == trade_duration:
                    next_sample_time += 5
                    # Unbiased back-transform with Duan smearing
                    roi_target = Predict_Max_ROI(model, scaler, time_since_market_open, curr_volatility, smearing_factor)
            elif (mode == 'max values'):
                roi_target = max_values[ticker]
            else:
                raise ValueError(f"bad mode value: {mode}")
            
            # check if we're at our roi prediction target
            if (curr_roi >= roi_target):
                results.append([ticker, date, curr_time, roi_target, 'roi hit'])
                break

            # Check if trade is done
            if curr_time == trade_end_timestamps[trade_id]:
                results.append([ticker, date, curr_time, curr_roi, 'roi not hit'])
                break

    # find sums
    ticker_date_sums = {}
    overall_date_sums = {}
    for result in results:
        ticker, date, time, roi, status = result
        if (date not in ticker_date_sums):
            ticker_date_sums[date] = {}
            overall_date_sums[date] = 0

        if (ticker not in ticker_date_sums[date]):
            ticker_date_sums[date][ticker] = 0

        ticker_date_sums[date][ticker] += roi
        overall_date_sums[date] += roi

    # Write results to text file
    file_path = f'{target_dir}/Diagnostics/roi_prediction_model_trade_results.txt'
    with open(file_path, 'w') as f:
        f.write(f"MODE = {mode}\n")
        for date in ticker_date_sums.keys():
            f.write(f"Date = {date}\n")
            
            for ticker, value in ticker_date_sums[date].items():
                f.write(f'{ticker} = {round(value, 2)}\n')
        
            f.write(f"overall = {round(overall_date_sums[date], 2)}\n\n")
        
        # now write the total sums across all days for each ticker and the overall sum for all days
        f.write("Totals Across All Dates:\n")
        ticker_totals = {}
        for date_totals in ticker_date_sums.values():
            for ticker, value in date_totals.items():
                ticker_totals[ticker] = ticker_totals.get(ticker, 0) + value
        for ticker, total in ticker_totals.items():
            f.write(f"{ticker} = {round(total, 2)}\n")

        days_count = len(overall_date_sums.keys())
        overall_sum = sum(overall_date_sums.values())
        overall_avg_per_day = round(overall_sum / days_count, 2)
        overall_avg_per_trade = round(overall_sum / len(bulk_df), 4)
        red_days_avg = 0
        green_days_avg = 0

        f.write(f"\nTrades: {trades}\n")
        f.write(f"Overall Total = {round(overall_sum, 2)}\n")
        f.write(f"days = {days_count}\n")
        f.write(f"Overall avg / trade = {overall_avg_per_trade}\n")
        f.write(f"**divided by 6 = {round(overall_avg_per_trade / 6, 4)}\n")
        f.write(f"Overall avg / day (BAD METRIC) = {overall_avg_per_day}\n")
        f.write(f"divided by 6 (BAD METRIC) = {round(overall_avg_per_day / 6, 2)}\n")

        # get red/green data data
        red_data = []
        green_data = []
        for date, value in overall_date_sums.items():
            if (value < 0):
                red_data.append(value)
            else: # including 0
                green_data.append(value)
        
        if (len(red_data) + len(green_data) != days_count):
            raise ValueError(f"red data and green data are both empty. this means there's a bug. length of df (for vol%): {len(bulk_df)}")

        if (len(red_data) != 0):
            red_days_avg = round(np.average(red_data) / 6, 2)
        if (len(green_data) != 0):
            green_days_avg = round(np.average(green_data) / 6, 2)

        f.write(f"red days: {len(red_data)}/{days_count}\n")
        f.write(f"avg red day: {red_days_avg}\n")
        f.write(f"avg green day: {green_days_avg}\n")


# keep: this is useful for assessing what optimizations we can do next. it's the distribution of y values
def Summarize_Response_Distribution(all_roi_samples_y, save_hist=True, show_plot=False):
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
    file_path = f'{target_dir}/Diagnostics/Response_Distribution_Summary_{version}.txt'
    with open(file_path, 'w') as f:
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

    print(f"\nResponse distribution summary written to: {file_path}")

    # Save histogram plot
    histogram_file_path = f'{target_dir}/Diagnostics/Response_Distribution_Hist_{version}.png'
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
        plt.savefig(histogram_file_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        else:
            plt.close()
        print(f'Histogram saved to: {histogram_file_path}')