import pandas as pd
from datetime import date, timedelta, datetime
import os
import shutil
import Main_Globals
import inspect
import sys
import numpy as np
from scipy import stats

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

'''
Compares time in trade to the outcome in order to show correlations. uses bulk_combined.csv
'''

bulk_combind_csv_path = "Csv_Files/3_Final_Trade_Csvs/Bulk_Combined.csv"
time_interval = 10 # minutes
times = ['6:30:00', '6:40:00', '6:50:00', '7:00:00', '7:10:00', '7:20:00', '7:30:00', '7:40:00', '7:50:00', '8:00:00', '8:30:00', '8:40:00','8:50:00','9:00:00']


# convert the csv info a dictionary of dataframes
# key = starting timestamp, it for time interval minutes
# value = dataframe of trades during that time interval
# time interval = how long each time period goes from its key
def Create_Dict_Of_DFs():
    try:
        # Load the bulk combined CSV file containing all trade data
        # This CSV should have an 'Entry Time' column for filtering by time
        df = pd.read_csv(bulk_combind_csv_path)
        
        # Initialize dictionary with time strings as keys and None as initial values
        # Each key represents the start of a time interval window
        time_dict = {}
        
        # Iterate through each time period and filter the main DataFrame
        # For each time key, create a subset DataFrame containing only trades where:
        # - 'Entry Time' >= current time key
        # - 'Entry Time' < (current time key + time_interval minutes)
        # This creates non-overlapping time buckets for analysis
        columns_to_keep = ['Date', 'Ticker', 'Entry Time', 'Best Exit Percent', 'Worst Exit Percent', 'Target 0.3,0.9,-0.5,-0.3', 
                           'Target 0.2,0.9,-0.5,-0.3', 'Target 0.5,0.9,-0.4,-0.3', 'Target 0.2,0.9,-0.5,-0.1', 'Time in Trade']
        for time_str in times:
            # Convert time strings to datetime objects for proper time comparison
            # Handle time parsing and interval calculation using timedelta
            start_time = datetime.strptime(time_str, '%H:%M:%S').time()
            end_time = (datetime.combine(date.today(), start_time) + timedelta(minutes=time_interval)).time()
            
            # Filter main DataFrame for each time window and store result
            # Each resulting DataFrame becomes the value for its corresponding time key
            
            # Convert Entry Time column to time objects for easier comparison
            df['Entry Time Parsed'] = pd.to_datetime(df['Entry Time'], format='%H:%M:%S').dt.time
            
            # Filter trades within the time window
            mask = (df['Entry Time Parsed'] >= start_time) & (df['Entry Time Parsed'] < end_time)

            # Store filtered DataFrame for this time window
            filtered_df = df[mask].copy()
            
            # Keep only the specified columns
            filtered_df = filtered_df[columns_to_keep]
            
            time_dict[time_str] = filtered_df
        
        # Clean up temporary column
        df.drop('Entry Time Parsed', axis=1, inplace=True)

        # Write results to a text file for inspection
        with open('time_dict_debug_output.txt', 'w') as f:
            for key, value in time_dict.items():
                f.write(f"key: {key}\n")
                f.write(f"{value.to_string()}\n")
                f.write("-" * 50 + "\n")
        
        # Return the populated dictionary for time-based trade analysis
        return time_dict
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Analyze_Results_Impact(dict_dfs):
    try:
        print("=== ANALYZING TIME IN TRADE IMPACT ON RESULTS ===\n")
        
        # Define the target columns to analyze
        target_columns = [
            'Target 0.3,0.9,-0.5,-0.3',
            'Target 0.2,0.9,-0.5,-0.3', 
            'Target 0.5,0.9,-0.4,-0.3',
            'Target 0.2,0.9,-0.5,-0.1'
        ]
        
        # Convert time in trade to seconds for numerical analysis
        def time_to_seconds(time_str):
            """Convert HH:MM:SS format to total seconds"""
            try:
                time_obj = datetime.strptime(time_str, '%H:%M:%S')
                return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
            except:
                return np.nan
        
        # Aggregate all data across time windows for overall analysis
        all_data = []
        time_window_results = {}
        
        for time_window, df in dict_dfs.items():
            if df.empty:
                continue
                
            # Convert time in trade to seconds
            df_copy = df.copy()
            df_copy['Time_in_Trade_Seconds'] = df_copy['Time in Trade'].apply(time_to_seconds)
            
            # Remove rows with invalid time data
            df_copy = df_copy.dropna(subset=['Time_in_Trade_Seconds'])
            
            if df_copy.empty:
                continue
            
            # Store for overall analysis
            all_data.append(df_copy)
            
            # Analyze each time window separately
            print(f"\n--- TIME WINDOW: {time_window} ---")
            print(f"Number of trades: {len(df_copy)}")
            
            if len(df_copy) > 1:  # Need at least 2 trades for correlation
                window_analysis = analyze_time_window(df_copy, target_columns, time_window)
                time_window_results[time_window] = window_analysis
        
        # Combine all data for overall analysis
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n=== OVERALL ANALYSIS (All Time Windows Combined) ===")
            print(f"Total trades analyzed: {len(combined_df)}")
            
            overall_analysis = analyze_overall_correlation(combined_df, target_columns)
            
            # Time bucket analysis
            time_bucket_analysis(combined_df, target_columns)
            
            # Generate summary report
            generate_summary_report(time_window_results, overall_analysis)
        
        return time_window_results
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def analyze_time_window(df, target_columns, time_window):
    """Analyze correlation for a specific time window"""
    try:
        results = {}
        
        # Basic statistics
        avg_time = df['Time_in_Trade_Seconds'].mean()
        median_time = df['Time_in_Trade_Seconds'].median()
        
        print(f"Average time in trade: {avg_time/60:.1f} minutes")
        print(f"Median time in trade: {median_time/60:.1f} minutes")
        
        # Analyze each target column
        for target_col in target_columns:
            if target_col in df.columns:
                # Convert results to numerical categories for analysis
                result_mapping = analyze_result_patterns(df, target_col, time_window)
                results[target_col] = result_mapping
        
        return results
        
    except Exception as e:
        print(f"Error analyzing time window {time_window}: {str(e)}")
        return {}


def analyze_result_patterns(df, target_col, time_window):
    """Analyze patterns between time in trade and results for a specific target"""
    try:
        # Get unique result values and their frequencies
        result_counts = df[target_col].value_counts()
        print(f"\n  {target_col}:")
        print(f"    Result distribution: {dict(result_counts)}")
        
        # Analyze time patterns for each result type
        result_time_stats = {}
        
        for result_value in result_counts.index:
            trades_with_result = df[df[target_col] == result_value]
            if len(trades_with_result) > 0:
                avg_time = trades_with_result['Time_in_Trade_Seconds'].mean()
                median_time = trades_with_result['Time_in_Trade_Seconds'].median()
                
                result_time_stats[result_value] = {
                    'count': len(trades_with_result),
                    'avg_time_minutes': avg_time / 60,
                    'median_time_minutes': median_time / 60
                }
                
                print(f"    Result {result_value}: {len(trades_with_result)} trades, "
                      f"avg time: {avg_time/60:.1f}min, median: {median_time/60:.1f}min")
        
        # Calculate correlation if we have numerical results
        numerical_results = convert_results_to_numerical(df[target_col])
        if len(numerical_results.dropna()) > 1:
            correlation = df['Time_in_Trade_Seconds'].corr(numerical_results)
            if not np.isnan(correlation):
                print(f"    Correlation with time: {correlation:.3f}")
                result_time_stats['correlation'] = correlation
        
        return result_time_stats
        
    except Exception as e:
        print(f"    Error analyzing {target_col}: {str(e)}")
        return {}


def convert_results_to_numerical(result_series):
    """Convert result values to numerical scores for correlation analysis"""
    # Create a mapping: higher positive values = better results
    # 0.9 (target2) = best, negative values = worse
    def map_result(value):
        try:
            if value == 0.9:
                return 3  # Hit target2 - best outcome
            elif value > 0:
                return 2  # Positive result but not target2
            elif value == 0.0:
                return 1  # Break-even
            elif value >= -0.3:
                return -1  # Small loss
            elif value >= -0.5:
                return -2  # Medium loss
            else:
                return -3  # Large loss
        except:
            return np.nan
    
    return result_series.apply(map_result)


def analyze_overall_correlation(combined_df, target_columns):
    """Analyze correlations across all time windows"""
    try:
        print(f"\nTime in trade statistics:")
        print(f"  Min: {combined_df['Time_in_Trade_Seconds'].min()/60:.1f} minutes")
        print(f"  Max: {combined_df['Time_in_Trade_Seconds'].max()/60:.1f} minutes")
        print(f"  Average: {combined_df['Time_in_Trade_Seconds'].mean()/60:.1f} minutes")
        print(f"  Median: {combined_df['Time_in_Trade_Seconds'].median()/60:.1f} minutes")
        
        overall_results = {}
        
        print(f"\nOverall correlations:")
        for target_col in target_columns:
            if target_col in combined_df.columns:
                numerical_results = convert_results_to_numerical(combined_df[target_col])
                correlation = combined_df['Time_in_Trade_Seconds'].corr(numerical_results)
                
                if not np.isnan(correlation):
                    print(f"  {target_col}: {correlation:.3f}")
                    overall_results[target_col] = correlation
                    
                    # Statistical significance test
                    if len(numerical_results.dropna()) > 10:
                        corr_coef, p_value = stats.pearsonr(
                            combined_df['Time_in_Trade_Seconds'][numerical_results.notna()], 
                            numerical_results.dropna()
                        )
                        significance = "significant" if p_value < 0.05 else "not significant"
                        print(f"    (p-value: {p_value:.3f} - {significance})")
        
        return overall_results
        
    except Exception as e:
        print(f"Error in overall correlation analysis: {str(e)}")
        return {}


def time_bucket_analysis(combined_df, target_columns):
    """Analyze results by time duration buckets"""
    try:
        print(f"\n=== TIME BUCKET ANALYSIS ===")
        
        # Create time buckets (in minutes)
        time_minutes = combined_df['Time_in_Trade_Seconds'] / 60
        
        # Define buckets
        bins = [0, 2, 5, 10, 15, 30, float('inf')]
        labels = ['0-2min', '2-5min', '5-10min', '10-15min', '15-30min', '30+min']
        
        combined_df['Time_Bucket'] = pd.cut(time_minutes, bins=bins, labels=labels, right=False)
        
        bucket_counts = combined_df['Time_Bucket'].value_counts().sort_index()
        print(f"Trade distribution by time buckets:")
        for bucket, count in bucket_counts.items():
            print(f"  {bucket}: {count} trades")
        
        # Analyze success rates by bucket for each target
        for target_col in target_columns:
            if target_col in combined_df.columns:
                print(f"\n{target_col} success rates by time bucket:")
                
                for bucket in labels:
                    bucket_data = combined_df[combined_df['Time_Bucket'] == bucket]
                    if len(bucket_data) > 0:
                        success_rate = (bucket_data[target_col] == 0.9).mean() * 100
                        loss_rate = (bucket_data[target_col] < 0).mean() * 100
                        
                        print(f"  {bucket}: {len(bucket_data)} trades, "
                              f"{success_rate:.1f}% hit target2, {loss_rate:.1f}% losses")
        
    except Exception as e:
        print(f"Error in time bucket analysis: {str(e)}")


def generate_summary_report(time_window_results, overall_analysis):
    """Generate a summary report of findings"""
    try:
        print(f"\n" + "="*60)
        print(f"SUMMARY REPORT: TIME IN TRADE IMPACT ANALYSIS")
        print(f"="*60)
        
        if overall_analysis:
            print(f"\nKEY FINDINGS:")
            
            # Identify strongest correlations
            strong_correlations = {k: v for k, v in overall_analysis.items() if abs(v) > 0.1}
            if strong_correlations:
                print(f"\nStrongest correlations (|r| > 0.1):")
                for target, corr in sorted(strong_correlations.items(), key=lambda x: abs(x[1]), reverse=True):
                    direction = "longer trades perform better" if corr > 0 else "shorter trades perform better"
                    print(f"  {target}: {corr:.3f} ({direction})")
            else:
                print(f"\nNo strong correlations found (all |r| < 0.1)")
                print(f"This suggests time in trade has minimal impact on results")
            
            # Interpretation guide
            print(f"\nINTERPRETATION GUIDE:")
            print(f"  Correlation > 0.3: Strong positive (longer trades = better results)")
            print(f"  Correlation 0.1-0.3: Moderate positive")
            print(f"  Correlation -0.1-0.1: Weak/no relationship")
            print(f"  Correlation -0.3--0.1: Moderate negative")
            print(f"  Correlation < -0.3: Strong negative (shorter trades = better results)")
        
        print(f"\nRECOMMENDations:")
        print(f"1. Review the time bucket analysis for actionable insights")
        print(f"2. Consider that market conditions (time windows) may matter more than trade duration")
        print(f"3. Look for patterns in specific time windows that show stronger correlations")
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")


# Run the analysis
dict_dfs = Create_Dict_Of_DFs()
if dict_dfs:
    results = Analyze_Results_Impact(dict_dfs)
    print("\nAnalysis complete! Check the output above for insights.")