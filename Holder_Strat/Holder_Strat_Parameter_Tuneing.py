'''
1) load each days market data into a dictionary date: df
2) load bulk csv into a df
3) make 'optimal sl result', 'optimal sl hit', 'optimal sl best exit %', 'optimal sl reached holding' as new df columns
3) get 'Entry Volatility Percent' bucket avg sl value, round down by 0.1%
4) iterate over 'Price Movement' list, if it hits the sl value, record the sl value in 'optimal sl result'
   if it reaches 0.6, change the sl value to 0, and write true for 'optimal sl reached holding'
   for the duration of the trade, record the best roi price. when the trade ends, record that price as a % roi for 'optimal sl best exit %'

next: write the max for each ticker and if best roi reaches the max, save as the max. output a text file saying the ticker results and total sum. number of sl's with -0.4 vs optimal and total results

'''

import pandas as pd
import os
import ast


# [exclusive low volatility %, inclusive high volatility %, optimal sl %]
sl_volatility_buckets = [
    [0, 0.2, -0.7],
    [0.2,0.4,-0.65],
    [0.4,0.6,-0.5],
    [0.6,0.8,-0.8],
    [0.8,1,None],
    [1,1.2,None],
    [1.2,1.4,None],
    [1.4,1.6,None],
    [1.6,1.8,None]
]

max_rois = {'HOOD': 3.0, 'IONQ': 4.0, 'MARA': 3.0, 'RDDT': 3.0, 'SMCI': 3.0, 'SOXL': 3.0, 'TSLA': 2.0}


def bulk_csv_date_converter(date):
    parts = date.split('-')

    if (len(parts[0]) == 1):
        parts[0] = f"0{parts[0]}"
    if (len(parts[1]) == 1):
        parts[1] = f"0{parts[1]}"
    if (len(parts[2]) == 2):
        parts[2] = f"20{parts[2]}"
    
    return '-'.join(parts)

def load_market_data_dictionary():
    """
    Load market data CSV files into a dictionary where key is date and value is dataframe.
    Only loads market data files that have corresponding dates in bulk_summaries.csv.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping date strings to market data dataframes
    """
    market_data_dir = "Holder_Strat/Approved_Checked_Market_Data"
    bulk_summaries_path = "Holder_Strat/Summary_Csvs/bulk_summaries.csv"
    market_data_dict = {}   # dictionary to store market data

    # Step 1: Load bulk_summaries.csv and extract unique dates
    bulk_df = pd.read_csv(bulk_summaries_path)
    
    # Extract unique dates from the 'Date' column
    unique_dates = bulk_df['Date'].unique().tolist()
    # date needs to be mm-dd-yyyy format
    for i in range (len(unique_dates)):
        unique_dates[i] = bulk_csv_date_converter(unique_dates[i])

    print(f"Found {len(unique_dates)} unique dates in bulk_summaries.csv")
    
    # Step 2: Get list of market data files
    market_data_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]    
    
    # Step 3: Process each market data file
    for filename in market_data_files:
        date_from_filename = filename.split('_')[3]
        if ('.csv' in date_from_filename):
            date_from_filename = date_from_filename[0:-4]

        # Only load if date is in our unique dates list
        if date_from_filename in unique_dates:
            file_path = os.path.join(market_data_dir, filename)
            try:
                # Load the market data CSV
                market_df = pd.read_csv(file_path)
                market_data_dict[date_from_filename] = market_df
                print(f"Loaded market data for {date_from_filename}: {len(market_df)} rows")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Skipping {filename} - date {date_from_filename} not in bulk_summaries.csv")
    
    print(f"\nSuccessfully loaded {len(market_data_dict)} market data files")
    return market_data_dict

def load_bulk_summaries():
    """
    Load the bulk_summaries.csv file into a dataframe.
    
    Returns:
        pd.DataFrame: The bulk summaries dataframe
    """
    bulk_summaries_path = "Holder_Strat/Summary_Csvs/bulk_summaries.csv"
    return pd.read_csv(bulk_summaries_path)

def add_optimal_sl_columns(df):
    """
    Add new columns for optimal stop loss analysis to the dataframe.
    
    Args:
        df: The bulk summaries dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new columns added
    """
    # Add new columns for optimal stop loss analysis
    df['optimal sl result'] = None
    df['optimal sl hit'] = False
    df['optimal sl best exit percent'] = None
    df['optimal sl reached holding'] = False

    # target estimate metrics
    df['optimal sl including max roi'] = 0
    
    return df


# ticker_date_market_data is market data for this trades ticker and date. indexes are likely the original full df's indexes
def process_price_movement(ticker_date_market_data, sl_value, trade_type, entry_time, entry_price, ticker):
    curr_roi_percent = 0
    best_exit_percent = 0
    reached_holding = False
    max_roi = max_rois[ticker]
    # this returns: (optimal sl result, optimal sl hit, optimal sl best exit percent, reached_holding)
    
    # Reset index to make iteration easier
    ticker_data = ticker_date_market_data.reset_index(drop=True)
    
    # Find the start row - either exact match or first time after entry_time
    start_row_idx = None
    for idx, row in ticker_data.iterrows():
        if row['Time'] == entry_time:
            start_row_idx = idx
            break
        elif row['Time'] > entry_time:
            start_row_idx = idx
            break
    
    # If we couldn't find a start row
    if start_row_idx is None:
        raise ValueError(f"couldn't find start row. entry time, entry price, trade type to id trade: {entry_time}, {entry_price}, {trade_type}")
    
    # Iterate through the data starting from the start row
    for idx in range(start_row_idx, len(ticker_data)):
        current_price = ticker_data.iloc[idx]['Price']
        
        # Calculate current ROI percent
        if trade_type == 'buy':
            curr_roi_percent = ((current_price - entry_price) / entry_price) * 100
            # For buy trades, best exit is highest price
            if curr_roi_percent > best_exit_percent:
                best_exit_percent = curr_roi_percent

        else:  # 'short'
            curr_roi_percent = ((entry_price - current_price) / entry_price) * 100
            # For short trades, best exit is when price goes down most (highest positive percent)
            if curr_roi_percent > best_exit_percent:
                best_exit_percent = curr_roi_percent
        
        # Check if stop loss is hit (sl_value is negative, so we check if curr_roi_percent <= sl_value)
        if curr_roi_percent <= sl_value:
            # (optimal sl result, optimal sl hit, optimal sl best exit percent, reached_holding, max_hit)
            return curr_roi_percent, True, best_exit_percent, reached_holding, False
        
        # Check if reached holding target of 0.6%
        if curr_roi_percent >= 0.6:
            reached_holding = True
            # Continue processing but with sl_value set to 0
            sl_value = 0

        if (curr_roi_percent >= max_roi):
            # (optimal sl result, optimal sl hit, optimal sl best exit percent, reached_holding, max_hit)
            return curr_roi_percent, False, best_exit_percent, reached_holding, True
    
    # If we get here, stop loss was never hit
    # (optimal sl result, optimal sl hit, optimal sl best exit percent, reached_holding, max_hit)
    return curr_roi_percent, False, best_exit_percent, reached_holding, False


def get_volatility_bucket_avg_sl(volatility):
    for range in sl_volatility_buckets:
        if (range[2] == None):
            return None

        if (volatility > range[0]):
            if (volatility <= range[1]):
                return range[2]

    raise ValueError(f"couldn't find the optimal sl for this range. volatility: {volatility}")


def analyze_optimal_stop_loss(bulk_df, market_data_dict):
    """
    Analyze optimal stop loss for each trade in the bulk summaries dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with optimal stop loss analysis completed
    """
    # Add the new columns
    bulk_df = add_optimal_sl_columns(bulk_df)
    
    # Process each row
    for idx, row in bulk_df.iterrows():
        # Get volatility bucket average stop loss
        ticker = row['Ticker']
        date = bulk_csv_date_converter(row['Date'])
        entry_time = row['Entry Time']
        entry_price = row['Entry Price']
        volatility_percent = row['Entry Volatility Percent']
        sl_value = get_volatility_bucket_avg_sl(volatility_percent)
        if (sl_value == None):
            # no data for this range, skip trade
            bulk_df.at[idx, 'optimal sl result'] = float('nan')
            bulk_df.at[idx, 'optimal sl hit'] = float('nan')
            bulk_df.at[idx, 'optimal sl best exit percent'] = float('nan')
            bulk_df.at[idx, 'optimal sl reached holding'] = float('nan')
            continue
        date_market_data = market_data_dict[date]
        ticker_date_market_data = date_market_data[date_market_data['Ticker'] == ticker] # this likely keeps the original indexes
        
        # Process price movement
        trade_type = row['Trade Type'] # 'buy' or 'short'
        sl_result, sl_hit, best_exit_percent, reached_holding, max_hit = process_price_movement(
            ticker_date_market_data, sl_value, trade_type, entry_time, entry_price, ticker
        )
        
        # Update the dataframe
        bulk_df.at[idx, 'optimal sl result'] = round(sl_result, 2)
        bulk_df.at[idx, 'optimal sl hit'] = sl_hit
        bulk_df.at[idx, 'optimal sl best exit percent'] = round(best_exit_percent, 2)
        bulk_df.at[idx, 'optimal sl reached holding'] = reached_holding
        if (max_hit == True):
            bulk_df.at[idx, 'optimal sl including max roi'] = max_rois[ticker]
        elif (sl_hit == True):
            bulk_df.at[idx, 'optimal sl including max roi'] = round(sl_result, 2)
        else:
            bulk_df.at[idx, 'optimal sl including max roi'] = round(sl_result, 2)
    
    return bulk_df

def generate_analysis_summary(bulk_df_analyzed):
    """
    Generate a text file summarizing the results of the parameter tuning analysis.
    
    Args:
        bulk_df_analyzed: The analyzed dataframe with optimal stop loss results
    """
    output_path = "Holder_Strat/Summary_Csvs/Tuned_Parameters_Analysis.txt"
    
    with open(output_path, 'w') as f:
        f.write("HOLDER STRATEGY PARAMETER TUNING ANALYSIS RESULTS\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-" * 20 + "\n\n")
        
        # Filter out NaN values for calculations
        valid_df = bulk_df_analyzed.dropna(subset=['optimal sl reached holding', 'optimal sl including max roi'])
        total_rows = len(valid_df)
        
        # Metric 1: Holding reached vs optimal sl reached holding vs total rows
        holding_reached_count = len(valid_df[valid_df['holding reached'] == True])
        optimal_sl_reached_holding_count = len(valid_df[valid_df['optimal sl reached holding'] == True])
        
        f.write(f"1. Trade Completion Analysis:\n")
        f.write(f"   - Total valid trades: {total_rows}\n")
        f.write(f"   - Original holding reached: {holding_reached_count} ({round((holding_reached_count/total_rows)*100, 2)}%)\n")
        f.write(f"   - Optimal SL reached holding: {optimal_sl_reached_holding_count} ({round((optimal_sl_reached_holding_count/total_rows)*100, 2)}%)\n\n")
        
        # Metric 2: Average Best Exit Percent comparison
        avg_best_exit = round(valid_df['Best Exit Percent'].mean(), 2)
        avg_optimal_sl_best_exit = round(valid_df['optimal sl best exit percent'].mean(), 2)
        
        f.write(f"2. Average Best Exit Percent Comparison:\n")
        f.write(f"   - Original strategy average: {avg_best_exit}%\n")
        f.write(f"   - Optimal SL strategy average: {avg_optimal_sl_best_exit}%\n")
        f.write(f"   - Improvement: {round(avg_optimal_sl_best_exit - avg_best_exit, 2)}%\n\n")
        
        # Metric 3: Sum comparison
        sum_optimal_sl_roi = round(valid_df['optimal sl including max roi'].sum(), 2)
        sum_percent_change = round(valid_df['Percent Change'].sum(), 2)
        
        f.write(f"3. Total Return Comparison:\n")
        f.write(f"   - Original strategy total return: {sum_percent_change}%\n")
        f.write(f"   - Optimal SL strategy total return: {sum_optimal_sl_roi}%\n")
        f.write(f"   - Improvement: {round(sum_optimal_sl_roi - sum_percent_change, 2)}%\n\n")
        
        # Individual ticker results
        f.write("INDIVIDUAL TICKER RESULTS:\n")
        f.write("-" * 30 + "\n\n")
        
        unique_tickers = valid_df['Ticker'].unique()
        
        for ticker in sorted(unique_tickers):
            ticker_df = valid_df[valid_df['Ticker'] == ticker]
            ticker_optimal_sl_sum = round(ticker_df['optimal sl including max roi'].sum(), 2)
            ticker_percent_change_sum = round(ticker_df['Percent Change'].sum(), 2)
            ticker_improvement = round(ticker_optimal_sl_sum - ticker_percent_change_sum, 2)
            trade_count = len(ticker_df)
            
            f.write(f"{ticker} ({trade_count} trades):\n")
            f.write(f"   - Original strategy total: {ticker_percent_change_sum}%\n")
            f.write(f"   - Optimal SL strategy total: {ticker_optimal_sl_sum}%\n")
            f.write(f"   - Improvement: {ticker_improvement}%\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Analysis completed successfully.\n")
    
    print(f"Analysis summary saved to: {output_path}")

# Main execution function
def main():
    """
    Main function to execute the parameter tuning analysis.
    """
    print("Starting Holder Strategy Parameter Tuning Analysis...")
    
    # Step 1: Load market data dictionary
    market_data_dict = load_market_data_dictionary()
    
    # Step 2: Load bulk summaries
    bulk_df = load_bulk_summaries()
    print(f"Loaded bulk summaries: {len(bulk_df)} trades")
    
    # Step 3: Analyze optimal stop loss
    bulk_df_analyzed = analyze_optimal_stop_loss(bulk_df, market_data_dict)
    
    # Step 4: Save results
    output_path = "Holder_Strat/Summary_Csvs/bulk_summaries_with_optimal_sl.csv"
    bulk_df_analyzed.to_csv(output_path, index=False)
    print(f"Saved analyzed results to: {output_path}")
    
    # Step 5: Generate analysis summary
    generate_analysis_summary(bulk_df_analyzed)
    
    return market_data_dict, bulk_df_analyzed



if __name__ == "__main__":
    market_data_dict, analyzed_df = main()