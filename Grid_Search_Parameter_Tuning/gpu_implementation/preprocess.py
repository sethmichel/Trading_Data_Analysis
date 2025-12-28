import pandas as pd
import numpy as np
import glob
import os
import re

# Constants
MARKET_DATA_DIR = '../cleaned_and_verified_market_data'
TRADE_LOGS_PATH = '../all_trade_logs.csv'
OUTPUT_DIR = 'data'
TICKERS = ['SOXL', 'IONQ', 'MARA']
TICKER_MAP = {'SOXL': 0, 'IONQ': 1, 'MARA': 2}

def load_and_process_market_data():
    print("Loading market data...")
    all_files = glob.glob(os.path.join(MARKET_DATA_DIR, "Raw_Market_Data_*.csv"))
    
    dfs = []
    
    for filename in all_files:
        # Extract date from filename: Raw_Market_Data_06-24-2025.csv
        basename = os.path.basename(filename)
        date_match = re.search(r'Raw_Market_Data_(\d{2}-\d{2}-\d{4})\.csv', basename)
        
        if not date_match:
            print(f"Skipping file with invalid name format: {basename}")
            continue
            
        date_str = date_match.group(1)
        
        try:
            df = pd.read_csv(filename)
            
            # Filter tickers
            df = df[df['Ticker'].isin(TICKERS)].copy()
            
            if df.empty:
                continue
                
            # Create Datetime column
            # Combine extracted date string with Time column
            # Time format is expected to be HH:MM:SS
            df['Datetime'] = pd.to_datetime(date_str + ' ' + df['Time'], format='%m-%d-%Y %H:%M:%S')
            
            # Keep only necessary columns
            # We need Ticker and Price primarily. Time is now in Datetime.
            # We preserve Datetime for sorting and merging.
            df = df[['Ticker', 'Datetime', 'Price']]
            
            dfs.append(df)
            
        except Exception as e:
            print(f"Error processing {basename}: {e}")

    if not dfs:
        raise ValueError("No market data loaded!")

    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Map tickers to integers
    combined_df['TickerID'] = combined_df['Ticker'].map(TICKER_MAP).astype(np.int32)
    
    # Sort: Primary TickerID, Secondary Datetime
    print("Sorting market data...")
    combined_df.sort_values(by=['TickerID', 'Datetime'], inplace=True)
    
    # Reset index to create the linear address space (0 to N)
    combined_df.reset_index(drop=True, inplace=True)
    
    # Type casting for GPU optimization
    combined_df['Price'] = combined_df['Price'].astype(np.float32)
    
    # Check size
    total_rows = len(combined_df)
    print(f"Total market data rows: {total_rows}")
    if total_rows > 2_000_000_000: # numbers at this scale behave weird. I don't want to get anywhere near the float32 limit
        print("WARNING: Market data exceeds 2 billion rows! 32-bit indexing will fail.")
        return None
    
    return combined_df

def load_and_process_trades(market_data_df):
    print("Loading trade logs...")
    df = pd.read_csv(TRADE_LOGS_PATH)
    
    # Filter tickers
    df = df[df['Ticker'].isin(TICKERS)].copy()
    
    # Parse datetimes
    # Date is MM-DD-YY (e.g., 06-24-25)
    # Entry Time/Exit Time are HH:MM:SS
    
    # Function to parse date and combine with time
    def combine_date_time(row, time_col):
        # Assuming Date is MM-DD-YY
        date_str = row['Date']
        # Fix year if necessary (e.g. if it's 2 digits)
        parts = date_str.split('-')
        if len(parts[2]) == 2:
             parts[2] = '20' + parts[2]
        date_str_full = "-".join(parts)
        
        return pd.to_datetime(date_str_full + ' ' + row[time_col], format='%m-%d-%Y %H:%M:%S')

    print("Parsing trade times...")
    df['EntryDatetime'] = df.apply(lambda row: combine_date_time(row, 'Entry Time'), axis=1)
    df['ExitDatetime'] = df.apply(lambda row: combine_date_time(row, 'Exit Time'), axis=1)
    
    # Map tickers
    df['TickerID'] = df['Ticker'].map(TICKER_MAP).astype(np.int32)
    
    # Calculate Pointers
    print("Calculating index pointers...")
    
    # We need to find the index in market_data_df where:
    # market_data_df['TickerID'] == trade['TickerID'] AND
    # market_data_df['Datetime'] >= trade['EntryDatetime'] (Start)
    
    # Since market_data is sorted by TickerID then Datetime, we can use searchsorted per ticker
    # But pd.merge_asof is cleaner for alignment, though searchsorted is often faster.
    # Given the scale, merge_asof is fine or we can iterate by ticker.
    # Let's use searchsorted approach for exactness and control on the "Ticker" block
    
    df['MarketDataStartIndex'] = -1
    df['MarketDataEndIndex'] = -1
    
    # Prepare market data arrays for search
    # We'll do this by ticker to ensure we don't cross ticker boundaries
    
    for ticker_name, ticker_id in TICKER_MAP.items():
        # Get market data subset indices
        market_data_mask = market_data_df['TickerID'] == ticker_id
        if not market_data_mask.any():
            continue
            
        market_data_subset = market_data_df[market_data_mask]
        market_data_times = market_data_subset['Datetime'].values
        global_indices = market_data_subset.index.values
        
        # Get trades subset
        trades_mask = df['TickerID'] == ticker_id
        trades_subset = df[trades_mask]
        
        if trades_subset.empty:
            continue
            
        # Find start indices (searchsorted finds where to insert to maintain order)
        # We want the first timestamp >= EntryDatetime
        entry_times = trades_subset['EntryDatetime'].values
        start_positions = np.searchsorted(market_data_times, entry_times, side='left')
        
        # Clip to valid range
        start_positions = np.clip(start_positions, 0, len(market_data_times) - 1)
        
        # Map back to global indices
        # If start_positions[i] points to index k in market_data_times, the global index is global_indices[k]
        # Check if the time matches or is reasonably close? 
        # Actually searchsorted guarantees >=, so we just take it.
        # However, if it's the last element and still less, it might be an issue, but usually Entry is within the day.
        
        # Assign global indices
        valid_indices = global_indices[start_positions]
        df.loc[trades_mask, 'MarketDataStartIndex'] = valid_indices.astype(np.int32)
        
        # Find end indices (ExitDatetime)
        # We want the last timestamp <= ExitDatetime? Or first >=?
        # Typically we want to simulate up to the exit time.
        # Let's find insertion point for ExitDatetime.
        exit_times = trades_subset['ExitDatetime'].values
        end_positions = np.searchsorted(market_data_times, exit_times, side='right') - 1
        end_positions = np.clip(end_positions, 0, len(market_data_times) - 1)
        
        valid_end_indices = global_indices[end_positions]
        df.loc[trades_mask, 'MarketDataEndIndex'] = valid_end_indices.astype(np.int32)

    # Filter out invalid pointers (start > end or -1)
    # This might happen if trade is outside available market data range
    valid_pointer_mask = (df['MarketDataStartIndex'] != -1) & \
                         (df['MarketDataEndIndex'] != -1) & \
                         (df['MarketDataStartIndex'] <= df['MarketDataEndIndex'])
                         
    print(f"Dropped {len(df) - valid_pointer_mask.sum()} trades due to missing market data.")
    df = df[valid_pointer_mask].copy()

    # Select columns to save
    cols_to_keep = [
        'Trade Id', 'TickerID', 'Entry Price', 'Exit Price', 
        'Worst Exit Percent', 'Entry Volatility Percent', 'Entry Volatility Ratio',
        'Entry Adx28', 'Entry Adx14', 'Entry Adx7', 
        'Trade Best Exit Percent', 'Trade Percent Change',
        'MarketDataStartIndex', 'MarketDataEndIndex'
    ]
    
    final_df = df[cols_to_keep].copy()
    
    # Type casting
    float_cols = ['Entry Price', 'Exit Price', 'Worst Exit Percent', 'Entry Volatility Percent', 
                  'Entry Volatility Ratio', 'Entry Adx28', 'Entry Adx14', 'Entry Adx7', 
                  'Trade Best Exit Percent', 'Trade Percent Change']
                  
    for col in float_cols:
        final_df[col] = final_df[col].astype(np.float32)
        
    final_df['MarketDataStartIndex'] = final_df['MarketDataStartIndex'].astype(np.int32)
    final_df['MarketDataEndIndex'] = final_df['MarketDataEndIndex'].astype(np.int32)
    final_df['Trade Id'] = final_df['Trade Id'].astype(np.int32)
    
    return final_df

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    market_data_df = load_and_process_market_data()
    trades_df = load_and_process_trades(market_data_df)
    
    print("Saving to parquet file...")
    market_data_df.to_parquet(os.path.join(OUTPUT_DIR, 'market_data.parquet'), index=False)
    trades_df.to_parquet(os.path.join(OUTPUT_DIR, 'trades.parquet'), index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()

