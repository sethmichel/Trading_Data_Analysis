import pandas as pd
import os
import inspect
import sys
import shutil
import Main_Globals
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
import os

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

def calculate_minute_ohlc(ticker_df):
    """
    Calculate OHLC (Open, High, Low, Close) for each minute from second-by-second data
    Returns dictionary with minute-by-minute OHLC data
    """
    try:
        ohlc_data = {
            'open': [],
            'high': [],
            'low': [],
            'close': []
        }
        
        current_minute = None
        minute_prices = []
        
        for idx, row in ticker_df.iterrows():
            price = row['Price']
            time_str = row['Time']
            
            # Extract minute from time string (HH:MM:SS)
            time_parts = time_str.split(':')
            minute = int(time_parts[1])
            
            # Check if we're in a new minute
            if current_minute != minute:
                # Process previous minute's data (if any)
                if minute_prices:
                    ohlc_data['open'].append(minute_prices[0])
                    ohlc_data['high'].append(max(minute_prices))
                    ohlc_data['low'].append(min(minute_prices))
                    ohlc_data['close'].append(minute_prices[-1])
                
                # Start new minute
                current_minute = minute
                minute_prices = [price]
            else:
                # Add price to current minute
                minute_prices.append(price)
        
        # Process the last minute
        if minute_prices:
            ohlc_data['open'].append(minute_prices[0])
            ohlc_data['high'].append(max(minute_prices))
            ohlc_data['low'].append(min(minute_prices))
            ohlc_data['close'].append(minute_prices[-1])
        
        return ohlc_data
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None

def calculate_true_range(ohlc_data):
    """
    Calculate True Range for each minute
    TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    """
    try:
        highs = ohlc_data['high']
        lows = ohlc_data['low']
        closes = ohlc_data['close']
        
        tr_values = []
        
        for i in range(len(highs)):
            if i == 0:
                # First minute: no previous close, so TR = High - Low
                tr = highs[i] - lows[i]
            else:
                # Standard TR calculation
                prev_close = closes[i-1]
                tr = max(
                    highs[i] - lows[i],
                    abs(highs[i] - prev_close),
                    abs(lows[i] - prev_close)
                )
            
            tr_values.append(round(tr, 4))
        
        return tr_values
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None

def calculate_directional_movement(ohlc_data):
    """
    Calculate +DM and -DM (Directional Movement)
    +DM = max(High[i] - High[i-1], 0) if High[i] - High[i-1] > Low[i-1] - Low[i], else 0
    -DM = max(Low[i-1] - Low[i], 0) if Low[i-1] - Low[i] > High[i] - High[i-1], else 0
    """
    try:
        highs = ohlc_data['high']
        lows = ohlc_data['low']
        
        plus_dm = []
        minus_dm = []
        
        for i in range(len(highs)):
            if i == 0:
                # First minute: no previous values
                plus_dm.append(0)
                minus_dm.append(0)
            else:
                # Calculate up and down moves
                up_move = highs[i] - highs[i-1]
                down_move = lows[i-1] - lows[i]
                
                # Calculate +DM and -DM
                if up_move > down_move and up_move > 0:
                    plus_dm.append(round(up_move, 4))
                    minus_dm.append(0)
                elif down_move > up_move and down_move > 0:
                    plus_dm.append(0)
                    minus_dm.append(round(down_move, 4))
                else:
                    plus_dm.append(0)
                    minus_dm.append(0)
        
        return plus_dm, minus_dm
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None, None

def wilders_smoothing(values, period=7):
    """
    Apply Wilder's smoothing (modified EMA) to a list of values
    """
    try:
        smoothed = []
        
        for i in range(len(values)):
            if i < period:
                # Not enough data for smoothing yet
                smoothed.append(None)
            elif i == period:
                # First smoothed value: simple average of first 'period' values
                # Only include non-None values in the calculation
                valid_values = [v for v in values[i-period+1:i+1] if v is not None]
                if len(valid_values) > 0:
                    avg = sum(valid_values) / len(valid_values)
                    smoothed.append(round(avg, 4))
                else:
                    smoothed.append(None)
            else:
                # Wilder's smoothing: previous_smooth + (current - previous_smooth) / period
                previous_smooth = smoothed[i-1]
                current = values[i]
                
                # Check if we have valid values to work with
                if previous_smooth is None or current is None:
                    smoothed.append(None)
                else:
                    new_smooth = previous_smooth + (current - previous_smooth) / period
                    smoothed.append(round(new_smooth, 4))
        
        return smoothed
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None

def calculate_di_values(smoothed_plus_dm, smoothed_minus_dm, smoothed_tr):
    """
    Calculate DI+ and DI- values
    DI+ = (Smoothed +DM / Smoothed TR) * 100
    DI- = (Smoothed -DM / Smoothed TR) * 100
    """
    try:
        di_plus = []
        di_minus = []
        
        for i in range(len(smoothed_tr)):
            if smoothed_tr[i] is None or smoothed_tr[i] == 0:
                di_plus.append(None)
                di_minus.append(None)
            else:
                di_plus.append(round((smoothed_plus_dm[i] / smoothed_tr[i]) * 100, 4))
                di_minus.append(round((smoothed_minus_dm[i] / smoothed_tr[i]) * 100, 4))
        
        return di_plus, di_minus
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None, None

def calculate_dx(di_plus, di_minus):
    """
    Calculate DX (Directional Index)
    DX = |DI+ - DI-| / (DI+ + DI-) * 100
    """
    try:
        dx_values = []
        
        for i in range(len(di_plus)):
            if di_plus[i] is None or di_minus[i] is None:
                dx_values.append(None)
            else:
                sum_di = di_plus[i] + di_minus[i]
                if sum_di == 0:
                    dx_values.append(0)
                else:
                    dx = abs(di_plus[i] - di_minus[i]) / sum_di * 100
                    dx_values.append(round(dx, 4))
        
        return dx_values
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None

def Add_ADX(market_data_csv_path):
    """
    Calculate ADX (Average Directional Index) for each ticker in the dataset
    """
    try:
        df = pd.read_csv(market_data_csv_path)
        ticker_values = {}
        unique_tickers = df['Ticker'].unique()
        
        # Split dataframe by ticker and process each one
        for ticker in unique_tickers:
            print(f"Processing ticker: {ticker}")
            
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df.reset_index(drop=True, inplace=True)
            
            # Step 1: Calculate minute-by-minute OHLC data
            ohlc_data = calculate_minute_ohlc(ticker_df)
            if ohlc_data is None:
                continue
            
            # Step 2: Calculate True Range
            tr_values = calculate_true_range(ohlc_data)
            if tr_values is None:
                continue
            
            # Step 3: Calculate Directional Movement (+DM, -DM)
            plus_dm, minus_dm = calculate_directional_movement(ohlc_data)
            if plus_dm is None or minus_dm is None:
                continue
            
            # Step 4: Apply Wilder's smoothing to TR, +DM, -DM
            smoothed_tr = wilders_smoothing(tr_values, 7)
            smoothed_plus_dm = wilders_smoothing(plus_dm, 7)
            smoothed_minus_dm = wilders_smoothing(minus_dm, 7)
            
            # Step 5: Calculate DI+ and DI-
            di_plus, di_minus = calculate_di_values(smoothed_plus_dm, smoothed_minus_dm, smoothed_tr)
            if di_plus is None or di_minus is None:
                continue
            
            # Step 6: Calculate DX (Directional Index)
            dx_values = calculate_dx(di_plus, di_minus)
            if dx_values is None:
                continue
            
            # Step 7: Calculate ADX (smoothed DX)
            adx_values = wilders_smoothing(dx_values, 7)
            
            # Store all calculated values
            ticker_values[ticker] = {
                'ohlc': ohlc_data,
                'true_range': tr_values,
                'plus_dm': plus_dm,
                'minus_dm': minus_dm,
                'smoothed_tr': smoothed_tr,
                'smoothed_plus_dm': smoothed_plus_dm,
                'smoothed_minus_dm': smoothed_minus_dm,
                'di_plus': di_plus,
                'di_minus': di_minus,
                'dx': dx_values,
                'adx': adx_values
            }
            
            # Print some debug info
            #print(f"  Minutes of data: {len(ohlc_data['high'])}")
            #print(f"  First few ADX values: {[x for x in adx_values[:10] if x is not None]}")
            #print(f"  Non-null ADX values: {len([x for x in adx_values if x is not None])}")
            
        return ticker_values
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return None


def Add_ADX_To_CSVs(file_path):
    """
    Process a CSV file, calculate ADX values for each ticker, and add them to the CSV
    """
    try:
        print(f"Processing file: {file_path}")
        
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        
        # Check if "My Adx" column already exists
        if 'My Adx' in df.columns:
            print(f"File {file_path} already has 'My Adx' column. Skipping.")
            return True
        
        # Calculate ADX values for all tickers in the file
        ticker_values = Add_ADX(file_path)
        if ticker_values is None:
            print("Failed to calculate ADX values")
            return False
        
        print(f"Calculated ADX for {len(ticker_values)} tickers")
        
        # Add "My Adx" column to the dataframe
        df['My Adx'] = None
        
        # Track current minute and ADX index for each ticker
        current_minute = None
        adx_indices = {}
        
        # Initialize ADX indices for each ticker
        for ticker in ticker_values.keys():
            adx_indices[ticker] = 0
        
        # Iterate through each row in the dataframe
        for idx, row in df.iterrows():
            ticker = row['Ticker']
            time_str = row['Time']
            
            # Extract minute from time string (HH:MM:SS)
            time_parts = time_str.split(':')
            minute = int(time_parts[1])
            
            # Check if we have ADX data for this ticker
            if ticker in ticker_values and 'adx' in ticker_values[ticker]:
                adx_list = ticker_values[ticker]['adx']
                
                # Check if minute has changed
                if current_minute != minute:
                    current_minute = minute
                    # Advance ADX index for all tickers when minute changes
                    for ticker_key in adx_indices:
                        adx_indices[ticker_key] += 1
                
                # Get the current ADX value for this ticker
                current_adx_index = adx_indices.get(ticker, 0)
                if current_adx_index < len(adx_list):
                    adx_value = adx_list[current_adx_index]
                    # Round ADX value to 2 decimal places if it's not None
                    if adx_value is not None:
                        adx_value = round(adx_value, 2)
                    df.at[idx, 'My Adx'] = adx_value
        
        # Reorder columns to put "My Adx" before "Time" (making "Time" the final column)
        columns = df.columns.tolist()
        if 'My Adx' in columns and 'Time' in columns:
            # Remove both columns from their current positions
            columns.remove('My Adx')
            columns.remove('Time')
            # Add "My Adx" before "Time" at the end
            columns.append('My Adx')
            columns.append('Time')
            df = df[columns]
        
        # Save the modified dataframe back to the same file
        df.to_csv(file_path, index=False)
        print(f"Saved updated file with ADX values to {file_path}")
        
        return True
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return False



# Run the ADX calculation
csv_dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"

# Test with one CSV file first

csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
for csv_file in csv_files:
    file_path = os.path.join(csv_dir, csv_file)
    print(f"Testing with file: {file_path}")
    success = Add_ADX_To_CSVs(file_path)
    if success:
        print("Successfully processed the test file!")
    else:
        print("Failed to process the test file.")

#ticker_values = Add_ADX("test market csv.csv")
#if ticker_values:
#    print(f"\nProcessed {len(ticker_values)} tickers successfully")
#    for ticker, data in ticker_values.items():
#        print(f"{ticker}: {len(data['adx'])} ADX values calculated")