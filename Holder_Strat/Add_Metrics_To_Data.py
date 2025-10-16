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


# MACD Z-SCORE ------------------------------------------------------------------------
def Add_Macd_Z_Score(market_data_csv_path):
    try:
        # Load the market data from the CSV file
        df = pd.read_csv(market_data_csv_path)
        
        # Define the lookback period for rolling calculations
        lookback_period = 300  # 300 seconds as specified
        
        # 1. Calculate MACD Histogram (MACD Value - MACD Average/Signal)
        df['MACD_Histogram'] = df['Val'] - df['Avg']
        
        # Define a function to calculate the Z-score for a series (a single ticker's data)
        def calculate_z_score(series):
            # 2. Calculate rolling mean and standard deviation
            rolling_mean = series.rolling(window=lookback_period).mean()
            rolling_std = series.rolling(window=lookback_period).std()
            
            # 3. Compute the Z-Score
            z_score = round((series - rolling_mean) / rolling_std, 4)
            
            # 4. Handle the edge case where standard deviation is zero.
            # This occurs if the histogram value is constant over the lookback period.
            # In this case, the deviation from the mean is zero, so the Z-score should be 0.
            z_score[rolling_std == 0] = 0
            return z_score
            
        # Use groupby().transform() to apply the z-score calculation to each ticker's data.
        # Transform ensures the output is aligned with the original DataFrame's index.
        df['Macd Z-Score'] = df.groupby('Ticker')['MACD_Histogram'].transform(calculate_z_score)

        # Drop the intermediate histogram column as it's no longer needed
        df.drop(columns=['MACD_Histogram'], inplace=True)
        
        # Reorder columns to place the new Z-Score column after the 'Avg' column for readability
        if 'Avg' in df.columns:
            cols = df.columns.tolist()
            avg_index = cols.index('Avg')
            cols.remove('Macd Z-Score')
            cols.insert(avg_index + 1, 'Macd Z-Score')
            df = df[cols]

        # 5. Save the modified dataframe back to the original CSV file
        df.to_csv(market_data_csv_path, index=False)
        
        print(f"Successfully added 'Macd Z-Score' column to: {market_data_csv_path}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, "Add_Macd_Z_Score", str(e), sys.exc_info()[2].tb_lineno)
# -------------------------------------------------------------------------------------


# ADX RATIO ---------------------------------------------------------------------------
def Add_ADX_Ratio(market_data_csv_path):
    try:
        df = pd.read_csv(market_data_csv_path)

        # Check if 'Adx7' or 'Adx28' columns contain any zero values.
        if (df['Adx7'] == 0).any() or (df['Adx28'] == 0).any():
            raise ValueError("ADX values cannot be zero for ratio calculation. Found zeros in 'Adx7' or 'Adx28'.")

        # Calculate the ADX Ratio and round to 3 decimal places.
        df['Adx Ratio'] = (df['Adx7'] / df['Adx28']).round(3)

        # Save the modified dataframe back to the original CSV file
        df.to_csv(market_data_csv_path, index=False)

        print(f"Successfully added 'Adx Ratio' column to: {market_data_csv_path}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, "Add_ADX_Ratio", str(e), sys.exc_info()[2].tb_lineno)
# -------------------------------------------------------------------------------------


# ADX SLOPE ---------------------------------------------------------------------------
def Add_ADX_Slope(market_data_csv_path):
    try:
        df = pd.read_csv(market_data_csv_path)

        # Sort values to ensure chronological order for each ticker, which is critical for the rolling window logic.
        df.sort_values(by=['Ticker', 'Time'], inplace=True)

        # A dictionary to hold the list of recent Adx7 values for each ticker.
        adx7_windows = {}
        # A list to store the calculated slope for each row.
        slopes = []
        window_size = 178

        # Iterate over each row to perform the calculation.
        for index, row in df.iterrows():
            ticker = row['Ticker']
            current_adx7 = row['Adx7']

            # Initialize a new list for a ticker if it's the first time we see it.
            if ticker not in adx7_windows:
                adx7_windows[ticker] = []
            
            # get that tickers values
            window = adx7_windows[ticker]

            # If the list has reached 178 entries, we can calculate the slope.
            if len(window) == window_size:
                # The slope is the difference between the current value and the oldest value in our list.
                slope = round(current_adx7 - window[0], 2)
                slopes.append(slope)
                # Remove the oldest value to maintain the window size for the next row.
                window.pop(0)
            else:
                # If we don't have 178 data points yet, we can't calculate the slope.
                slopes.append(pd.NA)
            
            # Add the current value to the end of the list for the next calculation.
            window.append(current_adx7)

        # Assign the list of calculated slopes to a new column in the DataFrame.
        df['Adx7 Slope'] = slopes

        # Construct the output file path by appending '_Final'
        output_dir = os.path.dirname(market_data_csv_path)
        base_filename = os.path.basename(market_data_csv_path)
        new_filename = os.path.splitext(base_filename)[0] + '_Final.csv'
        output_csv_path = os.path.join(output_dir, new_filename)

        # Save the processed dataframe to the new CSV file
        df.to_csv(output_csv_path, index=False)

        print(f"Successfully added 'Adx7 Slope' and saved to: {output_csv_path}")

    except Exception as e:
        # Use the existing error handler to log any issues
        Main_Globals.ErrorHandler(fileName, "Add_ADX_Slope", str(e), sys.exc_info()[2].tb_lineno)
# -------------------------------------------------------------------------------------

# ADX 7v14 & 7v28 ---------------------------------------------------------------------
def Add_ADX_Comparisons(market_data_csv_path):
    try:
        # Load the market data from the CSV file
        df = pd.read_csv(market_data_csv_path)

        # Check if the required ADX columns exist
        if 'Adx7' not in df.columns or 'Adx14' not in df.columns or 'Adx28' not in df.columns:
            raise ValueError("CSV file must contain 'Adx7', 'Adx14', and 'Adx28' columns.")

        # Calculate 'Adx7 Cross 14'
        df['Adx7 Cross 14'] = (df['Adx7'] - df['Adx14']).round(2)

        # Calculate 'Adx7 Cross 28'
        df['Adx7 Cross 28'] = (df['Adx7'] - df['Adx28']).round(2)

        # Reorder columns to place the new ADX comparison columns second-to-last, with only 'Time' after them
        if 'Time' in df.columns:
            cols = df.columns.tolist()
            time_index = cols.index('Time')
            
            # Remove the new columns from their current positions
            cols.remove('Adx7 Cross 14')
            cols.remove('Adx7 Cross 28')
            
            # Insert them before the 'Time' column
            cols.insert(time_index, 'Adx7 Cross 14')
            cols.insert(time_index + 1, 'Adx7 Cross 28')
            
            df = df[cols]

        # Save the modified dataframe back to the original CSV file
        df.to_csv(market_data_csv_path, index=False)

        print(f"Successfully added ADX comparison columns to: {market_data_csv_path}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, "Add_ADX_Comparisons", str(e), sys.exc_info()[2].tb_lineno)


# change past csv market data files to include the new volatility %
# (atr14 / price) * 100
def Add_Volatility_Percent():
    market_data_dir = 'Csv_Files/raw_Market_Data/Market_Data'
    csv_files_list = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files_list:
        file_path = os.path.join(market_data_dir, csv_file)
        temp_file_path = os.path.join(market_data_dir, f'temp_{csv_file}')
        
        print(f"Processing {csv_file}...")
        
        with open(file_path, 'r', newline='', encoding='utf-8') as input_file, \
             open(temp_file_path, 'w', newline='', encoding='utf-8') as output_file:
            
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            
            # Read header and check if Volatility column already exists
            header = next(reader)
            if 'Volatility Percent' in header:
                print(f"Skipping {csv_file} - Volatility column already exists")
                continue
            
            # Insert Volatility column between Volume and Time
            new_header = header[:8] + ['Volatility Percent'] + header[8:]
            writer.writerow(new_header)
            
            # Process each row
            for row in reader:
                if len(row) >= 9:  # Ensure we have enough columns
                    price = float(row[1])
                    atr14 = float(row[4])
                    
                    # Calculate volatility: (atr14 / price) * 100
                    volatility_percent = round((atr14 / price) * 100, 2)
                    
                    # Insert volatility value between Volume and Time
                    new_row = row[:8] + [volatility_percent] + row[8:]
                    writer.writerow(new_row)
        
        # Replace original file with modified file
        os.remove(file_path)
        os.rename(temp_file_path, file_path)
        
        print(f"Completed processing {csv_file}")
    
    print("All CSV files have been updated with the Volatility column.")


# add volaitlity ratio to all market data csv's
# REQUIRED: must have volatility percent already
def Add_Volatility_Ratio():
    market_data_dir = 'Csv_Files/raw_Market_Data/Market_Data'
    csv_files_list = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]
    
    for csv_file in csv_files_list:
        file_path = os.path.join(market_data_dir, csv_file)
        temp_file_path = os.path.join(market_data_dir, f'temp_{csv_file}')
        
        print(f"Processing {csv_file}...")
        
        with open(file_path, 'r', newline='', encoding='utf-8') as input_file, \
             open(temp_file_path, 'w', newline='', encoding='utf-8') as output_file:
            
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            
            # Read header and check if Volatility column already exists
            header = next(reader)
            if 'Volatility Ratio' in header:
                print(f"Skipping {csv_file} - Volatility column already exists")
                continue
            
            # Check if Volatility Percent column exists (required for calculation)
            if 'Volatility Percent' not in header:
                print(f"Skipping {csv_file} - 'Volatility Percent' column not found (required)")
                continue
            
            # Insert Volatility column between volume and Time
            new_header = header[:9] + ['Volatility Ratio'] + header[9:]
            writer.writerow(new_header)
            
            # Process each row
            for row in reader:
                if len(row) >= 9:  # Ensure we have enough columns
                    atr14 = float(row[4])
                    atr28 = float(row[5])
                    
                    # Calculate volatility: (atr14 / price) * 100
                    volatility_ratio = round((atr14/atr28), 2)
                    
                    # Insert volatility value between Volume and Time
                    new_row = row[:9] + [volatility_ratio] + row[9:]
                    writer.writerow(new_row)
        
        # Replace original file with modified file
        os.remove(file_path)
        os.rename(temp_file_path, file_path)
        
        print(f"Completed processing {csv_file}")
    
    print("All CSV files have been updated with the Volatility column.")



csv_dir = "Csv_Files/raw_Market_Data/Market_Data"

# macd z score
'''for filename in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, filename)
    print(f"Processing: {filename}")
    Add_Macd_Z_Score(file_path)'''
Add_Macd_Z_Score(f"{csv_dir}/Raw_Market_Data_06-24-2025.csv")

# adx slope
for filename in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, filename)
    print(f"Processing ADX Slope for: {filename}")
    Add_ADX_Slope(file_path)


# adx 7v14 & 7v28
for filename in os.listdir(csv_dir):
    file_path = os.path.join(csv_dir, filename)
    print(f"Processing ADX Comparisons for: {filename}")
    Add_ADX_Comparisons(file_path)


