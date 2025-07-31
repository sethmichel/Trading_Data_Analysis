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
import csv

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


# make a new csv of all 1 ticker
def Ticker_Csv():
    try:
        ticker = 'SOXL'
        file_path = 'Csv_Files/2_Raw_Market_Data/TODO_Market_Data/Raw_Market_Data_06-18-2025.csv'
        df = pd.read_csv(file_path)
        
        # Filter rows where the "Ticker" column matches the specified ticker
        filtered_df = df[df['Ticker'] == ticker]
        
        # Create output directory if it doesn't exist
        output_dir = 'Csv_Files/Testing_Csv_Data'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{output_dir}/{base_name}_{ticker}_filtered.csv"
        
        # Save the filtered data to a new CSV file
        filtered_df.to_csv(output_filename, index=False)
        
        print(f"Created filtered CSV: {output_filename}")
        print(f"Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")
        
        return output_filename
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# organize a csv by a column - highest to lowest
def Column_Sort_Csv():
    try:
        column = 'Volatility'
        file_path = "Csv_Files\Testing_Csv_Data\Raw_Market_Data_06-18-2025_SOXL_filtered.csv"
        
        df = pd.read_csv(file_path)
        sorted_df = df.sort_values(by=column, ascending=False)
        
        # Create output directory if it doesn't exist
        output_dir = 'Csv_Files/Testing_Csv_Data'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{output_dir}/{base_name}_sorted_by_{column}.csv"
        
        # Save the sorted data to a new CSV file
        sorted_df.to_csv(output_filename, index=False)
        
        print(f"Created sorted CSV: {output_filename}")
        print(f"Sorted by {column} column (highest to lowest)")
        print(f"Total rows: {len(sorted_df)}")
        print(f"Highest {column} value: {sorted_df[column].max()}")
        print(f"Lowest {column} value: {sorted_df[column].min()}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Only_Keep_Some_Columns_Csv():
    try:
        columns_to_keep = ['Atr14', 'Volatility', 'Time']
        file_path = "Csv_Files\Testing_Csv_Data\Raw_Market_Data_06-18-2025_SOXL_filtered.csv"
        output_dir = 'Csv_Files/Testing_Csv_Data'
        
        df = pd.read_csv(file_path)
        filtered_df = df[columns_to_keep]
        
        # Create the output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{output_dir}/{base_name}_columns_filtered.csv"
        
        # Save the filtered data to a new CSV file
        filtered_df.to_csv(output_filename, index=False)
        
        print(f"Created columns-filtered CSV: {output_filename}")
        print(f"Original columns: {list(df.columns)}")
        print(f"Kept columns: {columns_to_keep}")
        print(f"Total rows: {len(filtered_df)}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# prints a list of all dates of csv's in a dir
def Get_Group_Of_Dates():
    try:
        file_path = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
        
        # Extract dates from filenames
        dates = []
        for filename in csv_files:
            if filename.startswith('Raw_Market_Data_'):
                # Extract the date part after "Raw_Market_Data_"
                date_part = filename.replace('Raw_Market_Data_', '').replace('.csv', '')
                
                # Remove "_On_Demand" suffix if present
                if '_On_Demand' in date_part:
                    date_part = date_part.replace('_On_Demand', '')
                
                dates.append(date_part)
        
        # Sort dates chronologically
        dates.sort()
        
        # Output to terminal
        print("Available dates in CSV files:")
        print("['" + "', '".join(dates) + "']")
        
        print(f"\nTotal number of dates: {len(dates)}")
        
        return dates
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Print_Volatility_Counts():
    try:
        volatilities = [0.5, 0.6, 0.7, 0.8, 0.9]
        bulk_csv_data = "Csv_Files/3_Final_Trade_Csvs/Bulk_Combined.csv"
        df = pd.read_csv(bulk_csv_data)
        counts = {}

        for volatility in volatilities:
            filtered_df = df[(df['Entry Volatility Percent'] >= volatility) & (df['Entry Volatility Ratio'] >= 1)]
            counts[str(volatility)] = filtered_df.shape[0]

        print(counts)
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# finds the average time that ecah ticker reaches volatility x. reason is if it's still under by like 6:45 is that normal or bad?
def Volatility_Time_By_Ticker():
    try:
        data_dir = "Csv_Files/2_Raw_Market_Data/Used_Market_Data"
        volatility_target = 0.3
        
        # Dictionary to store results organized by date
        results_by_date = {}
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            # Extract date from filename (3rd index after splitting by '_')
            filename_parts = csv_file.split('_')
            if len(filename_parts) >= 4:
                date = filename_parts[3]  # This should be the date part
            else:
                date = "Unknown_Date"
            
            file_path = os.path.join(data_dir, csv_file)
            df = pd.read_csv(file_path)
            
            # Dictionary to store first volatility timestamp for each ticker
            ticker_volatility_times = {}
            
            # Group by ticker and find first occurrence where Volatility Percent >= volatility_target
            for ticker in df['Ticker'].unique():
                ticker_data = df[df['Ticker'] == ticker]
                
                # Find first row where Volatility Percent reaches or exceeds the target
                volatility_reached = ticker_data[ticker_data['Volatility Percent'] >= volatility_target]
                
                if not volatility_reached.empty:
                    # Get the first timestamp where volatility target was reached
                    first_timestamp = volatility_reached.iloc[0]['Time']
                    ticker_volatility_times[ticker] = first_timestamp
                else:
                    # If volatility target was never reached, mark as "Never Reached"
                    ticker_volatility_times[ticker] = "Never Reached"
            
            # Store results for this date
            results_by_date[date] = ticker_volatility_times
        
        # Create output text file
        output_filename = "volatility_time_by_ticker_results.txt"
        
        with open(output_filename, 'w') as f:
            f.write(f"Volatility Time By Ticker Analysis (Target: {volatility_target})\n")
            f.write("=" * 60 + "\n\n")
            
            # Sort dates for consistent output
            sorted_dates = sorted(results_by_date.keys())
            
            for date in sorted_dates:
                f.write(f"Date: {date}\n")
                f.write("-" * 30 + "\n")
                
                ticker_times = results_by_date[date]
                
                # Sort tickers alphabetically for consistent output
                for ticker in sorted(ticker_times.keys()):
                    timestamp = ticker_times[ticker]
                    f.write(f"{ticker}: {timestamp}\n")
                
                f.write("\n")
        
        # Also print to console
        print(f"Analysis complete! Results saved to: {output_filename}")
        print(f"Volatility target: {volatility_target}")
        print(f"Processed {len(csv_files)} CSV files")
        
        # Print summary to console
        for date in sorted_dates:
            print(f"\nDate: {date}")
            ticker_times = results_by_date[date]
            for ticker in sorted(ticker_times.keys()):
                timestamp = ticker_times[ticker]
                print(f"  {ticker}: {timestamp}")
        
        return results_by_date
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


'''
moves all raw trades and raw market data csv's back to the todo folder
'''
def move_all_csvs_back():
    try:
        used_market_data_dir = "Csv_Files/2_Raw_Market_Data/Used_Market_Data"
        todo_market_data_dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
        used_trade_data_dir = "Csv_Files/1_tos_Raw_Trades/Used_Trade_Data"
        todo_trade_data_dir = "Csv_Files/1_tos_Raw_Trades/TODO_Trade_Data"
        
        # Ensure the TODO directories exist
        os.makedirs(todo_market_data_dir, exist_ok=True)
        os.makedirs(todo_trade_data_dir, exist_ok=True)
        
        # Move files from used_market_data_dir to todo_market_data_dir
        market_files_moved = 0
        if os.path.exists(used_market_data_dir):
            for filename in os.listdir(used_market_data_dir):
                if filename.endswith('.csv'):
                    source_path = os.path.join(used_market_data_dir, filename)
                    destination_path = os.path.join(todo_market_data_dir, filename)
                    
                    # Move the file
                    shutil.move(source_path, destination_path)
                    market_files_moved += 1
                    print(f"Moved market data: {filename}")
        
        # Move files from used_trade_data_dir to todo_trade_data_dir
        trade_files_moved = 0
        if os.path.exists(used_trade_data_dir):
            for filename in os.listdir(used_trade_data_dir):
                if filename.endswith('.csv'):
                    source_path = os.path.join(used_trade_data_dir, filename)
                    destination_path = os.path.join(todo_trade_data_dir, filename)
                    
                    # Move the file
                    shutil.move(source_path, destination_path)
                    trade_files_moved += 1
                    print(f"Moved trade data: {filename}")
        
        print(f"\nFile moving complete!")
        print(f"Market data files moved: {market_files_moved}")
        print(f"Trade data files moved: {trade_files_moved}")
        print(f"Total files moved: {market_files_moved + trade_files_moved}")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# if I use the wrong starting timestamp in on demand, this'll correct it
def Change_Timestamps_of_Market_Data():
    try:
        correct_start = "06:30:00"
        file_path = "Raw_Market_Data_05-10-2025_On_Demand.csv"
        output_filename = "Raw_Market_Data_05-10-2025_On_Demand_correct.csv"
        
        df = pd.read_csv(file_path)
        
        # Convert time strings to datetime objects
        time_as_datetime = pd.to_datetime(df['Time'], format='%H:%M:%S')
        
        # Get the first timestamp as a datetime object
        erroneous_start_time = time_as_datetime.iloc[0]
        
        # Get the correct start time as a datetime object (with today's date, which is fine for delta calculation)
        correct_start_dt = pd.to_datetime(correct_start, format='%H:%M:%S')
        
        # Calculate the offset needed to shift the times
        time_offset = correct_start_dt - erroneous_start_time
        
        # Apply the offset to all timestamps and format back to string
        df['Time'] = (time_as_datetime + time_offset).dt.strftime('%H:%M:%S')
        
        # Save the updated dataframe to a new CSV file
        df.to_csv(output_filename, index=False)
        
        print(f"Timestamps corrected and saved to: {output_filename}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# for if you have market data, but take it again adding new indicators
# huge issue is idk if either of them used threading so the order might be off and the timestamps might be 1 second off
# strat: split them into ticker df's, find the start point, and just copy they by row. then merge them and order by timestamp
def Merge_Market_Data():
    try:
        original_path = f"Csv_Files/2_Raw_Market_Data/TODO_Market_Data/Raw_Market_Data_06-24-2025.csv"  # all other data
        new_data_path = f"Csv_Files/2_Raw_Market_Data/TODO_Market_Data/Raw_Market_Data_06-24-2025_On_Demand.csv"  # new indicator columns
        output_csv_path = f"Csv_Files/2_Raw_Market_Data/TODO_Market_Data/Raw_Market_Data_06-24-2025_Final.csv"

        df1 = pd.read_csv(original_path)
        df2 = pd.read_csv(new_data_path)
        
        # Get unique tickers
        tickers = df1['Ticker'].unique()
        
        merged_ticker_dfs = []
        
        for ticker in tickers:
            # Group each ticker into its own dataframe
            ticker_df1 = df1[df1['Ticker'] == ticker].copy()
            ticker_df2 = df2[df2['Ticker'] == ticker].copy()
            
            # Make sure the timestamps are in ascending order
            ticker_df1 = ticker_df1.sort_values(by='Time').reset_index(drop=True)
            ticker_df2 = ticker_df2.sort_values(by='Time').reset_index(drop=True)
            
            # Columns to merge from csv2
            columns_to_merge = ['Adx28', 'Adx14', 'Adx7']
            
            # Copy the columns
            for col in columns_to_merge:
                if col in ticker_df2.columns:
                    ticker_df1[col] = ticker_df2[col]
            
            merged_ticker_dfs.append(ticker_df1)
            
        # Merge the ticker dataframes back together
        final_df = pd.concat(merged_ticker_dfs, ignore_index=True)
        
        # Reorder the columns of the final dataframe
        final_column_order = [
            'Ticker', 'Price', 'Val', 'Avg', 'Atr14', 'Atr28', 'Rsi', 'Volume',
            'Adx28', 'Adx14', 'Adx7', 'Time'
        ]
        
        # Ensure all requested columns exist, creating them if they don't
        for col in final_column_order:
            if col not in final_df.columns:
                final_df[col] = None
        
        final_df = final_df[final_column_order]
        
        # Sort the final dataframe by time
        final_df = final_df.sort_values(by='Time').reset_index(drop=True)
        
        # Save the dataframe as a new csv file
        final_df.to_csv(output_csv_path, index=False)
        
        print(f"Successfully merged CSVs and saved to {output_csv_path}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Append_Csv_To_Other_Csv():
    try:
        base_csv = f"Csv_Files/csvs_to_edit/Raw_Market_Data_06-11-2025_On_Demand.csv"
        csv_to_append = f"Csv_Files/csvs_to_edit/6-11 vol data added.csv"
        dest_path = f"Csv_Files/csvs_to_edit/Raw_Market_Data_06-11-2025_On_Demand_Final.csv"
        
        print(f"Appending {csv_to_append} to {base_csv}...")
        
        # Read the base CSV
        base_df = pd.read_csv(base_csv)
        print(f"Base CSV has {len(base_df)} rows")
        
        # Read the CSV to append
        append_df = pd.read_csv(csv_to_append)
        print(f"CSV to append has {len(append_df)} rows")
        
        # Concatenate the dataframes
        combined_df = pd.concat([base_df, append_df], ignore_index=True)
        print(f"Combined CSV has {len(combined_df)} rows")
        
        # Save the combined dataframe to the destination path
        combined_df.to_csv(dest_path, index=False)
        
        print(f"Successfully appended CSV and saved to: {dest_path}")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# change past csv market data files to include the new volatility %
# (atr14 / price) * 100
def Add_Volatility_Percent():
    try:
        dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
        csv_files_list = ["Raw_Market_Data_06-24-2025.csv"]
        
        for csv_file in csv_files_list:
            file_path = f"{dir}/{csv_file}"
            temp_file_path = f"{dir}/'temp_{csv_file}"
            
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
                new_header = header[:11] + ['Volatility Percent'] + header[11:]
                writer.writerow(new_header)
                
                # Process each row
                for row in reader:
                    if len(row) >= 9:  # Ensure we have enough columns
                        if (row[1] != "" and row[4] != ""):
                            price = float(row[1])
                            atr14 = float(row[4])
                            
                            # Calculate volatility: (atr14 / price) * 100
                            volatility_percent = round((atr14 / price) * 100, 2)
                        else:
                            volatility_percent = ""

                        # Insert volatility percent
                        new_row = row[:11] + [volatility_percent] + row[11:]
                        writer.writerow(new_row)
            
            # Replace original file with modified file
            os.remove(file_path)
            os.rename(temp_file_path, file_path)
            
            print(f"Completed processing {csv_file}")
        
        print("All CSV files have been updated with the Volatility column.")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# add volaitlity ratio to all market data csv's
# REQUIRED: must have volatility percent already
def Add_Volatility_Ratio():
    dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
    csv_files_list = ["Raw_Market_Data_06-24-2025.csv"]

    for csv_file in csv_files_list:
        file_path = f"{dir}/{csv_file}"
        temp_file_path = f"{dir}/'temp_{csv_file}"
        
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
            new_header = header[:12] + ['Volatility Ratio'] + header[12:]
            writer.writerow(new_header)
            
            # Process each row
            for row in reader:
                if len(row) >= 9:  # Ensure we have enough columns
                    if (row[4] != "" and row[5] != ""):
                        atr14 = float(row[4])
                        atr28 = float(row[5])
                        
                        # Calculate volatility: (atr14 / price) * 100
                        volatility_ratio = round((atr14/atr28), 2)
                    else:
                        volatility_ratio = ""
                    
                    # Insert volatility value between Volume and Time
                    new_row = row[:12] + [volatility_ratio] + row[12:]
                    writer.writerow(new_row)
        
        # Replace original file with modified file
        os.remove(file_path)
        os.rename(temp_file_path, file_path)
        
        print(f"Completed processing {csv_file}")
    
    print("All CSV files have been updated with the Volatility column.")


# changes 1 column name in all csv files
def Change_Column_Name():
    #market_data_dir = 'Csv_Files/2_Raw_Market_Data/TODO_Market_Data'
    market_data_dir = 'Csv_Files/2_Raw_Market_Data/USED_Market_Data'
    csv_files_list = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]

    original_column_name = "Vol"
    new_column_name = "Volume"
    
    for csv_file in csv_files_list:
        file_path = os.path.join(market_data_dir, csv_file)
        temp_file_path = os.path.join(market_data_dir, f'temp_{csv_file}')
        
        print(f"Processing {csv_file}...")
        
        with open(file_path, 'r', newline='', encoding='utf-8') as input_file, \
             open(temp_file_path, 'w', newline='', encoding='utf-8') as output_file:
            
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            
            # Read header and check if original column name exists
            header = next(reader)
            if original_column_name not in header:
                print(f"Skipping {csv_file} - '{original_column_name}' column not found")
                continue
            
            # Replace the original column name with the new one
            new_header = [new_column_name if col == original_column_name else col for col in header]
            writer.writerow(new_header)
            
            # Copy all data rows without changes
            for row in reader:
                writer.writerow(row)
        
        # Replace original file with modified file
        os.remove(file_path)
        os.rename(temp_file_path, file_path)
        
        print(f"Completed processing {csv_file}")
    
    print("All CSV files have been updated with the column name change.")


# edits all values in 1 column in 1 csv file
# need this in case you add 0's to the end of numbers accidently. like 0.5800 instead of 0.58
def Edit_Values():
    market_data_dir = 'Csv_Files/2_Raw_Market_Data/TODO_Market_Data'
    file_path = f"{market_data_dir}/Raw_Market_Data_04-09-2025_On_Demand.csv"
    temp_file_path = f"{market_data_dir}/temp_Raw_Market_Data_04-09-2025_On_Demand.csv"
    
    print(f"Processing {file_path}...")
    
    with open(file_path, 'r', newline='', encoding='utf-8') as input_file, \
         open(temp_file_path, 'w', newline='', encoding='utf-8') as output_file:
        
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)
        
        # Read header and find the Volatility Percent column index
        header = next(reader)
        if 'Volatility Percent' not in header:
            print("Error: 'Volatility Percent' column not found in the CSV file")
            return
        
        volatility_index = header.index('Volatility Percent')
        writer.writerow(header)
        
        # Process each row
        for row in reader:
            if len(row) > volatility_index:
                # Get the volatility value and remove trailing zeros
                volatility_value = row[volatility_index]
                try:
                    # Convert to float and back to remove trailing zeros
                    cleaned_value = str(float(volatility_value))
                    row[volatility_index] = cleaned_value
                except ValueError:
                    # If conversion fails, keep the original value
                    pass
            
            writer.writerow(row)
    
    # Replace original file with modified file
    os.remove(file_path)
    os.rename(temp_file_path, file_path)
    
    print("Completed processing - trailing zeros removed from Volatility Percent column.")


# swtich the order of columns
def Re_Order_Columns():
    # --- Load and reorder columns for Raw_Market_Data_06-20-2025_Final.csv ---
    csv_path = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data/Raw_Market_Data_06-24-2025.csv"
    desired_order = [
        'Ticker', 'Price', 'Val', 'Avg', 'Atr14', 'Atr28', 'Rsi', 'Volume',
        'Adx28', 'Adx14', 'Adx7', 'Volatility Percent', 'Volatility Ratio', 'Time'
    ]
    try:
        df = pd.read_csv(csv_path)
        # Reorder columns
        df = df[desired_order]
        print(df.head(2))  # Show the first few rows to verify order
        # Save the reordered DataFrame back to the same file
        df.to_csv(csv_path, index=False)

    except Exception as e:
        print(f"Error loading or reordering columns: {e}")


# split a csv into 2 csv's. for when a csv has data up to a certain point
def Make_New_Csv_At_X_Line():
    try:
        csv_path = "Csv_Files/csvs_to_edit/Raw_Market_Data_06-11-2025_On_Demand.csv"
        dest_path = "Csv_Files/csvs_to_edit/6-11 vol data added.csv"
        split_line = 62832  # This is the line number in the file (1-based)

        print(f"Reading all lines from {csv_path}...")
        with open(csv_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        print(f"Read {len(lines)} lines from the file.")

        # The split_line is 1-based, list of lines is 0-indexed.
        split_index = split_line - 1

        if split_index <= 0 or split_index >= len(lines):
            print(f"Split line {split_line} is out of bounds for the file with {len(lines)} lines. No changes made.")
            return

        # The new file needs the header, which is the first line of the original file.
        header_line = lines[0]
        
        # Lines for the original file (up to the split point)
        original_lines = lines[:split_index]
        
        # Lines for the new file (header + lines from the split point)
        new_csv_lines = [header_line] + lines[split_index:]

        # Write to the destination file
        print(f"Writing {len(new_csv_lines)} lines to {dest_path}...")
        with open(dest_path, 'w', encoding='utf-8') as f_out_new:
            f_out_new.writelines(new_csv_lines)
        print(f"Created new CSV file: {dest_path}")

        # Write back to the original file
        print(f"Writing {len(original_lines)} lines back to {csv_path}...")
        with open(csv_path, 'w', encoding='utf-8') as f_out_orig:
            f_out_orig.writelines(original_lines)
        print(f"Modified original CSV file: {csv_path}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Check_Timestamp_Gaps():
    try:
        gap_size = 7 # seconds
        source_dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
        
        def time_to_seconds(time_str):
            """Convert HH:MM:SS to total seconds"""
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        
        # Process each CSV file in the directory
        csv_files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
        
        for filename in csv_files:
            file_path = os.path.join(source_dir, filename)            
            df = pd.read_csv(file_path)
            
            # Check if Time column exists
            if 'Time' not in df.columns:
                print(f"  Warning: No 'Time' column found in {filename}")
                return
            
            # Convert timestamps to seconds for comparison
            time_seconds = []
            for time_str in df['Time']:
                try:
                    seconds = time_to_seconds(str(time_str))
                    time_seconds.append(seconds)
                except:
                    print(f"  Warning: Invalid time format '{time_str}' in {filename}")
                    return
            
            # Check for gaps of x seconds or more
            gaps_found = False
            for i in range(1, len(time_seconds)):
                if time_seconds[i] is None or time_seconds[i-1] is None:
                    continue
                    
                gap = time_seconds[i] - time_seconds[i-1]
                
                # Check for gaps of X seconds or more
                if gap >= gap_size:
                    if not gaps_found:
                        print(f"  **BAD Gaps found in {filename}:")
                        gaps_found = True
                    
                    # Line number is i+2 because: i is 0-indexed, +1 for 1-indexed, +1 for header
                    line_number = i + 2
                    print(f"    Line {line_number}: Gap of {gap} seconds")
            
            if not gaps_found:
                print(f"  GOOD No gaps of {gap_size}+ seconds found in {filename}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
    


#Print_Volatility_Counts()
#Volatility_Time_By_Ticker()
#move_all_csvs_back()
#Change_Timestamps_of_Market_Data()
#Merge_Market_Data()
#Append_Csv_To_Other_Csv()
#Add_Volatility_Percent()
#Add_Volatility_Ratio()
#Re_Order_Columns()
#Make_New_Csv_At_X_Line()
#Check_Timestamp_Gaps()



'''market_data_dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
other_dir = "Csv_Files/2_Raw_Market_Data/Used_Market_Data"

# Process all files in market_data_dir
for filename in os.listdir(market_data_dir):
    file_path = f"{market_data_dir}/{filename}"
    print(f"Processing: {filename}")

    data_holder = Calcualte_Directional_Bias_V2(file_path)
    Add_Directional_Bias_To_Market_Data(data_holder, file_path)
'''





