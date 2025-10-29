import pandas as pd
import os
import inspect
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Main_Globals
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
import csv

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

'''
each bullet is its own function in this file

Editing:
-create a new csv of all 1 ticker (so split csv files into ticker files)
-reorganize csv by columns
-drop certain columns
-merge market data files into 1 file. works with the files having different columns
-move all files in a folder to another folder
-change all timestamps in 1 file. good for if I forget to set the starting timestamp in on demand
-add volatility $ column

Anaylsis
-find the avg time that each ticker takes to reach volatility % x. useful for seeing how long a ticker 
        takes to "normalize" at market open


Checking:
??check each csv columns are in right order
??Check counts of each ticker to tell if one was forgotten for some reason
'''



# make a new csv of all 1 ticker
def Ticker_Csv():
    try:
        ticker = 'SOXL'
        file_path = 'Csv_Files/raw_Market_Data/Market_Data/Raw_Market_Data_06-18-2025.csv'
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


# drop certain columns
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


# finds the average time that each ticker reaches volatility x. reason is if it's still under by like 6:45 is 
#      that normal or bad?
def Volatility_Time_By_Ticker():
    try:
        data_dir = "Csv_Files/raw_Market_Data/Used_Market_Data"
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


# change past csv market data files to include the new volatility %
# (atr14 / price) * 100
def Add_Volatility_Percent():
    try:
        dir = "Csv_Files/raw_Market_Data/Market_Data"
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
    dir = "Csv_Files/raw_Market_Data/Market_Data"
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
    #market_data_dir = 'Csv_Files/raw_Market_Data/Market_Data'
    market_data_dir = 'Csv_Files/raw_Market_Data/Market_Data'
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
    market_data_dir = 'Csv_Files/raw_Market_Data/Market_Data'
    file_path = f"{market_data_dir}/Raw_Market_Data_09-12-2025.csv"
    temp_file_path = f"{market_data_dir}/temp_Raw_Market_Data_09-12-2025.csv"
    
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


# replaces all values in a column with rounded values
# columns is a list of columns: ['Volatility Percent', 'Volatility Ratio']
# decimal_places is what we round to. 2 means round( ,2)
def Round_Whole_Column_Values(csv_path, columns, decimal_places=2):
    try:
        temp_file_path = csv_path.replace('.csv', '_temp.csv')
        
        print(f"Processing {csv_path}...")
        print(f"Rounding columns: {columns} to {decimal_places} decimal places")
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as input_file, \
             open(temp_file_path, 'w', newline='', encoding='utf-8') as output_file:
            
            reader = csv.reader(input_file)
            writer = csv.writer(output_file)
            
            # Read and write header row
            header = next(reader)
            writer.writerow(header)
            
            # Find column indices for the columns we want to round
            column_indices = []
            for column in columns:
                if column in header:
                    column_indices.append(header.index(column))
                    print(f"Found column '{column}' at index {header.index(column)}")
                else:
                    print(f"Warning: Column '{column}' not found in CSV file")
            
            if not column_indices:
                print("No valid columns found to round. Exiting without changes.")
                os.remove(temp_file_path)
                return
            
            # Process each data row
            rows_processed = 0
            for row in reader:
                if len(row) > max(column_indices):  # Ensure row has enough columns
                    # Round values in specified columns
                    for col_index in column_indices:
                        try:
                            # Only round if the value is not empty
                            if row[col_index] and row[col_index].strip():
                                original_value = float(row[col_index])
                                rounded_value = round(original_value, decimal_places)
                                row[col_index] = str(rounded_value)
                        except ValueError:
                            # If conversion fails, keep the original value
                            print(f"Warning: Could not convert '{row[col_index]}' to float, keeping original value")
                            pass
                
                writer.writerow(row)
                rows_processed += 1
        
        # Replace original file with modified file
        os.remove(csv_path)
        os.rename(temp_file_path, csv_path)
        
        print(f"Completed processing {rows_processed} rows")
        print(f"Rounded values in columns: {columns}")
        
    except Exception as e:
        # Clean up temp file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# swtich the order of columns
def Re_Order_Columns():
    # --- Load and reorder columns for Raw_Market_Data_06-20-2025_Final.csv ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "Csv_Files", "raw_Market_Data", "Raw_Market_Data_09-12-2025.csv")
    desired_order = [
        'Ticker', 'Price', 'Val', 'Avg', 'Atr14', 'Atr28', 'Rsi', 'Volume',
        'Adx28', 'Adx14', 'Adx7', 'Volatility Percent', 'Volatility Ratio', 'Time']

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



    
#------------------------------------------------------------------------------------------

# validate the file name is right and the date is correct format
def Check_File_Name(file_path, market_data_file):
    try:
        parts = market_data_file.split("_")
        parts[-1] = parts[-1][:-4]  # cut off '.csv' from the end (we already know it's there)

        if parts[:3] != ["Raw", "Market", "Data"]:
            print(f"BAD: bad filename: {market_data_file}")
            return False
        
        if len(parts) > 4:
            if (parts[4:6] != ['On', 'Demand']):
                print(f"BAD: bad filename: {market_data_file}")
                return False
        
        month, day, year = parts[3].split("-")
        if (len(month) != 2 or len(day) != 2 or len(year) != 4):
            print(f"BAD: bad filename: {market_data_file}")
            return False
        
        print(f"Filename valid: {market_data_file}")
        return True

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
    

# checks that each required column has values for each row. not all columns are checked
# if it's missing values I either must calculate it and add it, or there was an error in data recording
def Check_Required_Market_Values(file_path, market_data_file):
    try:
        required_column_values = ["Ticker","Price","Val","Avg","Atr14","Atr28","Rsi","Volume","Adx28","Adx14",
                                  "Adx7","Volatility Percent","Volatility Ratio","Time"]
        
        df = pd.read_csv(file_path)
        missing_counts = {col: 0 for col in required_column_values}
        missing_row_numbers = []
        
        for row_idx, row in df.iterrows():
            row_number = row_idx + 2  # +2 because: row_idx is 0-indexed, +1 for 1-indexed, +1 for header
            row_has_missing = False
            
            # Check each required column
            for col in required_column_values:
                if pd.isna(row[col]) or str(row[col]).strip() == '':
                    missing_counts[col] += 1
                    row_has_missing = True
            
            # If this row has any missing values, add to the list (limit to first 5)
            if row_has_missing and len(missing_row_numbers) < 5:
                missing_row_numbers.append(row_number)
        
        # Check if there are any missing values
        total_missing = sum(missing_counts.values())
        
        if total_missing > 0:
            print(f"  ✗ BAD: Missing values found in {market_data_file}")
            print(f"    Total missing values: {total_missing}")
            
            # Print missing counts for each column
            print("    Missing values by column:")
            for col, count in missing_counts.items():
                if count > 0:
                    print(f"      {col}: {count} missing")
            
            # Print first 5 row numbers with missing values
            if missing_row_numbers:
                print(f"    First {len(missing_row_numbers)} row(s) with missing values (regardless of column): {missing_row_numbers}")
            
            return False
        else:
            print(f"  ✓ GOOD: No missing values found in {market_data_file}")
            return True

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
    

# checks market data file and returns false if a row has a gap of at least x seconds compared to the previous row
# also checks if the data has a valid start time (at least 20 timestamps between 6:30:00 and 6:31:00)
def Check_Timestamp_Gaps(market_file_path, market_file):
    try:
        gap_size = 7 # seconds
        
        def time_to_seconds(time_str):
            # Convert HH:MM:SS to total seconds
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        
        df = pd.read_csv(market_file_path)
        
        # Check if Time column exists
        if 'Time' not in df.columns:
            print(f"  Warning: No 'Time' column found in {market_file}")
            return False
        
        # Convert timestamps to seconds for comparison
        time_seconds = []
        for time_str in df['Time']:
            try:
                seconds = time_to_seconds(str(time_str))
                time_seconds.append(seconds)
            except:
                print(f"  Warning: Invalid time format '{time_str}' in {market_file}")
                return False
        
        # Check for valid start time (at least 20 timestamps between 6:30:00 and 6:31:00)
        start_time_630 = time_to_seconds("06:30:00")  # 6:30:00 in seconds
        start_time_631 = time_to_seconds("06:31:00")  # 6:31:00 in seconds
        
        valid_start_timestamps = 0
        for seconds in time_seconds:
            if start_time_630 <= seconds < start_time_631:
                valid_start_timestamps += 1
        
        if valid_start_timestamps < 20:
            print(f"  ✗ BAD: Invalid start time in {market_file}")
            print(f"    Found only {valid_start_timestamps} timestamps between 6:30:00 and 6:31:00 (need at least 20)")
            if (market_file == 'Raw_Market_Data_06-24-2025.csv'):
                print(f"   OK: market data starts at 06:31:21, which is before the first trade (there was 1 trade but I cut it out). "
                      f"so accept this as it now doesn't have any impact. since it's in the morning I can't get reliable on demand "
                      f"data to fill it\n")
            elif (market_file == 'Raw_Market_Data_07-15-2025.csv'):
                print(f"   OK: market data starts at 06:34:52, and there currently aren't any trades for this. it should be noted where "
                      f"ever I keep track of this. Since it's in the morning I can't get reliable on demand data to fill it\n")
            elif (market_file == 'Raw_Market_Data_07-16-2025.csv'):
                print(f"   OK: market data starts at 06:33:18, and there currently aren't any trades for this. it should be noted where "
                      f"ever I keep track of this. Since it's in the morning I can't get reliable on demand data to fill it\n")
            elif (market_file == 'Raw_Market_Data_07-17-2025.csv'):
                print(f"   OK: market data starts at 06:31:22, and there currently aren't any trades for this. it should be noted where "
                      f"ever I keep track of this. Since it's in the morning I can't get reliable on demand data to fill it\n")
            elif (market_file == 'Raw_Market_Data_09-02-2025.csv'):
                print(f"   OK: market data starts at 06:32:22, and there currently aren't any trades for this. it should be noted where "
                      f"ever I keep track of this. Since it's in the morning I can't get reliable on demand data to fill it\n")
            elif (market_file == 'Raw_Market_Data_10-09-2025.csv'):
                print(f"   OK: market data starts at 06:34:19, but I didn't take a trade during that time so it's fine. Since it's in "
                      f"the morning I can't get reliable on demand data to fill it\n")
            else:
                return False
        
        # Check for gaps of x seconds or more (moved outside the conversion loop)
        gaps_found = False
        for i in range(1, len(time_seconds)):
            if time_seconds[i] is None or time_seconds[i-1] is None:
                continue
                
            gap = time_seconds[i] - time_seconds[i-1]
            
            # Check for gaps of X seconds or more
            if gap >= gap_size:
                if not gaps_found:
                    print(f"  **BAD Gaps found in {market_file}:")
                    gaps_found = True

                # Line number is i+2 because: i is 0-indexed, +1 for 1-indexed, +1 for header
                line_number = i + 2
                if (market_file == 'Raw_Market_Data_09-02-2025.csv' and line_number == 78):
                    print(f"{market_file} has gap at line 78 of 6 seconds. it's too early to get replacement data. live with it it's just 6 seconds")
                    return True
                elif (market_file == 'Raw_Market_Data_09-04-2025.csv' and line_number == 72):
                    print(f"{market_file} has gap at line 72 of 7 seconds. it's too early to get replacement data. live with it it's just 7 seconds")
                    return True
                elif (market_file == 'Raw_Market_Data_09-04-2025.csv' and line_number == 3075):
                    print(f"{market_file} has gap at line 3075 of 6 seconds. it's too early to get replacement data. live with it it's just 6 seconds")
                    return True
                elif (market_file == 'Raw_Market_Data_10-03-2025.csv' and line_number == 681):
                    print(f"{market_file} has gap at line {line_number} of {gap} seconds. it's too early to get replacement data. live with it it's just {gap} seconds")
                    return True
                elif (market_file == 'Raw_Market_Data_10-07-2025.csv' and line_number == 3432):
                    print(f"{market_file} has gap at line {line_number} of {gap} seconds. it's too early to get replacement data. live with it it's just {gap} seconds")
                    return True
                elif (market_file == 'Raw_Market_Data_10-16-2025.csv' and (line_number == 877 or line_number == 905)):
                    print(f"{market_file} has gap at line {line_number} of {gap} seconds. it's too early to get replacement data. live with it it's just {gap} seconds")
                    return True
                else:
                    print(f"    Line {line_number}: Gap of {gap} seconds")
        
        if not gaps_found:
            print(f"  ✓ GOOD: No gaps of {gap_size}+ seconds found and valid start time in {market_file}")
            print(f"    Found {valid_start_timestamps} timestamps between 6:30:00-6:31:00")
            return True
        else:
            raise ValueError()

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# checks if any tickers in market data were dropped from recording. it counts them and returns false if the max-min is > x
def Check_Ticker_Counts_Consistancy(market_file_path, market_file):
    try:
        df = pd.read_csv(market_file_path)
        
        # Count occurrences of each ticker in the 'Ticker' column
        ticker_counts = df['Ticker'].value_counts().to_dict()
        
        # Print the count of each ticker
        print(f"\nTicker counts for {market_file}:")
        print("  " + ", ".join(f"{ticker}: {count}" for ticker, count in ticker_counts.items()))
        
        # Check if the difference between largest and smallest count is >= 5
        if len(ticker_counts) > 0:
            max_count = max(ticker_counts.values())
            min_count = min(ticker_counts.values())
            difference = max_count - min_count
            
            if difference >= 5:
                print(f"  ✗ BAD: Ticker count inconsistency detected in {market_file}")
                print(f"    Difference ({difference}) >= 5")
                if (min_count == 16302 and difference == 79):
                    print("you know about this one, it's 5-20-2025, tesla, 79 difference. I don't really care so I'm skipping it")
                    return True
                if (market_file == 'Raw_Market_Data_09-02-2025.csv'):
                    print("you don't know what caused this, but you don't really care so you're skipping it")
                    return True
                return False
            else:
                print(f"  ✓ GOOD: Ticker counts are consistent in {market_file}")
                print(f"    Difference ({difference}) < 5")
                return True
        else:
            print(f"  ✗ BAD: No tickers found in {market_file}")
            return False

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
    

# checks if market data header row is in right order AND if there's no duplicate header rows
def Check_Market_Data_Column_Order(market_file_path, market_file):
    try:
        expected_headers = ["Ticker", "Price", "Val", "Avg", "Atr14", "Atr28", "Rsi", "Volume", "Adx28", 
                            "Adx14", "Adx7", "Volatility Percent", "Volatility Ratio", "Time"]
        
        df = pd.read_csv(market_file_path)
        actual_headers = list(df.columns)
        
        if actual_headers != expected_headers:
            print(f"✗ BAD: Header order mismatch in {market_file}")
            print(f"  Expected: {expected_headers}")
            print(f"  Actual:   {actual_headers}")

            if ('Macd Z-Score' in actual_headers):
                user_input = input(f"z score detected in {market_file}, do you want to remove the z score column from the data and header? "
                                   f"this affects every row. enter 'y' for yes, anything else for no: ")
                if user_input.lower() == "y":
                    # Remove the z score column from the dataframe
                    df = df.drop(columns=['Macd Z-Score'])
                    # Update the actual_headers list
                    actual_headers = list(df.columns)
                    print(f"✓ Removed 'Macd Z-Score' column from {market_file}")
                    
                    # Save the modified dataframe back to the CSV file
                    df.to_csv(market_file_path, index=False)
                    print(f"✓ Updated CSV file: {market_file}")
                    
                    # Return a special value to indicate the file was modified and needs re-checking
                    return "RECHECK"
                
            return False
        
        # Check for duplicate header rows throughout the file
        with open(market_file_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            header_row = next(csv_reader)  # Get the first header row
            
            line_number = 2  # Start from line 2 (after header)
            duplicate_headers_found = []
            
            for row in csv_reader:
                # Check if this row matches the header row
                if row == header_row:
                    duplicate_headers_found.append(line_number)
                    print(f"    ✗ Duplicate header found at line {line_number}")
                
                line_number += 1
        
        if duplicate_headers_found:
            print(f"  ✗ BAD: Found {len(duplicate_headers_found)} duplicate header row(s) at lines: {duplicate_headers_found}")
            return False
        else:
            print(f"    ✓ No duplicate header rows and headers are in right order for: {market_file}")
            return True

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Fix_Morning_Atr_Issue(file_path):
    try:
        df = pd.read_csv(file_path)

        if ('Early Morning Atr Warmup Fix' not in df.columns or isinstance(df.at[10, 'Early Morning Atr Warmup Fix'], float)):
            df['Early Morning Atr Warmup Fix'] = float('nan')  # Prepare new column with NaNs

            for idx, row in df.iterrows():
                row_vol_percent = row['Volatility Percent']
                row_time = row['Time']

                # get minutes since open
                time_obj = datetime.strptime(row_time, '%H:%M:%S').time()
                current_minutes = time_obj.hour * 60 + time_obj.minute
                minutes_since_open = round(current_minutes - (6*60+30)) # 6:30 am

                # change value based on time
                if (minutes_since_open < 9):
                    df.at[idx, 'Early Morning Atr Warmup Fix'] = round(row_vol_percent * 1.8, 2)
                elif (minutes_since_open >= 9 and minutes_since_open < 12):
                    df.at[idx, 'Early Morning Atr Warmup Fix'] = round(row_vol_percent * 1.6, 2)
                elif (minutes_since_open >= 12 and minutes_since_open < 14):
                    df.at[idx, 'Early Morning Atr Warmup Fix'] = round(row_vol_percent * 1.2, 2)
                else:
                    break
            
            # Place the new column immediately after 'Volatility Percent'
            df = df[["Ticker","Price","Val","Avg","Atr14","Atr28","Rsi","Volume","Adx28","Adx14","Adx7","Volatility Percent","Early Morning Atr Warmup Fix","Volatility Ratio","Time"]]

            df.to_csv(file_path, index=False)
            print(f"Saved warmup fix to: {file_path}")
            return True
        else:
            return True

    except Exception as e:
        print(f"***ERROR: problem with fixing the early monring volatility percent. file: {file_path}, error: {str(e)}")
        return False


# controller to handle all validity checks of csv files
def Authenticator_Freeway():
    try:
        market_data_dir = "Csv_Files/raw_Market_Data/market_data_to_check"
        #market_data_dir = 'Holder_Strat/Approved_Checked_Market_Data'
        market_data_csv_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]

        # the reason I split it into for loops and not 1 big for loop is so the results are organized

        # 1) check market order header is correct and there's not duplicate headers
        for market_data_file in market_data_csv_files:
            file_path = f"{market_data_dir}/{market_data_file}"
            result = Check_Market_Data_Column_Order(file_path, market_data_file)
            if result == "RECHECK":
                # Re-run the check for this file after modification
                print(f"Re-checking {market_data_file} after modification...")
                result = Check_Market_Data_Column_Order(file_path, market_data_file)
            if result == False or result == 'RECHECK':
                return False
        
        # 2) check counts of tickers for each file. this'll tell if a ticker stopped being recorded for some reason
        for market_data_file in market_data_csv_files:
            file_path = f"{market_data_dir}/{market_data_file}"
            if (Check_Ticker_Counts_Consistancy(file_path, market_data_file) == False):
                return False
        
        # 3) check time gaps (if there's a big gap in recording)
        for market_data_file in market_data_csv_files:
            file_path = f"{market_data_dir}/{market_data_file}"
            if (Check_Timestamp_Gaps(file_path, market_data_file) == False):
                return False
            
        # 4) check required row values: every row has Ticker,Price,Val,Avg,Atr14,Atr28,Rsi,Volume,Adx28,Adx14,Adx7,Volatility Percent,Volatility Ratio,Time
        for market_data_file in market_data_csv_files:
            file_path = f"{market_data_dir}/{market_data_file}"
            if (Check_Required_Market_Values(file_path, market_data_file) == False):
                return False
        print("\n")

        # 5) make sure the file name is correct and date is correct format
        for market_data_file in market_data_csv_files:
            file_path = f"{market_data_dir}/{market_data_file}"
            if (Check_File_Name(file_path, market_data_file) == False):
                return False
        
        # 6) fix the atr14 warmup period issue (volatility percent)
        print("\nchecking if files have the early morning volatility percent fix. if not then it's created and saved here")
        for market_data_file in market_data_csv_files:
            file_path = f"{market_data_dir}/{market_data_file}"
            if (Fix_Morning_Atr_Issue(file_path) != True):
                return False
        print("voaltility percent is valid for these files")

        print("\nFiles are valid")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
    

#Authenticator_Freeway()


Authenticator_Freeway()

#Re_Order_Columns()
#Round_Whole_Column_Values("Csv_Files/raw_Market_Data/Raw_Market_Data_09-12-2025.csv", ['Volatility Percent', 'Volatility Ratio'], 2)




