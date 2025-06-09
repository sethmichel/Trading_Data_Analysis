import pandas as pd
import numpy as np
import os
import re

file_dir = "Analyze_Raw_Market_Data/Combined_Data_Csvs"
destination_csv = "_Cross_Data.csv"         # will have a leading ticker in title
source_csv = f"Analyze_Raw_Market_Data/Single_Days_Cross_Data.csv"

def validate_date_input(date_str):
    """
    Validates date input in MM/DD/YYYY format.
    Returns the validated date string if valid, None if invalid.
    """
    # Check format using regex
    pattern = r'^(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/\d{4}$'
    if not re.match(pattern, date_str):
        print("Invalid date format. Please use MM/DD/YYYY format with leading zeros (e.g., 06/15/2024)")
        return None
    
    try:
        # Try to parse the date to ensure it's valid
        pd.to_datetime(date_str, format='%m/%d/%Y')
        return date_str
    except ValueError:
        print("Invalid date. Please enter a valid date.")
        return None


def process_ticker_data():
    # Create directory if it doesn't exist
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    # Read the source CSV
    with open(source_csv, 'r') as f:
        lines = f.readlines()
    
    current_ticker = None
    headers = None
    data_rows = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # If line doesn't contain a comma, it's a ticker symbol
        if ',' not in line:
            # If we have data from previous ticker, process it
            if current_ticker and headers and data_rows:
                destination_file = os.path.join(file_dir, f"{current_ticker}{destination_csv}")
                
                # Check if file exists and has content
                file_exists = os.path.exists(destination_file) and os.path.getsize(destination_file) > 0
                
                # Write to file
                with open(destination_file, 'a') as f:
                    # Write headers if file is new
                    if not file_exists:
                        f.write(','.join(headers) + '\n')
                    
                    # Write data rows
                    for row in data_rows:
                        f.write(','.join(row) + '\n')
            
            # Start new ticker
            current_ticker = line
            headers = None
            data_rows = []
            continue
        
        # If we don't have headers yet, this line contains them
        if not headers:
            headers = line.split(',')
            continue
        
        # Otherwise, this is a data row
        data_rows.append(line.split(','))
    
    # Process the last ticker's data
    if current_ticker and headers and data_rows:
        destination_file = os.path.join(file_dir, f"{current_ticker}{destination_csv}")
        
        # Check if file exists and has content
        file_exists = os.path.exists(destination_file) and os.path.getsize(destination_file) > 0
        
        # Write to file
        with open(destination_file, 'a') as f:
            # Write headers if file is new
            if not file_exists:
                f.write(','.join(headers) + '\n')
            
            # Write data rows
            for row in data_rows:
                f.write(','.join(row) + '\n')


''' WARNING - DANGEROUS '''
def delete_specific_date_ALL_CSVS():
    try:
        # Get date input from user
        while True:
            date_str = input("Enter the date to delete (MM/DD/YYYY format with leading zeros): ")
            validated_date = validate_date_input(date_str)
            if validated_date:
                break
        
        target_date = pd.to_datetime(validated_date, format='%m/%d/%Y').date()
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(file_dir) if f.endswith(destination_csv)]
        print(f"Found {len(csv_files)} CSV files to process")
        
        if not csv_files:
            print("No CSV files found to process.")
            return
            
        files_processed = False
        
        for csv_file in csv_files:
            try:
                file_path = os.path.join(file_dir, csv_file)
                print(f"\nProcessing {csv_file}...")
                
                # Skip if file is empty or only has header
                if os.path.getsize(file_path) <= 0:
                    print(f"Skipping {csv_file} - file is empty")
                    continue
                    
                # Read the CSV file
                df = pd.read_csv(file_path)
                print(f"File shape: {df.shape}")
                
                # Skip if only has header
                if len(df) == 0:
                    print(f"Skipping {csv_file} - file only contains headers")
                    continue
                    
                # Get the column name for the date (first column)
                date_col = df.columns[0]
                print(f"Date column name: {date_col}")
                
                # Convert all dates in the column to datetime
                df['parsed_date'] = pd.to_datetime(df[date_col], format='%m/%d/%Y').dt.date
                
                # Filter out rows with the target date
                original_len = len(df)
                df = df[df['parsed_date'] != target_date]
                rows_deleted = original_len - len(df)
                
                # Remove the temporary parsed_date column
                df = df.drop('parsed_date', axis=1)
                
                if rows_deleted > 0:
                    files_processed = True
                    print(f"Deleted {rows_deleted} rows from {csv_file} with date {target_date}")
                else:
                    print(f"No rows with date {target_date} found in {csv_file}")
                
                # Write back to file
                df.to_csv(file_path, index=False)
                
            except Exception as e:
                print(f"Error processing {csv_file}:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print(f"File path: {file_path}")
                if 'df' in locals():
                    print(f"DataFrame shape: {df.shape}")
                    print(f"DataFrame columns: {df.columns.tolist()}")
                    print(f"First few rows:\n{df.head()}")
                continue
        
        if not files_processed:
            print(f"No data was deleted for date {target_date} from any files.")
        else:
            print(f"\nSuccessfully deleted data for date {target_date}")
                
    except Exception as e:
        print("Error in delete_specific_date_ALL_CSVS:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Current directory: {os.getcwd()}")
        print(f"File directory: {file_dir}")
        print(f"Destination CSV pattern: {destination_csv}")

def delete_column_names():


if __name__ == "__main__":
    process_ticker_data()
    #delete_specific_date_ALL_CSVS()


