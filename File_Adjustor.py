import os
import csv

# change past csv market data files to include the new volatility %
# (atr14 / price) * 100
def Add_Volatility_Percent():
    market_data_dir = 'Csv_Files/2_Raw_Market_Data/Market_Data'
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
    market_data_dir = 'Csv_Files/2_Raw_Market_Data/Market_Data'
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


# changes 1 column name in all csv files
def Change_Column_Name():
    #market_data_dir = 'Csv_Files/2_Raw_Market_Data/Market_Data'
    market_data_dir = 'Csv_Files/2_Raw_Market_Data/Market_Data'
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
    market_data_dir = 'Csv_Files/2_Raw_Market_Data/Market_Data'
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


# Test the function
if __name__ == "__main__":
    Change_Column_Name()