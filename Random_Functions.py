import pandas as pd
import os

# make a new csv of all 1 ticker
def Ticker_Csv():
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


# organize a csv by a column - highest to lowest
def Column_Sort_Csv():
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


def Only_Keep_Some_Columns_Csv():
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


# prints a list of all dates of csv's in a dir
def Get_Group_Of_Dates():
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


Get_Group_Of_Dates()