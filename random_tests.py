'''
write code in random_tests.py that goes over each csv file in Csv_Files/1_tos_Raw_Trades-TODO/Old_Data and edits thier titles.
after 'Old_', which they all start with, write 'Raw_', keep the rest of the title the same
'''

import os
import glob

def rename_csv_files():
    """
    Rename CSV files in Csv_Files/1_tos_Raw_Trades-TODO/Old_Data directory
    Add 'Old_' to the beginning of each filename
    """
    # Path to the directory containing the CSV files
    csv_directory = "Csv_Files/1_tos_Raw_Trades-TODO/Old_Data"
    
    # Check if directory exists
    if not os.path.exists(csv_directory):
        print(f"Directory {csv_directory} does not exist!")
        return
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process...")
    
    # Process each CSV file
    for file_path in csv_files:
        # Get the filename from the path
        filename = os.path.basename(file_path)
        
        # Create new filename by adding 'Old_' to the beginning
        new_filename = "Old_" + filename
        new_file_path = os.path.join(csv_directory, new_filename)
        
        try:
            # Rename the file
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")
        except Exception as e:
            print(f"Error renaming {filename}: {e}")
    
    print("File renaming completed!")

if __name__ == "__main__":
    rename_csv_files()