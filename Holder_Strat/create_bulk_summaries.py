"""
Bulk Summary CSV Creator

This module provides functionality to combine multiple summary CSV files
into a single bulk_summaries.csv file. It processes all CSV files that 
start with "summary" and excludes files that start with "bulk" (case insensitive).
"""

import pandas as pd
import os
import glob
from pathlib import Path

def create_bulk_summaries_csv(output_dir="Summary_Csvs"):
    """
    Creates a bulk_summaries.csv file by combining all summary CSV files
    that do not start with "bulk" (case insensitive) from specified directories.
    
    Args:
        output_dir (str): Directory where the bulk file will be saved, 
                         relative to current working directory
    
    Returns:
        str: Path to the created bulk CSV file, or None if no files were found
    """
    
    # Define the paths to check for summary CSV files
    summary_dirs = [
        "Summary_Csvs/",
    ]
    
    # Output file path
    output_file = os.path.join(output_dir, "bulk_summaries.csv")
    
    # List to store all dataframes
    all_dataframes = []
    
    print("Looking for summary CSV files...")
    
    # Process each directory
    for summary_dir in summary_dirs:
        if os.path.exists(summary_dir):
            print(f"Checking directory: {summary_dir}")
            
            # Get all CSV files in the directory
            csv_files = glob.glob(os.path.join(summary_dir, "*.csv"))
            
            for csv_file in csv_files:
                filename = os.path.basename(csv_file).lower()
                
                # Skip files that start with "bulk" (case insensitive)
                if filename.startswith("bulk"):
                    print(f"Skipping bulk file: {csv_file}")
                    continue
                
                # Only process files that start with "summary"
                if filename.startswith("summary"):
                    print(f"Processing file: {csv_file}")
                    
                    try:
                        # Read the CSV file
                        df = pd.read_csv(csv_file)
                        
                        # Check if the dataframe is not empty
                        if not df.empty:
                            all_dataframes.append(df)
                            print(f"  Added {len(df)} rows from {os.path.basename(csv_file)}")
                        else:
                            print(f"  Skipped empty file: {os.path.basename(csv_file)}")
                    
                    except Exception as e:
                        print(f"  Error reading {csv_file}: {str(e)}")
        else:
            print(f"Directory does not exist: {summary_dir}")
    
    # Check if we found any files to combine
    if not all_dataframes:
        print("No summary CSV files found to combine!")
        return None
    
    print(f"\nCombining {len(all_dataframes)} CSV files...")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Sort by Date and then by Entry Time for better organization
    try:
        # Convert Date column to datetime for proper sorting
        combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%m-%d-%y', errors='coerce')
        combined_df = combined_df.sort_values(['Date', 'Entry Time'], ascending=[True, True])
        
        # Convert Date back to original format
        combined_df['Date'] = combined_df['Date'].dt.strftime('%m-%d-%y')
    except Exception as e:
        print(f"Warning: Could not sort by date/time: {str(e)}")
        print("Proceeding without sorting...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the combined dataframe to CSV
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nBulk summary file created successfully!")
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {len(combined_df.columns)}")
    
    # Display some basic statistics
    if 'Ticker' in combined_df.columns:
        print(f"Unique tickers: {combined_df['Ticker'].nunique()}")
        print(f"Most common tickers:")
        print(combined_df['Ticker'].value_counts().head(5))
    
    return output_file

def main():
    """
    Main function to run the bulk summaries creation when script is executed directly.
    """
    # Change to the script directory to ensure relative paths work correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Run the function
    result = create_bulk_summaries_csv()
    
    if result:
        print(f"\nSuccess! Bulk summaries file created at: {result}")
    else:
        print("\nFailed to create bulk summaries file.")

if __name__ == "__main__":
    main()
