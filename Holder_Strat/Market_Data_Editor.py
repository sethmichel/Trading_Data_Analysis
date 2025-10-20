import pandas as pd
import os
import inspect
from datetime import datetime
import os

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


def Remove_1_Column(csv_path):
    column_to_remove = 'Macd Z-Score'
    
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_path)
        
        # Check if the column exists
        if column_to_remove in df.columns:
            # Remove the specified column
            df = df.drop(columns=[column_to_remove])
            
            # Overwrite the existing CSV file
            df.to_csv(csv_path, index=False)
            print(f"Successfully removed column '{column_to_remove}' from {csv_path}")
        else:
            print(f"Column '{column_to_remove}' not found in {csv_path}")
            
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")




csv_dir = "Csv_Files/raw_Market_Data/market_data_to_check"
csv_path = f"{csv_dir}/Raw_Market_Data_09-05-2025.csv"

Remove_1_Column(csv_path)



