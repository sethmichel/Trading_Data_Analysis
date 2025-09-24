import pandas as pd
import numpy as np


# basically, this tells me info I'll use to choose values to test in grid search
def analyze_bulk_combined_data():
    """
    Analyzes the bulk_combined.csv file to count:
    1. Boolean values for 'last_result_must_be_4_minutes' column
    2. Float values >= threshold for 'result_of_last_2_trades' column
    """
    # Read the CSV file
    csv_path = "Csv_Files/3_Final_Trade_Csvs/Bulk_Combined.csv"
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data with {len(df)} rows")
        print(f"Columns in the dataset: {list(df.columns)}")
        print()
        
        # Check if required columns exist
        required_columns = ['result_of_last_2_trades', 'last_result_must_be_4_minutes']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing columns: {missing_columns}")
            return
        
        # Analyze boolean column: last_result_must_be_4_minutes
        print("=== Boolean Column Analysis: last_result_must_be_4_minutes ===")
        bool_col = 'last_result_must_be_4_minutes'
        
        # Count non-null values first
        non_null_bool = df[bool_col].notna()
        print(f"Total non-null values in {bool_col}: {non_null_bool.sum()}")
        
        if non_null_bool.sum() > 0:
            # Convert to boolean and count
            bool_values = df[bool_col].dropna()
            
            # Handle different possible boolean representations
            if bool_values.dtype == 'object':
                # Convert string representations to boolean
                bool_values = bool_values.map(lambda x: str(x).lower() in ['true', '1', 'yes'])
            
            true_count = bool_values.sum()
            false_count = len(bool_values) - true_count
            
            print(f"Count of True values: {true_count}")
            print(f"Count of False values: {false_count}")
            print(f"Percentage True: {true_count/len(bool_values)*100:.2f}%")
            print(f"Percentage False: {false_count/len(bool_values)*100:.2f}%")
        else:
            print("No valid boolean data found")
        
        print()
        
        # Analyze float column: result_of_last_2_trades
        print("=== Float Column Analysis: result_of_last_2_trades ===")
        float_col = 'result_of_last_2_trades'
        
        # Count non-null values
        non_null_float = df[float_col].notna()
        print(f"Total non-null values in {float_col}: {non_null_float.sum()}")
        
        if non_null_float.sum() > 0:
            # Get numeric values
            float_values = pd.to_numeric(df[float_col], errors='coerce').dropna()
            print(f"Successfully converted {len(float_values)} values to numeric")
            print(f"Range: {float_values.min():.3f} to {float_values.max():.3f}")
            print()
            
            # Test values as specified
            test_values = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            
            print("Threshold analysis (count of values >= threshold):")
            print("Threshold\tCount\tPercentage")
            print("-" * 35)
            
            for threshold in test_values:
                count_gte = (float_values >= threshold).sum()
                percentage = count_gte / len(float_values) * 100
                print(f"{threshold:8.1f}\t{count_gte:5d}\t{percentage:8.2f}%")
        else:
            print("No valid numeric data found")
            
    except FileNotFoundError:
        print(f"Error: Could not find file {csv_path}")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    analyze_bulk_combined_data()
