import pandas as pd
import os


'''
data.csv is the original data. 
force_Original_Data_Idx.csv is the data points of the force plot grouped by similarity, but showing which data.csv index it is
This function makes a new csv of only the data.csv rows used in a range of the force plot I give it (start to end of the force plot)

the result csv shows all the similar rows from the original data together
CRITICAL: looking at the force plot, 
'''
def Cut_Orginal_Csv_by_Force_Plot(start, end):
    force_plot_dir = f"{'XG_Boost/Output'}/Force_Plot_Stuff"
    csv_file = "force_Original_Data_Idx.csv"
    output_csv_name = "focused_original_data_by_force_plot.csv"
    
    # Read the force plot index CSV
    force_plot_path = os.path.join(force_plot_dir, csv_file)
    force_plot_df = pd.read_csv(force_plot_path)
    
    # Filter rows where Force_Plot_Index is between start and end (inclusive)
    filtered_df = force_plot_df[(force_plot_df['Force_Plot_Index'] >= start) & 
                               (force_plot_df['Force_Plot_Index'] <= end)]
    print(filtered_df.head(5))
    if filtered_df.empty:
        raise ValueError(f"No rows found with Force_Plot_Index between {start} and {end}")
    
    # Get the Data_csv_Index values in the order they appear
    data_indices = filtered_df['Data_csv_Index'].tolist()
    force_plot_indices = filtered_df['Force_Plot_Index'].tolist()
    
    # Read the original data CSV
    data_csv_path = os.path.join('XG_Boost', 'Data.csv')
    data_df = pd.read_csv(data_csv_path)
    
    # Check if any indices are out of range
    max_index = len(data_df) - 1  # Since we're using 0-based indexing for data rows
    invalid_indices = [idx for idx in data_indices if idx > max_index]
    if invalid_indices:
        raise ValueError(f"Data_csv_Index values {invalid_indices} are out of range. Max valid index is {max_index}")
    
    # Extract the rows from data_df using the indices
    # Note: data_indices are 0-based for the data rows (row 0 = first data row = row 2 of CSV)
    selected_rows = []
    for i, data_idx in enumerate(data_indices):
        row_data = data_df.iloc[data_idx].tolist()  # Get the actual data row
        # Add force_plot_idx and data_csv_idx as the first two columns
        row_with_metadata = [force_plot_indices[i], data_idx] + row_data
        selected_rows.append(row_with_metadata)
    
    # Create the output DataFrame
    # Get the original headers and add the new ones at the beginning
    original_headers = data_df.columns.tolist()
    new_headers = ['force_plot_idx', 'data_csv_idx'] + original_headers
    
    output_df = pd.DataFrame(selected_rows, columns=new_headers)

    output_df = Clean_Data_2(output_csv_name, output_df)
    
    # Write to output file
    output_path = os.path.join(force_plot_dir, output_csv_name)
    output_df.to_csv(output_path, index=False)
    
    print(f"Successfully created {output_csv_name} with {len(selected_rows)} rows")
    print(f"Output file location: {output_path}")


'''
custom data cleaning meant for specific output files
'''
def Clean_Data_2(file, df):
    if (file == "focused_original_data_by_force_plot.csv"):
        df["Target"] = df["Target"].fillna(0).astype(int)
        df['RSI_Entry_50_Baseline'] = round((df['Entry Rsi'] - 50).abs(), 0) # doesn't save direction. shap: 50 is blue, farther from 50 = red

        drop_cols = ["Date", "Exit Time", "Time in Trade", 'Dollar Change', 'Total Investment', 'Qty', 'Entry Price', 'Exit Price',
            'Best Exit Price', 'Best Exit Percent', 'Worst Exit Price', 'Worst Exit Percent', 'Prev 5 Min Avg Close Volume', 
            'Price_Movement', 'Percent Change']
        missing_cols = [col for col in drop_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")
        else:
            df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    return df


Cut_Orginal_Csv_by_Force_Plot(start=29, end=42)