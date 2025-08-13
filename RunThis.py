import Parser
import pandas as pd
import os
import sys
import inspect
import Main_Globals
import datetime

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

# 1) put raw tos csv's in 1_tos_Raw_Trades-TODO
# 2) put market data in 2_Raw_Market_Data
# 3) Running this makes new csv's in 3_Final_Trade_Csvs

actual_trade_logs_dir = 'Csv_Files/1_tos_Raw_Trades/TODO_Trade_Data'          # actual trade logs
edited_cross_logs_dir = 'Csv_Files/1_tos_Raw_Trades/Edited_All_Trades_Data'   # edited cross logs
market_data_logs_dir = 'Csv_Files/2_Raw_Market_Data/TODO_Market_Data'
summarized_final_trades_dir = 'Csv_Files/3_Final_Trade_Csvs'

target_dir = None # this'll be either actual_trade_logs_dir or edited_cross_logs_dir

headers = ["Date", "Amount", "Entry Time", "Exit Time", "Ticker", "Entry", "Exit", "Shares", "$ Change", "% Change", 
           "Buy/Short", "Technical Type", "High", "Low"]

# files are just the filename, not paths
def Create_Summary_Csv(date, market_data_file, target_log_file, mode):
    try:
        # prep work
        # 1: put raw trades in 1_tos_Raw_Trades-TODO
        # 2: put market data in 2_Raw_Market_Data
        # when it's done it'll move the tos raw trades csv into 1_tos_Raw_Trades-DONE
        # when it's done it'll make a new csv in 3_Final_Trade_Csvs
        
        if (market_data_file == None or target_log_file == None):
            return
        
        trade_summary_name = f"{summarized_final_trades_dir}/Summary_{date}.csv"
        market_file_path = f"{market_data_logs_dir}/{market_data_file}"
        if mode == 1:
            target_path = f"{actual_trade_logs_dir}/{target_log_file}"
        elif mode == 2:
            target_path = f"{edited_cross_logs_dir}/{target_log_file}"
        
        # 2) normalize raw trades so it's readable and each trade is 1 line.
        print(f"--{market_data_file} & {target_log_file}")

        raw_trade_df = Parser.CreateDf(target_path)            # creates the df of readable raw data
        normalized_df = Parser.Normalize_Raw_Trades(raw_trade_df)  # makes the data readable (not adding market data yet)
        normalized_df = Parser.Add_Running_Sums(normalized_df)

        # 3) add market data to the trade summary
        final_df = Parser.Add_Market_Data(normalized_df, market_file_path)

        # 4) save to new csv file
        final_df.to_csv(trade_summary_name, index=False)
        print(f"Trade summary saved to: {trade_summary_name}")

        # 5) move the used market data/trade log to used folders
        #Parser.Move_Processed_Files(raw_trades_name, raw_market_data_name, tos_raw_trades_DONE_dir, market_data_DONE_dir)
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# choose if we're using trade logs or edited cross logs
def Get_User_Input():
    global target_dir

    while True:
        mode = int(input("Select mode:\n1 - Use actual trade logs\n2 - Use edited cross logs\nEnter 1 or 2: "))
        if mode == 1:
            target_dir = actual_trade_logs_dir
            return mode
        elif mode == 2:
            target_dir = edited_cross_logs_dir
            return mode
        else:
            print("Invalid input. Please enter 1 or 2.")


# create any directories that don't exist for whatever reason
def Create_Directories_If_Not_Exist():
    """Check if required directories exist, create them if they don't"""
    print("Checking if any directories don't exist...")
    directories = [
        actual_trade_logs_dir,
        edited_cross_logs_dir,
        market_data_logs_dir,
        summarized_final_trades_dir
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


# actual trade logs are ordered bottom up, meaning earlist trades are the last row, newest trade is first row
# cross logs are opposite, so they need to be reordered
def Check_Edited_Cross_Times():
    try:
        print("Checking and fixing time order in edited cross logs...")
        files_processed = 0
        files_reordered = 0
        
        # Loop through all CSV files in the edited_cross_logs_dir
        for filename in os.listdir(edited_cross_logs_dir):
            if filename.endswith('.csv'):
                file_path = os.path.join(edited_cross_logs_dir, filename)
                print(f"Processing: {filename}")
                
                # Read the entire file to preserve structure
                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                
                # Find where data starts (line 7, index 6) and where it ends
                data_start_idx = 6  # Line 7 (0-indexed)
                data_end_idx = len(lines)
                
                # Find the end of data (first empty line after data starts)
                for i in range(data_start_idx + 1, len(lines)):
                    if lines[i].strip() == '':
                        data_end_idx = i
                        break
                
                # Extract data lines (excluding header at line 7)
                if data_end_idx > data_start_idx + 1:
                    header_line = lines[data_start_idx]
                    data_lines = lines[data_start_idx + 1:data_end_idx]
                    
                    # Create list of tuples (time_obj, original_line) for sorting
                    data_with_time = []
                    for line in data_lines:
                        if line.strip() and line.startswith(','):
                            # Parse the Exec Time (second column after first comma)
                            parts = line.split(',', 2)
                            if len(parts) >= 2:
                                exec_time_str = parts[1].strip()
                                try:
                                    # Parse datetime: MM/DD/YYYY HH:MM:SS
                                    exec_datetime = datetime.datetime.strptime(exec_time_str, '%m/%d/%Y %H:%M:%S')
                                    # We only care about time for sorting (ignore date)
                                    time_only = exec_datetime.time()
                                    data_with_time.append((time_only, line))
                                except ValueError:
                                    print(f"Warning: Could not parse time '{exec_time_str}' in {filename}")
                                    data_with_time.append((datetime.time.min, line))
                    
                    # Sort by time in descending order (latest time first)
                    data_with_time.sort(key=lambda x: x[0], reverse=True)
                    
                    # Check if order changed
                    original_order = [line for _, line in [(None, line) for line in data_lines]]
                    new_order = [line for _, line in data_with_time]
                    
                    if original_order != new_order:
                        print(f"  - Reordering data in {filename}")
                        
                        # Reconstruct the file with sorted data
                        new_lines = (lines[:data_start_idx + 1] +  # Everything before data (including header)
                                   new_order +                     # Sorted data
                                   lines[data_end_idx:])          # Everything after data
                        
                        # Write back to file
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.writelines(new_lines)
                        
                        files_reordered += 1
                    else:
                        print(f"  - {filename} already in correct order")
                
                files_processed += 1
        
        print(f"\nCompleted: {files_processed} files processed, {files_reordered} files reordered")
        return True
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
        return False


''' 1) deletes everything in 3_final_trade_csvs
    2) gets all valid dates used in both market data and my target trade logs
    3) calls Create_Summary_Csv() to make a summary csv for each date
    4) combines all csv's in 3_final_trade_csv into a bulk csv'''
def Bulk_Create_Summary_Csvs():
    try:
        mode = Get_User_Input()
        print('\n')
        if (mode == 2 and Check_Edited_Cross_Times() == False):
            return
        
        Create_Directories_If_Not_Exist()
        
        # 1) delete everyting in the final trade summary dir
        for filename in os.listdir(summarized_final_trades_dir):
            os.remove(f"{summarized_final_trades_dir}/{filename}")
            print(f"Deleted: {filename}")
        print("\n")

        # 2) Create dates list by going over all files in market_data
        market_data_dates = []
        market_data_file_names = []
        for file in os.listdir(market_data_logs_dir):
            if file.endswith('.csv'):
                # get the date from the file
                date = (file.split('_'))[3]  # month, day, year
                if ".csv" in date:
                    date = date[:-4]
                if date not in market_data_dates:
                    market_data_dates.append(date)
                    market_data_file_names.append(file)
        
        # now compare to the dates in target
        valid_dates = []
        target_file_names = []
        for file in os.listdir(target_dir):
            if file.endswith('.csv'):
                if (mode == 1):
                    parts = (file.split('_'))[0].split('-') # year, month, day
                    date = f"{parts[1]}-{parts[2]}-{parts[0]}"
                elif (mode == 2):
                    date = (file.split('_'))[3] # month, day, year
                    
                if date in market_data_dates:
                    valid_dates.append(date)
                    target_file_names.append(file)
    
        # 3) create the data
        for i in range(0, len(valid_dates)):
            Create_Summary_Csv(valid_dates[i], market_data_file_names[i], target_file_names[i], mode)

        # 4) Combine all individual CSV files into one combined file
        print("\nCombining all individual CSV files into Bulk_Combined.csv...")
        combined_df = pd.DataFrame()
        
        for date in valid_dates:
            csv_file_path = f"{summarized_final_trades_dir}/Summary_{date}.csv"
            if os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                print(f"Added {csv_file_path.split('/')[2]} to combined file")
            else:
                print(f"SKIPPING: {csv_file_path.split('/')[2]} not found")
        
        # Save the combined file
        combined_file_path = f"{summarized_final_trades_dir}/Bulk_Combined.csv"
        combined_df.to_csv(combined_file_path, index=False)
        print(f"\nCombined file saved to: {combined_file_path}")
        print(f"Total rows in combined file: {len(combined_df)}")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)



#Create_Summary_Csv(date="2025-04-09")
Bulk_Create_Summary_Csvs()

