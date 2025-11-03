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
import re
from io import StringIO

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


def Append_to_Df(manual_trade_state_df, filename, filepath, status, error_info=''):
    date = datetime.now().strftime('%Y-%m-%d')

    # if the file is already in here, it means it failed more than 2 check. just update the error info in this case
    files = manual_trade_state_df['filename'].dropna()
    if (filename in files):
        manual_trade_state_df.at[files.index(filename), 'error_info'] += f", {error_info}"
        manual_trade_state_df.at[files.index(filename), 'date checked'] = date
        
    else:
        manual_trade_state_df.loc[len(manual_trade_state_df)] = {
            'filename': filename,
            'filepath': filepath,
            'status': status,
            'error_info': error_info,
            'date checked': date
        }

    return manual_trade_state_df


def Delete_From_DF(manual_trade_state_df, old_filename):
    files = manual_trade_state_df['filename'].dropna()
    if (old_filename in files):
        row_index = files.index[files == old_filename][0]
        manual_trade_state_df = manual_trade_state_df.drop(row_index)
    
    return manual_trade_state_df 


# expected format: "MM-DD-YYYY-TradeActivity.csv" or "MM-DD-YYYY-TradeActivity_On_Demand.csv"
# return (status, filename) where status is True/'changed'/False
def Check_File_Name(filename, unvalidated_manual_trade_logs_dir):
    if not filename.endswith('.csv'):
        return False, filename
    
    pattern = r'^(\d{2})-(\d{2})-(\d{4})-TradeActivity(_On_Demand)?\.csv$'
    match = re.match(pattern, filename)
    
    if match:
        # Validate date components
        month, day, year = match.group(1), match.group(2), match.group(3)
        
        try:
            # Check if date is valid
            datetime.strptime(f"{month}-{day}-{year}", "%m-%d-%Y")
            return True, filename
        except ValueError:
            # Invalid date (e.g., 13-32-2025)
            return False, filename
    
    # Try to fix common issues
    corrected_filename = filename
    
    # Remove .csv for processing
    name_without_ext = filename[:-4]
    
    # Try different correction patterns
    
    # Pattern 1: Check if it's missing the "-TradeActivity" part
    # e.g., "09-22-2025.csv" -> "09-22-2025-TradeActivity.csv"
    date_pattern = r'^(\d{2})-(\d{2})-(\d{4})$'
    if re.match(date_pattern, name_without_ext):
        corrected_filename = f"{name_without_ext}-TradeActivity.csv"
        
    # Pattern 2: Check if date format is wrong (e.g., single digit month/day)
    # e.g., "9-22-2025-TradeActivity.csv" -> "09-22-2025-TradeActivity.csv"
    elif re.match(r'^(\d{1,2})-(\d{1,2})-(\d{4})-TradeActivity(_On_Demand)?$', name_without_ext):
        parts = name_without_ext.split('-')
        month = parts[0].zfill(2)
        day = parts[1].zfill(2)
        year = parts[2]
        
        # Check if there's "_On_Demand" in the rest
        rest = '-'.join(parts[3:])
        if '_On_Demand' in rest:
            corrected_filename = f"{month}-{day}-{year}-TradeActivity_On_Demand.csv"
        else:
            corrected_filename = f"{month}-{day}-{year}-TradeActivity.csv"
    
    # Pattern 3: Check if format is YYYY-MM-DD instead of MM-DD-YYYY
    # e.g., "2025-09-22-TradeActivity.csv" -> "09-22-2025-TradeActivity.csv"
    elif re.match(r'^(\d{4})-(\d{2})-(\d{2})-TradeActivity(_On_Demand)?$', name_without_ext):
        parts = name_without_ext.split('-')
        year = parts[0]
        month = parts[1]
        day = parts[2]
        rest = '-'.join(parts[3:])
        
        if '_On_Demand' in rest:
            corrected_filename = f"{month}-{day}-{year}-TradeActivity_On_Demand.csv"
        else:
            corrected_filename = f"{month}-{day}-{year}-TradeActivity.csv"
    
    # If we made a correction, validate it
    if corrected_filename != filename:
        # Validate the corrected filename
        match = re.match(pattern, corrected_filename)
        if match:
            month, day, year = match.group(1), match.group(2), match.group(3)
            try:
                datetime.strptime(f"{month}-{day}-{year}", "%m-%d-%Y")
                
                # Rename the actual file
                old_path = os.path.join(unvalidated_manual_trade_logs_dir, filename)
                new_path = os.path.join(unvalidated_manual_trade_logs_dir, corrected_filename)
                
                if os.path.exists(old_path):
                    # Read the data to ensure no data loss
                    try:
                        df = pd.read_csv(old_path)
                        # Save with new filename
                        df.to_csv(new_path, index=False)
                        # Verify new file exists and has same data
                        df_new = pd.read_csv(new_path)
                        if len(df) == len(df_new) and df.shape == df_new.shape:
                            # Delete old file
                            os.remove(old_path)
                            return 'changed', corrected_filename
                        else:
                            # Data mismatch, don't delete old file
                            if os.path.exists(new_path):
                                os.remove(new_path)
                            return False, filename
                    except Exception as e:
                        print(f"Error renaming file {filename}: {e}")
                        return False, filename
                else:
                    return False, filename
                    
            except ValueError:
                return False, filename
    
    # Could not correct the filename
    return False, filename


def Check_File_Content_Format(filename, path, manual_trade_state_df):
    ''' this is the format they're in straight from tos. note: trades are sorted by time latest to earliest
    Today's Trade Activity for  (Virtual Account) on 9/22/25 07:34:24

    Working Orders
    ,Time Placed,Spread,Side,Qty,Pos Effect,Symbol,Exp,Strike,Type,PRICE,,TIF,Mark,Status

    Filled Orders
    ,Exec Time,Spread,Side,Qty,Pos Effect,Symbol,Exp,Strike,Type,Price,Net Price,Price Improvement,Order Type
    ,9/22/25 07:34:23,STOCK,SELL,-4,AUTO,HOOD,,,STOCK,123.70,123.70,.00,STP
    ...
    .
    .
    .
    [skip line]
    Canceled Orders
    [random stuff we don't care about]
    '''
    # we want to delete everything before the column titles and everything after the last trade line. so basically make it into a formatted csv file
    
    # Read the file
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # find the header line (starts with ,Exec Time or ,,Exec Time)
    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(',Exec Time') or stripped.startswith(',,Exec Time') or stripped.startswith('Exec Time'):
            header_idx = i
            break
    
    if header_idx is None:
        message = f"Could not find 'Exec Time' header in {filename}"
        print(message)
        return False, message
    
    # Find the end of trade data (empty line or line starting with "Canceled Orders")
    trade_end_idx = None
    for i in range(header_idx + 1, len(lines)):
        stripped = lines[i].strip()
        # Empty line or "Canceled Orders" marks the end
        if stripped == '' or stripped.startswith('Canceled Orders'):
            trade_end_idx = i
            break
    
    if trade_end_idx is None:
        # If no end marker found, it means it's just a trade table. nothing after it
        trade_end_idx = len(lines)
    
    # Extract header and trade lines
    header_line = lines[header_idx].strip()
    trade_lines = [lines[i].strip() for i in range(header_idx + 1, trade_end_idx) if lines[i].strip() != '']
    
    # Remove leading commas (one or two) from header
    if header_line.startswith(',,'):
        header_line = header_line[2:]
    elif header_line.startswith(','):
        header_line = header_line[1:]
    
    # Remove leading commas from trade lines
    cleaned_trade_lines = []
    for line in trade_lines:
        if line.startswith(',,'):
            cleaned_trade_lines.append(line[2:])
        elif line.startswith(','):
            cleaned_trade_lines.append(line[1:])
        else:
            cleaned_trade_lines.append(line)
    
    # Parse the trades into a dataframe to check sorting
    csv_content = header_line + '\n' + '\n'.join(cleaned_trade_lines)
    df = pd.read_csv(StringIO(csv_content))
    
    # Check if 'Exec Time' column exists
    expected_columns = ['Exec Time', 'Spread', 'Side', 'Qty', 'Pos Effect', 'Symbol', 'Exp', 'Strike', 'Type', 'Price', 'Net Price', 'Price Improvement', 'Order Type']
    if list(df.columns) != expected_columns:
        message = f"Wrong columns in {filename}. columes: {df.columns}"
        print(message)
        return False, message
    
    # Parse the datetime and sort by time (newest to oldest)
    df['parsed_time'] = pd.to_datetime(df['Exec Time'], format='%m/%d/%y %H:%M:%S')
    
    # Check if already sorted (newest to oldest. descending)
    if not df['parsed_time'].equals(df['parsed_time'].sort_values(ascending=False).reset_index(drop=True)):
        # Sort if not already sorted
        df = df.sort_values('parsed_time', ascending=False).reset_index(drop=True)
    
    # Drop the helper column
    df = df.drop(columns=['parsed_time'])
    
    # Write the cleaned and sorted data back to the file
    df.to_csv(path, index=False)
    
    return True, None


def Authenticator_Freeway(unvalidated_manual_trade_logs_dir, filenames_to_validate, manual_trade_state_df):
    try:
        files_with_errors = []

        if not filenames_to_validate:
            print("No new manual trade log files to validate.")
            return True, manual_trade_state_df
        
        # 1) make sure the file name is correct and date is correct format
        for filename in filenames_to_validate:
            path = f"{unvalidated_manual_trade_logs_dir}/{filename}"

            result, new_filename = Check_File_Name(filename, unvalidated_manual_trade_logs_dir)
            if (isinstance(result, bool) and result == False):
                files_with_errors.append(filename)
                manual_trade_state_df = Append_to_Df(manual_trade_state_df, filename, path, 'failed validation', 'bad filename')
            
            elif (isinstance(result, str) and result == 'changed'):
                # put the fixed filename everywhere
                original_filename = filename
                filenames_to_validate[filenames_to_validate.index(filename)] = new_filename
                new_path = f"{unvalidated_manual_trade_logs_dir}/{new_filename}"
                # delete the entry for the OLD filename if it's there. it'll get added correctly later
                manual_trade_state_df = Delete_From_DF(manual_trade_state_df, original_filename)
                # rename the actual file
                os.rename(path, new_path)

        # 2) redo file format
        for filename in filenames_to_validate:
            path = f"{unvalidated_manual_trade_logs_dir}/{filename}"
            result, message = Check_File_Content_Format(path)
            if result == False:
                manual_trade_state_df = Append_to_Df(manual_trade_state_df, filename, path, 'failed validation', error_info=message)
                files_with_errors.append(filename)
   
        # now update the valid files status
        for filename in filenames_to_validate:
            if (filename not in files_with_errors):
                path = f"{unvalidated_manual_trade_logs_dir}/{filename}"
                manual_trade_state_df = Append_to_Df(manual_trade_state_df, filename, path, 'validated-cleaned')

        print("\nFiles have been checked. invalid files: \n")
        for filename in files_with_errors:
            print(f"    {filename}")

        return manual_trade_state_df, files_with_errors, filenames_to_validate

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)
    