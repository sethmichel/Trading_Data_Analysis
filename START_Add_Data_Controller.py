import pandas as pd
import os
import inspect
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import Main_Globals
import Validate_Market_Data as VMD
import Validate_Trade_Logs as VTL
import Bulk_Df_Creator as BDC

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

'''
Scenario: user adds new market data and/or trade data

If we add new market data
1) validate market data: check the txt file for any new market data files, call market data checker file on them
      -if basic errors: fix with market data editor
2) record the date and file name of each validated/erroneous market file in the txt file
3) check trade txt file for any new trade logs
-all files should be moved to their respective folders depending on their validation/cleaning results

if the txt files say we have trade logs and market data for new dates
3) merge market data with trade logs to make a summary file for that date
4) update bulk_df (combined summary file for all days)
5) run misc metrics on the new data via the metrics folder
6) optional: retrain stastics models and record their updated metrics - compare to prior version

optional for new data: run grid search to check parameter system still holds up
optional for new data: run xg boost tree to analyze feature importance
'''

# if the file isn't in the market data df tracker, or we're rechecking previosuly failed files
def Find_Market_Data_Files_To_Be_Checked(market_data_state_df, all_filenames_in_dir, erroneous_market_data_dir, recheck_failed_files):
    filenames_to_validate = []

    # determine which files need to be validated
    raw_tracked_filenames = set(market_data_state_df['filename'].dropna())
    for filename in all_filenames_in_dir:
        if (filename not in raw_tracked_filenames):
            filenames_to_validate.append(filename)
        
    if (recheck_failed_files == 'yes'):
        # we're rechecking everything in the failed validation folder
        filenames_to_validate += os.listdir(erroneous_market_data_dir)

    return filenames_to_validate


# moves both market data and manual trade logs
# filenames_to_validate has changed filenames, files with errors, and on demand (error) files
def Move_Files(filenames_to_validate, files_with_errors, on_demand_files, unvalidated_files_dir, 
                erroneous_files_dir, approved_cleaned_files_dir):
    # move each file based on whether it has errors
    for filename in filenames_to_validate:
        src_path = os.path.join(unvalidated_files_dir, filename)

        if filename in files_with_errors or filename in on_demand_files:
            dest_dir = erroneous_files_dir
        else:
            dest_dir = approved_cleaned_files_dir

        dest_path = f"{dest_dir}/{filename}"

        if os.path.exists(dest_path):
            raise FileExistsError(f"Duplicate file at destination: {dest_path}")

        shutil.move(src_path, dest_path)


# checks if all required files and directories exist. Creates them if they don't.
# csv files are created with headers: filename, filepath, status, error info, date checked
def Check_Files_And_Directories_Exist(*args):
    csv_columns = ['filename', 'filepath', 'status', 'error info', 'date checked']
    targets = list(args)
    
    # Create directories/files if they don't exist
    for target in targets:
        if target.endswith('.csv'):
            if not os.path.exists(target):
                # create empty CSV with headers
                pd.DataFrame(columns=csv_columns).to_csv(target, index=False)
        else:
            if not os.path.exists(target):
                os.makedirs(target, exist_ok=True)    


def Market_Data_Controller(recheck_failed_files):
    unvalidated_market_data_dir = "Data_Files/Market_Data/Unvalidated_Raw_Market_Data"
    erroneous_market_data_dir = "Data_Files/Market_Data/Erroneous_Market_Data"
    approved_cleaned_market_data_dir = "Data_Files/Market_Data/Approved_Cleaned_Market_Data"
    market_data_csv_state_manager_path = "Data_Files/Market_Data/Market_Data_Tracking.csv"

    Check_Files_And_Directories_Exist(unvalidated_market_data_dir, erroneous_market_data_dir, approved_cleaned_market_data_dir,
                                      market_data_csv_state_manager_path)
    
    # 1) check the txt file for any new market data files, call market data checker file on them
    # columns: filename, filepath, status, error info, date checked
    market_data_state_df = pd.read_csv(market_data_csv_state_manager_path)

    # collect all market data filenames from the unvalidated market data directory
    all_market_filenames_in_dir = os.listdir(unvalidated_market_data_dir)
    market_filenames_to_validate = Find_Market_Data_Files_To_Be_Checked(market_data_state_df, all_market_filenames_in_dir, 
                                                                        erroneous_market_data_dir, recheck_failed_files)
    if (market_filenames_to_validate == []):
        return

    # run authenticator/validator using only files not already tracked
    # market_filenames_to_validate can change if we changed a filename format
    market_data_state_df, filenames_to_validate, files_with_errors, on_demand_files = VMD.Authenticator_Freeway(
                            unvalidated_market_data_dir, market_filenames_to_validate, market_data_state_df)

    # move market files to reflect their validation status
    Move_Files(filenames_to_validate, files_with_errors, on_demand_files, unvalidated_market_data_dir, 
                erroneous_market_data_dir, approved_cleaned_market_data_dir)

    # 2) record the date and file name of each validated/erroneous market file in the csv file
    market_data_state_df.to_csv(market_data_csv_state_manager_path, index=False)
                                    

def Manual_Trade_Controller(recheck_failed_files):
    approved_cleaned_manual_trade_logs_dir = "Data_Files/Manual_Trade_Logs/Approved_Cleaned_Manual_Trade_Data"
    unvalidated_manual_trade_logs_dir = "Data_Files/Manual_Trade_Logs/Unvalidated_Manual_Trade_Logs"
    erroneous_manual_trade_logs_dir = "Data_Files/Manual_Trade_Logs/Erroneous_Manual_Trade_Logs"
    manual_trade_csv_state_manager_path = "Data_Files/Manual_Trade_Logs/Manual_Trade_Log_Status.csv"

    # check folders/files exist or create them
    Check_Files_And_Directories_Exist(approved_cleaned_manual_trade_logs_dir, unvalidated_manual_trade_logs_dir, 
                                      erroneous_manual_trade_logs_dir, manual_trade_csv_state_manager_path)

    # 3) check trade csv file for any new trade logs
    # columns: filename, filepath, status, error info, date checked
    manual_trade_state_df = pd.read_csv(manual_trade_csv_state_manager_path)

    all_trade_filenames_in_dir = os.listdir(unvalidated_manual_trade_logs_dir)
    trade_filenames_to_validate = Find_Market_Data_Files_To_Be_Checked(manual_trade_state_df, all_trade_filenames_in_dir,
                                                                       erroneous_manual_trade_logs_dir, recheck_failed_files)
    if (trade_filenames_to_validate == []):
        return

    # validate the trade logs
    manual_trade_state_df, filenames_to_validate, trade_files_with_errors = VTL.Authenticator_Freeway(
                       unvalidated_manual_trade_logs_dir, trade_filenames_to_validate, manual_trade_state_df)

    # move trade files to reflect their validation status
    Move_Files(filenames_to_validate, trade_files_with_errors, [], unvalidated_manual_trade_logs_dir, 
                erroneous_manual_trade_logs_dir, approved_cleaned_manual_trade_logs_dir)

    # record the date and file name of each validated/erroneous market file in the csv file
    manual_trade_state_df.to_csv(manual_trade_csv_state_manager_path, index=False)


def Main():
    trade_summaries_csv_dir = "Data_Files/Trade_Summaries/Summary_Csv_Files"
    trade_summaries_txt_dir = "Data_Files/Trade_Summaries/Summary_Text_Files"
    
    # check folders/files exist or create them
    Check_Files_And_Directories_Exist(trade_summaries_csv_dir, trade_summaries_txt_dir)

    # Ask user if they want to recheck files that previously failed validation
    while True:
        recheck_failed_files = input("Do you want to recheck files that previously failed validation? (yes/no): ").strip().lower()
        if recheck_failed_files == 'yes' or recheck_failed_files == 'no':
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    Market_Data_Controller(recheck_failed_files)

    Manual_Trade_Controller(recheck_failed_files)

    # create summaries and save bulk df
    BDC.Controller(do_all_trade_logs='yes')


Main()