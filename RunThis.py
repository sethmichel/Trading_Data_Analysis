import Parser
import pandas as pd
import os
import sys
import inspect
import Main_Globals

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

# 1) put raw tos csv's in 1_tos_Raw_Trades-TODO
# 2) put market data in 2_Raw_Market_Data
# 3) Running this makes new csv's in 3_Final_Trade_Csvs

tos_raw_trades_TODO_dir = 'Csv_Files/1_tos_Raw_Trades/TODO_Trade_Data'
tos_raw_trades_DONE_dir = 'Csv_Files/1_tos_Raw_Trades/Used_Trade_Data'
market_data_TODO_dir = 'Csv_Files/2_Raw_Market_Data/TODO_Market_Data'
market_data_DONE_dir = 'Csv_Files/2_Raw_Market_Data/Used_Market_Data'
summarized_final_trades_dir = 'Csv_Files/3_Final_Trade_Csvs'

headers = ["Date", "Amount", "Entry Time", "Exit Time", "Ticker", "Entry", "Exit", "Shares", "$ Change", "% Change", 
           "Buy/Short", "Technical Type", "High", "Low"]


def Get_File_Names(month, day, year):
    try:
        raw_trades_name = f"{tos_raw_trades_TODO_dir}/{year}-{month}-{day}-TradeActivity.csv"
        raw_market_data_name = f"{market_data_TODO_dir}/Raw_Market_Data_{month}-{day}-{year}.csv"
        file_names = []

        # Check if both files exist
        if (os.path.exists(raw_trades_name) == False):
            raw_trades_name_on_demand = raw_trades_name.replace('.csv', '_On_Demand.csv')
            if (os.path.exists(raw_trades_name_on_demand) == False):
                print(f"--SKIPPING: Could not find file {raw_trades_name.split('/')[3]} or {raw_trades_name_on_demand.split('/')[3]}")
                return None, None
            else:
                file_names.append(raw_trades_name_on_demand)
        else:
            file_names.append(raw_trades_name)

        if (os.path.exists(raw_market_data_name) == False):
            raw_market_data_name_on_demand = raw_market_data_name.replace('.csv', '_On_Demand.csv')
            if (os.path.exists(raw_market_data_name_on_demand) == False):
                print(f"--SKIPPING: Could not find file {raw_market_data_name.split('/')[3]} or {raw_market_data_name_on_demand.split('/')[3]}")
                return None, None
            else:  
                file_names.append(raw_market_data_name_on_demand)
        else:
            file_names.append(raw_market_data_name)

        return file_names[0], file_names[1]

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Create_Summary_Csv(date):
    try:
        # prep work
        # 1: put raw trades in 1_tos_Raw_Trades-TODO
        # 2: put market data in 2_Raw_Market_Data
        # when it's done it'll move the tos raw trades csv into 1_tos_Raw_Trades-DONE
        # when it's done it'll make a new csv in 3_Final_Trade_Csvs

        # 1) enter date in month-day-year format
        year, month, day = date.split('-')

        # use a function bec some files have 'On_Demand' in them (I want to keep that special title)
        raw_trades_name, raw_market_data_name = Get_File_Names(month, day, year)
        if (raw_trades_name == None or raw_market_data_name == None):
            return
        
        trade_summary_name = f"{summarized_final_trades_dir}/Summary_{date}.csv"

        # 2) normalize raw trades so it's readable and each trade is 1 line.
        print(f"--{raw_trades_name.split('/')[3]} & {raw_market_data_name.split('/')[3]}")

        raw_trade_df = Parser.CreateDf(raw_trades_name)            # creates the df of unreadable raw data
        normalized_df = Parser.Normalize_Raw_Trades(raw_trade_df)  # makes the data readable (not adding market data yet)
        normalized_df = Parser.Add_Running_Sums(normalized_df)

        # 3) add market data to the trade summary
        final_df = Parser.Add_Market_Data(normalized_df, raw_market_data_name)

        # 4) save to new csv file
        final_df.to_csv(trade_summary_name, index=False)
        print(f"Trade summary saved to: {trade_summary_name}")

        # 5) move the used market data/trade log to used folders
        #Parser.Move_Processed_Files(raw_trades_name, raw_market_data_name, tos_raw_trades_DONE_dir, market_data_DONE_dir)
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


''' 1) deletes everything in 3_final_trade_csvs
    2) gets all dates from todo_trade_data
    3) calls Create_Summary_Csv() to make a summary csv for each date
    4) combines all csv's in 3_final_trade_csv into a bulk csv
'''
def Bulk_Create_Summary_Csvs():
    try:
        dates_source = 'Csv_Files/1_tos_Raw_Trades/TODO_Trade_Data'
        
        # delete everyting in the final trade summary dir
        deletion_dir = 'Csv_Files/3_Final_Trade_Csvs'
        if os.path.exists(deletion_dir):
            for filename in os.listdir(deletion_dir):
                file_path = f"{deletion_dir}/{filename}"
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path.split('/')[2]}")
        print("\n")

        # Create dates list by going over all files in dates_source
        dates = []
        if os.path.exists(dates_source):
            files = os.listdir(dates_source)
            for file in files:
                if file.endswith('.csv'):
                    # Split filename by '-' and combine indexes 0-2 to get the date
                    parts = file.split('-')
                    if len(parts) >= 3:
                        date = f"{parts[0]}-{parts[1]}-{parts[2]}"
                        if date not in dates:  # Avoid duplicates
                            dates.append(date)
        
        valid_dates = []
        for date in dates:
            try:
                Create_Summary_Csv(date)
                valid_dates.append(date)
            except Exception as e: 
                print(f"error for date: {date}. this process uses each date in tos raw todo trades. it's doing it in bulk so I probably don't have market data for this day. error: {e}")
        
        # Combine all individual CSV files into one combined file
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

