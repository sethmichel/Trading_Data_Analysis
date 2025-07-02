import Parser
import pandas as pd
import os

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
    raw_trades_name = f"{tos_raw_trades_TODO_dir}/{year}-{month}-{day}-TradeActivity.csv"
    raw_market_data_name = f"{market_data_TODO_dir}/Raw_Market_Data_{month}-{day}-{year}.csv"
    file_names = []

    # Check if both files exist
    if (os.path.exists(raw_trades_name) == False):
        raw_trades_name_on_demand = raw_trades_name.replace('.csv', '_On_Demand.csv')
        if (os.path.exists(raw_trades_name_on_demand) == False):
                raise FileNotFoundError(f"Could not find file {raw_trades_name} or {raw_trades_name_on_demand}")
        else:
              file_names.append(raw_trades_name_on_demand)
    else:
          file_names.append(raw_trades_name)

    if (os.path.exists(raw_market_data_name) == False):
        raw_market_data_name_on_demand = raw_market_data_name.replace('.csv', '_On_Demand.csv')
        if (os.path.exists(raw_market_data_name_on_demand) == False):
                raise FileNotFoundError(f"Could not find file {raw_market_data_name} or {raw_market_data_name_on_demand}")
        else:  
              file_names.append(raw_market_data_name_on_demand)
    else:
          file_names.append(raw_market_data_name)

    return file_names[0], file_names[1]


def Create_Summary_Csv(date):
        # prep work
                # 1: put raw trades in 1_tos_Raw_Trades-TODO
                # 2: put market data in 2_Raw_Market_Data
                # when it's done it'll move the tos raw trades csv into 1_tos_Raw_Trades-DONE
                # when it's done it'll make a new csv in 3_Final_Trade_Csvs

        # 1) enter date in month-day-year format
        month, day, year = date.split('-')

        # use a function bec some files have 'On_Demand' in them (I want to keep that special title)
        raw_trades_name, raw_market_data_name = Get_File_Names(month, day, year)
        trade_summary_name = f"{summarized_final_trades_dir}/Summary_{date}.csv"

        # 2) normalize raw trades so it's readable and each trade is 1 line.
        #   -- uses raw trades in 1tosRawCsvs-TODO
        print(f"1) Normalizing raw trades")
        print(f"file names -> \nraw trades: {raw_trades_name}, \nraw market data {raw_market_data_name}")

        raw_trade_df = Parser.CreateDf(raw_trades_name)            # creates the df of unreadable raw data
        normalized_df = Parser.Normalize_Raw_Trades(raw_trade_df)  # makes the data readable (not adding market data yet)
        
        # 3) add market data to the trade summary
        final_df = Parser.Add_Market_Data(normalized_df, raw_market_data_name)

        # 4) save to new csv file
        final_df.to_csv(trade_summary_name, index=False)
        print(f"Trade summary saved to: {trade_summary_name}")

        # 5) move the used market data/trade log to used folders
        Parser.Move_Processed_Files(raw_trades_name, raw_market_data_name, tos_raw_trades_DONE_dir, market_data_DONE_dir)


# just calls the normal create summary csv function in a loop
def Bulk_Create_Summary_Csvs(dates):
        for date in dates:
              Create_Summary_Csv(date)
        
        # Combine all individual CSV files into one combined file
        print("Combining all individual CSV files into Bulk_Combined.csv...")
        combined_df = pd.DataFrame()
        
        for date in dates:
            csv_file_path = f"{summarized_final_trades_dir}/Summary_{date}.csv"
            if os.path.exists(csv_file_path):
                df = pd.read_csv(csv_file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
                print(f"Added {csv_file_path} to combined file")
            else:
                print(f"Warning: {csv_file_path} not found, skipping...")
        
        # Save the combined file
        combined_file_path = f"{summarized_final_trades_dir}/Bulk_Combined.csv"
        combined_df.to_csv(combined_file_path, index=False)
        print(f"Combined file saved to: {combined_file_path}")
        print(f"Total rows in combined file: {len(combined_df)}")


Create_Summary_Csv(date="06-11-2025")

'''Bulk_Create_Summary_Csvs(dates=['04-09-2025', '04-10-2025', '05-01-2025', '05-09-2025', 
                                '06-17-2025', '06-24-2025', 
                                '06-30-2025'])
'''
