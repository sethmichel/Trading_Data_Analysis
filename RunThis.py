import Parser
import pandas as pd
import os

# 1) put raw tos csv's in 1_tos_Raw_Trades-TODO
# 2) put market data in 2_Raw_Market_Data
# 3) Running this makes new csv's in 3_Final_Trade_Csvs

tosRawCsvDir_todo = 'Csv_Files/1_tos_Raw_Trades-TODO'
tosRawCsvDir_done = 'Csv_Files/1_tos_Raw_Trades-DONE'
marketDataDir = 'Csv_Files/2_Raw_Market_Data'
summarizedTradesDir = 'Csv_Files/3_Final_Trade_Csvs'

headers = ["Date", "Amount", "Entry Time", "Exit Time", "Ticker", "Entry", "Exit", "Shares", "$ Change", "% Change", 
           "Buy/Short", "Technical Type", "High", "Low"]


def Get_File_Names(month, day, year):
    raw_trades_name = f"{tosRawCsvDir_todo}/{year}-{month}-{day}-TradeActivity.csv"
    raw_market_data_name = f"{marketDataDir}/Raw_Market_Data_{month}-{day}-{year}.csv"
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


def CreateSummaryCsv():
        # prep work
                # 1: put raw trades in 1_tos_Raw_Trades-TODO
                # 2: put market data in 2_Raw_Market_Data
                # when it's done it'll move the tos raw trades csv into 1_tos_Raw_Trades-DONE
                # when it's done it'll make a new csv in 3_Final_Trade_Csvs

        # 1) enter date in month-day-year format
        date = "06-24-2025"
        month, day, year = date.split('-')

        # use a function bec some files have 'On_Demand' in them (I want to keep that special title)
        raw_trades_name, raw_market_data_name = Get_File_Names(month, day, year)
        trade_summary_name = f"{summarizedTradesDir}/Summary-{date}.csv"

        # 2) normalize raw trades so it's readable and each trade is 1 line.
        #   -- uses raw trades in 1tosRawCsvs-TODO
        print(f"1) Normalizing raw trades")
        print(f"file names -> \nraw trades: {raw_trades_name}, \nraw market data {raw_market_data_name}")

        raw_trade_df = Parser.CreateDf(raw_trades_name)            # creates the df of unreadable raw data
        normalized_df = Parser.Normalize_Raw_Trades(raw_trade_df)  # makes the data readable (not adding market data yet)
        
        # 3) add market data to the trade summary
        final_df = Parser.Add_Market_Data(normalized_df, raw_market_data_name)
        # TESTING - Write first 5 rows and header to testing.txt (too many rows to see in the terminal)
        with open('testing.txt', 'w') as f:
            f.write(final_df.head(20).to_string())

        # 4) save to new csv file



CreateSummaryCsv()