import pandas as pd
from pygam import LinearGAM, te
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import Helper_Functions
import json
import warnings


# WARNING: this turns all warnings into errors. it's so I can get the line number and more info
warnings.simplefilter('error', category=RuntimeWarning) # Turn all RuntimeWarnings into exceptions
np.seterr(divide='raise', invalid='raise')              # Have NumPy raise on divide-by-zero / invalid operations

'''
only meant to find metrics for different holding values
'''
def Basic_Grid_Search(bulk_df, market_data_dict_by_ticker, roi_dictionary, holding_value):
    unique_dates = bulk_df['Date'].unique()
    unique_tickers = bulk_df['Ticker'].unique()
    skip_dates = []
    overall_metrics = {}
    ticker_metrics = {}
    
    # 1) populate dictionaries
    for i in range(0, len(unique_dates)):
        unique_dates[i] = Helper_Functions.bulk_csv_date_converter(unique_dates[i])

    for ticker in unique_tickers:
        ticker_metrics[ticker] = {}
        for date in unique_dates:
            if (date not in overall_metrics):
                overall_metrics[date] = {'numb of holdings reached': 0, 'total roi': 0}
            ticker_metrics[ticker][date] = {'numb of holdings reached': 0, 'total roi': 0}

    # 2) grid search
    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        entry_time = row['Entry Time']
        trade_id = str(row['Trade Id'])
        date = Helper_Functions.bulk_csv_date_converter(row['Date'])  # 08-09-2025
        market_df = market_data_dict_by_ticker[date][ticker]  # market data df for this ticker and date
        
        if (date in skip_dates):
            continue
        # Check if we have market data for this date and ticker
        if date not in market_data_dict_by_ticker:
            print(f"No market data found for date {date}")
            skip_dates.append(date)
            continue    
        if ticker not in market_data_dict_by_ticker[date]:
            msg = f"No market data found for ticker {ticker} on date {date}"
            print(msg)
            raise ValueError(msg)
        # Skip trade if entry time is after final market data time
        if (Helper_Functions.Check_We_Have_Data_For_Trade(market_df, entry_time) == False):
            print(f"Skipping trade {trade_id}: entry time {entry_time} is after final market data time")
            continue

        roi_list = roi_dictionary[trade_id]
        
        # if it reached holding, then the final roi value will be either above holding value if we ran out of data
        #    or at the holding stop loss
        #    otherwise it failed
        final_roi = roi_list[-1]
        if (final_roi >= holding_value or (final_roi < 0.1 and final_roi > -0.1)): # that range should catch just holding stop losses and not starting stop losses
            overall_metrics[date]['numb of holdings reached'] += 1
            ticker_metrics[ticker][date]['numb of holdings reached'] += 1

        overall_metrics[date]['total roi'] += final_roi        
        ticker_metrics[ticker][date]['total roi'] += final_roi

    # grid search over - write results
    string_to_write = ''
    red_data = []
    green_data = []
    string_to_write += f"Data for all dates with holding value = {holding_value}\n"

    # get ticker info
    for ticker, date_data in ticker_metrics.items():
        total_holdings = 0
        total_roi = 0.0
     
        for date, metrics in date_data.items():
            total_holdings += metrics['numb of holdings reached']
            total_roi += metrics['total roi']

        string_to_write += f"{ticker} roi = {round(total_roi, 2)}\n"
        string_to_write += f"{ticker} holdings reached = {total_holdings}\n"

    # get overall & red/green info
    overall_sum = 0
    overall_holdings_reached = 0
    for date in unique_dates:
        date_sum = overall_metrics[date]['total roi']
        overall_sum += date_sum
        overall_holdings_reached += overall_metrics[date]['numb of holdings reached']

        # red/green info
        numb_of_days = len(unique_dates)
        if (date_sum < 0):
            red_data.append(date_sum)
        else: # including 0
            green_data.append(date_sum)

    if (len(red_data) + len(green_data) != numb_of_days):
        raise ValueError(f"red data and green data are both empty. this means there's a bug. holding value: {holding_value}, length of df (for vol%): {len(bulk_df)}")

    string_to_write += f"\nOverall total roi = {round(overall_sum, 2)}\n"
    string_to_write += f"Overall numb of holdings reached = {overall_holdings_reached}\n"

    # write metrics
    avg_roi_per_day = round(overall_sum / numb_of_days, 2)
    red_days_avg = 0
    green_days_avg = 0
    if (len(red_data) != 0):
        red_days_avg = round(np.average(red_data) / 6, 2)
    if (len(green_data) != 0):
        green_days_avg = round(np.average(green_data) / 6, 2)
    
    string_to_write += f"Days = {numb_of_days}\n"
    string_to_write += f"Overall avg / day = {avg_roi_per_day}\n"
    string_to_write += f"Divided by 6 = {round(avg_roi_per_day / 6, 2)}\n"
    string_to_write += f"Red days count: {len(red_data)}/{numb_of_days}\n"
    string_to_write += f"Avg green day / 6: {green_days_avg}\n"
    string_to_write += f"Avg red day / 6: {red_days_avg}\n"
    string_to_write +="-------------------------------------------\n\n"

    return string_to_write


def Write_Results(string_to_write, action):
    if (action == 'basic'):
        file_name = "Holding_Value_Basic_Grid_Search.txt"
    elif (action == 'volatility buckets'):
        file_name = "Holding_Value_Volatility_Buckets_Grid_Search.txt"
    else:
        raise ValueError(f"unclear action {action}")
    
    with open(f'Holder_Strat/Parameter_Tuning/model_files_and_data/{file_name}', 'w') as f:
        f.write(string_to_write)

        '''
        Totals Across All Dates:
        SOXL = 2.41
        SMCI = 0.48
        HOOD = 2.08
        MARA = 7.14
        IONQ = 4.48
        TSLA = 0.24
        RDDT = -1.92
        Overall Total = 14.92

        days = 10
        Overall avg / day = 1.49
        divided by 6 = 0.25
        red days: 2/10
        avg red day: -0.57
        avg green day: 2.01
        '''

def Run_Basic_Grid_Search():
    columns_to_keep = ["Date", "Trade Id", "Ticker", "Entry Time", "Entry Price", "Trade Type", "Entry Volatility Percent"]
    bulk_df = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[columns_to_keep]
    market_data_dict_by_ticker = Helper_Functions.Load_Market_Data_Dictionary(bulk_df) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    
    # we save each searches results as a string then write all results to a txt file at the end
    # we also need a custom roi dictionary for each holding_value
    grid_search_values = [0.5,0.6,0.7,0.8,0.9,1]
    string_to_write = f"Trades: {len(bulk_df)}\n"
    for holding_value in grid_search_values:
        roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Create_Roi_Dictionary_For_Trades(
                bulk_df, 
                market_data_dict_by_ticker, 
                largest_sl_value=-0.4, 
                holding_value=holding_value, 
                holding_sl_value=0
                )
    
        string_to_write += Basic_Grid_Search(bulk_df, market_data_dict_by_ticker, roi_dictionary, holding_value)
    
    Write_Results(string_to_write, action='basic')


def Run_Vol_Percent_Grid_Search():
    columns_to_keep = ["Date", "Trade Id", "Ticker", "Entry Time", "Entry Price", "Trade Type", "Entry Volatility Percent"]
    bulk_df = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[columns_to_keep]
    market_data_dict_by_ticker = Helper_Functions.Load_Market_Data_Dictionary(bulk_df) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}
    
    # we save each searches results as a string then write all results to a txt file at the end
    # we also need a custom roi dictionary for each holding_value
    grid_search_values = [0.4,0.5,0.6,0.7,0.8,0.9,1]
    vol_percent_ranges = [[0, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 5]] # inclusive, exclusive
    string_to_write = f"Trades: {len(bulk_df)}\n"
    for holding_value in grid_search_values:
        result = Helper_Functions.Load_Roi_Dictionary_And_Values(holding_value)
        if result != None:
            roi_dictionary, _, _ = result
        else:
            # probably happens if we're using a new holding_value. we don't have the roi dictionary saved
            roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Create_Roi_Dictionary_For_Trades(
                    bulk_df, 
                    market_data_dict_by_ticker, 
                    largest_sl_value=-0.4, 
                    holding_value=holding_value, 
                    holding_sl_value=0
                    )
                    
        # to do this, we can actually just give the basic grid search only the trades in each range (replace bulk_df)
        for vol_range in vol_percent_ranges:
            vol_df = bulk_df[(bulk_df['Entry Volatility Percent'] >= vol_range[0]) & (bulk_df['Entry Volatility Percent'] < vol_range[1])]
            vol_df = vol_df[vol_df['Entry Time'] >= '06:40:00']
            
            string_to_write += f"Volatility Range: {vol_range}. trades: {len(vol_df)}\n"
            string_to_write += Basic_Grid_Search(vol_df, market_data_dict_by_ticker, roi_dictionary, holding_value)
    
    Write_Results(string_to_write, action='volatility buckets')


if __name__ == "__main__":
    #Run_Basic_Grid_Search()
    Run_Vol_Percent_Grid_Search()



