import pandas as pd
import os
import inspect
import sys
import shutil
import Main_Globals
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))


# make a new csv of all 1 ticker
def Ticker_Csv():
    try:
        ticker = 'SOXL'
        file_path = 'Csv_Files/2_Raw_Market_Data/TODO_Market_Data/Raw_Market_Data_06-18-2025.csv'
        df = pd.read_csv(file_path)
        
        # Filter rows where the "Ticker" column matches the specified ticker
        filtered_df = df[df['Ticker'] == ticker]
        
        # Create output directory if it doesn't exist
        output_dir = 'Csv_Files/Testing_Csv_Data'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{output_dir}/{base_name}_{ticker}_filtered.csv"
        
        # Save the filtered data to a new CSV file
        filtered_df.to_csv(output_filename, index=False)
        
        print(f"Created filtered CSV: {output_filename}")
        print(f"Original rows: {len(df)}, Filtered rows: {len(filtered_df)}")
        
        return output_filename
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# organize a csv by a column - highest to lowest
def Column_Sort_Csv():
    try:
        column = 'Volatility'
        file_path = "Csv_Files\Testing_Csv_Data\Raw_Market_Data_06-18-2025_SOXL_filtered.csv"
        
        df = pd.read_csv(file_path)
        sorted_df = df.sort_values(by=column, ascending=False)
        
        # Create output directory if it doesn't exist
        output_dir = 'Csv_Files/Testing_Csv_Data'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{output_dir}/{base_name}_sorted_by_{column}.csv"
        
        # Save the sorted data to a new CSV file
        sorted_df.to_csv(output_filename, index=False)
        
        print(f"Created sorted CSV: {output_filename}")
        print(f"Sorted by {column} column (highest to lowest)")
        print(f"Total rows: {len(sorted_df)}")
        print(f"Highest {column} value: {sorted_df[column].max()}")
        print(f"Lowest {column} value: {sorted_df[column].min()}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Only_Keep_Some_Columns_Csv():
    try:
        columns_to_keep = ['Atr14', 'Volatility', 'Time']
        file_path = "Csv_Files\Testing_Csv_Data\Raw_Market_Data_06-18-2025_SOXL_filtered.csv"
        output_dir = 'Csv_Files/Testing_Csv_Data'
        
        df = pd.read_csv(file_path)
        filtered_df = df[columns_to_keep]
        
        # Create the output filename
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{output_dir}/{base_name}_columns_filtered.csv"
        
        # Save the filtered data to a new CSV file
        filtered_df.to_csv(output_filename, index=False)
        
        print(f"Created columns-filtered CSV: {output_filename}")
        print(f"Original columns: {list(df.columns)}")
        print(f"Kept columns: {columns_to_keep}")
        print(f"Total rows: {len(filtered_df)}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# prints a list of all dates of csv's in a dir
def Get_Group_Of_Dates():
    try:
        file_path = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(file_path) if f.endswith('.csv')]
        
        # Extract dates from filenames
        dates = []
        for filename in csv_files:
            if filename.startswith('Raw_Market_Data_'):
                # Extract the date part after "Raw_Market_Data_"
                date_part = filename.replace('Raw_Market_Data_', '').replace('.csv', '')
                
                # Remove "_On_Demand" suffix if present
                if '_On_Demand' in date_part:
                    date_part = date_part.replace('_On_Demand', '')
                
                dates.append(date_part)
        
        # Sort dates chronologically
        dates.sort()
        
        # Output to terminal
        print("Available dates in CSV files:")
        print("['" + "', '".join(dates) + "']")
        
        print(f"\nTotal number of dates: {len(dates)}")
        
        return dates
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


'''
dumbo google sheets can't make a sublist from my price movement. so this is going to read all of bulk data
formula: go through each price movement, takes actions on the first trigger number found
if target: add target
if sl: add sl
if sublist_trigger: make a sublist and add target/sl whichever comes first
how it's organized
-it tests tons of combos and skips impossible combos (those are the 'rules')
-it makes a list of each outcome for each trade, index 0-3 are the parameters, index 4 is the sum, the rest are the results
-at the end it saves the best x combos to a csv for google sheets
'''
def Stupid_Sublist_Calculation():
    try:
        bulk_csv_data = "Csv_Files/3_Final_Trade_Csvs/Bulk_Combined.csv"
        volatility = 0.7
        targets = [0.3,0.4,0.5,0.6]
        sublist_triggers = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        sublist_targets = [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
        stop_losss = [-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
        
        df = pd.read_csv(bulk_csv_data)
        
        # Filter dataframe to only include rows where Entry Volatility Percent is at least 0.7
        filtered_df = df[df['Entry Volatility Percent'] >= volatility]
        
        # Create a dictionary to store all sublists
        all_sublists = {}
        curr_paras = {'target': None, 'sublist_trigger': None, 'sublist_target': None, 'stop_loss': None}
        

        # RULES: skip the loop if any of these are not true
        # target must be more than sublist trigger 
        # target must be more than sublist_targets
        # sublist_trigger must be more than sublist_target
        # sublist_target must be more than stop loss
        for target in targets:
            curr_paras['target'] = target

            for sublist_trigger in sublist_triggers:
                if (sublist_trigger >= target):
                    continue

                curr_paras['sublist_trigger'] = sublist_trigger
                
                for sublist_target in sublist_targets:
                    if ((sublist_target >= target) or (sublist_target >= sublist_trigger)):
                        continue
            
                    curr_paras['sublist_target'] = sublist_target

                    for stop_loss in stop_losss:
                        if (stop_loss >= sublist_target):
                            continue
                        curr_paras['stop_loss'] = stop_loss
                        sublist = [target, sublist_trigger, sublist_target, stop_loss]
                        
                        for index, row in filtered_df.iterrows():
                            price_movement_str = str(row['Price Movement'])
                            
                            # Split the price movement string into a list of floats
                            if (len(price_movement_str) < 0):   # some rows are 0.0 but idk if they're 0 or 0.0 here
                                sublist.append(filtered_df.loc[index, 'Percent Change'])
                                continue
                            else:
                                price_movement_list = [float(x) for x in price_movement_str.split('|')]
                        
                            updated_flag = False
                            for j, value in enumerate(price_movement_list):
                                if value == target:
                                    sublist.append(target)
                                    updated_flag = True
                                    break

                                elif value == stop_loss:
                                    sublist.append(stop_loss)
                                    updated_flag = True
                                    break

                                elif value == sublist_trigger:
                                    # Copy from sublist_trigger to the end (excluding sublist_trigger itself)
                                    sublist_values = price_movement_list[j + 1:]
                                    for val in sublist_values:
                                        if (val == target):
                                            sublist.append(target)
                                            updated_flag = True
                                            break
                                        elif (val == stop_loss):
                                            sublist.append(stop_loss)
                                            updated_flag = True
                                            break
                                        elif (val == sublist_target):
                                            sublist.append(sublist_target)
                                            updated_flag = True
                                            break
                                        
                                    else: # for-else loop
                                        # it didn't find target/sl
                                        sublist.append(filtered_df.loc[index, 'Percent Change'])
                                        updated_flag = True

                                if (updated_flag == True):
                                    break
                            
                            # If no condition was met, append the Percent Change as default
                            if not updated_flag:
                                sublist.append(filtered_df.loc[index, 'Percent Change'])
                            
                        # find the sum
                        sublist_sum = 0
                        for val in sublist:
                            sublist_sum += val
                        sublist.insert(4, round(sublist_sum, 2))

                        # Store this sublist with the target index
                        values_str = ','.join(str(value) for value in curr_paras.values())
                        all_sublists[values_str] = sublist
        
        # Get the top x sublists based on the sum (index 4 of each sublist)
        best_sublists = {}
        sorted_items = sorted(all_sublists.items(), key=lambda x: x[1][4], reverse=True)
        
        for i in range(min(50, len(sorted_items))):
            key, sublist = sorted_items[i]
            best_sublists[key] = sublist

        # Create DataFrame from all sublists
        result_df = pd.DataFrame(best_sublists)

        output_filename = f"Csv_Files/dumb_code_instead_of_sheets_calculations/Stupid_Sublist_Thing.csv"
        result_df.to_csv(output_filename, index=False)
            
        print(f"Created sublist CSV: {output_filename}")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)



'''
move stop loss calculations
My target is actually an alert, and when it hits it I move the sl 0.x under the target and use a new upper target
'''
def Move_Targets_Calculations():
    try:
        volatilities = [0.5, 0.6, 0.7, 0.8, 0.9]
        alert_targets = [0.2,0.3,0.4,0.5,0.6]
        upper_targets = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        alert_stop_losss = [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
        normal_stop_losss = [-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
        bulk_csv_data = "Csv_Files/3_Final_Trade_Csvs/Bulk_Combined.csv"
        df = pd.read_csv(bulk_csv_data)
        volatility_dataframes = {}
        output_filename = "Csv_Files/dumb_code_instead_of_sheets_calculations/Move_Targets.csv"

        if os.path.exists(output_filename):
            os.remove(output_filename)
            
        for volatility in volatilities:
            all_sublists = {}
            curr_paras = {'alert_target': None, 'upper_target': None, 'alert_stop_loss': None, 'normal_stop_loss': None}
            filtered_df = df[(df['Entry Volatility Percent'] >= volatility) & (df['Entry Volatility Ratio'] >= 1)]

            for alert_target in alert_targets:
                curr_paras['alert_target'] = alert_target

                for upper_target in upper_targets:
                    if (upper_target <= alert_target):
                        continue

                    curr_paras['upper_target'] = upper_target
                    
                    for alert_stop_loss in alert_stop_losss:
                        if ((alert_stop_loss >= upper_target) or alert_stop_loss >= alert_target):
                            continue

                        curr_paras['alert_stop_loss'] = alert_stop_loss

                        for normal_stop_loss in normal_stop_losss:
                            if ((normal_stop_loss >= alert_target) or (normal_stop_loss >= alert_stop_loss)):
                                continue

                            curr_paras['normal_stop_loss'] = normal_stop_loss
                            sublist = [alert_target, upper_target, alert_stop_loss, normal_stop_loss]

                            for index, row in filtered_df.iterrows():
                                price_movement_str = str(row['Price Movement'])

                                if (len(price_movement_str) < 0):
                                    sublist.append(filtered_df.loc[index, 'Percent Change'])
                                    continue
                                else:
                                    price_movement_list = [float(x) for x in price_movement_str.split('|')]

                                updated_flag = False

                                for j, value in enumerate(price_movement_list):
                                    if (value == alert_target):
                                        for k, value in enumerate(price_movement_list[j + 1:], start=j + 1):
                                            if (value == upper_target):
                                                sublist.append(upper_target)
                                                updated_flag = True
                                                break
                                            elif (value == alert_stop_loss):
                                                sublist.append(alert_stop_loss)
                                                updated_flag = True
                                                break
                                        else:
                                            sublist.append(filtered_df.loc[index, 'Percent Change'])
                                            updated_flag = True

                                    elif (value == normal_stop_loss):
                                        sublist.append(normal_stop_loss)
                                        updated_flag = True
                                        break

                                    if (updated_flag == True):
                                        break

                                if (updated_flag == False):
                                    sublist.append(filtered_df.loc[index, 'Percent Change'])

                            sublist_sum = 0
                            for val in sublist[4:]:
                                sublist_sum += val
                            sublist.insert(4, round(sublist_sum, 2))
                            values_str = ','.join(str(value) for value in curr_paras.values())
                            all_sublists[values_str] = sublist

            best_sublists = {}
            sorted_items = sorted(all_sublists.items(), key=lambda x: x[1][4], reverse=True)
            for i in range(min(50, len(sorted_items))):
                key, sublist = sorted_items[i]
                best_sublists[key] = sublist

            if best_sublists:
                volatility_dataframes[volatility] = pd.DataFrame(best_sublists)

        final_result = []
        for volatility in volatilities:
            if volatility in volatility_dataframes:
                df_data = volatility_dataframes[volatility]
                header_data = [f"Volatility: {volatility}"] + [""] * (df_data.shape[1] - 1)
                final_result.append(header_data)

                for _, row in df_data.iterrows():
                    final_result.append(row.tolist())

                if volatility != volatilities[-1]:
                    final_result.append([""] * df_data.shape[1])
                    
        if final_result:
            first_volatility = next(iter(volatility_dataframes.values()))
            final_df = pd.DataFrame(final_result, columns=first_volatility.columns)
            final_df.to_csv(output_filename, index=False)
            print(f"Created Move Targets CSV: {output_filename}")
        else:
            print("No data to write to CSV")

    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def Print_Volatility_Counts():
    try:
        volatilities = [0.5, 0.6, 0.7, 0.8, 0.9]
        bulk_csv_data = "Csv_Files/3_Final_Trade_Csvs/Bulk_Combined.csv"
        df = pd.read_csv(bulk_csv_data)
        counts = {}

        for volatility in volatilities:
            filtered_df = df[(df['Entry Volatility Percent'] >= volatility) & (df['Entry Volatility Ratio'] >= 1)]
            counts[str(volatility)] = filtered_df.shape[0]

        print(counts)
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


# finds the average time that ecah ticker reaches volatility x. reason is if it's still under by like 6:45 is that normal or bad?
def Volatility_Time_By_Ticker():
    try:
        data_dir = "Csv_Files/2_Raw_Market_Data/Used_Market_Data"
        volatility_target = 0.5
        
        # Dictionary to store results organized by date
        results_by_date = {}
        
        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            # Extract date from filename (3rd index after splitting by '_')
            filename_parts = csv_file.split('_')
            if len(filename_parts) >= 4:
                date = filename_parts[3]  # This should be the date part
            else:
                date = "Unknown_Date"
            
            file_path = os.path.join(data_dir, csv_file)
            df = pd.read_csv(file_path)
            
            # Dictionary to store first volatility timestamp for each ticker
            ticker_volatility_times = {}
            
            # Group by ticker and find first occurrence where Volatility Percent >= volatility_target
            for ticker in df['Ticker'].unique():
                ticker_data = df[df['Ticker'] == ticker]
                
                # Find first row where Volatility Percent reaches or exceeds the target
                volatility_reached = ticker_data[ticker_data['Volatility Percent'] >= volatility_target]
                
                if not volatility_reached.empty:
                    # Get the first timestamp where volatility target was reached
                    first_timestamp = volatility_reached.iloc[0]['Time']
                    ticker_volatility_times[ticker] = first_timestamp
                else:
                    # If volatility target was never reached, mark as "Never Reached"
                    ticker_volatility_times[ticker] = "Never Reached"
            
            # Store results for this date
            results_by_date[date] = ticker_volatility_times
        
        # Create output text file
        output_filename = "volatility_time_by_ticker_results.txt"
        
        with open(output_filename, 'w') as f:
            f.write(f"Volatility Time By Ticker Analysis (Target: {volatility_target})\n")
            f.write("=" * 60 + "\n\n")
            
            # Sort dates for consistent output
            sorted_dates = sorted(results_by_date.keys())
            
            for date in sorted_dates:
                f.write(f"Date: {date}\n")
                f.write("-" * 30 + "\n")
                
                ticker_times = results_by_date[date]
                
                # Sort tickers alphabetically for consistent output
                for ticker in sorted(ticker_times.keys()):
                    timestamp = ticker_times[ticker]
                    f.write(f"{ticker}: {timestamp}\n")
                
                f.write("\n")
        
        # Also print to console
        print(f"Analysis complete! Results saved to: {output_filename}")
        print(f"Volatility target: {volatility_target}")
        print(f"Processed {len(csv_files)} CSV files")
        
        # Print summary to console
        for date in sorted_dates:
            print(f"\nDate: {date}")
            ticker_times = results_by_date[date]
            for ticker in sorted(ticker_times.keys()):
                timestamp = ticker_times[ticker]
                print(f"  {ticker}: {timestamp}")
        
        return results_by_date
    
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


'''
moves all raw trades and raw market data csv's back to the todo folder
'''
def move_all_csvs_back():
    try:
        used_market_data_dir = "Csv_Files/2_Raw_Market_Data/Used_Market_Data"
        todo_market_data_dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
        used_trade_data_dir = "Csv_Files/1_tos_Raw_Trades/Used_Trade_Data"
        todo_trade_data_dir = "Csv_Files/1_tos_Raw_Trades/TODO_Trade_Data"
        
        # Ensure the TODO directories exist
        os.makedirs(todo_market_data_dir, exist_ok=True)
        os.makedirs(todo_trade_data_dir, exist_ok=True)
        
        # Move files from used_market_data_dir to todo_market_data_dir
        market_files_moved = 0
        if os.path.exists(used_market_data_dir):
            for filename in os.listdir(used_market_data_dir):
                if filename.endswith('.csv'):
                    source_path = os.path.join(used_market_data_dir, filename)
                    destination_path = os.path.join(todo_market_data_dir, filename)
                    
                    # Move the file
                    shutil.move(source_path, destination_path)
                    market_files_moved += 1
                    print(f"Moved market data: {filename}")
        
        # Move files from used_trade_data_dir to todo_trade_data_dir
        trade_files_moved = 0
        if os.path.exists(used_trade_data_dir):
            for filename in os.listdir(used_trade_data_dir):
                if filename.endswith('.csv'):
                    source_path = os.path.join(used_trade_data_dir, filename)
                    destination_path = os.path.join(todo_trade_data_dir, filename)
                    
                    # Move the file
                    shutil.move(source_path, destination_path)
                    trade_files_moved += 1
                    print(f"Moved trade data: {filename}")
        
        print(f"\nFile moving complete!")
        print(f"Market data files moved: {market_files_moved}")
        print(f"Trade data files moved: {trade_files_moved}")
        print(f"Total files moved: {market_files_moved + trade_files_moved}")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)


def directional_bias_helper_get_high_low_close(market_data_df):
    data_holder = {}

    for idx, row in market_data_df.iterrows():
        ticker = row['Ticker']
        curr_price = row['Price']
        curr_time = row['Time']

        if ticker not in data_holder:
            data_holder[ticker] = {
                'high': [],
                'low': [],
                'close': [],
                'prev close': [],
                'TR': [],
                'upTR': [],
                'downTR': [],
                'upAtr': [],
                'downAtr': [],
                'directional bias': [],
                'current_minute': None,
                'prev_row_price': curr_price
            }

        time_parts = curr_time.split(':')
        curr_minute = int(time_parts[1])
        
        # Check if we're in a new minute
        if data_holder[ticker]['current_minute'] != curr_minute:
            # If this isn't the first minute for this ticker, save the close price
            if data_holder[ticker]['current_minute'] != None:
                data_holder[ticker]['close'].append(data_holder[ticker]['prev_row_price'])
            
            # Start new minute - initialize high and low with current price
            data_holder[ticker]['current_minute'] = curr_minute
            data_holder[ticker]['high'].append(curr_price)
            data_holder[ticker]['low'].append(curr_price)
            
        else:
            # Still in same minute - update high and low if needed
            if curr_price > data_holder[ticker]['high'][-1]:
                data_holder[ticker]['high'][-1] = curr_price
            elif curr_price < data_holder[ticker]['low'][-1]:
                data_holder[ticker]['low'][-1] = curr_price
        
        # Store current price as previous for next iteration
        data_holder[ticker]['prev_row_price'] = curr_price

    # After processing all rows, add the final close price for each ticker
    for ticker in data_holder:
        data_holder[ticker]['close'].append(data_holder[ticker]['prev_row_price'])

    return data_holder


def Calcualte_Directional_Bias_V2(market_data_csv_path):
    market_data_df = pd.read_csv(market_data_csv_path)
    ema_length = 7
    
    # 1) get high, low, close for every ticker
    data_holder = directional_bias_helper_get_high_low_close(market_data_df)

    # 2) populate prev close. it's i-1 of close
    for ticker in data_holder:
        close_list = data_holder[ticker]['close']
        prev_close_list = data_holder[ticker]['prev close']
        prev_close_list.append(None)  # first index is non computable
        
        for i in range(len(close_list) - 1):
            prev_close_list.append(close_list[i])

        data_holder[ticker]['prev close'] = prev_close_list

    # 3) compute True range
    '''TR = max(
        High - Low,
        abs(High - PrevClose),
        abs(Low - PrevClose)) '''
    for ticker in data_holder:
        high_list = data_holder[ticker]['high']
        low_list = data_holder[ticker]['low']
        prev_close_list = data_holder[ticker]['prev close']
        tr_list = data_holder[ticker]['TR']
        
        # Compute TR for each index
        for i in range(len(high_list)):
            high = high_list[i]
            low = low_list[i]
            prev_close = prev_close_list[i]
            
            if (prev_close == None):
                tr = round(high - low, 3)
            else:
                tr = round(max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                ), 3)

            tr_list.append(tr)

        data_holder[ticker]['TR'] = tr_list

    # 4) classify TR as upTR or downTR
    # If Close > PrevClose → UpTR = TR, DownTR = 0
    # If Close < PrevClose → DownTR = TR, UpTR = 0
    # else they're both 0
    for ticker in data_holder:
        close_list = data_holder[ticker]['close']
        prev_close_list = data_holder[ticker]['prev close']
        tr_list = data_holder[ticker]['TR']
        up_tr_list = data_holder[ticker]['upTR']
        down_tr_list = data_holder[ticker]['downTR']
        
        # Classify each TR value based on close vs prev_close
        for i in range(len(tr_list)):
            close = close_list[i]
            prev_close = prev_close_list[i]
            tr = tr_list[i]
            
            # prev_close[0] is none
            if  (prev_close == None or (close == prev_close)):
                up_tr_list.append(0)
                down_tr_list.append(0)

            elif close > prev_close:
                up_tr_list.append(tr)
                down_tr_list.append(0)

            else:
                up_tr_list.append(0)
                down_tr_list.append(tr)
                
        data_holder[ticker]['upTR'] = up_tr_list
        data_holder[ticker]['downTR'] = down_tr_list

    # 5) compute wilders EMA using ema_length (use up/downTR for data). The first x entries of up/downAtr should be None due to warmup period
    for ticker in data_holder:
        up_tr_list = data_holder[ticker]['upTR']
        down_tr_list = data_holder[ticker]['downTR']
        up_atr_list = data_holder[ticker]['upAtr']
        down_atr_list = data_holder[ticker]['downAtr']
        
        # Calculate Wilder EMA for both upTR and downTR
        for i in range(len(up_tr_list)):
            if i < ema_length:
                # First 7 indexes should be None
                up_atr_list.append(None)
                down_atr_list.append(None)
            elif i == ema_length:
                # First EMA calculation - simple average of first 7 values
                up_avg = sum(up_tr_list[:ema_length]) / ema_length
                down_avg = sum(down_tr_list[:ema_length]) / ema_length
                up_atr_list.append(round(up_avg, 4))
                down_atr_list.append(round(down_avg, 4))
            else:
                # Subsequent calculations - Wilder EMA formula
                current_up_tr = up_tr_list[i]
                current_down_tr = down_tr_list[i]
                previous_up_ema = up_atr_list[i-1]
                previous_down_ema = down_atr_list[i-1]
                
                new_up_ema = previous_up_ema + (current_up_tr - previous_up_ema) / ema_length
                new_down_ema = previous_down_ema + (current_down_tr - previous_down_ema) / ema_length
                
                up_atr_list.append(round(new_up_ema, 4))
                down_atr_list.append(round(new_down_ema, 4))
        
        data_holder[ticker]['upAtr'] = up_atr_list
        data_holder[ticker]['downAtr'] = down_atr_list

    # 6) compute directional bias
    for ticker in data_holder:
        up_atr_list = data_holder[ticker]['upAtr']
        down_atr_list = data_holder[ticker]['downAtr']
        directional_bias_list = data_holder[ticker]['directional bias']
        
        # Calculate directional bias for each index
        for i in range(len(up_atr_list)):
            if up_atr_list[i] is None or down_atr_list[i] is None:
                # During warmup period, directional bias is None
                directional_bias_list.append(None)
            else:
                up_atr = up_atr_list[i]
                down_atr = down_atr_list[i]
                denominator = up_atr + down_atr
                
                if denominator == 0:
                    # Neutral bias when both are 0
                    directional_bias = 0.5
                else:
                    directional_bias = up_atr / denominator
                
                directional_bias_list.append(round(directional_bias, 3))
        
        data_holder[ticker]['directional bias'] = directional_bias_list
    
    return data_holder
  

# for each ticker in data_holder
# write that minutes data_holder[ticker]['directional bias'] for the whole minute to each line
# when we detect a minute change move to the next index of data_holder[ticker]['directional bias']
# if it's none then write "NaN"
def Add_Directional_Bias_To_Market_Data(data_holder, market_data_csv_path):
    try:
        df = pd.read_csv(market_data_csv_path)
        df['Directional Bias'] = None   # Initialize new directional bias column
        ticker_minute_index = {}        # Dictionary to track current minute index for each ticker
        
        for index, row in df.iterrows():
            ticker = row['Ticker']
            time_str = row['Time']
            
            time_parts = time_str.split(':')
            current_minute = int(time_parts[1])
            
            # Initialize ticker tracking if first time seeing this ticker
            if ticker not in ticker_minute_index:
                ticker_minute_index[ticker] = {
                    'current_minute': current_minute,
                    'minute_index': 0
                }
            
            # Check if we've moved to a new minute for this ticker
            if current_minute != ticker_minute_index[ticker]['current_minute']:
                ticker_minute_index[ticker]['current_minute'] = current_minute
                ticker_minute_index[ticker]['minute_index'] += 1
            
            # Get the directional bias value
            directional_bias_list = data_holder[ticker]['directional bias']
            minute_index = ticker_minute_index[ticker]['minute_index']
            
            # Check if we have data for this minute index
            if minute_index < len(directional_bias_list):
                bias_value = directional_bias_list[minute_index]
                if bias_value is None:
                    df.at[index, 'Directional Bias'] = "NaN"
                else:
                    df.at[index, 'Directional Bias'] = bias_value
        
        # Reorder columns to put Time as the rightmost column
        columns = df.columns.tolist()
        columns.remove('Time')
        columns.append('Time')
        df = df[columns]
        
        # Save the modified dataframe back to the same file
        df.to_csv(market_data_csv_path, index=False)
        
        print(f"Added Directional Bias column to: {market_data_csv_path}\n")
        
    except Exception as e:
        Main_Globals.ErrorHandler(fileName, inspect.currentframe().f_code.co_name, str(e), sys.exc_info()[2].tb_lineno)




#Stupid_Sublist_Calculation()
#Move_Targets_Calculations()
#Print_Volatility_Counts()
#Volatility_Time_By_Ticker()
#move_all_csvs_back()


'''market_data_dir = "Csv_Files/2_Raw_Market_Data/TODO_Market_Data"
other_dir = "Csv_Files/2_Raw_Market_Data/Used_Market_Data"

# Process all files in market_data_dir
for filename in os.listdir(market_data_dir):
    file_path = f"{market_data_dir}/{filename}"
    print(f"Processing: {filename}")

    data_holder = Calcualte_Directional_Bias_V2(file_path)
    Add_Directional_Bias_To_Market_Data(data_holder, file_path)
'''