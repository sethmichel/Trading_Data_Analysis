import pandas as pd
import os

# make a new csv of all 1 ticker
def Ticker_Csv():
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


# organize a csv by a column - highest to lowest
def Column_Sort_Csv():
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


def Only_Keep_Some_Columns_Csv():
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


# prints a list of all dates of csv's in a dir
def Get_Group_Of_Dates():
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
                        price_movement_str = str(row['Price_Movement'])
                        
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



'''
move stop loss calculations
My target is actually an alert, and when it hits it I move the sl 0.x under the target and use a new upper target
'''
def Move_Targets_Calculations():
    volatility = 0.5
    alert_targets = [0.2,0.3,0.4,0.5,0.6]
    upper_targets = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    alert_stop_losss = [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
    normal_stop_losss = [-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
    bulk_csv_data = "Csv_Files/3_Final_Trade_Csvs/Bulk_Combined.csv"

    df = pd.read_csv(bulk_csv_data)
    filtered_df = df[df['Entry Volatility Percent'] >= volatility]
    
    all_sublists = {}
    curr_paras = {'alert_target': None, 'upper_target': None, 'alert_stop_loss': None, 'normal_stop_loss': None}

    # RULES
    # upper_target > alert_target
    # alert_stop_loss < upper_target
    # alert_stop_loss < alert_target
    # normal stop loss < alert target
    # normal stop loss < alert_stop_loss
    for alert_target in alert_targets:
        curr_paras['alert_target'] = alert_target

        for upper_target in upper_targets:
            if (upper_target <= alert_target): # rule 1
                continue

            curr_paras['upper_target'] = upper_target
            
            for alert_stop_loss in alert_stop_losss:
                if ((alert_stop_loss >= upper_target) or alert_stop_loss >= alert_target): # rule 2 and 3
                    continue
        
                curr_paras['alert_stop_loss'] = alert_stop_loss

                for normal_stop_loss in normal_stop_losss:
                    if ((normal_stop_loss >= alert_target) or (normal_stop_loss >= alert_stop_loss)): # rule 4 and 5
                        continue

                    curr_paras['normal_stop_loss'] = normal_stop_loss
                    sublist = [alert_target, upper_target, alert_stop_loss, normal_stop_loss]
                    
                    for index, row in filtered_df.iterrows():
                        price_movement_str = str(row['Price_Movement'])
                        
                        # Split the price movement string into a list of floats
                        if (len(price_movement_str) < 0):   # some rows are 0.0 but idk if they're 0 or 0.0 here
                            sublist.append(filtered_df.loc[index, 'Percent Change'])
                            continue
                        else:
                            price_movement_list = [float(x) for x in price_movement_str.split('|')]

                        updated_flag = False
                        for j, value in enumerate(price_movement_list):
                            if (value == alert_target):
                                # new target is upper target
                                # new sl is alert stop loss
                                for k, value in enumerate(price_movement_list[j + 1:], start=j + 1):
                                    if (value == upper_target):
                                        sublist.append(upper_target)
                                        updated_flag = True
                                        break

                                    elif (value == alert_stop_loss):
                                        sublist.append(alert_stop_loss)
                                        updated_flag = True
                                        break
                                else: # for-else loop
                                    # it didn't find target/sl
                                    sublist.append(filtered_df.loc[index, 'Percent Change'])
                                    updated_flag = True

                            elif (value == normal_stop_loss):
                                sublist.append(normal_stop_loss)
                                updated_flag = True
                                break

                            if (updated_flag == True):
                                break
                        
                        # If no condition was met, append the Percent Change as default
                        if (updated_flag == False):
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

    output_filename = f"Csv_Files/dumb_code_instead_of_sheets_calculations/Move_Targets.csv"
    result_df.to_csv(output_filename, index=False)
        
    print(f"Created sublist CSV: {output_filename}")



#Stupid_Sublist_Calculation()
Move_Targets_Calculations()