import pandas as pd
import os
import inspect
import sys
import shutil
import Main_Globals
from datetime import datetime


def Write_Analysis(message):
    # Write to Analysis_Results.txt - overwrite if exists, create if doesn't exist
    with open("Analysis_Results.txt", "a") as file:
        file.write(message)


# scans price movement for if price satisfied 0.3/-0.3. if neither are there then use the final value of the list
def Helper_Volatility_Percent_vs_Ratio_Target_Finder_1(df):
    # scan whole original df first, add a new column, then filter that df
    
    def find_target_result(price_movement_str):
        if price_movement_str == '':
            return None
            
        movements = [float(x.strip()) for x in price_movement_str.split('|')]
        for movement in movements:
            if movement == 0.3:
                return 0.3
            elif movement == -0.3:
                return -0.3
                
        # If neither 0.3 nor -0.3 found, return the final value
        return movements[-1]
    
    # Apply the function to create the new column
    df['target_Basic_0.3_-0.3_Result'] = df['Price Movement'].apply(find_target_result)
    
    return df

# scans price movement for if price satisfied target = 0.5, upper target = 0.9, upper stop loss = -0.1, stop loss = -0.5.
# if it doesn't then use the final value of the list
def Helper_Volatility_Percent_vs_Ratio_Target_Finder_2(df):
    target = 0.5
    upper_target = 0.9
    upper_stop_loss = -0.1
    stop_loss = -0.5
    
    def find_target_result(price_movement_str):
        if price_movement_str == '':
            return None
            
        movements = [float(x.strip()) for x in price_movement_str.split('|')]
        
        # Phase 1: Look for target or stop_loss
        for movement in movements:
            if movement == stop_loss:
                return stop_loss
            
            elif movement == target:
                # Switch to phase 2: look for upper_target or upper_stop_loss
                remaining_movements = movements[movements.index(movement) + 1:]
                for remaining_movement in remaining_movements:
                    if remaining_movement == upper_stop_loss:
                        return upper_stop_loss
                    
                    elif remaining_movement == upper_target:
                        return upper_target
                    
                # If we reach here, neither upper target nor upper stop loss found
                return movements[-1]
        
        # If neither target nor stop_loss found in phase 1, return final value
        return movements[-1]
    
    # Apply the function to create the new column
    df['Target_Complex_0.5_0.9_Result'] = df['Price Movement'].apply(find_target_result)
    
    return df


def Helper_Volailtity_Percent_vs_Ratio_Find_Top_5(result_data_basic, result_data_complex):
    best_basic_combos = []
    best_complex_combos = []
    
    # Round net change values to 2 decimal places
    for combo_id in result_data_basic:
        result_data_basic[combo_id]['net change'] = round(result_data_basic[combo_id]['net change'], 2)
    
    for combo_id in result_data_complex:
        result_data_complex[combo_id]['net change'] = round(result_data_complex[combo_id]['net change'], 2)
    
    # Find the top 5 combos in result_data_basic by 'net change'
    sorted_basic = sorted(
        result_data_basic.items(),
        key=lambda x: x[1]['net change'],
        reverse=True
    )
    best_basic_combos = []
    for combo_id, entry in sorted_basic[:5]:
        entry_with_id = entry.copy()
        # Move 'combo id' to be the first attribute
        entry_with_id = {'combo id': combo_id, **entry_with_id}
        best_basic_combos.append(entry_with_id)

    # Do the same for result_data_complex
    sorted_complex = sorted(
        result_data_complex.items(),
        key=lambda x: x[1]['net change'],
        reverse=True
    )
    best_complex_combos = []
    for combo_id, entry in sorted_complex[:5]:
        entry_with_id = entry.copy()
        # Move 'combo id' to be the first attribute
        entry_with_id = {'combo id': combo_id, **entry_with_id}
        best_complex_combos.append(entry_with_id)

    return best_basic_combos, best_complex_combos


'''
use all combos of volatility % and volatility ratio
do it on 0.3/-0.3 and on 
target = 0.5, upper target = 0.9, upper stop loss = -0.1, stop loss = -0.5
output: organized text saying what the best 5 combos are, their results, their counts
'''
def Volatility_Percent_vs_Ratio(df):
    global analysis_message

    percents = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    ratios = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
    result_data_basic = {}
    result_data_complex = {}
    
    # add 0.3/-0.3 target results and then the 0.5/0.9/-0.1/-0.5 one
    df = Helper_Volatility_Percent_vs_Ratio_Target_Finder_1(df)
    df = Helper_Volatility_Percent_vs_Ratio_Target_Finder_2(df)

    for percent in percents:
        for ratio in ratios:
            filtered_df = df[(df['Entry Volatility Percent'] >= percent) & (df['Entry Volatility Ratio'] >= ratio)]
            combo_id = f'{percent},{ratio}'
            result_data_basic[combo_id] = {"count": filtered_df.shape[0],  # total rows 
                                           "wins": 0,         # total wins
                                           'losses': 0,       # total losses
                                           'neither': 0,      # it didn't win or lose (exited at macd or something else)
                                           'net change': 0}   # % change
            
            result_data_complex[combo_id] = {"count": filtered_df.shape[0],
                                             "upper target wins": 0,
                                             'upper sl losses': 0,
                                             "lower sl losses": 0,
                                             'neither': 0,
                                             'net change': 0}

            for idx, row in filtered_df.iterrows():
                # 1) use 0.3 / -0.3. Get the count, win rate, percent change
                result_1 = row['target_Basic_0.3_-0.3_Result']
            
                if (result_1 == 0.3):
                    result_data_basic[combo_id]["wins"] += 1
                elif (result_1 == -0.3):
                    result_data_basic[combo_id]["losses"] += 1
                else:
                    result_data_basic[combo_id]["neither"] += 1

                result_data_basic[combo_id]['net change'] += result_1
    
                # 2) use the complex paramters. record the same things
                result_2 = row['Target_Complex_0.5_0.9_Result']

                if (result_2 == 0.9):
                    result_data_complex[combo_id]["upper target wins"] += 1
                elif (result_2 == -0.1):
                    result_data_complex[combo_id]["upper sl losses"] += 1
                elif (result_2 == -0.5):
                    result_data_complex[combo_id]["lower sl losses"] += 1
                else:
                    result_data_complex[combo_id]["neither"] += 1

                result_data_complex[combo_id]['net change'] += result_2

    # 3 find the best 5 combos for each subtest by net change. save the combo id as a new attribute
    best_basic_combos, best_complex_combos = Helper_Volailtity_Percent_vs_Ratio_Find_Top_5(result_data_basic, result_data_complex)

    # 4) write the results
    subtest_1_results = ""
    subtest_2_results = ""
    for i in range(0, len(best_basic_combos)):
        subtest_1_results += f"{i+1}) {best_basic_combos[i]}\n"

    for i in range(0, len(best_complex_combos)):
        subtest_2_results += f"{i+1}) {best_complex_combos[i]}\n"
    
    message = (f"TEST 1: testing all combos of volaility percent vs volatility ratio.\n"
               f"parameters: subtest 1: 0.3/-0.3, subtest 2: 0.5/0.9 lower/upper targets with -0.5/-0.1 lower/upper stop losses\n"
               f"subtest 1:\n"
               f"{subtest_1_results}\n"
               f"subtest 2:\n"
               f"{subtest_2_results}\n\n")
    message = message.replace("'", '').replace("{", '').replace("}", '')

    Write_Analysis(message)


def Write_Grid_Seach_Resutls(all_sublists):
    best_sublists_sum = {}
    best_sublists_winrate = {}
    
    # find top 10 sublists by sum
    sum_sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['sum'], reverse=True)
    for i in range(min(30, len(sum_sorted_items))):
        key, sublist = sum_sorted_items[i]
        sublist_rounded = sublist.copy()
        sublist_rounded['sum'] = round(sublist_rounded['sum'], 2)
        best_sublists_sum[key] = sublist_rounded

    # find top 10 sublists by winrate
    winrate_sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['winrate'], reverse=True)
    for i in range(min(30, len(winrate_sorted_items))):
        key, sublist = winrate_sorted_items[i]
        sublist_rounded = sublist.copy()
        sublist_rounded['winrate'] = sublist_rounded['winrate']
        best_sublists_winrate[key] = sublist_rounded

    message = (f"TEST 2: testing all combos of volatility percent vs volatility ratio vs parameters.\n"
        f"parameters: using an upper/lower target and upper/lower stop loss\n"
        f"Total combinations tested: {len(all_sublists)}\n"
        f"Results (Top 10 by sum):\n"
        f"volatility, ratio, adx28, 14, 7, ads zscore, rsi_type, normal_target, upper_target, normal_stop_loss, upper_stop_loss\n")

    for i, (key, sub_list) in enumerate(best_sublists_sum.items()):
        message += f"{i+1}) id: {sub_list['id']}, sum: {sub_list['sum']}, count: {sub_list['count']}, wins: {sub_list['wins']}, losses: {sub_list['losses']}, neither: {sub_list['neither']}\n"
        
    message += (f"\nResults (Top 10 by win rate):\n"
                f"volatility, ratio, adx28, 14, 7, ads zscore, rsi_type, normal_target, upper_target, normal_stop_loss, upper_stop_loss\n")

    for i, (key, sub_list) in enumerate(best_sublists_winrate.items()):
        message += f"{i+1}) id: {sub_list['id']}, sum: {sub_list['sum']}, count: {sub_list['count']}, wins: {sub_list['wins']}, losses: {sub_list['losses']}, neither: {sub_list['neither']}\n"

    message = message.replace("'", '').replace("{", '').replace("}", '')

    Write_Analysis(message)
    print("\nit's done\n")


'''
use all combos of volatility % and volatility ratio, and paramters 
parmater format is: ex) target = 0.5, upper target = 0.9, upper stop loss = -0.1, stop loss = -0.5
output: organized text saying what the best 5 combos are, their results, their counts
'''
def Grid_Search_Parameter_Optimization(df):
    volatilities = [0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9,1.0]
    ratios = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
    adx28s = [0.0,0.1,0.2,0.3,0.4,0.5]
    adx14s = [0.0,0.1,0.2,0.3,0.4,0.5]
    adx7s = [0.0,0.1,0.2,0.3,0.4,0.5]
    abs_macd_zScores = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]   # absolute value of z-score, not normal z-score
    extreme_rsis = [True, False, "either"]
    normal_targets = [0.2,0.3,0.4,0.5,0.6]
    upper_targets = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    upper_stop_losss = [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
    normal_stop_losss = [-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
    
    all_sublists = {}
    
    # Pre-process price movements to avoid repeated string operations
    df['price_movement_list'] = df['Price Movement'].apply(
        lambda x: [float(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else []
    )
    print("done - preprocessed price movements")
    
    for volatility in volatilities:
        for ratio in ratios:
            for adx28 in adx28s:
                for adx14 in adx14s:
                    for adx7 in adx7s:
                        for zscore in abs_macd_zScores:
                            for rsi_type in extreme_rsis:
                                filtered_df = df[(df['Entry Volatility Percent'] >= volatility) & 
                                                 (df['Entry Volatility Ratio'] >= ratio) & 
                                                 (df['Adx28'] >= adx28) & 
                                                 (df['Adx14'] >= adx14) & 
                                                 (df['Adx7'] >= adx7) &
                                                 (abs(df['Macd Z-Score']) >= zscore) &
                                                 (df['Rsi Extreme Prev Cross'] >= rsi_type)]

                                for normal_target in normal_targets:

                                    for upper_target in upper_targets:
                                        if (upper_target <= normal_target):
                                            continue
                                        
                                        for upper_stop_loss in upper_stop_losss:
                                            if ((upper_stop_loss >= upper_target) or upper_stop_loss >= normal_target):
                                                continue

                                            for normal_stop_loss in normal_stop_losss:
                                                if ((normal_stop_loss >= normal_target) or (normal_stop_loss >= upper_stop_loss)):
                                                    continue
                            
                                                sublist_key = f"{volatility}, {ratio}, {adx28}, {adx14}, {adx7}, {zscore}, {rsi_type}, {normal_target}, {upper_target}, {normal_stop_loss}, {upper_stop_loss}"
                                                sublist = {'id': sublist_key, 'sum': 0, 'count': len(filtered_df), 'wins': 0, 'losses': 0, 'neither': 0, 'winrate': 0}

                                                for index, row in filtered_df.iterrows():
                                                    price_movement_list = row['price_movement_list']

                                                    if not price_movement_list:  # Fixed: check if list is empty
                                                        sublist['neither'] += 1
                                                        # sum is unaffected
                                                        continue

                                                    updated_flag = False
                                                    curr_target = normal_target
                                                    curr_sl = normal_stop_loss
                                                    hit_target_once = False

                                                    for value in price_movement_list:
                                                        if value == curr_target:
                                                            if hit_target_once == False:
                                                                # Hit normal target - switch to upper targets
                                                                hit_target_once = True
                                                                curr_target = upper_target
                                                                curr_sl = upper_stop_loss
                                                            else:
                                                                # Hit upper target
                                                                sublist['sum'] += upper_target
                                                                sublist['wins'] += 1
                                                                updated_flag = True
                                                                break

                                                        elif (value == curr_sl):
                                                            # if we hit the normal OR upper stop loss. (target changes which is being used)
                                                            sublist['sum'] += curr_sl
                                                            sublist['losses'] += 1
                                                            updated_flag = True
                                                            break

                                                    # if we never hit a ending target - use the last value from price movement
                                                    if (updated_flag == False):
                                                        sublist['sum'] += price_movement_list[-1]
                                                        sublist['neither'] += 1

                                                sublist['winrate'] = round(sublist['count'] / sublist['wins'], 2)
                                                all_sublists[sublist_key] = sublist

    Write_Grid_Seach_Resutls(all_sublists)




# Ensure the text file "Analysis_Results.txt" exists (create if it doesn't)
if not os.path.exists("Analysis_Results.txt"):
    with open("Analysis_Results.txt", "w") as f_create:
        pass
# erase the text file if it exists
else:
    with open("Analysis_Results.txt", "w") as f:
        pass

data_dir = "Csv_Files/3_Final_Trade_Csvs"
data_file = "Bulk_Combined.csv"
df = pd.read_csv(f"{data_dir}/{data_file}")

#Volatility_Percent_vs_Ratio(df)
Grid_Search_Parameter_Optimization(df)
