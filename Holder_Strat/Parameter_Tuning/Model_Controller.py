import pandas as pd
import pickle
import numpy as np
import Helper_Functions
import json
import Target_Model.Target_Model_Training as Target_Model_Training
import Target_Model.Target_Model_Diagnostics as Target_Model_Diagnostics
import Sl_Model.Sl_Model_Training as Sl_Model_Training
import Sl_Model.Sl_Model_Diagnostics as Sl_Model_Diagnostics
import Success_Prob_Model.Success_Prob_Model_Training as Success_Prob_Model_Training
import Success_Prob_Model.Success_Prob_Model_Diagnostics as Success_Prob_Model_Diagnostics

''' Considersations
NOTE: market data has 'Volatility Percent' and 'Early Morning Atr Warmup Fix'. vol% takes 14 minutes to warm up so we use the other column
for 14 minutes. after 14 minutes the warmupfix column is nan. thus it's best to overwrite vol% column with warmup fix column for first 14 minutes

'''


target_model_version = 'v0.2'
sl_model_version = 'v0.2'
success_prob_model_version = 'v0.1'
Target_Model_Training.Set_Version(target_model_version)
Target_Model_Diagnostics.Set_Version(target_model_version)
Sl_Model_Training.Set_Version(sl_model_version)
Sl_Model_Diagnostics.Set_Version(sl_model_version)
Success_Prob_Model_Training.Set_Version(success_prob_model_version)
Success_Prob_Model_Diagnostics.Set_Version(success_prob_model_version)

columns_to_keep = ["Date", "Trade Id", "Ticker", "Entry Time", "Time in Trade", "Entry Price", "Exit Price", "Trade Type", "Exit Price", "Entry Volatility Percent", 'Entry Volatility Ratio', "Original Holding Reached", "Original Best Exit Percent", "Original Percent Change"]
bulk_df_all_values = pd.read_csv("Holder_Strat/Summary_Csvs/bulk_summaries.csv")[columns_to_keep]
LOAD_SAVED_MARKET_DATA = True
market_data_dict_by_ticker = Helper_Functions.Load_Market_Data_Dictionary(bulk_df_all_values, LOAD_SAVED_MARKET_DATA) # {date: {ticker: dataframe, ticker2: dataframe, ...}, date: ...}


# TARGET MODEL --------------------------------------------

def Load_Target_Model_And_Data():
    # load roi dictionary, trade start/end indexes
    model_folder = 'Target_Model'
    holding_value = 0.6
    holding_sl_value = 0
    largest_sl_value = -0.4

    roi_dictionary_path = f"Holder_Strat/Parameter_Tuning/{model_folder}/Data/roi_dictionary_saved_holder={holding_value},holderSL={holding_sl_value},startingSL={largest_sl_value}.json"
    roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Load_Roi_Dictionary_And_Values(roi_dictionary_path)

    # load model and training data
    all_data_samples_x, all_roi_samples_y = Target_Model_Training.Load_Test_Data()
    model, scaler, smearing_factor = Target_Model_Training.Load_Model_Data(holding_value, holding_sl_value, largest_sl_value)

    return roi_dictionary, trade_end_timestamps, trade_start_indexes, all_data_samples_x, all_roi_samples_y, model, scaler, smearing_factor


def Train_Target_Model():
    load_model_data = False
    holding_value = 0.6
    holding_sl_value = 0
    largest_sl_value = -0.4

    if (load_model_data == False):
        # make roi dictionary
        # NOTE: this function loads them if they already exist
        roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Create_Roi_Dictionary_For_Trades(
                bulk_df_all_values, market_data_dict_by_ticker, 
                model_folder='Target_Model', 
                holding_value=holding_value, 
                holding_sl_value=holding_sl_value, 
                largest_sl_value=largest_sl_value)

        # collect data
        all_data_samples_x, all_roi_samples_y, trade_count = Target_Model_Training.Collect_Data(
                bulk_df_all_values, market_data_dict_by_ticker, roi_dictionary, trade_end_timestamps, trade_start_indexes)

        # save that data
        Target_Model_Training.Save_Training_Data(all_data_samples_x, all_roi_samples_y)

        print("="*30)
        print(f"TARGET MODEL MODEL TRAINING SUMMARY {target_model_version}\n")

        # train model
        model, x_scaled, scaler, smearing_factor = Target_Model_Training.Train_Model(all_data_samples_x, all_roi_samples_y)

        # save model data
        Target_Model_Training.Save_Model_Data(model, scaler, smearing_factor, holding_value, holding_sl_value, largest_sl_value)

        print(f"Model: {trade_count} training samples")
        print("="*30)


def Target_Model_Run_Diagnostics():
    # load data/model
    roi_dictionary, trade_end_timestamps, trade_start_indexes, \
    all_data_samples_x, all_roi_samples_y, model, scaler, \
    smearing_factor = Load_Target_Model_And_Data()
    # NOTE: remember target model is trained on only winning trades. so depending on the test you may need to filter bulk_df_all_values

    # actual diagnostics
    #Target_Model_Diagnostics.Run_Model_Diagnostics(model, scaler, smearing_factor, all_data_samples_x, all_roi_samples_y)

    # run model over trade history
    mode = 'model values' # 'model values' means use model, 'max values' means use hard coded max values
    Target_Model_Diagnostics.Run_Model_Performance_Over_Trade_History(model, scaler, smearing_factor, bulk_df_all_values, market_data_dict_by_ticker, 
                                             roi_dictionary, trade_end_timestamps, trade_start_indexes, mode)    

    #Target_Model_Diagnostics.Get_Model_Test_Values(model, scaler, smearing_factor)

# ---------------------------------------------------------


# SL MODEL ------------------------------------------------
def Load_Sl_Model_And_Data():
    model_folder = 'Sl_Model'
    holding_value = 0.6
    holding_sl_value = 0
    largest_sl_value = -0.4

    # load roi dictionary, trade start/end indexes
    roi_dictionary_path = f"Holder_Strat/Parameter_Tuning/{model_folder}/Data/roi_dictionary_saved_holder={holding_value},holderSL={holding_sl_value},startingSL={largest_sl_value}.json"
    roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Load_Roi_Dictionary_And_Values(roi_dictionary_path)

    # load model and training data
    trades_processed, neither_trades_detected = Sl_Model_Training.Load_Training_data()
    model, scaler, results_df = Sl_Model_Training.Load_Model_data(holding_value, holding_sl_value, largest_sl_value)

    return roi_dictionary, trade_end_timestamps, trade_start_indexes, results_df, trades_processed, neither_trades_detected, model, scaler


def Train_Sl_Model():
    load_model_data = False
    sl_list = [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8] # must be highest to lowest order
    holding_value = 0.6
    holding_sl_value = 0

    if (load_model_data == False):
        # create roi dictionary (using the largetst sl so it includes everything)
        # NOTE: this function loads them if they already exist
        roi_dictionary, trade_end_timestamps, trade_start_indexes \
                = Helper_Functions.Create_Roi_Dictionary_For_Trades(
                    bulk_df_all_values, market_data_dict_by_ticker, 
                    model_folder='Sl_Model', 
                    holding_value=holding_value, 
                    holding_sl_value=holding_sl_value, 
                    largest_sl_value=min(sl_list))
        
        # collect training data
        results_df, trades_processed, neither_trades_detected \
                = Sl_Model_Training.Collect_data(
                        sl_list, bulk_df_all_values, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes)

        # save training data
        Sl_Model_Training.Save_Training_Data(results_df, trades_processed, neither_trades_detected)

        if len(results_df) > 0:
            # train model
            model, result_df, scaler = Sl_Model_Training.Train_Model(results_df)

            # save model
            Sl_Model_Training.Save_Model_Data(model, scaler, result_df)

        else:
            print("ERROR: No training data remaining after dropping 'neither' trades!")
            return
    
        print("="*30)
        print(f"SL MODEL MODEL TRAINING SUMMARY {sl_model_version}\n")
        print(f"Model (results_df): {len(results_df)} training samples")
        print(f"'Neither' trades detected (dropped): {neither_trades_detected}")
        print(f"'Neither' trades treated as failures: {neither_trades_detected}")
        print("="*30)


def Sl_Model_Run_Diagnostics():
    # load data/model
    roi_dictionary, trade_end_timestamps, trade_start_indexes, \
    results_df, trades_processed, neither_trades_detected, \
    model, scaler = Load_Sl_Model_And_Data()

    # actual diagnostics
    Sl_Model_Diagnostics.Model_Diagnostics(model, results_df, scaler)

    Sl_Model_Diagnostics.Give_Model_Test_Input(model, scaler)
# ---------------------------------------------------------


# SUCCESS PROBABILITY MODEL -------------------------------

def Load_Success_Prob_Model_And_Data():
    # load roi dictionary, trade start/end indexes
    model_folder = 'Success_Prob_Model'
    holding_value = 0.6
    holding_sl_value = 0
    largest_sl_value = -0.4

    roi_dictionary_path = f"Holder_Strat/Parameter_Tuning/{model_folder}/Data/roi_dictionary_saved_holder={holding_value},holderSL={holding_sl_value},startingSL={largest_sl_value}.json"
    roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Load_Roi_Dictionary_And_Values(roi_dictionary_path)

    # load model and training data
    results_df, neither_count, trade_count = Success_Prob_Model_Training.Load_Training_Data()
    model, scaler, x_scaled = Success_Prob_Model_Training.Load_Model_Data(holding_value, holding_sl_value, largest_sl_value)

    return roi_dictionary, trade_end_timestamps, trade_start_indexes, results_df, neither_count, trade_count, model, scaler, x_scaled


def Train_Success_Prob_Model():
    load_model_data = False
    holding_value = 0.6
    holding_sl_value = 0
    largest_sl_value = -0.4

    if (load_model_data == False):
        # make roi dictionary
        # NOTE: this function loads them if they already exist
        roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Create_Roi_Dictionary_For_Trades(
                bulk_df_all_values, market_data_dict_by_ticker, 
                model_folder='Success_Prob_Model', 
                holding_value=holding_value, 
                holding_sl_value=holding_sl_value, 
                largest_sl_value=largest_sl_value)

        # collect data
        results_df, neither_count, trade_count = Success_Prob_Model_Training.Collect_Training_Data(
                bulk_df_all_values, roi_dictionary, holding_value, largest_sl_value)

        # save that data
        Success_Prob_Model_Training.Save_Training_Data(results_df, neither_count, trade_count)

        print("="*30)
        print(f"SUCCESS PROBABILITY MODEL MODEL TRAINING SUMMARY {success_prob_model_version}\n")
        print(f"neither count: {neither_count}")
        print(f"data size (w/o neither trades): {len(bulk_df_all_values)}\n")

        # train model
        model, scaler, x_scaled = Success_Prob_Model_Training.Train_Model(results_df)

        # save model data
        Success_Prob_Model_Training.Save_Model_Data(model, scaler, x_scaled, holding_value, holding_sl_value, largest_sl_value)

        print("="*30)


def Success_Prob_Model_Run_Diagnostics():
    # load data/model
    roi_dictionary, trade_end_timestamps, trade_start_indexes, \
    results_df, neither_count, trade_count, model, scaler, x_scaled = Load_Target_Model_And_Data()
    # NOTE: remember target model is trained on only winning trades. so depending on the test you may need to filter bulk_df_all_values

    # actual diagnostics
    #Success_Prob_Model_Diagnostics.Run_Model_Diagnostics(model, scaler, all_data_samples_x, all_roi_samples_y)

    # run model over trade history
    mode = 'model values' # 'model values' means use model, 'max values' means use hard coded max values
    Success_Prob_Model_Diagnostics.Run_Model_Performance_Over_Trade_History(model, scaler, bulk_df_all_values, market_data_dict_by_ticker, 
                                             roi_dictionary, trade_end_timestamps, trade_start_indexes, mode)    

    #Success_Prob_Model_Diagnostics.Get_Model_Test_Values(model, scaler)

# ---------------------------------------------------------


#Train_Target_Model()
#Target_Model_Run_Diagnostics()

Train_Success_Prob_Model()