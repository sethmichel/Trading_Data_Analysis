import pandas as pd
import os
import sys
import inspect

fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

'''
convert spaces to '_', convert entry time to seconds since market open, change macd data to absolute vals
drop result revealing columns and non-computable columns
Converts categorical columns (Ticker, Trade Type)
rsi complexity: actually, 50 should be considered a low value and less than 30/more than 70 a high value. so distance from 50 should mean a higher value
    rsi goal is better representation on the SHAP chart. blue is around 50, red is farther away THIS DOES NOT SAVE RSI DIRECTION
Returns:
    X (pd.DataFrame): Feature matrix
    y (pd.Series): Target vector (0 = fail, 1 = success)
'''
def Clean_Data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Convert target to integer (1 or 0). it probably already is but this also makes sure it's not a string or something
        df["Target"] = df["Target"].fillna(0).astype(int)
        
        # Convert categorical data. 'Trade Type': 'buy' → 1, 'short' → 0. ticker to numeric category
        df["Trade Type"] = df["Trade Type"].map({"buy": 1, "short": 0})
        df["Ticker"] = df["Ticker"].astype("category").cat.codes

        # drop unused column and columns that give away the result. raise error is there's any mistakes in this step
        drop_cols = ["Date", "Exit Time", "Time in Trade", 'Dollar Change', 'Total Investment', 'Qty', 'Entry Price', 'Exit Price',
                    'Best Exit Price', 'Best Exit Percent', 'Worst Exit Price', 'Worst Exit Percent', 'Prev 5 Min Avg Close Volume', 
                    'Price_Movement', 'Percent Change']
        missing_cols = [col for col in drop_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")
        else:
            df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

        # convert entry time into seconds since market open
        # make sure it's in the right format
        seconds_since_open = []
        for time_str in df['Entry Time']:
            h, m, s = map(int, time_str.split(':'))
            seconds_630 = ((6 * 60) + 30) * 60 # hours to minutes + minutes * 60 + seconds
            entry_time_seconds = ((h * 60) + m) * 60 + s  # hours to minutes + minutes * 60 + seconds

            seconds_since_open.append(entry_time_seconds - seconds_630)

        df["Seconds_Since_Open"] = seconds_since_open

        # change entry macd val/avg to absolute value. it goes between + and - which the model doesn't understand
        for col in ['Entry Macd Val', 'Entry Macd Avg']:
            if col in df.columns:
                df[col] = df[col].abs()

        # now drop the entry_time column since I'm done with it
        df = df.drop(columns=['Entry Time'])

        # convert all spaces in headers to _ to avoid errors
        df.columns = df.columns.str.replace(' ', '_')

        # Ask user if they want to drop the Ticker column
        #drop_ticker = input("Do you want to drop the Ticker column from the features? (y/n): ").strip().lower()
        #if drop_ticker == 'y':
        if 'Ticker' in df.columns:
            df = df.drop(columns=["Ticker"])

        # deal with rsi (described above) 
        df['RSI_Entry_50_Baseline'] = (df['Entry_Rsi'] - 50).abs() # doesn't save direction. shap: 50 is blue, farther from 50 = red
        #df = df.drop(columns=['Entry_Rsi'])

        # Separate features and target (do this even if we're not making a predictive model). need dependent/independent vars
        X = df.drop(columns=["Target"])
        y = df["Target"]
        
        return X, y
    
    except Exception as e:
        print(f"ERROR: file: {fileName}, function: {inspect.currentframe().f_code.co_name}, line: {sys.exc_info()[2].tb_lineno}, error: {str(e)}, ")
        raise ValueError()
