import pandas as pd
from datetime import date, timedelta
import os
import shutil

def CreateDf(csvPath):
    df = pd.read_csv(csvPath, skiprows = 6)

    # delete all the canceled order rows
    index_to_delete = df[df['Unnamed: 0'] == 'Canceled Orders'].index[0]
    df = df[:index_to_delete]

    # drop useless columns
    x = list(df.columns)
    columns_to_drop = ["Unnamed: 0", "Unnamed: 1", "Spread", "Exp", "Strike", "Type", "Price Improvement", "Order Type"]
    df = df.drop(columns=columns_to_drop)  # 'errors="ignore"' prevents errors if a column doesn't exist

    # rename the first column to "Date"
    df.rename(columns={df.columns[0]: "Date"}, inplace = True)

    return df


def MoveProcessedFile(date_str, todoDir, doneDir):
    # Convert date format from MM-DD-YYYY to YYYY-MM-DD
    month, day, year = date_str.split('-')
    year = f'20{year}'
    if (len(month) == 1):
        month = f'0{month}'
    if (len(day) == 1):
        day = f'0{day}'
    formatted_date = f"{year}-{month}-{day}"
    
    # Construct the filename
    filename = f"{formatted_date}-TradeActivity.csv"
    source_path = os.path.join(todoDir, filename)
    dest_path = os.path.join(doneDir, filename)
    
    try:
        # Move the file
        shutil.move(source_path, dest_path)
        print(f"Moved {filename} from {todoDir} to {doneDir}")
    except FileNotFoundError:
        print(f"Warning: Could not find file {filename} in {todoDir}")
    except Exception as e:
        print(f"Error moving file: {str(e)}")


def GenerateTradeSummary(df):
    # Create a list to store trade summaries
    trade_summaries = []
    # Create a set to track processed rows
    processed_rows = set()
    # Dictionary to track current quantity for each symbol
    symbol_quantities = {}
    
    # Iterate through the dataframe from bottom to top
    for i in range(len(df) - 1, -1, -1):
        # Skip if row already processed
        if i in processed_rows:
            continue
            
        symbol = df.loc[i, "Symbol"]
        current_qty = QTYCorrector(df.loc[i, "Qty"])
        
        # Initialize quantity for symbol if not present
        if symbol not in symbol_quantities:
            symbol_quantities[symbol] = 0
            
        # Skip if this doesn't start a new position (quantity was already non-zero)
        if symbol_quantities[symbol] != 0:
            symbol_quantities[symbol] += current_qty
            continue
            
        # This is a new position (quantity was 0, now changing)
        entry_time = df.loc[i, "Date"].split()[1]  # Get just the time part
        initial_qty = abs(current_qty)  # Store the initial quantity
        current_price = PriceCorrector(df.loc[i, "Net Price"])
        entry_price = current_price  # Store initial entry price
        dollar_change = 0
        total_investment = abs(current_qty * current_price)  # Initial investment amount
        trade_type = "BUY" if current_qty > 0 else "SHORT"  # Determine if it's a buy or short trade
        processed_rows.add(i)
        has_multiple_entries = False  # Flag to track if there are multiple entries
        
        # Update the running quantity
        symbol_quantities[symbol] = current_qty
        
        # Look for matching trades going up the dataframe
        for j in range(i - 1, -1, -1):
            if j in processed_rows:
                continue
                
            if df.loc[j, "Symbol"] == symbol:
                new_qty = QTYCorrector(df.loc[j, "Qty"])
                new_price = PriceCorrector(df.loc[j, "Net Price"])
                
                # Check if this is another entry (quantity was 0 before this trade)
                if symbol_quantities[symbol] == 0:
                    has_multiple_entries = True
                
                # Calculate dollar change for this part of the trade
                if current_qty < 0:  # Short position
                    price_diff = current_price - new_price
                else:  # Long position
                    price_diff = new_price - current_price
                    
                # Adjust for the quantity being closed
                closing_qty = abs(new_qty) if new_qty < 0 else new_qty
                dollar_change += price_diff * closing_qty
                
                # Update current quantity and total investment
                current_qty += new_qty
                symbol_quantities[symbol] = current_qty
                
                # Only add to total investment if it's a new entry (quantity was 0)
                if symbol_quantities[symbol] - new_qty == 0:
                    total_investment += abs(new_qty * new_price)
                
                processed_rows.add(j)
                
                # If position is closed (qty = 0), record the trade
                if current_qty == 0:
                    exit_time = df.loc[j, "Date"].split()[1]  # Get just the time part
                    exit_price = new_price  # Store the exit price
                    
                    # Calculate time in trade
                    entry_datetime = pd.to_datetime(df.loc[i, "Date"])
                    exit_datetime = pd.to_datetime(df.loc[j, "Date"])
                    time_diff = exit_datetime - entry_datetime
                    time_str = str(time_diff).split('.')[0]  # Remove microseconds if present
                    time_in_trade = time_str.split('days ')[-1] if 'days' in time_str else time_str # get rid of days
                    
                    # Calculate percentage gain/loss
                    percent_change = (dollar_change / total_investment) * 100
                    
                    # Add trade summary
                    trade_summaries.append({
                        'Date': df.loc[i, "Date"].split()[0].replace("/","-"),  # Just the date part
                        'Symbol': symbol,
                        'Entry Time': entry_time,
                        'Exit Time': exit_time,
                        'Time in Trade': time_in_trade,
                        'Dollar Change': round(dollar_change, 2),
                        'Percent Change': round(percent_change, 2),
                        'Total Investment': round(total_investment, 2),
                        'Entry Price': round(entry_price, 4),
                        'Exit Price': round(exit_price, 4),
                        'Trade Type': trade_type,
                        'Quantity': None if has_multiple_entries else initial_qty,  # Use initial quantity unless multiple entries
                        'Best Exit Price': '',
                        'Best Exit Percent': '',
                        'Worst Exit Price': '',
                        'Worst Exit Percent': '',
                        'ATR14': None,
                        'ATR28': None
                    })
                    break
    
    # Create DataFrame from trade summaries
    summary_df = pd.DataFrame(trade_summaries)
    
    return summary_df


def QTYCorrector(qty):
    # Convert qty to a number, remove '+' if it's present
    if qty.startswith("+"):
        qty = qty[1:]
    if isinstance(qty, str):
        qty = float(qty)
    return qty


# Convert price to a number if it's a string
def PriceCorrector(price):
    if isinstance(price, str):
        price = float(price)
    
    return price


def AddMarketData(tradeDf, todoDir, doneDir):
    # Get the date from the first row (all rows have same date)
    date_str = tradeDf['Date'].iloc[0]
    # Convert date format to match CSV filename format
    month, day, year = date_str.split('-')
    if (len(year) == 2):
        year = f'20{year}'  # year was 25 instead of 2025
    if (len(day) == 1):
        day = f'0{day}'     # day was 4 instead of 04
    if (len(month) == 1):
        month = f'0{month}' # month was 4 instead of 04
    csv_filename = f"2MarketData/Data_{month}-{day}-{year}.csv"
    
    # Read the market data CSV
    market_df = pd.read_csv(csv_filename)
    
    # Convert time columns to datetime for easier comparison
    market_df['Time'] = pd.to_datetime(market_df['Time'], format='%H:%M:%S').dt.time
    tradeDf['Entry Time'] = pd.to_datetime(tradeDf['Entry Time'], format='%H:%M:%S').dt.time
    tradeDf['Exit Time'] = pd.to_datetime(tradeDf['Exit Time'], format='%H:%M:%S').dt.time
    
    # Process each trade
    for idx, trade in tradeDf.iterrows():
        symbol = trade['Symbol']
        entry_time = trade['Entry Time']
        exit_time = trade['Exit Time']
        
        # Find matching market data rows for this symbol
        symbol_data = market_df[market_df['Symbol'] == symbol].copy()
        
        if len(symbol_data) == 0:
            print(f"Warning: No market data found for symbol {symbol}")
            continue
        
        # Find the entry point (allowing 5 second window)
        entry_window = pd.Timedelta(seconds=5)
        entry_mask = symbol_data['Time'].apply(
            lambda x: abs(pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second) - 
                        pd.Timedelta(hours=entry_time.hour, minutes=entry_time.minute, seconds=entry_time.second)) <= entry_window
        )
        entry_data = symbol_data[entry_mask]
        
        if len(entry_data) == 0:
            print(f"Warning: No entry point found for {symbol} at {entry_time}")
            continue
        
        # Get ATR values from the entry point
        entry_idx = entry_data.index[0]
        tradeDf.at[idx, 'ATR14'] = round(symbol_data.loc[entry_idx, 'ATR14'], 4)
        tradeDf.at[idx, 'ATR28'] = round(symbol_data.loc[entry_idx, 'ATR28'], 4)
        
        # Find the exit point (allowing 5 second window)
        exit_mask = symbol_data['Time'].apply(
            lambda x: abs(pd.Timedelta(hours=x.hour, minutes=x.minute, seconds=x.second) - 
                        pd.Timedelta(hours=exit_time.hour, minutes=exit_time.minute, seconds=exit_time.second)) <= entry_window
        )
        exit_data = symbol_data[exit_mask]
        
        if len(exit_data) == 0:
            print(f"Warning: No exit point found for {symbol} at {exit_time}")
            continue
        
        # Get the index of entry and exit points
        exit_idx = exit_data.index[-1]
        
        # Get the relevant price data between entry and exit
        price_data = symbol_data.loc[entry_idx:exit_idx, 'Last']
        
        # Calculate best and worst prices based on trade type
        if trade['Trade Type'] == "BUY":
            best_price = price_data.max()  # Highest price is best for BUY
            worst_price = price_data.min()  # Lowest price is worst for BUY
        else:  # SHORT trade
            best_price = price_data.min()  # Lowest price is best for SHORT
            worst_price = price_data.max()  # Highest price is worst for SHORT
        
        # Update the trade dataframe
        tradeDf.at[idx, 'Best Exit Price'] = round(best_price, 4)
        tradeDf.at[idx, 'Worst Exit Price'] = round(worst_price, 4)
        
        # Calculate percentages based on trade type
        entry_price = trade['Entry Price']
        quantity = trade['Quantity']
        trade_type = trade['Trade Type']
        
        # Calculate initial position value
        initial_value = entry_price * quantity
        
        if trade_type == "BUY":
            best_percent = ((best_price - entry_price) * quantity / initial_value) * 100
            worst_percent = ((worst_price - entry_price) * quantity / initial_value) * 100
        else:  # SHORT trades
            best_percent = ((entry_price - worst_price) * quantity / initial_value) * 100
            worst_percent = ((entry_price - best_price) * quantity / initial_value) * 100
        
        tradeDf.at[idx, 'Best Exit Percent'] = round(best_percent, 2)
        tradeDf.at[idx, 'Worst Exit Percent'] = round(worst_percent, 2)
    
    return tradeDf


'''
WARNING ----------------------***********************---------------------********************
function is totally wrong. it's looking at price btw entry and exit times instead of entry and macd cross times.
To continue, I need to have macd calulations done and apply it to the market data files
'''
def AddEstimates(df, todoDir, doneDir):
    # Generate column names for target and stop loss combinations
    column_names = []
    target_values = [round(x * 0.1, 1) for x in range(2, 9)]  # 0.2 to 0.8
    stop_loss_values = [round(x * 0.1, 1) for x in range(3, 8)]  # 0.3 to 0.7
    
    for target in target_values:
        for stop_loss in stop_loss_values:
            column_name = f"{target}T&{stop_loss}SL"
            column_names.append(column_name)
            df[column_name] = None  # Initialize column with None values
    
    # Get the date from the first row
    date_str = df['Date'].iloc[0]
    month, day, year = date_str.split('-')
    if len(year) == 2:
        year = f'20{year}'
    if len(day) == 1:
        day = f'0{day}'
    if len(month) == 1:
        month = f'0{month}'
    
    # Read the market data CSV
    market_data_path = f"2MarketData/Data_{month}-{day}-{year}.csv"
    try:
        market_df = pd.read_csv(market_data_path)
        
        # Convert time columns to datetime for easier comparison
        market_df['Time'] = pd.to_datetime(market_df['Time'], format='%H:%M:%S').dt.time
        df['Entry Time'] = pd.to_datetime(df['Entry Time'], format='%H:%M:%S').dt.time
        df['Exit Time'] = pd.to_datetime(df['Exit Time'], format='%H:%M:%S').dt.time
        
        # Process each trade
        for idx, trade in df.iterrows():
            symbol = trade['Symbol']
            entry_time = trade['Entry Time']
            exit_time = trade['Exit Time']
            entry_price = trade['Entry Price']
            trade_type = trade['Trade Type']
            
            # Get price data for this symbol between entry and exit times
            symbol_data = market_df[
                (market_df['Symbol'] == symbol) & 
                (market_df['Time'] >= entry_time) & 
                (market_df['Time'] <= exit_time)
            ].copy()
            
            if len(symbol_data) == 0:
                print(f"Warning: No price data found for {symbol} between {entry_time} and {exit_time}")
                continue
            
            # Sort by time to ensure chronological order
            symbol_data = symbol_data.sort_values('Time')
            
            # For each target/stop loss combination
            for target in target_values:
                for stop_loss in stop_loss_values:
                    column_name = f"{target}T&{stop_loss}SL"
                    target_percent = target
                    stop_loss_percent = -stop_loss  # Make stop loss negative
                    
                    # Calculate price targets
                    if trade_type == "BUY":
                        target_price = entry_price * (1 + target_percent/100)
                        stop_loss_price = entry_price * (1 + stop_loss_percent/100)
                    else:  # SHORT trade
                        target_price = entry_price * (1 - target_percent/100)
                        stop_loss_price = entry_price * (1 - stop_loss_percent/100)
                    
                    # Check each price point to see which was hit first
                    hit_target = False
                    hit_stop_loss = False
                    
                    for price in symbol_data['Last']:
                        if trade_type == "BUY":
                            if price >= target_price:
                                hit_target = True
                                break
                            if price <= stop_loss_price:
                                hit_stop_loss = True
                                break
                        else:  # SHORT trade
                            if price <= target_price:
                                hit_target = True
                                break
                            if price >= stop_loss_price:
                                hit_stop_loss = True
                                break
                    
                    # Record the result
                    if hit_target:
                        df.at[idx, column_name] = target_percent
                    elif hit_stop_loss:
                        df.at[idx, column_name] = stop_loss_percent
                    else:
                        df.at[idx, column_name] = "NEITHER"

        # Save the resulting dataframe to a CSV file
        output_filename = f"3SummarizedTrades/Summary-{date_str}.csv"
        df.to_csv(output_filename, index=False)
        print(f"Saved trade summary to {output_filename}")

        # Move the processed file to done directory
        MoveProcessedFile(date_str, todoDir, doneDir)
        
    except FileNotFoundError:
        print(f"Warning: Market data file {market_data_path} not found")
    except Exception as e:
        print(f"Error processing market data: {str(e)}")
    
    return df













