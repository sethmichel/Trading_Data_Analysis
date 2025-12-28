import inspect
import psycopg2
import sys
import os
import time
import pandas as pd
from datetime import datetime
import Main_Globals
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SECRETS.DONOTOPEN import DB_SECRETS
fileName = os.path.basename(inspect.getfile(inspect.currentframe()))

conn = None # db connection


# backup in case kafka/trading system don't upload the data
def Bulk_Upload_Data_Manually():
    function_name = inspect.currentframe().f_code.co_name
    # include leading 0's
    dates_to_upload = ['07-30-2025','08-04-2025','08-25-2025','09-22-2025','09-23-2025',
                       '10-01-2025','10-02-2025','10-03-2025','10-06-2025','10-07-2025','10-08-2025',
                       '10-09-2025','10-13-2025','10-17-2025','10-20-2025','10-21-2025','10-24-2025',
                       '10-27-2025','10-28-2025','10-30-2025','01-31-2025'] 
    file_type = 'trade data'
    if (file_type == 'trade data'):
        source_dir = "Data_Files/Manual_Trade_Logs/Approved_Cleaned_Manual_Trade_Data"
    else:
        source_dir = "Data_Files/Market_Data/Approved_Cleaned_Market_Data"
    
    for date in dates_to_upload:
        try:
            # Find the CSV file for this date
            csv_filename = None
            for filename in os.listdir(source_dir):
                if filename.endswith('.csv'):
                    # For trade data: MM-DD-YYYY-TradeActivity.csv or MM-DD-YYYY-TradeActivity_On_Demand.csv
                    # For market data: MM-DD-YYYY_*.csv
                    if file_type == 'trade data':
                        file_date = filename.split('-TradeActivity')[0]
                    else:
                        file_date = filename.split('_')[0]
                    
                    if file_date == date:
                        csv_filename = filename
                        break
            
            if not csv_filename:
                msg = f"No CSV file found for date {date} in {source_dir}"
                Main_Globals.logger.warning(f"{fileName} - {function_name}() - {msg}")
                print(msg)
                continue
            
            csv_path = os.path.join(source_dir, csv_filename)
            msg = f"{fileName} - {function_name}() - manually uploading file to db: {csv_filename}"
            Main_Globals.logger.info(msg)
            print(msg)
            df = pd.read_csv(csv_path)

            # Convert column names to lowercase
            df.columns = df.columns.str.lower() 
            
            if file_type == 'market data':
                # Drop the 'early morning atr warmup fix' column if it exists
                if 'early morning atr warmup fix' in df.columns:
                    df = df.drop(columns=['early morning atr warmup fix'])
                
                date_parts = date.split('-')
                month, day, year = date_parts[0], date_parts[1], date_parts[2]
                
                # Convert 'time' column to datetime format M-D-YYYY HH:MM:SS
                df['date_time'] = df['time'].apply(
                    lambda t: datetime.strptime(f"{year}-{month.zfill(2)}-{day.zfill(2)} {t}", "%Y-%m-%d %H:%M:%S")
                )
                
                df = df.drop(columns=['time'])  # Drop the original 'time' column
                
                # Reorder columns to match database schema
                column_order = [
                    'ticker', 'price', 'val', 'avg', 'atr14', 'atr28', 
                    'rsi', 'volume', 'adx28', 'adx14', 'adx7', 
                    'volatility percent', 'volatility ratio', 'date_time'
                ]
                
                # Ensure all expected columns exist
                missing_cols = [col for col in column_order if col not in df.columns]
                if missing_cols:
                    msg = f"Missing columns in CSV. aborting upload: {missing_cols}"
                    Main_Globals.logger.warning(f"{fileName} - {function_name}() - {msg}")
                    continue
                
                df = df[column_order]
                
                # Rename columns to match database 
                df = df.rename(columns={
                    'val': 'macd_val',
                    'avg': 'macd_avg',
                    'volatility percent': 'volatility_percent',
                    'volatility ratio': 'volatility_ratio'
                })
                
                # Prepare the INSERT statement
                insert_query = """
                    INSERT INTO Raw_Market_Data_1 (
                        ticker, price, macd_val, macd_avg, atr14, atr28, 
                        rsi, volume, adx28, adx14, adx7, 
                        volatility_percent, volatility_ratio, date_time
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, date_time) DO NOTHING
                """
            
            elif file_type == 'trade data':
                # Convert 'exec time' from MM-DD-YYYY HH:MM:SS to YYYY-MM-DD HH:MM:SS
                df['exec_time'] = df['exec time'].apply(
                    lambda t: datetime.strptime(t, "%m-%d-%Y %H:%M:%S")
                )
                
                # Rename columns to match database
                df = df.rename(columns={
                    'pos effect': 'pos_effect',
                    'exp': 'trade_exp',
                    'type': 'trade_type',
                    'net price': 'net_price',
                    'price improvement': 'price_improvement',
                    'order type': 'order_type'
                })
                
                # Reorder columns to match database schema
                column_order = [
                    'exec_time', 'spread', 'side', 'qty', 'pos_effect', 
                    'symbol', 'trade_exp', 'strike', 'trade_type', 'price', 
                    'net_price', 'price_improvement', 'order_type'
                ]
                
                # Ensure all expected columns exist
                missing_cols = [col for col in column_order if col not in df.columns]
                if missing_cols:
                    msg = f"Missing columns in CSV. aborting upload: {missing_cols}"
                    Main_Globals.logger.warning(f"{fileName} - {function_name}() - {msg}")
                    continue
                
                df = df[column_order]
                
                # Replace empty strings with None for trade_exp and strike
                df['trade_exp'] = df['trade_exp'].replace('', None)
                df['strike'] = df['strike'].replace('', None)
                
                # Prepare the INSERT statement
                insert_query = """
                    INSERT INTO raw_trades_data_1 (
                        exec_time, spread, side, qty, pos_effect, 
                        symbol, trade_exp, strike, trade_type, price, 
                        net_price, price_improvement, order_type
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (exec_time, symbol) DO NOTHING
                """
            
            # Insert data into database using cursor
            cursor = conn.cursor()
            
            # Convert dataframe to list of tuples for batch insert
            data_tuples = [tuple(row) for row in df.to_numpy()]
            
            # Execute batch insert
            cursor.executemany(insert_query, data_tuples)
            conn.commit()
            
            Main_Globals.logger.info(f"{fileName} - {function_name}() - Successfully uploaded {len(data_tuples)} rows for date {date}")
            
            cursor.close()
            
        except Exception as e:
            msg = f"Error processing date {date}: {e}"
            Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
            if conn:
                conn.rollback()
            continue


# update db with the new validation status of the market data files
# df columns: filename,filepath,status,error info,date checked
def Upload_Market_Data_Validation_Df(market_data_validation_df):
    function_name = inspect.currentframe().f_code.co_name
    
    try:
        cursor = conn.cursor()
        
        # Process each row in the validation dataframe
        for _, row in market_data_validation_df.iterrows():
            # Extract date from filename (format: MM-DD-YYYY_...)
            # Convert from MM-DD-YYYY to YYYY-MM-DD
            date_str = row['filename'].split('_')[0]
            date_parts = date_str.split('-')
            month, day, year = date_parts[0], date_parts[1], date_parts[2]
            data_date = f"{year}-{month}-{day}"
            
            # Count rows in the data file if filepath exists and status is not error
            data_rows = None
            if pd.notna(row.get('filepath')) and row['status'] != 'error':
                try:
                    temp_df = pd.read_csv(row['filepath'])
                    data_rows = len(temp_df)
                except:
                    data_rows = None
            
            # Prepare the INSERT statement with ON CONFLICT UPDATE
            insert_query = """
                INSERT INTO market_data_validation (
                    data_date, filename, data_rows, status, error_info, date_checked
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (data_date) 
                DO UPDATE SET
                    filename = EXCLUDED.filename,
                    data_rows = EXCLUDED.data_rows,
                    status = EXCLUDED.status,
                    error_info = EXCLUDED.error_info,
                    date_checked = EXCLUDED.date_checked
            """
            
            # Execute the insert
            cursor.execute(insert_query, (
                data_date,
                row['filename'],
                data_rows,
                row['status'],
                row.get('error info', None),
                row['date checked']
            ))
        
        conn.commit()
        cursor.close()
        
        msg = f"Successfully uploaded {len(market_data_validation_df)} validation records to database"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
        
    except Exception as e:
        msg = f"Error uploading market data validation: {e}"
        Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
        if conn:
            conn.rollback()
        raise


# download the market data validation table from the database
# returns df with columns: filename,filepath,status,error info,date checked
def Download_Market_Data_Validation_Df():
    function_name = inspect.currentframe().f_code.co_name
    
    try:
        cursor = conn.cursor()
        
        # Query to get all records from market_data_validation table
        query = """
            SELECT data_date, filename, data_rows, status, error_info, date_checked
            FROM market_data_validation
            ORDER BY data_date DESC
        """
        
        cursor.execute(query)
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Get column names from cursor description
        columns = [desc[0] for desc in cursor.description]
        
        # Create dataframe
        df = pd.DataFrame(results, columns=columns)
        
        # Rename columns to match expected format
        # Note: filepath is not stored in DB, will be None
        df = df.rename(columns={
            'data_date': 'data_date',  # Keep for potential use
            'error_info': 'error info',
            'date_checked': 'date checked'
        })
        
        # Add filepath column (not stored in database)
        df['filepath'] = None
        
        # Reorder columns to match expected format
        df = df[['filename', 'filepath', 'status', 'error info', 'date checked']]
        
        cursor.close()
        
        msg = f"Successfully downloaded {len(df)} validation records from database"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
        
        return df
        
    except Exception as e:
        msg = f"Error downloading market data validation: {e}"
        Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
        raise


# update db with the new validation status of the trade data files
# df columns: filename,filepath,status,error info,date checked
def Upload_Trade_Data_Validation_Df(trade_data_validation_df):
    function_name = inspect.currentframe().f_code.co_name
    
    try:
        cursor = conn.cursor()
        
        # Process each row in the validation dataframe
        for _, row in trade_data_validation_df.iterrows():
            # Extract date from filename (format: MM-DD-YYYY-TradeActivity*.csv)
            # Convert from MM-DD-YYYY to YYYY-MM-DD
            date_str = row['filename'].split('-TradeActivity')[0]
            date_parts = date_str.split('-')
            month, day, year = date_parts[0], date_parts[1], date_parts[2]
            data_date = f"{year}-{month}-{day}"
            
            # Count rows in the data file if filepath exists and status is not error
            data_rows = None
            if pd.notna(row.get('filepath')) and row['status'] != 'error':
                try:
                    temp_df = pd.read_csv(row['filepath'])
                    data_rows = len(temp_df)
                except:
                    data_rows = None
            
            # Prepare the INSERT statement with ON CONFLICT UPDATE
            insert_query = """
                INSERT INTO real_trades_validation (
                    data_date, filename, data_rows, status, error_info, date_checked
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (data_date) 
                DO UPDATE SET
                    filename = EXCLUDED.filename,
                    data_rows = EXCLUDED.data_rows,
                    status = EXCLUDED.status,
                    error_info = EXCLUDED.error_info,
                    date_checked = EXCLUDED.date_checked
            """
            
            # Execute the insert
            cursor.execute(insert_query, (
                data_date,
                row['filename'],
                data_rows,
                row['status'],
                row.get('error info', None),
                row['date checked']
            ))
        
        conn.commit()
        cursor.close()
        
        msg = f"Successfully uploaded {len(trade_data_validation_df)} validation records to database"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
        
    except Exception as e:
        msg = f"Error uploading trade data validation: {e}"
        Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
        if conn:
            conn.rollback()
        raise


# download the trade data validation table from the database
# returns df with columns: filename,filepath,status,error info,date checked
def Download_Trade_Data_Validation_Df():
    function_name = inspect.currentframe().f_code.co_name
    
    try:
        cursor = conn.cursor()
        
        # Query to get all records from trade_data_validation table
        query = """
            SELECT data_date, filename, data_rows, status, error_info, date_checked
            FROM real_trades_validation
            ORDER BY data_date DESC
        """
        
        cursor.execute(query)
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Get column names from cursor description
        columns = [desc[0] for desc in cursor.description]
        
        # Create dataframe
        df = pd.DataFrame(results, columns=columns)
        
        # Rename columns to match expected format
        # Note: filepath is not stored in DB, will be None
        df = df.rename(columns={
            'data_date': 'data_date',  # Keep for potential use
            'error_info': 'error info',
            'date_checked': 'date checked'
        })
        
        # Add filepath column (not stored in database)
        df['filepath'] = None
        
        # Reorder columns to match expected format
        df = df[['filename', 'filepath', 'status', 'error info', 'date checked']]
        
        cursor.close()
        
        msg = f"Successfully downloaded {len(df)} validation records from database"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
        
        return df
        
    except Exception as e:
        msg = f"Error downloading trade data validation: {e}"
        Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
        raise


def Upload_Trade_Summaries_From_CSV():
    """
    Upload trade summaries from bulk_summaries.csv to the trade_summaries table.
    Only inserts new records (skips existing primary keys).
    Converts date format from MM-DD-YY to YYYY-MM-DD.
    Converts column names from Title Case to lowercase_with_underscores.
    """
    function_name = inspect.currentframe().f_code.co_name
    
    try:
        # Read the CSV file
        csv_path = "Data_Files/Trade_Summaries/Summary_Csv_Files/bulk_summaries.csv"
        df = pd.read_csv(csv_path)
        
        msg = f"Read {len(df)} rows from {csv_path}"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
        
        # Convert column names to lowercase with underscores
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert date from MM-DD-YY to YYYY-MM-DD format
        df['date'] = pd.to_datetime(df['date'], format='%m-%d-%y').dt.strftime('%Y-%m-%d')
        
        # Rename 'date' column to 'trade_date' to match the database schema
        df.rename(columns={'date': 'trade_date'}, inplace=True)
        
        # Prepare the INSERT statement with ON CONFLICT DO NOTHING
        # This will skip rows where the primary key (trade_date, entry_time) already exists
        insert_query = """
            INSERT INTO trade_summaries (
                trade_date, trade_id, ticker, entry_time, exit_time, time_in_trade,
                dollar_change, running_percent_by_ticker, running_percent_all,
                total_investment, entry_price, exit_price, trade_type, qty,
                best_exit_price, best_exit_time_in_trade, worst_exit_price,
                worst_exit_percent, worst_exit_time_in_trade, entry_atr14,
                entry_atr28, entry_volatility_percent, entry_volatility_ratio,
                entry_adx28, entry_adx14, entry_adx7, trade_holding_reached,
                trade_best_exit_percent, trade_percent_change
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (trade_date, entry_time) DO NOTHING
        """
        
        # Insert data into database using cursor
        cursor = conn.cursor()
        
        # Convert dataframe to list of tuples for batch insert
        data_tuples = [tuple(row) for row in df.to_numpy()]
        
        # Execute batch insert
        cursor.executemany(insert_query, data_tuples)
        
        # Get the number of rows actually inserted
        rows_inserted = cursor.rowcount
        
        conn.commit()
        cursor.close()
        
        msg = f"Successfully processed {len(data_tuples)} rows. Inserted {rows_inserted} new rows (skipped {len(data_tuples) - rows_inserted} duplicates)"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
        
        return True
        
    except Exception as e:
        conn.rollback()
        msg = f"Error uploading trade summaries: {e}"
        Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
        print(msg)
        return False


def Download_Trade_Summaries_To_Dataframe():
    """
    Download all trade summaries from the trade_summaries table and return as a DataFrame.
    Converts column names back to Title Case format matching the original CSV.
    Converts trade_date back to 'Date' column.
    """
    function_name = inspect.currentframe().f_code.co_name
    
    try:
        cursor = conn.cursor()
        
        # Query to get all records from trade_summaries table
        query = """
            SELECT 
                trade_date, trade_id, ticker, entry_time, exit_time, time_in_trade,
                dollar_change, running_percent_by_ticker, running_percent_all,
                total_investment, entry_price, exit_price, trade_type, qty,
                best_exit_price, best_exit_time_in_trade, worst_exit_price,
                worst_exit_percent, worst_exit_time_in_trade, entry_atr14,
                entry_atr28, entry_volatility_percent, entry_volatility_ratio,
                entry_adx28, entry_adx14, entry_adx7, trade_holding_reached,
                trade_best_exit_percent, trade_percent_change
            FROM trade_summaries
            ORDER BY trade_date, entry_time
        """
        
        cursor.execute(query)
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Get column names from cursor description
        columns = [desc[0] for desc in cursor.description]
        
        cursor.close()
        
        # Create DataFrame
        df = pd.DataFrame(results, columns=columns)
        
        msg = f"Downloaded {len(df)} rows from trade_summaries table"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
        
        # Convert column names back to Title Case with spaces (matching original CSV format)
        # Map from database column names to CSV column names
        column_mapping = {
            'trade_date': 'Date',
            'trade_id': 'Trade Id',
            'ticker': 'Ticker',
            'entry_time': 'Entry Time',
            'exit_time': 'Exit Time',
            'time_in_trade': 'Time in Trade',
            'dollar_change': 'Dollar Change',
            'running_percent_by_ticker': 'Running Percent By Ticker',
            'running_percent_all': 'Running Percent All',
            'total_investment': 'Total Investment',
            'entry_price': 'Entry Price',
            'exit_price': 'Exit Price',
            'trade_type': 'Trade Type',
            'qty': 'Qty',
            'best_exit_price': 'Best Exit Price',
            'best_exit_time_in_trade': 'Best Exit Time In Trade',
            'worst_exit_price': 'Worst Exit Price',
            'worst_exit_percent': 'Worst Exit Percent',
            'worst_exit_time_in_trade': 'Worst Exit Time In Trade',
            'entry_atr14': 'Entry Atr14',
            'entry_atr28': 'Entry Atr28',
            'entry_volatility_percent': 'Entry Volatility Percent',
            'entry_volatility_ratio': 'Entry Volatility Ratio',
            'entry_adx28': 'Entry Adx28',
            'entry_adx14': 'Entry Adx14',
            'entry_adx7': 'Entry Adx7',
            'trade_holding_reached': 'Trade Holding Reached',
            'trade_best_exit_percent': 'Trade Best Exit Percent',
            'trade_percent_change': 'Trade Percent Change'
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        return df
        
    except Exception as e:
        msg = f"Error downloading trade summaries: {e}"
        Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
        print(msg)
        return None


def Get_Model_Diagnostics_Column_Names():
    """
    Query the database to get all column names from model_diagnostics table,
    excluding the metadata columns (model_name, version_id, created, features).
    """
    function_name = inspect.currentframe().f_code.co_name
    
    try:
        cursor = conn.cursor()
        
        # Query to get column names from the table
        query = """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'model_diagnostics' 
            AND column_name NOT IN ('model_name', 'version_id', 'created', 'features')
            ORDER BY ordinal_position
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        
        # Extract column names from results
        column_names = [row[0] for row in results]
        
        msg = f"Retrieved {len(column_names)} column names from model_diagnostics table"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        
        return column_names
        
    except Exception as e:
        msg = f"Error getting column names from model_diagnostics table: {e}"
        Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
        raise


def Find_New_Model_Versions(model_names):
    function_name = inspect.currentframe().f_code.co_name
    cursor = conn.cursor()
    
    # Get current max version_id for each model
    version_ids = {}
    for model_key, model_name in model_names.items():
        cursor.execute("""
            SELECT MAX(version_id) 
            FROM model_diagnostics 
            WHERE model_name = %s
        """, (model_name,))
        
        result = cursor.fetchone()
        max_version = result[0] if result[0] is not None else 0.09
        
        # Increment by 0.01
        new_version = round(max_version + 0.01, 2)
        version_ids[model_key] = new_version
        
        msg = f"Model '{model_name}': previous version = {max_version}, new version = {new_version}"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
    
    cursor.close()
    
    msg = "Version IDs determined for all models"
    Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
    print(msg)

    return version_ids


def Upload_Model_Diagnostics(new_version_ids, success_prob_response_distribution_results, success_prob_diagnostics_results, 
             success_prob_trade_diagnostic_results,sl_response_distribution_results, sl_diagnostics_results,
             sl_trade_diagnostic_results,target_response_distribution_results, target_diagnostics_results,
             target_trade_diagnostic_results):
    """
    Upload model diagnostics for all three models (success_prob, sl, target).
    Automatically increments version_id by 0.01 for each model based on their current max version.
    Each model gets its own row in the table.
    """
    try:
        function_name = inspect.currentframe().f_code.co_name
        today = datetime.now().date()
        model_names = {
            'success_prob': 'success probability',
            'sl': 'stop loss',
            'target': 'target'
        }
        model_features = {
            'success probability': 2,
            'stop loss': 2,
            'target': 2
        }
        
        all_columns = Get_Model_Diagnostics_Column_Names()
        
        # Organize all diagnostic data by model
        model_data = {
            'success_prob': {
                'response_dist': success_prob_response_distribution_results,
                'diagnostics': success_prob_diagnostics_results,
                'trade_diagnostics': success_prob_trade_diagnostic_results
            },
            'sl': {
                'response_dist': sl_response_distribution_results,
                'diagnostics': sl_diagnostics_results,
                'trade_diagnostics': sl_trade_diagnostic_results
            },
            'target': {
                'response_dist': target_response_distribution_results,
                'diagnostics': target_diagnostics_results,
                'trade_diagnostics': target_trade_diagnostic_results
            }
        }
        
        cursor = conn.cursor()
        rows_inserted = 0
        
        # Insert one row for each model
        for model_key, model_full_name in model_names.items():
            # Create a dictionary to hold all column values (initialized to None)
            column_values = {col: None for col in all_columns}
            
            # Get data for this specific model
            data = model_data[model_key]
            
            # 1. Process response distribution results
            # Column names like: response_dist_count, response_dist_min, response_dist_p1, etc.
            if data['response_dist']:
                for key, value in data['response_dist'].items():
                    if key == 'percentiles' and isinstance(value, dict):
                        # Handle nested percentiles
                        for percentile_key, percentile_value in value.items():
                            column_name = f"response_dist_{percentile_key}"
                            if column_name in column_values:
                                column_values[column_name] = percentile_value
                    else:
                        column_name = f"response_dist_{key}"
                        if column_name in column_values:
                            column_values[column_name] = value
            
            # 2. Process diagnostics results (all_samples and per_trade)
            # Column names like: all_samples_data_points_count, per_trade_trades_count, etc.
            if data['diagnostics']:
                if 'all_samples' in data['diagnostics']:
                    for key, value in data['diagnostics']['all_samples'].items():
                        column_name = f"all_samples_{key}"
                        if column_name in column_values:
                            column_values[column_name] = value
                
                if 'per_trade' in data['diagnostics']:
                    for key, value in data['diagnostics']['per_trade'].items():
                        column_name = f"per_trade_{key}"
                        if column_name in column_values:
                            column_values[column_name] = value
            
            # 3. Process trade diagnostic results
            # Column names like: trade_test_overall_roi_total, etc.
            if data['trade_diagnostics']:
                for key, value in data['trade_diagnostics'].items():
                    column_name = f"trade_test_{key}"
                    if column_name in column_values:
                        column_values[column_name] = value
            
            # Build the INSERT statement
            # Columns: model_name, version_id, created, features, + all diagnostic columns
            insert_columns = ['model_name', 'version_id', 'created', 'features'] + all_columns
            placeholders = ', '.join(['%s'] * len(insert_columns))
            columns_str = ', '.join(insert_columns)
            
            insert_query = f"""
                INSERT INTO model_diagnostics ({columns_str})
                VALUES ({placeholders})
            """
            
            # Build the values tuple
            values = [
                model_full_name,
                new_version_ids[model_key],
                today,
                model_features[model_full_name]
            ] + [column_values[col] for col in all_columns]
            
            # Execute the insert
            cursor.execute(insert_query, values)
            rows_inserted += 1
            
            msg = f"Inserted diagnostics for model '{model_full_name}' version {new_version_ids[model_key]}"
            Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
            print(msg)
        
        conn.commit()
        cursor.close()
        
        msg = f"Successfully uploaded diagnostics for {rows_inserted} models"
        Main_Globals.logger.info(f"{fileName} - {function_name}() - {msg}")
        print(msg)
        
        return True
        
    except Exception as e:
        conn.rollback()
        msg = f"Error uploading model diagnostics: {e}"
        Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
        print(msg)
        return False


# connect to and return a db connection, retrying x number of times
def Connect_To_Db():
    global conn

    function_name = inspect.currentframe().f_code.co_name
    retry_attempts = 5
    
    for i in range(retry_attempts):
        try:
            conn = psycopg2.connect(
                dbname='Market_Data',
                user='postgres',
                password=DB_SECRETS.password,
                host=DB_SECRETS.host_computer_address, # your PostgreSQL server IP
                port=5432
            )
            conn.autocommit = False
            msg = "Connected to PostgreSQL"
            Main_Globals.logger.info(msg)
            print(msg)

            return True

        # error but retry and don't notify user
        except Exception as e:
            msg = f"Failed to connect to PostgreSQL, attempt {i+1}/{retry_attempts}: {e}"
            Main_Globals.logger.error(f"{fileName} - {function_name}() - Line {sys.exc_info()[2].tb_lineno}: {msg}")
            
            # Only sleep if we're going to retry again
            if i < retry_attempts - 1:
                time.sleep(3)

    # If we get here, all retries failed - notify user
    msg = f"unable to connect to db in {retry_attempts} tries. Please do something about it."
    Main_Globals.ErrorHandler(fileName, function_name, msg, None)
    return None

