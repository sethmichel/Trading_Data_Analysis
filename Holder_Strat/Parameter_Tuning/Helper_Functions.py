def bulk_csv_date_converter(date):
    parts = date.split('-')

    if (len(parts[0]) == 1):
        parts[0] = f"0{parts[0]}"
    if (len(parts[1]) == 1):
        parts[1] = f"0{parts[1]}"
    if (len(parts[2]) == 2):
        parts[2] = f"20{parts[2]}"
    
    return '-'.join(parts)


def Check_We_Have_Data_For_Trade(market_df, entry_time):
    # Get the final timestamp from market data
    final_market_time = market_df.iloc[-1]['Time']
    
    # Convert times to datetime objects for comparison
    entry_time_obj = datetime.datetime.strptime(entry_time, '%H:%M:%S').time()
    final_market_time_obj = datetime.datetime.strptime(final_market_time, '%H:%M:%S').time()
    
    # Convert to seconds for easier comparison
    entry_seconds = entry_time_obj.hour * 3600 + entry_time_obj.minute * 60 + entry_time_obj.second
    final_market_seconds = final_market_time_obj.hour * 3600 + final_market_time_obj.minute * 60 + final_market_time_obj.second
    
    if entry_seconds > final_market_seconds:
        return False
    else:
        return True
    