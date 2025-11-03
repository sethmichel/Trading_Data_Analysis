import pandas as pd

'''
loop over 1 day of market data, give it trade entry times, it'll tell me risk and gain. basically takes a 40 min proceess into 3 seconds
'''
class Ticker:
    def __init__(self, ticker):
        self.name = ticker
        self.buyOrShort = None                        # str: 'buy' or 'short'
        self.entryPrice = 0
        self.allTradesChangePercent = 0               # change over all trades on this ticker
        self.tradeChange = {"worstExitPrice": None,   # individual trade
                            "bestExitPrice": None,
                            "worstExitPercent": None,   # individual trade
                            "bestExitPercent": None}     
        self.tradeTracker = {}                        #['time']: [worstexitpercent, best exit percent, actual change]}


def CreateDf(csvPath):
    df = pd.read_csv(csvPath)

    return df


def analyze_trades(df, trade_times, change_tracker):
    tickers = {}
    
    # Create Ticker objects for each ticker in trade_times
    for ticker in trade_times:
        tickers[ticker] = Ticker(ticker)
    
    # Process each ticker's trades
    for ticker, trade_data in trade_times.items():
        entry_types = trade_data[0]  # List of 'buy' or 'short'
        entry_times = trade_data[1]  # List of timestamps
        
        print(f"\nProcessing trades for {ticker}")
        print(f"Entry times to process: {entry_times}")
        
        # Convert entry times to datetime for easier comparison
        entry_times_processed = set()  # Keep track of which entry times we've processed
        
        for i, entry_time in enumerate(entry_times):
            print(f"\nAttempting to process trade at {entry_time}")
            
            # Initialize trade tracking
            tickers[ticker].buyOrShort = entry_types[i]
            tickers[ticker].tradeChange = {
                "worstExitPrice": None,
                "bestExitPrice": None,
                "worstExitPercent": None,
                "bestExitPercent": None
            }
            
            # Find entry point
            entry_found = False
            entry_price = None
            entry_idx = None
            
            # Find the entry point in the dataframe
            for idx, row in df.iterrows():
                if row['Symbol'] == ticker and row['Time'].startswith(entry_time):
                    entry_found = True
                    entry_price = row['Last']
                    entry_idx = idx
                    print(f"Found entry point at index {idx} with price {entry_price}")
                    break
            
            if not entry_found:
                print(f"Warning: Could not find entry point for {ticker} at {entry_time}")
                continue
            
            # Track price movements after entry
            trade_active = True
            tickers[ticker].entryPrice = entry_price
            tickers[ticker].tradeChange["worstExitPrice"] = entry_price
            tickers[ticker].tradeChange["bestExitPrice"] = entry_price
            
            # Process all rows after entry point
            for idx, row in df.iloc[entry_idx:].iterrows():
                if not trade_active:
                    break
                    
                if row['Symbol'] == ticker:
                    current_price = row['Last']
                    
                    # Update best/worst prices based on trade direction
                    if tickers[ticker].buyOrShort == 'short':
                        # For shorts, lower price is better (profit), higher price is worse (loss)
                        if current_price < tickers[ticker].tradeChange["bestExitPrice"]:
                            tickers[ticker].tradeChange["bestExitPrice"] = current_price
                        if current_price > tickers[ticker].tradeChange["worstExitPrice"]:
                            tickers[ticker].tradeChange["worstExitPrice"] = current_price
                            
                        # Calculate percent changes for shorts (inverted)
                        best_change = ((entry_price - tickers[ticker].tradeChange["bestExitPrice"]) / entry_price) * 100
                        worst_change = ((entry_price - tickers[ticker].tradeChange["worstExitPrice"]) / entry_price) * 100
                    else:  # buy
                        if current_price > tickers[ticker].tradeChange["bestExitPrice"]:
                            tickers[ticker].tradeChange["bestExitPrice"] = current_price
                        if current_price < tickers[ticker].tradeChange["worstExitPrice"]:
                            tickers[ticker].tradeChange["worstExitPrice"] = current_price
                            
                        # Calculate percent changes for buys
                        best_change = ((tickers[ticker].tradeChange["bestExitPrice"] - entry_price) / entry_price) * 100
                        worst_change = ((tickers[ticker].tradeChange["worstExitPrice"] - entry_price) / entry_price) * 100
                    
                    tickers[ticker].tradeChange["bestExitPercent"] = best_change
                    tickers[ticker].tradeChange["worstExitPercent"] = worst_change
                    
                    # Check for exit conditions
                    if best_change >= change_tracker["target"] or worst_change <= change_tracker["stop loss"]:
                        exit_change = best_change if best_change >= change_tracker["target"] else worst_change
                        tickers[ticker].allTradesChangePercent += exit_change
                        tickers[ticker].tradeTracker[entry_time] = [
                            tickers[ticker].tradeChange["worstExitPercent"],
                            tickers[ticker].tradeChange["bestExitPercent"],
                            exit_change
                        ]
                        print(f"Trade at {entry_time} exited with change {exit_change:.2f}%")
                        trade_active = False
                        break
    
    # Print results
    print("\nTrading Analysis Results:")
    print("=" * 50)
    for ticker_name, ticker_obj in tickers.items():
        print(f"\nTicker: {ticker_name}")
        print(f"Total Change: {ticker_obj.allTradesChangePercent:.2f}%")
        print("\nIndividual Trades:")
        for time, trade_data in ticker_obj.tradeTracker.items():
            print(f"Entry Time: {time}")
            print(f"  Worst Exit: {trade_data[0]:.2f}%")
            print(f"  Best Exit: {trade_data[1]:.2f}%")
            print(f"  Actual Exit: {trade_data[2]:.2f}%")
            print("-" * 30)



marketDataCsv = '2MarketData/Data_04-30-2025.csv'
tradeTimes = {'SOXL': [['buy','short','buy','buy','short','short','short'],
                      ['06:49','07:35','09:36','10:17','10:45','11:24','11:58']]
              }
changeTracker = {"target": 0.3, "stop loss": -0.7}
totalChange = 0

df = CreateDf(marketDataCsv)

# Run the analysis
analyze_trades(df, tradeTimes, changeTracker)
