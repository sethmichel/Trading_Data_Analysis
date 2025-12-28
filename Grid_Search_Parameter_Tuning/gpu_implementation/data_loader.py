import cudf
import cupy as cp
import config
import os

def load_data():
    """
    Loads market data and trades from Parquet files into GPU memory (CuPy arrays).
    Uses DLPack for zero-copy transfer from cuDF to CuPy.
    """
    print("Loading data into VRAM...")
    
    # 1. Load Market Data
    if not os.path.exists(config.MARKET_DATA_PATH):
        raise FileNotFoundError(f"Market data not found at {config.MARKET_DATA_PATH}. Run preprocess.py first.")
        
    print(f"Reading {config.MARKET_DATA_PATH}...")
    market_df = cudf.read_parquet(config.MARKET_DATA_PATH)
    
    # Convert Price column to CuPy array
    # We only need the Price array for the simulation kernel as the "Texture"
    # The indices in trades.parquet already point to the correct locations in this array.
    market_prices = cp.from_dlpack(market_df['Price'].to_dlpack())
    
    # Ensure it's float32
    if market_prices.dtype != cp.float32:
        market_prices = market_prices.astype(cp.float32)
        
    print(f"Market Data Loaded: {len(market_prices)} rows.")
    
    # 2. Load Trades
    if not os.path.exists(config.TRADES_PATH):
        raise FileNotFoundError(f"Trades data not found at {config.TRADES_PATH}. Run preprocess.py first.")

    print(f"Reading {config.TRADES_PATH}...")
    trades_df = cudf.read_parquet(config.TRADES_PATH)
    
    # Extract columns needed for the kernel
    # We use a dictionary to organize them
    
    # Metrics for filtering
    trade_arrays = {}
    
    # Mapping column names to what we'll use in the kernel
    # Note: Ensure dtypes match what Numba expects (float32 for metrics, int32 for indices)
    
    columns_to_load = {
        'Entry Price': 'entry_price',
        'Exit Price': 'exit_price', # Used for fallback exit
        'MarketDataStartIndex': 'start_index',
        'MarketDataEndIndex': 'end_index',
        'Entry Volatility Percent': 'volatility',
        'Entry Volatility Ratio': 'ratio',
        'Entry Adx28': 'adx28',
        'Entry Adx14': 'adx14',
        'Entry Adx7': 'adx7',
        'Trade Percent Change': 'roi_percent', # Original ROI (fallback)
        # 'Worst Exit Percent' might be needed if we want to check if stop loss was hit in original data? 
        # But we are re-simulating, so we check prices directly.
    }
    
    for col_name, key in columns_to_load.items():
        # dlpack transfer
        arr = cp.from_dlpack(trades_df[col_name].to_dlpack())
        trade_arrays[key] = arr
    
    print(f"Trades Loaded: {len(trades_df)} rows.")
    
    return market_prices, trade_arrays

if __name__ == "__main__":
    # Test loader
    import memory
    memory.initialize_memory()
    prices, trades = load_data()
    print("Market Prices shape:", prices.shape)
    print("Trade keys:", trades.keys())
    print("First trade entry price:", trades['entry_price'][0])

