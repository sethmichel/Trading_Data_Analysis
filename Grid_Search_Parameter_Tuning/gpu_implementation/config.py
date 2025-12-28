import os

# File Paths
DATA_DIR = 'data'
MARKET_DATA_PATH = os.path.join(DATA_DIR, 'market_data.parquet')
TRADES_PATH = os.path.join(DATA_DIR, 'trades.parquet')
OUTPUT_FILE = 'top_strategies.csv'

# GPU Settings
THREADS_PER_BLOCK = 256  # Standard CUDA block size
# Batch size for strategy combinations to avoid TDR timeouts on Windows and memory limits
# critical when scaling to billions.
BATCH_SIZE = 1_000_000 

# Strategy Parameters (Grid Search Space)
# These define the possible values for each parameter.
PARAM_RANGES = {
    'volatilities': [0.0, 0.3, 0.4, 0.5, 0.6],           # Entry Volatility Percent
    'ratios':       [0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0], # Entry Volatility Ratio
    'adx28s':       [0, 20, 30],
    'adx14s':       [0, 20, 30],
    'adx7s':        [0, 20, 30],
    'targets':      [0.2, 0.3, 0.4, 0.5],            # Target ROI %
    'stop_losses':  [-0.5, -0.4, -0.3]               # Stop Loss %
}

# The order of parameters expected by the kernel
PARAM_NAMES = [
    'volatility', 'ratio', 'adx28', 'adx14', 'adx7', 'target', 'stop_loss'
]

