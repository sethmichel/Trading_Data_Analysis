from numba import cuda
import math

@cuda.jit
def simulate_strategies_kernel(
    market_prices,
    trade_entry_idxs,
    trade_end_idxs,
    trade_entry_prices,
    trade_volatilities,
    trade_ratios,
    trade_adx28s,
    trade_adx14s,
    trade_adx7s,
    strategy_params,
    results
):
    """
    GPU Kernel to simulate trading strategies.
    
    Grid: 1D grid where each thread represents ONE strategy combination.
    Loop: Each thread iterates through ALL trades.
    
    strategy_params: [N_strategies, 7]
        Cols: 0:volatility, 1:ratio, 2:adx28, 3:adx14, 4:adx7, 5:target, 6:stop_loss
    """
    
    # Global thread index = Strategy Index
    strat_idx = cuda.grid(1)
    
    if strat_idx < strategy_params.shape[0]:
        
        # Load Strategy Parameters into local registers
        param_vol = strategy_params[strat_idx, 0]
        param_ratio = strategy_params[strat_idx, 1]
        param_adx28 = strategy_params[strat_idx, 2]
        param_adx14 = strategy_params[strat_idx, 3]
        param_adx7 = strategy_params[strat_idx, 4]
        param_target = strategy_params[strat_idx, 5]     # e.g., 0.05 for 5%
        param_stop = strategy_params[strat_idx, 6]       # e.g., -0.02 for -2%
        
        # Accumulator for ROI
        # Using summation of percentages (simple return) to avoid float overflow/underflow on large datasets
        total_roi = 0.0
        
        num_trades = trade_entry_idxs.shape[0]
        
        for i in range(num_trades):
            # 1. Filter Check
            # All conditions must be met to take the trade
            # "volatility = 0.5 means we exclude all trades that had volatility < 0.5"
            if trade_volatilities[i] < param_vol:
                continue
            if trade_ratios[i] < param_ratio:
                continue
            if trade_adx28s[i] < param_adx28:
                continue
            if trade_adx14s[i] < param_adx14:
                continue
            if trade_adx7s[i] < param_adx7:
                continue
                
            # 2. Simulate Trade
            start_idx = trade_entry_idxs[i]
            end_idx = trade_end_idxs[i]
            entry_price = trade_entry_prices[i]
            
            # Calculate exit levels
            # Note: param_target is positive, param_stop is negative
            target_price = entry_price * (1.0 + param_target)
            stop_price = entry_price * (1.0 + param_stop)
            
            trade_result = 0.0
            closed = False
            
            # Iterate through market data for this trade's duration
            # All threads access the same market_prices[k] roughly at the same time -> High L2 Cache Hit Rate
            for k in range(start_idx, end_idx + 1):
                current_price = market_prices[k]
                
                # Check Stop Loss (Usually checked first or same tick? We check stop first for safety)
                # Assuming Stop Loss is a hard stop
                if current_price <= stop_price:
                    trade_result = param_stop
                    closed = True
                    break
                    
                # Check Target
                if current_price >= target_price:
                    trade_result = param_target
                    closed = True
                    break
            
            # If trade didn't hit stop or target by end of window
            if not closed:
                exit_price = market_prices[end_idx]
                trade_result = (exit_price - entry_price) / entry_price
                
            # Accumulate
            total_roi += trade_result
            
        # Write result
        results[strat_idx] = total_roi

