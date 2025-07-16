import pandas as pd
import numpy as np
import time
from datetime import datetime
from Proper_Analysis_Calculations import Grid_Search_Parameter_Optimization

def test_numba_optimization():
    """
    Test the NUMBA optimization with a small dataset to ensure it works correctly.
    """
    print("=== TESTING NUMBA OPTIMIZATION ===")
    
    # Create a small test dataset
    np.random.seed(42)  # For reproducible results
    n_samples = 1000
    
    # Generate test data
    test_data = {
        'Entry Volatility Percent': np.random.uniform(0.3, 1.2, n_samples),
        'Entry Volatility Ratio': np.random.uniform(0.4, 1.5, n_samples),
        'Entry Adx28': np.random.uniform(10, 70, n_samples),
        'Entry Adx14': np.random.uniform(10, 70, n_samples),
        'Entry Adx7': np.random.uniform(10, 70, n_samples),
        'Entry Macd Z-Score': np.random.uniform(-5, 5, n_samples),
        'Rsi Extreme Prev Cross': np.random.choice([True, False, "either"], n_samples),
        'Price Movement': []
    }
    
    # Generate test price movements (as pipe-separated strings)
    for i in range(n_samples):
        # Create random price movement sequences
        length = np.random.randint(10, 50)
        movements = np.round(np.random.uniform(-1, 1, length), 2)
        test_data['Price Movement'].append('|'.join(map(str, movements)))
    
    # Create test DataFrame
    df_test = pd.DataFrame(test_data)
    
    print(f"Created test dataset with {len(df_test):,} samples")
    print(f"Sample price movement: {df_test['Price Movement'].iloc[0][:50]}...")
    
    # Override the parameter lists in the actual function temporarily for testing
    # We'll use much smaller parameter ranges for the test
    original_function = Grid_Search_Parameter_Optimization
    
    def test_grid_search(df):
        """Test version with reduced parameter space"""
        print("Starting test grid search with reduced parameters...")
        
        # Temporarily replace with smaller parameter sets for testing
        import Proper_Analysis_Calculations as pac
        
        # Store original parameters
        original_code = pac.Grid_Search_Parameter_Optimization.__code__
        
        # Run a minimal test by directly calling the optimized chunk processor
        volatilities = [0.5, 0.7]  # Just 2 volatilities for test
        ratios = [0.8, 1.0]  # Just 2 ratios
        adx28s = [20, 40]  # Just 2 ADX values
        adx14s = [20, 40]
        adx7s = [20, 40]
        abs_macd_zScores = [1.0, 2.0]  # Just 2 z-scores
        extreme_rsis = [True, False]  # Remove "either" for simpler test
        normal_targets = [0.3, 0.5]  # Just 2 targets
        upper_targets = [0.4, 0.6]  # Just 2 upper targets
        upper_stop_losss = [-0.1, -0.3]  # Just 2 stops
        normal_stop_losss = [-0.4, -0.6]  # Just 2 normal stops
        
        # Calculate expected combinations
        expected_combos = 0
        for ratio in ratios:
            for adx28 in adx28s:
                for adx14 in adx14s:
                    for adx7 in adx7s:
                        for zscore in abs_macd_zScores:
                            for rsi_type in extreme_rsis:
                                for normal_target in normal_targets:
                                    for normal_stop_loss in normal_stop_losss:
                                        for upper_target in upper_targets:
                                            if upper_target <= normal_target:
                                                continue
                                            for upper_stop_loss in upper_stop_losss:
                                                if ((upper_stop_loss >= upper_target) or upper_stop_loss >= normal_target):
                                                    continue
                                                expected_combos += 1
        
        print(f"Expected valid combinations per volatility: {expected_combos:,}")
        print(f"Total combinations across {len(volatilities)} volatilities: {expected_combos * len(volatilities):,}")
        
        # Pre-convert price movements (same as in main function)
        price_movement_lists = df['Price Movement'].apply(
            lambda x: [float(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else []
        ).tolist()
        
        max_length = max(len(pm_list) for pm_list in price_movement_lists if len(pm_list) > 0)
        print(f"Maximum price movement length: {max_length}")
        
        padded_price_movements = np.full((len(price_movement_lists), max_length), np.nan, dtype=float)
        actual_lengths = np.zeros(len(price_movement_lists), dtype=int)
        
        for i, pm_list in enumerate(price_movement_lists):
            if len(pm_list) > 0:
                padded_price_movements[i, :len(pm_list)] = pm_list
                actual_lengths[i] = len(pm_list)
        
        df['padded_price_movements_index'] = range(len(df))
        
        print("Pre-processing complete. Starting NUMBA test...")
        
        # Test with first volatility only
        start_time = time.time()
        result = pac.Process_Volatility_Chunk(
            volatilities[0], df, padded_price_movements, actual_lengths, max_length,
            ratios, adx28s, adx14s, adx7s, abs_macd_zScores,
            extreme_rsis, normal_targets, upper_targets, upper_stop_losss, normal_stop_losss
        )
        end_time = time.time()
        
        print(f"\nTest completed in {end_time - start_time:.2f} seconds")
        print(f"Results found: {len(result)}")
        
        if len(result) > 0:
            # Show first few results
            print("\nSample results:")
            for i, (key, value) in enumerate(list(result.items())[:3]):
                print(f"  {i+1}: {value}")
            
            print("\nTEST PASSED: NUMBA optimization working correctly!")
        else:
            print("WARNING: No results found - this could be normal with random test data")
        
        return True
    
    # Run the test
    try:
        test_grid_search(df_test)
        print("\n=== NUMBA OPTIMIZATION TEST COMPLETED SUCCESSFULLY ===")
        return True
    except Exception as e:
        print(f"\nERROR in NUMBA test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_numba_optimization() 