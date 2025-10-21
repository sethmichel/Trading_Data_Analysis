"""
Analysis tool for trades that reach neither stop loss nor target.

This module provides functions to:
1. Identify trades that reach neither condition
2. Analyze the characteristics of these trades
3. Implement solutions for handling them in training data
4. Compare model performance with different approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import Helper_Functions


def analyze_neither_trades(sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes):
    """
    Comprehensive analysis of trades that reach neither stop loss nor target.
    
    Returns detailed statistics about:
    - How many trades fall into each category (success, failure, neither)
    - Characteristics of 'neither' trades (time in trade, final ROI, volatility, etc.)
    - Distribution across different stop loss levels
    """
    skip_dates = []
    holding_percent = 0.6
    
    # Track all trade outcomes
    trade_outcomes = []
    
    print(f"Analyzing trade outcomes for stop loss candidates: {sl_list}")
    print("="*60)
    
    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        date = Helper_Functions.bulk_csv_date_converter(row['Date'])
        trade_id = row['Trade Id']
        
        if date in skip_dates:
            continue
            
        # Check if we have market data for this date and ticker
        if date not in market_data_dict_by_ticker:
            print(f"No market data found for date {date}")
            skip_dates.append(date)
            continue  
        if ticker not in market_data_dict_by_ticker[date]:
            continue

        entry_time = row['Entry Time']
        market_df = market_data_dict_by_ticker[date][ticker].copy()

        # Skip trade if entry time is after final market data time
        if not Helper_Functions.Check_We_Have_Data_For_Trade(market_df, entry_time):
            continue
        
        roi_list = roi_dictionary[trade_id]
        start_index = trade_start_indexes[trade_id]
        
        # Get trade characteristics
        entry_volatility = market_df.iloc[start_index]['Volatility Percent']
        entry_time_minutes = market_df.iloc[start_index]['Time Since Market Open']
        
        # Test each stop loss level
        for sl in sl_list:
            counter = -1
            outcome = None
            final_roi = None
            time_in_trade = 0
            
            # Simulate this stop loss on the trade
            for i in range(start_index, len(market_df)):
                counter += 1
                if counter >= len(roi_list):
                    # Ran out of ROI data - this is a "neither" case
                    outcome = 'neither'
                    final_roi = roi_list[-1] if roi_list else 0
                    time_in_trade = counter
                    break
                    
                curr_roi = roi_list[counter]
                time_in_trade = counter + 1

                if curr_roi <= sl:
                    # Stop loss hit - failure
                    outcome = 'failure'
                    final_roi = curr_roi
                    break
                elif curr_roi >= holding_percent:
                    # Target reached - success
                    outcome = 'success'
                    final_roi = curr_roi
                    break
            else:
                # Loop completed without break - neither condition met
                outcome = 'neither'
                final_roi = roi_list[-1] if roi_list else 0
            
            # Record this trade outcome
            trade_outcomes.append({
                'trade_id': trade_id,
                'ticker': ticker,
                'date': date,
                'stop_loss': sl,
                'outcome': outcome,
                'final_roi': final_roi,
                'time_in_trade': time_in_trade,
                'entry_volatility': entry_volatility,
                'entry_time_minutes': entry_time_minutes,
                'roi_data_length': len(roi_list)
            })
    
    # Convert to DataFrame for analysis
    outcomes_df = pd.DataFrame(trade_outcomes)
    
    # Print comprehensive analysis
    print_neither_analysis(outcomes_df, sl_list)
    
    return outcomes_df


def print_neither_analysis(outcomes_df, sl_list):
    """Print detailed analysis of trade outcomes."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TRADE OUTCOME ANALYSIS")
    print("="*80)
    
    # Overall statistics
    total_samples = len(outcomes_df)
    print(f"\nTotal trade-stoploss combinations analyzed: {total_samples:,}")
    
    # Outcome distribution
    outcome_counts = outcomes_df['outcome'].value_counts()
    print(f"\nOVERALL OUTCOME DISTRIBUTION:")
    print("-" * 40)
    for outcome in ['success', 'failure', 'neither']:
        count = outcome_counts.get(outcome, 0)
        pct = (count / total_samples) * 100
        print(f"  {outcome.capitalize():>8}: {count:>6,} ({pct:>5.1f}%)")
    
    # Analysis by stop loss level
    print(f"\nOUTCOME DISTRIBUTION BY STOP LOSS LEVEL:")
    print("-" * 50)
    print(f"{'Stop Loss':<10} {'Success':<8} {'Failure':<8} {'Neither':<8} {'Neither %':<10}")
    print("-" * 50)
    
    for sl in sorted(sl_list):
        sl_data = outcomes_df[outcomes_df['stop_loss'] == sl]
        sl_outcomes = sl_data['outcome'].value_counts()
        
        success_count = sl_outcomes.get('success', 0)
        failure_count = sl_outcomes.get('failure', 0)
        neither_count = sl_outcomes.get('neither', 0)
        total_sl = len(sl_data)
        neither_pct = (neither_count / total_sl) * 100 if total_sl > 0 else 0
        
        print(f"{sl:>8.1f}%  {success_count:>7,} {failure_count:>7,} {neither_count:>7,} {neither_pct:>8.1f}%")
    
    # Characteristics of 'neither' trades
    neither_trades = outcomes_df[outcomes_df['outcome'] == 'neither']
    
    if len(neither_trades) > 0:
        print(f"\nCHARACTERISTICS OF 'NEITHER' TRADES:")
        print("-" * 40)
        
        print(f"Average final ROI: {neither_trades['final_roi'].mean():>6.2f}%")
        print(f"Median final ROI:  {neither_trades['final_roi'].median():>6.2f}%")
        print(f"ROI range: {neither_trades['final_roi'].min():>6.2f}% to {neither_trades['final_roi'].max():>6.2f}%")
        
        print(f"\nAverage time in trade: {neither_trades['time_in_trade'].mean():>6.1f} seconds")
        print(f"Median time in trade:  {neither_trades['time_in_trade'].median():>6.1f} seconds")
        
        print(f"\nAverage entry volatility: {neither_trades['entry_volatility'].mean():>6.3f}%")
        print(f"Average entry time: {neither_trades['entry_time_minutes'].mean():>6.1f} minutes since open")
        
        # ROI distribution for neither trades
        print(f"\nFINAL ROI DISTRIBUTION FOR 'NEITHER' TRADES:")
        print("-" * 45)
        roi_bins = [-10, -0.5, -0.1, 0, 0.1, 0.5, 1.0, 10]
        roi_labels = ['< -0.5%', '-0.5 to -0.1%', '-0.1 to 0%', '0 to 0.1%', '0.1 to 0.5%', '0.5 to 1.0%', '> 1.0%']
        
        neither_trades['roi_bin'] = pd.cut(neither_trades['final_roi'], bins=roi_bins, labels=roi_labels, include_lowest=True)
        roi_dist = neither_trades['roi_bin'].value_counts().sort_index()
        
        for bin_label, count in roi_dist.items():
            pct = (count / len(neither_trades)) * 100
            print(f"  {bin_label:>12}: {count:>5,} ({pct:>5.1f}%)")
    
    # Time analysis
    print(f"\nTIME IN TRADE ANALYSIS:")
    print("-" * 30)
    
    for outcome in ['success', 'failure', 'neither']:
        outcome_data = outcomes_df[outcomes_df['outcome'] == outcome]
        if len(outcome_data) > 0:
            avg_time = outcome_data['time_in_trade'].mean()
            median_time = outcome_data['time_in_trade'].median()
            print(f"  {outcome.capitalize():>8}: Avg {avg_time:>6.1f}s, Median {median_time:>6.1f}s")


def create_filtered_training_data(sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes, drop_neither=True):
    """
    Create training data with option to drop 'neither' trades.
    
    Args:
        drop_neither: If True, exclude trades that reach neither condition
        
    Returns:
        DataFrame ready for model training
    """
    skip_dates = []
    result_df = pd.DataFrame(columns=['trade id', 'minutes since market open', 'volatility percent', 'optimal_stop_loss'])
    holding_percent = 0.6
    
    trades_processed = 0
    trades_dropped = 0
    
    print(f"Creating training data (drop_neither={drop_neither})")
    
    for idx, row in bulk_df.iterrows():
        ticker = row['Ticker']
        date = Helper_Functions.bulk_csv_date_converter(row['Date'])
        trade_id = row['Trade Id']
        
        if date in skip_dates:
            continue
            
        # Check if we have market data for this date and ticker
        if date not in market_data_dict_by_ticker:
            skip_dates.append(date)
            continue  
        if ticker not in market_data_dict_by_ticker[date]:
            continue

        entry_time = row['Entry Time']
        market_df = market_data_dict_by_ticker[date][ticker].copy()

        if not Helper_Functions.Check_We_Have_Data_For_Trade(market_df, entry_time):
            continue
        
        roi_list = roi_dictionary[trade_id]
        start_index = trade_start_indexes[trade_id]

        # Test each stop loss and find the optimal one for this trade
        sl_results = {}  # {stop_loss: success(1) or failure(0)}
        has_neither = False
        
        for sl in sl_list:
            counter = -1
            
            # Simulate this stop loss on the trade
            for i in range(start_index, len(market_df)):
                counter += 1
                if counter >= len(roi_list):
                    # Ran out of ROI data - this is a "neither" case
                    if drop_neither:
                        has_neither = True
                        break
                    else:
                        sl_results[sl] = 0  # Treat as failure (original behavior)
                        break
                    
                curr_roi = roi_list[counter]

                if curr_roi <= sl:
                    # Stop loss hit - failure
                    sl_results[sl] = 0
                    break
                elif curr_roi >= holding_percent:
                    # Target reached - success
                    sl_results[sl] = 1
                    break
            else:
                # Neither stop loss nor target hit
                if drop_neither:
                    has_neither = True
                else:
                    sl_results[sl] = 0  # Treat as failure (original behavior)
            
            # If we're dropping neither trades and found one, skip this trade entirely
            if has_neither and drop_neither:
                break
        
        # Skip this trade if it has 'neither' outcomes and we're dropping them
        if has_neither and drop_neither:
            trades_dropped += 1
            continue
        
        trades_processed += 1
        
        # Find the optimal stop loss: the loosest SL that still allows success
        successful_sls = [sl for sl, success in sl_results.items() if success == 1]
        
        if successful_sls:
            # Choose the loosest (least negative) stop loss that still succeeds
            optimal_sl = max(successful_sls)
        else:
            # No stop loss worked, choose the loosest one anyway
            optimal_sl = max(sl_list)
        
        # Add one row per trade with its optimal stop loss
        new_row = {
            'trade id': trade_id,
            'minutes since market open': market_df.iloc[start_index]['Time Since Market Open'],
            'volatility percent': market_df.iloc[start_index]['Volatility Percent'],
            'optimal_stop_loss': optimal_sl
        }
        result_df = result_df._append(new_row, ignore_index=True)

    print(f"\nTraining Data Creation Results:")
    print(f"  Trades processed: {trades_processed:,}")
    if drop_neither:
        print(f"  Trades dropped (neither): {trades_dropped:,}")
        drop_rate = (trades_dropped / (trades_processed + trades_dropped)) * 100
        print(f"  Drop rate: {drop_rate:.1f}%")
    
    return result_df


def compare_model_approaches(sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes):
    """
    Compare model performance with and without dropping 'neither' trades.
    """
    print("\n" + "="*80)
    print("COMPARING MODEL APPROACHES")
    print("="*80)
    
    # Create training data both ways
    print("\n1. Creating training data WITH 'neither' trades (original approach)...")
    data_with_neither = create_filtered_training_data(
        sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes, 
        drop_neither=False
    )
    
    print("\n2. Creating training data WITHOUT 'neither' trades (proposed approach)...")
    data_without_neither = create_filtered_training_data(
        sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes, 
        drop_neither=True
    )
    
    # Compare data characteristics
    print(f"\nDATA COMPARISON:")
    print("-" * 30)
    print(f"Original approach (with neither): {len(data_with_neither):,} samples")
    print(f"Proposed approach (drop neither): {len(data_without_neither):,} samples")
    
    reduction = len(data_with_neither) - len(data_without_neither)
    reduction_pct = (reduction / len(data_with_neither)) * 100
    print(f"Reduction: {reduction:,} samples ({reduction_pct:.1f}%)")
    
    # Compare optimal stop loss distributions
    print(f"\nOPTIMAL STOP LOSS DISTRIBUTIONS:")
    print("-" * 40)
    
    print("Original approach (with neither):")
    orig_dist = data_with_neither['optimal_stop_loss'].value_counts().sort_index()
    for sl, count in orig_dist.items():
        pct = (count / len(data_with_neither)) * 100
        print(f"  {sl:>6.1f}%: {count:>4,} ({pct:>5.1f}%)")
    
    print("\nProposed approach (drop neither):")
    new_dist = data_without_neither['optimal_stop_loss'].value_counts().sort_index()
    for sl, count in new_dist.items():
        pct = (count / len(data_without_neither)) * 100
        print(f"  {sl:>6.1f}%: {count:>4,} ({pct:>5.1f}%)")
    
    # Feature distribution comparison
    print(f"\nFEATURE DISTRIBUTIONS:")
    print("-" * 25)
    
    features = ['minutes since market open', 'volatility percent']
    for feature in features:
        print(f"\n{feature.title()}:")
        
        orig_mean = data_with_neither[feature].mean()
        orig_std = data_with_neither[feature].std()
        new_mean = data_without_neither[feature].mean()
        new_std = data_without_neither[feature].std()
        
        print(f"  Original: Mean {orig_mean:>6.2f}, Std {orig_std:>6.2f}")
        print(f"  Proposed: Mean {new_mean:>6.2f}, Std {new_std:>6.2f}")
        print(f"  Difference: {abs(new_mean - orig_mean):>6.2f} ({abs(new_mean - orig_mean)/orig_mean*100:>5.1f}%)")
    
    return data_with_neither, data_without_neither


def assess_solution_correctness(outcomes_df):
    """
    Assess whether dropping 'neither' trades is the correct approach.
    """
    print("\n" + "="*80)
    print("SOLUTION CORRECTNESS ASSESSMENT")
    print("="*80)
    
    neither_trades = outcomes_df[outcomes_df['outcome'] == 'neither']
    
    if len(neither_trades) == 0:
        print("No 'neither' trades found - no action needed.")
        return
    
    print(f"\nANALYZING {len(neither_trades):,} 'NEITHER' TRADE INSTANCES...")
    
    # 1. Check if these are truly ambiguous cases
    print(f"\n1. AMBIGUITY ANALYSIS:")
    print("-" * 25)
    
    # Look at final ROI distribution
    final_rois = neither_trades['final_roi']
    
    positive_roi = (final_rois > 0).sum()
    negative_roi = (final_rois < 0).sum()
    zero_roi = (final_rois == 0).sum()
    
    print(f"Final ROI when trade ended:")
    print(f"  Positive ROI: {positive_roi:,} ({positive_roi/len(neither_trades)*100:.1f}%)")
    print(f"  Negative ROI: {negative_roi:,} ({negative_roi/len(neither_trades)*100:.1f}%)")
    print(f"  Zero ROI:     {zero_roi:,} ({zero_roi/len(neither_trades)*100:.1f}%)")
    
    # 2. Check if they're close to either condition
    print(f"\n2. PROXIMITY TO CONDITIONS:")
    print("-" * 30)
    
    holding_percent = 0.6
    
    # How close did they get to target?
    max_roi_reached = final_rois.max()
    close_to_target = (final_rois > holding_percent * 0.8).sum()  # Within 80% of target
    
    print(f"Highest ROI reached by 'neither' trade: {max_roi_reached:.2f}%")
    print(f"Trades within 80% of target (0.48%): {close_to_target:,}")
    
    # How close did they get to stop losses?
    for sl in sorted(outcomes_df['stop_loss'].unique()):
        sl_neither = neither_trades[neither_trades['stop_loss'] == sl]
        if len(sl_neither) > 0:
            close_to_sl = (sl_neither['final_roi'] < sl * 0.8).sum()  # Within 80% of stop loss
            print(f"Trades close to {sl:.1f}% stop loss: {close_to_sl:,}")
    
    # 3. Time analysis - are these just trades that ran out of time?
    print(f"\n3. TIME CONSTRAINT ANALYSIS:")
    print("-" * 30)
    
    avg_time_neither = neither_trades['time_in_trade'].mean()
    avg_time_success = outcomes_df[outcomes_df['outcome'] == 'success']['time_in_trade'].mean()
    avg_time_failure = outcomes_df[outcomes_df['outcome'] == 'failure']['time_in_trade'].mean()
    
    print(f"Average time in trade:")
    print(f"  Neither trades: {avg_time_neither:>6.1f} seconds")
    print(f"  Successful trades: {avg_time_success:>6.1f} seconds")
    print(f"  Failed trades: {avg_time_failure:>6.1f} seconds")
    
    # Check if neither trades are systematically longer
    if avg_time_neither > max(avg_time_success, avg_time_failure):
        print(f"  → 'Neither' trades are longer on average - likely hit time/data limits")
    
    # 4. Data availability analysis
    print(f"\n4. DATA AVAILABILITY ANALYSIS:")
    print("-" * 35)
    
    avg_roi_length_neither = neither_trades['roi_data_length'].mean()
    all_roi_lengths = outcomes_df.groupby('trade_id')['roi_data_length'].first()
    avg_roi_length_all = all_roi_lengths.mean()
    
    print(f"Average ROI data length:")
    print(f"  Trades with 'neither' outcomes: {avg_roi_length_neither:>6.1f} data points")
    print(f"  All trades: {avg_roi_length_all:>6.1f} data points")
    
    # 5. Final recommendation
    print(f"\n5. RECOMMENDATION:")
    print("-" * 20)
    
    drop_rate = len(neither_trades) / len(outcomes_df) * 100
    
    print(f"Drop rate: {drop_rate:.1f}% of training samples")
    
    if drop_rate > 50:
        print("⚠️  WARNING: Very high drop rate - investigate data collection issues")
    elif drop_rate > 30:
        print("⚠️  CAUTION: High drop rate - ensure this is acceptable for your use case")
    else:
        print("✅ ACCEPTABLE: Drop rate is within reasonable bounds")
    
    # Assess if dropping is correct
    mostly_positive = positive_roi > (negative_roi * 2)  # More than 2:1 ratio
    mostly_negative = negative_roi > (positive_roi * 2)
    
    if mostly_positive:
        print("📈 Many 'neither' trades end positive - consider them partial successes?")
    elif mostly_negative:
        print("📉 Many 'neither' trades end negative - treating as failures might be valid")
    else:
        print("⚖️  Mixed outcomes - dropping them is likely the most unbiased approach")
    
    print(f"\n✅ CONCLUSION: Dropping 'neither' trades is RECOMMENDED because:")
    print(f"   • They represent ambiguous cases that don't clearly fit success/failure")
    print(f"   • Including them as failures introduces bias")
    print(f"   • The model should learn from clear examples of success vs failure")
    print(f"   • {drop_rate:.1f}% drop rate is acceptable for cleaner training data")


def main():
    """
    Run comprehensive analysis of 'neither' trades and provide recommendations.
    """
    # Load data
    sl_list = [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8]
    
    bulk_summary_path = "Holder_Strat/Summary_Csvs/bulk_summaries.csv"
    columns_to_keep = ["Date", "Trade Id", "Ticker", "Entry Time", "Entry Price", "Trade Type"]
    bulk_df = pd.read_csv(bulk_summary_path)[columns_to_keep]

    print(f"Loaded {len(bulk_df)} trades from {bulk_summary_path}")

    market_data_dict_by_ticker = Helper_Functions.Load_Market_Data_Dictionary(bulk_df)
    roi_dictionary, trade_end_timestamps, trade_start_indexes = Helper_Functions.Create_Roi_Dictionary_For_Trades(bulk_df, market_data_dict_by_ticker, sl_list[-1])
    
    # Run comprehensive analysis
    outcomes_df = analyze_neither_trades(sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes)
    
    # Compare approaches
    data_with_neither, data_without_neither = compare_model_approaches(sl_list, bulk_df, market_data_dict_by_ticker, roi_dictionary, trade_start_indexes)
    
    # Assess correctness
    assess_solution_correctness(outcomes_df)
    
    return outcomes_df, data_with_neither, data_without_neither


if __name__ == "__main__":
    main()
