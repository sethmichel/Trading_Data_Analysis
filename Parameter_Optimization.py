import pandas as pd
import os
import inspect
import sys
import shutil
import Main_Globals
from datetime import datetime
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
import optuna

# Global list to collect all results
all_results = []


def simulate_trades(volatility, ratio, bias, normal_target, upper_target, upper_stop_loss, normal_stop_loss, df):
    """
    Simulate trades based on the given parameters.
    This is essentially the same as Volatility_Percent_vs_Ratio_vs_Parameters but with a cleaner interface.
    """
    filtered_df = df[(df['Entry Volatility Percent'] >= volatility) & 
                     (df['Entry Volatility Ratio'] >= ratio) & 
                     (df['Entry Directional Bias Abs Distance'] >= bias)]
                                
    # Create result dictionary
    result = {
        'id': (volatility, ratio, bias, normal_target, upper_target, upper_stop_loss, normal_stop_loss),
        'sum': 0, 
        'count': len(filtered_df), 
        'wins': 0, 
        'losses': 0, 
        'neither': 0
    }

    for index, row in filtered_df.iterrows():
        price_movement_list = row['price_movement_list']

        if not price_movement_list:  # Check if list is empty
            result['neither'] += 1
            continue

        updated_flag = False
        curr_target = normal_target
        curr_sl = normal_stop_loss
        hit_target_once = False

        for value in price_movement_list:
            if value == curr_target:
                if hit_target_once == False:
                    # Hit normal target - switch to upper targets
                    hit_target_once = True
                    curr_target = upper_target
                    curr_sl = upper_stop_loss
                else:
                    # Hit upper target
                    result['sum'] += upper_target
                    result['wins'] += 1
                    updated_flag = True
                    break

            elif (value == curr_sl):
                # Hit stop loss (normal or upper)
                result['sum'] += curr_sl
                result['losses'] += 1
                updated_flag = True
                break

        # If we never hit a target or stop loss, use the last value
        if (updated_flag == False):
            result['sum'] += price_movement_list[-1]
            result['neither'] += 1

    return result


def Volatility_Percent_vs_Ratio_vs_Parameters(df, volatility, ratio, dir_bias, normal_target, upper_target, upper_stop_loss, normal_stop_loss, sublist_key):
    """Keep the original function for backward compatibility"""
    result = simulate_trades(volatility, ratio, dir_bias, normal_target, upper_target, upper_stop_loss, normal_stop_loss, df)
    result['id'] = sublist_key  # Override the id with the string key for backward compatibility
    return result


def objective(trial, df):
    """
    Objective function for Optuna optimization.
    Returns the total return (sum) to be maximized.
    """
    global all_results
    
    # Define parameter spaces
    volatilities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ratios = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    dir_bias_dist = [0.0, 0.1, 0.2, 0.3, 0.4]
    normal_targets = [0.2, 0.3, 0.4, 0.5, 0.6]
    upper_targets = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    upper_stop_losss = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
    normal_stop_losss = [-0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
    
    # Suggest parameters
    volatility = trial.suggest_categorical("volatility", volatilities)
    ratio = trial.suggest_categorical("ratio", ratios)
    bias = trial.suggest_categorical("bias", dir_bias_dist)
    normal_target = trial.suggest_categorical("nt", normal_targets)
    upper_target = trial.suggest_categorical("ut", upper_targets)
    upper_stop_loss = trial.suggest_categorical("usl", upper_stop_losss)
    normal_stop_loss = trial.suggest_categorical("nsl", normal_stop_losss)
    
    # Apply constraints - return very low score if constraints are violated
    if upper_target <= normal_target:
        return -1e9
    
    if upper_stop_loss >= upper_target or upper_stop_loss >= normal_target:
        return -1e9
    
    if normal_stop_loss >= normal_target or normal_stop_loss >= upper_stop_loss:
        return -1e9
    
    # Simulate trades with the suggested parameters
    result = simulate_trades(volatility, ratio, bias, normal_target, upper_target, upper_stop_loss, normal_stop_loss, df)
    
    # Add parameter information to the result
    result['parameters'] = {
        'volatility': volatility,
        'ratio': ratio,
        'bias': bias,
        'normal_target': normal_target,
        'upper_target': upper_target,
        'upper_stop_loss': upper_stop_loss,
        'normal_stop_loss': normal_stop_loss
    }
    
    # Collect results for later analysis
    all_results.append(result.copy())
    
    # Return the sum (total return) to maximize
    return result['sum']


def bayesian_optimization_optuna(df, n_trials=1000):
    """
    Perform Bayesian optimization using Optuna.
    """
    global all_results
    all_results = []  # Reset results list
    
    # Pre-process price movements to avoid repeated string operations
    df['price_movement_list'] = df['Price Movement'].apply(
        lambda x: [float(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else []
    )
    print("Done - preprocessed price movements")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    print(f"Starting Bayesian optimization with {n_trials} trials...")
    study.optimize(lambda trial: objective(trial, df), n_trials=n_trials)
    
    # Print best results
    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION RESULTS")
    print("="*80)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Best Value (Total Return): {trial.value:.2f}")
    print("  Best Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Get detailed results for best parameters
    best_params = trial.params
    best_result = simulate_trades(
        best_params['volatility'], 
        best_params['ratio'], 
        best_params['bias'], 
        best_params['nt'], 
        best_params['ut'], 
        best_params['usl'], 
        best_params['nsl'], 
        df
    )
    
    print(f"\nDetailed results for best parameters:")
    print(f"  Total return: {best_result['sum']:.2f}")
    print(f"  Number of trades: {best_result['count']}")
    print(f"  Wins: {best_result['wins']}")
    print(f"  Losses: {best_result['losses']}")
    print(f"  Neither: {best_result['neither']}")
    if best_result['count'] > 0:
        print(f"  Win rate: {(best_result['wins'] / best_result['count']) * 100:.1f}%")
        print(f"  Average return per trade: {best_result['sum'] / best_result['count']:.4f}")
    
    # Sort all results by sum (total return) descending and show top 10
    print(f"\n" + "="*80)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("="*80)
    
    # Filter out invalid results (those with -1e9 return)
    valid_results = [r for r in all_results if r['sum'] > -1e8]
    
    # Sort by sum descending
    sorted_results = sorted(valid_results, key=lambda x: x['sum'], reverse=True)
    
    print(f"\nFound {len(valid_results)} valid parameter combinations out of {len(all_results)} total trials")
    print(f"Showing top {min(10, len(sorted_results))} results:\n")
    
    for i, result in enumerate(sorted_results[:10], 1):
        params = result['parameters']
        print(f"#{i}")
        print(f"  Parameters: (vol={params['volatility']}, ratio={params['ratio']}, bias={params['bias']}, "
              f"nt={params['normal_target']}, ut={params['upper_target']}, "
              f"nsl={params['normal_stop_loss']}, usl={params['upper_stop_loss']})")
        print(f"  Total Return: {result['sum']:.2f}")
        print(f"  Trades: {result['count']}, Wins: {result['wins']}, Losses: {result['losses']}, Neither: {result['neither']}")
        if result['count'] > 0:
            print(f"  Win Rate: {(result['wins'] / result['count']) * 100:.1f}%, "
                  f"Avg Return/Trade: {result['sum'] / result['count']:.4f}")
        print()
    
    # Additional statistics
    if valid_results:
        print("="*80)
        print("OPTIMIZATION STATISTICS")
        print("="*80)
        returns = [r['sum'] for r in valid_results]
        print(f"Mean return: {sum(returns) / len(returns):.2f}")
        print(f"Best return: {max(returns):.2f}")
        print(f"Worst return: {min(returns):.2f}")
        print(f"Standard deviation: {(sum([(r - sum(returns)/len(returns))**2 for r in returns]) / len(returns))**0.5:.2f}")
    
    return study, best_result, sorted_results[:10]


def bayesian_optimization(df):
    """Original grid search function - kept for backward compatibility"""
    volatilities = [0.2,0.3,0.4,0.5, 0.6, 0.7, 0.8, 0.9,1.0]
    ratios = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4]
    dir_bias_dist = [0.0,0.1,0.2,0.3,0.4] # these are absolute value distance from 0.5. bec 0.5 is neutral and upper/lower values are good
    normal_targets = [0.2,0.3,0.4,0.5,0.6]
    upper_targets = [0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    upper_stop_losss = [0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
    normal_stop_losss = [-0.3,-0.4,-0.5,-0.6,-0.7,-0.8,-0.9]
    all_sublists = {}

    # Pre-process price movements to avoid repeated string operations
    df['price_movement_list'] = df['Price Movement'].apply(
        lambda x: [float(val) for val in str(x).split('|')] if str(x) and str(x) != 'nan' else []
    )
    print("done - preprocessed price movements")

    for volatility in volatilities:
        for ratio in ratios:
            for dir_bias in dir_bias_dist:
                for normal_target in normal_targets:
                    for upper_target in upper_targets:
                        if (upper_target <= normal_target):
                            continue
                        
                        for upper_stop_loss in upper_stop_losss:
                            if ((upper_stop_loss >= upper_target) or upper_stop_loss >= normal_target):
                                continue

                            for normal_stop_loss in normal_stop_losss:
                                if ((normal_stop_loss >= normal_target) or (normal_stop_loss >= upper_stop_loss)):
                                    continue

                                sublist_key = f"{volatility}, {ratio}, {dir_bias}, {normal_target}, {upper_target}, {normal_stop_loss}, {upper_stop_loss}"
                                sublist = Volatility_Percent_vs_Ratio_vs_Parameters(df, volatility, ratio, dir_bias, normal_target, upper_target, upper_stop_loss, normal_stop_loss, sublist_key)
                                all_sublists[sublist_key] = sublist

    # find the top x sublists based on sum
    best_sublists = {}
    sorted_items = sorted(all_sublists.items(), key=lambda x: x[1]['sum'], reverse=True)
    
    for i in range(min(10, len(sorted_items))):
        key, sublist = sorted_items[i]
        sublist_rounded = sublist.copy()
        sublist_rounded['sum'] = round(sublist_rounded['sum'], 2)
        best_sublists[key] = sublist_rounded


# Main execution
if __name__ == "__main__":
    data_dir = "Csv_Files/3_Final_Trade_Csvs"
    data_file = "Bulk_Combined.csv"
    df = pd.read_csv(f"{data_dir}/{data_file}")

    print("Choose optimization method:")
    print("1. Bayesian Optimization (Optuna) - Recommended")
    print("2. Grid Search (Original)")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        n_trials = int(input("Enter number of trials for Bayesian optimization (default 1000): ") or "1000")
        study, best_result, top_results = bayesian_optimization_optuna(df, n_trials)
    else:
        print("Running original grid search...")
        bayesian_optimization(df)