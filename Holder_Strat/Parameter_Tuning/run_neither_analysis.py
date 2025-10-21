"""
Simple script to run the 'neither trades' analysis.

This script will:
1. Analyze the current handling of trades that reach neither condition
2. Show statistics about these trades
3. Compare model performance with and without dropping them
4. Provide recommendations
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Neither_Trades_Analysis import main as run_analysis

if __name__ == "__main__":
    print("="*80)
    print("NEITHER TRADES ANALYSIS")
    print("="*80)
    print("This analysis will help determine if dropping 'neither' trades is correct.")
    print("Please wait while we process your trading data...")
    print("="*80)
    
    try:
        # Run the comprehensive analysis
        outcomes_df, data_with_neither, data_without_neither = run_analysis()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("Key findings:")
        print("1. Check the 'Drop rate' - if it's around 37%, this matches your observation")
        print("2. Look at the 'SOLUTION CORRECTNESS ASSESSMENT' for recommendations")
        print("3. The analysis shows whether dropping these trades is the right approach")
        print("\nNext steps:")
        print("- If the analysis recommends dropping 'neither' trades, use the updated")
        print("  Stop_Loss_Tuning.py with drop_neither_trades=True")
        print("- The updated code will automatically handle this for you")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("\nThis might be due to:")
        print("1. Missing data files")
        print("2. Incorrect file paths")
        print("3. Data format issues")
        print("\nPlease check your data files and paths, then try again.")
